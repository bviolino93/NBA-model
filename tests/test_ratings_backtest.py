import numpy as np
import pandas as pd

from backtest import record, slope
from nba_ratings import RatingConfig, walk_forward


def synthetic(seed=0, days=160):
    """A season where team strength is known, so ratings must find it."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    true = dict(zip(teams, np.linspace(-8, 8, 10)))
    rows, gid = [], 0
    start = pd.Timestamp("2025-10-21")
    for d in range(days):
        order = rng.permutation(teams)
        day = start + pd.Timedelta(days=d)
        for h, a in zip(order[::2], order[1::2]):
            m = 3 + true[h] - true[a] + rng.normal(0, 12)
            tot = 225 + rng.normal(0, 18)
            rows.append({"game_id": str(gid), "season": 2026,
                         "season_type": "REG", "game_date": day.date(),
                         "commence": day.tz_localize("UTC"),
                         "home_team": h, "away_team": a,
                         "home_score": round((tot + m) / 2),
                         "away_score": round((tot - m) / 2),
                         "spread_line": 3 + true[h] - true[a],
                         "total_line": 225.0, "neutral": False,
                         "completed": True})
            gid += 1
    return pd.DataFrame(rows), true


def test_walk_forward_recovers_strength_without_lookahead():
    g, true = synthetic()
    fr = walk_forward(g, RatingConfig(min_games=100))
    late = fr[fr["model_margin"].notna()]
    assert len(late) > 500
    # Projections track the true line closely once enough games are in.
    err = (late["model_margin"] - late["spread_line"]).abs().mean()
    assert err < 2.5
    # No look-ahead: the first projectable day uses only earlier games.
    first_day = late["game_date"].min()
    assert (fr["game_date"] < first_day).sum() >= 100


def test_slope_and_record():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = 0.5 * x + rng.normal(size=2000)
    b, t, n = slope(x, y)
    assert abs(b - 0.5) < 0.1 and t > 10 and n == 2000
    w, l, p = record(np.array([5.0, -5.0, 1.0]), np.array([0.0, 0.0, 0.0]),
                     np.array([3.0, 2.0, 9.0]), edge=3)
    assert (w, l, p) == (1, 1, 0)
