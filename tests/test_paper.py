import numpy as np
import pandas as pd

import paper
from selector import Fair


def _games():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    rows = []
    d0 = pd.Timestamp("2026-10-01")
    for n in range(300):
        h, a = rng.choice(teams, 2, replace=False)
        hs = 112 + (8 if h == "T0" else 0) + rng.normal(0, 10)
        as_ = 110 + rng.normal(0, 10)
        rows.append({"game_id": str(n), "season": 2027, "game_date": (d0 + pd.Timedelta(days=n // 4)).date(),
                     "home_team": h, "away_team": a, "home_score": round(hs), "away_score": round(as_),
                     "completed": True, "neutral": False, "spread_line": np.nan, "total_line": np.nan})
    return pd.DataFrame(rows)


def test_model_lines_log_and_grade():
    g = _games()
    now = pd.Timestamp("2027-01-20T12:00Z")
    fair = {"x": Fair("T0", "T1", "2027-01-21T00:00:00Z", market_margin=-2.0, market_total=260.0)}
    lines = paper.model_lines(g, fair, now)
    assert lines["x"]["margin"] > 3          # T0 is strong at home
    t = paper.log([], fair, lines, now.isoformat())
    sides = {b["market"]: b["side"] for b in t}
    assert sides == {"spreads": "T0", "totals": "Under"}
    sp = [b for b in t if b["market"] == "spreads"][0]
    assert sp["point"] == 2.0 and sp["line"] == -2.0
    assert paper.log(t, fair, lines, now.isoformat()) == t      # logged once
    fin = pd.DataFrame([{"home_team": "T0", "away_team": "T1", "game_date": pd.Timestamp("2027-01-20").date(),
                         "completed": True, "home_score": 100, "away_score": 101,
                         "spread_line": 1.0, "total_line": 258.0}])
    t = paper.grade(t, fin)
    sp = [b for b in t if b["market"] == "spreads"][0]
    tot = [b for b in t if b["market"] == "totals"][0]
    assert sp["result"] == "win" and sp["clv"] == 3.0     # T0 +2, lost by 1; line moved -2 -> +1
    assert tot["result"] == "win" and tot["clv"] == 2.0   # under 260, 201 scored; closed 258
    assert paper.summary(t)["wins"] == 2


def test_anchored_lines_use_smaller_gaps():
    fair = {"x": Fair("T0", "T1", "2027-01-21T00:00:00Z", market_margin=-2.0, market_total=230.0)}
    lines = {"x": {"margin": -0.8, "total": 231.5, "anchored": True}}
    t = paper.log([], fair, lines, "r")
    assert [(b["market"], b["side"]) for b in t] == [("spreads", "T0")]
