"""
Team ratings from scores alone, walk-forward, as the baseline to beat.

Each game day the ratings are refitted on every earlier game, weighted by
recency (half-life in days, with the offseason counted as extra days so
last season fades over summer), by ridge least squares:

    home_margin  = hca + r[home] - r[away]
    total points = mu + t[home] + t[away]

Ridge shrinks every team toward average, which matters in the first weeks
of a season. The margin model stands in for net rating and the total model
for pace times efficiency; box-score ratings can replace them later, but
the NFL lesson is that the score-based baseline already ties the close,
so the work belongs in the adjustments, not the ratings.

Nothing here bets. backtest.py scores these against the closing line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RatingConfig:
    half_life: float = 120.0     # days (tools/tune_ratings.py, 2026-11-02)
    offseason_days: float = 150.0  # summer counts as this many extra days
    ridge_margin: float = 3.0
    ridge_total: float = 3.0
    min_games: int = 200          # need this many prior games to project


def _age(dates: np.ndarray, seasons: np.ndarray, on, season: int,
         cfg: RatingConfig) -> np.ndarray:
    days = np.asarray((pd.Timestamp(on) - pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))).days, float)
    return days + cfg.offseason_days * (season - seasons)


def fit(prior: pd.DataFrame, on, season: int, cfg: RatingConfig,
        teams: list[str]):
    """Ratings from games before `on`. Returns (hca, r, mu, t) dicts."""
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(prior), len(teams)
    age = _age(prior["game_date"], prior["season"].to_numpy(), on, season, cfg)
    w = 0.5 ** (age / cfg.half_life)
    hi = prior["home_team"].map(idx).to_numpy()
    ai = prior["away_team"].map(idx).to_numpy()
    rows = np.arange(n)
    neutral = prior["neutral"].astype(float).to_numpy()

    # Margin: [hca | teams]
    X = np.zeros((n, k + 1))
    X[:, 0] = 1.0 - neutral
    X[rows, hi + 1] += 1.0
    X[rows, ai + 1] -= 1.0
    y = (prior["home_score"] - prior["away_score"]).to_numpy(float)
    P = np.eye(k + 1) * cfg.ridge_margin
    P[0, 0] = 1e-6
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + P, X.T @ (w * y))

    # Total: [mu | teams]
    Z = np.zeros((n, k + 1))
    Z[:, 0] = 1.0
    Z[rows, hi + 1] += 1.0
    Z[rows, ai + 1] += 1.0
    yt = (prior["home_score"] + prior["away_score"]).to_numpy(float)
    Q = np.eye(k + 1) * cfg.ridge_total
    Q[0, 0] = 1e-6
    gam = np.linalg.solve(Z.T @ (Z * w[:, None]) + Q, Z.T @ (w * yt))
    return (beta[0], dict(zip(teams, beta[1:])),
            gam[0], dict(zip(teams, gam[1:])))


def walk_forward(g: pd.DataFrame, cfg: RatingConfig | None = None,
                 first_season: int | None = None) -> pd.DataFrame:
    """Project every completed game from games before its date only.

    Adds model_margin and model_total. Games before `first_season`, or
    with fewer than cfg.min_games prior games, are left NaN.
    """
    cfg = cfg or RatingConfig()
    g = g.sort_values("commence").reset_index(drop=True).copy()
    done = g[g["completed"]].copy()
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    g["model_margin"] = np.nan
    g["model_total"] = np.nan
    for day, todays in g.groupby("game_date", sort=True):
        season = int(todays["season"].iloc[0])
        if first_season and season < first_season:
            continue
        prior = done[(done["game_date"] < day)
                     & (done["season"] >= season - 2)]
        if len(prior) < cfg.min_games:
            continue
        hca, r, mu, t = fit(prior, day, season, cfg, teams)
        h, a = todays["home_team"], todays["away_team"]
        home_adv = hca * (1.0 - todays["neutral"].astype(float))
        g.loc[todays.index, "model_margin"] = (home_adv + h.map(r).fillna(0)
                                               - a.map(r).fillna(0))
        g.loc[todays.index, "model_total"] = (mu + h.map(t).fillna(0)
                                              + a.map(t).fillna(0))
    return g
