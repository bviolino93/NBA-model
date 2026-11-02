"""
Player prop projections: points, rebounds, assists, threes made.

    projection = minutes x per-minute rate x pace

each piece built from earlier games only:

  minutes   exponentially weighted minutes in games he played, scaled up
            when rotation teammates are missing: their expected minutes go
            to whoever is left, in proportion to each player's own minutes
            (capped at 42)
  rate      season-to-date stat per minute, shrunk toward last season's
            rate (or the league's) with PRIOR_MIN minutes of weight
  pace      tonight's closing total / the season's average total so far;
            more points in the game means more of every counting stat

Props are void when a player does not play, so the backtest scores only
games he played, like a book does. The test that matters is against prop
lines (logged nightly by log_lines.py); until there are enough, it is
scored against the naive projections a book has to beat: season average
and last-5 average.

    python nba_props.py --first 2016 --split 2022
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

STATS = ["pts", "reb", "ast", "fg3m"]
PRIOR_MIN = 250.0
HALF_LIFE = 8.0        # games, for minutes
MAX_MIN = 42.0


def _shift_ewm(s: pd.Series, half_life: float) -> pd.Series:
    return s.ewm(halflife=half_life, min_periods=1).mean().shift(1)


def league_rates(p: pd.DataFrame) -> pd.DataFrame:
    """Per-minute league rate per season, for shrinkage priors."""
    pl = p[p["played"]]
    agg = pl.groupby("season")[STATS + ["min"]].sum()
    return agg[STATS].div(agg["min"], axis=0)


def project(p: pd.DataFrame, ab: pd.DataFrame | None = None,
            games: pd.DataFrame | None = None,
            half_life: float = HALF_LIFE, prior_min: float = PRIOR_MIN,
            use_absences: bool = True) -> pd.DataFrame:
    """One row per player-game PLAYED with projections and baselines.

    p      box rows (nba_players.load_box)
    ab     absences per (game_id, team) (nba_players.absences)
    games  game rows with total_line, for pace
    """
    pl = p[p["played"]].sort_values(["commence", "game_id"]).copy()
    key = ["season", "player_id"]
    g = pl.groupby(key, sort=False)

    pl["n_prev"] = g.cumcount()
    pl["base_min"] = g["min"].transform(lambda s: _shift_ewm(s, half_life))

    # Prior rate: the player's own last season, else the league's.
    lr = league_rates(p)
    own = (pl.groupby(key)[STATS + ["min"]].sum().reset_index())
    own["season"] = own["season"] + 1           # applies to the next season
    for s in STATS:
        own[f"prior_{s}"] = own[s] / own["min"].clip(lower=1)
    pl = pl.merge(own[key + [f"prior_{s}" for s in STATS]], on=key, how="left")
    # Last season's league rate; the first season loaded has none, so it
    # uses its own (a small look-ahead confined to that season).
    prev_league = lr.shift(1).fillna(lr)
    for s in STATS:
        lg = pl["season"].map(prev_league[s])
        pl[f"prior_{s}"] = pl[f"prior_{s}"].fillna(lg)

    g = pl.groupby(key, sort=False)
    cum_min = g["min"].transform(lambda s: s.cumsum().shift(1)).fillna(0)
    for s in STATS:
        cum = g[s].transform(lambda x: x.cumsum().shift(1)).fillna(0)
        pl[f"rate_{s}"] = ((cum + pl[f"prior_{s}"] * prior_min)
                           / (cum_min + prior_min))
        # Naive baselines a book would beat
        pl[f"avg_{s}"] = g[s].transform(lambda x: x.expanding().mean().shift(1))
        pl[f"l5_{s}"] = g[s].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1))

    # Minutes: share out missing teammates' minutes.
    pl["proj_min"] = pl["base_min"]
    if use_absences and ab is not None and len(ab):
        a = ab[["game_id", "team", "lost_min"]]
        pl = pl.merge(a, on=["game_id", "team"], how="left")
        pl["lost_min"] = pl["lost_min"].fillna(0.0)
        pl["proj_min"] = (pl["base_min"] * 240.0
                          / (240.0 - pl["lost_min"].clip(upper=120)))
    pl["proj_min"] = pl["proj_min"].clip(upper=MAX_MIN)

    # Pace from the game's closing total vs the season's average so far.
    pl["pace"] = 1.0
    if games is not None and "total_line" in games:
        gg = games.sort_values("commence")[["game_id", "season", "total_line"]].copy()
        gg["season_avg_total"] = (gg.groupby("season")["total_line"]
                                    .transform(lambda s: s.expanding().mean()
                                               .shift(1)))
        gg["pace"] = (gg["total_line"] / gg["season_avg_total"]).clip(0.85, 1.15)
        pl = pl.drop(columns="pace").merge(
            gg[["game_id", "pace"]], on="game_id", how="left")
        pl["pace"] = pl["pace"].fillna(1.0)

    for s in STATS:
        pl[f"proj_{s}"] = pl["proj_min"] * pl[f"rate_{s}"] * pl["pace"]
    return pl


def mae_table(fr: pd.DataFrame, min_prev: int = 5,
              min_base: float = 15.0) -> pd.DataFrame:
    """MAE of each projection on games of established rotation players."""
    f = fr[(fr["n_prev"] >= min_prev) & (fr["base_min"] >= min_base)]
    rows = []
    for s in STATS:
        y = f[s]
        rows.append({"stat": s, "n": len(f),
                     "model": (f[f"proj_{s}"] - y).abs().mean(),
                     "season_avg": (f[f"avg_{s}"] - y).abs().mean(),
                     "last5": (f[f"l5_{s}"] - y).abs().mean()})
    return pd.DataFrame(rows).set_index("stat")


def main():
    from nba_data import load_games
    from nba_players import absences, load_box
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2016)
    ap.add_argument("--split", type=int, default=2022)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    g = load_games(a.first)
    box = load_box(g)
    ab = absences(box)
    fr = project(box, ab, g)
    flat = project(box, ab, g, use_absences=False)
    fr["proj_noinj_pts"] = flat["proj_pts"].to_numpy()
    for lab, part in (("tuning", fr[fr["season"] < a.split]),
                      ("held out", fr[fr["season"] >= a.split])):
        print(f"== {lab}: MAE, rotation players (5+ games, 15+ min) ==")
        print(mae_table(part).round(3).to_string())
        f = part[(part["n_prev"] >= 5) & (part["base_min"] >= 15)]
        hit = f[f["lost_min"] >= 20]
        if len(hit):
            print(f"  games with 20+ teammate minutes missing: {len(hit)}; "
                  f"points MAE with/without the boost "
                  f"{(hit['proj_pts'] - hit['pts']).abs().mean():.3f} / "
                  f"{(hit['proj_noinj_pts'] - hit['pts']).abs().mean():.3f}; "
                  f"mean error with/without "
                  f"{(hit['proj_pts'] - hit['pts']).mean():+.3f} / "
                  f"{(hit['proj_noinj_pts'] - hit['pts']).mean():+.3f}")
        print()
    if a.out:
        fr.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
