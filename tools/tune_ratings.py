"""
Tune the ratings' memory on past seasons: half-life, how much of last
season carries over (offseason_days) and ridge strength.

Scored by MAE of the predicted home margin and total against final
scores, on tuning seasons only, overall and for the early season (either
team under 15 games in, where tonight's ratings live). Also reports how
often each setting disagrees with the closing line by 3+ points; a
sound model shouldn't do that on most games.

    python tools/tune_ratings.py --first 2016 --split 2022
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_data import load_games, team_games  # noqa: E402
from nba_ratings import RatingConfig, walk_forward  # noqa: E402


def early_flag(g: pd.DataFrame, n: int = 15) -> np.ndarray:
    t = team_games(g).sort_values("commence")
    t["k"] = t.groupby(["team", "season"]).cumcount()
    s = t.set_index(["game_id", "team"])["k"]
    h = s.reindex(list(zip(g["game_id"], g["home_team"]))).to_numpy()
    a = s.reindex(list(zip(g["game_id"], g["away_team"]))).to_numpy()
    return np.fmin(h, a) < n


def score(fr: pd.DataFrame) -> dict:
    m = fr["home_score"] - fr["away_score"]
    p = fr["home_score"] + fr["away_score"]
    em = (fr["model_margin"] - m).abs()
    et = (fr["model_total"] - p).abs()
    dm = (fr["model_margin"] - fr["spread_line"]).abs()
    dt = (fr["model_total"] - fr["total_line"]).abs()
    e = fr["early"]
    return {"mae_m": em.mean(), "mae_m_early": em[e].mean(),
            "mae_t": et.mean(), "mae_t_early": et[e].mean(),
            "close_m": (fr["spread_line"] - m).abs().mean(),
            "close_m_early": (fr["spread_line"] - m)[e].abs().mean(),
            "gap3_m": (dm >= 3).mean(), "gap3_m_early": (dm[e] >= 3).mean(),
            "gap3_t": (dt >= 3).mean(), "gap3_t_early": (dt[e] >= 3).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2016)
    ap.add_argument("--split", type=int, default=2022)
    a = ap.parse_args()
    g = load_games(a.first)
    g = g[g["season_type"].isin(["REG", "PLAYIN"])].reset_index(drop=True)
    g["early"] = early_flag(g)
    grid = list(itertools.product((30.0, 45.0, 75.0, 120.0),
                                  (0.0, 30.0, 60.0, 150.0),
                                  (3.0, 15.0, 40.0)))
    rows = []
    for hl, off, ridge in grid:
        t0 = time.time()
        cfg = RatingConfig(half_life=hl, offseason_days=off,
                           ridge_margin=ridge, ridge_total=ridge)
        fr = walk_forward(g, cfg, first_season=a.first + 1)
        fr = fr[fr["completed"] & fr["model_margin"].notna()]
        for part, d in (("tune", fr[fr["season"] < a.split]),
                        ("held", fr[fr["season"] >= a.split])):
            rows.append({"half_life": hl, "offseason": off, "ridge": ridge,
                         "part": part, **score(d)})
        print(f"hl {hl:g} off {off:g} ridge {ridge:g}: "
              f"{time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(rows)
    tune = r[r["part"] == "tune"].copy()
    # Pick on the tuning seasons: margin + total MAE, early games doubled
    # because that's where the live model is weakest.
    tune["obj"] = (tune["mae_m"] + tune["mae_m_early"]
                   + 0.5 * (tune["mae_t"] + tune["mae_t_early"]))
    tune = tune.sort_values("obj")
    pd.set_option("display.width", 250)
    print("\n== tuning seasons, best 12 ==")
    print(tune.head(12).round(3).to_string(index=False))
    best = tune.iloc[0]
    held = r[(r["part"] == "held") & (r["half_life"] == best["half_life"])
             & (r["offseason"] == best["offseason"])
             & (r["ridge"] == best["ridge"])]
    cur = r[(r["part"] == "held") & (r["half_life"] == 45.0)
            & (r["offseason"] == 60.0) & (r["ridge"] == 3.0)]
    print("\n== held out: chosen vs current (45/60/3) ==")
    print(pd.concat([held, cur]).round(3).to_string(index=False))
    print(f"\nCHOSEN half_life={best['half_life']:g} "
          f"offseason_days={best['offseason']:g} ridge={best['ridge']:g}")


if __name__ == "__main__":
    main()
