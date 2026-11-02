"""
Does a team's recent against-the-spread record predict its next cover?

For every game, each team's covers over its previous N games this season
(at ESPN's closing spread), home minus away. Tested walk-forward against
the next game's result vs the close: slope with t-stat, and the record
betting the hotter ATS team when the gap is big. Split into tuning and
held-out seasons, like the backtest.

    python tools/ats_trend.py --first 2016 --split 2022
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest import slope  # noqa: E402
from nba_data import load_games  # noqa: E402


def trends(g: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    g = g[g["completed"] & g["spread_line"].notna()].sort_values("commence")
    g = g.reset_index(drop=True)
    g["home_ats"] = (g["home_score"] - g["away_score"]) - g["spread_line"]
    rows = []
    for side, sign in (("home", 1), ("away", -1)):
        rows.append(pd.DataFrame({
            "idx": g.index, "season": g["season"], "team": g[f"{side}_team"],
            "commence": g["commence"], "ats": sign * g["home_ats"]}))
    t = pd.concat(rows).sort_values(["commence", "idx"])
    grp = t.groupby(["team", "season"])
    for n in windows:
        t[f"cov{n}"] = grp["ats"].transform(
            lambda s: (np.sign(s) > 0).astype(float)
            .rolling(n, min_periods=n).mean().shift(1))
        t[f"mar{n}"] = grp["ats"].transform(
            lambda s: s.rolling(n, min_periods=n).mean().shift(1))
    h = t[t["team"].values == g.loc[t["idx"], "home_team"].values].set_index("idx")
    a = t[t["team"].values == g.loc[t["idx"], "away_team"].values].set_index("idx")
    for n in windows:
        g[f"cov_diff{n}"] = h[f"cov{n}"] - a[f"cov{n}"]
        g[f"mar_diff{n}"] = h[f"mar{n}"] - a[f"mar{n}"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2016)
    ap.add_argument("--split", type=int, default=2022)
    a = ap.parse_args()
    g = trends(load_games(a.first))
    y = g["home_ats"].to_numpy(float)
    for lab, m in (("tuning", g["season"] < a.split),
                   ("held out", g["season"] >= a.split)):
        print(f"== {lab}: {int(m.sum())} games with a closing spread ==")
        for n in (5, 10, 20):
            for kind in ("cov", "mar"):
                x = g.loc[m, f"{kind}_diff{n}"].to_numpy(float)
                b, t, k = slope(x, y[m.to_numpy()])
                what = "cover rate" if kind == "cov" else "cover margin"
                print(f"  last {n:2d}, {what:12s} home-away: slope {b:+.3f} "
                      f"(t {t:+.1f}, n {k})")
            # Bet the hotter team when cover rates differ by 40+ points
            x = g.loc[m, f"cov_diff{n}"].to_numpy(float)
            r = y[m.to_numpy()]
            sel = np.isfinite(x) & (np.abs(x) >= 0.4) & (r != 0)
            won = (np.sign(x[sel]) == np.sign(r[sel])).sum()
            lost = sel.sum() - won
            pct = won / max(1, won + lost)
            print(f"  last {n:2d}: back the hotter ATS team (gap 40%+): "
                  f"{won}-{lost} ({pct:.1%}); fade it: {lost}-{won} "
                  f"({1 - pct:.1%}). Break-even 52.4%.")
        print()


if __name__ == "__main__":
    main()
