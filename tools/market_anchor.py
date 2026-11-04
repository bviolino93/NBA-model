"""
Does our line add anything the market's doesn't already have?

The no-money test logs a bet when our line is 3+ points off the market.
Our total is built from scores alone, so it is pulled toward the league
average and "disagrees" toward the middle whenever the market posts an
extreme number. This checks that, and an alternative: fit the final
score on the market's closing line plus our features, so a gap only
appears where the features say something beyond the line.

For each test season, weights fit on earlier seasons only:
  raw       our line as now (nba_model.fit_weights)
  anchored  actual ~ market line + our features
Prints, by gap size: bets, win rate against the closing line, and for
totals the share that are unders on high totals / overs on low ones.

    python tools/market_anchor.py --first 2019 --test 3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_data import load_games  # noqa: E402
from nba_model import (ANCHOR_M, ANCHOR_T, MARGIN_X, TOTAL_X, anchor_cols,  # noqa: E402
                       anchor_parts, fit_anchor, fit_weights, history)
from nba_players import load_box  # noqa: E402


def pred(d, w, cols):
    return sum(w[c] * d[c].fillna(0.0) for c in cols) + w["const"]


def anchored(te, w, cols, line):
    x = te.copy()
    anchor_cols(x, x["spread_line"], x["total_line"])
    for c in cols:
        x[c] = x[c].fillna(0.0)
    return (line + sum(anchor_parts(w, x, cols).values())).to_numpy(float)


def report(name, p, line, actual, other=None):
    gap = p - line
    res = np.sign(gap) * (actual - line)
    print(f"  {name}: mean |gap| {np.abs(gap).mean():.2f}")
    for t in (0.25, 0.5, 0.75, 1, 2, 3):
        m = (np.abs(gap) >= t) & (res != 0)
        if not m.sum():
            continue
        win = (res[m] > 0).mean()
        extra = ""
        if other is None:                  # spreads: share on the favourite
            fav = np.sign(gap) == np.sign(line)
            extra = f" | on favourite {fav[m].mean():.0%}"
        else:
            mid = np.median(line)
            toward = ((gap < 0) & (line > mid)) | ((gap > 0) & (line < mid))
            extra = f" | toward average {toward[m].mean():.0%}"
        print(f"    gap {t:g}+: {m.sum():5d} bets, win {win:.1%}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    ap.add_argument("--test", type=int, default=3)
    a = ap.parse_args()
    g = load_games(a.first)
    g = g[g["season_type"].isin(["REG", "PLAYIN"])]
    h = history(g, load_box(g))
    h = h[h["completed"] & h["model_margin"].notna()].copy()
    h["mkt_margin"], h["mkt_total"] = h["spread_line"], h["total_line"]
    seasons = sorted(h["season"].unique())
    for s in seasons[-a.test:]:
        tr, te = h[h["season"] < s], h[h["season"] == s]
        te = te[te["spread_line"].notna() & te["total_line"].notna()]
        trm = tr[tr["spread_line"].notna() & tr["total_line"].notna()].copy()
        for c in set(MARGIN_X + TOTAL_X):
            trm[c] = trm[c].fillna(0.0)
        f = fit_weights(tr)
        am, at = fit_anchor(trm)
        margin = (te["home_score"] - te["away_score"]).to_numpy(float)
        total = (te["home_score"] + te["away_score"]).to_numpy(float)
        sl, tl = te["spread_line"].to_numpy(float), te["total_line"].to_numpy(float)
        print(f"\n{s}: {len(te)} games (fit on {len(trm)})")
        print("  anchored margin weights: "
              + ", ".join(f"{k} {v:+.2f}" for k, v in am.items() if ":" not in k))
        print("  anchored total weights:  "
              + ", ".join(f"{k} {v:+.2f}" for k, v in at.items() if ":" not in k))
        print(" SPREADS")
        report("raw", pred(te, f.margin, MARGIN_X).to_numpy(float), sl, margin)
        report("anchored", anchored(te, am, ANCHOR_M, sl), sl, margin)
        print(" TOTALS")
        report("raw", pred(te, f.total, TOTAL_X).to_numpy(float), tl, total, 1)
        report("anchored", anchored(te, at, ANCHOR_T, tl), tl, total, 1)


if __name__ == "__main__":
    main()
