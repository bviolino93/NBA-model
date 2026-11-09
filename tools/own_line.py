"""
Our own line, with no market input at all: would betting it have won?

The model (nba_model.fit_weights: team ratings from scores, rest,
back-to-backs, injuries in minutes and production) is fit on earlier
seasons only and tested on each later one, against ESPN's pregame line.
For each gap size and for "top N gaps a night" it prints the record
against the line, the win rate (52.4% breaks even at -110) and how many
bets a night that rule makes. Pooled over the test seasons, with a
standard error, so one lucky season doesn't read as an edge.

    python tools/own_line.py --first 2017 --test 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_data import load_games  # noqa: E402
from nba_model import MARGIN_X, TOTAL_X, fit_weights, history  # noqa: E402
from nba_players import load_box  # noqa: E402

GAPS = (1, 2, 3, 4, 5, 6, 8)
TOPS = (1, 2, 3, 5)
BREAK_EVEN = 110 / 210


def pred(d, w, cols):
    return sum(w[c] * d[c].fillna(0.0) for c in cols) + w["const"]


def bets(te, market):
    """One row per game: gap, result against the line (+1/-1/0), night."""
    if market == "spreads":
        p = pred(te, te.attrs["f"].margin, MARGIN_X)
        line = te["spread_line"]
        actual = te["home_score"] - te["away_score"]
    else:
        p = pred(te, te.attrs["f"].total, TOTAL_X)
        line = te["total_line"]
        actual = te["home_score"] + te["away_score"]
    gap = p - line
    return pd.DataFrame({"season": te["season"], "night": te["game_date"],
                         "gap": gap, "res": np.sign(gap * (actual - line))})


def line(label, b, nights):
    b = b[b["res"] != 0]
    n = len(b)
    if not n:
        return f"    {label:14s} no bets"
    w = (b["res"] > 0).mean()
    se = np.sqrt(w * (1 - w) / n)
    per = ", ".join(f"{(g['res'] > 0).mean():.0%}"
                    for _, g in b.groupby("season"))
    return (f"    {label:14s} {n:5d} bets  win {w:.1%} (se {se:.1%})  "
            f"{'ABOVE' if w - 2 * se > BREAK_EVEN else 'not above'} break-even"
            f"  {n / nights:.1f} a night  by season {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2017)
    ap.add_argument("--test", type=int, default=5)
    a, _ = ap.parse_known_args()
    g = load_games(a.first)
    g = g[g["season_type"].isin(["REG", "PLAYIN"])]
    h = history(g, load_box(g))
    h = h[h["completed"] & h["model_margin"].notna()].copy()
    seasons = sorted(h["season"].unique())
    out = {"spreads": [], "totals": []}
    nights = 0
    for s in seasons[-a.test:]:
        tr, te = h[h["season"] < s], h[h["season"] == s]
        te = te[te["spread_line"].notna() & te["total_line"].notna()].copy()
        te.attrs["f"] = fit_weights(tr)
        nights += te["game_date"].nunique()
        for m in out:
            out[m].append(bets(te, m))
    print(f"test seasons {seasons[-a.test:]}, {nights} nights; "
          f"break-even at -110 is {BREAK_EVEN:.1%}")
    for m, parts in out.items():
        b = pd.concat(parts, ignore_index=True)
        b = b[b["gap"].notna()]
        print(f"\n{m.upper()}: mean |gap| {b['gap'].abs().mean():.2f}")
        for t in GAPS:
            print(line(f"gap {t}+", b[b["gap"].abs() >= t], nights))
        r = b.assign(a=b["gap"].abs()).sort_values("a", ascending=False)
        for n in TOPS:
            print(line(f"top {n} a night", r.groupby("night").head(n), nights))


if __name__ == "__main__":
    main()
