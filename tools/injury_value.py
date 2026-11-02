"""
Which measure of a missing player's value matches how the market prices
absences, and how results go?

For each measure (minutes lost, star count, plus-minus value, box
production value), home-minus-away:
  * open-to-close spread move on it (2023-24 on, where ESPN has opens):
    how many points the market moves per unit, and how well (t, R^2)
  * final margin on it, holding the walk-forward ratings fixed: what the
    model's fit sees
Ratings are built from scores, so a player out for weeks is partly in
them already; the line move isolates fresh news.

    python tools/injury_value.py --first 2019
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_data import load_games  # noqa: E402
from nba_players import absences, game_absences, load_box  # noqa: E402
from nba_ratings import walk_forward  # noqa: E402

MEASURES = {"minutes": "inj_min_diff", "stars (30+ min)": "inj_star_diff",
            "plus-minus value": "inj_val_diff", "box production": "inj_prod_diff"}


def ols(X: np.ndarray, y: np.ndarray):
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[m], y[m]
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    s2 = r @ r / (len(y) - X1.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b[1:], (b / se)[1:], r2, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    a = ap.parse_args()
    g = load_games(a.first)
    g = g[g["season_type"].isin(["REG", "PLAYIN"])]
    box = load_box(g)
    g = game_absences(g, absences(box))
    g = walk_forward(g)
    g = g[g["completed"]]
    move = (g["spread_line"] - g["spread_open"]).to_numpy(float)
    margin = (g["home_score"] - g["away_score"]).to_numpy(float)
    mm = g["model_margin"].to_numpy(float)
    has_open = np.isfinite(move)
    print(f"{len(g)} games, {has_open.sum()} with an opening spread\n")
    print(f"{'measure':18s} | move per unit (t, R2)        | margin per unit, "
          f"ratings held (t)")
    for name, col in MEASURES.items():
        x = g[col].to_numpy(float)
        b1, t1, r1, n1 = ols(x[has_open, None], move[has_open])
        b2, t2, r2, n2 = ols(np.column_stack([mm, x]), margin)
        print(f"{name:18s} | {b1[0]:+.3f} (t {t1[0]:+6.1f}, R2 {r1:.3f}) "
              f"| {b2[1]:+.3f} (t {t2[1]:+5.1f}), ratings {b2[0]:.2f}")
    # All together, so the best mix shows
    cols = list(MEASURES.values())
    X = g[cols].to_numpy(float)
    b, t, r, n = ols(X[has_open], move[has_open])
    print("\njoint, line move: " + ", ".join(
        f"{k} {bb:+.2f} (t {tt:+.1f})" for k, bb, tt in zip(MEASURES, b, t))
        + f"; R2 {r:.3f}")
    b, t, r, n = ols(np.column_stack([mm, X]), margin)
    print("joint, margin:    " + ", ".join(
        f"{k} {bb:+.2f} (t {tt:+.1f})" for k, bb, tt in
        zip(["ratings"] + list(MEASURES), b, t)) + f"; R2 {r:.3f}")
    rot = box[box["played"] & (box["min"] >= 15)]
    p36 = 36 * (rot["pts"] + 0.5 * rot["reb"] + 0.7 * rot["ast"]) / rot["min"]
    print(f"\nproduction per 36, rotation games: quartiles "
          f"{np.percentile(p36, [25, 50, 75]).round(1).tolist()}")


if __name__ == "__main__":
    main()
