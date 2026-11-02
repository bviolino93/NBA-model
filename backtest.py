"""
Walk-forward backtest against the closing line, the same test the NFL
model failed honestly: does (model - close) predict (result - close)?

Reports, split into tuning seasons and held-out seasons:

  * slope of (result - close) on (model - close), with its t-stat; a slope
    near zero means the close already knows what the model knows
  * ATS / over-under record when the model disagrees by `edge`+ points
  * MAE of model vs close
  * the same residual test for rest, back-to-backs and schedule density,
    the NBA's best-known non-rating factors

    python backtest.py --first 2016 --split 2022 [--out games.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from nba_data import load_games, schedule_features
from nba_ratings import RatingConfig, walk_forward


def slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """OLS slope of y on x (with intercept) and its t-stat."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30 or x.std() == 0:
        return float("nan"), float("nan"), int(len(x))
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / (len(x) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(b[1]), float(b[1] / se), int(len(x))


def record(model, close, result, edge: float) -> tuple[int, int, int]:
    """Wins, losses, pushes betting the model's side when |gap| >= edge."""
    gap = model - close
    m = np.isfinite(gap) & np.isfinite(result) & (np.abs(gap) >= edge)
    side = np.sign(gap[m])
    out = np.sign(result[m] - close[m]) * side
    return int((out > 0).sum()), int((out < 0).sum()), int((out == 0).sum())


def report(fr: pd.DataFrame, label: str, edge: float) -> list[str]:
    margin = (fr["home_score"] - fr["away_score"]).to_numpy(float)
    points = (fr["home_score"] + fr["away_score"]).to_numpy(float)
    sp, tl = fr["spread_line"].to_numpy(float), fr["total_line"].to_numpy(float)
    mm, mt = fr["model_margin"].to_numpy(float), fr["model_total"].to_numpy(float)
    out = [f"== {label} =="]
    for name, model, close, res in (("Spreads", mm, sp, margin),
                                    ("Totals", mt, tl, points)):
        b, t, n = slope(model - close, res - close)
        w, l, p = record(model, close, res, edge)
        ok = np.isfinite(model) & np.isfinite(close) & np.isfinite(res)
        mae_m = np.abs(model[ok] - res[ok]).mean() if ok.any() else np.nan
        mae_c = np.abs(close[ok] - res[ok]).mean() if ok.any() else np.nan
        pct = w / max(1, w + l)
        out.append(f"{name:8s} n={n:5d}  slope {b:+.3f} (t {t:+.2f})  "
                   f"edge>={edge:g}: {w}-{l}-{p} ({pct:.1%})  "
                   f"MAE model/close {mae_m:.2f}/{mae_c:.2f}")
    res_m = margin - sp
    res_t = points - tl
    for f in ("rest_diff", "b2b_diff", "away_b2b_road", "home_last5",
              "away_last5"):
        if f not in fr:
            continue
        x = fr[f].to_numpy(float)
        b, t, n = slope(x, res_m)
        bt, tt, _ = slope(x, res_t)
        out.append(f"  {f:14s} vs spread resid {b:+.3f} (t {t:+.2f})   "
                   f"vs total resid {bt:+.3f} (t {tt:+.2f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2016)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--split", type=int, default=2022)
    ap.add_argument("--edge", type=float, default=3.0)
    ap.add_argument("--half-life", type=float, default=RatingConfig.half_life)
    ap.add_argument("--lines", default="", help="closing-line archive CSV")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    g = load_games(a.first, a.last, lines_csv=a.lines or None)
    g = schedule_features(g)
    cfg = RatingConfig(half_life=a.half_life)
    fr = walk_forward(g, cfg, first_season=a.first + 1)
    fr = fr[fr["completed"] & (fr["season_type"] == "REG")]
    cover = fr[["spread_line", "total_line"]].notna().mean()
    print(f"{len(fr)} completed regular-season games; lines on "
          f"{cover['spread_line']:.0%} (spread) / {cover['total_line']:.0%} (total)")
    for lab, part in ((f"{a.first + 1}-{a.split - 1} (tuning)",
                       fr[fr["season"] < a.split]),
                      (f"{a.split}-now (held out)",
                       fr[fr["season"] >= a.split])):
        print("\n".join(report(part, lab, a.edge)))
        print()
    if a.out:
        fr.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
