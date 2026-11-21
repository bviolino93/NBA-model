"""
Which inputs and which model forms predict NBA games best, judged two
ways on seasons the models never saw:

  1. accuracy against the final score: mean absolute error of the home
     margin and the total, next to the closing line's own error
  2. against the closing line: does the model know anything the line
     doesn't? The slope of (final - close) on (model - close) with its
     t-stat (0 = no information beyond the line), and the betting record
     at each size of disagreement and for the top N disagreements a night
     (52.4% breaks even at -110)

Each test season is predicted by models fitted on every earlier season
(the features themselves are walk-forward too). Variants, each adding to
the one before:

  ratings      final-score team ratings alone (the old baseline)
  +schedule    rest, back-to-backs, density, travel, time zones, altitude
  +possessions pace and efficiency ratings, four factors
  +players     player impact and box value over tonight's players
  trees        the full set, ridge plus boosted trees on its misses
  blend        trees and the closing line, weighted on training seasons
               (not a model of our own; shows how much the line adds)

    python tools/model_lab.py --first 2017 --test 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nba_predict as npr  # noqa: E402
from nba_data import load_games  # noqa: E402
from nba_players import load_box  # noqa: E402

BREAK_EVEN = 110 / 210
GAPS = {"spreads": (0, 1, 2, 3, 4, 6), "totals": (0, 2, 3, 4, 6, 8)}
TOPS = (1, 2, 3, 5)

SCHED_M = ["rest_diff", "b2b_diff", "away_b2b_road", "last5_diff", "km_diff",
           "away_tz_abs", "alt_home", "early", "neutral_f"]
SCHED_T = ["b2b_sum", "last5_sum", "km_sum", "alt_home", "early", "month"]
POSS_M = ["eff_margin", "ff_efg_m", "ff_tov_m", "ff_orb_m", "ff_ftr_m",
          "ff_3p_m"]
POSS_T = ["eff_total", "pace_hat", "ff_efg_s", "ff_tov_s", "ff_orb_s",
          "ff_ftr_s", "ff_3p_s"]
PLAY_M = ["apm_diff", "box_diff"]
PLAY_T = ["apm_sum", "box_sum"]
VARIANTS = {
    "ratings": (["model_margin"], ["model_total"], False),
    "+schedule": (["model_margin"] + SCHED_M, ["model_total"] + SCHED_T, False),
    "+possessions": (["model_margin"] + SCHED_M + POSS_M,
                     ["model_total"] + SCHED_T + POSS_T, False),
    "+players": (["model_margin"] + SCHED_M + POSS_M + PLAY_M,
                 ["model_total"] + SCHED_T + POSS_T + PLAY_T, False),
    "players only": (["apm_diff", "box_diff"] + SCHED_M,
                     ["model_total"] + SCHED_T + PLAY_T, False),
    "trees": (npr.MARGIN_F, npr.TOTAL_F, True),
}


def record(pred, line, actual):
    gap = pred - line
    res = np.sign(gap * (actual - line))
    return gap, res


def win_line(label, res, nights, seasons):
    keep = res != 0
    r, s = res[keep], seasons[keep]
    n = len(r)
    if not n:
        return f"    {label:16s} no bets"
    w = (r > 0).mean()
    se = np.sqrt(w * (1 - w) / n)
    per = " ".join(f"{(r[s == x] > 0).mean():.0%}" for x in sorted(set(s)))
    flag = "ABOVE" if w - 2 * se > BREAK_EVEN else ("above" if w > BREAK_EVEN
                                                     else "below")
    return (f"    {label:16s} {n:5d} bets {n / nights:4.1f}/night  win {w:.1%} "
            f"(se {se:.1%}) {flag} 52.4%   by season {per}")


def info(pred, line, actual):
    """Slope and t of (actual - line) on (pred - line)."""
    x, y = pred - line, actual - line
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 50 or x.std() == 0:
        return float("nan"), float("nan")
    xc = x - x.mean()
    b = (xc @ (y - y.mean())) / (xc @ xc)
    resid = y - y.mean() - b * xc
    se = np.sqrt(resid @ resid / (len(x) - 2) / (xc @ xc))
    return b, b / se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2017)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--apm-ridge", type=float, default=None)
    ap.add_argument("--apm-half-life", type=float, default=None)
    ap.add_argument("--save", default="")
    ap.add_argument("--load", default="")
    ap.add_argument("--out", default="", help="write per-game held-out predictions here")
    a, _ = ap.parse_known_args()
    t0 = time.time()
    cfg = npr.PredictConfig()
    if a.apm_ridge:
        cfg.apm_ridge = a.apm_ridge
    if a.apm_half_life:
        cfg.apm_half_life = a.apm_half_life
    if a.load:
        h = pd.read_pickle(a.load)
    else:
        g = load_games(a.first - 1)
        g = g[g["season_type"].isin(["REG", "PLAYIN"])].reset_index(drop=True)
        print(f"loaded {len(g)} games ({time.time() - t0:.0f}s)", flush=True)
        box = load_box(g)
        print(f"box rows {len(box)}, games with box "
              f"{box['game_id'].nunique()} ({time.time() - t0:.0f}s)",
              flush=True)
        h = npr.build(g, box, cfg, first_season=a.first, verbose=True)
        print(f"features built ({time.time() - t0:.0f}s)", flush=True)
        if a.save:
            h.to_pickle(a.save)
    h = h[h["completed"] & (h["season"] >= a.first)
          & h["model_margin"].notna()].copy()
    cov = h.groupby("season").agg(games=("game_id", "size"),
                                  line=("spread_line", lambda s: s.notna().mean()),
                                  apm=("apm_diff", lambda s: s.notna().mean()),
                                  eff=("eff_margin", lambda s: s.notna().mean()))
    print("\ncoverage by season\n" + cov.round(3).to_string())
    seasons = sorted(h["season"].unique())
    tests = [s for s in seasons[-a.test:] if (h["season"] < s).sum() >= 2000]
    print(f"\ntest seasons {tests}; each fitted on seasons {seasons[0]}.."
          "the one before")

    preds = {k: [] for k in list(VARIANTS) + ["blend", "close"]}
    frames = []
    weights = None
    for s in tests:
        tr, te = h[h["season"] < s], h[h["season"] == s].copy()
        te = te[te["spread_line"].notna() & te["total_line"].notna()]
        frames.append(te)
        for name, (mc, tc, trees) in VARIANTS.items():
            m = npr.fit_models(tr, trees=trees, margin_cols=mc, total_cols=tc)
            p = npr.predict_frame(m, te, "trees" if trees else "ridge")
            preds[name].append(p)
            if name == "trees":
                weights = m
                # blend weights on the training seasons' own (in-sample)
                # predictions would overfit; use the season before instead
                last = tr[tr["season"] == tr["season"].max()]
                last = last[last["spread_line"].notna() & last["total_line"].notna()]
                m2 = npr.fit_models(tr[tr["season"] < tr["season"].max()],
                                    trees=True)
                if m2.margin and len(last) > 300:
                    pl = npr.predict_frame(m2, last)
                    bw = {}
                    for k, line, y in (("margin", "spread_line",
                                        last["home_score"] - last["away_score"]),
                                       ("total", "total_line",
                                        last["home_score"] + last["away_score"])):
                        X = np.column_stack([pl[f"pred_{k}"], last[line],
                                             np.ones(len(last))])
                        bw[k] = np.linalg.lstsq(X, y.to_numpy(float),
                                                rcond=None)[0]
                    preds["blend"].append(pd.DataFrame({
                        "pred_margin": bw["margin"][0] * p["pred_margin"]
                        + bw["margin"][1] * te["spread_line"] + bw["margin"][2],
                        "pred_total": bw["total"][0] * p["pred_total"]
                        + bw["total"][1] * te["total_line"] + bw["total"][2]},
                        index=te.index))
                    print(f"  {s}: blend weight on our line {bw['margin'][0]:.2f} "
                          f"(spread), {bw['total'][0]:.2f} (total)")
                else:
                    preds["blend"].append(pd.DataFrame(
                        {"pred_margin": te["spread_line"],
                         "pred_total": te["total_line"]}, index=te.index))
        preds["close"].append(pd.DataFrame({"pred_margin": te["spread_line"],
                                            "pred_total": te["total_line"]},
                                           index=te.index))
        print(f"  {s} done ({time.time() - t0:.0f}s)", flush=True)
    te = pd.concat(frames)
    nights = te["game_date"].nunique()
    am = (te["home_score"] - te["away_score"]).to_numpy(float)
    at = (te["home_score"] + te["away_score"]).to_numpy(float)
    sm = te["spread_line"].to_numpy(float)
    st = te["total_line"].to_numpy(float)
    ss = te["season"].to_numpy()
    print(f"\n{len(te)} test games, {nights} nights. Break-even at -110: "
          f"{BREAK_EVEN:.1%}")

    print("\nACCURACY vs final score (lower is better), and information "
          "beyond the closing line")
    print(f"  {'model':16s} {'margin MAE':>10s} {'total MAE':>10s}  "
          f"{'spread slope':>12s} {'t':>6s}  {'total slope':>11s} {'t':>6s}")
    P = {}
    for name, parts in preds.items():
        p = pd.concat(parts).reindex(te.index)
        P[name] = p
        pm, pt = p["pred_margin"].to_numpy(float), p["pred_total"].to_numpy(float)
        bm, tm = info(pm, sm, am)
        bt, tt = info(pt, st, at)
        print(f"  {name:16s} {np.nanmean(np.abs(pm - am)):10.2f} "
              f"{np.nanmean(np.abs(pt - at)):10.2f}  {bm:12.2f} {tm:6.1f}  "
              f"{bt:11.2f} {tt:6.1f}")
    print("  (slope 1 = the model's disagreement with the line is fully "
          "right, 0 = worthless; |t| > 2 is real)")

    print("\nMAE by season (margin / total)")
    for name in ("close", "ratings", "+players", "trees"):
        p = P[name]
        row = []
        for s in tests:
            k = ss == s
            row.append(f"{s}: {np.nanmean(np.abs(p['pred_margin'].to_numpy()[k] - am[k])):.2f}"
                       f"/{np.nanmean(np.abs(p['pred_total'].to_numpy()[k] - at[k])):.2f}")
        print(f"  {name:16s} " + "  ".join(row))

    for name in ("ratings", "+schedule", "+possessions", "+players",
                 "players only", "trees", "blend"):
        p = P[name]
        print(f"\n{name.upper()} against the closing line")
        for market, pred, line, act in (
                ("spreads", p["pred_margin"].to_numpy(float), sm, am),
                ("totals", p["pred_total"].to_numpy(float), st, at)):
            gap, res = record(pred, line, act)
            print(f"  {market}: mean |gap| {np.nanmean(np.abs(gap)):.2f}")
            for t in GAPS[market]:
                k = np.abs(gap) >= max(t, 1e-9)
                print(win_line(f"gap {t}+", res[k], nights, ss[k]))
            d = pd.DataFrame({"night": te["game_date"].to_numpy(),
                              "a": np.abs(gap), "res": res, "s": ss})
            d = d[np.isfinite(d["a"])].sort_values("a", ascending=False)
            for n in TOPS:
                top = d.groupby("night").head(n)
                print(win_line(f"top {n} a night", top["res"].to_numpy(),
                               nights, top["s"].to_numpy()))

    # Opening lines (ESPN carries them from about 2023-24): does our line
    # see where the market is going? If the gap to the opener predicts the
    # move to the close, betting early numbers captures closing-line value.
    if "spread_open" in te:
        for name in ("ratings", "+players", "trees"):
            p = P[name]
            print(f"\n{name.upper()} against the OPENING line")
            for market, pred, line, close, act in (
                    ("spreads", p["pred_margin"].to_numpy(float),
                     te["spread_open"].to_numpy(float), sm, am),
                    ("totals", p["pred_total"].to_numpy(float),
                     te["total_open"].to_numpy(float), st, at)):
                ok = np.isfinite(line)
                if ok.sum() < 200:
                    print(f"  {market}: only {ok.sum()} games with an opener")
                    continue
                gap = pred[ok] - line[ok]
                move = close[ok] - line[ok]
                b, t = info(pred[ok], line[ok], close[ok])
                toward = np.sign(gap) * np.sign(move)
                print(f"  {market}: {ok.sum()} games; line moved toward us "
                      f"{(toward > 0).sum()}, away {(toward < 0).sum()}; "
                      f"slope of move on gap {b:.2f} (t {t:.1f})")
                _, res = record(pred[ok], line[ok], act[ok])
                for th in GAPS[market][1:4]:
                    k = np.abs(gap) >= th
                    print(win_line(f"gap {th}+ at open", res[k],
                                   te.loc[ok, "game_date"].nunique(),
                                   ss[ok][k]))

    if weights is not None and weights.margin:
        print("\nLatest fitted ridge weights (points per 1 sd of input)")
        print("  margin:", npr.ridge_weights(weights.margin))
        print("  total: ", npr.ridge_weights(weights.total))
    if a.out:
        keep = [c for c in ("game_id", "season", "game_date", "home_team", "away_team",
                            "home_score", "away_score", "spread_line", "total_line",
                            "spread_open", "total_open", "rest_diff", "b2b_diff",
                            "home_b2b", "away_b2b", "early", "month")
                if c in te.columns]
        o = te[keep].copy()
        for name in ("ratings", "+players", "trees"):
            o[f"{name}_margin"] = P[name]["pred_margin"]
            o[f"{name}_total"] = P[name]["pred_total"]
        o.to_csv(a.out, index=False)
        print(f"wrote {len(o)} games to {a.out}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
