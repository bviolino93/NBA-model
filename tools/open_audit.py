"""
Are ESPN's opening lines real openers? model_lab showed our line winning
60-68% against them, which would be a famous edge if it were true. Before
believing it, check the data:

  1. by season: how far the "open" sits from the close, how often the
     favourite flips, and how often the move is 3+ points (real NBA
     openers move a point or so; injury news moves a few by 3+)
  2. this season: ESPN's open and close next to our own Odds API
     snapshots (first and last pregame median across books) for the
     same games
  3. our line against the opener, keeping only games where the move to
     the close was small (a sign error or a stale number makes huge moves)
     and, separately, against our own first logged snapshot (a number we
     really could have bet), graded by the final score

    python tools/open_audit.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nba_predict as npr  # noqa: E402
from nba_data import load_games  # noqa: E402
from nba_players import load_box  # noqa: E402

BE = 110 / 210


def rec(label, pred, line, act):
    ok = np.isfinite(pred) & np.isfinite(line) & np.isfinite(act)
    res = np.sign((pred - line) * (act - line))[ok]
    gap = np.abs(pred - line)[ok]
    out = []
    for t in (0, 1, 2, 3):
        r = res[(gap >= max(t, 1e-9)) & (res != 0)]
        if len(r):
            w = (r > 0).mean()
            out.append(f"{t}+: {len(r)} at {w:.1%} (se {np.sqrt(w*(1-w)/len(r)):.1%})")
    print(f"  {label:34s} " + " | ".join(out))


def snapshots(path_glob="data/lines/games_*.csv") -> pd.DataFrame:
    """Per event: first and last pregame median home spread and total."""
    files = glob.glob(path_glob)
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d["market"].isin(["spreads", "totals"])]
    d["snap"] = pd.to_datetime(d["snapshot_utc"], utc=True)
    d["tip"] = pd.to_datetime(d["commence_time"], utc=True)
    d = d[d["snap"] < d["tip"]]
    sp = d[(d["market"] == "spreads") & (d["outcome"] == d["home_team"])]
    sp = sp.assign(v=-sp["point"])          # home -5 -> home margin +5
    tt = d[(d["market"] == "totals") & (d["outcome"].str.lower() == "over")]
    tt = tt.assign(v=tt["point"])
    out = []
    for m, x in (("spread", sp), ("total", tt)):
        med = (x.groupby(["event_id", "home_team", "away_team", "tip", "snap"])
               ["v"].median().reset_index().sort_values("snap"))
        g = med.groupby(["event_id", "home_team", "away_team", "tip"])["v"]
        out.append(pd.DataFrame({f"{m}_first": g.first(), f"{m}_last": g.last()}))
    return pd.concat(out, axis=1).reset_index()


def main():
    g = load_games(2023)
    g = g[g["season_type"].isin(["REG", "PLAYIN"])].reset_index(drop=True)
    done = g[g["completed"]].copy()
    done["move"] = done["spread_line"] - done["spread_open"]
    done["tmove"] = done["total_line"] - done["total_open"]
    print("1. ESPN open vs close, by season (spreads; totals in brackets)")
    for s, x in done.groupby("season"):
        k = x["spread_open"].notna()
        if not k.any():
            print(f"  {s}: no openers")
            continue
        mv = x.loc[k, "move"].abs()
        flip = (np.sign(x.loc[k, "spread_open"]) * np.sign(x.loc[k, "spread_line"]) < 0) \
            & (x.loc[k, "spread_open"].abs() >= 3)
        tm = x["tmove"].abs().dropna()
        print(f"  {s}: {k.sum():4d} games, median |move| {mv.median():.1f}, "
              f"mean {mv.mean():.2f}, 3+ moves {(mv >= 3).mean():.0%}, "
              f"favourite flips (open 3+) {flip.mean():.1%}  "
              f"[totals median {tm.median():.1f}, 3+ {(tm >= 3).mean():.0%}]")

    snap = snapshots()
    if not snap.empty:
        snap["game_date"] = (snap["tip"] - pd.Timedelta(hours=5)).dt.date
        m = done.merge(snap, on=["game_date", "home_team", "away_team"])
        print(f"\n2. this season, {len(m)} games logged by us and by ESPN")
        for a, b in (("spread_open", "spread_first"), ("spread_line", "spread_last"),
                     ("total_open", "total_first"), ("total_line", "total_last")):
            dlt = (m[a] - m[b]).dropna()
            if len(dlt):
                print(f"  ESPN {a:12s} - ours {b:13s}: median {dlt.median():+.2f}, "
                      f"mean |diff| {dlt.abs().mean():.2f}, 2+ apart "
                      f"{(dlt.abs() >= 2).mean():.0%} (n {len(dlt)})")

    print("\n3. our line (trees, all inputs) against openers")
    box = load_box(g)
    h = npr.build(g, box, first_season=2024)
    h = h[h["completed"] & h["model_margin"].notna()]
    preds = []
    for s in sorted(h["season"].unique()):
        if s < 2025:
            continue
        m = npr.fit_models(h[h["season"] < s])
        te = h[h["season"] == s]
        preds.append(te.assign(**npr.predict_frame(m, te)))
    p = pd.concat(preds)
    am = (p["home_score"] - p["away_score"]).to_numpy(float)
    at = (p["home_score"] + p["away_score"]).to_numpy(float)
    pm, pt = p["pred_margin"].to_numpy(float), p["pred_total"].to_numpy(float)
    rec("spreads vs ESPN close", pm, p["spread_line"].to_numpy(float), am)
    rec("spreads vs ESPN open", pm, p["spread_open"].to_numpy(float), am)
    small = (p["spread_line"] - p["spread_open"]).abs() <= 1.5
    rec("  ... open, move to close <= 1.5", np.where(small, pm, np.nan),
        p["spread_open"].to_numpy(float), am)
    rec("  ... the CLOSE on those same games", np.where(small, pm, np.nan),
        p["spread_line"].to_numpy(float), am)
    rec("totals vs ESPN open", pt, p["total_open"].to_numpy(float), at)
    if not snap.empty:
        q = p.merge(snap, on=["game_date", "home_team", "away_team"])
        if len(q):
            aq = (q["home_score"] - q["away_score"]).to_numpy(float)
            tq = (q["home_score"] + q["away_score"]).to_numpy(float)
            print(f"  this season, our own snapshots ({len(q)} games):")
            rec("spreads vs our first snapshot", q["pred_margin"].to_numpy(float),
                q["spread_first"].to_numpy(float), aq)
            rec("spreads vs our last snapshot", q["pred_margin"].to_numpy(float),
                q["spread_last"].to_numpy(float), aq)
            rec("totals vs our first snapshot", q["pred_total"].to_numpy(float),
                q["total_first"].to_numpy(float), tq)
            mv = q["spread_last"] - q["spread_first"]
            gp = q["pred_margin"] - q["spread_first"]
            ok = mv.notna() & gp.notna() & (mv != 0)
            print(f"  first->last snapshot moved toward us "
                  f"{(np.sign(mv[ok]) == np.sign(gp[ok])).sum()} of {ok.sum()}")


if __name__ == "__main__":
    main()
