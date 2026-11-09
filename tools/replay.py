"""
Replay the stale-line selector over every logged snapshot and score what
it would have flagged against the sharp closing line. Runs on Actions:
"Research script", script replay.py, arguments --min-ev 0.02.

The real ledger gets a handful of alerts a week, too few to judge. Every
logged snapshot (data/lines/games_*.csv, taken twice a day) is priced the
way picks.py prices the live feed, and every DraftKings / FanDuel offer it
flags at min_ev or better gets its EV at the close (clv.py), counted once
per game, market and side, at the first snapshot that flagged it. Two fair
lines are compared:

  sharp   the fair line from sharp books only (selector, since the review)
  all     every book's line, sharp books weighted up (before the review)

Also shown: every DraftKings / FanDuel offer, flagged or not, both sides,
as the baseline: its EV at the close should be about the vig (-4 to -5%)
and its points average to zero.

    python tools/replay.py [--min-ev 0.02] [--books draftkings,fanduel]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clv  # noqa: E402
import selector as sel  # noqa: E402

BUCKETS = [(0, 8, "under 8 h to tip"), (8, 24, "8-24 h"), (24, 99, "24 h+")]


def replay(snaps: pd.DataFrame, mm, tm, books=("draftkings", "fanduel"),
           min_ev: float = 0.02, modes=("sharp", "all")) -> pd.DataFrame:
    """One row per flagged offer (first flag per game, market, side and
    mode) plus every offer at `books` as mode "baseline"; each with its
    EV at the sharp close."""
    rows = []
    src = snaps[snaps["_src"] == "games"] if "_src" in snaps else snaps
    for at, s in src.groupby("_at", sort=True):
        s = s.drop(columns=["_tip", "_at", "_src"], errors="ignore")
        for mode in modes:
            fair = sel.consensus(s, mm, tm, sharp_first=(mode == "sharp"))
            pol = sel.Policy(min_ev=min_ev, require_sharp=(mode == "sharp"))
            pr = sel.price_offers(s, fair, mm, tm, pol)
            if pr.empty:
                continue
            pr = pr[pr["market"].isin(["spreads", "totals"])
                    & pr["book"].isin(books)]
            if mode == modes[0]:
                rows.append(pr.assign(mode="baseline", at=at))
            rows.append(pr[pr["ev_market"] >= min_ev].assign(mode=mode, at=at))
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    flagged = d["mode"] != "baseline"
    d = pd.concat([d[~flagged],
                   d[flagged].sort_values("at").drop_duplicates(
                       ["mode", "event_id", "market", "pick"])])
    out, cache = [], {}
    for r in d.itertuples():
        tip = pd.Timestamp(r.commence)
        tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip
        b = {}
        if tip < pd.Timestamp.now(tz="UTC"):
            close = clv.close_rows(snaps, r.home, r.away, r.commence, r.at)
            if len(close):
                k = (r.home, r.away, close["_at"].iloc[0])
                if k not in cache:
                    cache[k] = clv.close_fair(close, mm, tm)
                if cache[k] is not None:
                    b = clv.score(r.market, r.pick, float(r.point),
                                  float(r.price), cache[k], close, mm, tm)
        out.append({"mode": r.mode, "market": r.market, "book": r.book,
                    "hours": (tip - r.at).total_seconds() / 3600,
                    "ev_claimed": r.ev_market, "clv_ev": b.get("clv_ev"),
                    "clv_pts": b.get("clv_pts"),
                    "close_min": b.get("close_min")})
    return pd.DataFrame(out)


def report(r: pd.DataFrame) -> str:
    lines = []
    r = r[r["clv_ev"].notna()]
    for mode, g in r.groupby("mode", sort=False):
        lines.append(f"== {mode}: {len(g)} offers with a close ==")
        for (lo, hi, lab) in [(0, 999, "all")] + BUCKETS:
            for mk in ("all", "spreads", "totals"):
                x = g[(g["hours"] >= lo) & (g["hours"] < hi)]
                if mk != "all":
                    x = x[x["market"] == mk]
                if len(x) == 0:
                    continue
                se = x["clv_ev"].std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                lines.append(
                    f"  {lab:16s} {mk:8s} n={len(x):4d}  EV claimed "
                    f"{x['ev_claimed'].mean():+.1%}  EV at close "
                    f"{x['clv_ev'].mean():+.1%} (se {se:.1%})  beat close "
                    f"{(x['clv_ev'] > 0).mean():.0%}  pts {x['clv_pts'].mean():+.2f}"
                    f"  close {x['close_min'].median():.0f} min before tip")
    return "\n".join(lines)


def main():
    from nba_data import load_games
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--books", default="draftkings,fanduel")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    snaps = clv.load_snapshots()
    mm, tm = sel.build_outcome_models(load_games())
    print(f"{snaps['_at'].nunique()} pre-tip snapshots, "
          f"{snaps['event_id'].nunique()} games")
    r = replay(snaps, mm, tm, tuple(b.strip() for b in a.books.split(",")),
               a.min_ev)
    if r.empty:
        print("nothing to score yet")
        return
    print(report(r))
    if a.out:
        r.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
