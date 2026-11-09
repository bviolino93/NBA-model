"""
How often each betting rule would have picked, and how those picks did
against the sharp close, over every logged snapshot. For the question
"what would it take to get a real pick every day". Runs on Actions:
"Research script", script pick_rate.py.

Rules, all on DraftKings / FanDuel spreads and totals within 8 h of tip
(the live alert window), one side per game and market:

  ev >= t     the live rule at threshold t (first snapshot it qualifies)
  top N       the N best offers each day by EV, whatever the EV

Fair line from sharp books only (as live) and, for comparison, from every
book (the old rule).

    python tools/pick_rate.py [--hours 8]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clv  # noqa: E402
import selector as sel  # noqa: E402

THRESHOLDS = (-1.0, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03)
TOP_N = (1, 2, 3)


def offers(snaps, mm, tm, books, hours, sharp):
    rows = []
    src = snaps[snaps["_src"] == "games"] if "_src" in snaps else snaps
    for at, s in src.groupby("_at", sort=True):
        s = s.drop(columns=["_tip", "_at", "_src"], errors="ignore")
        fair = sel.consensus(s, mm, tm, sharp_first=sharp)
        pr = sel.price_offers(s, fair, mm, tm,
                              sel.Policy(min_ev=0, require_sharp=sharp))
        if pr.empty:
            continue
        pr = pr[pr["market"].isin(["spreads", "totals"]) & pr["book"].isin(books)]
        tip = pd.to_datetime(pr["commence"], utc=True)
        pr = pr.assign(at=at, tip=tip, hours=(tip - at).dt.total_seconds() / 3600)
        rows.append(pr[(pr["hours"] > 0) & (pr["hours"] <= hours)])
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    d["day"] = d["tip"].dt.tz_convert("US/Eastern").dt.date
    return d


def scorer(snaps, mm, tm):
    cache = {}

    def f(r):
        close = clv.close_rows(snaps, r.home, r.away, r.commence, r.at)
        if close.empty:
            return {}
        k = (r.home, r.away, close["_at"].iloc[0])
        if k not in cache:
            cache[k] = clv.close_fair(close, mm, tm)
        if cache[k] is None:
            return {}
        return clv.score(r.market, r.pick, float(r.point), float(r.price),
                         cache[k], close, mm, tm)
    return f


def stats(label, p, days, score):
    if p.empty:
        return f"  {label:12s} no picks"
    s = pd.DataFrame([score(r) for r in p.itertuples()])
    ev = s["clv_ev"].dropna() if "clv_ev" in s else pd.Series(dtype=float)
    pts = s["clv_pts"].dropna() if "clv_pts" in s else pd.Series(dtype=float)
    se = ev.std(ddof=1) / np.sqrt(len(ev)) if len(ev) > 1 else np.nan
    out = (f"  {label:12s} picks {len(p):4d}  days with a pick "
           f"{p['day'].nunique():2d}/{days}  per day {len(p) / days:4.1f}  "
           f"EV claimed {p['ev_market'].mean():+.1%}")
    if len(ev):
        out += (f"  graded {len(ev):3d}  EV at close {ev.mean():+.1%} "
                f"(se {se:.1%})  beat close {(ev > 0).mean():.0%}  "
                f"pts {pts.mean():+.2f}")
    return out


def one_side(d):
    """Best side per game and market (the live select)."""
    return (d.sort_values("ev_market", ascending=False)
             .drop_duplicates(["event_id", "market"]))


def main():
    from nba_data import load_games
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--books", default="draftkings,fanduel")
    a, _ = ap.parse_known_args()
    snaps = clv.load_snapshots()
    mm, tm = sel.build_outcome_models(load_games())
    books = tuple(b.strip() for b in a.books.split(","))
    score = scorer(snaps, mm, tm)
    for sharp in (True, False):
        d = offers(snaps, mm, tm, books, a.hours, sharp)
        print(f"== fair line from {'sharp books' if sharp else 'every book'}: "
              f"{d['at'].nunique() if len(d) else 0} snapshots within "
              f"{a.hours:.0f} h of tip ==")
        if d.empty:
            continue
        days = d["day"].nunique()
        for t in THRESHOLDS:
            q = d[d["ev_market"] >= t].sort_values("at")
            q = q.drop_duplicates(["event_id", "market", "pick"])
            q = q.drop_duplicates(["event_id", "market"])
            print(stats("any offer" if t < -0.5 else f"ev >= {t:.1%}",
                        q, days, score))
        best = one_side(d)
        for n in TOP_N:
            q = (best.sort_values("ev_market", ascending=False)
                     .groupby("day").head(n))
            print(stats(f"top {n}/day", q, days, score))
        top = best.sort_values("ev_market", ascending=False).groupby("day").head(1)
        print("  best offer each day:")
        for r in top.sort_values("day").itertuples():
            print(f"    {r.day} {r.pick} {r.point:+g} ({r.price:+g}) {r.book} "
                  f"{r.away}@{r.home} EV {r.ev_market:+.1%} {r.hours:.1f} h out")


if __name__ == "__main__":
    main()
