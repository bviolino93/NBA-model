"""Early-line timing test: does DraftKings/FanDuel follow Pinnacle?

For every logged pre-tip snapshot where DK or FD hangs a spread 1+ point
away from Pinnacle's number in the same snapshot, the "pick" is the DK/FD
side Pinnacle points to. Graded by points of closing-line value: the book's
own last logged pre-tip number minus the number we'd have taken (positive
= the line moved our way). No result grading: CLV is the fast signal.

Also lists the open candidates in the latest snapshot.

    python tools/early_steam.py [--min-gap 1.0]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS = ("draftkings", "fanduel")


def load():
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "lines", "games_*.csv"))
                + glob.glob(os.path.join(ROOT, "data", "lines", "close_*.csv")))
    g = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = g[(g.market == "spreads") & (g.outcome == g.home_team)].copy()
    g["t"] = pd.to_datetime(g.snapshot_utc, utc=True)
    g["tip"] = pd.to_datetime(g.commence_time, utc=True)
    g = g[g.t < g.tip]
    g["hrs"] = (g.tip - g.t).dt.total_seconds() / 3600
    return g.drop_duplicates(["event_id", "book", "snapshot_utc"])


def signals(g, min_gap):
    pin = g[g.book == "pinnacle"][["event_id", "snapshot_utc", "point"]].rename(
        columns={"point": "pin"})
    s = g[g.book.isin(BOOKS)].merge(pin, on=["event_id", "snapshot_utc"])
    s["d"] = s.pin - s.point          # home spread: Pinnacle minus book
    s = s[s.d.abs() >= min_gap]
    # first time each book x game shows the gap = when we'd bet
    return s.sort_values("t").drop_duplicates(["event_id", "book"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--alert", default="", help="write open signals here (markdown) if any")
    a, _ = ap.parse_known_args()
    g = load()
    last = (g.sort_values("t").drop_duplicates(["event_id", "book"], keep="last")
            [["event_id", "book", "point", "t"]].rename(columns={"point": "close", "t": "tc"}))
    s = signals(g, a.min_gap).merge(last, on=["event_id", "book"], how="left")
    # Pinnacle's home number lower = home more favoured there, so DK/FD's
    # home number is cheap: take the home side (and the away side otherwise).
    s["side"] = np.where(s.d < 0, "home", "away")
    mv = s.close - s.point
    s["clv"] = np.where(s.side == "home", -mv, mv)  # home spread dropping = home got dearer
    done = s[s.tc > s.t]
    print(f"{g.event_id.nunique()} games logged; {len(s)} early-line signals "
          f"(DK/FD {a.min_gap:g}+ pts off Pinnacle in the same snapshot)")
    if len(done):
        print(f"graded on a later snapshot: {len(done)}; moved our way "
              f"{(done.clv > 0).sum()}, against {(done.clv < 0).sum()}, flat "
              f"{(done.clv == 0).sum()}; average {done.clv.mean():+.2f} pts "
              f"(median hours before tip at signal {done.hrs.median():.0f})")
        for b, x in done.groupby("book"):
            print(f"  {b:11s} {len(x):3d}  avg {x.clv.mean():+.2f} pts, our way {(x.clv > 0).mean():.0%}")
    now = g.t.max()
    open_ = s[(s.t == now)]
    print(f"\nOPEN in the latest snapshot ({now:%Y-%m-%d %H:%M} UTC): {len(open_)}")
    lines = []
    for _, r in open_.sort_values("tip").iterrows():
        team = r.home_team if r.side == "home" else r.away_team
        num = r.point if r.side == "home" else -r.point
        pnum = r.pin if r.side == "home" else -r.pin
        tip_et = (r.tip - pd.Timedelta(hours=5)).strftime("%a %-I:%M %p ET")
        lines.append(f"- **{team} {num:+.1f}** at {r.book}, {r.away_team} @ "
                     f"{r.home_team}, {tip_et}. Pinnacle has {pnum:+.1f}.")
        print("  " + lines[-1])
    if a.alert and lines:
        rec = (f"Record so far: {len(done)} graded, moved our way {(done.clv > 0).sum()}, "
               f"against {(done.clv < 0).sum()}, average {done.clv.mean():+.2f} pts."
               if len(done) else "")
        with open(a.alert, "w") as f:
            f.write("**Early-line pick (NBA, $5 test):** DraftKings/FanDuel is "
                    f"{a.min_gap:g}+ pts off Pinnacle. Bet before it moves.\n\n"
                    + "\n".join(lines) + "\n\n" + rec + "\n")


if __name__ == "__main__":
    main()
