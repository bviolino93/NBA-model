"""
Closing-line value for every bet, from the line snapshots in data/lines/.

A few hundred results are mostly luck; whether the price beat the sharp
closing line is the early read on an edge. The close for a game is its
last snapshot before tip that has a sharp book in it (log_lines.py takes
one in the half hour before each tip, --close), priced the same way as
the picks (selector.consensus, sharp books only). Each bet gets:

    close_fair   the sharp fair home margin (spreads) or total at the close
    clv_pts      points the close moved past the bet's number, its way
    clv_ev       EV of the bet as placed, priced at the close (the price
                 counts: -105 beats -115 on the same number)
    close_min    minutes before tip the close was taken

A bet is graded once its game has tipped, and only against a snapshot
taken after the bet was first seen.

    python clv.py site/history.json     # print CLV for a ledger file
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import selector as sel

LINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "lines")
PATTERNS = ("games_*.csv", "close_*.csv")
MATCH_HOURS = 12.0        # a bet's commence vs the feed's, same game


def load_snapshots(lines_dir: str = LINES_DIR) -> pd.DataFrame:
    """Every logged game-line row taken before its game's tip, with _tip,
    _at (snapshot time) and _src ("games" or "close")."""
    files = [f for p in PATTERNS
             for f in sorted(glob.glob(os.path.join(lines_dir, p)))]
    frames = [pd.read_csv(f).assign(_src=os.path.basename(f).split("_")[0])
              for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[d["market"].isin(["spreads", "totals", "h2h"])].copy()
    d["_tip"] = pd.to_datetime(d["commence_time"], utc=True, errors="coerce")
    d["_at"] = pd.to_datetime(d["snapshot_utc"], utc=True, errors="coerce")
    return d[d["_at"] < d["_tip"]]


def close_rows(snaps: pd.DataFrame, home: str, away: str, commence,
               after=None) -> pd.DataFrame:
    """The game's last pre-tip snapshot with a sharp quote, if it was taken
    after `after` (when the bet was first seen); else empty."""
    if snaps.empty:
        return snaps
    tip = pd.Timestamp(commence)
    tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
    g = snaps[(snaps["home_team"] == home) & (snaps["away_team"] == away)
              & ((snaps["_tip"] - tip).abs()
                 <= pd.Timedelta(hours=MATCH_HOURS))]
    sharp = g[g["book"].isin(sel.SHARP_BOOKS)]
    if sharp.empty:
        return sharp
    last = sharp["_at"].max()
    if after is not None:
        a = pd.Timestamp(after)
        a = a.tz_localize("UTC") if a.tzinfo is None else a.tz_convert("UTC")
        if last <= a:
            return sharp.iloc[0:0]
    return g[g["_at"] == last]


def points_gained(market: str, pick: str, point: float, f) -> float:
    """Points the fair close sits past the bet's number, the bet's way."""
    if market == "spreads":
        cm = f.market_margin
        return cm + point if pick == f.home_team else point - cm
    if market == "totals":
        ct = f.market_total
        return ct - point if str(pick).lower() == "over" else point - ct
    return float("nan")


def grade(bets: list[dict], snaps: pd.DataFrame, margin_model, total_model,
          now=None, pick_key: str = "pick", seen_key: str = "first_seen",
          price: float | None = None) -> list[dict]:
    """Add close_fair, clv_pts, clv_ev and close_min to every bet whose game
    has tipped and has a sharp close. `price` stands in for bets logged
    without one (the model test, at -110)."""
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    for b in bets:
        if b.get("clv_ev") is not None or b.get("market") not in ("spreads",
                                                                  "totals"):
            continue
        tip = pd.Timestamp(b["commence"])
        tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
        if tip > now:
            continue
        rows = close_rows(snaps, b["home"], b["away"], b["commence"],
                          b.get(seen_key))
        if rows.empty:
            continue
        f = close_fair(rows, margin_model, total_model)
        if f is None:
            continue
        b.update(score(b["market"], b[pick_key], float(b["point"]),
                       b.get("price", price) if price is None else price,
                       f, rows, margin_model, total_model))
    return bets


def close_fair(rows: pd.DataFrame, margin_model, total_model):
    """The sharp fair line (selector.Fair) from a game's closing rows."""
    fair = sel.consensus(rows.drop(columns=["_tip", "_at", "_src"],
                                   errors="ignore"), margin_model, total_model)
    return next(iter(fair.values()), None)


def score(market: str, pick: str, point: float, price: float, f, rows,
          margin_model, total_model) -> dict:
    """CLV fields for one bet against a closing Fair; {} when the close has
    no sharp quote for the market."""
    spread = market == "spreads"
    if not (f.sharp_margin if spread else f.sharp_total):
        return {}
    bet = {"market": market, "pick": pick, "point": point, "price": price}
    return {
        "close_fair": round(float(f.market_margin if spread
                                  else f.market_total), 2),
        "clv_pts": round(float(points_gained(market, pick, point, f)), 2),
        "clv_ev": round(float(sel.clv(bet, f, margin_model, total_model)), 4),
        "close_min": round((rows["_tip"].iloc[0] - rows["_at"].iloc[0])
                           .total_seconds() / 60.0),
    }


def summary(bets: list[dict]) -> dict:
    c = [b for b in bets if b.get("clv_ev") is not None and not b.get("pulled")]
    if not c:
        return {"clv_n": 0, "avg_clv_ev": None, "avg_clv_pts": None,
                "beat_close": None}
    ev = np.array([b["clv_ev"] for b in c], float)
    pts = np.array([b["clv_pts"] for b in c], float)
    return {"clv_n": len(c), "avg_clv_ev": round(float(ev.mean()), 4),
            "avg_clv_pts": round(float(pts.mean()), 2),
            "beat_close": round(float((ev > 0).mean()), 4)}


def main():
    from nba_data import load_games
    path = sys.argv[1] if len(sys.argv) > 1 else "site/history.json"
    with open(path) as fh:
        bets = json.load(fh).get("bets", [])
    mm, tm = sel.build_outcome_models(load_games())
    bets = grade(bets, load_snapshots(), mm, tm)
    for b in bets:
        if b.get("clv_ev") is not None:
            print(f"{b['commence'][:10]} {b['pick']} {b['point']} "
                  f"({b['price']}) at {b['book']}: close {b['close_fair']}, "
                  f"{b['clv_pts']:+.1f} pts, EV at close {b['clv_ev']:+.1%} "
                  f"({b['close_min']} min before tip)")
    print(summary(bets))


if __name__ == "__main__":
    main()
