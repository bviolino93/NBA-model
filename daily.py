"""
The daily picks: DAILY_PICKS small real bets on every day with games,
even when nothing clears the regular bar.

Regular bets need a DraftKings / FanDuel line 2% or more off the sharp
market, and inside 8 hours of tip that almost never happens (replaying
Nov 1-8 2026, tools/pick_rate.py: 2 days of 7 had anything at 0% or
better; the day's best offer averaged about -0.5% against the sharp line).
So the daily pick is the best-priced line of the day where our own model
(nba_model, the no-money test line) leans the same way. It is close to a
fair bet on price, and the model lean is the part being tested.

It is staked flat at DAILY_STAKE (the size of the largest regular bet) and
kept in its own ledger (site/daily.json), with its own record and CLV, so
it never muddies the regular record. Size it up only if it beats the
sharp close over a few hundred picks.

Rules:
  - up to DAILY_PICKS a day (ET date of tip), one per game; each is
    locked when first chosen, and later runs only top the day up
    (games tipping later come inside the window on later runs)
  - candidates: spreads and totals at your books, tipping within
    ALERT_HOURS, with a sharp quote on the market (selector.price_offers)
  - not a game and market that already has a regular bet
  - an offer the model leans to, priced within PRICE_FLOOR of fair, first;
    then the best price against the sharp line. When no offer has both,
    the best price alone
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

import ledger
import selector as sel

DAILY_STAKE = 5.0           # dollars, flat, on the $1,000 bankroll
DAILY_PICKS = 3             # a day, one per game
# The model's lean only wins over a better price when its own offer is
# close to fair: at -110 with no edge an offer is about -4.5%, and an
# unproven lean is not worth paying the full vig for.
PRICE_FLOOR = -0.025
MARKETS = ("spreads", "totals")


def _day(commence) -> str:
    return (pd.Timestamp(commence).tz_convert("America/New_York")
            .strftime("%Y-%m-%d"))


def lean(r, fair: dict, lines: dict) -> float:
    """How far our model sits from the market on this offer's side, in
    points (positive = the model likes this side). NaN with no model line."""
    f, m = fair.get(r.event_id), lines.get(r.event_id)
    if f is None or not m:
        return float("nan")
    if r.market == "spreads":
        d = m.get("margin", float("nan")) - f.market_margin
        return d if r.pick == f.home_team else -d
    d = m.get("total", float("nan")) - f.market_total
    return d if str(r.pick).lower() == "over" else -d


def candidates(priced: pd.DataFrame, fair: dict, lines: dict, books,
               now: datetime, hours: float) -> pd.DataFrame:
    """Offers that could be the daily pick, best first within each day."""
    if priced is None or priced.empty:
        return pd.DataFrame()
    now = pd.Timestamp(now)
    tip = pd.to_datetime(priced["commence"], utc=True, errors="coerce")
    c = priced[priced["market"].isin(MARKETS) & priced["book"].isin(books)
               & (tip > now) & (tip <= now + pd.Timedelta(hours=hours))
               & priced["ev_market"].notna()].copy()
    if c.empty:
        return c
    c["lean"] = [lean(r, fair, lines) for r in c.itertuples()]
    c["agrees"] = (c["lean"] > 0) & (c["ev_market"] >= PRICE_FLOOR)
    c["day"] = [_day(x) for x in c["commence"]]
    return c.sort_values(["day", "agrees", "ev_market"],
                         ascending=[True, False, False])


def _record(r, books_name: dict, run: str) -> dict:
    pp = float(r.p_push) if math.isfinite(r.p_push) else 0.0
    b = sel.payout(r.price)
    pw = (float(r.ev_market) + 1.0 - pp) / (b + 1.0)   # at the sharp line
    pt = f"{r.point:+g}" if r.market == "spreads" else f"{r.point:g}"
    if r.agrees:
        why = (f"Daily pick: one of today's best-priced lines that our model "
               f"also likes ({r.lean:.1f} pts its way). The price is "
               f"{r.ev_market * 100:+.1f}% against the sharp line.")
    else:
        why = (f"Daily pick: one of today's best-priced lines (the model's "
               f"other sides aren't priced near fair today). The price is "
               f"{r.ev_market * 100:+.1f}% against the sharp line.")
    return {
        "away": r.away, "home": r.home, "commence": str(r.commence),
        "market": r.market, "pick": r.pick, "point": float(r.point),
        "price": float(r.price), "book": books_name.get(r.book, r.book),
        "stake": DAILY_STAKE, "win_chance": round(pw / max(1e-9, 1 - pp), 4),
        "ev": round(float(r.ev_market), 4),
        "fair_market": float(r.fair_market),
        "lean": None if not math.isfinite(r.lean) else round(float(r.lean), 2),
        "model_agrees": bool(r.agrees), "day": r.day,
        "label": f"{r.pick} {pt} ({int(r.price):+d})", "why": why,
        "first_seen": run, "first_point": float(r.point),
        "first_price": float(r.price), "result": None, "profit": None,
        "pulled": False,
    }


def choose(history: list[dict], regular: list[dict], priced: pd.DataFrame,
           fair: dict, lines: dict, books, now: datetime, hours: float,
           books_name: dict | None = None,
           per_day: int | None = None) -> list[dict]:
    """New daily picks, topping each day up to `per_day`, one per game."""
    per_day = DAILY_PICKS if per_day is None else per_day
    c = candidates(priced, fair, lines, books, now, hours)
    if c.empty:
        return []
    game = lambda b: (b["away"], b["home"], b["day"])  # noqa: E731
    count = {}
    for b in history:
        count[b["day"]] = count.get(b["day"], 0) + 1
    used = {game(b) for b in history}
    taken = {ledger._game(b) for b in regular if not b.get("pulled")}
    new = []
    for day, g in c.groupby("day", sort=True):
        for r in g.itertuples():
            if count.get(day, 0) >= per_day:
                break
            rec = _record(r, books_name or {}, pd.Timestamp(now).isoformat())
            if ledger._game(rec) in taken or game(rec) in used:
                continue
            new.append(rec)
            used.add(game(rec))
            count[day] = count.get(day, 0) + 1
    return new


def render_md(history: list[dict]) -> str:
    out = ["## Daily picks\n"]
    upcoming = [b for b in history if not b.get("result")]
    for b in sorted(upcoming, key=lambda b: b["commence"]):
        out.append(f"**{b['label']} at {b['book']}**, ${b['stake']:,.2f} · "
                   f"{b['away']} @ {b['home']}\n\n{b['why']}\n")
    if not upcoming:
        out.append("Today's picks go out from the first run within 8 hours "
                   "of tip.\n")
    s = ledger.summary(history)
    if s["wins"] + s["losses"] + s["pushes"]:
        roi = f"{s['roi'] * 100:+.1f}%" if s["roi"] is not None else "n/a"
        out.append(f"Daily-pick record: {s['wins']}-{s['losses']}-"
                   f"{s['pushes']}, ${s['profit']:+,.2f} (ROI {roi}).")
    if s.get("clv_n"):
        out.append(f" Against the sharp close ({s['clv_n']} picks): "
                   f"{s['beat_close'] * 100:.0f}% beat it, average EV "
                   f"{s['avg_clv_ev'] * 100:+.1f}%.")
    out.append(f"\nUp to {DAILY_PICKS} flat small bets a day, kept apart "
               "from the regular record. They get bigger only if they beat "
               "the sharp close.\n")
    return "\n".join(out)


def render_alert(bets: list[dict]) -> str:
    out = []
    for b in bets:
        out.append(f"**Daily pick: {b['label']}** at {b['book']}, "
                   f"${b['stake']:,.2f} · {b['away']} @ {b['home']}, "
                   + pd.Timestamp(b["commence"]).tz_convert("America/New_York")
                   .strftime("%a %-I:%M %p ET"))
    return "\n".join(out) + ("\n" if out else "")
