"""
NBA picks: bets at your books whose line is stale against the sharp market.

    ODDS_API_KEY=... python picks.py --md site/README.md --history site/history.json

The backtests (backtest.py, nba_props.py) found nothing that beats the
NBA closing line yet, so no model moves these picks. What is left is the
part of the NFL picks that needs no model: every book's spread and total
is de-vigged into a sharp-weighted fair line (Pinnacle counts 4x), and a
bet goes out only when DraftKings or FanDuel offers a number or price that
fair line says is +EV by at least min_ev. That is a soft book that hasn't
caught up with a move, the most reliable edge a non-syndicate bettor has.

Stakes are quarter Kelly, halved while unproven, capped at 1% a bet and
8% a night. The ledger grades every alerted bet against ESPN finals and
review.py-style CLV will judge it.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import ledger
import nba_injuries as inj_mod
import nba_model
import paper
import selector as sel
from log_lines import _rows
from nba_data import current_season, load_games

ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
MY_BOOKS = "draftkings,fanduel"
BOOK_NAME = {"draftkings": "DraftKings", "fanduel": "FanDuel",
             "betmgm": "BetMGM", "williamhill_us": "Caesars",
             "espnbet": "ESPN BET", "betrivers": "BetRivers",
             "fanatics": "Fanatics"}
DASHBOARD = "https://bviolino93.github.io/NBA-model/"
STAKE_SCALE = 0.5          # half stakes until the record says otherwise
HOURS_AHEAD = 30.0         # tonight's slate (and an early look at tomorrow's)
MARKETS = ("spreads", "totals")


def fetch_rows(key: str, regions: str = "us,eu") -> pd.DataFrame:
    r = requests.get(ODDS_URL, params={
        "apiKey": key, "regions": regions, "markets": "spreads,totals,h2h",
        "oddsFormat": "american"}, timeout=30)
    if r.status_code == 401:
        raise SystemExit(f"The Odds API rejected the key (401): {r.text[:200]}")
    r.raise_for_status()
    print(f"odds: {len(r.json())} events, credits left "
          f"{r.headers.get('x-requests-remaining')}")
    snap = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame([x for ev in r.json() for x in _rows(ev, snap)])


def upcoming(rows: pd.DataFrame, now: datetime | None = None,
             hours: float = HOURS_AHEAD) -> pd.DataFrame:
    """Games not yet tipped (live lines can't be priced) within `hours`."""
    now = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC")
    tip = pd.to_datetime(rows["commence_time"], utc=True, errors="coerce")
    return rows[(tip > now) & (tip < now + pd.Timedelta(hours=hours))]


def explain(p) -> str:
    """Why this offer is a bet: the book's number vs the fair number."""
    book = BOOK_NAME.get(p.book, p.book)
    if p.market == "totals":
        side = str(p.pick).lower()
        return (f"{book} has the total at {p.point:g} ({int(p.price):+d}); "
                f"the sharp market's fair total is {p.fair_market:.1f}, so "
                f"the {side} wins {p.win_chance * 100:.1f}% of the time it "
                f"doesn't push.")
    home = p.pick == p.home
    fair_pt = -p.fair_market if home else p.fair_market
    return (f"{book} has {p.pick} at {p.point:+g} ({int(p.price):+d}); the "
            f"sharp market makes it {fair_pt:+.1f}, so it covers "
            f"{p.win_chance * 100:.1f}% of the time it doesn't push.")


def build_picks(rows: pd.DataFrame, games: pd.DataFrame, my_books,
                bankroll: float = 1000.0, min_ev: float = 0.02,
                now: datetime | None = None,
                injuries: pd.DataFrame | None = None):
    """(picks, fair, priced) for upcoming games.

    injuries: nba_injuries.with_roles output. Rotation players ruled out
    are listed on the pick; ones questionable or day-to-day mark it
    "wait for news", since the line can still jump before tip."""
    rows = upcoming(rows, now)
    if rows.empty:
        return pd.DataFrame(), {}, pd.DataFrame()
    mm, tm = sel.build_outcome_models(games)
    fair = sel.consensus(rows, mm, tm)
    pol = sel.Policy(min_ev=min_ev)
    priced = sel.price_offers(rows, fair, mm, tm, pol)
    if priced.empty:
        return priced, fair, priced
    bettable = priced[priced["market"].isin(MARKETS)
                      & priced["book"].isin(my_books)
                      & (priced["ev_market"] >= min_ev)]
    picks = sel.select(bettable, bankroll, pol)
    if len(picks):
        picks["stake"] = (picks["stake"] * STAKE_SCALE).round(2)
        picks = picks[picks["stake"] >= 1.0].copy()
    if len(picks):
        picks["win_chance"] = picks["p_win"] / (1 - picks["p_push"])
        picks["why"] = [explain(p) for p in picks.itertuples()]
        notes = [inj_mod.game_notes(injuries, p.home, p.away)
                 for p in picks.itertuples()]
        picks["out"] = ["; ".join(n["out"]) for n in notes]
        picks["watch"] = ["; ".join(n["watch"]) for n in notes]
        picks["book"] = picks["book"].map(lambda b: BOOK_NAME.get(b, b))
    return picks, fair, priced


def _label(p) -> str:
    pt = f"{p.point:+g}" if p.market == "spreads" else f"{p.point:g}"
    return f"{p.pick} {pt} ({int(p.price):+d})"


def _tip(commence) -> str:
    return (pd.to_datetime(commence, utc=True).tz_convert("America/New_York")
            .strftime("%a %-I:%M %p ET"))


def _injury_md(p) -> str:
    out, watch = getattr(p, "out", ""), getattr(p, "watch", "")
    lines = []
    if watch:
        lines.append(f"⚠️ **Wait for news:** {watch}. The line can still "
                     f"move either way; the evening run re-checks it.")
    if out:
        lines.append(f"Already out (priced in): {out}.")
    return "\n".join(lines) + ("\n" if lines else "")


def render_md(picks: pd.DataFrame, made: str, history: list[dict]) -> str:
    out = [f"# NBA picks\n\n_Updated {made}_ · "
           f"[Dashboard]({DASHBOARD})\n"]
    if picks.empty:
        out.append("**No bets right now.** No DraftKings or FanDuel line is "
                   "far enough off the sharp market. That is the usual answer.\n")
    for p in picks.itertuples():
        out.append(f"### {_label(p)} at {p.book}\n\n"
                   f"**Stake ${p.stake:,.2f}** · win chance "
                   f"{p.win_chance * 100:.1f}% · EV {p.ev * 100:.1f}%  \n"
                   f"{p.away} @ {p.home} · {_tip(p.commence)}\n\n{p.why}\n"
                   + _injury_md(p))
    s = ledger.summary(history)
    graded = s["wins"] + s["losses"] + s["pushes"]
    if graded:
        roi = f"{s['roi'] * 100:+.1f}%" if s["roi"] is not None else "n/a"
        out.append(f"## Record\n\n{s['wins']}-{s['losses']}-{s['pushes']}, "
                   f"${s['profit']:+,.2f} on ${s['staked']:,.2f} staked "
                   f"(ROI {roi}).\n")
    out.append("---\nStale-line bets only: no NBA model has beaten the closing "
               "line yet. Quarter-Kelly, half stakes, on a $1,000 bankroll, "
               "capped at 1% a bet and 8% a night. 21+ where legal; "
               "1-800-GAMBLER.\n")
    return "\n".join(out)


def _clean(x):
    """NaN -> None, so the page's JSON.parse accepts the blob."""
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def render_html(picks: list[dict], fair: dict, made: str, bankroll: float,
                history: list[dict], tests: list[dict] = (),
                lines: dict | None = None) -> str:
    """The dashboard: dashboard.html with this run's data embedded, so the
    page works from GitHub Pages, the picks branch or a saved copy."""
    data = {
        "updated": made, "bankroll": bankroll, "picks": picks,
        "fair": [{"away": f.away_team, "home": f.home_team,
                  "commence": str(f.commence),
                  "market_margin": f.market_margin,
                  "market_total": f.market_total,
                  "model_margin": (lines or {}).get(k, {}).get("margin"),
                  "model_total": (lines or {}).get(k, {}).get("total"),
                  "parts": (lines or {}).get(k, {}).get("parts"),
                  "missing": (lines or {}).get(k, {}).get("missing")}
                 for k, f in fair.items()],
        "history": {"summary": ledger.summary(history), "bets": history},
        "paper": {"summary": paper.summary(list(tests)), "gap": paper.GAP,
                  "bets": list(tests)},
    }
    blob = json.dumps(_clean(json.loads(json.dumps(data, default=str))),
                      allow_nan=False).replace("</", "<\\/")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "dashboard.html")) as fh:
        return fh.read().replace("/*DATA*/", blob)


def render_alert(bets: list[dict]) -> str:
    out = [f"**{len(bets)} new NBA bet{'s' if len(bets) != 1 else ''}**\n"]
    for b in bets:
        p = pd.Series(b)
        out.append(f"- **{_label(p)}** at {p.book}, ${p.stake:,.2f} "
                   f"({p.win_chance * 100:.0f}% to win) · {p.away} @ "
                   f"{p.home}, {_tip(p.commence)}"
                   + (f"\n  ⚠️ wait for news: {p.watch}"
                      if p.get("watch") else ""))
    out.append(f"\nPicks: {DASHBOARD}")
    return "\n".join(out) + "\n"


def render_dropped(bets: list[dict]) -> str:
    one = len(bets) == 1
    out = [f"**Skip {'this' if one else 'these'} if you haven't bet "
           f"{'it' if one else 'them'}: {len(bets)} NBA bet"
           f"{'' if one else 's'} dropped off the board**\n"]
    for b in bets:
        p = pd.Series(b)
        out.append(f"- ~~{_label(p)} at {p.book}~~ · {p.away} @ {p.home}, "
                   f"{_tip(p.commence)}")
    out.append("\nThe book caught up with the market. If you already bet it, "
               "it still counts in the record at the alerted price.")
    return "\n".join(out) + "\n"


def load_box_recent(games: pd.DataFrame, seasons: int = 5):
    """Box scores for the last few seasons (cached), or None."""
    try:
        from nba_players import load_box
        return load_box(games[games["season"] > current_season() - seasons])
    except Exception as exc:
        print(f"box scores unavailable: {exc}")
        return None


def load_injuries(box) -> pd.DataFrame:
    """ESPN's report with each player's recent minutes; empty on failure."""
    rep = inj_mod.fetch_injuries()
    cur = None if box is None else box[box["season"] == current_season()]
    return inj_mod.with_roles(rep, inj_mod.recent_minutes(cur))


def model_lines(games, fair, box, injuries, now) -> dict:
    """Our own line per game (nba_model); ratings only if that fails."""
    try:
        recent = games[games["season"] > current_season() - 5]
        fitted = nba_model.fit_weights(nba_model.history(recent, box))
        print(f"model weights from {fitted.n} games: margin "
              f"{ {k: round(v, 2) for k, v in fitted.margin.items()} }, total "
              f"{ {k: round(v, 2) for k, v in fitted.total.items()} }")
        missing = nba_model.tonight_missing(
            injuries, nba_model.player_values(box))
        return nba_model.predict(games, fair, fitted, missing, now)
    except Exception as exc:
        print(f"full model failed, ratings only: {exc}")
        return paper.model_lines(games, fair, now)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument("--md", default="site/README.md")
    ap.add_argument("--json", default="site/picks.json")
    ap.add_argument("--history", default="site/history.json")
    ap.add_argument("--alert", default="alert.md")
    ap.add_argument("--paper", default="site/paper.json")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-ev", type=float, default=0.02)
    ap.add_argument("--books", default=os.environ.get("MY_BOOKS") or MY_BOOKS)
    ap.add_argument("--lines-csv", default="",
                    help="use a logged snapshot instead of the live feed")
    a = ap.parse_args()
    if a.lines_csv:
        rows = sel.latest_snapshot(pd.read_csv(a.lines_csv))
    else:
        key = (os.environ.get("ODDS_API_KEY") or "").strip().strip("\"'()")
        if not key:
            raise SystemExit("set ODDS_API_KEY")
        rows = fetch_rows(key)
    books = [b.strip() for b in a.books.split(",") if b.strip()]
    games = load_games()
    now = datetime.now(timezone.utc)
    box = load_box_recent(games)
    injuries = load_injuries(box)
    picks, fair, priced = build_picks(rows, games, books, a.bankroll,
                                      a.min_ev, now, injuries)
    made = now.strftime("%a %b %-d, %H:%M UTC")
    for path in (a.out, a.md, a.json, a.history):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = ["away", "home", "commence", "market", "pick", "point", "price",
            "book", "stake", "win_chance", "ev", "fair_market", "why",
            "out", "watch"]
    out = picks[cols].to_dict("records") if len(picks) else []
    out = json.loads(json.dumps(out, default=str))
    history = ledger.load(a.history)
    out = ledger.one_side(history, out)
    if len(picks):
        keep = {ledger._key(p) for p in out}
        picks = picks[[ledger._key(r) in keep for r in
                       json.loads(json.dumps(picks[cols].to_dict("records"),
                                             default=str))]]
    seen = {ledger._key(b) for b in history}
    was = ledger.on_board(history)
    history = ledger.merge(history, out, now, now.isoformat())
    gone = ledger.dropped(was, history, now)
    try:
        history = ledger.grade(history, games)
    except Exception as exc:          # scores can wait for the next run
        print(f"grading skipped: {exc}")
    ledger.save(a.history, history)
    tests, lines = paper.load(a.paper), {}
    try:
        lines = model_lines(games, fair, box, injuries, now)
        tests = paper.log(tests, fair, lines, now.isoformat())
        tests = paper.grade(tests, games)
    except Exception as exc:          # the test never blocks real picks
        print(f"model test skipped: {exc}")
    paper.save(a.paper, tests)
    print(f"model test: {len(lines)} games projected, {len(tests)} test bets "
          f"logged, {paper.summary(tests)}")
    with open(a.md, "w") as fh:
        fh.write(render_md(picks, made, history))
    with open(a.out, "w") as fh:
        fh.write(render_html(out, fair, made, a.bankroll, history, tests,
                             lines))
    with open(a.json, "w") as fh:
        json.dump({"updated": made, "picks": out}, fh, indent=2, default=str)
    fresh = [p for p in out if ledger._key(p) not in seen]
    note = "\n".join(r(b) for b, r in ((fresh, render_alert),
                                       (gone, render_dropped)) if b)
    if note:
        with open(a.alert, "w") as fh:
            fh.write(note)
    print(f"{len(fair)} games priced, {len(out)} bets, {len(fresh)} new, "
          f"{len(gone)} dropped")
    for p in out:
        print(f"  {p['pick']} {p['point']} ({p['price']}) at {p['book']}: "
              f"${p['stake']} | {p['win_chance']:.1%} | {p['why']}")
        if p.get("out") or p.get("watch"):
            print(f"    out: {p.get('out') or '-'} | watch: {p.get('watch') or '-'}")


if __name__ == "__main__":
    main()
