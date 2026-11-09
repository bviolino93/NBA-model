"""
Snapshot NBA lines from The Odds API into CSVs, so every model can be
scored against numbers you could actually bet, not just the close.

Public NBA data has no line history worth trusting and no prop lines at
all. Run on a schedule (.github/workflows/log_lines.yml) and the history
builds itself. Each run appends one row per book x market x outcome with a
`snapshot_utc` stamp, so the earliest snapshot per game is the opener-ish
number and the last pre-tip one is the close.

Credit cost (The Odds API bills markets x regions; approximate):
    game lines   3 markets x 2 regions (us,eu for Pinnacle)  =  6 a run
    props        4 markets x 1 region  x ~8 games a night    = ~32 a run
The schedule takes game lines twice a day and props once, before tip:
about 360 + 1,000 credits a month in season. Props need a paid plan.
Games that have already tipped are skipped: the feed returns in-play
prices for them, which once made up 37% of the prop rows and 17% of the
game-line rows.

--close takes the closing line for clv.py: it runs every few minutes in
the evening, checks ESPN's free scoreboard for a tip in the next
`--close` minutes, and only then spends credits (2 a call: spreads and
totals from CLOSE_BOOKS, which the API bills as one region). Rows go to
data/lines/close_<season>.csv. With --injuries DIR it also logs every
change to ESPN's injury report (free) to DIR/changes_<season>.csv, and
takes an extra snapshot when a player on a team playing later today
changes status (at most NEWS_PULLS_PER_DAY a day), plus one follow-up on
the next run, for measuring how long DraftKings and FanDuel lag the sharp
books after news.

    ODDS_API_KEY=... python log_lines.py [--props] [--out-dir data/lines]
    ODDS_API_KEY=... python log_lines.py --close 30
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timedelta, timezone

import requests

from nba_data import current_season

API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
GAME_MARKETS = "spreads,totals,h2h"
PROP_MARKETS = "player_points,player_rebounds,player_assists,player_threes"
# Sharp books for the close, plus the books bet at (all seen in the us,eu
# feed). Up to 10 bill as one region.
CLOSE_BOOKS = ("pinnacle,betonlineag,lowvig,matchbook,betfair_ex_eu,"
               "draftkings,fanduel,betmgm,williamhill_us")
FIELDS = ["snapshot_utc", "event_id", "commence_time", "home_team",
          "away_team", "book", "market", "outcome", "player", "point",
          "price", "last_update"]


def _get(url, params):
    r = requests.get(url, params=params, timeout=30)
    if r.status_code >= 400:
        # The Odds API says why in the body: a bad key and an exhausted
        # monthly quota are both 401s.
        print(f"  HTTP {r.status_code}: {r.text[:300]}")
    r.raise_for_status()
    left = r.headers.get("x-requests-remaining")
    if left is not None:
        print(f"  credits remaining: {left}")
    return r.json()


def _rows(event, snap):
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                yield {
                    "snapshot_utc": snap,
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "book": bk.get("key"),
                    "market": mk.get("key"),
                    "outcome": oc.get("name"),
                    # Props put the player in `description`.
                    "player": oc.get("description", ""),
                    "point": oc.get("point"),
                    "price": oc.get("price"),
                    "last_update": mk.get("last_update", bk.get("last_update")),
                }


def _append(path, rows):
    rows = list(rows)
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def pregame(events, now):
    """Events that haven't tipped (the feed also returns live ones)."""
    out = []
    for e in events:
        try:
            tip = datetime.fromisoformat(
                str(e.get("commence_time")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if tip > now:
            out.append(e)
    return out


def todays_games(now) -> list:
    """Today's ESPN games (US date). Free; no Odds API credits."""
    from nba_data import _get, parse_scoreboard
    from zoneinfo import ZoneInfo
    day = now.astimezone(ZoneInfo("America/New_York")).date()
    return parse_scoreboard(_get({"dates": f"{day:%Y%m%d}", "limit": 100}))


def tips_soon(now, minutes: float, games=None) -> list:
    """ESPN games tipping in (now, now + minutes]."""
    games = todays_games(now) if games is None else games
    end = now + timedelta(minutes=minutes)
    return [r for r in games
            if r["commence"] is not None and now < r["commence"] <= end]


# Injury news (--injuries): an extra snapshot when a player on a team that
# plays later today changes status, and one more on the next run, so the
# lag between Pinnacle and DraftKings / FanDuel after news can be measured.
NEWS_PULLS_PER_DAY = 30      # 2 credits each, plus the follow-ups
NEWS_STATUSES = ("out", "doubtful", "questionable", "day-to-day",
                 "game time", "removed")


def _state(path: str, day: str) -> dict:
    import json
    try:
        with open(path) as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        st = {}
    if st.get("day") != day:
        st = {"day": day, "news_pulls": 0, "followup_due": None}
    return st


def injury_news(now, out_dir: str, games: list) -> list:
    """Log the report's changes; return the ones that are news for a game
    later today (none on the very first run, which logs the whole report)."""
    import nba_injuries as inj
    path = os.path.join(out_dir, f"changes_{current_season(now.date())}.csv")
    first = not os.path.exists(path)
    ch = inj.log_changes(inj.fetch_injuries(), path,
                         now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    print(f"injury report: {len(ch)} changes logged")
    if first or ch.empty:
        return []
    later = {t for r in games if r["commence"] is not None and r["commence"] > now
             for t in (r["home_team"], r["away_team"])}
    st = ch["status"].str.lower()
    news = ch[ch["team"].isin(later)
              & st.apply(lambda s: any(k in s for k in NEWS_STATUSES))]
    return [f"{r.player} ({r.team}) {r.status}" for r in news.itertuples()]


def log_close(key: str, out_dir: str, minutes: float, now,
              injuries_dir: str = "") -> None:
    """Snapshot sharp + DK/FD lines when a game tips within `minutes`, when
    injury news hits a game later today (capped), or as the follow-up to
    the last news snapshot."""
    import json
    from zoneinfo import ZoneInfo
    games = todays_games(now)
    soon = tips_soon(now, minutes, games)
    why = []
    if soon:
        why.append("tipping soon: " + ", ".join(
            f"{r['away_team']} @ {r['home_team']}" for r in soon))
    st, state_path = None, ""
    if injuries_dir:
        day = str(now.astimezone(ZoneInfo("America/New_York")).date())
        state_path = os.path.join(injuries_dir, "news_state.json")
        st = _state(state_path, day)
        news = injury_news(now, injuries_dir, games)
        due = st.get("followup_due")
        if due and datetime.fromisoformat(due) <= now:
            why.append("follow-up to injury news")
            st["followup_due"] = None
        if news and st["news_pulls"] < NEWS_PULLS_PER_DAY:
            why.append("injury news: " + "; ".join(news[:6]))
            st["news_pulls"] += 1
            st["followup_due"] = (now + timedelta(minutes=5)).isoformat()
    if why:
        print("\n".join(why))
        snap = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        until = (now + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = pregame(_get(f"{API}/odds", {
            "apiKey": key, "bookmakers": CLOSE_BOOKS,
            "markets": "spreads,totals", "oddsFormat": "american",
            "commenceTimeTo": until}), now)
        n = _append(os.path.join(out_dir,
                                 f"close_{current_season(now.date())}.csv"),
                    (r for e in events for r in _rows(e, snap)))
        print(f"  {len(events)} pregame events, {n} rows")
    else:
        print(f"no tip in the next {minutes:g} minutes and no injury news; "
              f"no credits spent")
    if st is not None:
        with open(state_path, "w") as fh:
            json.dump(st, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/lines")
    ap.add_argument("--props", action="store_true",
                    help="also snapshot player props (per-game credits)")
    ap.add_argument("--days-ahead", type=float, default=1.5,
                    help="only events tipping off within this window")
    ap.add_argument("--regions", default="us,eu")
    ap.add_argument("--prop-regions", default="us")
    ap.add_argument("--close", type=float, default=0.0, metavar="MINUTES",
                    help="closing-line mode: log only if a game tips within "
                         "MINUTES (checked on ESPN first)")
    ap.add_argument("--injuries", default="", metavar="DIR",
                    help="with --close: log the injury report's changes to "
                         "DIR and snapshot lines on news")
    a = ap.parse_args()
    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    if not key:
        raise SystemExit("set ODDS_API_KEY")

    now = datetime.now(timezone.utc)
    if a.close:
        log_close(key, a.out_dir, a.close, now, a.injuries)
        return
    snap = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    until = (now + timedelta(days=a.days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
    season = current_season(now.date())

    print("game lines")
    events = pregame(_get(f"{API}/odds", {
        "apiKey": key, "regions": a.regions, "markets": GAME_MARKETS,
        "oddsFormat": "american", "commenceTimeTo": until}), now)
    n = _append(os.path.join(a.out_dir, f"games_{season}.csv"),
                (r for e in events for r in _rows(e, snap)))
    print(f"  {len(events)} events, {n} rows")

    if not a.props:
        return
    print("props")
    total = 0
    for e in events:
        try:
            ev = _get(f"{API}/events/{e['id']}/odds", {
                "apiKey": key, "regions": a.prop_regions,
                "markets": PROP_MARKETS, "oddsFormat": "american"})
        except requests.HTTPError as err:
            print(f"  {e.get('away_team')} @ {e.get('home_team')}: {err}")
            continue
        total += _append(os.path.join(a.out_dir, f"props_{season}.csv"),
                         _rows(ev, snap))
    print(f"  {total} rows")


if __name__ == "__main__":
    main()
