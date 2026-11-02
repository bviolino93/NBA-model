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

    ODDS_API_KEY=... python log_lines.py [--props] [--out-dir data/lines]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/lines")
    ap.add_argument("--props", action="store_true",
                    help="also snapshot player props (per-game credits)")
    ap.add_argument("--days-ahead", type=float, default=1.5,
                    help="only events tipping off within this window")
    ap.add_argument("--regions", default="us,eu")
    ap.add_argument("--prop-regions", default="us")
    a = ap.parse_args()
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("set ODDS_API_KEY")

    now = datetime.now(timezone.utc)
    snap = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    until = (now + timedelta(days=a.days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")
    season = current_season(now.date())

    print("game lines")
    events = _get(f"{API}/odds", {
        "apiKey": key, "regions": a.regions, "markets": GAME_MARKETS,
        "oddsFormat": "american", "commenceTimeTo": until})
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
