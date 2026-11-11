"""Print ESPN final scores for one date (NBA + men's college). Run on Actions
(Actions -> "Scores"); the sandbox can't reach ESPN."""
import sys

import requests

DAY = sys.argv[1] if len(sys.argv) > 1 else "20261110"
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/{}/scoreboard"
for lg, extra in [("nba", {}),
                  ("mens-college-basketball", {"groups": "50", "limit": "500"})]:
    js = requests.get(BASE.format(lg), params={"dates": DAY, **extra},
                      timeout=30).json()
    print(f"===== {lg} {DAY}: {len(js.get('events', []))} events")
    for ev in js.get("events", []):
        c = ev["competitions"][0]
        t = {x["homeAway"]: x for x in c["competitors"]}
        a, h = t["away"], t["home"]
        print(f"{a['team']['displayName']} {a.get('score')} @ "
              f"{h['team']['displayName']} {h.get('score')} | "
              f"{c['status']['type']['description']} | neutral={c.get('neutralSite')}")
