"""
Print where ESPN keeps pregame lines for past games, so nba_data can read
them. Run on Actions (Actions -> "ESPN probe"); the sandbox can't reach ESPN.
"""
import json
import sys

import requests

SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
CORE = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
        "events/{id}/competitions/{id}/odds")


def show(label, obj, n=1500):
    print(f"--- {label}")
    print(json.dumps(obj, indent=1)[:n])


for day in sys.argv[1:] or ["20240115", "20181115", "20261030"]:
    js = requests.get(SB, params={"dates": day}, timeout=30).json()
    ev = js["events"][0]
    comp = ev["competitions"][0]
    print(f"===== {day} event {ev['id']}  competition keys: {sorted(comp)}")
    show("scoreboard odds", comp.get("odds"))
    s = requests.get(SUM, params={"event": ev["id"]}, timeout=30).json()
    print("summary keys:", sorted(s))
    show("summary pickcenter", s.get("pickcenter"), 2500)
    show("summary odds", s.get("odds"), 800)
    c = requests.get(CORE.format(id=ev["id"]), timeout=30)
    print("core odds status", c.status_code)
    if c.ok:
        show("core odds", c.json(), 2500)
