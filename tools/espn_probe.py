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


for day in ([] if "--box" in sys.argv else
            [a for a in sys.argv[1:] if not a.startswith("--")]
            or ["20240115", "20181115", "20261030"]):
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


def box(day="20240115"):
    """Box score shape: per-team stat labels and the first few athletes."""
    js = requests.get(SB, params={"dates": day}, timeout=30).json()
    ev = js["events"][0]
    s = requests.get(SUM, params={"event": ev["id"]}, timeout=30).json()
    for tm in s.get("boxscore", {}).get("players", []):
        print("team", tm.get("team", {}).get("abbreviation"))
        for st in tm.get("statistics", []):
            print(" names", st.get("names"), "\n labels", st.get("labels"),
                  "\n keys", st.get("keys"))
            for a in st.get("athletes", [])[:2] + st.get("athletes", [])[-3:]:
                print("  ", {k: v for k, v in a.items() if k != "athlete"},
                      a.get("athlete", {}).get("id"),
                      a.get("athlete", {}).get("displayName"))
    inj = s.get("injuries") or []
    print("summary injuries: teams", len(inj), "first entry keys",
          sorted((inj[0].get("injuries") or [{}])[0]) if inj else None)


def open_close_coverage(days=("20171115", "20200115", "20220115",
                              "20240115", "20260115")):
    """Which seasons carry open/close in the core odds items."""
    for day in days:
        js = requests.get(SB, params={"dates": day}, timeout=30).json()
        ev = js["events"][0]
        c = requests.get(CORE.format(id=ev["id"]), timeout=30).json()
        for it in c.get("items", []):
            h = it.get("homeTeamOdds", {})
            print(day, it.get("provider", {}).get("name"),
                  "open" in h, "close" in h,
                  h.get("open", {}).get("pointSpread"),
                  h.get("close", {}).get("pointSpread"),
                  it.get("open", {}).get("total"), it.get("close", {}).get("total"))


if __name__ == "__main__" and "--box" in sys.argv:
    box()
