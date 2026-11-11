"""Tonight's slate for handicapping by hand: line, total, ESPN's win
projection, records, last five, injuries, per game, from ESPN's
scoreboard and game summaries. Run on Actions ("Scores" workflow); the
sandbox can't reach ESPN.

    python tools/slate.py nba 20261111
    python tools/slate.py mens-college-basketball 20261111
"""
import sys

import requests

LG = sys.argv[1] if len(sys.argv) > 1 else "nba"
DAY = sys.argv[2] if len(sys.argv) > 2 else "20261111"
B = f"https://site.api.espn.com/apis/site/v2/sports/basketball/{LG}"
X = {"groups": "50", "limit": "500"} if "college" in LG else {}


def get(url, **p):
    try:
        r = requests.get(url, params=p, timeout=30)
        return r.json() if r.ok else {}
    except Exception:
        return {}


sb = get(f"{B}/scoreboard", dates=DAY, **X)
evs = sb.get("events", [])
print(f"===== {LG} {DAY}: {len(evs)} games")
for ev in evs:
    c = ev["competitions"][0]
    t = {x["homeAway"]: x for x in c["competitors"]}
    a, h = t["away"], t["home"]
    odds = (c.get("odds") or [{}])[0]
    if "college" in LG and not odds.get("details"):
        continue
    rec = lambda x: " ".join(f"{r.get('type', r.get('name', ''))} {r.get('summary')}"
                             for r in x.get("records", []))
    s = get(f"{B}/summary", event=ev["id"])
    print(f"\n### {a['team']['displayName']} @ {h['team']['displayName']}"
          f"{' (neutral)' if c.get('neutralSite') else ''} | {ev.get('date')} | "
          f"{c['status']['type']['shortDetail']}")
    print(f"  line: {odds.get('details')} total {odds.get('overUnder')} "
          f"({(odds.get('provider') or {}).get('name')})")
    for pc in s.get("pickcenter", [])[:1]:
        print(f"  pickcenter: {pc.get('details')} o/u {pc.get('overUnder')} "
              f"ML away {(pc.get('awayTeamOdds') or {}).get('moneyLine')} "
              f"home {(pc.get('homeTeamOdds') or {}).get('moneyLine')}")
    pr = s.get("predictor") or {}
    if pr:
        print(f"  ESPN projection: home win {(pr.get('homeTeam') or {}).get('gameProjection')}% "
              f"away {(pr.get('awayTeam') or {}).get('gameProjection')}%")
    print(f"  away records: {rec(a)} | home records: {rec(h)}")
    for lf in s.get("lastFiveGames", []):
        nm = (lf.get("team") or {}).get("abbreviation")
        g5 = []
        for e in lf.get("events", []):
            g5.append(f"{e.get('gameResult')}{e.get('score')}"
                      f"{'@' if e.get('atVs') == '@' else 'v'}"
                      f"{(e.get('opponent') or {}).get('abbreviation')}")
        print(f"  last5 {nm}: {', '.join(g5)}")
    for inj in s.get("injuries", []):
        nm = (inj.get("team") or {}).get("abbreviation")
        lst = [f"{(i.get('athlete') or {}).get('displayName')}"
               f" ({((i.get('athlete') or {}).get('position') or {}).get('abbreviation')})"
               f" {i.get('status')}" for i in inj.get("injuries", [])]
        if lst:
            print(f"  injuries {nm}: {'; '.join(lst)}")
    for tm in (s.get("boxscore") or {}).get("teams", []):
        st = {x.get("name") or x.get("label"): x.get("displayValue")
              for x in tm.get("statistics", [])}
        keep = ["avgPoints", "avgPointsAgainst", "fieldGoalPct", "threePointFieldGoalPct",
                "avgRebounds", "avgTurnovers", "streak"]
        got = {k: st[k] for k in keep if k in st}
        if got:
            print(f"  season stats {(tm.get('team') or {}).get('abbreviation')}: {got}")
