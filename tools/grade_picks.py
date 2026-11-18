"""
Grade the game-by-game picks (NBA and college) and keep the record.

Picks live on the `breakdowns` branch as picks/<date>.json:
    {"date": "2026-11-18", "picks": [
        {"sport": "nba" | "cbb", "home": "...", "away": "...",
         "market": "spread" | "total", "side": "<team>" | "Over" | "Under",
         "line": 12.5,        # the side's spread (+ = getting points) or the total
         "confidence": 1-5, "stake": 5, "why": "..."}]}
Team names are ESPN display names (as in the breakdown sheets).

This fetches the final scores from ESPN's scoreboard, writes
picks/graded/<date>.json and prints the day and the running record by
sport, market and confidence.

    python tools/grade_picks.py --dir ../bd            # every ungraded day
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import requests

BOARD = {"nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
         "cbb": ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
                 "mens-college-basketball/scoreboard")}
ALIASES = {"LA Clippers": "Los Angeles Clippers", "LA Lakers": "Los Angeles Lakers"}


def _canon(x: str) -> str:
    x = str(x).strip()
    return ALIASES.get(x, x)


def finals(sport: str, day: str) -> dict:
    """{(home, away): (home_score, away_score)} for finished games."""
    params = {"dates": day.replace("-", ""), "limit": 500}
    if sport == "cbb":
        params["groups"] = 50
    js = requests.get(BOARD[sport], params=params, timeout=30).json()
    out = {}
    for ev in js.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        t = {c["homeAway"]: c for c in comp.get("competitors", [])}
        try:
            out[(_canon(t["home"]["team"]["displayName"]),
                 _canon(t["away"]["team"]["displayName"]))] = (
                float(t["home"]["score"]), float(t["away"]["score"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def grade(p: dict, score) -> str:
    """'W', 'L', 'P' or '' (not final)."""
    if score is None:
        return ""
    hs, as_ = score
    if p["market"] == "spread":
        m = (hs - as_) if p["side"] == p["home"] else (as_ - hs)
        v = m + float(p["line"])
    else:
        v = (hs + as_) - float(p["line"])
        if p["side"].lower() == "under":
            v = -v
    return "W" if v > 0 else "L" if v < 0 else "P"


def profit(res: str, stake: float, price: int = -110) -> float:
    if res == "W":
        return stake * (100 / -price if price < 0 else price / 100)
    return -stake if res == "L" else 0.0


def grade_day(path: str, out_dir: str, fetch=finals) -> dict:
    with open(path) as fh:
        day = json.load(fh)
    cache = {}
    for p in day["picks"]:
        key = p["sport"]
        if key not in cache:
            cache[key] = fetch(key, day["date"])
        sc = cache[key].get((p["home"], p["away"]))
        p["final"] = None if sc is None else list(sc)
        p["result"] = grade(p, sc)
        p["profit"] = round(profit(p["result"], float(p.get("stake", 5)),
                                   int(p.get("price", -110))), 2)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, os.path.basename(path)), "w") as fh:
        json.dump(day, fh, indent=1)
    return day


def record(days: list[dict]) -> dict:
    agg = defaultdict(lambda: [0, 0, 0, 0.0])
    for d in days:
        for p in d["picks"]:
            if not p.get("result"):
                continue
            for k in ("all", p["sport"], f"{p['sport']} {p['market']}",
                      f"confidence {p.get('confidence', '?')}"):
                a = agg[k]
                a["WLP".index(p["result"])] += 1
                a[3] += p.get("profit", 0.0)
    return dict(agg)


def _ln(p) -> str:
    x = float(p["line"])
    return f"{x:+g}" if p["market"] == "spread" else f"{x:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--regrade", action="store_true")
    a = ap.parse_args()
    pdir = os.path.join(a.dir, "picks")
    gdir = os.path.join(pdir, "graded")
    graded = []
    for path in sorted(glob.glob(os.path.join(pdir, "*.json"))):
        g = os.path.join(gdir, os.path.basename(path))
        done = None
        if os.path.exists(g) and not a.regrade:
            with open(g) as fh:
                done = json.load(fh)
            if all(p.get("result") for p in done["picks"]):
                graded.append(done)
                continue
        d = grade_day(path, gdir)
        graded.append(d)
        print(f"\n{d['date']}")
        for p in d["picks"]:
            print(f"  {p['result'] or '-'}  {p['sport']} {p['away']} @ {p['home']}: "
                  f"{p['side']} {_ln(p)} ({p['market']}, conf "
                  f"{p.get('confidence')}) final {p['final']} {p['profit']:+.2f}")
    print("\nRecord")
    for k, (w, l, pu, pr) in sorted(record(graded).items()):
        print(f"  {k:16s} {w}-{l}-{pu}  {pr:+.2f}")


if __name__ == "__main__":
    main()
