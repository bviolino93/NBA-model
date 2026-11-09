"""
Tonight's injury report from ESPN, weighed by each player's role.

    https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries

The backtest found the closing spread already prices who sits, so the
report doesn't move the fair line here: the sharp market moves within
minutes of news and the fair line follows it. What the report adds to a
pick is risk. A rotation player listed questionable or day-to-day means
the line can still jump before tip, either way, so the pick is flagged
"wait for news" and the alert says who to watch. Players already ruled
out are listed for context only; their absence is in the price.

Role = average minutes over the player's last 10 games this season, from
ESPN box scores (nba_players), matched on ESPN athlete id, else name.
"""

from __future__ import annotations

import pandas as pd
import requests

from nba_data import canon

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
UNCERTAIN = ("questionable", "doubtful", "day-to-day", "game time decision",
             "gtd", "probable")
KEY_MIN = 20.0          # minutes a game that make a player worth flagging
COLS = ["team", "player_id", "player", "status", "detail", "date"]
# Logged changes (log_changes): one row per player whose status changed.
LOG_COLS = ["snapshot_utc", "team", "player_id", "player", "status", "detail",
            "date"]
REMOVED = "Removed"       # dropped off the report: cleared to play


def _status(it: dict) -> str:
    s = it.get("status")
    if isinstance(s, dict):
        s = s.get("name") or s.get("description") or s.get("type")
    if not s:
        t = it.get("type") or {}
        s = t.get("description") or t.get("name") or ""
    s = str(s).replace("INJURY_STATUS_", "").replace("_", " ").strip()
    return s.title() if s.isupper() or s.islower() else s


def parse_injuries(js: dict) -> pd.DataFrame:
    """One row per listed player: team (Odds API name), id, name, status."""
    rows = []
    for tm in js.get("injuries", []) or []:
        team = canon(tm.get("displayName")
                     or (tm.get("team") or {}).get("displayName") or "")
        for it in tm.get("injuries", []) or []:
            ath = it.get("athlete") or {}
            if not team:
                team = canon((ath.get("team") or {}).get("displayName", ""))
            det = it.get("details") or {}
            detail = " ".join(str(x) for x in (det.get("type"), det.get("side"),
                                                det.get("detail")) if x)
            rows.append({"team": team, "player_id": str(ath.get("id", "")),
                         "player": ath.get("displayName", ""),
                         "status": _status(it),
                         "detail": detail or it.get("shortComment", "") or "",
                         # ESPN's time for the entry: when the news broke,
                         # closer than our polling interval
                         "date": str(it.get("date") or "")})
    return pd.DataFrame(rows, columns=COLS)


def log_changes(report: pd.DataFrame, path: str, snap: str) -> pd.DataFrame:
    """Append to `path` every player whose status is new or changed since
    the last logged state, and every player who dropped off the report
    (status REMOVED). Returns the changes. The first run logs the whole
    report; the log is then a complete history of the report at the
    polling interval.

    Returns an empty frame, and logs nothing, when the report came back
    empty (a failed fetch must not mark the whole league as cleared)."""
    import os
    if report is None or report.empty:
        return pd.DataFrame(columns=LOG_COLS)
    rep = report.assign(key=report["team"] + "|" + report["player_id"])
    rep = rep.drop_duplicates("key", keep="last")
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str).fillna("")
        old["key"] = old["team"] + "|" + old["player_id"]
        last = old.drop_duplicates("key", keep="last").set_index("key")
        last = last[last["status"] != REMOVED]
    else:
        last = pd.DataFrame(columns=LOG_COLS).set_index(pd.Index([], name="key"))
    prev = rep["key"].map(last["status"]) if len(last) else pd.Series(
        None, index=rep.index, dtype=object)
    changed = rep[prev.isna() | (prev != rep["status"])]
    gone = last[~last.index.isin(rep["key"])]
    gone = gone.reset_index().assign(status=REMOVED, detail="", date="")
    out = pd.concat([changed, gone], ignore_index=True).assign(snapshot_utc=snap)
    out = out[LOG_COLS]
    if len(out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        out.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return out


def fetch_injuries() -> pd.DataFrame:
    try:
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        df = parse_injuries(r.json())
        print(f"injury report: {len(df)} players on {df['team'].nunique()} teams")
        return df
    except Exception as exc:              # picks still run without it
        print(f"injury report unavailable: {exc}")
        return pd.DataFrame(columns=COLS)


def recent_minutes(box: pd.DataFrame, last: int = 10) -> pd.DataFrame:
    """player_id, player, team, mpg over the team's last `last` games."""
    if box is None or box.empty:
        return pd.DataFrame(columns=["player_id", "player", "team", "mpg"])
    b = box.sort_values("commence")
    tg = (b.drop_duplicates(["team", "game_id"])
            .groupby("team").tail(last)[["team", "game_id"]])
    b = b.merge(tg, on=["team", "game_id"])
    n = tg.groupby("team").size().rename("n_games")
    m = (b.groupby(["team", "player_id", "player"])["min"].sum()
           .reset_index().merge(n, on="team"))
    m["mpg"] = m["min"] / m["n_games"]
    return m[["player_id", "player", "team", "mpg"]]


def with_roles(inj: pd.DataFrame, mins: pd.DataFrame) -> pd.DataFrame:
    """Add mpg (0 when unknown) and an `uncertain` flag."""
    inj = inj.copy()
    if inj.empty:
        return inj.assign(mpg=[], uncertain=[])
    by_id = mins.drop_duplicates("player_id").set_index("player_id")["mpg"]
    by_name = (mins.sort_values("mpg").drop_duplicates("player", keep="last")
                   .set_index("player")["mpg"])
    mpg = inj["player_id"].map(by_id)
    inj["mpg"] = mpg.fillna(inj["player"].map(by_name)).fillna(0.0)
    st = inj["status"].str.lower()
    inj["uncertain"] = st.apply(lambda s: any(u in s for u in UNCERTAIN)
                                and "probable" not in s)
    return inj


def game_notes(inj: pd.DataFrame, home: str, away: str,
               key_min: float = KEY_MIN) -> dict:
    """Who matters in this game: {'out': [...], 'watch': [...]} strings."""
    out, watch = [], []
    if inj is None or inj.empty:
        return {"out": out, "watch": watch}
    g = inj[inj["team"].isin([home, away]) & (inj["mpg"] >= key_min)]
    for r in g.sort_values("mpg", ascending=False).itertuples():
        label = f"{r.player} ({_short(r.team)}, {r.mpg:.0f} min)"
        if r.uncertain:
            watch.append(f"{label} {r.status.lower()}")
        elif "out" in r.status.lower():
            out.append(label)
    return {"out": out, "watch": watch}


def _short(team: str) -> str:
    return team.split()[-1]
