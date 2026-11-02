"""
NBA games, scores and a market line per game, from ESPN's public JSON.

There is no nflverse for basketball. ESPN's scoreboard endpoint is free,
needs no key, works from GitHub Actions, and for recent seasons carries one
book's pregame spread and total on every game, which is a usable stand-in
for the closing line until a better archive is loaded (see load_lines).

The frame load_games() returns uses the NFL repo's column names, so
selector.build_outcome_models and the backtest code read it unchanged:

    game_id, season, season_type, game_date, commence, home_team,
    away_team, home_score, away_score, spread_line, total_line, neutral

  * season is the year the season ENDS (2026 = the 2025-26 season), which
    is ESPN's convention
  * spread_line is the expected HOME margin (positive = home favoured), as
    in nflverse; ESPN states it from the favourite, so it is converted
  * team names are the full names The Odds API uses, so lines and games
    join on name

    python nba_data.py --first 2016 --last 2026     # fill the cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

CACHE_DIR = os.environ.get("NBA_CACHE", ".cache_nba")
SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
              "scoreboard")
SEASON_TYPES = {1: "PRE", 2: "REG", 3: "POST", 5: "PLAYIN"}

# ESPN display name -> The Odds API name, where they differ.
ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",
}

COLS = ["game_id", "season", "season_type", "game_date", "commence",
        "home_team", "away_team", "home_abbr", "away_abbr", "home_score",
        "away_score", "spread_line", "total_line", "neutral", "completed",
        "line_books"]


def canon(name: str) -> str:
    name = str(name).strip()
    return ALIASES.get(name, name)


def current_season(today: date | None = None) -> int:
    """Season label (end year). A season starts in October."""
    today = today or datetime.now(timezone.utc).date()
    return today.year + 1 if today.month >= 8 else today.year


# ----------------------------------------------------------------------
# Parsing (pure, tested on fixtures)
# ----------------------------------------------------------------------
_DETAILS = re.compile(r"^\s*([A-Z]{2,4})\s+([-+]?\d+(?:\.\d+)?)\s*$")


def _home_spread(odds: dict, home_abbr: str, away_abbr: str) -> float:
    """Expected home margin from ESPN's odds block, or NaN.

    `details` reads like "BOS -8.5" (the favourite and its number) or
    "EVEN". It is the most reliable field across seasons; the numeric
    `spread` field has not always been stated from the same side.
    """
    det = str(odds.get("details") or "").strip()
    if det.upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = _DETAILS.match(det)
    if m:
        abbr, pts = m.group(1), float(m.group(2))
        if abbr == home_abbr:
            return -pts
        if abbr == away_abbr:
            return pts
    # Newer feeds: homeTeamOdds.spreadOdds / pointSpread.home.close.line
    try:
        line = odds["pointSpread"]["home"]["close"]["line"]
        return -float(str(line).replace("+", ""))
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _total(odds: dict) -> float:
    for k in ("overUnder",):
        try:
            v = float(odds.get(k))
            if v > 100:
                return v
        except (TypeError, ValueError):
            pass
    return float("nan")


def parse_scoreboard(js: dict) -> list[dict]:
    """One row per game in a scoreboard response."""
    rows = []
    for ev in js.get("events", []):
        try:
            comp = ev["competitions"][0]
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            h, a = teams["home"], teams["away"]
        except (KeyError, IndexError):
            continue
        status = comp.get("status", ev.get("status", {})).get("type", {})
        done = bool(status.get("completed"))
        odds = (comp.get("odds") or [{}])[0]
        ha, aa = h["team"].get("abbreviation", ""), a["team"].get("abbreviation", "")
        season = ev.get("season", {})
        commence = pd.to_datetime(ev.get("date"), utc=True)

        def score(c):
            try:
                return float(c.get("score")) if done else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        rows.append({
            "game_id": str(ev.get("id")),
            "season": int(season.get("year", 0)) or None,
            "season_type": SEASON_TYPES.get(int(season.get("type", 2)), "REG"),
            # US calendar date of tip-off (ET), the date bettors use.
            "game_date": (commence - pd.Timedelta(hours=5)).date()
            if pd.notna(commence) else None,
            "commence": commence,
            "home_team": canon(h["team"].get("displayName", "")),
            "away_team": canon(a["team"].get("displayName", "")),
            "home_abbr": ha, "away_abbr": aa,
            "home_score": score(h), "away_score": score(a),
            "spread_line": _home_spread(odds, ha, aa),
            "total_line": _total(odds),
            "neutral": bool(comp.get("neutralSite", False)),
            "completed": done,
        })
    return rows


# ----------------------------------------------------------------------
# Fetching with a per-day cache
# ----------------------------------------------------------------------
def _get(params: dict, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            r = requests.get(SCOREBOARD, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"ESPN scoreboard failed for {params}")


def fetch_day(d: date, cache_dir: str = CACHE_DIR) -> list[dict]:
    """A day's games. Days fully in the past are cached for good."""
    os.makedirs(os.path.join(cache_dir, "days"), exist_ok=True)
    path = os.path.join(cache_dir, "days", f"{d:%Y%m%d}.json")
    today = datetime.now(timezone.utc).date()
    if os.path.exists(path) and d < today - timedelta(days=1):
        with open(path) as fh:
            return parse_scoreboard(json.load(fh))
    js = _get({"dates": f"{d:%Y%m%d}", "limit": 100})
    with open(path, "w") as fh:
        json.dump(js, fh)
    return parse_scoreboard(js)


def season_days(season: int):
    """Every date that could hold a game: Oct 1 to Jun 30."""
    d, end = date(season - 1, 10, 1), date(season, 6, 30)
    today = datetime.now(timezone.utc).date()
    while d <= min(end, today + timedelta(days=7)):
        yield d
        d += timedelta(days=1)


def load_season(season: int, cache_dir: str = CACHE_DIR,
                pause: float = 0.15) -> pd.DataFrame:
    rows = []
    for d in season_days(season):
        cached = os.path.exists(os.path.join(cache_dir, "days",
                                             f"{d:%Y%m%d}.json"))
        rows.extend(fetch_day(d, cache_dir))
        if not cached:
            time.sleep(pause)
    return pd.DataFrame(rows, columns=COLS)


# ----------------------------------------------------------------------
# Lines for past games: ESPN's core odds API
# ----------------------------------------------------------------------
# The scoreboard drops its odds block once a game is final. The core API
# keeps every book ESPN carried for the game (Caesars, DraftKings, ...;
# newer seasons also have open and close). Probed 2026-11-02 on games
# from 2018-19 and 2026-27 (tools/espn_probe.py).
CORE_ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
             "nba/events/{id}/competitions/{id}/odds")
# Providers that are projections or in-game prices, not pregame lines.
SKIP_PROVIDER = re.compile(r"live|numberfire|teamrankings|projection|"
                           r"accuscore|predict", re.I)


def parse_core_odds(js: dict, home_abbr: str, away_abbr: str
                    ) -> tuple[float, float, int]:
    """(home margin line, total, books used): medians across books."""
    spreads, totals = [], []
    for it in js.get("items", []):
        name = str((it.get("provider") or {}).get("name", ""))
        if SKIP_PROVIDER.search(name):
            continue
        sp = _home_spread(it, home_abbr, away_abbr)
        if pd.notna(sp) and abs(sp) <= 30:
            spreads.append(sp)
        tt = _total(it)
        if pd.notna(tt):
            totals.append(tt)
    med = lambda v: float(pd.Series(v).median()) if v else float("nan")
    return med(spreads), med(totals), max(len(spreads), len(totals))


def fetch_core_odds(game_id: str, cache_dir: str = CACHE_DIR) -> dict:
    os.makedirs(os.path.join(cache_dir, "odds"), exist_ok=True)
    path = os.path.join(cache_dir, "odds", f"{game_id}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    js = {}
    for i in range(3):
        try:
            r = requests.get(CORE_ODDS.format(id=game_id), timeout=30)
            if r.status_code == 200:
                js = r.json()
                break
            if r.status_code == 404:
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    with open(path, "w") as fh:
        json.dump(js, fh)
    return js


def fill_lines(g: pd.DataFrame, cache_dir: str = CACHE_DIR,
               workers: int = 8) -> pd.DataFrame:
    """Fill missing spread/total on completed games from the core API."""
    from concurrent.futures import ThreadPoolExecutor
    need = g[g["completed"] & (g["spread_line"].isna()
                               | g["total_line"].isna())]
    if need.empty:
        return g
    with ThreadPoolExecutor(workers) as ex:
        docs = list(ex.map(lambda i: fetch_core_odds(i, cache_dir),
                           need["game_id"]))
    g = g.copy()
    g["line_books"] = g.get("line_books", pd.Series(0, index=g.index))
    for idx, js, ha, aa in zip(need.index, docs, need["home_abbr"],
                               need["away_abbr"]):
        sp, tt, n = parse_core_odds(js, ha, aa)
        if pd.isna(g.at[idx, "spread_line"]):
            g.at[idx, "spread_line"] = sp
        if pd.isna(g.at[idx, "total_line"]):
            g.at[idx, "total_line"] = tt
        g.at[idx, "line_books"] = n
    return g


def load_games(first: int = 2016, last: int | None = None,
               cache_dir: str = CACHE_DIR, regular_only: bool = False,
               lines_csv: str | None = None,
               past_lines: bool = True) -> pd.DataFrame:
    """All games first..last (season end years), oldest first.

    past_lines fetches each completed game's lines from ESPN's core odds
    API (one request per game, cached for good; ~45 minutes cold for ten
    seasons)."""
    last = last or current_season()
    g = pd.concat([load_season(s, cache_dir) for s in range(first, last + 1)],
                  ignore_index=True)
    g = g[g["season_type"] != "PRE"]
    if regular_only:
        g = g[g["season_type"] == "REG"]
    g = g.drop_duplicates("game_id").sort_values("commence")
    if past_lines:
        g = fill_lines(g, cache_dir)
    if lines_csv:
        g = merge_lines(g, load_lines(lines_csv))
    return g.reset_index(drop=True)


# ----------------------------------------------------------------------
# A better closing-line archive, when one is loaded
# ----------------------------------------------------------------------
def load_lines(path: str) -> pd.DataFrame:
    """
    Closing lines from any archive, normalised to
        game_date, home_team, away_team, spread_line, total_line
    with spread_line as the expected home margin. Expected columns are the
    same names; convert an archive to them once and keep it in data/.
    """
    d = pd.read_csv(path)
    d["game_date"] = pd.to_datetime(d["game_date"]).dt.date
    for c in ("home_team", "away_team"):
        d[c] = d[c].map(canon)
    return d[["game_date", "home_team", "away_team", "spread_line",
              "total_line"]]


def merge_lines(g: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Archive lines replace ESPN's where both exist."""
    m = g.merge(lines, on=["game_date", "home_team", "away_team"], how="left",
                suffixes=("", "_arch"))
    for c in ("spread_line", "total_line"):
        m[c] = m[f"{c}_arch"].fillna(m[c])
    return m.drop(columns=["spread_line_arch", "total_line_arch"])


# ----------------------------------------------------------------------
# Schedule features (rest, back-to-backs, density)
# ----------------------------------------------------------------------
def team_games(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with rest computed from its schedule."""
    cols = ["game_id", "season", "game_date", "commence"]
    home = g[cols].assign(team=g["home_team"], opp=g["away_team"], home=1)
    away = g[cols].assign(team=g["away_team"], opp=g["home_team"], home=0)
    t = pd.concat([home, away]).sort_values(["team", "commence"])
    gd = pd.to_datetime(t["game_date"])
    prev = gd.groupby(t["team"]).shift(1)
    same = t["season"] == t.groupby("team")["season"].shift(1)
    t["rest"] = (gd - prev).dt.days.where(same).clip(upper=7)
    t["b2b"] = (t["rest"] == 1).astype(int)
    # Games in the previous 5 nights (excluding this one).
    t["_d"] = gd
    t["last5"] = (t.groupby("team", group_keys=False)
                    .apply(lambda x: x.rolling("5D", on="_d", closed="left")
                                      ["home"].count()))
    t["prev_away"] = t.groupby("team")["home"].shift(1).eq(0).astype(int)
    return t.drop(columns="_d")


def schedule_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per game: rest difference, back-to-back flags, road b2b, density."""
    t = team_games(g).set_index(["game_id", "team"])
    out = g.copy()

    def pick(side, col):
        idx = list(zip(out["game_id"], out[f"{side}_team"]))
        return t[col].reindex(idx).to_numpy()

    for side in ("home", "away"):
        out[f"{side}_rest"] = pick(side, "rest")
        out[f"{side}_b2b"] = pick(side, "b2b")
        out[f"{side}_last5"] = pick(side, "last5")
    # A back-to-back whose first night was on the road (travel overnight).
    out["away_b2b_road"] = out["away_b2b"] * pick("away", "prev_away")
    out["rest_diff"] = (out["home_rest"].fillna(3).clip(upper=4)
                        - out["away_rest"].fillna(3).clip(upper=4))
    out["b2b_diff"] = out["home_b2b"] - out["away_b2b"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2016)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    g = load_games(a.first, a.last)
    done = g[g["completed"]]
    print(f"{len(g)} games, {len(done)} completed")
    print(done.groupby("season").agg(
        games=("game_id", "size"),
        with_spread=("spread_line", lambda s: s.notna().mean().round(3)),
        with_total=("total_line", lambda s: s.notna().mean().round(3)),
        home_margin=("home_score", lambda s: (
            s - done.loc[s.index, "away_score"]).mean().round(2)),
    ).to_string())
    if a.out:
        g.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
