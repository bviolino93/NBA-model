"""
Player box scores from ESPN's game summaries, and who was missing.

One row per player per game:

    game_id, season, game_date, team, opp, home, player_id, player,
    starter, played, reason, min, pts, reb, ast, fg3m, pm

`played` is False for players on the roster who did not play (DNP,
inactive, injured); ESPN gives a reason for most of them.

Absences come from comparing each game's box with the player's recent
role: a player who averaged real minutes over his team's last games and
then sat is a missing player for that game, whether or not ESPN listed a
reason. What the absence is worth is measured walk-forward (backtest),
never assumed.

    python nba_players.py --first 2016     # fill the per-game cache
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from nba_data import CACHE_DIR, canon, load_games

SUMMARY = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
           "summary")
STATS = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast",
         "+/-": "pm", "3PT": "fg3", "TO": "tov", "STL": "stl", "BLK": "blk"}
COLS = ["game_id", "team_abbr", "player_id", "player", "starter", "played",
        "reason", "min", "pts", "reb", "ast", "fg3m", "pm"]


def _num(x) -> float:
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return float("nan")


def _minutes(x) -> float:
    s = str(x)
    if ":" in s:                      # "34:12"
        m, sec = s.split(":", 1)
        return _num(m) + _num(sec) / 60.0
    return _num(s)


def parse_box(js: dict, game_id: str) -> list[dict]:
    """Player rows from a summary response."""
    rows = []
    for tm in (js.get("boxscore") or {}).get("players", []) or []:
        abbr = (tm.get("team") or {}).get("abbreviation", "")
        for st in tm.get("statistics", []) or []:
            labels = st.get("labels") or st.get("names") or []
            idx = {lab: i for i, lab in enumerate(labels)}
            for a in st.get("athletes", []) or []:
                ath = a.get("athlete") or {}
                stats = a.get("stats") or []
                dnp = bool(a.get("didNotPlay")) or not stats

                def get(lab):
                    i = idx.get(lab)
                    return stats[i] if i is not None and i < len(stats) else None

                fg3 = str(get("3PT") or "")
                rows.append({
                    "game_id": str(game_id), "team_abbr": abbr,
                    "player_id": str(ath.get("id", "")),
                    "player": ath.get("displayName", ""),
                    "starter": bool(a.get("starter")),
                    "played": not dnp and (_minutes(get("MIN")) or 0) > 0,
                    "reason": a.get("reason", "") or "",
                    "min": 0.0 if dnp else _minutes(get("MIN")),
                    "pts": 0.0 if dnp else _num(get("PTS")),
                    "reb": 0.0 if dnp else _num(get("REB")),
                    "ast": 0.0 if dnp else _num(get("AST")),
                    "fg3m": 0.0 if dnp else _num(fg3.split("-")[0]) if fg3 else float("nan"),
                    "pm": 0.0 if dnp else _num(get("+/-")),
                })
    return rows


def fetch_summary(game_id: str, cache_dir: str = CACHE_DIR) -> dict:
    """A final game's summary; only the box score is kept on disk. A 404 is
    cached as empty; other failures are not cached (next run retries)."""
    from nba_data import NONE, cached_json
    os.makedirs(os.path.join(cache_dir, "box"), exist_ok=True)
    path = os.path.join(cache_dir, "box", f"{game_id}.json")
    js = cached_json(path)
    if js is not None:
        return js
    for i in range(3):
        try:
            r = requests.get(SUMMARY, params={"event": game_id}, timeout=30)
            if r.status_code == 200:
                js = {"boxscore": {"players": (r.json().get("boxscore") or {})
                                   .get("players", [])}}
                break
            if r.status_code == 404:
                js = NONE
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    if js is None:
        return {}
    with open(path, "w") as fh:
        json.dump(js, fh)
    return js


def load_box(games: pd.DataFrame, cache_dir: str = CACHE_DIR,
             workers: int = 8) -> pd.DataFrame:
    """Player rows for every completed game in `games`."""
    done = games[games["completed"]]
    ids = done["game_id"].astype(str).tolist()
    with ThreadPoolExecutor(workers) as ex:
        docs = list(ex.map(lambda i: fetch_summary(i, cache_dir), ids))
    rows = [r for gid, js in zip(ids, docs) for r in parse_box(js, gid)]
    p = pd.DataFrame(rows, columns=COLS)
    return attach_games(p, done)


def attach_games(p: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Add season, date, team name, opponent and home flag to box rows."""
    g = games[["game_id", "season", "game_date", "commence", "home_team",
               "away_team", "home_abbr", "away_abbr"]].copy()
    g["game_id"] = g["game_id"].astype(str)
    p = p.merge(g, on="game_id", how="inner")
    home = p["team_abbr"] == p["home_abbr"]
    p["home"] = home.astype(int)
    p["team"] = np.where(home, p["home_team"], p["away_team"])
    p["opp"] = np.where(home, p["away_team"], p["home_team"])
    return p.drop(columns=["home_team", "away_team", "home_abbr",
                           "away_abbr"]).sort_values(["commence", "game_id"])


# ----------------------------------------------------------------------
# Roles and absences, walk-forward
# ----------------------------------------------------------------------
def team_minutes(p: pd.DataFrame) -> pd.DataFrame:
    """Team-game x player minutes, 0 where the player had no minutes.

    Players ruled out before the game often have no box row at all, so a
    missing player can only be found by looking at the team's earlier
    games, never at this game's box.
    """
    p = p.copy()
    p["key"] = p["season"].astype(str) + "|" + p["team"]
    order = (p.drop_duplicates(["key", "game_id"])
               .sort_values("commence")[["key", "game_id", "commence"]])
    order["n"] = order.groupby("key").cumcount()
    p = p.merge(order[["key", "game_id", "n"]], on=["key", "game_id"])
    return p


PROD_PRIOR = 18.0     # per 36: about a replacement-level rotation player
PROD_REPL = 18.0      # production per 36 a fill-in gives back


def roles(p: pd.DataFrame, window: int = 10, recent: int = 5,
          prior_min: float = 300.0) -> pd.DataFrame:
    """Every player's expected role going INTO each of his team's games.

    One row per team game per player who appeared for the team earlier in
    the season, built from earlier games only:
      exp_min   mean minutes over the team's last `window` games (0 when
                he did not play)
      recent    played in at least one of the team's last `recent` games
                (drops players who were traded or are out long term)
      pm36      season-to-date plus-minus per 36, shrunk with `prior_min`
      prod36    season-to-date box production per 36 (points + 0.5
                rebounds + 0.7 assists), shrunk toward PROD_PRIOR
      min_now   minutes in THIS game (0 = did not play)
    """
    p = team_minutes(p)
    out = []
    for key, t in p.groupby("key", sort=False):
        mat = t.pivot_table(index="n", columns="player_id", values="min",
                            aggfunc="sum").sort_index()
        mat = mat.reindex(range(int(mat.index.max()) + 1))
        seen = mat.notna().cumsum().shift(1).fillna(0) > 0
        m = mat.fillna(0.0)
        pmm = (t.pivot_table(index="n", columns="player_id", values="pm",
                             aggfunc="sum").reindex(mat.index).fillna(0.0))
        exp_min = m.rolling(window, min_periods=1).mean().shift(1)
        rec = (m > 0).rolling(recent, min_periods=1).max().shift(1).fillna(0) > 0
        pm36 = 36.0 * pmm.cumsum().shift(1) / (m.cumsum().shift(1) + prior_min)
        prd = (t.assign(prod=t["pts"] + 0.5 * t["reb"] + 0.7 * t["ast"])
                .pivot_table(index="n", columns="player_id", values="prod",
                             aggfunc="sum").reindex(mat.index).fillna(0.0))
        prod36 = 36.0 * (prd.cumsum().shift(1) + PROD_PRIOR / 36.0 * prior_min) \
            / (m.cumsum().shift(1) + prior_min)
        stacked = pd.DataFrame({
            "exp_min": exp_min.stack(future_stack=True),
            "recent": rec.stack(future_stack=True),
            "pm36": pm36.stack(future_stack=True),
            "prod36": prod36.stack(future_stack=True),
            "min_now": m.stack(future_stack=True),
            "seen": seen.stack(future_stack=True),
        }).reset_index()
        stacked = stacked[stacked["seen"]]
        stacked["key"] = key
        out.append(stacked)
    r = pd.concat(out, ignore_index=True)
    games = p.drop_duplicates(["key", "n"])[["key", "n", "game_id", "season",
                                             "team", "commence"]]
    names = p.drop_duplicates("player_id").set_index("player_id")["player"]
    r = r.merge(games, on=["key", "n"])
    r["player"] = r["player_id"].map(names)
    return r.drop(columns=["seen"])


def absences(p: pd.DataFrame, min_minutes: float = 15.0) -> pd.DataFrame:
    """Per team per game: missing rotation minutes and their value.

    A player is missing when he averaged `min_minutes`+ over the team's
    last 10 games, played in one of the last 5, and has no minutes in
    this one. value = expected minutes x max(plus-minus per minute, 0)
    plus a flat 1 point per 48 minutes for the drop to a replacement;
    losing a bad player is not treated as a gain.
    """
    r = roles(p)
    miss = r[(r["exp_min"] >= min_minutes) & r["recent"] & (r["min_now"] <= 0)]
    miss = miss.assign(
        lost_min=miss["exp_min"],
        lost_val=miss["exp_min"] / 36.0 * miss["pm36"].fillna(0).clip(lower=0)
                 + miss["exp_min"] / 48.0,
        lost_star=(miss["exp_min"] >= 30).astype(int),
        # Production above a fill-in's, over the minutes he'd have played
        lost_prod=miss["exp_min"] / 36.0
                  * (miss["prod36"].fillna(PROD_REPL) - PROD_REPL).clip(lower=0),
    )
    out = (miss.groupby(["game_id", "team"])
               .agg(lost_min=("lost_min", "sum"), lost_val=("lost_val", "sum"),
                    lost_star=("lost_star", "sum"),
                    lost_prod=("lost_prod", "sum"),
                    missing=("player", lambda s: ", ".join(map(str, s))))
               .reset_index())
    return out


def game_absences(games: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away absence features on each game row."""
    g = games.copy()
    g["game_id"] = g["game_id"].astype(str)
    ab = ab.copy()
    for c in ("lost_min", "lost_val", "lost_star", "lost_prod"):
        if c not in ab:
            ab[c] = 0.0
    for side in ("home", "away"):
        a = ab.rename(columns={c: f"{side}_{c}" for c in
                               ("lost_min", "lost_val", "lost_star",
                                "lost_prod", "missing")})
        g = g.merge(a.rename(columns={"team": f"{side}_team"}),
                    on=["game_id", f"{side}_team"], how="left")
    for c in ("lost_min", "lost_val", "lost_star", "lost_prod"):
        g[f"home_{c}"] = g[f"home_{c}"].fillna(0)
        g[f"away_{c}"] = g[f"away_{c}"].fillna(0)
    # Positive = the HOME side is more depleted.
    g["inj_val_diff"] = g["home_lost_val"] - g["away_lost_val"]
    g["inj_min_diff"] = g["home_lost_min"] - g["away_lost_min"]
    g["inj_val_sum"] = g["home_lost_val"] + g["away_lost_val"]
    g["inj_star_diff"] = g["home_lost_star"] - g["away_lost_star"]
    g["inj_prod_diff"] = g["home_lost_prod"] - g["away_lost_prod"]
    g["inj_prod_sum"] = g["home_lost_prod"] + g["away_lost_prod"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2016)
    ap.add_argument("--last", type=int, default=None)
    a = ap.parse_args()
    g = load_games(a.first, a.last)
    p = load_box(g)
    print(f"{len(p)} player rows, {p['game_id'].nunique()} games")
    print(p.groupby("season").agg(
        rows=("player_id", "size"), played=("played", "mean"),
        min_per_team_game=("min", lambda s: s.sum() / max(1, 2 * p.loc[
            s.index, "game_id"].nunique()))).round(3).to_string())
    ab = absences(p)
    print(ab.describe().round(2).to_string())


if __name__ == "__main__":
    main()
