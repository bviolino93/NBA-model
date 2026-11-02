"""
No-money test of the ratings model's own lines against the market.

nba_ratings builds a home margin and a total for every game from scores
alone (opponent-adjusted, home court, recency-weighted). Backtested
against closing lines it lost: 49% ATS when it disagreed by 3+ points.
This test checks it live instead of trusting the backtest either way.
The first time a game is seen, if the model's margin or total is GAP+
points off the sharp fair line, a test bet goes in on the model's side at
that line. Each is graded two ways:

  * against the final score, at the logged line
  * against the closing line (CLV, in points moved the model's way): the
    honest early read, since a few hundred results are mostly luck

Nothing here is staked. If the logged bets beat the close over a few
hundred games, the model can move into the real picks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nba_ratings import RatingConfig, fit

GAP = 3.0
# Bump when the model changes in a way that makes earlier test bets
# unrepresentative; bets logged by older versions are dropped on load.
VERSION = 3


def load(path: str) -> list[dict]:
    try:
        with open(path) as fh:
            return [b for b in json.load(fh).get("bets", [])
                    if b.get("v") == VERSION]
    except (OSError, ValueError):
        return []


def _key(market: str, home: str, away: str, commence: str) -> str:
    return f"{market}|{away}|{home}|{str(commence)[:10]}"


def model_lines(games: pd.DataFrame, fair: dict, now=None,
                cfg: RatingConfig | None = None) -> dict[str, dict]:
    """{fair key: {"margin", "total"}} from ratings fit on games before
    today (this season and the two before)."""
    cfg = cfg or RatingConfig()
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    done = g[g["completed"] & (g["game_date"] < today)]
    if done.empty:
        return {}
    season = int(done["season"].max())
    prior = done[done["season"] >= season - 2]
    if len(prior) < cfg.min_games:
        return {}
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    hca, r, mu, t = fit(prior, today, season, cfg, teams)
    out = {}
    for k, f in fair.items():
        h, a = f.home_team, f.away_team
        if h not in r or a not in r:
            continue
        out[k] = {"margin": float(hca + r[h] - r[a]),
                  "total": float(mu + t.get(h, 0.0) + t.get(a, 0.0))}
    return out


def log(history: list[dict], fair: dict, lines: dict, run: str,
        gap: float = GAP) -> list[dict]:
    """Add a test bet for each market seen for the first time where the
    model and the fair line disagree by `gap` points or more."""
    seen = {b["key"] for b in history}
    for k, f in fair.items():
        if k not in lines:
            continue
        mine = lines[k]
        for market, fair_pt, model_pt in (
                ("spreads", f.market_margin, mine["margin"]),
                ("totals", f.market_total, mine["total"])):
            key = _key(market, f.home_team, f.away_team, f.commence)
            if key in seen or fair_pt is None or np.isnan(fair_pt):
                continue
            seen.add(key)
            line = round(fair_pt * 2) / 2              # bookable half point
            d = model_pt - line
            if abs(d) < gap:
                continue
            if market == "spreads":
                home = d > 0
                side = f.home_team if home else f.away_team
                point = -line if home else line        # side's own line
            else:
                side, point = ("Over" if d > 0 else "Under"), line
            history.append({
                "v": VERSION, "key": key, "market": market, "away": f.away_team,
                "home": f.home_team, "commence": str(f.commence),
                "logged": run, "side": side, "point": point,
                "line": line, "model": round(model_pt, 1),
                "parts": mine.get("parts") if market == "spreads" else None,
                "gap": round(d, 1), "close": None, "clv": None,
                "result": None, "final": None})
    return history


def grade(history: list[dict], games: pd.DataFrame) -> list[dict]:
    g = games.assign(day=pd.to_datetime(games["game_date"]).dt.normalize())
    for b in history:
        if b["result"]:
            continue
        day = (pd.Timestamp(b["commence"]).tz_convert("America/New_York")
               .tz_localize(None).normalize())
        m = g[(g["home_team"] == b["home"]) & (g["away_team"] == b["away"])
              & ((g["day"] - day).abs() <= pd.Timedelta(days=1))]
        if m.empty:
            continue
        m = m.iloc[0]
        spread = b["market"] == "spreads"
        close = m["spread_line"] if spread else m["total_line"]
        # +1 when the model's side is the home team or the over
        sign = (1 if b["side"] in (b["home"], "Over") else -1)
        # Before tip ESPN's line is the current one, not the close.
        if bool(m["completed"]) and pd.notna(close):
            b["close"] = float(close)
            b["clv"] = round(sign * (float(close) - b["line"]), 1)
        if bool(m["completed"]) and pd.notna(m["home_score"]):
            hs, as_ = float(m["home_score"]), float(m["away_score"])
            actual = hs - as_ if spread else hs + as_
            diff = sign * (actual - b["line"])
            b["result"] = "win" if diff > 0 else "loss" if diff < 0 else "push"
            b["final"] = f"{int(as_)}-{int(hs)}"
    return history


def summary(history: list[dict]) -> dict:
    done = [b for b in history if b["result"]]
    w = sum(b["result"] == "win" for b in done)
    lo = sum(b["result"] == "loss" for b in done)
    clv = [b["clv"] for b in history if b["clv"] is not None]
    return {"bets": len(history), "wins": w, "losses": lo,
            "pushes": len(done) - w - lo,
            "win_rate": round(w / (w + lo), 4) if w + lo else None,
            "avg_clv": round(float(np.mean(clv)), 2) if clv else None,
            "beat_close": round(float(np.mean([c > 0 for c in clv])), 4)
            if clv else None}


def save(path: str, history: list[dict]) -> None:
    with open(path, "w") as fh:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(),
                   "gap": GAP, "summary": summary(history),
                   "bets": history}, fh, indent=2, default=str)
