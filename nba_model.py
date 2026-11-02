"""
Our own line for each game: ratings plus rest plus tonight's injuries.

    home margin = a * ratings margin + b * back-to-back diff
                  + c * rest diff + d * road back-to-back
                  + e * missing value (home minus away) + const
    total       = a * ratings total + b * missing value (both teams)
                  + c * back-to-backs (both teams) + const

ratings   nba_ratings: opponent-adjusted margin and total from scores,
          recency-weighted, home court
rest      days off, back-to-backs, second night of a road back-to-back
injuries  value of missing rotation players: expected minutes x their
          plus-minus per 36 (shrunk) plus a replacement drop (as in
          nba_players.absences). Historically from who actually sat;
          tonight from ESPN's report, weighted by status (out 1,
          doubtful 0.8, questionable or day-to-day 0.4)

The weights are fitted on completed games, each game's ratings built
from earlier games only, so the fit sees what a live model would have.
The backtest says this doesn't beat closing lines, so it runs as a
no-money test (paper.py) until it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_data import schedule_features
from nba_ratings import RatingConfig, fit, walk_forward

STATUS_WEIGHT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.4,
                 "day-to-day": 0.4, "game time decision": 0.4}
MARGIN_X = ["model_margin", "b2b_diff", "rest_diff", "away_b2b_road",
            "inj_val_diff"]
TOTAL_X = ["model_total", "inj_val_sum", "b2b_sum"]


@dataclass
class Fitted:
    margin: dict = field(default_factory=dict)     # name -> weight, "const"
    total: dict = field(default_factory=dict)
    n: int = 0


def _ols(df: pd.DataFrame, y: pd.Series, cols: list[str]) -> dict:
    X = np.column_stack([df[c].to_numpy(float) for c in cols]
                        + [np.ones(len(df))])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)
    return dict(zip(cols + ["const"], map(float, beta)))


def history(games: pd.DataFrame, box: pd.DataFrame | None,
            cfg: RatingConfig | None = None) -> pd.DataFrame:
    """Completed games with walk-forward ratings and every feature."""
    g = schedule_features(games)
    g["b2b_sum"] = g["home_b2b"].fillna(0) + g["away_b2b"].fillna(0)
    if box is not None and len(box):
        from nba_players import absences, game_absences
        g = game_absences(g, absences(box))
    else:
        g["inj_val_diff"] = 0.0
        g["inj_val_sum"] = 0.0
    return walk_forward(g, cfg)


def fit_weights(h: pd.DataFrame) -> Fitted:
    d = h[h["completed"] & h["model_margin"].notna()].copy()
    for c in set(MARGIN_X + TOTAL_X):
        d[c] = d[c].fillna(0.0)
    if len(d) < 200:
        return Fitted(n=len(d))
    margin = d["home_score"] - d["away_score"]
    total = d["home_score"] + d["away_score"]
    return Fitted(_ols(d, margin, MARGIN_X), _ols(d, total, TOTAL_X), len(d))


def player_values(box: pd.DataFrame, last: int = 10,
                  prior_min: float = 300.0) -> pd.DataFrame:
    """Each current player's value if he sits tonight (same formula as
    nba_players.absences): exp_min/36 * max(pm36, 0) + exp_min/48."""
    if box is None or box.empty:
        return pd.DataFrame(columns=["player_id", "player", "team", "value",
                                     "exp_min"])
    season = box["season"].max()
    b = box[box["season"] == season].sort_values("commence")
    tg = (b.drop_duplicates(["team", "game_id"]).groupby("team").tail(last)
            [["team", "game_id"]])
    n = tg.groupby("team").size().rename("n")
    recent = (b.merge(tg, on=["team", "game_id"])
                .groupby(["team", "player_id", "player"])["min"].sum()
                .reset_index().merge(n, on="team"))
    recent["exp_min"] = recent["min"] / recent["n"]
    tot = b.groupby(["team", "player_id"])[["pm", "min"]].sum().reset_index()
    tot["pm36"] = 36.0 * tot["pm"] / (tot["min"] + prior_min)
    v = recent.merge(tot[["team", "player_id", "pm36"]],
                     on=["team", "player_id"], how="left")
    v["value"] = (v["exp_min"] / 36.0 * v["pm36"].fillna(0).clip(lower=0)
                  + v["exp_min"] / 48.0)
    return v[["player_id", "player", "team", "value", "exp_min"]]


def tonight_missing(report: pd.DataFrame | None, values: pd.DataFrame,
                    min_minutes: float = 15.0) -> dict:
    """team -> (expected missing value, [player notes]) from the report."""
    out: dict = {}
    if report is None or report.empty or values.empty:
        return out
    by_id = values.drop_duplicates("player_id").set_index("player_id")
    by_name = values.drop_duplicates("player").set_index("player")
    for r in report.itertuples():
        st = str(r.status).lower()
        w = next((v for k, v in STATUS_WEIGHT.items() if k in st), 0.0)
        if not w:
            continue
        row = (by_id.loc[r.player_id] if r.player_id in by_id.index
               else by_name.loc[r.player] if r.player in by_name.index
               else None)
        if row is None or row["exp_min"] < min_minutes:
            continue
        val, notes = out.get(r.team, (0.0, []))
        val += w * float(row["value"])
        notes.append(f"{r.player} {st}")
        out[r.team] = (val, notes)
    return out


def predict(games: pd.DataFrame, fair: dict, fitted: Fitted,
            missing: dict, now=None,
            cfg: RatingConfig | None = None) -> dict[str, dict]:
    """{fair key: margin, total and the parts behind them}."""
    cfg = cfg or RatingConfig()
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    today = now.tz_convert("America/New_York").tz_localize(None).normalize()
    g = schedule_features(games)
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
    M, T = fitted.margin, fitted.total
    out = {}
    for k, f in fair.items():
        h, a = f.home_team, f.away_team
        if h not in r or a not in r:
            continue
        day = (pd.Timestamp(f.commence).tz_convert("America/New_York")
               .tz_localize(None).normalize())
        row = g[(g["home_team"] == h) & (g["away_team"] == a)
                & ((g["game_date"] - day).abs() <= pd.Timedelta(days=1))]
        x = {c: 0.0 for c in MARGIN_X + TOTAL_X}
        if len(row):
            row = row.iloc[0]
            for c in ("b2b_diff", "rest_diff", "away_b2b_road"):
                x[c] = 0.0 if pd.isna(row[c]) else float(row[c])
            x["b2b_sum"] = float(np.nansum([row["home_b2b"], row["away_b2b"]]))
        hv, hn = missing.get(h, (0.0, []))
        av, an = missing.get(a, (0.0, []))
        x["inj_val_diff"], x["inj_val_sum"] = hv - av, hv + av
        x["model_margin"] = float(hca + r[h] - r[a])
        x["model_total"] = float(mu + t.get(h, 0.0) + t.get(a, 0.0))
        if M:
            ratings = M["model_margin"] * x["model_margin"] + M["const"]
            rest = sum(M[c] * x[c] for c in ("b2b_diff", "rest_diff",
                                             "away_b2b_road"))
            inj = M["inj_val_diff"] * x["inj_val_diff"]
            tot = sum(T[c] * x[c] for c in TOTAL_X) + T["const"]
        else:                                   # no history: ratings only
            ratings, rest, inj, tot = x["model_margin"], 0.0, 0.0, x["model_total"]
        out[k] = {"margin": ratings + rest + inj, "total": tot,
                  "parts": {"ratings": round(ratings, 1),
                            "rest": round(rest, 1),
                            "injuries": round(inj, 1)},
                  "missing": {"home": hn, "away": an}}
    return out
