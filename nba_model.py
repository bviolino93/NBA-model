"""
Our own line for each game: ratings plus rest plus tonight's injuries.

    home margin = a * ratings margin + b * back-to-back diff
                  + c * rest diff + d * road back-to-back
                  + e * minutes missing + f * production missing
                  (home minus away) + const
    total       = a * ratings total + b * minutes and c * production
                  missing (both teams) + d * back-to-backs + const

ratings   nba_ratings: opponent-adjusted margin and total from scores,
          recency-weighted, home court
rest      days off, back-to-backs, second night of a road back-to-back
injuries  missing rotation players' expected minutes, and their box
          production above a fill-in's (points + 0.5 rebounds + 0.7
          assists per 36, shrunk; nba_players.absences) over those
          minutes. tools/injury_value.py: these are what the market's
          line moves price (t -16 and -14); plus-minus value isn't
          (t +1.6), so it was dropped. Historically from who actually sat;
          tonight from ESPN's report, weighted by status (out 1,
          doubtful 0.8, questionable or day-to-day 0.4)

The weights are fitted on completed games, each game's ratings built
from earlier games only, so the fit sees what a live model would have.

Fitted on scores alone, the line is pulled toward the league average and
"disagrees" with every extreme market number (tools/market_anchor.py:
80% of 3+ point total gaps were toward the average; 47-49% against the
close). So the line we test is anchored: the same fit with the market's
own line as one more input (ESPN closing lines in history, tonight's
sharp fair line live). A gap then only appears where rest, injuries or
ratings say something the market line doesn't. It runs as a no-money
test (paper.py) until it beats closing lines live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_data import schedule_features, team_games
from nba_ratings import RatingConfig, fit, walk_forward

STATUS_WEIGHT = {"out": 1.0, "doubtful": 0.8, "questionable": 0.4,
                 "day-to-day": 0.4, "game time decision": 0.4}
# Early in a season the ratings rest on a handful of games; these terms
# let the fit trust them less until both teams have EARLY games in.
EARLY = 15
MARGIN_X = ["model_margin", "mm_early", "b2b_diff", "rest_diff", "away_b2b_road",
            "inj_min_diff", "inj_prod_diff"]
TOTAL_X = ["model_total", "mt_early", "early", "inj_min_sum", "inj_prod_sum",
           "b2b_sum"]
INJ = ("min", "prod")
# The anchored line is the market's number plus what these say beyond it.
# The market's own weight is fixed at 1: fitted freely it drifted (0.99 to
# 1.10 early season), which turned every big line into a test bet on the
# favourite. Fitted on (final - market) with an intercept, but intercept
# and feature means are left out of the line, so no input is a standing
# lean toward home teams, favourites, overs or unders.
ANCHOR_M = ["d_margin", "d_margin_early", "b2b_diff", "rest_diff",
            "away_b2b_road", "inj_min_diff", "inj_prod_diff"]
ANCHOR_T = ["d_total", "d_total_early", "inj_min_sum", "inj_prod_sum",
            "b2b_sum"]
PARTS = {"ratings": ("d_margin", "d_margin_early", "d_total", "d_total_early"),
         "rest": ("b2b_diff", "rest_diff", "away_b2b_road", "b2b_sum"),
         "injuries": ("inj_min_diff", "inj_prod_diff", "inj_min_sum",
                      "inj_prod_sum")}


@dataclass
class Fitted:
    margin: dict = field(default_factory=dict)     # name -> weight, "const"
    total: dict = field(default_factory=dict)
    n: int = 0
    anchor_margin: dict = field(default_factory=dict)   # with "mkt_margin"
    anchor_total: dict = field(default_factory=dict)    # with "mkt_total"


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
        g["inj_min_sum"] = g["home_lost_min"] + g["away_lost_min"]
    else:
        for c in ("inj_min_diff", "inj_prod_diff", "inj_min_sum",
                  "inj_prod_sum"):
            g[c] = 0.0
    g = walk_forward(g, cfg)
    t = team_games(g).sort_values("commence")
    t["n_before"] = t.groupby(["team", "season"]).cumcount()
    n = t.set_index(["game_id", "team"])["n_before"]
    hn = n.reindex(list(zip(g["game_id"], g["home_team"]))).to_numpy()
    an = n.reindex(list(zip(g["game_id"], g["away_team"]))).to_numpy()
    _early_terms(g, (np.fmin(hn, an) < EARLY).astype(float))
    return g


def _early_terms(x, early) -> None:
    """Interaction columns on a frame or a dict of features."""
    x["early"] = early
    x["mm_early"] = x["model_margin"] * early
    x["mt_early"] = x["model_total"] * early


def fit_weights(h: pd.DataFrame) -> Fitted:
    d = h[h["completed"] & h["model_margin"].notna()].copy()
    for c in set(MARGIN_X + TOTAL_X):
        d[c] = d[c].fillna(0.0)
    if len(d) < 200:
        return Fitted(n=len(d))
    margin = d["home_score"] - d["away_score"]
    total = d["home_score"] + d["away_score"]
    f = Fitted(_ols(d, margin, MARGIN_X), _ols(d, total, TOTAL_X), len(d))
    if {"spread_line", "total_line"} <= set(d.columns):
        m = d[d["spread_line"].notna() & d["total_line"].notna()].copy()
        if len(m) >= 200:
            f.anchor_margin, f.anchor_total = fit_anchor(m)
    return f


def anchor_cols(x, mkt_margin, mkt_total) -> None:
    """Add the anchored features to a frame or a dict."""
    x["d_margin"] = x["model_margin"] - mkt_margin
    x["d_margin_early"] = x["d_margin"] * x["early"]
    x["d_total"] = x["model_total"] - mkt_total
    x["d_total_early"] = x["d_total"] * x["early"]


def fit_anchor(m: pd.DataFrame) -> tuple[dict, dict]:
    """Weights on (final - market line); each also carries the training
    mean of its feature under "mean:<name>"."""
    m = m.copy()
    anchor_cols(m, m["spread_line"], m["total_line"])
    out = []
    for cols, y in ((ANCHOR_M, m["home_score"] - m["away_score"]
                     - m["spread_line"]),
                    (ANCHOR_T, m["home_score"] + m["away_score"]
                     - m["total_line"])):
        w = _ols(m, y, cols)
        w.update({f"mean:{c}": float(m[c].mean()) for c in cols})
        out.append(w)
    return out[0], out[1]


def anchor_parts(w: dict, x, cols: list[str]) -> dict:
    """{"ratings", "rest", "injuries"}: points each moves the line off the
    market's, features centred on their training means."""
    parts = {}
    for name, members in PARTS.items():
        parts[name] = sum(w[c] * (x[c] - w[f"mean:{c}"])
                          for c in cols if c in members)
    return parts


def player_values(box: pd.DataFrame, last: int = 10,
                  prior_min: float = 300.0) -> pd.DataFrame:
    """Each current player's minutes and production value if he sits
    tonight (as nba_players.absences). Production per 36 uses this season
    and last together, so a star six games in isn't shrunk to a bench
    player's level."""
    from nba_players import PROD_PRIOR, PROD_REPL
    cols = ["player_id", "player", "team", "exp_min", "prod_val"]
    if box is None or box.empty:
        return pd.DataFrame(columns=cols)
    season = box["season"].max()
    b = box[box["season"] == season].sort_values("commence")
    tg = (b.drop_duplicates(["team", "game_id"]).groupby("team").tail(last)
            [["team", "game_id"]])
    n = tg.groupby("team").size().rename("n")
    v = (b.merge(tg, on=["team", "game_id"])
           .groupby(["team", "player_id", "player"])["min"].sum()
           .reset_index().merge(n, on="team"))
    v["exp_min"] = v["min"] / v["n"]
    two = box[box["season"] >= season - 1]
    two = two.assign(prod=two["pts"] + 0.5 * two["reb"] + 0.7 * two["ast"])
    pr = two.groupby("player_id")[["prod", "min"]].sum()
    prod36 = 36.0 * (pr["prod"] + PROD_PRIOR / 36.0 * prior_min) \
        / (pr["min"] + prior_min)
    v["prod36"] = v["player_id"].map(prod36).fillna(PROD_PRIOR)
    v["prod_val"] = v["exp_min"] / 36.0 * (v["prod36"] - PROD_REPL).clip(lower=0)
    return v[cols]


def tonight_missing(report: pd.DataFrame | None, values: pd.DataFrame,
                    min_minutes: float = 15.0) -> dict:
    """team -> ({"min": minutes, "prod": production} expected missing,
    [player notes]) from the report, weighted by status."""
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
        val, notes = out.get(r.team, ({"min": 0.0, "prod": 0.0}, []))
        val["min"] += w * float(row["exp_min"])
        val["prod"] += w * float(row["prod_val"])
        notes.append(f"{r.player} {st}")
        out[r.team] = (val, notes)
    return out


def remove_slate_lean(out: dict, fair: dict, min_games: int = 6) -> dict:
    """Take out the night's common lean from anchored lines.

    Tonight's injuries come from the report (status-weighted) while the
    fit learned from who actually sat, and early-season ratings drift, so
    the inputs can push every game the same way (Nov 4: 13 of 14 totals
    under the market, by about a point). The market prices any real
    league-wide shift, so a game only counts as a disagreement by how far
    it stands out from the rest of the slate: the median gap is removed
    and shown as the "slate" part."""
    for kind, mkt_attr, parts_key in (("margin", "market_margin", "parts"),
                                      ("total", "market_total", "total_parts")):
        keys = [k for k, v in out.items() if v.get("anchored")
                and getattr(fair[k], mkt_attr, None) is not None
                and not np.isnan(getattr(fair[k], mkt_attr))]
        if len(keys) < min_games:
            continue
        lean = float(np.median([out[k][kind] - getattr(fair[k], mkt_attr)
                                for k in keys]))
        for k in keys:
            out[k][kind] -= lean
            if out[k].get(parts_key) is not None:
                out[k][parts_key]["slate"] = round(-lean, 1)
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
    cur = done[done["season"] == season]
    played = pd.concat([cur["home_team"], cur["away_team"]]).value_counts()
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
        none = ({"min": 0.0, "prod": 0.0}, [])
        hv, hn = missing.get(h, none)
        av, an = missing.get(a, none)
        for c in INJ:
            x[f"inj_{c}_diff"] = hv[c] - av[c]
            x[f"inj_{c}_sum"] = hv[c] + av[c]
        x["model_margin"] = float(hca + r[h] - r[a])
        x["model_total"] = float(mu + t.get(h, 0.0) + t.get(a, 0.0))
        _early_terms(x, float(min(played.get(h, 0), played.get(a, 0)) < EARLY))
        mm = getattr(f, "market_margin", None)
        mt = getattr(f, "market_total", None)
        AM, AT = fitted.anchor_margin, fitted.anchor_total
        if AM and mm is not None and mt is not None \
                and not (np.isnan(mm) or np.isnan(mt)):
            anchor_cols(x, float(mm), float(mt))
            mp = anchor_parts(AM, x, ANCHOR_M)
            tp = anchor_parts(AT, x, ANCHOR_T)
            out[k] = {"margin": float(mm) + sum(mp.values()),
                      "total": float(mt) + sum(tp.values()),
                      "total_parts": {"market": round(float(mt), 1),
                                      **{n: round(v, 1) for n, v in tp.items()}},
                      "parts": {"market": round(float(mm), 1),
                                **{n: round(v, 1) for n, v in mp.items()}},
                      "anchored": True,
                      "missing": {"home": hn, "away": an}}
            continue
        if M:
            ratings = (M["model_margin"] * x["model_margin"]
                       + M["mm_early"] * x["mm_early"] + M["const"])
            rest = sum(M[c] * x[c] for c in ("b2b_diff", "rest_diff",
                                             "away_b2b_road"))
            inj = sum(M[f"inj_{c}_diff"] * x[f"inj_{c}_diff"] for c in INJ)
            tot = sum(T[c] * x[c] for c in TOTAL_X) + T["const"]
        else:                                   # no history: ratings only
            ratings, rest, inj, tot = x["model_margin"], 0.0, 0.0, x["model_total"]
        out[k] = {"margin": ratings + rest + inj, "total": tot,
                  "parts": {"ratings": round(ratings, 1),
                            "rest": round(rest, 1),
                            "injuries": round(inj, 1)},
                  "missing": {"home": hn, "away": an}}
    return out
