"""
Our own prediction of every game: home margin and total points, built
from players, possessions and the schedule, with no market input.

The first own line (nba_model) used team ratings from final scores, rest,
back-to-backs and missing minutes. This one adds what a basketball
analyst would look at first:

players     who is actually playing tonight and how good each is.
            * apm: each player's impact on the scoreboard per 48 minutes,
              a ridge regression of every game's margin on the minutes of
              the players in it (adjusted plus-minus at game level),
              recency-weighted, refitted every game day on earlier games
            * box: each player's box-score production per 36 minutes
              (Hollinger game score), shrunk toward a bench player
            A team's number for a game is the minutes-weighted sum over the
            players available for it, each at his expected minutes (his
            last 10 games played). Injuries, trades and rest days are in
            here directly instead of as a separate "minutes missing" term.
possessions offense and defense per 100 possessions and pace, opponent-
            adjusted (ridge, recency-weighted), so the line reads
            "pace x efficiency" rather than raw scores; plus each team's
            four factors (shooting, turnovers, offensive rebounds, free
            throws) and three-point rate, for and against
schedule    rest, back-to-backs, games in the last 5 nights, travel
            distance and time zones crossed since the last game, altitude
            (Denver, Utah)

Everything for a game is built from games before its date only, so a
fitted model sees what a live one would have. The models that turn these
into a line (ridge, and boosted trees on the ridge's misses) are fitted in
fit_models; tools/model_lab.py tests them season by season against the
final scores and the closing line.

Historically "available" means the players who played (the closing line
knew about late scratches too). Tonight it comes from ESPN's injury
report: out players are dropped, questionable ones count at partial
minutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_data import schedule_features
from nba_ratings import RatingConfig, walk_forward

# ----------------------------------------------------------------------
# Arenas: latitude, longitude, UTC offset in winter, altitude flag
# ----------------------------------------------------------------------
ARENAS = {
    "Atlanta Hawks": (33.757, -84.396, -5, 0),
    "Boston Celtics": (42.366, -71.062, -5, 0),
    "Brooklyn Nets": (40.683, -73.976, -5, 0),
    "Charlotte Hornets": (35.225, -80.839, -5, 0),
    "Chicago Bulls": (41.881, -87.674, -6, 0),
    "Cleveland Cavaliers": (41.496, -81.688, -5, 0),
    "Dallas Mavericks": (32.790, -96.810, -6, 0),
    "Denver Nuggets": (39.749, -105.008, -7, 1),
    "Detroit Pistons": (42.341, -83.055, -5, 0),
    "Golden State Warriors": (37.768, -122.388, -8, 0),
    "Houston Rockets": (29.751, -95.362, -6, 0),
    "Indiana Pacers": (39.764, -86.155, -5, 0),
    "Los Angeles Clippers": (33.945, -118.343, -8, 0),
    "Los Angeles Lakers": (34.043, -118.267, -8, 0),
    "Memphis Grizzlies": (35.138, -90.051, -6, 0),
    "Miami Heat": (25.781, -80.188, -5, 0),
    "Milwaukee Bucks": (43.045, -87.917, -6, 0),
    "Minnesota Timberwolves": (44.979, -93.276, -6, 0),
    "New Orleans Pelicans": (29.949, -90.082, -6, 0),
    "New York Knicks": (40.751, -73.993, -5, 0),
    "Oklahoma City Thunder": (35.463, -97.515, -6, 0),
    "Orlando Magic": (28.539, -81.384, -5, 0),
    "Philadelphia 76ers": (39.901, -75.172, -5, 0),
    "Phoenix Suns": (33.446, -112.071, -7, 0),
    "Portland Trail Blazers": (45.532, -122.667, -8, 0),
    "Sacramento Kings": (38.580, -121.500, -8, 0),
    "San Antonio Spurs": (29.427, -98.437, -6, 0),
    "Toronto Raptors": (43.643, -79.379, -5, 0),
    "Utah Jazz": (40.768, -111.901, -7, 1),
    "Washington Wizards": (38.898, -77.021, -5, 0),
}


def _km(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


@dataclass
class PredictConfig:
    ratings: RatingConfig = field(default_factory=RatingConfig)
    eff_half_life: float = 120.0     # days, possessions ratings
    eff_ridge: float = 3.0
    apm_half_life: float = 365.0     # days, player impact
    apm_ridge: float = 8.0
    apm_seasons: int = 2             # earlier seasons kept in the apm fit
    exp_games: int = 10              # expected minutes: last N games played
    box_prior_min: float = 400.0
    box_prior: float = 9.0           # game score per 36 of a bench player
    ff_half_life: float = 15.0       # four factors, in team games
    min_prior_games: int = 200


# ----------------------------------------------------------------------
# Travel and altitude
# ----------------------------------------------------------------------
def travel_features(g: pd.DataFrame) -> pd.DataFrame:
    """Kilometres and time zones each team moved since its last game, and
    the altitude flag of the arena."""
    cols = ["game_id", "season", "commence"]
    home = g[cols].assign(team=g["home_team"], site=g["home_team"])
    away = g[cols].assign(team=g["away_team"], site=g["home_team"])
    t = pd.concat([home, away]).sort_values(["team", "commence"])
    t["prev_site"] = t.groupby("team")["site"].shift(1)
    same = t["season"] == t.groupby("team")["season"].shift(1)
    t["km"] = [(_km(ARENAS.get(p), ARENAS.get(s)) if isinstance(p, str)
                else float("nan")) for p, s in zip(t["prev_site"], t["site"])]
    tz = lambda x: ARENAS.get(x, (0, 0, float("nan"), 0))[2]  # noqa: E731
    t["tz"] = [tz(s) - tz(p) if isinstance(p, str) else float("nan")
               for p, s in zip(t["prev_site"], t["site"])]
    t.loc[~same, ["km", "tz"]] = np.nan
    k = t.set_index(["game_id", "team"])
    out = g.copy()
    for side in ("home", "away"):
        idx = list(zip(out["game_id"], out[f"{side}_team"]))
        out[f"{side}_km"] = k["km"].reindex(idx).to_numpy() / 1000.0
        out[f"{side}_tz"] = k["tz"].reindex(idx).to_numpy()
    out["km_diff"] = out["home_km"].fillna(0) - out["away_km"].fillna(0)
    out["away_tz_abs"] = out["away_tz"].abs().fillna(0)
    out["alt_home"] = [ARENAS.get(h, (0, 0, 0, 0))[3] for h in out["home_team"]]
    return out


# ----------------------------------------------------------------------
# Team box scores, possessions and four factors
# ----------------------------------------------------------------------
STAT = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
        "tov", "min"]


def team_box(p: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: totals, possessions, the four factors,
    for and against."""
    if p is None or p.empty or "fga" not in p:
        return pd.DataFrame()
    t = (p.groupby(["game_id", "team"], as_index=False)[STAT].sum(min_count=1))
    o = t.rename(columns={c: f"o_{c}" for c in STAT})
    o = o.rename(columns={"team": "opp"})
    m = t.merge(p.drop_duplicates(["game_id", "team"])[["game_id", "team", "opp"]],
                on=["game_id", "team"]).merge(o, on=["game_id", "opp"])
    own = m["fga"] - m["oreb"] + m["tov"] + 0.44 * m["fta"]
    opp = m["o_fga"] - m["o_oreb"] + m["o_tov"] + 0.44 * m["o_fta"]
    m["poss"] = 0.5 * (own + opp)
    m["pace"] = m["poss"] * 240.0 / m["min"].where(m["min"] > 0)
    m["ortg"] = 100.0 * m["pts"] / m["poss"]
    m["efg"] = (m["fgm"] + 0.5 * m["fg3m"]) / m["fga"]
    m["tovr"] = m["tov"] / m["poss"]
    m["orbr"] = m["oreb"] / (m["oreb"] + m["o_dreb"])
    m["ftr"] = m["ftm"] / m["fga"]
    m["par3"] = m["fg3a"] / m["fga"]
    m["o_efg"] = (m["o_fgm"] + 0.5 * m["o_fg3m"]) / m["o_fga"]
    m["o_tovr"] = m["o_tov"] / m["poss"]
    m["o_orbr"] = m["o_oreb"] / (m["o_oreb"] + m["dreb"])
    m["o_ftr"] = m["o_ftm"] / m["o_fga"]
    m["o_par3"] = m["o_fg3a"] / m["o_fga"]
    ok = (m["fga"] > 40) & (m["o_fga"] > 40) & (m["min"] >= 200)
    return m[ok].replace([np.inf, -np.inf], np.nan)


FF = ["efg", "tovr", "orbr", "ftr", "par3"]


def four_factors(g: pd.DataFrame, tb: pd.DataFrame,
                 half_life: float) -> pd.DataFrame:
    """Each team's recent four factors going into each game (exponentially
    weighted over its earlier games, for and against)."""
    if tb.empty:
        return pd.DataFrame(columns=["game_id", "team"])
    cols = FF + [f"o_{c}" for c in FF]
    # Every team-game, played or not, so a game not played yet gets the
    # team's latest numbers.
    base = pd.concat([g[["game_id", "commence", "home_team"]]
                      .rename(columns={"home_team": "team"}),
                      g[["game_id", "commence", "away_team"]]
                      .rename(columns={"away_team": "team"})])
    t = base.merge(tb[["game_id", "team"] + cols], on=["game_id", "team"],
                   how="left")
    t = t.sort_values(["team", "commence"]).reset_index(drop=True)
    ew = (t.groupby("team")[cols]
           .transform(lambda s: s.shift(1).ewm(halflife=half_life,
                                               min_periods=3,
                                               ignore_na=True).mean()
                      .ffill()))
    ew.columns = [f"ff_{c}" for c in cols]
    return pd.concat([t[["game_id", "team"]], ew], axis=1)


# ----------------------------------------------------------------------
# Walk-forward ridge helpers
# ----------------------------------------------------------------------
def _weights(dates, seasons, day, season, half_life, offseason):
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    age = np.asarray((pd.Timestamp(day) - d).days, float)
    age = age + offseason * (season - np.asarray(seasons, float))
    return 0.5 ** (age / half_life)


def efficiency_ratings(g: pd.DataFrame, tb: pd.DataFrame,
                       cfg: PredictConfig) -> pd.DataFrame:
    """Per game: predicted pace and each side's points per 100 possessions,
    from opponent-adjusted ratings fitted on earlier games.

        ortg(team vs opp) = mu + home + off[team] - def[opp]
        pace(game)        = p0 + pace[home] + pace[away]
    """
    out = pd.DataFrame({"game_id": g["game_id"]})
    for c in ("pace_hat", "home_ortg_hat", "away_ortg_hat"):
        out[c] = np.nan
    if tb.empty:
        return out
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    info = g.set_index("game_id")[["home_team", "game_date", "season",
                                   "neutral"]]
    r = tb.merge(info, left_on="game_id", right_index=True)
    r = r[r["poss"].notna() & r["ortg"].notna()]
    r["is_home"] = ((r["team"] == r["home_team"])
                    & ~r["neutral"].astype(bool)).astype(float)
    r["ti"] = r["team"].map(idx)
    r["oi"] = r["opp"].map(idx)
    r = r.sort_values("game_date").reset_index(drop=True)
    gd = pd.to_datetime(r["game_date"]).to_numpy()
    res = {}
    for day, todays in g.groupby("game_date", sort=True):
        season = int(todays["season"].iloc[0])
        m = (gd < np.datetime64(pd.Timestamp(day))) & \
            (r["season"].to_numpy() >= season - 2)
        prior = r[m]
        if len(prior) < 2 * cfg.min_prior_games:
            continue
        w = _weights(prior["game_date"], prior["season"], day, season,
                     cfg.eff_half_life, cfg.ratings.offseason_days)
        n = len(prior)
        rows = np.arange(n)
        X = np.zeros((n, 2 + 2 * k))
        X[:, 0] = 1.0
        X[:, 1] = prior["is_home"].to_numpy()
        X[rows, 2 + prior["ti"].to_numpy()] = 1.0
        X[rows, 2 + k + prior["oi"].to_numpy()] = -1.0
        P = np.eye(2 + 2 * k) * cfg.eff_ridge
        P[0, 0] = P[1, 1] = 1e-6
        beta = np.linalg.solve(X.T @ (X * w[:, None]) + P,
                               X.T @ (w * prior["ortg"].to_numpy()))
        # Pace: one row per team-game is fine (each game counted twice)
        Z = np.zeros((n, 1 + k))
        Z[:, 0] = 1.0
        Z[rows, 1 + prior["ti"].to_numpy()] += 1.0
        Z[rows, 1 + prior["oi"].to_numpy()] += 1.0
        Q = np.eye(1 + k) * cfg.eff_ridge
        Q[0, 0] = 1e-6
        gam = np.linalg.solve(Z.T @ (Z * w[:, None]) + Q,
                              Z.T @ (w * prior["pace"].to_numpy()))
        off, dfn = beta[2:2 + k], beta[2 + k:]
        for row in todays.itertuples():
            hi, ai = idx[row.home_team], idx[row.away_team]
            hm = 0.0 if row.neutral else 1.0
            res[row.game_id] = (gam[0] + gam[1 + hi] + gam[1 + ai],
                                beta[0] + hm * beta[1] + off[hi] - dfn[ai],
                                beta[0] + off[ai] - dfn[hi])
    v = pd.DataFrame.from_dict(res, orient="index",
                               columns=["pace_hat", "home_ortg_hat",
                                        "away_ortg_hat"])
    return out[["game_id"]].merge(v, left_on="game_id", right_index=True,
                                  how="left")


# ----------------------------------------------------------------------
# Players: expected minutes, box value, adjusted plus-minus
# ----------------------------------------------------------------------
def game_score(p: pd.DataFrame) -> pd.Series:
    """Hollinger game score."""
    f = lambda c: p[c].fillna(0.0) if c in p else 0.0  # noqa: E731
    return (f("pts") + 0.4 * f("fgm") - 0.7 * f("fga")
            - 0.4 * (f("fta") - f("ftm")) + 0.7 * f("oreb") + 0.3 * f("dreb")
            + f("stl") + 0.7 * f("ast") + 0.7 * f("blk") - 0.4 * f("pf")
            - f("tov"))


def player_form(p: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    """Per player row (played games only), going INTO that game:
    exp_min (mean minutes over his last cfg.exp_games games played) and
    box36 (game score per 36 over his last 164 games, shrunk)."""
    q = p[p["min"] > 0].sort_values(["player_id", "commence"]).copy()
    q["gs"] = game_score(q)
    grp = q.groupby("player_id")
    q["exp_min"] = grp["min"].transform(
        lambda s: s.shift(1).rolling(cfg.exp_games, min_periods=1).mean())
    sg = grp["gs"].transform(lambda s: s.shift(1).rolling(164, min_periods=1)
                             .sum()).fillna(0.0)
    sm = grp["min"].transform(lambda s: s.shift(1).rolling(164, min_periods=1)
                              .sum()).fillna(0.0)
    q["box36"] = 36.0 * (sg + cfg.box_prior / 36.0 * cfg.box_prior_min) \
        / (sm + cfg.box_prior_min)
    q["exp_min"] = q["exp_min"].fillna(12.0)
    return q


def latest_form(p: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    """Each player's exp_min and box36 going into his NEXT game."""
    q = p[p["min"] > 0].sort_values(["player_id", "commence"]).copy()
    q["gs"] = game_score(q)
    grp = q.groupby("player_id")
    last = grp.tail(cfg.exp_games).groupby("player_id")["min"].mean()
    tail = grp.tail(164).groupby("player_id")[["gs", "min"]].sum()
    box36 = 36.0 * (tail["gs"] + cfg.box_prior / 36.0 * cfg.box_prior_min) \
        / (tail["min"] + cfg.box_prior_min)
    return pd.DataFrame({"exp_min": last, "box36": box36})


def apm_ratings(g: pd.DataFrame, p: pd.DataFrame, cfg: PredictConfig,
                days=None):
    """Walk-forward player impact (points of margin per 48 minutes).

    Returns {game day: {player_id: rating}, "_hca": {day: hca}} for every
    day in `days` (default: every day in g with games), each fitted on
    completed games before that day."""
    from scipy import sparse
    done = g[g["completed"]].copy()
    q = p[(p["min"] > 0) & p["game_id"].isin(done["game_id"])]
    tm = q.groupby(["game_id", "team"])["min"].transform("sum")
    q = q.assign(share=5.0 * q["min"] / tm)
    home = done.set_index("game_id")["home_team"]
    q = q.assign(sign=np.where(q["team"].to_numpy()
                               == home.reindex(q["game_id"]).to_numpy(),
                               1.0, -1.0))
    gids = done["game_id"][done["game_id"].isin(q["game_id"])].tolist()
    gi = {x: i for i, x in enumerate(gids)}
    pids = sorted(q["player_id"].unique())
    pi = {x: i for i, x in enumerate(pids)}
    X = sparse.csr_matrix(((q["share"] * q["sign"]).to_numpy(),
                           (q["game_id"].map(gi).to_numpy(),
                            q["player_id"].map(pi).to_numpy())),
                          shape=(len(gids), len(pids)))
    d = done.set_index("game_id").loc[gids]
    y = (d["home_score"] - d["away_score"]).to_numpy(float)
    hca = 1.0 - d["neutral"].astype(float).to_numpy()
    dates = pd.to_datetime(d["game_date"])
    dnp = dates.to_numpy()
    seasons = d["season"].to_numpy()
    out, hcas = {}, {}
    if days is None:
        days = sorted(g["game_date"].unique())
    seasons_of = g.groupby("game_date")["season"].first()
    for day in days:
        season = int(seasons_of.get(day, seasons.max() if len(seasons) else 0))
        m = (dnp < np.datetime64(pd.Timestamp(day))) & \
            (seasons >= season - cfg.apm_seasons)
        if m.sum() < cfg.min_prior_games:
            continue
        rows = np.flatnonzero(m)
        Xs = X[rows]
        cols = np.unique(Xs.indices)
        Xs = Xs[:, cols]
        w = _weights(dates.iloc[rows], seasons[rows], day, season,
                     cfg.apm_half_life, cfg.ratings.offseason_days)
        h = hca[rows]
        W = sparse.diags(w)
        A = (Xs.T @ W @ Xs).toarray() + cfg.apm_ridge * np.eye(len(cols))
        xh = Xs.T @ (w * h)
        top = np.concatenate([[float(h @ (w * h)) + 1e-6], xh])
        A = np.block([[top[None, :]], [np.column_stack([xh, A])]])
        b = np.concatenate([[float(h @ (w * y[rows]))],
                            Xs.T @ (w * y[rows])])
        beta = np.linalg.solve(A, b)
        hcas[day] = float(beta[0])
        out[day] = dict(zip((pids[c] for c in cols), beta[1:]))
    out["_hca"] = hcas
    return out


def lineup_strength(avail: pd.DataFrame, ratings: dict) -> pd.DataFrame:
    """Per (game_id, team): minutes-weighted apm and box36 over the
    available players.

    avail: game_id, team, game_date, player_id, exp_min, box36, weight
    (1 = available; less for questionable)."""
    a = avail.copy()
    a["m"] = a["exp_min"].fillna(12.0) * a.get("weight", 1.0)
    tot = a.groupby(["game_id", "team"])["m"].transform("sum")
    a["share"] = 5.0 * a["m"] / tot.where(tot > 0)
    a["apm"] = [ratings.get(d, {}).get(pid, 0.0)
                for d, pid in zip(a["game_date"], a["player_id"])]
    a["has"] = [d in ratings for d in a["game_date"]]
    a["apm_x"] = a["share"] * a["apm"]
    a["box_x"] = a["share"] * a["box36"]
    s = a.groupby(["game_id", "team"]).agg(apm=("apm_x", "sum"),
                                           box=("box_x", "sum"),
                                           has=("has", "max"),
                                           n_avail=("player_id", "size"))
    s.loc[~s["has"].astype(bool), "apm"] = np.nan
    return s.drop(columns="has").reset_index()


# ----------------------------------------------------------------------
# Everything per game
# ----------------------------------------------------------------------
MARGIN_F = ["model_margin", "eff_margin", "apm_diff", "box_diff",
            "rest_diff", "b2b_diff", "away_b2b_road", "last5_diff",
            "km_diff", "away_tz_abs", "alt_home", "early", "neutral_f",
            "ff_efg_m", "ff_tov_m", "ff_orb_m", "ff_ftr_m", "ff_3p_m"]
TOTAL_F = ["model_total", "eff_total", "pace_hat", "b2b_sum", "last5_sum",
           "km_sum", "alt_home", "early", "ff_efg_s", "ff_tov_s",
           "ff_orb_s", "ff_ftr_s", "ff_3p_s", "box_sum", "apm_sum",
           "month"]


def build(games: pd.DataFrame, box: pd.DataFrame,
          cfg: PredictConfig | None = None, upcoming: pd.DataFrame | None = None,
          first_season: int | None = None, verbose: bool = False
          ) -> pd.DataFrame:
    """Every game in `games` with the features above, walk-forward.

    upcoming: availability rows for games not played yet (game_id, team,
    player_id, weight), from tonight_availability. Completed games use
    the players who played."""
    cfg = cfg or PredictConfig()
    g = schedule_features(games)
    g = travel_features(g)
    g["b2b_sum"] = g["home_b2b"].fillna(0) + g["away_b2b"].fillna(0)
    g["last5_diff"] = g["home_last5"].fillna(0) - g["away_last5"].fillna(0)
    g["last5_sum"] = g["home_last5"].fillna(0) + g["away_last5"].fillna(0)
    g["km_sum"] = g["home_km"].fillna(0) + g["away_km"].fillna(0)
    g["neutral_f"] = g["neutral"].astype(float)
    g["month"] = pd.to_datetime(g["game_date"]).dt.month.map(
        lambda m: (m - 10) % 12)
    if verbose:
        print("  score ratings ...", flush=True)
    g = walk_forward(g, cfg.ratings, first_season)
    gp = pd.concat([g[["game_id", "season", "commence", "home_team"]]
                    .rename(columns={"home_team": "team"}),
                    g[["game_id", "season", "commence", "away_team"]]
                    .rename(columns={"away_team": "team"})])
    gp = gp.sort_values("commence")
    gp["n"] = gp.groupby(["team", "season"]).cumcount()
    n = gp.set_index(["game_id", "team"])["n"]
    hn = n.reindex(list(zip(g["game_id"], g["home_team"]))).to_numpy()
    an = n.reindex(list(zip(g["game_id"], g["away_team"]))).to_numpy()
    g["early"] = (np.fmin(hn, an) < 15).astype(float)

    have_box = box is not None and len(box) and "fga" in box
    if not have_box:
        return g
    box = box.copy()
    box["game_id"] = box["game_id"].astype(str)
    tb = team_box(box)
    if verbose:
        print("  possessions ratings ...", flush=True)
    er = efficiency_ratings(g, tb, cfg)
    g = g.merge(er, on="game_id", how="left")
    g["eff_margin"] = g["pace_hat"] * (g["home_ortg_hat"]
                                       - g["away_ortg_hat"]) / 100.0
    g["eff_total"] = g["pace_hat"] * (g["home_ortg_hat"]
                                      + g["away_ortg_hat"]) / 100.0
    ff = four_factors(g, tb, cfg.ff_half_life).set_index(["game_id", "team"])
    for side in ("home", "away"):
        idx = list(zip(g["game_id"], g[f"{side}_team"]))
        for c in ff.columns:
            g[f"{side}_{c}"] = ff[c].reindex(idx).to_numpy()
    # Matchup: my offense's factor and what the opponent's defense allows
    for f, lab, sign in (("efg", "efg", 1), ("tovr", "tov", -1),
                         ("orbr", "orb", 1), ("ftr", "ftr", 1),
                         ("par3", "3p", 1)):
        hm = g[f"home_ff_{f}"] + g[f"away_ff_o_{f}"]
        am = g[f"away_ff_{f}"] + g[f"home_ff_o_{f}"]
        g[f"ff_{lab}_m"] = sign * (hm - am)
        g[f"ff_{lab}_s"] = hm + am

    if verbose:
        print("  player form and impact ...", flush=True)
    form = player_form(box, cfg)
    av = form[["game_id", "team", "game_date", "player_id", "exp_min",
               "box36"]].assign(weight=1.0)
    if upcoming is not None and len(upcoming):
        lf = latest_form(box, cfg)
        u = upcoming.copy()
        u["game_id"] = u["game_id"].astype(str)
        u = u.merge(g[["game_id", "game_date"]], on="game_id")
        u = u.join(lf, on="player_id")
        u["box36"] = u["box36"].fillna(cfg.box_prior)
        av = pd.concat([av, u[av.columns]], ignore_index=True)
    days = sorted(set(av["game_date"]))
    if first_season:
        keep = set(g.loc[g["season"] >= first_season, "game_date"])
        days = [d for d in days if d in keep]
    rat = apm_ratings(g, box, cfg, days=days)
    hcas = rat.pop("_hca")
    ls = lineup_strength(av, rat).set_index(["game_id", "team"])
    for side in ("home", "away"):
        idx = list(zip(g["game_id"], g[f"{side}_team"]))
        for c in ("apm", "box", "n_avail"):
            g[f"{side}_{c}"] = ls[c].reindex(idx).to_numpy()
    g["apm_hca"] = g["game_date"].map(hcas)
    g["apm_diff"] = (g["home_apm"] - g["away_apm"]
                     + g["apm_hca"].fillna(0) * (1 - g["neutral_f"]))
    g["apm_sum"] = g["home_apm"] + g["away_apm"]
    g["box_diff"] = g["home_box"] - g["away_box"]
    g["box_sum"] = g["home_box"] + g["away_box"]
    return g


# ----------------------------------------------------------------------
# Models: ridge on the features, boosted trees on the ridge's misses
# ----------------------------------------------------------------------
@dataclass
class Models:
    margin: dict = field(default_factory=dict)   # {"ridge", "gbm", "cols", "fill"}
    total: dict = field(default_factory=dict)
    n: int = 0


def _X(d: pd.DataFrame, cols, fill: dict) -> np.ndarray:
    return np.column_stack([d[c].astype(float).fillna(fill.get(c, 0.0))
                            .to_numpy() if c in d else
                            np.full(len(d), fill.get(c, 0.0)) for c in cols])


def _fit_one(d: pd.DataFrame, y: np.ndarray, cols, base: str,
             trees: bool) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fill = {c: float(d[c].median()) if c in d and d[c].notna().any() else 0.0
            for c in cols}
    # The ratings baseline fills where a newer input is missing.
    X = _X(d, cols, fill)
    ridge = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 3, 12)))
    ridge.fit(X, y)
    out = {"ridge": ridge, "cols": list(cols), "fill": fill, "gbm": None}
    if trees:
        from sklearn.ensemble import HistGradientBoostingRegressor
        resid = y - ridge.predict(X)
        Xn = np.column_stack([d[c].astype(float).to_numpy() if c in d else
                              np.full(len(d), np.nan) for c in cols])
        gbm = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300,
            min_samples_leaf=100, l2_regularization=1.0, random_state=0)
        gbm.fit(Xn, resid)
        out["gbm"] = gbm
    return out


def fit_models(h: pd.DataFrame, trees: bool = True,
               margin_cols=None, total_cols=None) -> Models:
    d = h[h["completed"] & h["model_margin"].notna()]
    if len(d) < 500:
        return Models(n=len(d))
    mc = [c for c in (margin_cols or MARGIN_F) if c in d]
    tc = [c for c in (total_cols or TOTAL_F) if c in d]
    ym = (d["home_score"] - d["away_score"]).to_numpy(float)
    yt = (d["home_score"] + d["away_score"]).to_numpy(float)
    return Models(_fit_one(d, ym, mc, "model_margin", trees),
                  _fit_one(d, yt, tc, "model_total", trees), len(d))


def _pred_one(m: dict, d: pd.DataFrame, kind: str) -> np.ndarray:
    X = _X(d, m["cols"], m["fill"])
    r = m["ridge"].predict(X)
    if kind == "ridge" or m.get("gbm") is None:
        return r
    Xn = np.column_stack([d[c].astype(float).to_numpy() if c in d else
                          np.full(len(d), np.nan) for c in m["cols"]])
    return r + m["gbm"].predict(Xn)


def predict_frame(models: Models, d: pd.DataFrame, kind: str = "trees"
                  ) -> pd.DataFrame:
    """pred_margin and pred_total for every row of d."""
    out = pd.DataFrame(index=d.index)
    if not models.margin:
        out["pred_margin"] = d.get("model_margin")
        out["pred_total"] = d.get("model_total")
        return out
    out["pred_margin"] = _pred_one(models.margin, d, kind)
    out["pred_total"] = _pred_one(models.total, d, kind)
    return out


def ridge_weights(m: dict) -> dict:
    """Points per one standard deviation of each input (for reading)."""
    ridge = m["ridge"]
    coef = ridge[-1].coef_
    return dict(zip(m["cols"], np.round(coef, 2)))


# ----------------------------------------------------------------------
# Tonight
# ----------------------------------------------------------------------
STATUS_KEEP = {"out": 0.0, "doubtful": 0.2, "questionable": 0.6,
               "day-to-day": 0.7, "game time decision": 0.6}


def tonight_availability(box: pd.DataFrame, upcoming: pd.DataFrame,
                         report: pd.DataFrame | None,
                         recent_games: int = 5) -> pd.DataFrame:
    """Who is expected to play in each not-yet-played game: every player
    who played for the team in its last `recent_games` games (his latest
    team only), weighted down or dropped by the injury report.

    upcoming: game_id, home_team, away_team. report: team, player,
    player_id, status (nba_injuries)."""
    cols = ["game_id", "team", "player_id", "weight", "player"]
    if box is None or box.empty or upcoming is None or upcoming.empty:
        return pd.DataFrame(columns=cols)
    b = box[box["min"] > 0].sort_values("commence")
    latest = b.groupby("player_id").tail(1).set_index("player_id")["team"]
    tg = (b.drop_duplicates(["team", "game_id"]).groupby("team")
           .tail(recent_games)[["team", "game_id"]])
    roster = (b.merge(tg, on=["team", "game_id"])
                .drop_duplicates(["team", "player_id"])[["team", "player_id",
                                                          "player"]])
    roster = roster[roster["player_id"].map(latest) == roster["team"]]
    keep = {}
    if report is not None and len(report):
        for r in report.itertuples():
            st = str(getattr(r, "status", "")).lower()
            w = next((v for k, v in STATUS_KEEP.items() if k in st), None)
            if w is None:
                continue
            pid = str(getattr(r, "player_id", "") or "")
            keep[(r.team, pid)] = w
            keep[(r.team, str(getattr(r, "player", "")))] = w
    rows = []
    for u in upcoming.itertuples():
        for side in (u.home_team, u.away_team):
            for p in roster[roster["team"] == side].itertuples():
                w = keep.get((side, str(p.player_id)),
                             keep.get((side, str(p.player)), 1.0))
                if w > 0:
                    rows.append((str(u.game_id), side, str(p.player_id), w,
                                 p.player))
    return pd.DataFrame(rows, columns=cols)


def _match(games: pd.DataFrame, f) -> str | None:
    """ESPN game_id of a priced game (same teams, tip within a day)."""
    tip = pd.Timestamp(f.commence)
    c = games[(games["home_team"] == f.home_team)
              & (games["away_team"] == f.away_team)
              & ((pd.to_datetime(games["commence"], utc=True) - tip).abs()
                 <= pd.Timedelta(hours=30))]
    return None if c.empty else str(c["game_id"].iloc[0])


def live_lines(games: pd.DataFrame, box: pd.DataFrame, fair: dict,
               report: pd.DataFrame | None, now=None, seasons: int = 4,
               cfg: PredictConfig | None = None) -> dict[str, dict]:
    """Our own margin and total for every priced game tonight, from the
    model fitted on the last `seasons` seasons. No market input: the
    lines in `fair` are only used to find the games.

    {fair key: {"margin", "total", "own": True, "parts": {...}}}"""
    cfg = cfg or PredictConfig()
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    g = games[games["season_type"].isin(["REG", "PLAYIN", "POST"])].copy()
    g["game_id"] = g["game_id"].astype(str)
    last = int(g["season"].max())
    g = g[g["season"] > last - seasons].reset_index(drop=True)
    keys = {}
    for k, f in fair.items():
        gid = _match(g, f)
        if gid is not None and not bool(
                g.loc[g["game_id"] == gid, "completed"].iloc[0]):
            keys[k] = gid
    if not keys:
        return {}
    b = box[box["season"] > last - seasons] if box is not None else None
    up = g[g["game_id"].isin(set(keys.values()))]
    avail = tonight_availability(b, up, report)
    h = build(g, b, cfg, upcoming=avail, first_season=last - seasons + 2)
    models = fit_models(h)
    if not models.margin:
        return {}
    print(f"own model v2 fitted on {models.n} games")
    tonight = h[h["game_id"].isin(set(keys.values()))].set_index("game_id")
    p = predict_frame(models, tonight)
    out = {}
    for k, gid in keys.items():
        if gid not in p.index or not np.isfinite(p.at[gid, "pred_margin"]):
            continue
        r = tonight.loc[gid]
        out[k] = {"margin": float(p.at[gid, "pred_margin"]),
                  "total": float(p.at[gid, "pred_total"]), "own": True,
                  "parts": {"players": _r(r.get("apm_diff")),
                            "box": _r(r.get("box_diff")),
                            "possessions": _r(r.get("eff_margin")),
                            "scores": _r(r.get("model_margin"))},
                  "total_parts": {"possessions": _r(r.get("eff_total")),
                                  "pace": _r(r.get("pace_hat")),
                                  "scores": _r(r.get("model_total"))}}
    return out


def _r(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 1) if np.isfinite(x) else None
