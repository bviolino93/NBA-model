"""
Do our prop projections know anything the prop market doesn't?

nba_props.py was only scored by average error against naive baselines. A
prop bet needs the chance of going over a line, and the test that matters
is against the lines themselves. For every prop line logged before tip
(data/lines/props_*.csv, two-way Over/Under at the same number):

  market   the book's de-vigged chance of the over (pushes aside)
  model    the chance from our projection (nba_props.project, minutes x
           rate x pace, built from earlier games only, no knowledge of who
           sat) with a negative binomial spread around it; the spread per
           stat is fitted on earlier seasons

Reported, on players who played (a prop is void otherwise):

  1. log loss and calibration of market, model and a blend, at each
     book's last pre-tip line
  2. a logistic fit of the result on the market's logit and the model's
     difference from it: a weight on the difference that is clearly above
     zero means the model adds information the price lacks
  3. betting the model's side at the FIRST logged price when its EV is
     3%+, 6%+, 10%+: record, ROI at the real prices, and CLV against the
     same book's closing price at the same number
  4. the same by stat and by book

    python tools/props_vs_lines.py [--min-prev 5]
"""
import argparse
import glob
import math
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selector as sel  # noqa: E402

LINES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "lines")
MARKET_STAT = {"player_points": "pts", "player_rebounds": "reb",
               "player_assists": "ast", "player_threes": "fg3m"}
EV_CUTS = (0.03, 0.06, 0.10)
YMAX = 80


# ----------------------------------------------------------------------
# Lines
# ----------------------------------------------------------------------
def norm_name(s) -> str:
    """'Nikola Jokić' -> 'nikola jokic'; drops Jr., III and punctuation."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    return " ".join(w for w in s.split() if w not in ("jr", "sr", "ii", "iii",
                                                       "iv", "v"))


def prop_pairs(props: pd.DataFrame) -> pd.DataFrame:
    """One row per pre-tip snapshot x book x player x market x number with
    both sides priced: over/under prices and the de-vigged over chance."""
    d = props[props["market"].isin(MARKET_STAT)].copy()
    d["_tip"] = pd.to_datetime(d["commence_time"], utc=True, errors="coerce")
    d["_at"] = pd.to_datetime(d["snapshot_utc"], utc=True, errors="coerce")
    d = d[d["_at"] < d["_tip"]]
    d["side"] = d["outcome"].str.lower()
    key = ["snapshot_utc", "event_id", "book", "market", "player", "point"]
    w = (d[d["side"].isin(["over", "under"])]
         .pivot_table(index=key, columns="side", values="price", aggfunc="first")
         .dropna().reset_index())
    meta = d.drop_duplicates(key)[key + ["home_team", "away_team",
                                         "commence_time", "_tip", "_at"]]
    w = w.merge(meta, on=key)
    po = w["over"].map(sel.american_to_prob)
    pu = w["under"].map(sel.american_to_prob)
    w["p_mkt"] = po / (po + pu)
    w["stat"] = w["market"].map(MARKET_STAT)
    w["name"] = w["player"].map(norm_name)
    return w


def first_and_last(w: pd.DataFrame) -> pd.DataFrame:
    """Per book x player x market x number: the first and last pre-tip
    snapshot (open_/close_ prices and chances)."""
    key = ["event_id", "book", "market", "player", "point"]
    w = w.sort_values("_at")
    first = w.drop_duplicates(key, keep="first")
    last = w.drop_duplicates(key, keep="last")[key + ["over", "under", "p_mkt",
                                                      "_at"]]
    return first.merge(last, on=key, suffixes=("", "_close"))


# ----------------------------------------------------------------------
# The model's chance of the over
# ----------------------------------------------------------------------
def nb_cdf_table(mu: np.ndarray, k: float) -> np.ndarray:
    """P(Y <= y) for y = 0..YMAX, one row per mean (negative binomial with
    mean mu and size k; variance mu + mu^2 / k)."""
    mu = np.clip(np.asarray(mu, float), 1e-6, None)[:, None]
    y = np.arange(YMAX + 1)[None, :]
    lg = np.vectorize(math.lgamma)
    logp = (lg(y + k) - math.lgamma(k) - lg(y + 1)
            + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))
    return np.cumsum(np.exp(logp), axis=1)


def nb_over(mu, k: float, line) -> tuple[np.ndarray, np.ndarray]:
    """(P(over), P(push)) at each line."""
    cdf = nb_cdf_table(mu, k)
    line = np.asarray(line, float)
    fl = np.clip(np.floor(line).astype(int), 0, YMAX)
    rows = np.arange(len(fl))
    below = cdf[rows, fl]                      # P(Y <= floor(line))
    whole = np.isclose(line, np.floor(line))
    at = np.where(whole, below - np.where(fl > 0, cdf[rows, fl - 1], 0.0), 0.0)
    return 1.0 - below, at


def fit_k(mu: np.ndarray, y: np.ndarray) -> float:
    """Negative binomial size by likelihood over a grid."""
    m = np.isfinite(mu) & np.isfinite(y) & (mu > 0)
    mu, y = np.asarray(mu, float)[m], np.asarray(y, float)[m].round()
    if len(mu) > 20000:
        idx = np.random.default_rng(0).choice(len(mu), 20000, replace=False)
        mu, y = mu[idx], y[idx]
    lg = np.vectorize(math.lgamma)
    best, ll_best = 50.0, -np.inf
    for k in np.exp(np.linspace(np.log(0.5), np.log(500), 40)):
        ll = np.sum(lg(y + k) - math.lgamma(k) - lg(y + 1)
                    + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))
        if ll > ll_best:
            best, ll_best = float(k), ll
    return best


# ----------------------------------------------------------------------
# Matching lines to projections and results
# ----------------------------------------------------------------------
def attach(lines: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Join each line to the player's projection and result for that game
    (name, team in the game, US date of tip)."""
    p = proj.copy()
    p["name"] = p["player"].map(norm_name)
    p["day"] = pd.to_datetime(p["game_date"]).dt.date
    lines = lines.copy()
    lines["day"] = (lines["_tip"].dt.tz_convert("America/New_York")
                    .dt.tz_localize(None).dt.date)
    cols = ["name", "day", "team", "n_prev", "base_min"] + [
        c for s in MARKET_STAT.values() for c in (s, f"proj_{s}")]
    m = lines.merge(p[cols], on=["name", "day"], how="inner")
    m = m[(m["team"] == m["home_team"]) | (m["team"] == m["away_team"])]
    m["y"] = [r[s] for r, s in zip(m.to_dict("records"), m["stat"])]
    m["mu"] = [r[f"proj_{s}"] for r, s in zip(m.to_dict("records"), m["stat"])]
    return m


def add_model(m: pd.DataFrame, ks: dict) -> pd.DataFrame:
    m = m.copy()
    m["p_over"], m["p_push"] = np.nan, np.nan
    for s, k in ks.items():
        i = m.index[(m["stat"] == s) & m["mu"].notna()]
        if len(i):
            po, pp = nb_over(m.loc[i, "mu"].to_numpy(), k, m.loc[i, "point"])
            m.loc[i, "p_over"], m.loc[i, "p_push"] = po, pp
    m["p_model"] = m["p_over"] / (1.0 - m["p_push"]).clip(lower=1e-9)
    m["result"] = np.sign(m["y"] - m["point"])          # +1 over, -1 under
    return m


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logloss(p, o) -> float:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def logistic(X: np.ndarray, y: np.ndarray, iters: int = 50):
    """Coefficients and z-scores (intercept first) by Newton's method."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, b / se


def information_test(m: pd.DataFrame) -> list[str]:
    """At the closing line: does the model add to the market's price?"""
    c = m[(m["result"] != 0) & m["p_model"].notna()]
    o = (c["result"] > 0).astype(float).to_numpy()
    lm, lmod = _logit(c["p_mkt_close"]), _logit(c["p_model"])
    out = [f"  {len(c)} closing lines with a result (pushes left out)",
           f"  log loss: market {logloss(c['p_mkt_close'], o):.4f}, model "
           f"{logloss(c['p_model'], o):.4f}, coin flip 0.6931"]
    if len(c) >= 50:
        b, z = logistic(np.column_stack([lm, lmod - lm]), o)
        out.append(f"  result ~ market logit {b[1]:+.2f} (z {z[1]:+.1f}) + "
                   f"(model - market) {b[2]:+.2f} (z {z[2]:+.1f})")
        out.append("  a weight on (model - market) clearly above 0 (z 2+) "
                   "means the model adds information")
        bl = 1 / (1 + np.exp(-(b[0] + b[1] * lm + b[2] * (lmod - lm))))
        out.append(f"  blend log loss {logloss(bl, o):.4f} (in-sample, "
                   f"an upper bound on the gain)")
    return out


def betting(m: pd.DataFrame, cuts=EV_CUTS) -> list[str]:
    """Bet the model's side at the first logged price when EV clears cut;
    graded at that price, with CLV against the same book's close."""
    d = m[m["p_model"].notna()].copy()
    po, pp = d["p_over"], d["p_push"]
    pu = 1 - po - pp
    ev_o = po * d["over"].map(sel.payout) - pu
    ev_u = pu * d["under"].map(sel.payout) - po
    d["bet_over"] = ev_o >= ev_u
    d["ev"] = np.where(d["bet_over"], ev_o, ev_u)
    d["price"] = np.where(d["bet_over"], d["over"], d["under"])
    sign = np.where(d["bet_over"], 1, -1)
    win = d["result"] * sign
    d["profit"] = np.where(win > 0, d["price"].map(sel.payout),
                           np.where(win < 0, -1.0, 0.0))
    # CLV: the closing de-vigged chance of our side, paid at our price
    pc = np.where(d["bet_over"], d["p_mkt_close"], 1 - d["p_mkt_close"])
    d["clv_ev"] = pc * d["price"].map(sel.payout) - (1 - pc)
    d["moved"] = d["_at_close"] > d["_at"]
    out = []
    for cut in cuts:
        b = d[d["ev"] >= cut]
        if b.empty:
            continue
        w, l = int((b["profit"] > 0).sum()), int((b["profit"] < 0).sum())
        mv = b[b["moved"]]
        out.append(f"  EV {cut:.0%}+: {len(b)} bets, {w}-{l}, ROI "
                   f"{b['profit'].mean():+.1%} (se "
                   f"{b['profit'].std(ddof=1) / max(1, np.sqrt(len(b))):.1%}); "
                   f"CLV on {len(mv)} with a later close: "
                   f"{mv['clv_ev'].mean():+.1%}, beat close "
                   f"{(mv['clv_ev'] > 0).mean():.0%}"
                   + ("" if len(mv) else " (none)"))
        for by in ("stat", "book"):
            for g, x in b.groupby(by):
                if len(x) >= 20:
                    mx = x[x["moved"]]
                    out.append(f"      {by} {g:14s} {len(x):5d} bets, ROI "
                               f"{x['profit'].mean():+.1%}, CLV "
                               f"{mx['clv_ev'].mean():+.1%}" if len(mx)
                               else f"      {by} {g:14s} {len(x):5d} bets, "
                               f"ROI {x['profit'].mean():+.1%}")
    return out or ["  no bets clear the cuts"]


def walk_forward(m: pd.DataFrame, cuts=(0.0, 0.02, 0.04, 0.06), min_train: int = 1500
                 ) -> list[str]:
    """The honest version of 3: the raw model overstates its edge, so shrink
    it with a blend (result ~ market logit + (model - market)) fitted ONLY on
    earlier days, then bet later days at the first logged price when the
    blended EV clears the cut. Expanding window, refit each day."""
    d = m[m["p_model"].notna() & (m["result"] != 0)].copy()
    d["day"] = d["_tip"].dt.tz_convert("America/New_York").dt.date
    d["lm"] = _logit(d["p_mkt"])
    d["dm"] = _logit(d["p_model"]) - d["lm"]
    d["o"] = (d["result"] > 0).astype(float)
    days = sorted(d["day"].unique())
    rows = []
    for day in days:
        tr, te = d[d["day"] < day], d[d["day"] == day]
        if len(tr) < min_train or te.empty:
            continue
        b, _ = logistic(tr[["lm", "dm"]].to_numpy(), tr["o"].to_numpy())
        pb = 1 / (1 + np.exp(-(b[0] + b[1] * te["lm"] + b[2] * te["dm"])))
        ev_o = pb * te["over"].map(sel.payout) - (1 - pb)
        ev_u = (1 - pb) * te["under"].map(sel.payout) - pb
        t = te.assign(ev=np.maximum(ev_o, ev_u), bet_over=ev_o >= ev_u, w_model=b[2])
        rows.append(t)
    if not rows:
        return [f"  not enough days yet (need {min_train} earlier lines)"]
    t = pd.concat(rows)
    price = np.where(t["bet_over"], t["over"], t["under"])
    win = t["result"] * np.where(t["bet_over"], 1, -1)
    t["profit"] = np.where(win > 0, pd.Series(price, index=t.index).map(sel.payout), -1.0)
    out = [f"  {t['day'].nunique()} days bet ({t['day'].min()} .. {t['day'].max()}), "
           f"blend weight on model gap: first {t['w_model'].iloc[0]:+.2f}, "
           f"last {t['w_model'].iloc[-1]:+.2f}"]
    for cut in cuts:
        b = t[t["ev"] > cut]
        if b.empty:
            continue
        se = b["profit"].std(ddof=1) / max(1, np.sqrt(len(b)))
        per = b.groupby("day").size().mean()
        out.append(f"  blended EV > {cut:.0%}: {len(b)} bets ({per:.0f}/day), win "
                   f"{(b['profit'] > 0).mean():.1%}, ROI {b['profit'].mean():+.1%} (se {se:.1%})")
        for g, x in b.groupby("stat"):
            out.append(f"      {g:5s} {len(x):5d} bets ROI {x['profit'].mean():+.1%}")
        if cut == cuts[-1]:
            for bk in ("draftkings", "fanduel"):
                x = b[b["book"] == bk]
                if len(x):
                    out.append(f"      at {bk}: {len(x)} bets ROI {x['profit'].mean():+.1%}")
    return out


def calibration(m: pd.DataFrame) -> list[str]:
    c = m[(m["result"] != 0) & m["p_model"].notna()]
    o = (c["result"] > 0).astype(float)
    out = ["  chance of the over   market: n, actual   model: n, actual"]
    bins = [0, .3, .4, .45, .5, .55, .6, .7, 1]
    mk = pd.cut(c["p_mkt_close"], bins)
    md = pd.cut(c["p_model"], bins)
    a, b = o.groupby(mk, observed=False), o.groupby(md, observed=False)
    for iv in mk.cat.categories:
        out.append(f"  {str(iv):12s} {a.size().get(iv, 0):7d} "
                   f"{a.mean().get(iv, np.nan):6.1%}   "
                   f"{b.size().get(iv, 0):7d} {b.mean().get(iv, np.nan):6.1%}")
    return out


def run(props: pd.DataFrame, proj: pd.DataFrame, ks: dict,
        min_prev: int = 5) -> tuple[pd.DataFrame, list[str]]:
    w = prop_pairs(props)
    lines = first_and_last(w)
    m = attach(lines, proj)
    m = m[m["n_prev"] >= min_prev]
    m = add_model(m, ks)
    rep = [f"{lines['player'].nunique()} players, {len(lines)} lines; "
           f"{len(m)} matched to a box score with {min_prev}+ earlier games "
           f"(players who sat drop out: props are void)",
           "NB size per stat: " + ", ".join(f"{s} {k:.1f}"
                                             for s, k in ks.items()),
           "", "1-2. Information at the close"]
    rep += information_test(m)
    for s in MARKET_STAT.values():
        x = m[m["stat"] == s]
        if len(x) >= 50:
            rep.append(f"   {s}:")
            rep += ["  " + r for r in information_test(x)[1:3]]
    rep += ["", "Calibration at the close"] + calibration(m)
    rep += ["", "3-4. Betting the model's side at the first logged price"]
    rep += betting(m)
    rep += ["", "5. Walk-forward: model shrunk by a blend fitted on earlier days only"]
    rep += walk_forward(m)
    return m, rep


def main():
    from nba_data import current_season, load_games
    from nba_players import load_box
    import nba_props
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prev", type=int, default=5)
    ap.add_argument("--half-life", type=float, default=8.0)
    ap.add_argument("--prior-min", type=float, default=100.0)
    ap.add_argument("--out", default="")
    a, _ = ap.parse_known_args()   # the Research workflow passes its default args too
    files = sorted(glob.glob(os.path.join(LINES_DIR, "props_*.csv")))
    props = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    season = current_season()
    g = load_games(season - 2)
    box = load_box(g)
    proj = nba_props.project(box, None, g, half_life=a.half_life,
                             prior_min=a.prior_min, use_absences=False)
    # Spread per stat from earlier seasons only
    past = proj[(proj["season"] < season) & (proj["n_prev"] >= a.min_prev)]
    ks = {s: fit_k(past[f"proj_{s}"].to_numpy(), past[s].to_numpy())
          for s in MARKET_STAT.values()}
    m, rep = run(props, proj[proj["season"] == season], ks, a.min_prev)
    print("\n".join(rep))
    if a.out:
        m.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
