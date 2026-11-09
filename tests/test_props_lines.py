import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import nba_props  # noqa: E402
import props_vs_lines as pvl  # noqa: E402


def test_names_and_nb():
    assert pvl.norm_name("Nikola Jokić") == "nikola jokic"
    assert pvl.norm_name("Jaren Jackson Jr.") == "jaren jackson"
    assert pvl.norm_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
    po, pp = pvl.nb_over(np.array([20.0, 20.0]), 1e6, [19.5, 20.0])
    # Huge size = Poisson: P(Y >= 20 | 20) ~ 0.53, P(Y == 20) ~ 0.089
    assert abs(po[0] - 0.5297) < 0.01 and pp[0] == 0
    assert abs(pp[1] - 0.0888) < 0.01 and abs(po[1] - (po[0] - pp[1])) < 1e-9
    rng = np.random.default_rng(0)
    mu = rng.uniform(5, 25, 5000)
    k = 8.0
    y = rng.negative_binomial(k, k / (k + mu))
    assert 5 < pvl.fit_k(mu, y) < 13


def _box(n_games=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(n_games):
        t = pd.Timestamp("2026-10-21T23:30", tz="UTC") + pd.Timedelta(days=2 * n)
        for pid, name, mpts in (("1", "Star Guard", 28), ("2", "Role Wing", 10)):
            rows.append(dict(game_id=f"g{n}", season=2027, game_date=t.date(),
                             commence=t, team="Atlanta Hawks", opp="Z", home=1,
                             team_abbr="ATL", reason="", starter=True,
                             played=True, pm=0.0, player_id=pid, player=name,
                             min=34.0 if pid == "1" else 22.0,
                             pts=rng.poisson(mpts), reb=rng.poisson(5),
                             ast=rng.poisson(4), fg3m=rng.poisson(2)))
    return pd.DataFrame(rows)


def _props(box):
    rows = []
    for r in box[box["game_id"].isin([f"g{n}" for n in range(10, 30)])].itertuples():
        tip = pd.Timestamp(r.commence)
        for snap in (tip - pd.Timedelta(hours=6), tip - pd.Timedelta(hours=1)):
            for side, price in (("Over", -110), ("Under", -110)):
                rows.append(dict(snapshot_utc=snap.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 event_id=r.game_id,
                                 commence_time=tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 home_team="Atlanta Hawks", away_team="Z",
                                 book="fanduel", market="player_points",
                                 outcome=side, player=r.player + ("" if r.player_id == "1" else " Jr."),
                                 point=24.5 if r.player_id == "1" else 13.5,
                                 price=price, last_update=""))
    return pd.DataFrame(rows)


def test_run_end_to_end():
    b = _box()
    proj = nba_props.project(b, None, None, half_life=8, prior_min=100,
                             use_absences=False)
    m, rep = pvl.run(_props(b), proj, {"pts": 50.0, "reb": 50.0, "ast": 50.0,
                                       "fg3m": 50.0})
    assert len(m) == 40                                  # 20 games x 2 players
    assert set(m["name"]) == {"star guard", "role wing"}
    assert m["p_model"].between(0, 1).all()
    # Star projects ~28 vs a 24.5 line: the model likes the over
    assert m.loc[m["name"] == "star guard", "p_model"].mean() > 0.6
    text = "\n".join(rep)
    assert "log loss" in text and "EV 3%+" in text
