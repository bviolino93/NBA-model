import numpy as np
import pandas as pd

import nba_predict as npr

TEAMS = sorted(npr.ARENAS)[:10]


def league(seasons=(2024, 2025, 2026), days=60, seed=0):
    """A fake league: 10 teams of 10 players with known impact. The star
    of team 0 (+8 per 48) sits every third game."""
    rng = np.random.default_rng(seed)
    true = {(t, j): rng.normal(0, 2) for t in range(10) for j in range(10)}
    true[(0, 0)] = 8.0
    games, box = [], []
    gid = 0
    for s in seasons:
        start = pd.Timestamp(f"{s - 1}-10-25", tz="UTC")
        for d in range(days):
            t = start + pd.Timedelta(days=d)
            order = rng.permutation(10)
            for k in range(5):
                h, a = order[2 * k], order[2 * k + 1]
                gid += 1
                lineups = {}
                for side in (h, a):
                    play = [j for j in range(10)
                            if not (side == 0 and j == 0 and gid % 3 == 0)]
                    mins = np.array([36 if j < 5 else 12 for j in play], float)
                    mins *= 240 / mins.sum()
                    lineups[side] = list(zip(play, mins))
                strength = {side: sum(m / 48 * true[(side, j)]
                                      for j, m in lineups[side])
                            for side in (h, a)}
                margin = 3 + strength[h] - strength[a] + rng.normal(0, 8)
                total = 220 + rng.normal(0, 12)
                hs, as_ = round((total + margin) / 2), round((total - margin) / 2)
                games.append(dict(
                    game_id=str(gid), season=s, season_type="REG",
                    game_date=(t - pd.Timedelta(hours=5)).date(), commence=t,
                    home_team=TEAMS[h], away_team=TEAMS[a], home_abbr=str(h),
                    away_abbr=str(a), home_score=hs, away_score=as_,
                    spread_line=3 + strength[h] - strength[a],
                    total_line=220.0, neutral=False, completed=True))
                for side, pts in ((h, hs), (a, as_)):
                    opp = a if side == h else h
                    for j, m in lineups[side]:
                        share = m / 240
                        box.append(dict(
                            game_id=str(gid), season=s,
                            game_date=games[-1]["game_date"], commence=t,
                            team=TEAMS[side], opp=TEAMS[opp],
                            home=int(side == h), player_id=f"{side}-{j}",
                            player=f"P{side}-{j}", starter=j < 5, played=True,
                            reason="", min=m, pts=pts * share,
                            reb=10 * share * 4, ast=5 * share * 4, fg3m=12 * share,
                            pm=0.0, fgm=40 * share, fga=88 * share,
                            fg3a=35 * share, ftm=18 * share, fta=23 * share,
                            oreb=10 * share, dreb=34 * share, stl=7 * share,
                            blk=5 * share, tov=13 * share, pf=19 * share))
    return pd.DataFrame(games), pd.DataFrame(box)


def test_team_box_possessions():
    g, b = league(seasons=(2026,), days=2)
    tb = npr.team_box(b)
    assert len(tb) == 2 * len(g)
    r = tb.iloc[0]
    assert abs(r["poss"] - (88 - 10 + 13 + 0.44 * 23)) < 1e-6
    assert abs(r["efg"] - (40 + 6) / 88) < 1e-6
    assert abs(r["orbr"] - 10 / 44) < 1e-6


def test_travel_and_altitude():
    g = pd.DataFrame({
        "game_id": ["1", "2"], "season": [2026, 2026],
        "commence": pd.to_datetime(["2025-11-01", "2025-11-03"], utc=True),
        "home_team": ["Boston Celtics", "Denver Nuggets"],
        "away_team": ["Miami Heat", "Boston Celtics"]})
    t = npr.travel_features(g)
    assert np.isnan(t.loc[0, "home_km"])
    # Boston -> Denver is about 2,800 km and two time zones west
    assert 2.6 < t.loc[1, "away_km"] < 3.0
    assert t.loc[1, "away_tz"] == -2
    assert t.loc[1, "alt_home"] == 1


def test_build_finds_the_star_and_his_absences():
    g, b = league()
    cfg = npr.PredictConfig(min_prior_games=100)
    cfg.ratings.min_games = 100
    h = npr.build(g, b, cfg, first_season=2025)
    last = h[(h["season"] == 2026) & h["apm_diff"].notna()]
    assert len(last) > 200
    # The star's games: team 0 is better with him than without
    t0 = TEAMS[0]
    side = np.where(last["home_team"] == t0, 1.0,
                    np.where(last["away_team"] == t0, -1.0, np.nan))
    k = np.isfinite(side)
    sits = last["game_id"].astype(int) % 3 == 0
    with_star = (side * last["apm_diff"])[k & ~sits].mean()
    without = (side * last["apm_diff"])[k & sits].mean()
    assert with_star - without > 1.5
    # The player line tracks the true strength (the fake "close")
    c = np.corrcoef(last["apm_diff"], last["spread_line"])[0, 1]
    assert c > 0.5
    m = npr.fit_models(h[h["season"] == 2025], trees=True)
    p = npr.predict_frame(m, last)
    assert p["pred_margin"].notna().all() and p["pred_total"].notna().all()


def test_tonight_availability_drops_out_players():
    g, b = league(seasons=(2026,), days=5)
    up = pd.DataFrame({"game_id": ["x"], "home_team": [TEAMS[0]],
                       "away_team": [TEAMS[1]]})
    rep = pd.DataFrame({"team": [TEAMS[0], TEAMS[1]],
                        "player": ["P0-0", "P1-1"], "player_id": ["0-0", "1-1"],
                        "status": ["Out", "Questionable"]})
    av = npr.tonight_availability(b, up, rep)
    assert "0-0" not in set(av["player_id"])
    assert av.set_index("player_id").loc["1-1", "weight"] == 0.6
    assert len(av) == 19


def test_live_lines_for_a_game_not_played_yet():
    from types import SimpleNamespace
    g, b = league()
    t = pd.Timestamp("2026-03-01", tz="UTC")
    g = pd.concat([g, pd.DataFrame([dict(
        game_id="new", season=2026, season_type="REG",
        game_date=(t - pd.Timedelta(hours=5)).date(), commence=t,
        home_team=TEAMS[0], away_team=TEAMS[1], home_abbr="0", away_abbr="1",
        home_score=np.nan, away_score=np.nan, spread_line=np.nan,
        total_line=np.nan, neutral=False, completed=False)])],
        ignore_index=True)
    fair = {"e1": SimpleNamespace(home_team=TEAMS[0], away_team=TEAMS[1],
                                  commence=t)}
    rep = pd.DataFrame({"team": [TEAMS[0]], "player": ["P0-0"],
                        "player_id": ["0-0"], "status": ["Out"]})
    cfg = npr.PredictConfig(min_prior_games=100)
    cfg.ratings.min_games = 100
    with_star = npr.live_lines(g, b, fair, None, cfg=cfg, seasons=3)
    without = npr.live_lines(g, b, fair, rep, cfg=cfg, seasons=3)
    assert with_star["e1"]["own"]
    assert with_star["e1"]["margin"] - without["e1"]["margin"] > 0.5
