import numpy as np
import pandas as pd

import nba_players as npl


def box_rows():
    """Team A, 8 games. Star S plays 36 min (+5 a game) through game 5,
    then sits games 6-8 with no box row. Bench B plays 20 every game."""
    rows = []
    for n in range(8):
        gid = f"g{n}"
        t = pd.Timestamp("2025-10-21", tz="UTC") + pd.Timedelta(days=2 * n)
        common = dict(game_id=gid, season=2026, game_date=t.date(),
                      commence=t, team="A", opp="Z", home=1, team_abbr="A",
                      reason="", reb=0, ast=0, fg3m=0)
        if n < 5:
            rows.append(dict(common, player_id="S", player="Star", starter=True,
                             played=True, min=36.0, pts=25, pm=5.0))
        rows.append(dict(common, player_id="B", player="Bench", starter=False,
                         played=True, min=20.0, pts=8, pm=-2.0))
    return pd.DataFrame(rows)


def test_missing_star_found_without_a_box_row():
    ab = npl.absences(box_rows()).set_index("game_id")
    assert "g4" not in ab.index            # he played
    assert ab.loc["g5", "missing"] == "Star"
    assert ab.loc["g5", "lost_min"] == 36.0
    assert ab.loc["g5", "lost_star"] == 1
    # Value: 36/36 * pm36 + 36/48, pm36 = 36*25/(180+300) = 1.875
    assert abs(ab.loc["g5", "lost_val"] - (1.875 + 0.75)) < 1e-6
    # Still missing two games later (played within the last 5 team games)
    assert "g7" in ab.index
    assert ab.loc["g7", "lost_min"] < 36.0   # rolling mean decays


def test_parse_box_from_espn_shape():
    js = {"boxscore": {"players": [{"team": {"abbreviation": "HOU"},
          "statistics": [{"labels": ["MIN", "PTS", "FG", "3PT", "FT", "REB",
                                     "AST", "TO", "STL", "BLK", "OREB", "DREB",
                                     "PF", "+/-"],
          "athletes": [
            {"athlete": {"id": "1", "displayName": "Dillon Brooks"},
             "starter": True, "didNotPlay": False,
             "stats": ["31", "18", "7-9", "2-3", "2-4", "6", "0", "0", "2",
                       "0", "2", "4", "2", "-6"]},
            {"athlete": {"id": "2", "displayName": "Boban Marjanovic"},
             "starter": False, "didNotPlay": True,
             "reason": "COACH'S DECISION", "stats": []}]}]}]}}
    rows = npl.parse_box(js, "g1")
    a, b = rows
    assert a["min"] == 31 and a["pts"] == 18 and a["fg3m"] == 2
    assert a["reb"] == 6 and a["pm"] == -6 and a["played"]
    assert not b["played"] and b["min"] == 0


def test_game_absences_home_minus_away():
    g = pd.DataFrame({"game_id": ["g5", "g6"], "home_team": ["A", "Z"],
                      "away_team": ["Z", "A"]})
    ab = pd.DataFrame({"game_id": ["g5", "g6"], "team": ["A", "A"],
                       "lost_min": [36.0, 30.0], "lost_val": [2.0, 1.5],
                       "lost_star": [1, 1], "missing": ["Star", "Star"]})
    f = npl.game_absences(g, ab).set_index("game_id")
    assert f.loc["g5", "inj_val_diff"] == 2.0      # home A depleted
    assert f.loc["g6", "inj_val_diff"] == -1.5     # away A depleted
    assert f.loc["g6", "away_missing"] == "Star"
    assert f.loc["g5", "inj_val_sum"] == 2.0
