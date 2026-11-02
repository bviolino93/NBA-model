import json
import os

import numpy as np
import pandas as pd

import nba_data

FIX = os.path.join(os.path.dirname(__file__), "fixtures",
                   "scoreboard_20251022.json")


def rows():
    with open(FIX) as fh:
        return pd.DataFrame(nba_data.parse_scoreboard(json.load(fh)))


def test_parse_home_spread_sign_and_names():
    d = rows().set_index("game_id")
    # Home favourite: BOS -7.5 -> expected home margin +7.5
    assert d.loc["401809001", "spread_line"] == 7.5
    # Away favourite: DEN -2 at LAC -> home margin -2
    assert d.loc["401809002", "spread_line"] == -2.0
    assert d.loc["401809003", "spread_line"] == 0.0
    assert d.loc["401809002", "home_team"] == "Los Angeles Clippers"
    assert d.loc["401809001", "total_line"] == 224.5


def test_scores_only_when_completed_and_dates_are_us():
    d = rows().set_index("game_id")
    assert d.loc["401809001", "home_score"] == 117
    assert np.isnan(d.loc["401809003", "home_score"])
    # 02:00 UTC tip is still the 22nd in the US.
    assert str(d.loc["401809002", "game_date"]) == "2025-10-22"
    assert d.loc["401809001", "season"] == 2026


def test_current_season():
    from datetime import date
    assert nba_data.current_season(date(2026, 11, 2)) == 2027
    assert nba_data.current_season(date(2027, 4, 1)) == 2027


def _sched():
    days = ["2025-10-22", "2025-10-23", "2025-10-25", "2025-10-26"]
    g = pd.DataFrame({
        "game_id": ["1", "2", "3", "4"],
        "season": 2026,
        "game_date": pd.to_datetime(days).date,
        "commence": pd.to_datetime(days, utc=True),
        "home_team": ["A", "C", "A", "B"],
        "away_team": ["B", "A", "C", "A"],
    })
    return g


def test_rest_and_back_to_backs():
    f = nba_data.schedule_features(_sched()).set_index("game_id")
    # A plays 22nd (home), 23rd (away at C): road game on a b2b
    assert f.loc["2", "away_b2b"] == 1
    assert f.loc["2", "away_rest"] == 1
    # A then plays 25th at home (rest 2) and 26th away at B (b2b, but the
    # first night was at home)
    assert f.loc["3", "home_rest"] == 2
    assert f.loc["4", "away_b2b"] == 1
    assert f.loc["4", "away_b2b_road"] == 0
    assert f.loc["2", "away_b2b_road"] == 0  # A was home on the 22nd
    # First game of the season has unknown rest
    assert np.isnan(f.loc["1", "home_rest"])
    # Games in the previous 5 nights for A before the 26th: 22, 23, 25
    assert f.loc["4", "away_last5"] == 3


def test_merge_lines_prefers_archive(tmp_path):
    g = rows()
    p = tmp_path / "lines.csv"
    pd.DataFrame([{"game_date": "2025-10-22", "home_team": "Boston Celtics",
                   "away_team": "Philadelphia 76ers", "spread_line": 8.0,
                   "total_line": 225.0}]).to_csv(p, index=False)
    m = nba_data.merge_lines(g, nba_data.load_lines(str(p))).set_index("game_id")
    assert m.loc["401809001", "spread_line"] == 8.0
    assert m.loc["401809002", "spread_line"] == -2.0


def test_core_odds_median_skips_projections_and_live():
    fix = os.path.join(os.path.dirname(__file__), "fixtures",
                       "core_odds_401070894.json")
    with open(fix) as fh:
        sp, tt, n = nba_data.parse_core_odds(json.load(fh), "HOU", "MIA")
    assert n == 3
    assert sp == 4.5          # HOU home favourite by 4.5 (median of 4.5, 4.5, 5)
    assert tt == 217.0


def test_core_odds_empty():
    sp, tt, n = nba_data.parse_core_odds({}, "HOU", "MIA")
    assert np.isnan(sp) and np.isnan(tt) and n == 0


def test_core_open_lines():
    fix = os.path.join(os.path.dirname(__file__), "fixtures",
                       "core_odds_401070894.json")
    with open(fix) as fh:
        sp, tt = nba_data.parse_core_open(json.load(fh))
    assert sp == 5.5            # median of home -5 and -6 at the open
    assert tt == 221.5
