from datetime import datetime, timezone

import pandas as pd

import ledger
import picks
from test_selector import snap


def games_frame():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 6000
    spread = rng.choice(np.arange(-14, 14.5, 0.5), n)
    total = rng.choice(np.arange(205, 245.5, 0.5), n)
    margin = np.round(spread + rng.normal(0, 12, n))
    points = np.round(total + rng.normal(0, 18, n))
    return pd.DataFrame({"season": 2024, "spread_line": spread,
                         "total_line": total,
                         "home_score": np.round((points + margin) / 2),
                         "away_score": np.round((points - margin) / 2)})


def test_stale_fanduel_line_is_picked_with_half_stake():
    rows = snap()
    now = datetime(2025, 10, 22, 18, tzinfo=timezone.utc)
    p, fair, priced = picks.build_picks(rows, games_frame(), ["fanduel"],
                                        1000.0, 0.02, now)
    assert len(p) == 1
    b = p.iloc[0]
    assert b["book"] == "FanDuel" and b["pick"] == "Boston Celtics"
    assert b["stake"] <= 5.0                 # 1% cap, halved
    assert "FanDuel has Boston Celtics at -6.5" in b["why"]


def test_no_alerts_more_than_eight_hours_out():
    early = datetime(2025, 10, 22, 12, tzinfo=timezone.utc)   # 11.5 h to tip
    p, fair, priced = picks.build_picks(snap(), games_frame(), ["fanduel"],
                                        1000.0, 0.02, early)
    assert p.empty and len(fair) == 1


def test_no_sharp_book_no_bet():
    rows = snap()
    rows = rows[~rows["book"].isin(["pinnacle", "circasports"])]
    rows = pd.concat([rows, rows.assign(book="betmgm")])
    now = datetime(2025, 10, 22, 18, tzinfo=timezone.utc)
    p, fair, priced = picks.build_picks(rows, games_frame(), ["fanduel"],
                                        1000.0, 0.02, now)
    assert p.empty and fair["e1"].sharp_margin == 0


def test_games_already_tipped_are_skipped():
    rows = snap()
    late = datetime(2025, 10, 23, 0, tzinfo=timezone.utc)
    p, fair, priced = picks.build_picks(rows, games_frame(), ["fanduel"],
                                        1000.0, 0.02, late)
    assert p.empty


def test_ledger_grades_from_espn_finals():
    bet = {"away": "Philadelphia 76ers", "home": "Boston Celtics",
           "commence": "2025-10-22T23:30:00+00:00", "market": "spreads",
           "pick": "Boston Celtics", "point": -6.5, "price": -105,
           "stake": 5.0}
    g = pd.DataFrame([{"home_team": "Boston Celtics",
                       "away_team": "Philadelphia 76ers",
                       "game_date": pd.Timestamp("2025-10-22").date(),
                       "home_score": 117.0, "away_score": 105.0,
                       "completed": True}])
    out = ledger.grade([dict(bet)], g)
    assert out[0]["result"] == "win" and out[0]["final"] == "105-117"
    assert out[0]["profit"] == round(5.0 * 100 / 105, 2)
