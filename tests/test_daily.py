from datetime import datetime, timezone

import pandas as pd

import daily
import ledger
import picks
import selector as sel
from test_picks import games_frame
from test_selector import snap

NOW = datetime(2025, 10, 22, 18, tzinfo=timezone.utc)      # 5.5 h to tip
NAMES = picks.BOOK_NAME


def priced_fair(s=None):
    mm, tm = sel.build_outcome_models(games_frame())
    s = snap() if s is None else s
    fair = sel.consensus(s, mm, tm)
    return sel.price_offers(s, fair, mm, tm), fair


def choose(history=(), regular=(), lines=None, now=NOW, rows=None):
    pr, fair = priced_fair(rows)
    return daily.choose(list(history), list(regular), pr, fair, lines or {},
                        ["draftkings", "fanduel"], now, 8.0, NAMES)


def test_best_price_when_the_model_has_no_line():
    new = choose()
    assert len(new) == 1
    b = new[0]
    assert (b["pick"], b["point"], b["book"]) == ("Boston Celtics", -6.5,
                                                  "FanDuel")
    assert b["stake"] == daily.DAILY_STAKE and b["day"] == "2025-10-22"
    assert b["ev"] > 0 and not b["model_agrees"]


def test_model_lean_beats_a_better_price():
    # Market has Boston by about 7.5 and 224.5. FanDuel's Over is at -102
    # (about -1%, near fair); the model says Boston by 6 (likes
    # Philadelphia) and 230 (likes the Over), so the Over goes even though
    # FanDuel's stale Boston -6.5 is the better price.
    s = snap()
    s.loc[(s["book"] == "fanduel") & (s["outcome"] == "Over"), "price"] = -102
    new = choose(lines={"e1": {"margin": 6.0, "total": 230.0}}, rows=s)
    assert len(new) == 1
    assert (new[0]["pick"], new[0]["book"]) == ("Over", "FanDuel")
    assert new[0]["model_agrees"] and new[0]["lean"] > 5


def test_model_lean_does_not_pay_the_full_vig():
    # Every Over is -110 at a fair 224.5 (about -4.5%): the model's lean
    # isn't worth that, so the better price goes.
    new = choose(lines={"e1": {"margin": 6.0, "total": 230.0}})
    assert new[0]["pick"] == "Boston Celtics" and not new[0]["model_agrees"]


def two_games():
    a = snap()
    b = snap().assign(event_id="e2", home_team="New York Knicks",
                      away_team="Miami Heat")
    b["outcome"] = b["outcome"].replace({"Boston Celtics": "New York Knicks",
                                         "Philadelphia 76ers": "Miami Heat"})
    return pd.concat([a, b], ignore_index=True)


def test_up_to_three_a_day_one_per_game():
    new = choose(rows=two_games())
    assert len(new) == 2
    assert {b["home"] for b in new} == {"Boston Celtics", "New York Knicks"}
    pr, fair = priced_fair(two_games())
    one = daily.choose([], [], pr, fair, {}, ["fanduel"], NOW, 8.0, NAMES,
                       per_day=1)
    assert len(one) == 1


def test_later_runs_only_top_up():
    first = choose()                       # one game on the board
    more = choose(history=first, rows=two_games())
    assert [b["home"] for b in more] == ["New York Knicks"]


def test_one_pick_a_game_and_it_stays():
    first = choose()
    assert choose(history=first) == []


def test_not_on_a_market_already_bet():
    regular = [{"away": "Philadelphia 76ers", "home": "Boston Celtics",
                "commence": "2025-10-22T23:30:00Z", "market": "spreads",
                "pick": "Boston Celtics"}]
    new = choose(regular=regular)
    assert len(new) == 1 and new[0]["market"] == "totals"


def test_nothing_more_than_eight_hours_out():
    assert choose(now=datetime(2025, 10, 22, 12, tzinfo=timezone.utc)) == []


def test_daily_pick_grades_like_a_bet():
    b = choose()[0]
    games = pd.DataFrame([{"home_team": "Boston Celtics",
                           "away_team": "Philadelphia 76ers",
                           "game_date": "2025-10-22", "completed": True,
                           "home_score": 110, "away_score": 100}])
    g = ledger.grade([b], games)[0]
    assert g["result"] == "win" and g["profit"] > 0
    assert "Daily-pick record: 1-0-0" in daily.render_md([g])
    assert "Daily pick" in daily.render_alert([choose()[0]])
