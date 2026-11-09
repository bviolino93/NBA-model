import json

import pandas as pd
from datetime import datetime, timezone


import clv
import ledger
import log_lines
import nba_data
import paper
from test_selector import outcome_models, snap


def _logged(tmp_path, at="2025-10-22T23:00:00Z", name="close_2026.csv"):
    s = snap().assign(snapshot_utc=at, last_update=at, player="",
                      )[log_lines.FIELDS]
    s.to_csv(tmp_path / name, index=False)
    return clv.load_snapshots(str(tmp_path))


def _bet(**kw):
    b = {"away": "Philadelphia 76ers", "home": "Boston Celtics",
         "commence": "2025-10-22T23:30:00Z", "market": "spreads",
         "pick": "Boston Celtics", "point": -6.5, "price": -105,
         "stake": 5.0, "book": "FanDuel",
         "first_seen": "2025-10-22T18:00:00+00:00"}
    b.update(kw)
    return b


def test_bet_graded_against_sharp_close(tmp_path):
    mm, tm = outcome_models()
    snaps = _logged(tmp_path)
    after = datetime(2025, 10, 23, 1, tzinfo=timezone.utc)
    b = clv.grade([_bet(), _bet(market="totals", pick="Under", point=226.5,
                                price=-110)], snaps, mm, tm, after)
    sp, tot = b
    assert abs(sp["close_fair"] - 7.5) < 0.6
    assert abs(sp["clv_pts"] - 1.0) < 0.6 and sp["clv_ev"] > 0
    assert sp["close_min"] == 30
    assert tot["clv_pts"] > 0
    s = ledger.summary([dict(x, result="win", profit=1.0) for x in b])
    assert s["clv_n"] == 2 and s["beat_close"] == 1.0


def test_no_clv_before_tip_or_from_an_older_snapshot(tmp_path):
    mm, tm = outcome_models()
    snaps = _logged(tmp_path, at="2025-10-22T17:00:00Z")
    before = datetime(2025, 10, 22, 20, tzinfo=timezone.utc)
    after = datetime(2025, 10, 23, 1, tzinfo=timezone.utc)
    assert clv.grade([_bet()], snaps, mm, tm, before)[0].get("clv_ev") is None
    # The only snapshot predates the alert: nothing to compare against.
    assert clv.grade([_bet()], snaps, mm, tm, after)[0].get("clv_ev") is None


def test_in_play_rows_ignored(tmp_path):
    snaps = _logged(tmp_path, at="2025-10-23T00:00:00Z")
    assert snaps.empty


def test_model_test_takes_the_sharp_close(tmp_path):
    mm, tm = outcome_models()
    snaps = _logged(tmp_path)
    t = [{"market": "spreads", "home": "Boston Celtics",
          "away": "Philadelphia 76ers", "commence": "2025-10-22T23:30:00Z",
          "logged": "2025-10-22T18:00:00+00:00", "side": "Philadelphia 76ers",
          "point": 8.5, "line": -8.5, "close": None, "clv": None}]
    t = paper.grade_sharp(t, snaps, mm, tm,
                          datetime(2025, 10, 23, 1, tzinfo=timezone.utc))
    assert t[0]["clv_src"] == "sharp" and abs(t[0]["clv"] - 1.0) < 0.6


def test_pregame_filter():
    now = datetime(2025, 10, 22, 23, 45, tzinfo=timezone.utc)
    ev = [{"commence_time": "2025-10-22T23:30:00Z"},
          {"commence_time": "2025-10-23T02:00:00Z"}]
    assert log_lines.pregame(ev, now) == ev[1:]


def test_day_cached_for_good_only_when_final():
    def day(*states):
        return {"events": [{"competitions": [{"status": {"type": s}}]}
                           for s in states]}
    assert nba_data.day_final(day({"completed": True, "state": "post"}))
    assert nba_data.day_final(day({"completed": True},
                                  {"name": "STATUS_POSTPONED"}))
    assert not nba_data.day_final(day({"completed": True},
                                      {"state": "pre",
                                       "name": "STATUS_SCHEDULED"}))
    assert nba_data.day_final({"events": []})


def test_failed_fetch_not_cached(tmp_path, monkeypatch):
    class R:
        status_code = 503
    monkeypatch.setattr(nba_data.requests, "get", lambda *a, **k: R())
    monkeypatch.setattr(nba_data.time, "sleep", lambda s: None)
    assert nba_data.fetch_core_odds("1", str(tmp_path)) == {}
    assert not (tmp_path / "odds" / "1.json").exists()
    R.status_code = 404
    assert nba_data.fetch_core_odds("1", str(tmp_path)) == nba_data.NONE
    assert json.loads((tmp_path / "odds" / "1.json").read_text()) == nba_data.NONE
    # An old cached failure ({}) is fetched again
    (tmp_path / "odds" / "2.json").write_text("{}")
    assert nba_data.fetch_core_odds("2", str(tmp_path)) == nba_data.NONE


def test_replay_scores_flagged_offers(tmp_path):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import replay
    mm, tm = outcome_models()
    early = snap().assign(snapshot_utc="2025-10-22T18:00:00Z",
                          last_update="2025-10-22T18:00:00Z", player="")
    early[log_lines.FIELDS].to_csv(tmp_path / "games_2026.csv", index=False)
    _logged(tmp_path)                       # the close at 23:00
    r = replay.replay(clv.load_snapshots(str(tmp_path)), mm, tm)
    sharp = r[r["mode"] == "sharp"]
    assert len(sharp) == 1 and sharp.iloc[0]["clv_ev"] > 0
    assert (r["mode"] == "baseline").sum() == 8   # 2 books x 2 markets x 2 sides


def test_close_mode_spends_credits_only_on_tips_or_news(tmp_path, monkeypatch):
    import nba_injuries as inj
    from test_injuries import REPORT
    now = datetime(2025, 10, 22, 20, 0, tzinfo=timezone.utc)
    game = {"commence": pd.Timestamp("2025-10-22T23:30:00Z"),
            "home_team": "Atlanta Hawks", "away_team": "Boston Celtics"}
    calls = []
    monkeypatch.setattr(log_lines, "todays_games", lambda now: [game])
    monkeypatch.setattr(log_lines, "_get",
                        lambda url, p: calls.append(p) or [])
    report = {"js": REPORT}
    monkeypatch.setattr(inj, "fetch_injuries",
                        lambda: inj.parse_injuries(report["js"]))
    lines, injd = str(tmp_path / "lines"), str(tmp_path / "inj")
    log_lines.log_close("k", lines, 30, now, injd)       # first run: baseline
    assert calls == []
    log_lines.log_close("k", lines, 30, now, injd)       # no change
    assert calls == []
    js = json.loads(json.dumps(REPORT))
    js["injuries"][0]["injuries"][1]["status"] = "Out"   # Wing Two ruled out
    report["js"] = js
    log_lines.log_close("k", lines, 30, now, injd)       # news
    assert len(calls) == 1 and calls[0]["bookmakers"] == log_lines.CLOSE_BOOKS
    later = now + pd.Timedelta(minutes=10)
    log_lines.log_close("k", lines, 30, later, injd)     # follow-up
    assert len(calls) == 2
    log_lines.log_close("k", lines, 30, later + pd.Timedelta(minutes=10), injd)
    assert len(calls) == 2                               # quiet again
    log_lines.log_close("k", lines, 30, datetime(2025, 10, 22, 23, 10,
                                                 tzinfo=timezone.utc), injd)
    assert len(calls) == 3                               # tip in 20 minutes
