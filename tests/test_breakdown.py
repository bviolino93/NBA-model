import pandas as pd

from tools import breakdown as bd


def _games():
    rows = []
    t = pd.Timestamp("2026-11-01T00:00Z")
    # Heat at home 4 times, winning by 10 vs a line of -5 (home margin +5)
    for i in range(4):
        rows.append(dict(game_id=str(i), commence=t + pd.Timedelta(days=i),
                         home_team="Miami Heat", away_team=f"T{i}",
                         home_score=110, away_score=100, spread_line=5.0,
                         total_line=200.0, completed=True))
    # one road loss by 3, line Heat +1 (home margin +1), total 230 vs 220
    rows.append(dict(game_id="9", commence=t + pd.Timedelta(days=5),
                     home_team="X", away_team="Miami Heat", home_score=116,
                     away_score=113, spread_line=1.0, total_line=220.0,
                     completed=True))
    rows.append(dict(game_id="10", commence=t + pd.Timedelta(days=9),
                     home_team="Miami Heat", away_team="Y", home_score=None,
                     away_score=None, spread_line=3.0, total_line=220.0,
                     completed=False))
    return pd.DataFrame(rows)


def test_team_form():
    f = bd.team_form(_games(), "Miami Heat", "2026-11-09T00:00Z")
    assert f["n"] == 5 and f["w"] == 4 and f["l"] == 1
    assert f["cover"] == 4 and f["no_cover"] == 1   # road: lost by 3, line +1
    assert f["over"] == 5 and f["under"] == 0
    assert f["streak"] == "L1"
    assert f["margin"] == round((40 - 3) / 5, 1)


def test_team_form_before_cutoff():
    f = bd.team_form(_games(), "Miami Heat", "2026-11-02T12:00Z")
    assert f["n"] == 2


def test_line_history():
    rows = []
    for snap, sp, tot in (("2026-11-18T01:00Z", -12.5, 236.5),
                          ("2026-11-18T13:00Z", -10.5, 234.0)):
        for book in ("draftkings", "fanduel"):
            rows.append(dict(snapshot_utc=snap, commence_time="2026-11-19T00:10Z",
                             home_team="Miami Heat", away_team="Milwaukee Bucks",
                             book=book, market="spreads", outcome="Miami Heat",
                             point=sp))
            rows.append(dict(snapshot_utc=snap, commence_time="2026-11-19T00:10Z",
                             home_team="Miami Heat", away_team="Milwaukee Bucks",
                             book=book, market="totals", outcome="Over",
                             point=tot))
    # after tip: ignored
    rows.append(dict(snapshot_utc="2026-11-19T01:00Z",
                     commence_time="2026-11-19T00:10Z", home_team="Miami Heat",
                     away_team="Milwaukee Bucks", book="draftkings",
                     market="spreads", outcome="Miami Heat", point=-20))
    lh = bd.line_history(pd.DataFrame(rows), "Miami Heat", "Milwaukee Bucks",
                         "2026-11-19T00:10Z")
    assert lh["bet_spread_open"] == -12.5 and lh["bet_spread_now"] == -10.5
    assert lh["bet_total_now"] == 234.0 and lh["snapshots"] == 2
    assert "sharp_spread_now" not in lh


def _b():
    return {"home": "Miami Heat", "away": "Milwaukee Bucks",
            "home_short": "Heat", "away_short": "Bucks", "tip_et": "7:10 PM",
            "line": {"spread": -10.5, "total": 234.0, "source": "DK/FD",
                     "bet_spread_open": -12.5, "bet_spread_now": -10.5,
                     "bet_total_open": 236.5, "bet_total_now": 234.0,
                     "snapshots": 2},
            "projection": {"margin": 6.0, "total": 240.0, "home_pts": 123.0,
                           "away_pts": 117.0, "parts": {}},
            "matchup": {"pace": 101.0, "home_ortg": 118.0, "away_ortg": 112.0,
                        "four_factors": {"efg": 0.02}, "apm_diff": 3.0,
                        "box_diff": 1.0},
            "form": {"home": {"n": 0}, "away": {"n": 0}},
            "schedule": {"away_b2b": 1.0, "away_prev_away": 1.0,
                         "rest_diff": 1.0},
            "injuries": {"home": [{"player": "Bam Adebayo", "status": "Day-To-Day",
                                   "mpg": 34.0, "detail": "Hip"}], "away": []}}


def test_notes():
    n = bd.notes(_b())
    assert "Bucks on a back-to-back after a road game" in n
    assert any("Adebayo" in x for x in n)
    assert any(x.startswith("Spread moved 2.0 toward Bucks") for x in n)
    assert any("4.5 pts better for Bucks" in x for x in n)
    assert any("Our total is 6.0 over" in x for x in n)


def test_markdown():
    b = _b()
    b["notes"] = bd.notes(b)
    md = bd.to_markdown("2026-11-18", [b])
    assert "## Milwaukee Bucks @ Miami Heat" in md
    assert "Heat 123, Bucks 117" in md
