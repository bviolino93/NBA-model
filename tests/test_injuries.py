import pandas as pd

import nba_injuries as inj
from picks import render_alert, render_md

REPORT = {"injuries": [
    {"displayName": "Atlanta Hawks", "injuries": [
        {"status": "Out", "athlete": {"id": "1", "displayName": "Star Guard"},
         "details": {"type": "Knee"}},
        {"status": "Day-To-Day", "athlete": {"id": "2", "displayName": "Wing Two"},
         "type": {"description": "day-to-day"}},
        {"status": "Questionable", "athlete": {"id": "3", "displayName": "Bench Guy"}},
    ]},
    {"displayName": "LA Clippers", "injuries": [
        {"type": {"name": "INJURY_STATUS_QUESTIONABLE"},
         "athlete": {"id": "9", "displayName": "Clip Big"}},
    ]},
]}


def _mins():
    return pd.DataFrame({"player_id": ["1", "2", "3", "9"],
                         "player": ["Star Guard", "Wing Two", "Bench Guy", "Clip Big"],
                         "team": ["Atlanta Hawks"] * 3 + ["Los Angeles Clippers"],
                         "mpg": [34.0, 26.0, 8.0, 30.0]})


def test_parse_and_notes():
    df = inj.parse_injuries(REPORT)
    assert len(df) == 4
    assert set(df["team"]) == {"Atlanta Hawks", "Los Angeles Clippers"}
    assert df.loc[df.player == "Clip Big", "status"].item() == "Questionable"
    r = inj.with_roles(df, _mins())
    n = inj.game_notes(r, "Indiana Pacers", "Atlanta Hawks")
    assert n["out"] == ["Star Guard (Hawks, 34 min)"]
    assert n["watch"] == ["Wing Two (Hawks, 26 min) day-to-day"]   # bench guy too small
    assert inj.game_notes(None, "a", "b") == {"out": [], "watch": []}


def test_recent_minutes_uses_last_team_games():
    box = pd.DataFrame({
        "team": ["T"] * 4, "game_id": ["1", "2", "3", "3"],
        "commence": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-03"],
        "player_id": ["a", "a", "a", "b"], "player": ["A", "A", "A", "B"],
        "min": [30.0, 20.0, 10.0, 40.0]})
    m = inj.recent_minutes(box, last=2).set_index("player")["mpg"]
    assert m["A"] == 15.0 and m["B"] == 20.0


def test_watch_shows_on_page_and_alert():
    bet = {"away": "Atlanta Hawks", "home": "Indiana Pacers",
           "commence": "2026-11-03T00:40:00Z", "market": "spreads",
           "pick": "Atlanta Hawks", "point": -5.5, "price": -105,
           "book": "FanDuel", "stake": 5.0, "win_chance": 0.535, "ev": 0.04,
           "fair_market": -6.25, "why": "x", "out": "Star Guard (Hawks, 34 min)",
           "watch": "Wing Two (Hawks, 26 min) day-to-day"}
    md = render_md(pd.DataFrame([bet]), "now", [])
    assert "Wait for news" in md and "Already out" in md
    assert "wait for news: Wing Two" in render_alert([bet])


def test_dashboard_embeds_data_without_nan():
    import json, re
    from picks import render_html
    from selector import Fair
    bet = {"away": "A", "home": "B", "commence": "2026-11-03T00:40:00Z",
           "market": "spreads", "pick": "A", "point": 5.5, "price": -110,
           "book": "DraftKings", "stake": 3.0, "win_chance": 0.53, "ev": 0.02,
           "fair_market": float("nan"), "why": "x", "out": "", "watch": "w"}
    fair = {"g": Fair("B", "A", "2026-11-03T00:40:00Z", market_margin=-4.8,
                      market_total=float("nan"))}
    html = render_html([bet], fair, "now", 1000, [])
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                     html, re.S).group(1)
    d = json.loads(blob)
    assert d["picks"][0]["fair_market"] is None and d["fair"][0]["market_total"] is None


def test_log_changes_records_only_changes(tmp_path):
    path = str(tmp_path / "changes.csv")
    rep = inj.parse_injuries(REPORT)
    first = inj.log_changes(rep, path, "t1")
    assert len(first) == 4                               # whole report
    assert inj.log_changes(rep, path, "t2").empty        # nothing changed
    rep2 = rep.copy()
    rep2.loc[rep2.player == "Wing Two", "status"] = "Out"
    rep2 = rep2[rep2.player != "Clip Big"]
    ch = inj.log_changes(rep2, path, "t3")
    assert sorted(zip(ch.player, ch.status)) == [("Clip Big", "Removed"),
                                                 ("Wing Two", "Out")]
    # A failed fetch logs nothing (doesn't clear the league)
    assert inj.log_changes(inj.parse_injuries({}), path, "t4").empty
    # Back on the report after being removed: logged again
    ch = inj.log_changes(rep, path, "t5")
    assert set(ch.player) == {"Clip Big", "Wing Two"}
    log = pd.read_csv(path)
    assert list(log.columns) == inj.LOG_COLS and len(log) == 8
