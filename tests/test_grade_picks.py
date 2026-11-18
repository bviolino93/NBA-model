import json

from tools import grade_picks as gp


def P(**k):
    base = {"sport": "nba", "home": "Miami Heat", "away": "Milwaukee Bucks",
            "stake": 5}
    base.update(k)
    return base


def test_grade_spread_and_total():
    sc = (120.0, 110.0)                       # Heat by 10, total 230
    assert gp.grade(P(market="spread", side="Milwaukee Bucks", line=12.5), sc) == "W"
    assert gp.grade(P(market="spread", side="Miami Heat", line=-12.5), sc) == "L"
    assert gp.grade(P(market="spread", side="Miami Heat", line=-10), sc) == "P"
    assert gp.grade(P(market="total", side="Over", line=229.5), sc) == "W"
    assert gp.grade(P(market="total", side="Under", line=229.5), sc) == "L"
    assert gp.grade(P(market="total", side="Under", line=229.5), None) == ""


def test_grade_day_and_record(tmp_path):
    day = {"date": "2026-11-18", "picks": [
        P(market="spread", side="Milwaukee Bucks", line=12.5, confidence=3),
        P(market="total", side="Over", line=240, confidence=2),
        P(sport="cbb", home="Idaho Vandals", away="Cal Poly Mustangs",
          market="spread", side="Idaho Vandals", line=-1.5, confidence=4)]}
    (tmp_path / "picks").mkdir()
    path = tmp_path / "picks" / "2026-11-18.json"
    path.write_text(json.dumps(day))
    fin = {"nba": {("Miami Heat", "Milwaukee Bucks"): (120.0, 110.0)},
           "cbb": {}}
    d = gp.grade_day(str(path), str(tmp_path / "g"), fetch=lambda s, _: fin[s])
    assert [p["result"] for p in d["picks"]] == ["W", "L", ""]
    assert abs(d["picks"][0]["profit"] - 5 * 100 / 110) < 1e-2
    r = gp.record([d])
    assert r["all"][:3] == [1, 1, 0]
    assert r["nba spread"][:3] == [1, 0, 0]
    assert "cbb" not in r
