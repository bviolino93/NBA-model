import numpy as np
import pandas as pd

import nba_model
from selector import Fair


def _games(n=600):
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(10)]
    d0 = pd.Timestamp("2025-10-20")
    rows = []
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        day = d0 + pd.Timedelta(days=i // 5)
        season = 2026 if day < pd.Timestamp("2026-07-01") else 2027
        hs = 113 + (6 if h == "T0" else 0) - (6 if a == "T0" else 0) + rng.normal(0, 11)
        rows.append({"game_id": str(i), "season": season, "game_date": day.date(),
                     "commence": (day + pd.Timedelta(hours=24)).isoformat() + "Z",
                     "home_team": h, "away_team": a, "home_score": round(hs),
                     "away_score": round(111 + rng.normal(0, 11)), "completed": True,
                     "neutral": False, "spread_line": np.nan, "total_line": np.nan})
    return pd.DataFrame(rows)


def test_fit_and_predict_with_injury():
    g = _games()
    h = nba_model.history(g, None)
    f = nba_model.fit_weights(h)
    assert f.n > 100 and 0.3 < f.margin["model_margin"] < 1.5
    box = pd.DataFrame({"season": 2027, "commence": "2027-01-01", "game_id": "x",
                        "team": "T0", "player_id": "7", "player": "Star",
                        "min": 34.0, "pm": 20.0, "pts": 30.0, "reb": 8.0,
                        "ast": 6.0}, index=[0])
    vals = nba_model.player_values(box)
    rep = pd.DataFrame({"team": ["T0"], "player_id": ["7"], "player": ["Star"],
                        "status": ["Out"], "detail": [""]})
    miss = nba_model.tonight_missing(rep, vals)
    assert miss["T0"][0]["min"] == 34.0 and miss["T0"][0]["prod"] > 0
    assert miss["T0"][1] == ["Star out"]
    now = pd.Timestamp(g["game_date"].max()) + pd.Timedelta(days=1, hours=17)
    now = now.tz_localize("UTC")
    fair = {"k": Fair("T0", "T1", (now + pd.Timedelta(hours=7)).isoformat(),
                      market_margin=3.0, market_total=224.0)}
    healthy = nba_model.predict(g, fair, f, {}, now)["k"]
    hurt = nba_model.predict(g, fair, f, miss, now)["k"]
    assert healthy["margin"] > 3
    assert set(hurt["parts"]) == {"ratings", "rest", "injuries"}
    assert hurt["missing"]["home"] == ["Star out"]


def test_anchored_parts_add_up():
    g = _games()
    rng = np.random.default_rng(2)
    g["spread_line"] = (g["home_score"] - g["away_score"]) * 0.3 + rng.normal(0, 3, len(g))
    g["total_line"] = 224 + rng.normal(0, 5, len(g))
    f = nba_model.fit_weights(nba_model.history(g, None))
    assert f.anchor_margin and f.anchor_total
    now = (pd.Timestamp(g["game_date"].max()) + pd.Timedelta(days=1, hours=17)).tz_localize("UTC")
    fair = {"k": Fair("T0", "T1", (now + pd.Timedelta(hours=7)).isoformat(),
                      market_margin=3.0, market_total=224.0)}
    p = nba_model.predict(g, fair, f, {}, now)["k"]
    assert p["anchored"]
    assert abs(sum(p["parts"].values()) - p["margin"]) < 0.3
    assert abs(sum(p["total_parts"].values()) - p["total"]) < 0.3
