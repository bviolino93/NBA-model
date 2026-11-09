import numpy as np
import pandas as pd

import selector as sel


def outcome_models(seed=0):
    rng = np.random.default_rng(seed)
    n = 6000
    spread = rng.choice(np.arange(-14, 14.5, 0.5), n)
    total = rng.choice(np.arange(205, 245.5, 0.5), n)
    margin = np.round(spread + rng.normal(0, 12, n))
    points = np.round(total + rng.normal(0, 18, n))
    g = pd.DataFrame({"season": 2024, "spread_line": spread,
                      "total_line": total,
                      "home_score": np.round((points + margin) / 2),
                      "away_score": np.round((points - margin) / 2)})
    return sel.build_outcome_models(g, since=2016)


def snap():
    rows = []
    home, away = "Boston Celtics", "Philadelphia 76ers"
    for book, hp, ap_, tp in (("pinnacle", -110, -110, -110),
                              ("circasports", -110, -110, -110),
                              ("draftkings", -110, -110, -110),
                              ("fanduel", -105, -115, -110)):
        line = -7.5 if book != "fanduel" else -6.5   # FanDuel is stale
        rows += [
            dict(event_id="e1", commence_time="2025-10-22T23:30:00Z",
                 home_team=home, away_team=away, book=book, market="spreads",
                 outcome=home, point=line, price=hp),
            dict(event_id="e1", commence_time="2025-10-22T23:30:00Z",
                 home_team=home, away_team=away, book=book, market="spreads",
                 outcome=away, point=-line, price=ap_),
            dict(event_id="e1", commence_time="2025-10-22T23:30:00Z",
                 home_team=home, away_team=away, book=book, market="totals",
                 outcome="Over", point=224.5, price=tp),
            dict(event_id="e1", commence_time="2025-10-22T23:30:00Z",
                 home_team=home, away_team=away, book=book, market="totals",
                 outcome="Under", point=224.5, price=tp),
        ]
    return pd.DataFrame(rows)


def test_stale_book_is_the_bet():
    mm, tm = outcome_models()
    s = snap()
    fair = sel.consensus(s, mm, tm)
    assert abs(fair["e1"].market_margin - 7.5) < 1.0
    picks = sel.select(sel.price_offers(s, fair, mm, tm), bankroll=1000)
    sp = picks[picks["market"] == "spreads"]
    assert len(sp) == 1
    assert sp.iloc[0]["book"] == "fanduel"
    assert sp.iloc[0]["pick"] == "Boston Celtics"
    assert sp.iloc[0]["stake"] <= 10.0     # 1% cap on $1,000


def test_fair_line_ignores_soft_books_and_stale_quotes():
    mm, tm = outcome_models()
    s = snap()
    soft = s[s["book"] == "draftkings"]
    # Ten soft books all hanging -5.5 don't move the sharp -7.5
    many = pd.concat([soft.assign(book=f"soft{i}",
                                  point=soft["point"].where(
                                      soft["market"] != "spreads",
                                      soft["point"].clip(-5.5, 5.5)))
                      for i in range(10)])
    f = sel.consensus(pd.concat([s, many]), mm, tm)["e1"]
    assert abs(f.market_margin - 7.5) < 0.6 and f.sharp_margin == 2
    # A Pinnacle quote 2 hours old is dropped
    s = s.assign(snapshot_utc="2025-10-22T20:00:00Z",
                 last_update="2025-10-22T19:59:00Z")
    s.loc[s["book"] == "pinnacle", "last_update"] = "2025-10-22T18:00:00Z"
    f = sel.consensus(s, mm, tm)["e1"]
    assert f.sharp_margin == 1               # Circa only
