import ledger


def _bet(pick, point, seen="2026-11-02T01:00:00+00:00"):
    return {"away": "Atlanta Hawks", "home": "Indiana Pacers",
            "commence": "2026-11-02T23:00:00Z", "market": "spreads",
            "pick": pick, "point": point, "price": -110, "stake": 10,
            "first_seen": seen}


def test_other_side_never_added():
    hist = ledger.merge([], [_bet("Atlanta Hawks", -5.5)],
                        "2026-11-02T01:00:00Z", "r1")
    picks = [_bet("Indiana Pacers", 5.5)]
    assert ledger.one_side(hist, picks) == []
    hist = ledger.merge(hist, picks, "2026-11-02T03:00:00Z", "r2")
    assert [b["pick"] for b in hist] == ["Atlanta Hawks"]


def test_existing_duplicate_keeps_first_alert():
    hist = [_bet("Indiana Pacers", 5.5, "2026-11-02T03:42:00+00:00"),
            _bet("Atlanta Hawks", -5.5, "2026-11-02T01:00:00+00:00")]
    out = ledger.merge(hist, [], "2026-11-02T04:00:00Z", "r3")
    assert [b["pick"] for b in out] == ["Atlanta Hawks"]


def test_same_side_still_tracked():
    hist = ledger.merge([], [_bet("Atlanta Hawks", -5.5)],
                        "2026-11-02T01:00:00Z", "r1")
    assert len(ledger.one_side(hist, [_bet("Atlanta Hawks", -6)])) == 1
