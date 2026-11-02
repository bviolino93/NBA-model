import numpy as np
import pandas as pd

import nba_players as npl
import nba_props


def box(n_games=12, star_out_from=8):
    rows = []
    for n in range(n_games):
        t = pd.Timestamp("2025-10-21", tz="UTC") + pd.Timedelta(days=2 * n)
        common = dict(game_id=f"g{n}", season=2026, game_date=t.date(),
                      commence=t, team="A", opp="Z", home=1, team_abbr="A",
                      reason="", starter=True, played=True, pm=0.0)
        if n < star_out_from:
            rows.append(dict(common, player_id="S", player="Star", min=36.0,
                             pts=30, reb=6, ast=8, fg3m=3))
        rows.append(dict(common, player_id="R", player="Role", min=24.0,
                         pts=12, reb=6, ast=2, fg3m=2))
    return pd.DataFrame(rows)


def test_projection_uses_only_earlier_games_and_boosts_minutes():
    b = box()
    ab = npl.absences(b)
    fr = nba_props.project(b, ab)
    r = fr[fr["player_id"] == "R"].set_index("game_id")
    # First game: nothing earlier, so no projection.
    assert np.isnan(r.loc["g0", "base_min"])
    # Before the star sits, role player projects at his 24 minutes.
    assert abs(r.loc["g5", "proj_min"] - 24.0) < 1e-9
    # Star out (36 expected minutes): 24 * 240 / 204
    assert abs(r.loc["g8", "proj_min"] - 24.0 * 240 / 204) < 1e-6
    # Rate is pulled toward the league prior early, toward his own later.
    assert r.loc["g7", "rate_pts"] > 0
    assert abs(r.loc["g8", "proj_pts"]
               - r.loc["g8", "proj_min"] * r.loc["g8", "rate_pts"]) < 1e-9


def test_mae_table_runs():
    b = box(n_games=20, star_out_from=20)
    fr = nba_props.project(b, npl.absences(b))
    t = nba_props.mae_table(fr, min_prev=3)
    assert set(t.index) == {"pts", "reb", "ast", "fg3m"}
    # Constant lines: the shrunk rate starts near the league prior and
    # converges on the player's own; errors are finite and shrinking.
    assert np.isfinite(t.loc["pts", "model"])
    r = fr[fr["player_id"] == "R"]
    err = (r["proj_pts"] - r["pts"]).abs().dropna()
    assert err.iloc[-1] < err.iloc[0]
