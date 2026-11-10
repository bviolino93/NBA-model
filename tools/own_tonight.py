"""
Our own line (nba_predict) for the next two days' games, next to ESPN's
pregame line where it has one. A dry run of what picks.py computes, with
no Odds API credits: ESPN's games stand in for the priced games.

    python tools/own_tonight.py
"""
import os
import sys
import time
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nba_injuries as inj  # noqa: E402
import nba_predict as npr  # noqa: E402
from nba_data import load_games  # noqa: E402
from nba_players import load_box  # noqa: E402


def main():
    t0 = time.time()
    g = load_games(2021)
    now = pd.Timestamp.now(tz="UTC")
    up = g[~g["completed"] & (pd.to_datetime(g["commence"], utc=True) > now)
           & (pd.to_datetime(g["commence"], utc=True) < now + pd.Timedelta(days=2))]
    fair = {r.game_id: SimpleNamespace(home_team=r.home_team, away_team=r.away_team,
                                       commence=r.commence) for r in up.itertuples()}
    box = load_box(g)
    try:
        rep = inj.fetch_injuries()
    except Exception as exc:
        print(f"no injury report: {exc}")
        rep = None
    out = npr.live_lines(g, box, fair, rep, now)
    print(f"{len(out)} of {len(up)} games projected ({time.time() - t0:.0f}s)")
    for r in up.itertuples():
        v = out.get(r.game_id)
        if not v:
            continue
        print(f"  {r.away_team} @ {r.home_team}: ours home {v['margin']:+.1f}, "
              f"total {v['total']:.1f} | ESPN {r.spread_line:+.1f} / {r.total_line:.1f}"
              f" | parts {v['parts']}")


if __name__ == "__main__":
    main()
