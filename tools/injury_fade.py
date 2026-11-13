"""
Does the closing spread overreact when a team is missing key players?

For every regular-season game with a closing spread, find who was missing
on each side (nba_players.roles: averaged 15+ min over the last 10 and
played in one of the last 5, then sat). A "star" averaged 30+ minutes.
A "fresh" absence is a player who played in the team's previous game,
so the news is new and the market had to react to it.

Then, from the shorthanded team's side, count covers against the close:
  * missing a star, opponent missing none
  * missing a fresh star, opponent missing none
  * by size of the value gap (plus-minus value lost, home minus away)
plus a regression of (margin - close) on the absence gaps. A positive
cover rate above 52.4% for the shorthanded side (or below 47.6%) would
be an edge after the vig; within +-2 standard errors of 50% is nothing.

    python tools/injury_fade.py --first 2016
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nba_data import load_games  # noqa: E402
from nba_players import load_box, roles  # noqa: E402


def team_missing(box: pd.DataFrame, min_minutes: float = 15.0) -> pd.DataFrame:
    r = roles(box).sort_values(["key", "player_id", "n"])
    r["prev_min"] = r.groupby(["key", "player_id"])["min_now"].shift(1)
    miss = r[(r["exp_min"] >= min_minutes) & r["recent"] & (r["min_now"] <= 0)]
    miss = miss.assign(
        star=(miss["exp_min"] >= 30).astype(int),
        fresh=(miss["prev_min"].fillna(0) > 0).astype(int),
        val=miss["exp_min"] / 36.0 * miss["pm36"].fillna(0).clip(lower=0)
            + miss["exp_min"] / 48.0,
    )
    miss["fresh_star"] = miss["star"] * miss["fresh"]
    miss["fresh_val"] = miss["val"] * miss["fresh"]
    return (miss.groupby(["game_id", "team"])
                .agg(stars=("star", "sum"), fresh_stars=("fresh_star", "sum"),
                     val=("val", "sum"), fresh_val=("fresh_val", "sum"),
                     n_out=("player", "size"))
                .reset_index())


def rate(cover: pd.Series) -> str:
    c = cover.dropna()
    n = len(c)
    if n == 0:
        return "n 0"
    p = c.mean()
    se = np.sqrt(0.25 / n)
    return f"{p*100:5.1f}% of {n:5d} (se {se*100:.1f}, z {(p-0.5)/se:+.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2016)
    a = ap.parse_args()
    g = load_games(a.first)
    g = g[g["season_type"].isin(["REG", "PLAYIN"]) & g["completed"]]
    g = g[np.isfinite(g["spread_line"].astype(float))].copy()
    g["game_id"] = g["game_id"].astype(str)
    box = load_box(g)
    tm = team_missing(box)
    tm["game_id"] = tm["game_id"].astype(str)
    for side in ("home", "away"):
        s = tm.rename(columns={c: f"{side}_{c}" for c in
                               ("stars", "fresh_stars", "val", "fresh_val", "n_out")})
        g = g.merge(s.rename(columns={"team": f"{side}_team"}),
                    on=["game_id", f"{side}_team"], how="left")
        for c in ("stars", "fresh_stars", "val", "fresh_val", "n_out"):
            g[f"{side}_{c}"] = g[f"{side}_{c}"].fillna(0)
    g["ats"] = (g["home_score"] - g["away_score"]) - g["spread_line"].astype(float)
    g = g[g["ats"] != 0]
    home_cover = (g["ats"] > 0).astype(float)
    print(f"{len(g)} games with a closing spread (pushes dropped), "
          f"seasons {int(g['season'].min())}-{int(g['season'].max())}\n")

    def side_view(home_mask, away_mask, label):
        """Cover rate of the shorthanded side."""
        c = pd.concat([home_cover[home_mask], 1 - home_cover[away_mask]])
        print(f"{label:52s} {rate(c)}")
        return c

    print("Cover rate of the SHORTHANDED team vs the closing spread")
    side_view((g.home_stars >= 1) & (g.away_stars == 0),
              (g.away_stars >= 1) & (g.home_stars == 0),
              "missing 1+ star (30+ min), opponent none")
    side_view((g.home_stars >= 2) & (g.away_stars == 0),
              (g.away_stars >= 2) & (g.home_stars == 0),
              "missing 2+ stars, opponent none")
    side_view((g.home_fresh_stars >= 1) & (g.away_stars == 0),
              (g.away_fresh_stars >= 1) & (g.home_stars == 0),
              "missing a FRESH star (played last game), opp none")
    side_view((g.home_stars >= 1) & (g.home_fresh_stars == 0) & (g.away_stars == 0),
              (g.away_stars >= 1) & (g.away_fresh_stars == 0) & (g.home_stars == 0),
              "missing a star for 2+ games already, opp none")
    d = g["home_val"] - g["away_val"]
    for lo in (2, 4, 6, 8):
        side_view(d >= lo, d <= -lo, f"value gap {lo}+ (pts of plus-minus value)")
    fd = g["home_fresh_val"] - g["away_fresh_val"]
    for lo in (2, 4, 6):
        side_view(fd >= lo, fd <= -lo, f"FRESH value gap {lo}+")

    print("\nShorthanded side as underdog vs favourite (1+ star, opp none)")
    hm = (g.home_stars >= 1) & (g.away_stars == 0)
    am = (g.away_stars >= 1) & (g.home_stars == 0)
    sp = g["spread_line"].astype(float)
    side_view(hm & (sp < 0), am & (sp > 0), "  shorthanded team is the underdog")
    side_view(hm & (sp > 0), am & (sp < 0), "  shorthanded team is still favoured")

    print("\nBy season (1+ star missing, opponent none)")
    for s, gs in g.groupby("season"):
        hs = (gs.home_stars >= 1) & (gs.away_stars == 0)
        as_ = (gs.away_stars >= 1) & (gs.home_stars == 0)
        hc = (gs["ats"] > 0).astype(float)
        c = pd.concat([hc[hs], 1 - hc[as_]])
        print(f"  {int(s)}: {rate(c)}")

    print("\nRegression: (margin - close) on home-minus-away absence gaps")
    y = g["ats"].to_numpy(float)
    for name, x in [("value gap", d), ("fresh value gap", fd),
                    ("star gap", g.home_stars - g.away_stars)]:
        X = np.column_stack([np.ones(len(y)), x.to_numpy(float)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        se = np.sqrt(np.diag(r @ r / (len(y) - 2) * np.linalg.inv(X.T @ X)))
        print(f"  {name:16s} {b[1]:+.3f} pts per unit (t {b[1]/se[1]:+.1f}); "
              f"positive = shorthanded HOME side beats the close")

    if "spread_open" in g:
        o = g[np.isfinite(g["spread_open"].astype(float))]
        print(f"\nOpen-to-close (2023-24 on; ESPN opens are noisy): {len(o)} games")
        mv = (o["spread_line"] - o["spread_open"]).astype(float)
        od = (o["home_fresh_val"] - o["away_fresh_val"])
        X = np.column_stack([np.ones(len(o)), od.to_numpy(float)])
        b, *_ = np.linalg.lstsq(X, mv.to_numpy(float), rcond=None)
        print(f"  line moves {b[1]:+.2f} pts per point of fresh value gap")


if __name__ == "__main__":
    main()
