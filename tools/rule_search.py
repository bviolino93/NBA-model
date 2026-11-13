"""Is there ANY slice of NBA games where our own line beats the close?

Reads model_lab.py's per-game held-out predictions (--out lab_games.csv;
each season predicted by models fit on earlier seasons only). Tries a grid
of pick rules for spreads and totals, CHOOSES on the early test seasons and
REPORTS the chosen rules on later seasons the search never saw. The same
method found the college favourite rule (cbb-model tools/rule_search.py).

    python tools/rule_search.py lab_games.csv [--split 2025]
"""
import argparse
import itertools

import numpy as np
import pandas as pd

BE = 110 / 210


def frame(d, market):
    if market == "spreads":
        line, act = d["spread_line"], d["home_score"] - d["away_score"]
        pred, alt = d["trees_margin"], d["+players_margin"]
    else:
        line, act = d["total_line"], d["home_score"] + d["away_score"]
        pred, alt = d["trees_total"], d["+players_total"]
    x = pd.DataFrame({"season": d["season"], "game_date": d["game_date"],
                      "line": line, "gap": pred - line, "alt": alt - line,
                      "act": act}, index=d.index)
    x = x[x["line"].notna() & x["gap"].notna()]
    x["res"] = np.sign(x["gap"] * (x["act"] - x["line"]))
    x = x[x["res"] != 0].copy()
    x["win"] = (x["res"] > 0).astype(int)
    x["ag"] = x["gap"].abs()
    x["agree"] = np.sign(x["alt"]) == np.sign(x["gap"])
    x["month"] = pd.to_datetime(x["game_date"]).dt.month
    hb = d.get("home_b2b", pd.Series(0, index=d.index)).reindex(x.index).fillna(0)
    ab = d.get("away_b2b", pd.Series(0, index=d.index)).reindex(x.index).fillna(0)
    x["any_b2b"] = (hb + ab) > 0
    x["bet_side_b2b"] = np.where(x["gap"] > 0, hb > 0, ab > 0) if market == "spreads" else False
    x["opp_b2b"] = np.where(x["gap"] > 0, ab > 0, hb > 0) if market == "spreads" else False
    if market == "spreads":
        bet_home = x["gap"] > 0
        x["bet_home"] = bet_home
        x["bet_dog"] = np.where(x["line"] > 0, ~bet_home, bet_home) & (x["line"] != 0)
        x["al"] = x["line"].abs()
    else:
        x["bet_home"] = x["gap"] > 0  # = over
        x["bet_dog"] = False
        x["al"] = 0.0
    return x


def filters(market):
    gaps = [(1, 99), (2, 99), (3, 99), (4, 99), (6, 99), (2, 5)] if market == "spreads" \
        else [(2, 99), (3, 99), (4, 99), (6, 99), (8, 99), (3, 6)]
    f = {"gap": {f"{lo}+" if hi == 99 else f"{lo}-{hi}":
                 (lambda lo, hi: lambda x: (x.ag >= lo) & (x.ag < hi))(lo, hi) for lo, hi in gaps},
         "agree": {"any": None, "ridge agrees": lambda x: x.agree},
         "season part": {"any": None, "Oct-Dec": lambda x: x.month.isin([10, 11, 12]),
                         "Jan+": lambda x: ~x.month.isin([10, 11, 12])},
         "b2b": {"any": None, "a b2b in game": lambda x: x.any_b2b,
                 "no b2b": lambda x: ~x.any_b2b}}
    if market == "spreads":
        f["side"] = {"any": None, "home": lambda x: x.bet_home, "away": lambda x: ~x.bet_home,
                     "dog": lambda x: x.bet_dog, "fav": lambda x: ~x.bet_dog}
        f["line"] = {"any": None, "|line|<5": lambda x: x.al < 5,
                     "|line| 5-10": lambda x: (x.al >= 5) & (x.al < 10),
                     "|line| 10+": lambda x: x.al >= 10}
        f["b2b"]["opp on b2b"] = lambda x: x.opp_b2b
    else:
        f["side"] = {"any": None, "over": lambda x: x.bet_home, "under": lambda x: ~x.bet_home}
    return f


def rec(x):
    n = len(x)
    if not n:
        return "    0 bets"
    p = x["win"].mean()
    return f"{n:5d} bets {p:6.1%} (se {np.sqrt(p * (1 - p) / n):.1%})"


def search(x, market, split, min_bets):
    tr, te = x[x.season < split], x[x.season >= split]
    print(f"\n===== {market.upper()} ===== choose on {sorted(tr.season.unique())} ({len(tr)} games), "
          f"confirm on {sorted(te.season.unique())} ({len(te)})")
    print(f"every game, model side: choose {tr.win.mean():.1%}, confirm {te.win.mean():.1%}")
    F = filters(market)
    keys = list(F)
    rows = []
    for combo in itertools.product(*[F[k].items() for k in keys]):
        def mask(z):
            m = np.ones(len(z), bool)
            for _, f in combo:
                if f is not None:
                    m &= np.asarray(f(z), bool)
            return m
        mt, mc = mask(tr), mask(te)
        n = int(mt.sum())
        if n < min_bets:
            continue
        p = tr.win.to_numpy()[mt].mean()
        q = te.win.to_numpy()[mc]
        rows.append({"rule": ", ".join(f"{k}={v}" for k, (v, _) in zip(keys, combo) if v != "any"),
                     "n1": n, "w1": p, "lb": p - 1.5 * np.sqrt(p * (1 - p) / n),
                     "n2": len(q), "w2": q.mean() if len(q) else np.nan, "combo": combo})
    r = pd.DataFrame(rows).sort_values("lb", ascending=False)
    print(f"{len(r)} rules with {min_bets}+ bets. Average confirm win {r.w2.mean():.1%}; "
          f"top 20 by choose score {r.head(20).w2.mean():.1%}; bottom half "
          f"{r.tail(len(r) // 2).w2.mean():.1%}; choose/confirm correlation "
          f"{r[['w1', 'w2']].corr().iloc[0, 1]:+.2f}")
    print("(top 20 must clearly beat the average on unseen seasons, or the search found noise)")
    for _, z in r.head(20).iterrows():
        print(f"  {z.w1:5.1%} on {z.n1:4d} -> {z.w2:5.1%} on {z.n2:4d}   {z.rule}")
    print("single filters (all seasons, then by season)")
    for k in keys:
        for v, f in F[k].items():
            if f is None:
                continue
            s = x[np.asarray(f(x), bool) & (x.ag >= (2 if market == "spreads" else 3))]
            by = " ".join(f"{g.win.mean():.0%}" for _, g in s.groupby("season"))
            print(f"  {k:11s} {v:14s} (+gap {2 if market == 'spreads' else 3}+) {rec(s)}  {by}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="lab_games.csv")
    ap.add_argument("--split", type=int, default=2025)
    ap.add_argument("--min-bets", type=int, default=250)
    a = ap.parse_args()
    d = pd.read_csv(a.path)
    print(f"{len(d)} held-out games, seasons {sorted(d.season.unique())}. Break-even {BE:.1%}")
    for m in ("spreads", "totals"):
        search(frame(d, m), m, a.split, a.min_bets)


if __name__ == "__main__":
    main()
