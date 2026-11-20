"""
Game-by-game breakdown for one NBA day: everything a bettor would look at
for each game, in one sheet, so the picks can be made game by game.

For every game on the date it lists
  - the line now (DraftKings/FanDuel, Pinnacle) and how it moved since
    our first snapshot (data/lines/games_<season>.csv), ESPN's otherwise
  - our projected score (nba_predict trees model, no market input)
  - matchup: possessions ratings, pace, four-factor edges, player impact
  - recent form: last 10 record, average margin, covers, overs
  - schedule: rest, back-to-backs, games in 5 nights, travel
  - injuries: ESPN report, with each player's recent minutes
and a list of plain-English notes. It writes breakdown/nba/<date>.md and
.json. It makes no picks: the picks are made from the sheet.

    python tools/breakdown.py [--date 2026-11-18] [--out breakdown]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nba_injuries as inj  # noqa: E402
import nba_predict as npr  # noqa: E402
from nba_data import canon, current_season, load_games  # noqa: E402
from nba_players import load_box  # noqa: E402

BET_BOOKS = ("draftkings", "fanduel")
SHARP = "pinnacle"


# ----------------------------------------------------------------------
# Pieces (pure, tested in tests/test_breakdown.py)
# ----------------------------------------------------------------------
def team_form(g: pd.DataFrame, team: str, before, last: int = 10) -> dict:
    """Last `last` completed games of `team` before `before`: record,
    average margin and points, covers and overs against the line."""
    d = g[g["completed"] & (pd.to_datetime(g["commence"], utc=True)
                            < pd.Timestamp(before))
          & ((g["home_team"] == team) | (g["away_team"] == team))]
    d = d.sort_values("commence").tail(last)
    if d.empty:
        return {"n": 0}
    home = (d["home_team"] == team).to_numpy()
    hm = (d["home_score"] - d["away_score"]).to_numpy(float)
    m = np.where(home, hm, -hm)
    pf = np.where(home, d["home_score"], d["away_score"]).astype(float)
    pa = np.where(home, d["away_score"], d["home_score"]).astype(float)
    sl = d["spread_line"].to_numpy(float)       # expected home margin
    vs = np.where(home, hm - sl, sl - hm)       # beat the line by
    tot = (d["home_score"] + d["away_score"]).to_numpy(float)
    tl = d["total_line"].to_numpy(float)
    ok, okt = np.isfinite(vs), np.isfinite(tl)
    out = {"n": int(len(d)), "w": int((m > 0).sum()), "l": int((m < 0).sum()),
           "margin": round(float(m.mean()), 1),
           "pf": round(float(pf.mean()), 1), "pa": round(float(pa.mean()), 1),
           "cover": int((vs[ok] > 0).sum()), "no_cover": int((vs[ok] < 0).sum()),
           "vs_line": round(float(vs[ok].mean()), 1) if ok.any() else None,
           "over": int((tot[okt] > tl[okt]).sum()),
           "under": int((tot[okt] < tl[okt]).sum()),
           "streak": _streak(m)}
    return out


def _streak(m) -> str:
    if len(m) == 0:
        return ""
    last = m[-1] > 0
    k = 0
    for x in m[::-1]:
        if (x > 0) != last:
            break
        k += 1
    return f"{'W' if last else 'L'}{k}"


def line_history(lines: pd.DataFrame, home: str, away: str, commence
                 ) -> dict:
    """First and latest spread (home number, negative = home favoured as
    books quote it) and total, at the books we bet and at Pinnacle."""
    if lines is None or lines.empty:
        return {}
    tip = pd.Timestamp(commence)
    x = lines[(lines["home_team"] == home) & (lines["away_team"] == away)]
    if x.empty:
        return {}
    x = x[(pd.to_datetime(x["commence_time"], utc=True) - tip).abs()
          <= pd.Timedelta(hours=30)]
    x = x[pd.to_datetime(x["snapshot_utc"], utc=True)
          < pd.to_datetime(x["commence_time"], utc=True)]
    out = {}
    for key, books in (("bet", BET_BOOKS), ("sharp", (SHARP,))):
        b = x[x["book"].isin(books)]
        sp = b[(b["market"] == "spreads") & (b["outcome"] == home)]
        to = b[(b["market"] == "totals") & (b["outcome"] == "Over")]
        for lab, part in (("spread", sp), ("total", to)):
            if part.empty:
                continue
            by = part.groupby("snapshot_utc")["point"].median().sort_index()
            out[f"{key}_{lab}_open"] = float(by.iloc[0])
            out[f"{key}_{lab}_now"] = float(by.iloc[-1])
            out[f"{key}_{lab}_first_snap"] = str(by.index[0])
    if out:
        out["snapshots"] = int(x["snapshot_utc"].nunique())
    return out


def injuries_for(rep: pd.DataFrame, team: str, min_mpg: float = 10.0
                 ) -> list[dict]:
    if rep is None or rep.empty:
        return []
    r = rep[(rep["team"] == team) & (rep["mpg"] >= min_mpg)]
    r = r.sort_values("mpg", ascending=False)
    return [{"player": p.player, "status": p.status, "mpg": round(float(p.mpg), 1),
             "detail": str(getattr(p, "detail", "") or "")}
            for p in r.itertuples()]


def notes(b: dict) -> list[str]:
    """Plain-English flags from one game's breakdown."""
    out = []
    h, a = b["home"], b["away"]
    hs, as_ = b["home_short"], b["away_short"]
    s = b["schedule"]
    for side, name in (("home", hs), ("away", as_)):
        if s.get(f"{side}_b2b"):
            road = " after a road game" if s.get(f"{side}_prev_away") else ""
            out.append(f"{name} on a back-to-back{road}")
        if (s.get(f"{side}_last5") or 0) >= 4:
            out.append(f"{name}: 4th game in 5 nights or worse")
        if (s.get(f"{side}_km") or 0) >= 2.5:
            out.append(f"{name} travelled {s[f'{side}_km'] * 1000:,.0f} km")
    rd = s.get("rest_diff")
    if rd is not None and abs(rd) >= 2:
        more = hs if rd > 0 else as_
        out.append(f"{more} has {abs(rd):.0f} more days of rest")
    for side, name in (("home", hs), ("away", as_)):
        f = b["form"][side]
        if f.get("n", 0) >= 6:
            if f["cover"] >= f["n"] - 2 or f["no_cover"] >= f["n"] - 2:
                out.append(f"{name} {f['cover']}-{f['no_cover']} against the "
                           f"spread last {f['n']}")
            if f["over"] >= f["n"] - 2 or f["under"] >= f["n"] - 2:
                out.append(f"{name} games {f['over']}-{f['under']} over/under "
                           f"last {f['n']}")
        for p in b["injuries"][side]:
            if p["mpg"] >= 24:
                out.append(f"{name}: {p['player']} ({p['mpg']:.0f} min) "
                           f"{p['status'].lower()}")
    ln = b["line"]
    for lab, unit in (("spread", "pts"), ("total", "pts")):
        o, n = ln.get(f"bet_{lab}_open"), ln.get(f"bet_{lab}_now")
        if o is not None and n is not None and abs(n - o) >= 1:
            if lab == "spread":
                toward = hs if n < o else as_
                out.append(f"Spread moved {abs(n - o):.1f} toward {toward} "
                           f"({o:+.1f} to {n:+.1f})")
            else:
                out.append(f"Total moved {'up' if n > o else 'down'} "
                           f"{abs(n - o):.1f} ({o:.1f} to {n:.1f})")
        sh, bn = ln.get(f"sharp_{lab}_now"), n
        if sh is not None and bn is not None and abs(sh - bn) >= 1:
            out.append(f"Pinnacle {lab} {sh:+.1f} vs DK/FD {bn:+.1f}"
                       if lab == "spread" else
                       f"Pinnacle total {sh:.1f} vs DK/FD {bn:.1f}")
    p = b["projection"]
    if p.get("margin") is not None and b["line"].get("spread") is not None:
        gap = p["margin"] + b["line"]["spread"]   # line quoted home number
        if abs(gap) >= 3:
            side = hs if gap > 0 else as_
            out.append(f"Our projection is {abs(gap):.1f} pts better for "
                       f"{side} than the line")
    if p.get("total") is not None and b["line"].get("total") is not None:
        tg = p["total"] - b["line"]["total"]
        if abs(tg) >= 4:
            out.append(f"Our total is {abs(tg):.1f} {'over' if tg > 0 else 'under'}"
                       " the line")
    return out


def _f(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def short(team: str) -> str:
    return team.split()[-1] if team else team


# ----------------------------------------------------------------------
# Situational adjustment and the call (fatigue and missing players)
# ----------------------------------------------------------------------
# Points taken off a team's projected margin. Not fitted: the old backtests
# found the market prices these, so they are deliberate caution, set after
# 11-18 (Lakers on a b2b without Kessler and Reaves lost by 12 as 10.5 favs).
B2B = 2.0            # second night of a back-to-back
B2B_ROAD = 1.0       # extra when the first night was on the road
DENSE = 1.0          # 4th game in 5 nights or worse
TRAVEL = 0.5         # 2,500+ km since the last game
OUT_STAR = 1.0       # each 24+ min player out (beyond the model's roster drop)
DTD_STAR = 0.5       # each 20+ min player questionable / day-to-day
TOTAL_B2B = 2.0      # points off the total per team on a back-to-back
TOTAL_STAR = 1.0     # points off the total per 24+ min player out

UNSURE = ("questionable", "doubtful", "day-to-day", "game time")


def situation(b: dict, side: str) -> tuple[float, float, list[str]]:
    """(margin penalty, total penalty, reasons) for one team."""
    s, name = b["schedule"], b[f"{side}_short"]
    pen, tpen, why = 0.0, 0.0, []
    if s.get(f"{side}_b2b"):
        pen += B2B
        tpen += TOTAL_B2B
        road = bool(s.get(f"{side}_prev_away"))
        if road:
            pen += B2B_ROAD
        why.append(f"{name} on a back-to-back{' after a road game' if road else ''}")
    if (s.get(f"{side}_last5") or 0) >= 4:
        pen += DENSE
        why.append(f"{name} 4th game in 5 nights")
    if (s.get(f"{side}_km") or 0) >= 2.5:
        pen += TRAVEL
        why.append(f"{name} travelled {s[f'{side}_km'] * 1000:,.0f} km")
    outs = [p for p in b["injuries"][side]
            if "out" in p["status"].lower() and p["mpg"] >= 24]
    unsure = [p for p in b["injuries"][side]
              if any(u in p["status"].lower() for u in UNSURE) and p["mpg"] >= 20]
    if outs:
        pen += OUT_STAR * len(outs)
        tpen += TOTAL_STAR * len(outs)
        why.append(f"{name} without " + ", ".join(p["player"] for p in outs))
    if unsure:
        pen += DTD_STAR * len(unsure)
        why.append(f"{name}: " + ", ".join(f"{p['player']} {p['status'].lower()}"
                                             for p in unsure))
    return pen, tpen, why


def _fmt_line(x: float) -> str:
    return "PK" if abs(x) < 0.01 else f"{x:+g}"


BIG_GAP = 8.0       # pts between our margin and the line that caps confidence
SPREAD_MAX = 3      # NBA spread calls are 10-22 so far; keep them off the top


def assess(b: dict) -> dict:
    """The call for one game: adjusted projection, spread and total sides,
    confidence 1-5 from how many factors back each side, minus fatigue and
    injury doubts on the side we pick, with the reasons spelled out."""
    p, ln = b["projection"], b["line"]
    if p.get("margin") is None or ln.get("spread") is None:
        return {}
    hp, htp, hwhy = situation(b, "home")
    ap, atp, awhy = situation(b, "away")
    adj_m = p["margin"] - hp + ap                 # + = home
    adj_t = (p["total"] or 0) - htp - atp
    hs, as_ = b["home_short"], b["away_short"]
    home_m = -ln["spread"]                        # market's home margin
    pick_home = adj_m > home_m
    side, other = ("home", "away") if pick_home else ("away", "home")
    sgn = 1 if pick_home else -1
    fh, fa = b["form"]["home"], b["form"]["away"]
    backs, against = [], []

    def vote(ok, yes, no):
        if ok is None:
            return
        (backs if ok else against).append(yes if ok else no)

    vote(True, f"projection {hs} {adj_m:+.1f} vs line {home_m:+.1f}", "")
    if fh.get("n") and fa.get("n"):
        d = sgn * (fh["margin"] - fa["margin"])
        vote(d > 3 if abs(d) > 3 else None, "recent margins favour our side",
             "recent margins favour the other side")
        if fh.get("vs_line") is not None and fa.get("vs_line") is not None:
            v = sgn * (fh["vs_line"] - fa["vs_line"])
            vote(v > 2 if abs(v) > 2 else None, "beating the line lately",
                 "other side beating the line lately")
    rest = sgn * ((ap + 0.0) - hp)                # fatigue gap in our favour
    # A tired or short-handed side counts against the pick, but the
    # opponent being tired no longer adds confidence (11-19: rest-edge
    # top plays went 0-3 while the tired-team flag held up).
    vote(False if rest <= -1 else None, "", "fatigue/injuries work against our side")
    notes = ["note: fatigue/injury edge (not counted)"] if rest >= 1 else []
    mv = None
    if ln.get("bet_spread_open") is not None and ln.get("bet_spread_now") is not None:
        mv = -(ln["bet_spread_now"] - ln["bet_spread_open"]) * sgn
    vote(mv > 0 if mv and abs(mv) >= 1 else None, "line moving our way",
         "line moving against us")
    conf = 1 + max(0, len(backs) - 1) - (1 if len(against) >= 2 else 0)
    gap = abs(adj_m - home_m)
    my_why = hwhy if pick_home else awhy
    caution = []
    if my_why:
        conf -= 1
        caution = my_why
    if gap >= BIG_GAP:
        # A projection this far from the line usually means the model is
        # missing something (11-19: the big-gap top plays went 0-5). We
        # still take our side, but never as a top play.
        conf = min(conf, 2)
        against.append(f"projection {gap:.1f} pts from the line, model likely missing something")
    conf = int(min(SPREAD_MAX, max(1, conf)))
    team = b["home"] if pick_home else b["away"]
    line = ln["spread"] if pick_home else -ln["spread"]
    spread = {"side": team, "line": line, "confidence": conf,
              "why": "; ".join(backs + [f"against: {x}" for x in against]
                               + [f"caution: {x}" for x in caution] + notes)}
    tot = None
    if ln.get("total") is not None and p.get("total") is not None:
        over = adj_t > ln["total"]
        tc = 1 + (1 if abs(adj_t - ln["total"]) >= 4 else 0)
        ou = [fh.get("over", 0) - fh.get("under", 0), fa.get("over", 0) - fa.get("under", 0)]
        if all((x > 0) == over and x != 0 for x in ou):
            tc += 1
        tired = [w for w in hwhy + awhy if "back-to-back" in w or "without" in w]
        if over and tired:
            tc -= 1
        tot = {"side": "Over" if over else "Under", "line": ln["total"],
               "confidence": int(min(5, max(1, tc))),
               "why": f"adjusted total {adj_t:.1f} (model {p['total']:.1f})"
                      + (f"; caution: {'; '.join(tired)}" if over and tired else "")
                      + (f"; fatigue lowers scoring: {'; '.join(tired)}"
                         if not over and tired else "")}
    return {"adj_margin": round(adj_m, 1), "adj_total": round(adj_t, 1),
            "fatigue": hwhy + awhy, "spread": spread, "total": tot}


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------
def _form_line(name, f):
    if not f.get("n"):
        return f"- {name}: no games yet"
    vl = f.get("vs_line")
    return (f"- {name}: {f['w']}-{f['l']} last {f['n']} ({f['streak']}), "
            f"margin {f['margin']:+.1f}, scores {f['pf']:.0f} allows "
            f"{f['pa']:.0f}; ATS {f['cover']}-{f['no_cover']}"
            + (f" (beat line by {vl:+.1f})" if vl is not None else "")
            + f"; O/U {f['over']}-{f['under']}")


def to_markdown(day, games: list[dict]) -> str:
    out = [f"# NBA breakdown {day}", "",
           "Line: home team's spread as books quote it (negative = home "
           "favoured). Projection: our model, no market input.", ""]
    for b in games:
        hs, as_ = b["home_short"], b["away_short"]
        ln, p, m, s = b["line"], b["projection"], b["matchup"], b["schedule"]
        out.append(f"## {b['away']} @ {b['home']} ({b['tip_et']} ET)")
        sp, tl = ln.get("spread"), ln.get("total")
        src = ln.get("source", "")
        out.append(f"**Line:** {hs} {sp:+.1f}, total {tl:.1f} ({src})"
                   if sp is not None and tl is not None else
                   f"**Line:** none yet ({src})")
        if ln.get("bet_spread_open") is not None:
            out.append(f"- DK/FD open {hs} {ln['bet_spread_open']:+.1f} / "
                       f"{ln.get('bet_total_open', float('nan')):.1f}, now "
                       f"{ln['bet_spread_now']:+.1f} / "
                       f"{ln.get('bet_total_now', float('nan')):.1f} "
                       f"({ln.get('snapshots', 0)} snapshots)")
        if ln.get("sharp_spread_now") is not None:
            out.append(f"- Pinnacle now {hs} {ln['sharp_spread_now']:+.1f} / "
                       f"{ln.get('sharp_total_now', float('nan')):.1f}")
        if p.get("margin") is not None:
            out.append(f"**Projection:** {hs} {p['home_pts']:.0f}, {as_} "
                       f"{p['away_pts']:.0f} ({hs} {p['margin']:+.1f}, total "
                       f"{p['total']:.1f})")
            out.append(f"- parts (home margin): players {p['parts'].get('players')}, "
                       f"box {p['parts'].get('box')}, possessions "
                       f"{p['parts'].get('possessions')}, scores "
                       f"{p['parts'].get('scores')}")
        out.append("**Matchup:**")
        if m.get("pace") is not None:
            out.append(f"- Pace {m['pace']:.1f}; expected offense per 100: "
                       f"{hs} {m.get('home_ortg')}, {as_} {m.get('away_ortg')}")
        ff = [f"{k} {v:+.3f}" for k, v in m.get("four_factors", {}).items()
              if v is not None]
        if ff:
            out.append(f"- Four-factor edges for {hs} (+ = {hs}): "
                       + ", ".join(ff))
        if m.get("apm_diff") is not None:
            out.append(f"- Player impact of who's expected to play: {hs} "
                       f"{m['apm_diff']:+.1f} (box {m.get('box_diff')})")
        out.append("**Form:**")
        out.append(_form_line(hs, b["form"]["home"]))
        out.append(_form_line(as_, b["form"]["away"]))
        out.append("**Schedule:**")
        for side, name in (("home", hs), ("away", as_)):
            r = s.get(f"{side}_rest")
            km = s.get(f"{side}_km")
            out.append(f"- {name}: rest {'?' if r is None else f'{r:.0f}'} days"
                       f"{', back-to-back' if s.get(f'{side}_b2b') else ''}, "
                       f"{s.get(f'{side}_last5') or 0:.0f} games in last 5 "
                       f"nights, travel {0 if km is None else km * 1000:,.0f} km")
        out.append("**Injuries (10+ min players):**")
        for side, name in (("home", hs), ("away", as_)):
            lst = b["injuries"][side]
            out.append(f"- {name}: " + ("none" if not lst else "; ".join(
                f"{p['player']} {p['status']} ({p['mpg']:.0f} min"
                + (f", {p['detail']}" if p['detail'] else "") + ")"
                for p in lst)))
        c = b.get("call") or {}
        if c:
            out.append(f"**Call:** adjusted projection {hs} {c['adj_margin']:+.1f}, "
                       f"total {c['adj_total']:.1f}"
                       + (f" (fatigue/injuries: {'; '.join(c['fatigue'])})"
                          if c["fatigue"] else ""))
            sp = c["spread"]
            out.append(f"- Spread: **{sp['side']} {_fmt_line(sp['line'])}** "
                       f"({sp['confidence']}/5): {sp['why']}")
            if c.get("total"):
                t = c["total"]
                out.append(f"- Total: **{t['side']} {t['line']:g}** "
                           f"({t['confidence']}/5): {t['why']}")
        if b["notes"]:
            out.append("**Notes:**")
            out += [f"- {n}" for n in b["notes"]]
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------
def build_day(day, first: int = 2021, lines_path: str | None = None
              ) -> list[dict]:
    t0 = time.time()
    g = load_games(first)
    g["game_id"] = g["game_id"].astype(str)
    up = g[(pd.to_datetime(g["game_date"]).dt.date == day) & ~g["completed"]]
    print(f"{len(up)} games on {day} not final ({time.time() - t0:.0f}s)",
          flush=True)
    if up.empty:
        return []
    box = load_box(g)
    rep = inj.fetch_injuries()
    if rep is not None and len(rep):
        rep = inj.with_roles(rep, inj.recent_minutes(box))
    lines = None
    lp = lines_path or f"data/lines/games_{current_season()}.csv"
    if os.path.exists(lp):
        lines = pd.read_csv(lp)
        for c in ("home_team", "away_team"):
            lines[c] = lines[c].map(canon)

    # Model features and projections (as nba_predict.live_lines, keeping
    # the feature rows)
    gg = g[g["season_type"].isin(["REG", "PLAYIN", "POST"])].copy()
    last = int(gg["season"].max())
    seasons = 6
    gg = gg[gg["season"] > last - seasons].reset_index(drop=True)
    b6 = box[box["season"] > last - seasons] if box is not None else None
    upg = gg[gg["game_id"].isin(set(up["game_id"]))]
    avail = npr.tonight_availability(b6, upg, rep)
    h = npr.build(gg, b6, npr.PredictConfig(), upcoming=avail,
                  first_season=last - seasons + 2, verbose=True)
    models = npr.fit_models(h)
    print(f"model fitted on {models.n} games ({time.time() - t0:.0f}s)",
          flush=True)
    tn = h[h["game_id"].isin(set(up["game_id"]))].set_index("game_id")
    pr = npr.predict_frame(models, tn)

    out = []
    for r in up.sort_values("commence").itertuples():
        gid = r.game_id
        f = tn.loc[gid] if gid in tn.index else pd.Series(dtype=float)
        pm = _f(pr.at[gid, "pred_margin"]) if gid in pr.index else None
        pt = _f(pr.at[gid, "pred_total"]) if gid in pr.index else None
        lh = line_history(lines, r.home_team, r.away_team, r.commence)
        if lh.get("bet_spread_now") is not None:
            line = {"spread": lh["bet_spread_now"],
                    "total": lh.get("bet_total_now"), "source": "DK/FD"}
        else:
            esp = _f(r.spread_line)
            line = {"spread": None if esp is None else -esp,
                    "total": _f(r.total_line), "source": "ESPN"}
        line.update(lh)
        b = {"game_id": gid, "home": r.home_team, "away": r.away_team,
             "home_short": short(r.home_team), "away_short": short(r.away_team),
             "commence": str(r.commence),
             "tip_et": (pd.Timestamp(r.commence) - pd.Timedelta(hours=5)
                        ).strftime("%-I:%M %p"),
             "line": line,
             "projection": {"margin": pm, "total": pt,
                            "home_pts": None if pm is None or pt is None
                            else round((pt + pm) / 2, 1),
                            "away_pts": None if pm is None or pt is None
                            else round((pt - pm) / 2, 1),
                            "parts": {"players": _f(f.get("apm_diff")),
                                      "box": _f(f.get("box_diff")),
                                      "possessions": _f(f.get("eff_margin")),
                                      "scores": _f(f.get("model_margin"))}},
             "matchup": {"pace": _f(f.get("pace_hat")),
                         "home_ortg": _f(f.get("home_ortg_hat")),
                         "away_ortg": _f(f.get("away_ortg_hat")),
                         "four_factors": {k: _f(f.get(f"ff_{k}_m"), 3)
                                          for k in ("efg", "tov", "orb",
                                                    "ftr", "3p")},
                         "apm_diff": _f(f.get("apm_diff")),
                         "box_diff": _f(f.get("box_diff"))},
             "form": {"home": team_form(g, r.home_team, r.commence),
                      "away": team_form(g, r.away_team, r.commence)},
             "schedule": {k: _f(f.get(k)) for k in
                          ("home_rest", "away_rest", "home_b2b", "away_b2b",
                           "home_last5", "away_last5", "home_km", "away_km",
                           "home_tz", "away_tz", "rest_diff")},
             "injuries": {"home": injuries_for(rep, r.home_team),
                          "away": injuries_for(rep, r.away_team)}}
        b["schedule"]["away_prev_away"] = _f(f.get("away_b2b_road"))
        b["notes"] = notes(b)
        b["call"] = assess(b)
        out.append(b)
    print(f"breakdown done ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="")
    ap.add_argument("--first", type=int, default=2021)
    ap.add_argument("--out", default="breakdown")
    a, _ = ap.parse_known_args()
    day = (pd.Timestamp(a.date).date() if a.date else
           (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5)).date())
    games = build_day(day, a.first)
    os.makedirs(os.path.join(a.out, "nba"), exist_ok=True)
    md = to_markdown(day, games)
    with open(os.path.join(a.out, "nba", f"{day}.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(a.out, "nba", f"{day}.json"), "w") as fh:
        json.dump({"date": str(day), "games": games}, fh, indent=1,
                  default=str)
    print(md)


if __name__ == "__main__":
    main()
