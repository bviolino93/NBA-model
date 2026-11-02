# NBA model

The NFL model's recipe applied to the NBA: start from the sharp market's
fair line, add only adjustments that beat the closing line walk-forward,
and bet stale numbers at DraftKings and FanDuel. Plan and reasoning:
the "NBA betting model: plan" doc in the project.

| File | What it does |
| --- | --- |
| `nba_data.py` | Games, scores and a pregame spread/total per game from ESPN's public scoreboard, cached per day; rest, back-to-back and density features |
| `nba_ratings.py` | Recency-weighted ridge ratings for margin and total, refitted every game day on earlier games only |
| `backtest.py` | Scores the ratings and the schedule factors against the close: slope, ATS at an edge, MAE, split tuning / held out |
| `nba_players.py` | Player box scores from ESPN summaries; each player's expected minutes and value going into a game; missing-player loads per team |
| `selector.py` | Ported from the NFL repo: de-vig, sharp-weighted fair line, empirical outcome model, offer pricing, Kelly sizing, CLV |
| `picks.py` | Daily picks: DraftKings/FanDuel lines that are stale against the sharp market, staked and explained; alerts and a graded record on the `picks` branch |
| `ledger.py` | Every alerted bet, locked at its alerted price, graded from ESPN finals |
| `nba_props.py` | Player prop projections (minutes x rate x pace), walk-forward vs naive baselines |
| `log_lines.py` | Snapshots game lines (and props) from The Odds API into `data/lines/` |

Seasons are labelled by the year they end (2026 = 2025-26), as ESPN does.
`spread_line` is the expected home margin (positive = home favoured).

## Run

    pip install -r requirements.txt
    python -m pytest -q tests
    python nba_data.py --first 2016          # fill the cache, coverage table
    python backtest.py --first 2016 --split 2022
    ODDS_API_KEY=... python log_lines.py [--props]
    python selector.py data/lines/games_2027.csv --bankroll 1000

On GitHub: Actions -> "Backtest vs closing lines" (no secrets), and
"Log NBA lines" (needs secret `ODDS_API_KEY`; set variable `LOG_PROPS=true`
to add the nightly props snapshot).

## Not built yet

1. Timing of injury news (official injury report) to separate news before and after the opener
2. `nba_combined.py`: ridge on result minus line, the NFL combined recipe
3. Weekly review (record and CLV by market), ported from the NFL repo
4. Props projections (minutes x per-minute rates x pace)
