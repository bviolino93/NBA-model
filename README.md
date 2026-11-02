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
| `selector.py` | Ported from the NFL repo: de-vig, sharp-weighted fair line, empirical outcome model, offer pricing, Kelly sizing, CLV |
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

1. Injury loads in minutes (official injury report + recent minutes)
2. `nba_combined.py`: ridge on result minus line, the NFL combined recipe
3. `picks.py` + picks/ledger/review workflows, ported from the NFL repo
4. Props projections (minutes x per-minute rates x pace)
