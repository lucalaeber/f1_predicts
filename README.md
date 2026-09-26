# F1 Race Winner Predictor

Predicts Formula 1 race winners from grid position, championship standings,
and rolling driver/team form, using [FastF1](https://docs.fastf1.dev/) for
data and a LightGBM ranker for win probabilities.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

LightGBM needs OpenMP on macOS: `brew install libomp` if `import lightgbm` fails.

## Modules

- **`data_loader.py`** — pulls race results, qualifying grids, and driver
  championship standings (via FastF1's Ergast wrapper) for a given season.
  `load_all_data(year)` returns the prior season (full) + current season
  (completed rounds so far), chronologically sorted.
- **`features.py`** — turns raw results into a leak-free feature table. Every
  feature for a race (grid, championship position, rolling form, per-track
  history, team form) is computed from a `HistoryState` that is only updated
  *after* that race is featurized, so nothing about a race leaks into its own
  prediction. The same state can be queried for a race that hasn't happened
  yet.
- **`model.py`** — trains an `LGBMRanker` (`lambdarank`) with one ranking
  group per race, label = finishing-position relevance. Raw scores are
  softmaxed per race into win probabilities that sum to 1.
- **`backtest.py`** — expanding-window backtest: for each race of the current
  season, retrains on everything known before that race (prior season +
  earlier current-season races) and predicts it. Reports top-1/top-3
  accuracy, mean rank of the actual winner, and log loss.
- **`predict.py`** — trains on all available data and predicts the next race
  on the calendar. If qualifying hasn't run yet, grid position is estimated
  from each driver's recent average grid.

## Usage

```bash
python3 backtest.py --year 2026 --out backtest_results.csv
python3 predict.py --year 2026 --out next_race_predictions.csv
```

## Report

[`report.html`](report.html) is a self-contained (no build step, no external
data fetch) dashboard built from the two CSVs above — backtest accuracy, the
race-by-race predicted-vs-actual table, and win probabilities for the next
Grand Prix. Open it directly in a browser, or enable **GitHub Pages** (repo
Settings → Pages → deploy from this branch) to get a shareable link, since
GitHub's own file viewer shows `.html` files as source, not rendered.

The data inside `report.html` is a snapshot from whenever it was last
generated — it doesn't re-fetch anything. After rerunning `backtest.py` /
`predict.py`, update the `bakuDrivers` and `backtest` arrays near the bottom
of the file with the new CSV rows (or regenerate the whole file with your own
script) to keep it in sync.

## Current results (2026 season, run 2026-09-26)

Backtesting all 14 completed 2026 races (expanding window, retrained per race):

| Metric | Value |
|---|---|
| Top-1 accuracy | 50.0% |
| Top-3 accuracy | 92.9% |
| Mean rank of actual winner | 1.86 |
| Mean probability assigned to actual winner | 0.501 |
| Log loss (actual winner) | 1.293 |

See `backtest_results.csv` for the full predicted-vs-actual table.

Predicted win probabilities for the **Azerbaijan Grand Prix (Baku), Round 15**,
using Saturday's actual qualifying grid:

| Driver | Team | Grid | Win probability |
|---|---|---|---|
| RUS | Mercedes | P1 | 49.8% |
| LEC | Ferrari | P2 | 26.4% |
| ANT | Mercedes | P16 | 10.8% |
| HAD | Red Bull Racing | P4 | 3.9% |
| PIA | McLaren | P3 | 3.5% |
| NOR | McLaren | P5 | 2.9% |
| HAM | Ferrari | P6 | 1.9% |

Full table in `next_race_predictions.csv`. An earlier run made before
qualifying had ANT as the 28% favorite from an estimated grid — his actual
P16 qualifying result dropped him to 10.8%, and pole-sitter RUS jumped from
5.2% to 49.8%, a reminder of how much weight the model puts on grid
position.

## Notes / limitations

- No weather, tyre-strategy, or track-specific pace data — only results-derived
  features (grid, standings, rolling form, per-circuit history).
- Small dataset (~1.5 seasons): the ranker is prone to overfitting to a few
  dominant drivers early in a season; accuracy should be read as directional,
  not calibrated.
- `career_avg_finish_at_track` is sparse for tracks a driver has visited 0–1
  times (rookies, new circuits) — LightGBM handles the resulting NaNs natively.
