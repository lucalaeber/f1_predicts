"""Predicts win probabilities for the next Grand Prix on the calendar,
using only data available right now (grid/quali falls back to each
driver's recent average grid position if qualifying hasn't run yet)."""

import argparse

import numpy as np
import pandas as pd

import fastf1

from data_loader import (get_driver_standings_before_round, get_schedule,
                          load_all_data, load_quali_results)
from features import FEATURE_COLUMNS, build_feature_table
from model import predict_win_probabilities, train_ranker


def get_next_round(year: int) -> int:
    sched = get_schedule(year)
    now = pd.Timestamp.utcnow().tz_localize(None)
    dates = pd.to_datetime(sched["EventDate"]).dt.tz_localize(None)
    upcoming = sched[dates >= now]
    if upcoming.empty:
        raise RuntimeError(f"No upcoming rounds found for {year}")
    return int(upcoming.iloc[0]["RoundNumber"])


def get_entry_list(year: int, round_number: int) -> pd.DataFrame:
    """Best-known grid for the upcoming race: real quali results if already
    run, else the prior race's classified entrants as a grid proxy."""
    quali = load_quali_results(year, round_number)
    if quali is not None and not quali.empty:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        info = session.results[["DriverNumber", "Abbreviation", "TeamName"]].drop_duplicates()
        entries = info.merge(quali, on="DriverNumber", how="left")
        entries = entries.rename(columns={"QualiPosition": "GridPosition"})
        entries["grid_is_estimated"] = False
        return entries[["Abbreviation", "TeamName", "GridPosition", "grid_is_estimated"]]

    # fall back: most recent completed race's entry list
    from data_loader import get_completed_rounds, load_race_results
    completed = get_completed_rounds(year)
    last_round = completed[-1] if completed else None
    if last_round is None:
        raise RuntimeError("No completed race to source an entry list from.")
    last_race = load_race_results(year, last_round)
    entries = last_race[["Abbreviation", "TeamName"]].drop_duplicates().reset_index(drop=True)
    entries["GridPosition"] = np.nan  # filled from rolling avg grid at predict time
    entries["grid_is_estimated"] = True
    return entries


def predict_next_race(current_year: int) -> tuple[pd.DataFrame, dict]:
    next_round = get_next_round(current_year)
    sched = get_schedule(current_year)
    event_row = sched[sched["RoundNumber"] == next_round].iloc[0]
    country, event_name = event_row["Country"], event_row["EventName"]

    results = load_all_data(current_year)
    feat_df, state = build_feature_table(results)

    entries = get_entry_list(current_year, next_round)
    standings = get_driver_standings_before_round(current_year, next_round)
    standings_map = standings.set_index("Abbreviation").to_dict("index")

    rows = []
    for _, e in entries.iterrows():
        drv, team = e["Abbreviation"], e["TeamName"]
        grid = e["GridPosition"]
        if pd.isna(grid):
            hist = list(state.driver_recent[drv])[-5:]
            grid = np.mean([h["grid"] for h in hist]) if hist else 10.0
        feat = state.compute_features(drv, team, country, grid, standings_map.get(drv), next_round)
        feat["Abbreviation"] = drv
        feat["TeamName"] = team
        feat["grid_is_estimated"] = bool(e["grid_is_estimated"])
        rows.append(feat)
    race_df = pd.DataFrame(rows)

    model = train_ranker(feat_df)
    race_df["win_prob"] = predict_win_probabilities(model, race_df)
    race_df = race_df.sort_values("win_prob", ascending=False).reset_index(drop=True)
    race_df["win_prob_pct"] = (race_df["win_prob"] * 100).round(1)

    meta = {"year": current_year, "round": next_round, "event_name": event_name, "country": country}
    cols = ["Abbreviation", "TeamName", "GridPosition", "win_prob_pct", "grid_is_estimated"]
    return race_df[cols], meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--out", type=str, default="next_race_predictions.csv")
    args = parser.parse_args()

    preds, meta = predict_next_race(args.year)
    print(f"=== {meta['event_name']} ({meta['country']}) — {meta['year']} Round {meta['round']} ===")
    if preds["grid_is_estimated"].any():
        print("(qualifying not yet available — grid positions are rolling-average estimates)")
    print(preds.to_string(index=False))
    preds.to_csv(args.out, index=False)
