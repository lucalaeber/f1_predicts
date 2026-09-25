"""Expanding-window backtest: for each race of the current season, train only
on data known before that race started, then predict its winner."""

import argparse

import numpy as np
import pandas as pd

from data_loader import load_all_data
from features import build_feature_table
from model import predict_win_probabilities, train_ranker


def run_backtest(current_year: int, feat_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if feat_df is None:
        results = load_all_data(current_year)
        feat_df, _ = build_feature_table(results)

    current_rounds = sorted(feat_df.loc[feat_df["Year"] == current_year, "Round"].unique())

    records = []
    for rnd in current_rounds:
        train_df = feat_df[(feat_df["Year"] < current_year) |
                            ((feat_df["Year"] == current_year) & (feat_df["Round"] < rnd))]
        race_df = feat_df[(feat_df["Year"] == current_year) & (feat_df["Round"] == rnd)].copy()
        if train_df.empty or race_df.empty:
            continue

        model = train_ranker(train_df)
        race_df["win_prob"] = predict_win_probabilities(model, race_df)
        race_df = race_df.sort_values("win_prob", ascending=False).reset_index(drop=True)

        actual_winner_row = race_df[race_df["Position"] == 1]
        if actual_winner_row.empty:
            continue
        actual_winner = actual_winner_row.iloc[0]["Abbreviation"]
        predicted_winner = race_df.iloc[0]["Abbreviation"]
        predicted_prob = race_df.iloc[0]["win_prob"]
        actual_rank = int(race_df.index[race_df["Abbreviation"] == actual_winner][0]) + 1
        prob_on_actual = float(race_df.loc[race_df["Abbreviation"] == actual_winner, "win_prob"].iloc[0])

        records.append({
            "Year": current_year,
            "Round": rnd,
            "EventName": race_df.iloc[0]["EventName"],
            "predicted_winner": predicted_winner,
            "predicted_prob": round(float(predicted_prob), 3),
            "actual_winner": actual_winner,
            "correct": predicted_winner == actual_winner,
            "top3_hit": actual_rank <= 3,
            "actual_winner_rank": actual_rank,
            "prob_assigned_to_actual_winner": round(prob_on_actual, 3),
        })

    return pd.DataFrame(records)


def summarize(backtest_df: pd.DataFrame) -> dict:
    if backtest_df.empty:
        return {}
    eps = 1e-15
    probs = backtest_df["prob_assigned_to_actual_winner"].clip(eps, 1 - eps)
    return {
        "races_evaluated": len(backtest_df),
        "top1_accuracy": round(backtest_df["correct"].mean(), 3),
        "top3_accuracy": round(backtest_df["top3_hit"].mean(), 3),
        "mean_rank_of_actual_winner": round(backtest_df["actual_winner_rank"].mean(), 2),
        "mean_prob_assigned_to_actual_winner": round(backtest_df["prob_assigned_to_actual_winner"].mean(), 3),
        "log_loss_on_actual_winner": round(float(-np.mean(np.log(probs))), 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--out", type=str, default="backtest_results.csv")
    args = parser.parse_args()

    bt = run_backtest(args.year)
    bt.to_csv(args.out, index=False)
    print(bt.to_string(index=False))
    print("\n=== Backtest metrics ===")
    for k, v in summarize(bt).items():
        print(f"{k}: {v}")
