"""Builds a leak-free feature table: every feature for a given race uses only
information available before that race's lights-out."""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

SHORT_WINDOW = 5
LONG_WINDOW = 10

FEATURE_COLUMNS = [
    "GridPosition",
    "StandingPosition", "StandingPoints", "StandingWins",
    "roll_avg_finish", "roll_avg_grid", "roll_avg_points",
    "roll_podium_rate", "roll_dnf_rate", "roll_win_rate",
    "career_avg_finish_at_track", "career_starts_at_track",
    "team_roll_avg_finish", "team_roll_avg_points", "team_roll_podium_rate",
    "driver_races_completed", "RoundNumber",
]


def _safe_mean(values):
    return float(np.mean(values)) if values else np.nan


class HistoryState:
    """Running per-driver / per-team history used to compute causal features.
    Replaying it over past results and then querying it (without updating)
    gives identical features for a race that hasn't happened yet.
    """

    def __init__(self):
        self.driver_recent = defaultdict(lambda: deque(maxlen=LONG_WINDOW))
        self.driver_track_history = defaultdict(list)
        self.driver_race_count = defaultdict(int)
        self.team_recent = defaultdict(lambda: deque(maxlen=LONG_WINDOW))

    def compute_features(self, drv, team, country, grid, standings_row, round_number) -> dict:
        drv_hist = list(self.driver_recent[drv])
        short_hist = drv_hist[-SHORT_WINDOW:]
        team_hist = list(self.team_recent[team])
        short_team_hist = team_hist[-SHORT_WINDOW:]
        track_key = (drv, country)

        standings_row = standings_row or {}
        return {
            "GridPosition": grid,
            "StandingPosition": standings_row.get("StandingPosition", np.nan),
            "StandingPoints": standings_row.get("StandingPoints", np.nan),
            "StandingWins": standings_row.get("StandingWins", np.nan),
            "roll_avg_finish": _safe_mean([h["finish"] for h in short_hist]),
            "roll_avg_grid": _safe_mean([h["grid"] for h in short_hist]),
            "roll_avg_points": _safe_mean([h["points"] for h in short_hist]),
            "roll_podium_rate": _safe_mean([h["podium"] for h in drv_hist]),
            "roll_dnf_rate": _safe_mean([h["dnf"] for h in drv_hist]),
            "roll_win_rate": _safe_mean([h["win"] for h in drv_hist]),
            "career_avg_finish_at_track": _safe_mean(self.driver_track_history[track_key]),
            "career_starts_at_track": len(self.driver_track_history[track_key]),
            "team_roll_avg_finish": _safe_mean([h["finish"] for h in short_team_hist]),
            "team_roll_avg_points": _safe_mean([h["points"] for h in short_team_hist]),
            "team_roll_podium_rate": _safe_mean([h["podium"] for h in team_hist]),
            "driver_races_completed": self.driver_race_count[drv],
            "RoundNumber": round_number,
        }

    def update(self, drv, team, country, grid, finish, points, dnf):
        podium = 1 if finish <= 3 else 0
        win = 1 if finish == 1 else 0
        self.driver_recent[drv].append({
            "finish": finish, "grid": grid, "points": points,
            "podium": podium, "dnf": int(dnf), "win": win,
        })
        self.driver_track_history[(drv, country)].append(finish)
        self.driver_race_count[drv] += 1
        self.team_recent[team].append({"finish": finish, "points": points, "podium": podium})


def build_feature_table(results: pd.DataFrame) -> tuple[pd.DataFrame, HistoryState]:
    """`results` must be sorted chronologically (Year, Round) and contain the
    columns produced by data_loader.load_all_data. Returns one row per
    driver-race with causal features plus the actual outcome columns
    (Position, DNF, Points) for training/evaluation, and the final
    HistoryState (usable to compute features for the next, unplayed race).
    """
    results = results.sort_values(["Year", "Round"]).reset_index(drop=True)
    state = HistoryState()

    rows = []
    for (year, rnd), event in results.groupby(["Year", "Round"], sort=False):
        for _, r in event.iterrows():
            drv, team, country = r["Abbreviation"], r["TeamName"], r["Country"]
            standings_row = {
                "StandingPosition": r.get("StandingPosition", np.nan),
                "StandingPoints": r.get("StandingPoints", np.nan),
                "StandingWins": r.get("StandingWins", np.nan),
            }
            feat = state.compute_features(drv, team, country, r["GridPosition"], standings_row, rnd)
            feat.update({
                "Year": year, "Round": rnd, "EventName": r["EventName"], "Country": country,
                "Abbreviation": drv, "TeamName": team,
                "Position": r["Position"], "Points": r["Points"], "DNF": bool(r["DNF"]),
            })
            rows.append(feat)
            state.update(drv, team, country, r["GridPosition"], r["Position"], r["Points"], r["DNF"])

    feat_df = pd.DataFrame(rows)
    n_drivers = feat_df.groupby(["Year", "Round"])["Abbreviation"].transform("count")
    feat_df["relevance"] = (n_drivers - feat_df["Position"] + 1).clip(lower=0)
    feat_df["IsWinner"] = (feat_df["Position"] == 1).astype(int)
    return feat_df, state
