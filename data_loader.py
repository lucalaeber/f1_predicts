"""Fetches race results, qualifying grids, and championship standings via FastF1."""

from pathlib import Path

import fastf1
import pandas as pd
from fastf1.ergast import Ergast

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

_ergast = Ergast()

RESULT_COLUMNS = [
    "DriverNumber", "Abbreviation", "FullName", "TeamName",
    "GridPosition", "Position", "ClassifiedPosition", "Points", "Status", "Laps",
]


def get_schedule(year: int) -> pd.DataFrame:
    return fastf1.get_event_schedule(year, include_testing=False)


def get_completed_rounds(year: int, as_of: pd.Timestamp | None = None) -> list[int]:
    """Round numbers whose race session is already in the past."""
    sched = get_schedule(year)
    now = as_of or pd.Timestamp.utcnow().tz_localize(None)
    dates = pd.to_datetime(sched["EventDate"]).dt.tz_localize(None)
    completed = sched[dates < now]
    return sorted(int(r) for r in completed["RoundNumber"].tolist())


def load_race_results(year: int, round_number: int) -> pd.DataFrame | None:
    """Full classified race results for one round, or None if unavailable."""
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception:
        return None
    if session.results is None or session.results.empty:
        return None

    res = session.results.copy()
    res = res[RESULT_COLUMNS]
    res["Year"] = year
    res["Round"] = round_number
    res["EventName"] = session.event["EventName"]
    res["EventDate"] = session.event["EventDate"]
    res["Country"] = session.event["Country"]
    res["DNF"] = ~res["Status"].astype(str).str.contains("Finished|Lap", na=False)
    return res.reset_index(drop=True)


def load_quali_results(year: int, round_number: int) -> pd.DataFrame | None:
    """Qualifying classification (Q1/Q2/Q3 elimination order), or None if unavailable."""
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception:
        return None
    if session.results is None or session.results.empty:
        return None
    q = session.results.copy()[["DriverNumber", "Position"]]
    q = q.rename(columns={"Position": "QualiPosition"})
    q["Year"] = year
    q["Round"] = round_number
    return q.reset_index(drop=True)


def load_season_results(year: int, rounds: list[int] | None = None) -> pd.DataFrame:
    """Race + qualifying results for every completed round of a season, concatenated."""
    if rounds is None:
        rounds = get_completed_rounds(year)
    frames = []
    for rnd in rounds:
        race = load_race_results(year, rnd)
        if race is None:
            continue
        quali = load_quali_results(year, rnd)
        if quali is not None:
            race = race.merge(quali, on=["DriverNumber", "Year", "Round"], how="left")
        else:
            race["QualiPosition"] = race["GridPosition"]
        race["QualiPosition"] = race["QualiPosition"].fillna(race["GridPosition"])
        frames.append(race)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_driver_standings_before_round(year: int, round_number: int) -> pd.DataFrame:
    """Driver championship standings known BEFORE `round_number` starts.

    Round 1 (or earlier) falls back to the previous season's final standings,
    which is all a model could know before a season has begun.
    """
    try:
        if round_number <= 1:
            resp = _ergast.get_driver_standings(season=year - 1)
        else:
            resp = _ergast.get_driver_standings(season=year, round=round_number - 1)
        df = resp.content[-1] if resp.content else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=["Abbreviation", "StandingPosition", "StandingPoints", "StandingWins"])

    df = df.rename(columns={
        "driverCode": "Abbreviation",
        "position": "StandingPosition",
        "points": "StandingPoints",
        "wins": "StandingWins",
    })
    return df[["Abbreviation", "StandingPosition", "StandingPoints", "StandingWins"]]


def load_all_data(current_year: int) -> pd.DataFrame:
    """Prior season (full) + current season (completed rounds), chronologically sorted."""
    prior = load_season_results(current_year - 1)
    current = load_season_results(current_year)
    all_results = pd.concat([prior, current], ignore_index=True)
    all_results = all_results.sort_values(["Year", "Round"]).reset_index(drop=True)

    standings_cache: dict[tuple[int, int], pd.DataFrame] = {}
    merged_frames = []
    for (yr, rnd), grp in all_results.groupby(["Year", "Round"], sort=False):
        key = (yr, rnd)
        if key not in standings_cache:
            standings_cache[key] = get_driver_standings_before_round(yr, rnd)
        standings = standings_cache[key]
        grp = grp.merge(standings, on="Abbreviation", how="left")
        merged_frames.append(grp)

    out = pd.concat(merged_frames, ignore_index=True)
    out = out.sort_values(["Year", "Round"]).reset_index(drop=True)
    return out
