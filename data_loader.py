"""
data_loader.py

Handles all interaction with FastF1: loading sessions, caching,
and pulling out the raw lap/tire/track-status data we need.

Run this file directly to sanity-check that a session loads correctly:
    python src/data_loader.py
"""

import os
import fastf1
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "fastf1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_session(year: int, gp: str, session_type: str = "R"):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def get_clean_laps(session) -> pd.DataFrame:
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()]
    laps = laps[(laps["PitOutTime"].isna()) & (laps["PitInTime"].isna())]

    if "TrackStatus" in laps.columns:
        laps = laps[laps["TrackStatus"] == "1"]

    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
    return laps


def get_sc_affected_laps(session) -> list:
    """
    Return the list of lap numbers where a safety car or VSC was active,
    based on each lap's TrackStatus code.

    session.laps has a TrackStatus column per lap (a string that can
    contain multiple codes, e.g. "14" if two statuses applied during
    that lap). Status codes: '4' = Safety Car, '6'/'7' = VSC.
    """
    laps = session.laps.copy()
    laps = laps[laps["TrackStatus"].notna()]

    sc_mask = laps["TrackStatus"].apply(
        lambda status: any(code in str(status) for code in ["4", "6", "7"])
    )
    sc_laps = laps[sc_mask]["LapNumber"].dropna().unique().tolist()
    return sc_laps


if __name__ == "__main__":
    session = load_race_session(2023, "Bahrain", "R")
    laps = get_clean_laps(session)
    print(f"Loaded {len(laps)} clean laps")
    print(laps[["Driver", "LapNumber", "Compound", "TyreLife", "LapTimeSeconds"]].head(10))