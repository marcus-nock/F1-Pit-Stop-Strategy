"""
degradation_model.py

Fits a lap-time-vs-tire-age model per compound, per track.

The core idea: as tire age (laps on that set) increases, lap time
increases (tires degrade). We fit this relationship so we can predict
lap time for ANY stint length/compound combo later in the simulator.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_stint_data(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns we need for degradation modeling, and drop
    any remaining rows with missing tire age or compound info.
    """
    cols = ["Driver", "LapNumber", "Compound", "TyreLife", "LapTimeSeconds", "Stint"]
    data = laps[cols].dropna()
    return data


def fit_degradation_by_compound(data: pd.DataFrame, min_samples: int = 20) -> dict:
    """
    Fits a separate regression model per tire compound.

    Model: LapTimeSeconds ~ TyreLife + LapNumber

    Why both features: LapNumber acts as a rough proxy for fuel load
    (cars get lighter and faster as the race goes on, since fuel burns
    off roughly linearly). Without it, fuel-driven speedup and tire-driven
    slowdown get tangled together in a single coefficient, and TyreLife's
    effect can look artificially flat or even negative.

    With two features, model.coef_ has two values:
      coef_[0] = effect of tyre age (holding lap number constant)
      coef_[1] = effect of lap number / fuel burn (holding tyre age constant)

    min_samples: compounds with fewer laps than this are skipped and
    reported, since a regression on a handful of points isn't trustworthy.

    Returns: dict like {"SOFT": model, "MEDIUM": model, "HARD": model}
    """
    models = {}

    for compound in data["Compound"].unique():
        subset = data[data["Compound"] == compound]

        if len(subset) < min_samples:
            print(f"{compound}: skipped - only {len(subset)} laps, "
                  f"need at least {min_samples} for a reliable fit")
            continue

        X = subset[["TyreLife", "LapNumber"]].values
        y = subset["LapTimeSeconds"].values

        model = LinearRegression()
        model.fit(X, y)

        # Attach the max tyre age actually observed for this compound.
        # This guards against the simulator recommending stint lengths
        # that extrapolate way past real data (linear regression will
        # happily produce nonsense predictions otherwise).
        model.max_observed_age = int(subset["TyreLife"].max())

        models[compound] = model

        tyre_coef, lap_coef = model.coef_
        print(f"{compound}: lap_time = {model.intercept_:.3f} "
              f"+ {tyre_coef:.4f}*tyre_age + {lap_coef:.4f}*lap_number  (n={len(subset)})")

    return models


def predict_lap_time(models: dict, compound: str, tyre_age: int, lap_number: int) -> float:
    """
    Predict lap time in seconds for a given compound, at a given tire
    age AND a given point in the race (lap_number matters now too,
    since it captures the fuel-load effect).
    """
    if compound not in models:
        raise ValueError(f"No fitted model for compound: {compound}")
    return float(models[compound].predict([[tyre_age, lap_number]])[0])


def predict_stint_total(models: dict, compound: str, stint_length: int,
                         start_age: int = 0, start_lap: int = 1) -> float:
    """
    Predict total time (seconds) for an entire stint of a given length,
    starting from a given tire age (0 = fresh tires) and a given lap
    number in the race (needed now to account for fuel burn-off).
    """
    total = 0.0
    for lap_in_stint in range(stint_length):
        age = start_age + lap_in_stint
        lap_number = start_lap + lap_in_stint
        total += predict_lap_time(models, compound, age, lap_number)
    return total


if __name__ == "__main__":
    # Example usage once you've got real data flowing in from data_loader.py
    from data_loader import load_race_session, get_clean_laps

    session = load_race_session(2023, "Spain", "R")
    laps = get_clean_laps(session)
    data = prepare_stint_data(laps)
    models = fit_degradation_by_compound(data)

    for compound in models:
        subset = data[data["Compound"] == compound]
        print(f"{compound}: tyre_age range in training data = "
              f"{subset['TyreLife'].min()} to {subset['TyreLife'].max()} laps")

    # sanity check: predict a 20-lap medium stint
    if "MEDIUM" in models:
        total = predict_stint_total(models, "MEDIUM", stint_length=20)
        print(f"\nPredicted total time for a 20-lap MEDIUM stint: {total:.1f}s")
