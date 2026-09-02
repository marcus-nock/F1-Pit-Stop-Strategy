"""
validate_strategy.py

Compares your model's top recommended strategy against what actually
happened in a real race - specifically, what strategy the race winner
actually ran. This is the step that tells you whether your model is
actually good, not just running without errors.
"""

import pandas as pd
from data_loader import load_race_session, get_clean_laps
from degradation_model import prepare_stint_data, fit_degradation_by_compound
from strategy_simulator import rank_strategies


def get_actual_strategy(session, driver_code: str) -> list:
    laps = session.laps.copy()
    driver_laps = laps[laps["Driver"] == driver_code]

    if driver_laps.empty:
        raise ValueError(f"No laps found for driver code: {driver_code}")

    strategy = []
    for stint_number in sorted(driver_laps["Stint"].dropna().unique()):
        stint_laps = driver_laps[driver_laps["Stint"] == stint_number]
        compound = stint_laps["Compound"].iloc[0]
        stint_length = len(stint_laps)
        strategy.append((compound, stint_length))

    return strategy


def verify_real_pit_stops(session, driver_code: str) -> list:
    laps = session.laps.copy()
    driver_laps = laps[laps["Driver"] == driver_code].sort_values("LapNumber")

    real_stops = driver_laps[driver_laps["PitInTime"].notna()]
    stop_laps = real_stops["LapNumber"].tolist()

    print(f"Real pit stops for {driver_code} (laps with recorded PitInTime): {stop_laps}")
    return stop_laps


def get_winner_code(session) -> str:
    results = session.results
    winner_row = results[results["Position"] == 1.0]
    return winner_row["Abbreviation"].iloc[0]


def compare_strategies(models: dict, model_strategy: list, actual_strategy: list, model_time: float):
    """
    Prints a side-by-side comparison, including a numeric prediction for
    BOTH strategies, so you see how many seconds apart they actually are.
    """
    from strategy_simulator import evaluate_strategy

    model_desc = " -> ".join(f"{c} x{n}" for c, n in model_strategy)
    actual_desc = " -> ".join(f"{c} x{n}" for c, n in actual_strategy)
    actual_predicted_time = evaluate_strategy(models, actual_strategy)

    print(f"Model's top recommendation: {model_desc}  |  predicted: {model_time:.1f}s")
    print(f"What actually happened:     {actual_desc}  |  predicted: {actual_predicted_time:.1f}s")
    print(f"Difference: {abs(model_time - actual_predicted_time):.1f}s")

    model_compounds = {c for c, _ in model_strategy}
    actual_compounds = {c for c, _ in actual_strategy}

    if model_compounds != actual_compounds:
        print(f"Note: model used {model_compounds}, reality used {actual_compounds} "
              "(check degradation_model output for skipped compounds)")


if __name__ == "__main__":
    YEAR = 2023
    GP = "Spain"
    TOTAL_LAPS = 66

    session = load_race_session(YEAR, GP, "R")
    laps = get_clean_laps(session)
    data = prepare_stint_data(laps)
    models = fit_degradation_by_compound(data)

    top_strategies = rank_strategies(models, TOTAL_LAPS, list(models.keys()), top_n=1)
    best_model_strategy, best_model_time = top_strategies[0]

    winner_code = get_winner_code(session)
    actual_strategy = get_actual_strategy(session, winner_code)
    verify_real_pit_stops(session, winner_code)

    print(f"\n{GP} {YEAR} - Race Winner: {winner_code}\n")
    compare_strategies(models, best_model_strategy, actual_strategy, best_model_time)