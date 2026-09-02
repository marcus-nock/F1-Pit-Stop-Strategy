"""
safety_car_model.py

Builds a per-track probability model for safety car occurrence by lap
number, then uses it to adjust strategy rankings with an expected-value
layer: some strategies become more attractive once you account for the
chance of a cheap pit stop under a safety car.
"""

import pandas as pd
from data_loader import get_sc_affected_laps

SC_PIT_SAVINGS_SECONDS = 15.0


def build_sc_probability_by_lap(sessions: list) -> pd.Series:
    sc_laps = []

    for session in sessions:
        laps_with_sc = get_sc_affected_laps(session)
        sc_laps.extend(laps_with_sc)

    if not sc_laps:
        print("No SC events found - check track status data or add more sessions")
        return pd.Series(dtype=float)

    max_lap = int(max(sc_laps))
    probs = {}
    for lap in range(1, max_lap + 1):
        occurrences = sum(1 for l in sc_laps if l <= lap)
        probs[lap] = occurrences / len(sessions)

    return pd.Series(probs)


def expected_value_adjustment(strategy: list, sc_prob_by_lap: pd.Series) -> float:
    if sc_prob_by_lap.empty:
        return 0.0

    expected_savings = 0.0
    lap_counter = 0

    for i, (compound, stint_length) in enumerate(strategy):
        lap_counter += stint_length
        if i < len(strategy) - 1:
            prob_at_lap = sc_prob_by_lap.get(lap_counter, 0.0)
            expected_savings += prob_at_lap * SC_PIT_SAVINGS_SECONDS

    return expected_savings


def rank_strategies_with_ev(scored_strategies: list, sc_prob_by_lap: pd.Series) -> list:
    adjusted = []
    for strategy, predicted_time in scored_strategies:
        ev_adjustment = expected_value_adjustment(strategy, sc_prob_by_lap)
        adjusted_time = predicted_time - ev_adjustment
        adjusted.append((strategy, predicted_time, ev_adjustment, adjusted_time))

    adjusted.sort(key=lambda x: x[3])
    return adjusted


if __name__ == "__main__":
    from data_loader import load_race_session, get_clean_laps
    from degradation_model import prepare_stint_data, fit_degradation_by_compound
    from strategy_simulator import rank_strategies

    sessions = [
        load_race_session(2021, "Bahrain", "R"),
        load_race_session(2022, "Bahrain", "R"),
        load_race_session(2023, "Bahrain", "R"),
    ]

    sc_probs = build_sc_probability_by_lap(sessions)

    laps = get_clean_laps(sessions[-1])
    data = prepare_stint_data(laps)
    models = fit_degradation_by_compound(data)

    top_strategies = rank_strategies(models, total_laps=57, compounds=list(models.keys()))
    adjusted_ranking = rank_strategies_with_ev(top_strategies, sc_probs)

    print("\nStrategies ranked with safety car EV adjustment:")
    for strategy, predicted, ev, adjusted in adjusted_ranking:
        stint_desc = " -> ".join(f"{c} x{n}" for c, n in strategy)
        print(f"  {stint_desc} | base: {predicted:.1f}s | EV savings: {ev:.1f}s | adjusted: {adjusted:.1f}s")