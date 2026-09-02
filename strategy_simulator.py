"""
strategy_simulator.py

Enumerates feasible pit strategies for a race and ranks them by
predicted total race time, using the degradation models from
degradation_model.py.

A "strategy" here is represented as a list of (compound, stint_length)
tuples, e.g. [("MEDIUM", 20), ("HARD", 37)] for a 1-stop strategy
on a 57-lap race.
"""

from itertools import product
from degradation_model import predict_stint_total

PIT_STOP_LOSS_SECONDS = 22.0


def generate_two_compound_strategies(total_laps: int, compounds: list, min_stint: int = 5) -> list:
    strategies = []

    for c1, c2 in product(compounds, repeat=2):
        if c1 == c2:
            continue
        for stint1_len in range(min_stint, total_laps - min_stint + 1):
            stint2_len = total_laps - stint1_len
            if stint2_len < min_stint:
                continue
            strategies.append([(c1, stint1_len), (c2, stint2_len)])

    return strategies


def generate_three_stint_strategies(total_laps: int, compounds: list, min_stint: int = 5, step: int = 4) -> list:
    strategies = []

    for c1, c2, c3 in product(compounds, repeat=3):
        if len({c1, c2, c3}) < 2:
            continue

        for stint1_len in range(min_stint, total_laps - 2 * min_stint + 1, step):
            for stint2_len in range(min_stint, total_laps - stint1_len - min_stint + 1, step):
                stint3_len = total_laps - stint1_len - stint2_len
                if stint3_len < min_stint:
                    continue
                strategies.append([(c1, stint1_len), (c2, stint2_len), (c3, stint3_len)])

    return strategies


def evaluate_strategy(models: dict, strategy: list) -> float:
    total_time = 0.0
    current_lap = 1

    for i, (compound, stint_length) in enumerate(strategy):
        total_time += predict_stint_total(models, compound, stint_length, start_lap=current_lap)
        current_lap += stint_length
        if i > 0:
            total_time += PIT_STOP_LOSS_SECONDS

    return total_time


def is_within_observed_range(models: dict, strategy: list, buffer_laps: int = 3) -> bool:
    """
    Rejects strategies that ask a compound to run longer than it was
    ever observed running in real data (plus a small buffer). Without
    this, linear regression will extrapolate way past its training
    range and produce nonsense predictions.
    """
    for compound, stint_length in strategy:
        max_observed = getattr(models[compound], "max_observed_age", None)
        if max_observed is not None and stint_length > max_observed + buffer_laps:
            return False
    return True


def rank_strategies(models: dict, total_laps: int, compounds: list, top_n: int = 5,
                     include_two_stop: bool = True) -> list:
    strategies = generate_two_compound_strategies(total_laps, compounds)

    if include_two_stop:
        strategies += generate_three_stint_strategies(total_laps, compounds)

    strategies = [s for s in strategies if is_within_observed_range(models, s)]

    if not strategies:
        print("Warning: no strategies fit within observed tyre-age ranges - "
              "falling back to unfiltered strategies (predictions may be unreliable)")
        strategies = generate_two_compound_strategies(total_laps, compounds)
        if include_two_stop:
            strategies += generate_three_stint_strategies(total_laps, compounds)

    scored = [(s, evaluate_strategy(models, s)) for s in strategies]
    scored.sort(key=lambda x: x[1])
    return scored[:top_n]


if __name__ == "__main__":
    from data_loader import load_race_session, get_clean_laps
    from degradation_model import prepare_stint_data, fit_degradation_by_compound

    session = load_race_session(2023, "Bahrain", "R")
    laps = get_clean_laps(session)
    data = prepare_stint_data(laps)
    models = fit_degradation_by_compound(data)

    available_compounds = list(models.keys())
    total_laps = 57

    top_strategies = rank_strategies(models, total_laps, available_compounds)

    print("\nTop predicted strategies:")
    for strategy, predicted_time in top_strategies:
        stint_desc = " -> ".join(f"{c} x{n}" for c, n in strategy)
        print(f"  {stint_desc}  |  predicted total: {predicted_time:.1f}s")