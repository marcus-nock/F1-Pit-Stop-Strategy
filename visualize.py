"""
visualize.py

Generates the core plots for the project:
1. Degradation curves per compound (lap time vs tyre age)
2. A bar chart comparing the model's recommended strategy against
   what actually happened in the real race

Run directly to generate and save both plots as PNG files.
"""

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_race_session, get_clean_laps
from degradation_model import prepare_stint_data, fit_degradation_by_compound


def plot_degradation_curves(models: dict, data, track_name: str, save_path: str = None):
    """
    Plots the fitted degradation line per compound, alongside the raw
    scatter of real laps, so you can visually judge model fit.

    Held at a fixed lap_number (mid-race) so the plot isolates the
    tyre-age effect cleanly, since lap_number is a second variable now.
    """
    compound_colors = {"SOFT": "red", "MEDIUM": "gold", "HARD": "gray"}
    mid_race_lap = int(data["LapNumber"].median())

    plt.figure(figsize=(9, 6))

    for compound, model in models.items():
        subset = data[data["Compound"] == compound]
        color = compound_colors.get(compound, "blue")

        plt.scatter(subset["TyreLife"], subset["LapTimeSeconds"],
                    alpha=0.2, color=color, label=f"{compound} (raw laps)")

        max_age = getattr(model, "max_observed_age", int(subset["TyreLife"].max()))
        age_range = np.arange(0, max_age + 1)
        predicted = [model.predict([[age, mid_race_lap]])[0] for age in age_range]
        plt.plot(age_range, predicted, color=color, linewidth=2.5,
                  label=f"{compound} (fitted, at lap {mid_race_lap})")

    plt.xlabel("Tyre Age (laps)")
    plt.ylabel("Lap Time (seconds)")
    plt.title(f"Tire Degradation by Compound - {track_name}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_strategy_comparison(model_strategy: list, model_time: float,
                              actual_strategy: list, actual_time: float,
                              track_name: str, save_path: str = None):
    """
    Simple bar chart: model's predicted time vs the actual strategy's
    predicted time, using the model's own time function for both.
    """
    model_desc = " -> ".join(f"{c}x{n}" for c, n in model_strategy)
    actual_desc = " -> ".join(f"{c}x{n}" for c, n in actual_strategy)

    labels = [f"Model's pick:\n{model_desc}", f"What actually happened:\n{actual_desc}"]
    times = [model_time, actual_time]
    colors = ["#2E86AB", "#A23B72"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, times, color=colors)
    plt.ylabel("Predicted Total Race Time (seconds)")
    plt.title(f"Model Recommendation vs Reality - {track_name}")
    plt.figtext(0.5, 0.01, "Note: y-axis is zoomed in to make the time gap visible - it does not start at 0",
                ha="center", fontsize=8, style="italic")

    # Zoom the y-axis to the range that actually matters. Without this,
    # a real but small gap (a few seconds on a ~5000+ second scale) is
    # invisible - both bars look identical even though they aren't.
    min_time, max_time = min(times), max(times)
    padding = max((max_time - min_time) * 2, 5)
    plt.ylim(min_time - padding, max_time + padding)

    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + padding * 0.1,
                  f"{time:.1f}s", ha="center", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    from strategy_simulator import rank_strategies, evaluate_strategy
    from validate_strategy import get_winner_code, get_actual_strategy

    YEAR = 2023
    GP = "Bahrain"
    TOTAL_LAPS = 57
    TRACK_LABEL = "Bahrain 2023"

    session = load_race_session(YEAR, GP, "R")
    laps = get_clean_laps(session)
    data = prepare_stint_data(laps)
    models = fit_degradation_by_compound(data)

    plot_degradation_curves(models, data, TRACK_LABEL, save_path="outputs/degradation_curves.png")

    top_strategies = rank_strategies(models, TOTAL_LAPS, list(models.keys()), top_n=1)
    best_strategy, best_time = top_strategies[0]

    winner_code = get_winner_code(session)
    actual_strategy = get_actual_strategy(session, winner_code)
    actual_time = evaluate_strategy(models, actual_strategy)

    plot_strategy_comparison(best_strategy, best_time, actual_strategy, actual_time,
                              TRACK_LABEL, save_path="outputs/strategy_comparison.png")