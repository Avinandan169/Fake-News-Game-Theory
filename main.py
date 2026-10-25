"""Fake News Propagation and Platform Moderation.

This beginner-friendly project has two independent parts:
1. a small game-theory model of a creator and a platform;
2. a simplified simulation of how fake news can spread under moderation.

The simulation is illustrative only. It is not a prediction about any real
platform or population.
"""

from __future__ import annotations

import argparse
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

CREATOR_STRATEGIES = ("Spread fake news", "Share responsible content")
PLATFORM_STRATEGIES = ("Weak moderation", "Strong moderation")

# Rows are creator strategies; columns are platform strategies.
# Each cell is (creator payoff, platform payoff).
PAYOFF_MATRIX = np.array(
    [
        [(4, -4), (-3, 3)],
        [(2, 2), (1, 1)],
    ],
    dtype=int,
)


@dataclass(frozen=True)
class ModerationStrategy:
    """A named moderation level and its effect on the sharing probability."""

    name: str
    sharing_multiplier: float


MODERATION_STRATEGIES = (
    ModerationStrategy("No moderation", 1.00),
    ModerationStrategy("Moderate moderation", 0.35),
    ModerationStrategy("Strict moderation", 0.10),
)


@dataclass(frozen=True)
class SimulationSettings:
    """Inputs used by every moderation scenario in a comparison."""

    population: int = 1000
    initially_exposed: int = 10
    rounds: int = 12
    sharing_probability: float = 0.35
    seed: int = 42
    average_contacts: int = 5


@dataclass(frozen=True)
class SimulationResult:
    """The cumulative exposure after each round for one strategy."""

    strategy_name: str
    history: tuple[int, ...]

    @property
    def final_exposure(self) -> int:
        return self.history[-1]


def validate_settings(settings: SimulationSettings) -> None:
    """Raise ValueError when a simulation input is outside its safe range."""
    if settings.population <= 0:
        raise ValueError("Population must be greater than 0.")
    if not 0 <= settings.initially_exposed <= settings.population:
        raise ValueError("Initially exposed users must be between 0 and the population.")
    if settings.rounds < 0:
        raise ValueError("Number of rounds cannot be negative.")
    if not 0 <= settings.sharing_probability <= 1:
        raise ValueError("Sharing probability must be between 0% and 100%.")
    if settings.average_contacts < 0:
        raise ValueError("Average contacts cannot be negative.")


def print_payoff_matrix() -> None:
    """Print the matrix in a presentation-friendly terminal format."""
    print("\nGAME THEORY MODEL")
    print("Creator payoff, Platform payoff")
    print(f"{'':30}{PLATFORM_STRATEGIES[0]:>23}{PLATFORM_STRATEGIES[1]:>23}")
    for row, creator_strategy in enumerate(CREATOR_STRATEGIES):
        weak = tuple(int(value) for value in PAYOFF_MATRIX[row, 0])
        strong = tuple(int(value) for value in PAYOFF_MATRIX[row, 1])
        print(f"{creator_strategy:30}{str(weak):>23}{str(strong):>23}")


def mixed_nash_equilibrium(matrix: np.ndarray = PAYOFF_MATRIX) -> tuple[float, float]:
    """Return (P(creator spreads fake news), P(platform uses weak moderation)).

    The formula applies to a 2-by-2 game. Each player mixes so that the
    other player is indifferent between their two strategies.
    """
    if matrix.shape != (2, 2, 2):
        raise ValueError("Payoff matrix must have shape (2, 2, 2).")

    creator = matrix[:, :, 0].astype(float)
    platform = matrix[:, :, 1].astype(float)

    # p is creator's probability of row 0. It makes platform indifferent.
    denominator_p = (platform[0, 0] - platform[1, 0]) - (
        platform[0, 1] - platform[1, 1]
    )
    # q is platform's probability of column 0. It makes creator indifferent.
    denominator_q = (creator[0, 0] - creator[0, 1]) - (
        creator[1, 0] - creator[1, 1]
    )
    if denominator_p == 0 or denominator_q == 0:
        raise ValueError("This game does not have a unique mixed equilibrium formula.")

    p_creator_fake = (platform[1, 1] - platform[1, 0]) / denominator_p
    q_platform_weak = (creator[1, 1] - creator[0, 1]) / denominator_q
    return float(p_creator_fake), float(q_platform_weak)


def describe_equilibrium() -> None:
    """Print the mixed equilibrium for the selected payoff matrix."""
    creator_fake, platform_weak = mixed_nash_equilibrium()
    print("\nNASH EQUILIBRIUM")
    print("There is no pure-strategy equilibrium in this payoff matrix.")
    print(f"Creator spreads fake news: {creator_fake:.1%}")
    print(f"Creator shares responsible content: {1 - creator_fake:.1%}")
    print(f"Platform uses weak moderation: {platform_weak:.1%}")
    print(f"Platform uses strong moderation: {1 - platform_weak:.1%}")


def simulate_propagation(
    settings: SimulationSettings, strategy: ModerationStrategy
) -> SimulationResult:
    """Simulate cumulative exposure one round at a time.

    Each newly exposed user gets one chance to share. A sharing user exposes
    ``average_contacts`` new people, subject to the remaining population.
    The same random seed is used for each strategy comparison so every
    scenario begins under the same initial conditions.
    """
    validate_settings(settings)
    if not 0 <= strategy.sharing_multiplier <= 1:
        raise ValueError("A moderation multiplier must be between 0 and 1.")

    rng = np.random.default_rng(settings.seed)
    effective_probability = settings.sharing_probability * strategy.sharing_multiplier
    cumulative = settings.initially_exposed
    newly_exposed = settings.initially_exposed
    history = [cumulative]

    for _ in range(settings.rounds):
        sharers = rng.binomial(newly_exposed, effective_probability)
        possible_new_exposures = sharers * settings.average_contacts
        new_exposures = min(settings.population - cumulative, possible_new_exposures)
        cumulative += new_exposures
        newly_exposed = new_exposures
        history.append(cumulative)

    return SimulationResult(strategy.name, tuple(history))


def compare_strategies(settings: SimulationSettings) -> list[SimulationResult]:
    """Run all moderation strategies with identical input settings."""
    return [simulate_propagation(settings, strategy) for strategy in MODERATION_STRATEGIES]


def print_comparison(results: list[SimulationResult], population: int) -> None:
    """Print final exposure, percentage, and reduction from no moderation."""
    no_moderation = results[0].final_exposure
    print("\nSIMULATION RESULTS")
    print(f"{'Strategy':<24}{'Final exposed':>16}{'% of population':>20}{'Reduction vs no moderation':>30}")
    for result in results:
        exposure = result.final_exposure
        percentage = 100 * exposure / population
        reduction = 0 if no_moderation == 0 else 100 * (no_moderation - exposure) / no_moderation
        print(f"{result.strategy_name:<24}{exposure:>16}{percentage:>19.1f}%{reduction:>29.1f}%")


def save_graphs(results: list[SimulationResult], settings: SimulationSettings, output_dir: Path) -> None:
    """Create a line chart and a final-exposure bar chart."""
    # Import Matplotlib only when graphs are requested. This keeps the automated
    # tests fast and lets them run on machines without a graphical display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = range(settings.rounds + 1)

    plt.figure(figsize=(9, 5))
    for result in results:
        plt.plot(rounds, result.history, marker="o", label=result.strategy_name)
    plt.title("Fake-news propagation under moderation strategies")
    plt.xlabel("Simulation round")
    plt.ylabel("Cumulative users exposed")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "propagation_over_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    names = [result.strategy_name.replace(" moderation", "") for result in results]
    values = [result.final_exposure for result in results]
    bars = plt.bar(names, values, color=["#d95f02", "#7570b3", "#1b9e77"])
    plt.title("Final fake-news exposure by moderation strategy")
    plt.ylabel("Users exposed")
    plt.bar_label(bars)
    plt.tight_layout()
    plt.savefig(output_dir / "final_exposure_comparison.png", dpi=150)
    plt.close()


def prompt_value(
    label: str, default: int | float, converter: Callable[[str], int | float], validator: Callable[[int | float], bool], error: str
) -> int | float:
    """Ask repeatedly until the user enters a valid value or accepts a default."""
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = converter(raw)
        except ValueError:
            print(error)
            continue
        if validator(value):
            return value
        print(error)


def collect_settings() -> SimulationSettings:
    """Collect validated terminal inputs, with defaults for a quick run."""
    print("\nEnter values, or press Enter to use each default.")
    population = int(prompt_value("Population", 1000, int, lambda x: x > 0, "Enter a whole number above 0."))
    initially_exposed = int(prompt_value(
        "Initially exposed users", 10, int, lambda x: 0 <= x <= population,
        f"Enter a whole number from 0 to {population}."
    ))
    rounds = int(prompt_value("Number of rounds", 12, int, lambda x: x >= 0, "Enter a whole number of 0 or more."))
    sharing_percent = float(prompt_value(
        "Base sharing probability (%)", 35.0, float, lambda x: 0 <= x <= 100,
        "Enter a number from 0 to 100."
    ))
    seed = int(prompt_value("Random seed", 42, int, lambda x: True, "Enter a whole number."))
    return SimulationSettings(
        population=population,
        initially_exposed=initially_exposed,
        rounds=rounds,
        sharing_probability=sharing_percent / 100,
        seed=seed,
    )


def run_tests() -> bool:
    """Run the included tests when the user chooses `python main.py --test`."""
    test_dir = Path(__file__).parent / "tests"
    suite = unittest.defaultTestLoader.discover(str(test_dir))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main() -> None:
    parser = argparse.ArgumentParser(description="Game theory and fake-news propagation project")
    parser.add_argument("--test", action="store_true", help="Run the built-in test suite and exit")
    parser.add_argument("--no-graphs", action="store_true", help="Do not save PNG graph files")
    args = parser.parse_args()

    if args.test:
        sys.exit(0 if run_tests() else 1)

    print("Fake News Propagation and Platform Moderation")
    print("This is a simplified classroom model, not real-world data.")
    settings = collect_settings()
    print_payoff_matrix()
    describe_equilibrium()
    results = compare_strategies(settings)
    print_comparison(results, settings.population)
    if not args.no_graphs:
        save_graphs(results, settings, Path(__file__).parent)
        print("\nSaved: propagation_over_time.png and final_exposure_comparison.png")


if __name__ == "__main__":
    main()
