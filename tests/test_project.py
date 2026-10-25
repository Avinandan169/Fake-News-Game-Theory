"""Tests for the beginner-friendly game theory project."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import (  # noqa: E402
    MODERATION_STRATEGIES,
    PAYOFF_MATRIX,
    SimulationSettings,
    mixed_nash_equilibrium,
    simulate_propagation,
    validate_settings,
)


class PayoffAndEquilibriumTests(unittest.TestCase):
    def test_payoff_matrix_has_expected_shape(self):
        self.assertEqual(PAYOFF_MATRIX.shape, (2, 2, 2))

    def test_mixed_nash_equilibrium(self):
        creator_fake, platform_weak = mixed_nash_equilibrium()
        self.assertAlmostEqual(creator_fake, 0.125)
        self.assertAlmostEqual(platform_weak, 2 / 3)
        self.assertTrue(0 <= creator_fake <= 1)
        self.assertTrue(0 <= platform_weak <= 1)


class SimulationTests(unittest.TestCase):
    def test_zero_initial_exposure_stays_zero(self):
        settings = SimulationSettings(initially_exposed=0)
        result = simulate_propagation(settings, MODERATION_STRATEGIES[0])
        self.assertEqual(result.history, (0,) * 13)

    def test_zero_rounds_keeps_initial_exposure(self):
        settings = SimulationSettings(initially_exposed=7, rounds=0)
        result = simulate_propagation(settings, MODERATION_STRATEGIES[0])
        self.assertEqual(result.history, (7,))

    def test_zero_sharing_probability_stops_spread(self):
        settings = SimulationSettings(initially_exposed=9, sharing_probability=0)
        result = simulate_propagation(settings, MODERATION_STRATEGIES[0])
        self.assertEqual(result.final_exposure, 9)

    def test_exposure_never_exceeds_population(self):
        settings = SimulationSettings(population=25, initially_exposed=20, rounds=20, sharing_probability=1)
        result = simulate_propagation(settings, MODERATION_STRATEGIES[0])
        self.assertLessEqual(result.final_exposure, 25)
        self.assertTrue(all(a <= b for a, b in zip(result.history, result.history[1:])))

    def test_same_seed_gives_repeatable_result(self):
        settings = SimulationSettings(seed=123)
        first = simulate_propagation(settings, MODERATION_STRATEGIES[1])
        second = simulate_propagation(settings, MODERATION_STRATEGIES[1])
        self.assertEqual(first.history, second.history)

    def test_invalid_initial_exposure_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_settings(SimulationSettings(population=10, initially_exposed=11))

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_settings(SimulationSettings(sharing_probability=1.01))


if __name__ == "__main__":
    unittest.main()
