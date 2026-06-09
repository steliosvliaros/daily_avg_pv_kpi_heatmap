import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.metrics_calculator import calculate_revenue_from_energy, load_park_prices


class MetricsCalculatorRegressionTests(unittest.TestCase):
    def test_load_park_prices_removes_blank_and_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "park_metadata.csv"
            pd.DataFrame(
                [
                    {"park_id": "park_a", "price_euro_to_kwh": 0.10},
                    {"park_id": "park_a", "price_euro_to_kwh": 0.20},
                    {"park_id": "", "price_euro_to_kwh": 0.30},
                    {"park_id": None, "price_euro_to_kwh": 0.40},
                ]
            ).to_csv(metadata_path, index=False)

            prices = load_park_prices(metadata_path)

            self.assertTrue(prices.index.is_unique, "Prices should not contain duplicate park IDs")
            self.assertTrue(prices.index.notna().all(), "Prices should not keep blank park IDs")
            self.assertIn("park_a", prices.index)
            self.assertAlmostEqual(prices["park_a"], 0.15)

    def test_calculate_revenue_from_energy_accepts_deduped_prices(self):
        energy = pd.DataFrame({"park_a": [10.0, 20.0]}, index=[2024, 2025])

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "park_metadata.csv"
            pd.DataFrame(
                [
                    {"park_id": "park_a", "price_euro_to_kwh": 0.10},
                    {"park_id": "park_a", "price_euro_to_kwh": 0.20},
                    {"park_id": "", "price_euro_to_kwh": 0.30},
                ]
            ).to_csv(metadata_path, index=False)

            prices = load_park_prices(metadata_path)
            revenue = calculate_revenue_from_energy(energy, price_per_kwh=prices)

            self.assertEqual(revenue.shape, energy.shape)
            self.assertIn("park_a_revenue", revenue.columns)
            self.assertTrue(revenue.notna().all().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
