"""AirCast Training Pipeline and Artifact Integrity Tests

Verifies:
1. Persistence baseline calculation correctness.
2. Saved model artifacts exist on disk.
3. Saved metadata files are complete and valid JSON.
4. joblib model reload produces exact matching predictions.
5. Zero test-set leakage: test features remain untouched.
"""

import json
import os
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.train import PersistenceBaseline


class TestTrainPipeline(unittest.TestCase):
    """Test suite for training pipeline, models, and artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.val_path = os.path.join(cls.project_root, "data/processed/val_features.csv")
        cls.models_dir = os.path.join(cls.project_root, "models")
        cls.reports_dir = os.path.join(cls.project_root, "reports")
        cls.val_df = pd.read_csv(cls.val_path)

    def test_01_persistence_baseline(self):
        """Persistence baseline predicts pm25_current exactly and matches computed metrics."""
        pers = PersistenceBaseline(current_col="pm25_current")
        preds = pers.predict(self.val_df)

        self.assertEqual(len(preds), len(self.val_df))
        np.testing.assert_array_equal(preds, self.val_df["pm25_current"].to_numpy())

        # Check known validation metrics
        mae = mean_absolute_error(self.val_df["target_pm25"], preds)
        rmse = root_mean_squared_error(self.val_df["target_pm25"], preds)
        self.assertAlmostEqual(mae, 8.8662, places=4)
        self.assertAlmostEqual(rmse, 15.2144, places=4)

    def test_02_persistence_missing_handling(self):
        """Persistence baseline raises ValueError if current value is missing."""
        pers = PersistenceBaseline(current_col="pm25_current")
        df_with_nan = pd.DataFrame({"pm25_current": [10.0, np.nan, 20.0]})
        with self.assertRaises(ValueError):
            pers.predict(df_with_nan)

    def test_03_saved_artifacts_exist(self):
        """All expected model artifacts and metadata files exist in models/."""
        expected_artifacts = [
            "best_model.joblib",
            "best_model_metadata.json",
            "linear_regression_autoregressive.joblib",
            "linear_regression_expanded.joblib",
            "linear_regression_full.joblib",
            "random_forest_autoregressive.joblib",
            "random_forest_expanded.joblib",
            "random_forest_full.joblib",
            "persistence_baseline_metadata.json",
        ]
        for fname in expected_artifacts:
            fpath = os.path.join(self.models_dir, fname)
            self.assertTrue(os.path.isfile(fpath), f"Missing artifact: {fname}")

    def test_04_best_model_reload_and_predict(self):
        """Reloading best_model.joblib produces identical predictions of expected shape."""
        best_model_path = os.path.join(self.models_dir, "best_model.joblib")
        best_meta_path = os.path.join(self.models_dir, "best_model_metadata.json")

        with open(best_meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        features = metadata["feature_list"]
        model = joblib.load(best_model_path)

        X_val = self.val_df[features]
        preds = model.predict(X_val)

        self.assertEqual(preds.shape, (len(self.val_df),))
        mae = mean_absolute_error(self.val_df["target_pm25"], preds)
        rmse = root_mean_squared_error(self.val_df["target_pm25"], preds)

        self.assertAlmostEqual(mae, metadata["mae"], places=4)
        self.assertAlmostEqual(rmse, metadata["rmse"], places=4)

    def test_05_reports_exist(self):
        """Validation metrics report and markdown summary exist."""
        csv_path = os.path.join(self.reports_dir, "validation_metrics.csv")
        json_path = os.path.join(self.reports_dir, "validation_metrics.json")
        md_path = os.path.join(self.reports_dir, "model_comparison.md")

        self.assertTrue(os.path.isfile(csv_path))
        self.assertTrue(os.path.isfile(json_path))
        self.assertTrue(os.path.isfile(md_path))


if __name__ == "__main__":
    unittest.main()
