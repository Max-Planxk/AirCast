"""Automated Tests for Evaluation, Residual Diagnostics, and Metric Correctness

Verifies:
1. Metric calculations on known synthetic test vectors.
2. Fair model vs. persistence baseline comparison on identical rows.
3. Strict residual definition: residual = actual - predicted.
4. Exclusion of target, timestamp, and metadata columns from evaluation inputs.
5. Correct output dimensions, zero nulls, and non-negative predictions.
6. All diagnostic figures and evaluation reports exist on disk.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.evaluate import (
    load_model_and_test_data,
    evaluate_test_predictions,
)
from src.features import TARGET_COLUMN


class TestEvaluationPipeline(unittest.TestCase):
    """Test suite for out-of-sample evaluation logic and diagnostic outputs."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.model_path = os.path.join(cls.project_root, "models/best_model.joblib")
        cls.metadata_path = os.path.join(cls.project_root, "models/best_model_metadata.json")
        cls.test_csv_path = os.path.join(cls.project_root, "data/processed/test_features.csv")
        cls.figures_dir = os.path.join(cls.project_root, "reports/figures")
        cls.reports_dir = os.path.join(cls.project_root, "reports")

        cls.model, cls.metadata, cls.test_df = load_model_and_test_data(
            cls.model_path, cls.metadata_path, cls.test_csv_path
        )
        cls.feature_list = cls.metadata["feature_list"]
        cls.metrics, cls.y_pred_rf, cls.y_pred_pers, cls.residuals = (
            evaluate_test_predictions(cls.test_df, cls.model, cls.feature_list)
        )

    def test_01_metric_calculation_on_known_samples(self):
        """MAE and RMSE calculations match exact analytical formulas on known vectors."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 44.0])

        # Absolute errors: [2, 2, 3, 4] -> sum = 11, mean = 2.75
        expected_mae = 2.75
        # Squared errors: [4, 4, 9, 16] -> sum = 33, mean = 8.25 -> sqrt = 2.8722813
        expected_rmse = np.sqrt(8.25)

        computed_mae = mean_absolute_error(y_true, y_pred)
        computed_rmse = root_mean_squared_error(y_true, y_pred)

        self.assertAlmostEqual(computed_mae, expected_mae, places=7)
        self.assertAlmostEqual(computed_rmse, expected_rmse, places=7)

    def test_02_fair_comparison_on_identical_rows(self):
        """Random Forest and Persistence Baseline are evaluated on identical test rows."""
        self.assertEqual(len(self.y_pred_rf), len(self.y_pred_pers))
        self.assertEqual(len(self.y_pred_rf), len(self.test_df))
        self.assertEqual(self.metrics["num_test_examples"], 2736)

    def test_03_residual_definition(self):
        """Residuals strictly follow the definition: residual = actual - predicted."""
        y_true = self.test_df[TARGET_COLUMN].to_numpy()
        expected_residuals = y_true - self.y_pred_rf
        np.testing.assert_allclose(self.residuals, expected_residuals, rtol=1e-6)

    def test_04_feature_list_excludes_target_and_metadata(self):
        """Model input feature list excludes target, timestamps, and database indices."""
        forbidden_cols = [
            TARGET_COLUMN,
            "target_datetime",
            "datetime",
            "No",
            "station",
            "split",
            "is_eligible",
        ]
        for col in forbidden_cols:
            self.assertNotIn(col, self.feature_list, f"Forbidden column '{col}' found in model feature list!")

    def test_05_output_dimensions_and_validity(self):
        """Predictions have expected dimensions, zero nulls/infs, and physically valid values."""
        self.assertEqual(self.y_pred_rf.shape, (len(self.test_df),))
        self.assertFalse(np.isnan(self.y_pred_rf).any(), "NaN found in test predictions!")
        self.assertFalse(np.isinf(self.y_pred_rf).any(), "Inf found in test predictions!")
        self.assertTrue((self.y_pred_rf >= 0).all(), "Negative predictions found!")

    def test_06_saved_figures_and_reports_exist(self):
        """All 5 diagnostic plots and academic evaluation reports exist on disk."""
        expected_figures = [
            "test_actual_vs_predicted.png",
            "test_residual_distribution.png",
            "test_residuals_over_time.png",
            "test_error_by_hour.png",
            "test_error_by_pm25_range.png",
        ]
        for fig_name in expected_figures:
            fig_path = os.path.join(self.figures_dir, fig_name)
            self.assertTrue(os.path.isfile(fig_path), f"Missing plot: {fig_name}")
            self.assertGreater(os.path.getsize(fig_path), 10000, f"Plot {fig_name} is unexpectedly empty!")

        expected_reports = [
            "final_test_evaluation.md",
            "test_evaluation_metrics.json",
            "test_evaluation_metrics.csv",
        ]
        for rep_name in expected_reports:
            rep_path = os.path.join(self.reports_dir, rep_name)
            self.assertTrue(os.path.isfile(rep_path), f"Missing report: {rep_name}")


if __name__ == "__main__":
    unittest.main()
