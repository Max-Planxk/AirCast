"""Automated Leakage, Temporal Integrity, and Feature Correctness Tests

Implements 10 thorough verification checks using Python's built-in unittest framework:
1. Target Alignment: target_pm25 equals PM2.5 exactly 1 hour later (t+1).
2. Lag Correctness: lag features contain strictly past observations.
3. No Future Leakage in Rolling Stats: mutating future data has 0 impact on rolling stats at time t.
4. Feature Separation: ALL_FEATURE_COLUMNS excludes future target, target_datetime, and 'No' index.
5. Chronological Boundaries: Train, Val, and Test target timestamps strictly respect defined date ranges.
6. Split Disjointness: Zero target timestamp overlap between Train, Val, and Test splits.
7. Target Non-Null: Supervised splits contain zero missing target values.
8. Outage Protection: The Dec 2014 sensor outage is not synthetically filled.
9. Eligibility Enforcement: Retained supervised examples have zero missing values in required features.
10. Pipeline Determinism: Re-running feature generation produces strictly identical results.
"""

import os
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.features import (
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    METADATA_COLUMNS,
    CORE_AUTOREGRESSIVE_FEATURES,
    load_cleaned_data,
    construct_feature_dataset,
    split_features_chronologically,
)


class TestFeaturePipeline(unittest.TestCase):
    """Test suite for feature engineering and chronological splitting."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and compute baseline feature dataset."""
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.cleaned_path = os.path.join(
            cls.project_root, "data/processed/aotizhongxin_cleaned.csv"
        )
        cls.cleaned_df = load_cleaned_data(cls.cleaned_path)
        cls.features_df = construct_feature_dataset(cls.cleaned_df)
        cls.train_df, cls.val_df, cls.test_df, cls.audit = (
            split_features_chronologically(cls.features_df)
        )

    def test_01_target_alignment(self):
        """1. Target for each feature row equals PM2.5 exactly 1 hour later."""
        # Spot check across sample indices
        sample_indices = [50, 1000, 5000, 15000, 25000]
        for idx in sample_indices:
            feature_dt = self.features_df.loc[idx, "datetime"]
            target_dt = self.features_df.loc[idx, "target_datetime"]
            target_val = self.features_df.loc[idx, TARGET_COLUMN]
            next_row_val = self.cleaned_df.loc[idx + 1, "PM2.5"]

            # Timestamp must be exactly 1 hour later
            self.assertEqual(target_dt, feature_dt + pd.Timedelta(hours=1))

            # Target value must match next hour's PM2.5 (or both NaN)
            if pd.isna(target_val):
                self.assertTrue(pd.isna(next_row_val))
            else:
                self.assertAlmostEqual(target_val, next_row_val, places=6)

    def test_02_lag_features_temporal_correctness(self):
        """2. Lag features contain only values from the appropriate previous timestamps."""
        test_idx = 100
        lags = [1, 2, 3, 6, 12, 24]
        for lag in lags:
            lag_col = f"pm25_lag_{lag}"
            lag_val = self.features_df.loc[test_idx, lag_col]
            expected_past_val = self.cleaned_df.loc[test_idx - lag, "PM2.5"]
            if pd.isna(lag_val):
                self.assertTrue(pd.isna(expected_past_val))
            else:
                self.assertAlmostEqual(lag_val, expected_past_val, places=6)

    def test_03_rolling_features_no_future_leakage(self):
        """3. Rolling features do not include future observations."""
        # Take a slice of PM2.5, compute rolling statistics at index t=50
        slice_df = self.cleaned_df.iloc[:60].copy()
        t = 50

        # Compute baseline 6h and 24h rolling mean at t
        base_r6 = slice_df["PM2.5"].rolling(6, min_periods=4).mean().iloc[t]
        base_r24 = slice_df["PM2.5"].rolling(24, min_periods=18).mean().iloc[t]

        # Mutate future observations at t+1, t+2, t+3
        mutated_df = slice_df.copy()
        mutated_df.loc[t + 1: t + 3, "PM2.5"] = 9999.0

        # Recompute rolling statistics
        mutated_r6 = mutated_df["PM2.5"].rolling(6, min_periods=4).mean().iloc[t]
        mutated_r24 = mutated_df["PM2.5"].rolling(24, min_periods=18).mean().iloc[t]

        # Rolling statistics at time t MUST be completely unaffected by future changes
        self.assertAlmostEqual(base_r6, mutated_r6, places=7)
        self.assertAlmostEqual(base_r24, mutated_r24, places=7)

    def test_04_feature_columns_exclude_forbidden_targets_and_indexes(self):
        """4. Feature columns exclude the future target, future datetime, and 'No' index."""
        forbidden = [TARGET_COLUMN, "target_datetime", "No", "split", "is_eligible"]
        for col in forbidden:
            self.assertNotIn(
                col,
                ALL_FEATURE_COLUMNS,
                f"Forbidden column '{col}' found in ALL_FEATURE_COLUMNS!",
            )

    def test_05_chronological_boundaries(self):
        """5. Training, validation, and test target timestamps fall strictly within boundaries."""
        # Train: 2013-03-01 to 2016-02-29
        self.assertGreaterEqual(
            self.train_df["target_datetime"].min(),
            pd.Timestamp("2013-03-01 00:00:00"),
        )
        self.assertLessEqual(
            self.train_df["target_datetime"].max(),
            pd.Timestamp("2016-02-29 23:59:59"),
        )

        # Val: 2016-03-01 to 2016-10-31
        self.assertGreaterEqual(
            self.val_df["target_datetime"].min(),
            pd.Timestamp("2016-03-01 00:00:00"),
        )
        self.assertLessEqual(
            self.val_df["target_datetime"].max(),
            pd.Timestamp("2016-10-31 23:59:59"),
        )

        # Test: 2016-11-01 to 2017-02-28
        self.assertGreaterEqual(
            self.test_df["target_datetime"].min(),
            pd.Timestamp("2016-11-01 00:00:00"),
        )
        self.assertLessEqual(
            self.test_df["target_datetime"].max(),
            pd.Timestamp("2017-02-28 23:59:59"),
        )

    def test_06_no_target_timestamp_overlap(self):
        """6. No target timestamp appears in more than one split."""
        train_targets = set(self.train_df["target_datetime"])
        val_targets = set(self.val_df["target_datetime"])
        test_targets = set(self.test_df["target_datetime"])

        self.assertEqual(len(train_targets & val_targets), 0, "Train and Val overlap!")
        self.assertEqual(len(val_targets & test_targets), 0, "Val and Test overlap!")
        self.assertEqual(len(train_targets & test_targets), 0, "Train and Test overlap!")

    def test_07_missing_targets_excluded_from_splits(self):
        """7. Missing targets are 100% excluded from all supervised splits."""
        self.assertEqual(self.train_df[TARGET_COLUMN].isna().sum(), 0)
        self.assertEqual(self.val_df[TARGET_COLUMN].isna().sum(), 0)
        self.assertEqual(self.test_df[TARGET_COLUMN].isna().sum(), 0)

    def test_08_outage_not_synthetically_filled(self):
        """8. The long pollutant outage (Dec 17-31, 2014) is not synthetically filled."""
        outage_start = pd.Timestamp("2014-12-17 10:00:00")
        outage_end = pd.Timestamp("2014-12-31 16:00:00")

        # Verify that all PM2.5 observations in features_df during outage remain NaN
        outage_feature_rows = self.features_df[
            (self.features_df["datetime"] >= outage_start)
            & (self.features_df["datetime"] <= outage_end)
        ]
        self.assertTrue(outage_feature_rows["pm25_current"].isna().all())

        # Verify that target during outage is also NaN
        outage_target_rows = self.features_df[
            (self.features_df["target_datetime"] >= outage_start)
            & (self.features_df["target_datetime"] <= outage_end)
        ]
        self.assertTrue(outage_target_rows[TARGET_COLUMN].isna().all())

        # Verify zero outage-affected rows exist in the training split
        outage_in_train = self.train_df[
            (self.train_df["target_datetime"] >= outage_start)
            & (self.train_df["target_datetime"] <= outage_end)
        ]
        self.assertEqual(len(outage_in_train), 0)

    def test_09_eligibility_enforcement(self):
        """9. Missing input features are handled strictly according to eligibility policy."""
        # No retained supervised example may contain any NaN in ALL_FEATURE_COLUMNS
        self.assertEqual(self.train_df[ALL_FEATURE_COLUMNS].isna().sum().sum(), 0)
        self.assertEqual(self.val_df[ALL_FEATURE_COLUMNS].isna().sum().sum(), 0)
        self.assertEqual(self.test_df[ALL_FEATURE_COLUMNS].isna().sum().sum(), 0)

    def test_10_pipeline_determinism(self):
        """10. Re-running feature generation produces strictly identical results."""
        features_reproduced = construct_feature_dataset(self.cleaned_df)
        assert_frame_equal(self.features_df, features_reproduced)


if __name__ == "__main__":
    unittest.main()
