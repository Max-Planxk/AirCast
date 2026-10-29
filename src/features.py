"""AirCast Feature Engineering and Chronological Splitting Module

Provides modular, reproducible functions for:
1. One-hour-ahead target construction (PM2.5 at t+1 aligned with features at t).
2. Autoregressive lag features (t-1, t-2, t-3, t-6, t-12, t-24) and current observation (t).
3. Past-only rolling window statistics (6-hour and 24-hour mean, std, min, max).
4. Cyclical temporal encodings (hour of day, day of year, month using sine/cosine).
5. Circular wind direction and velocity vector features (u, v components).
6. Derived meteorological dynamics (dew point depression, 3-hour barometric trend).
7. Strict chronological splitting (Train: 2013-03 to 2016-02, Val: 2016-03 to 2016-10, Test: 2016-11 to 2017-02).
8. Data leakage prevention and missing-value audit tracking.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# 16-point compass directions mapped to meteorological azimuth degrees (0 deg = North)
WIND_DIRECTION_DEGREES: Dict[str, float] = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

# Explicit definition of feature subsets
CORE_AUTOREGRESSIVE_FEATURES: List[str] = [
    "pm25_current",
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_lag_6",
    "pm25_lag_12",
    "pm25_lag_24",
    "pm25_rolling_mean_6h",
    "pm25_rolling_std_6h",
    "pm25_rolling_min_6h",
    "pm25_rolling_max_6h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_std_24h",
    "pm25_rolling_min_24h",
    "pm25_rolling_max_24h",
]

CALENDAR_FEATURES: List[str] = [
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]

WIND_FEATURES: List[str] = [
    "wd_sin",
    "wd_cos",
    "wind_u",
    "wind_v",
    "wd_is_missing",
]

METEOROLOGICAL_FEATURES: List[str] = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "dew_point_depression",
    "pressure_trend_3h",
]

CO_POLLUTANT_FEATURES: List[str] = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
]

# All input features provided for modeling
ALL_FEATURE_COLUMNS: List[str] = (
    CORE_AUTOREGRESSIVE_FEATURES
    + CALENDAR_FEATURES
    + WIND_FEATURES
    + METEOROLOGICAL_FEATURES
    + CO_POLLUTANT_FEATURES
)

TARGET_COLUMN: str = "target_pm25"

METADATA_COLUMNS: List[str] = [
    "datetime",
    "target_datetime",
    "No",
    "station",
    "split",
    "is_imputed_short_gap",
]


def load_cleaned_data(filepath: str) -> pd.DataFrame:
    """Load cleaned station dataset and verify chronological integrity."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Cleaned data file not found at: {filepath}")

    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Verify chronological order and uniqueness
    if not df["datetime"].is_monotonic_increasing:
        df = df.sort_values("datetime").reset_index(drop=True)

    if df["datetime"].duplicated().sum() > 0:
        raise ValueError("Duplicate timestamps found in cleaned dataset.")

    return df


def build_target_column(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Construct one-hour-ahead forecasting target (PM2.5 at t+1).

    For a feature row at time t, the target is PM2.5 at time t+1.
    The target datetime is explicitly recorded for full temporal auditability.
    """
    df = df.copy()
    df[TARGET_COLUMN] = df["PM2.5"].shift(-horizon)
    df["target_datetime"] = df["datetime"].shift(-horizon)
    return df


def build_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 6, 12, 24]
) -> pd.DataFrame:
    """Construct autoregressive lag features for PM2.5.

    - pm25_current: Observation at time t.
    - pm25_lag_k: Observation at time t-k.
    Uses strictly past observations.
    """
    df = df.copy()
    df["pm25_current"] = df["PM2.5"]
    for k in lags:
        df[f"pm25_lag_{k}"] = df["PM2.5"].shift(k)
    return df


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct past-only rolling window summary statistics.

    Windows:
    - 6-hour window: mean, std, min, max (min_periods=4, requiring >=67% completeness)
    - 24-hour window: mean, std, min, max (min_periods=18, requiring >=75% completeness)

    Justification for min and max:
    PM2.5 exhibits rapid pollution accumulation and cold-front cleanout events.
    The rolling minimum captures baseline background clean air, while the rolling
    maximum captures acute episodic peak exposure over the preceding window.

    Leakage Prevention:
    In pandas, rolling(w) at row t includes observations [t - w + 1, ..., t].
    This uses strictly historical data up to prediction time t and zero future observations.
    """
    df = df.copy()

    # 6-hour window
    r6 = df["PM2.5"].rolling(window=6, min_periods=4)
    df["pm25_rolling_mean_6h"] = r6.mean()
    df["pm25_rolling_std_6h"] = r6.std()
    df["pm25_rolling_min_6h"] = r6.min()
    df["pm25_rolling_max_6h"] = r6.max()

    # 24-hour window
    r24 = df["PM2.5"].rolling(window=24, min_periods=18)
    df["pm25_rolling_mean_24h"] = r24.mean()
    df["pm25_rolling_std_24h"] = r24.std()
    df["pm25_rolling_min_24h"] = r24.min()
    df["pm25_rolling_max_24h"] = r24.max()

    return df


def build_cyclical_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode calendar time cyclically using sine and cosine transformations.

    - Hour of day (0-23): 24-hour diurnal cycle.
    - Day of year (1-365/366): 365.25-day annual seasonal cycle.
    - Month of year (1-12): 12-month annual macro cycle.
    """
    df = df.copy()
    hours = df["datetime"].dt.hour
    day_of_year = df["datetime"].dt.day_of_year
    months = df["datetime"].dt.month

    # Diurnal cycle (24 hours)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Seasonal cycle (365.25 days to accommodate leap year 2016)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Month cycle (12 months)
    df["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12.0)

    return df


def build_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode wind direction circularly and compute meteorological velocity vectors.

    Conventions:
    - Wind direction is mapped to compass azimuth angles (0 deg to 337.5 deg).
    - Radians: theta = deg * (pi / 180).
    - Directional components: wd_sin = sin(theta), wd_cos = cos(theta).
    - Meteorological velocity vectors (direction toward which wind blows):
      wind_u = -WSPM * sin(theta) (zonal / east-west component)
      wind_v = -WSPM * cos(theta) (meridional / north-south component)
    - Handling 'Missing' wind direction:
      When wd is 'Missing', wd_sin, wd_cos, wind_u, and wind_v are assigned 0.0,
      and wd_is_missing is flagged as 1.0 (0.0 otherwise).
      This avoids arbitrary ordinal numeric encoding while preserving speed data.
    """
    df = df.copy()
    wd_series = df["wd"].astype(str)
    wspm_series = df["WSPM"].fillna(0.0)

    # Map string directions to angles
    angles_deg = wd_series.map(WIND_DIRECTION_DEGREES)
    is_missing = angles_deg.isna()

    angles_rad = np.radians(angles_deg.fillna(0.0))

    wd_sin = np.where(is_missing, 0.0, np.sin(angles_rad))
    wd_cos = np.where(is_missing, 0.0, np.cos(angles_rad))

    # Meteorological vectors (toward East and North)
    wind_u = np.where(is_missing, 0.0, -wspm_series * np.sin(angles_rad))
    wind_v = np.where(is_missing, 0.0, -wspm_series * np.cos(angles_rad))

    df["wd_sin"] = wd_sin
    df["wd_cos"] = wd_cos
    df["wind_u"] = wind_u
    df["wind_v"] = wind_v
    df["wd_is_missing"] = is_missing.astype(float)

    return df


def build_meteorological_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived physically meaningful meteorological indicators.

    - dew_point_depression: TEMP - DEWP. Indicator of relative atmospheric moisture
      and fog/saturation potential (near 0 implies saturation / condensation).
    - pressure_trend_3h: PRES(t) - PRES(t-3). Barometric change over 3 hours,
      indicating approaching frontal weather systems or pressure troughs.
    """
    df = df.copy()
    df["dew_point_depression"] = df["TEMP"] - df["DEWP"]
    df["pressure_trend_3h"] = df["PRES"] - df["PRES"].shift(3)
    return df


def construct_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete feature engineered dataset with all candidate features.

    Chains all deterministic domain feature constructors and annotates outage metadata.
    Does not drop rows or perform learned transformations.
    """
    df = build_target_column(df, horizon=1)
    df = build_lag_features(df)
    df = build_rolling_features(df)
    df = build_cyclical_calendar_features(df)
    df = build_wind_features(df)
    df = build_meteorological_derived_features(df)

    # Annotate Dec 2014 outage window (2014-12-17 10:00 to 2014-12-31 16:00)
    outage_mask = (df["datetime"] >= "2014-12-17 10:00:00") & (
        df["datetime"] <= "2014-12-31 16:00:00"
    )
    df["is_outage_affected"] = outage_mask

    return df


def audit_feature_eligibility(
    df: pd.DataFrame,
    required_features: List[str]
) -> Tuple[pd.Series, Dict[str, int]]:
    """Audit supervised eligibility and decompose exclusion reasons.

    Exclusion Reasons:
    1. Initial Warmup: Rows < 24 hours lacking historical lag-24.
    2. Outage Window & Trailing Shadow: December 2014 blackout and subsequent 24 hours.
    3. Missing Target: Unobserved target at t+1 (outside outage/warmup).
    4. Missing Features: Missing required input features (outside outage/warmup).

    Returns:
        Tuple of (boolean_eligibility_mask, audit_reasons_dict).
    """
    target_valid = ~df[TARGET_COLUMN].isna()
    features_valid = ~df[required_features].isna().any(axis=1)
    is_eligible = target_valid & features_valid

    # Hierarchical audit breakdown
    warmup_mask = df.index < 24
    outage_shadow_mask = (
        (df["datetime"] >= "2014-12-17 10:00:00")
        & (df["datetime"] <= "2015-01-01 16:00:00")
        & ~warmup_mask
    )
    target_missing_mask = ~target_valid & ~warmup_mask & ~outage_shadow_mask
    features_missing_mask = (
        target_valid & ~features_valid & ~warmup_mask & ~outage_shadow_mask
    )

    audit: Dict[str, int] = {
        "total_records": len(df),
        "eligible_supervised_examples": int(is_eligible.sum()),
        "total_excluded": int((~is_eligible).sum()),
        "excluded_initial_warmup_24h": int(warmup_mask.sum()),
        "excluded_dec2014_outage_and_shadow": int(outage_shadow_mask.sum()),
        "excluded_unobserved_target_t_plus_1": int(target_missing_mask.sum()),
        "excluded_missing_required_features": int(features_missing_mask.sum()),
    }

    return is_eligible, audit


def split_features_chronologically(
    df: pd.DataFrame,
    required_features: List[str] = ALL_FEATURE_COLUMNS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Assign examples to Train, Validation, and Test splits based on target timestamp.

    Chronological Boundaries (Target Timestamp t+1):
    - Training:   2013-03-01 00:00:00 through 2016-02-29 23:59:59 (36 months / 3 years)
    - Validation: 2016-03-01 00:00:00 through 2016-10-31 23:59:59 (8 months)
    - Test:       2016-11-01 00:00:00 through 2017-02-28 23:59:59 (4 peak winter months)

    Separation Guarantee:
    - Features at the start of Validation and Test use observations available at prediction
      time from the immediately preceding history, but NEVER future observations.
    - Zero target timestamps overlap between splits.
    - Ineligible rows (missing target or missing required features) are excluded from splits.

    Returns:
        Tuple of (train_df, val_df, test_df, audit_summary_dict).
    """
    df = df.copy()

    # Determine eligibility
    is_eligible, audit = audit_feature_eligibility(df, required_features)
    df["is_eligible"] = is_eligible

    # Assign split labels based on target timestamp
    target_dt = df["target_datetime"]

    train_mask = (
        is_eligible
        & (target_dt >= "2013-03-01 00:00:00")
        & (target_dt <= "2016-02-29 23:59:59")
    )
    val_mask = (
        is_eligible
        & (target_dt >= "2016-03-01 00:00:00")
        & (target_dt <= "2016-10-31 23:59:59")
    )
    test_mask = (
        is_eligible
        & (target_dt >= "2016-11-01 00:00:00")
        & (target_dt <= "2017-02-28 23:59:59")
    )

    df["split"] = "excluded"
    df.loc[train_mask, "split"] = "train"
    df.loc[val_mask, "split"] = "val"
    df.loc[test_mask, "split"] = "test"

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    audit["train_examples"] = len(train_df)
    audit["val_examples"] = len(val_df)
    audit["test_examples"] = len(test_df)

    return train_df, val_df, test_df, audit


def save_split_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """Save clearly named split datasets under data/processed/."""
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {
        "train": os.path.join(output_dir, "train_features.csv"),
        "val": os.path.join(output_dir, "val_features.csv"),
        "test": os.path.join(output_dir, "test_features.csv"),
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return paths


def run_feature_pipeline(
    cleaned_data_path: str,
    output_dir: str,
    feature_set: List[str] = ALL_FEATURE_COLUMNS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Execute end-to-end feature engineering, auditing, and chronological splitting."""
    print("=" * 65)
    print("AirCast Feature Pipeline: Feature Construction & Chronological Split")
    print("=" * 65)

    # 1. Load cleaned dataset
    cleaned_df = load_cleaned_data(cleaned_data_path)
    print(f"1. Loaded cleaned data: {len(cleaned_df):,} rows from '{os.path.basename(cleaned_data_path)}'.")

    # 2. Construct feature set
    features_df = construct_feature_dataset(cleaned_df)
    print(f"2. Constructed {len(ALL_FEATURE_COLUMNS)} candidate features.")

    # 3. Chronological splitting & eligibility audit
    train_df, val_df, test_df, audit = split_features_chronologically(
        features_df, required_features=feature_set
    )

    print("\n--- Feature Eligibility & Exclusion Audit ---")
    for k, v in audit.items():
        print(f"  {k:38s}: {v:,}")

    # 4. Save split datasets
    saved_paths = save_split_datasets(train_df, val_df, test_df, output_dir)
    print("\n--- Saved Split Datasets ---")
    for split_name, path in saved_paths.items():
        print(f"  {split_name:5s} -> '{path}'")

    print("=" * 65)
    return train_df, val_df, test_df, audit


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLEANED_PATH = os.path.join(PROJECT_ROOT, "data/processed/aotizhongxin_cleaned.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed")
    run_feature_pipeline(CLEANED_PATH, OUTPUT_DIR)
