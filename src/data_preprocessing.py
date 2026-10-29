"""AirCast Data Preprocessing Module

Provides reusable, modular functions for loading, validating, cleaning,
and auditing the Beijing Multi-Site Air Quality dataset for the Aotizhongxin station.

Key Design Principles:
1. Target Integrity: The PM2.5 target column is never fabricated or imputed.
2. Leakage Prevention: Only forward-fill from past observations is used for short gaps;
   no bidirectional or lookahead interpolation.
3. Outage Protection: The 343-hour sensor outage (and any gap > 2 hours) is preserved as NaN.
4. Traceability: The original row index 'No' and an imputation flag are preserved.
5. Immutability: The raw input dataset is never altered on disk.
"""

import os
import hashlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


EXPECTED_COLUMNS: List[str] = [
    "No", "year", "month", "day", "hour",
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station"
]

NUMERIC_POLLUTANTS: List[str] = ["PM10", "SO2", "NO2", "CO", "O3"]
NUMERIC_WEATHER: List[str] = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file to verify immutability."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_station_data(filepath: str) -> pd.DataFrame:
    """Load station CSV and validate the presence of expected columns.

    Args:
        filepath: Absolute or relative path to the station CSV file.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If expected columns are missing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Station data file does not exist at '{filepath}'")

    df = pd.read_csv(filepath)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"File '{filepath}' is missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def construct_and_verify_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Construct an ISO datetime column from calendar parts and verify continuity.

    Args:
        df: DataFrame containing 'year', 'month', 'day', and 'hour'.

    Returns:
        DataFrame sorted chronologically with a validated 'datetime' column.

    Raises:
        ValueError: If duplicate timestamps or missing hourly intervals are found.
    """
    df = df.copy()

    # Construct datetime
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # Sort chronologically
    df = df.sort_values("datetime").reset_index(drop=True)

    # Check monotonicity
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly monotonic after sorting.")

    # Check for duplicates
    dup_count = df["datetime"].duplicated().sum()
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate timestamp records.")

    # Check hourly continuity
    min_date = df["datetime"].min()
    max_date = df["datetime"].max()
    expected_range = pd.date_range(start=min_date, end=max_date, freq="h")

    if len(df) != len(expected_range):
        missing_count = len(expected_range) - len(df)
        raise ValueError(
            f"Timestamp grid has gaps: expected {len(expected_range)} hours, "
            f"but found {len(df)} records ({missing_count} missing slots)."
        )

    return df


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate counts and percentages of missing values for all columns."""
    total_rows = len(df)
    null_counts = df.isnull().sum()
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "null_count": null_counts.values,
        "null_pct": [(c / total_rows) * 100 for c in null_counts.values]
    })
    return summary


def _fill_short_gaps(series: pd.Series, max_gap: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Forward-fill isolated missing gaps whose continuous length <= max_gap.

    Crucially, longer gaps (e.g. sensor outages > max_gap) are NOT filled at all.
    Uses forward-fill only, preventing future lookahead / forecasting leakage.

    Args:
        series: Pandas Series with potential missing values.
        max_gap: Maximum consecutive missing observations allowed to be imputed.

    Returns:
        Tuple of (imputed_series, boolean_mask_of_imputed_positions).
    """
    is_na = series.isna()
    if not is_na.any():
        return series.copy(), pd.Series(False, index=series.index)

    # Group consecutive runs of NaNs
    run_id = (is_na != is_na.shift()).cumsum()
    gap_sizes = is_na.groupby(run_id).transform("sum")

    # Only fill positions belonging to a run of length <= max_gap
    can_fill = is_na & (gap_sizes <= max_gap)

    # Forward fill (past observation only)
    filled = series.ffill()

    result = series.copy()
    result[can_fill] = filled[can_fill]

    return result, can_fill


def clean_aotizhongxin_data(
    df: pd.DataFrame,
    max_gap_hours: int = 2
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean the Aotizhongxin air quality data with strict conservative policies.

    Missing-Value Policy:
    1. PM2.5 (Target): Preserved completely intact (all NaNs remain NaN).
       No target values are ever invented or filled.
    2. Input Pollutants & Weather: Short gaps (<= max_gap_hours) are forward-filled
       using past observations only (no lookahead / future leakage).
       Gaps > max_gap_hours (e.g., the 343-hour Dec 2014 outage) are left as NaN.
    3. Wind Direction ('wd'): Short gaps (<= max_gap_hours) forward-filled; remaining
       unfilled entries are explicitly labeled as 'Missing' without arbitrary numeric encoding.
    4. Imputation Flag: A boolean column 'is_imputed_short_gap' is added to explicitly
       track every row that received an imputed feature value.
    5. Traceability: The original 'No' row index is retained for auditing purposes.

    Args:
        df: Input DataFrame after timeline construction.
        max_gap_hours: Maximum allowable consecutive missing hours to impute (default: 2).

    Returns:
        Tuple of (cleaned_df, imputation_stats_dict).
    """
    cleaned = df.copy()
    stats: Dict[str, int] = {}
    any_imputed_mask = pd.Series(False, index=cleaned.index)

    # 1. PM2.5 Target: NEVER imputed
    stats["PM2.5_unfilled_nulls"] = int(cleaned["PM2.5"].isna().sum())

    # 2. Numeric Input Features (Pollutants & Weather)
    features_to_impute = NUMERIC_POLLUTANTS + NUMERIC_WEATHER
    for col in features_to_impute:
        orig_nulls = int(cleaned[col].isna().sum())
        imputed_series, was_filled = _fill_short_gaps(cleaned[col], max_gap=max_gap_hours)
        cleaned[col] = imputed_series
        filled_count = int(was_filled.sum())
        any_imputed_mask = any_imputed_mask | was_filled
        stats[f"{col}_filled"] = filled_count
        stats[f"{col}_remaining_nulls"] = orig_nulls - filled_count

    # 3. Wind Direction (wd): Categorical handling
    orig_wd_nulls = int(cleaned["wd"].isna().sum())
    wd_imputed, wd_was_filled = _fill_short_gaps(cleaned["wd"], max_gap=max_gap_hours)
    cleaned["wd"] = wd_imputed.fillna("Missing")
    wd_filled_count = int(wd_was_filled.sum())
    any_imputed_mask = any_imputed_mask | wd_was_filled
    stats["wd_filled"] = wd_filled_count
    stats["wd_remaining_missing_labeled"] = orig_wd_nulls - wd_filled_count

    # 4. Add Imputation Traceability Flag
    cleaned["is_imputed_short_gap"] = any_imputed_mask
    stats["total_rows_with_any_imputation"] = int(any_imputed_mask.sum())

    return cleaned, stats


def verify_data_quality(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    raw_filepath: str,
    expected_sha256: str
) -> bool:
    """Run assertions to verify data quality and integrity.

    Verifies:
    - Expected columns present in both.
    - Datetime correctly parsed and monotonic.
    - No duplicate timestamps.
    - Continuous hourly coverage from 2013-03-01 to 2017-02-28.
    - Target PM2.5 values are 100% authentic (never fabricated or altered).
    - Raw input file on disk remains unchanged.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: If any verification check fails.
    """
    # 1. Verify raw file immutability
    current_hash = compute_file_sha256(raw_filepath)
    assert current_hash == expected_sha256, (
        f"Raw data file was modified! Expected {expected_sha256}, got {current_hash}"
    )

    # 2. Check dimensions and continuity
    assert len(cleaned_df) == 35064, f"Expected 35064 rows, got {len(cleaned_df)}"
    assert cleaned_df["datetime"].is_monotonic_increasing, "Datetime is not monotonic"
    assert cleaned_df["datetime"].duplicated().sum() == 0, "Duplicate timestamps found"
    assert cleaned_df["datetime"].min() == pd.Timestamp("2013-03-01 00:00:00"), "Invalid start date"
    assert cleaned_df["datetime"].max() == pd.Timestamp("2017-02-28 23:00:00"), "Invalid end date"

    # 3. Target column authenticity
    raw_pm25_nulls = raw_df["PM2.5"].isna().sum()
    clean_pm25_nulls = cleaned_df["PM2.5"].isna().sum()
    assert raw_pm25_nulls == clean_pm25_nulls, (
        f"PM2.5 missing count changed! Raw: {raw_pm25_nulls}, Cleaned: {clean_pm25_nulls}"
    )

    # Verify that existing non-null values were not modified
    raw_valid = raw_df["PM2.5"].dropna()
    clean_valid = cleaned_df["PM2.5"].dropna()
    assert np.allclose(raw_valid.values, clean_valid.values, equal_nan=True), (
        "Non-null PM2.5 values were altered during cleaning!"
    )

    # 4. Outage preservation check (Dec 17-31, 2014)
    outage_mask = (cleaned_df["datetime"] >= "2014-12-17 10:00:00") & (
        cleaned_df["datetime"] <= "2014-12-31 16:00:00"
    )
    assert cleaned_df.loc[outage_mask, "PM2.5"].isna().all(), (
        "PM2.5 outage was improperly filled!"
    )
    assert cleaned_df.loc[outage_mask, "CO"].isna().all(), (
        "CO outage was improperly filled!"
    )

    return True


def save_cleaned_data(df: pd.DataFrame, output_filepath: str) -> None:
    """Save cleaned DataFrame to CSV, ensuring parent directories exist."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)


def run_pipeline(
    raw_filepath: str,
    output_filepath: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Execute end-to-end loading, timeline construction, cleaning, and verification."""
    print("=" * 60)
    print("AirCast Preprocessing Pipeline: Aotizhongxin Station")
    print("=" * 60)

    # Compute baseline SHA-256 hash of raw file
    initial_hash = compute_file_sha256(raw_filepath)
    print(f"1. Raw file verified. SHA-256: {initial_hash}")

    # Load and validate columns
    raw_df = load_station_data(raw_filepath)
    print(f"2. Loaded {len(raw_df)} rows and {len(raw_df.columns)} columns.")

    # Timeline construction
    timed_df = construct_and_verify_timeline(raw_df)
    print("3. Timeline verified: complete hourly grid (2013-03-01 to 2017-02-28).")

    # Missing value audit before cleaning
    before_summary = summarize_missing_values(timed_df)
    print("\n--- Missing Values Before Cleaning ---")
    print(before_summary.to_string(index=False))

    # Apply conservative cleaning policy
    cleaned_df, stats = clean_aotizhongxin_data(timed_df, max_gap_hours=2)

    # Missing value audit after cleaning
    after_summary = summarize_missing_values(cleaned_df)
    print("\n--- Missing Values After Cleaning ---")
    print(after_summary.to_string(index=False))

    print("\n--- Imputation Summary (Gaps <= 2 hours forward-filled) ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Verify quality and immutability
    verify_data_quality(raw_df, cleaned_df, raw_filepath, initial_hash)
    print("\n4. Quality verification passed: Target untouched, outage preserved, raw file unchanged.")

    # Save cleaned output
    save_cleaned_data(cleaned_df, output_filepath)
    print(f"5. Saved cleaned dataset to '{output_filepath}'.")
    print("=" * 60)

    return cleaned_df, stats


if __name__ == "__main__":
    RAW_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/raw/beijing+multi+site+air+quality+data/PRSA2017_Data_20130301-20170228/PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv"
    )
    OUT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/processed/aotizhongxin_cleaned.csv"
    )
    run_pipeline(RAW_PATH, OUT_PATH)
