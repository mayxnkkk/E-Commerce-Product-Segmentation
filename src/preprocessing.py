"""
Data Preprocessing and Outlier Treatment Module.

Performs data quality checks, missing value validation, duplicate removal,
and IQR-based outlier detection and capping (winsorization) on price and sales_volume.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import DATA_RAW_DIR, DATA_PROCESSED_DIR, TABLES_DIR, FEATURE_COLUMNS


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform thorough data hygiene and quality checks.

    Returns:
        dict: Summary of missing values, duplicates, and invalid ranges.
    """
    summary = {
        "initial_shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["product_id"].duplicated().sum()),
        "invalid_ratings": int(((df["rating"] < 1.0) | (df["rating"] > 5.0)).sum()),
        "negative_prices": int((df["price"] < 0).sum()),
        "negative_sales": int((df["sales_volume"] < 0).sum()),
        "invalid_return_rates": int(((df["return_rate"] < 0) | (df["return_rate"] > 100)).sum()),
        "invalid_discounts": int(((df["discount_frequency"] < 0) | (df["discount_frequency"] > 100)).sum()),
    }
    return summary


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle duplicates and missing values systematically.
    """
    cleaned_df = df.copy()

    # Drop exact duplicates if any
    duplicates_count = cleaned_df.duplicated().sum()
    if duplicates_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {duplicates_count} duplicate rows.")

    # Impute missing values with column median if present
    for col in FEATURE_COLUMNS:
        if cleaned_df[col].isnull().sum() > 0:
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
            print(f"Imputed missing values in '{col}' with median: {median_val}")

    return cleaned_df


def detect_outliers_iqr(df: pd.DataFrame, feature: str) -> dict:
    """
    Calculate IQR outlier boundaries and statistics for a given feature.

    Parameters:
        df (pd.DataFrame): Input dataframe
        feature (str): Column name

    Returns:
        dict: Q1, Q3, IQR, lower_bound, upper_bound, outlier_count, outlier_pct
    """
    series = df[feature]
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = float(q3 - q1)

    # Lower bound cannot be negative for physical metrics like price/sales
    lower_bound = float(max(0.0, q1 - 1.5 * iqr))
    upper_bound = float(q3 + 1.5 * iqr)

    outliers_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = int(outliers_mask.sum())
    outlier_pct = float(outlier_count / len(series) * 100)

    return {
        "feature": feature,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
    }


def treat_outliers_capping(
    df: pd.DataFrame, features: list = ["price", "sales_volume"]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generate two datasets:
    1. Dataset A: Cleaned dataset before outlier capping
    2. Dataset B: Dataset after IQR capping (winsorization)

    Rationale:
    Some extreme values represent genuine blockbuster or luxury products rather
    than data errors. Capping bounds the variance for distance-based clustering
    without discarding valuable operational data points.
    """
    df_before = df.copy()
    df_after = df.copy()

    stats = {}
    for feat in features:
        feat_stats = detect_outliers_iqr(df_before, feat)
        stats[feat] = feat_stats

        # Apply capping to df_after
        df_after[feat] = df_after[feat].clip(
            lower=feat_stats["lower_bound"],
            upper=feat_stats["upper_bound"]
        )

    return df_before, df_after, stats


def run_preprocessing_pipeline(
    raw_path: str = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Execute full preprocessing pipeline and persist processed datasets.
    """
    if raw_path is None:
        raw_path = os.path.join(DATA_RAW_DIR, "ecommerce_products.csv")

    df_raw = pd.read_csv(raw_path)
    print(f"Loaded raw dataset from: {raw_path} with shape {df_raw.shape}")

    # Quality check
    quality_summary = check_data_quality(df_raw)
    print(f"Data Quality Check:\n  - Missing: {quality_summary['total_missing']}\n  - Duplicates: {quality_summary['duplicate_rows']}")

    # Clean dataset
    df_cleaned = clean_dataset(df_raw)

    # Perform outlier detection and capping on price and sales_volume
    df_before, df_after, outlier_stats = treat_outliers_capping(
        df_cleaned, features=["price", "sales_volume"]
    )

    # Save processed datasets
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    before_path = os.path.join(DATA_PROCESSED_DIR, "products_before_outlier_treatment.csv")
    after_path = os.path.join(DATA_PROCESSED_DIR, "products_after_outlier_treatment.csv")

    df_before.to_csv(before_path, index=False)
    df_after.to_csv(after_path, index=False)
    print(f"Saved Dataset A (Before Capping): {before_path}")
    print(f"Saved Dataset B (After Capping):  {after_path}")

    # Save outlier stats summary table
    os.makedirs(TABLES_DIR, exist_ok=True)
    outlier_df = pd.DataFrame(outlier_stats).T
    outlier_table_path = os.path.join(TABLES_DIR, "outlier_detection_summary.csv")
    outlier_df.to_csv(outlier_table_path, index=True)
    print(f"Saved Outlier Summary Table: {outlier_table_path}")

    return df_before, df_after, outlier_stats


if __name__ == "__main__":
    run_preprocessing_pipeline()
