from __future__ import annotations

import json
import os

import pandas as pd

from agents.logging_utils import with_agent_logging


@with_agent_logging("duplicate_removal")
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows from the DataFrame"""
    rows_before = len(df)
    cleaned = df.drop_duplicates()
    rows_after = len(cleaned)
    duplicates_removed = rows_before - rows_after
    duplicate_percentage = (
        round((duplicates_removed / rows_before) * 100, 2)
        if rows_before > 0
        else 0.0
    )

    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "duplicate_percentage": duplicate_percentage,
    }

    return cleaned, summary


def save_versioned_artifact(
    df: pd.DataFrame,
    stage_name: str,
    version: int = 1,
    artifacts_dir: str = "artifacts",
) -> str:
    """Save a DataFrame as a versioned CSV file inside the artifacts directory"""
    os.makedirs(artifacts_dir, exist_ok=True)
    file_name = f"{stage_name}_v{version}.csv"
    file_path = os.path.join(artifacts_dir, file_name)
    df.to_csv(file_path, index=False)
    return os.path.abspath(file_path)


@with_agent_logging("missing_value_imputation")
def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    result_df = df.copy()
    total_rows = len(result_df)

    columns_imputed: list[dict] = []
    columns_flagged_high_missing: list[dict] = []
    columns_skipped_no_missing: list[str] = []

    for col in result_df.columns:
        missing_count = int(result_df[col].isna().sum())

        # no missing values - skip entirely
        if missing_count == 0:
            columns_skipped_no_missing.append(str(col))
            continue

        missing_percentage = round((missing_count / total_rows) * 100, 2)

        # >50 % missing - flag and leave untouched
        if missing_percentage > 50:
            columns_flagged_high_missing.append(
                {
                    "column": str(col),
                    "missing_percentage": missing_percentage,
                }
            )
            continue

        dtype = str(result_df[col].dtype)

        if dtype in ("int64", "float64"):
            # numeric column, ≤50 % missing
            skew_val = float(result_df[col].skew())

            if abs(skew_val) > 1:
                strategy = "median"
                fill_value = result_df[col].median()
            else:
                strategy = "mean"
                fill_value = result_df[col].mean()

            if isinstance(fill_value, (pd.Timestamp, pd.Timedelta)):
                fill_value_plain = str(fill_value)
            else:
                fill_value_plain = fill_value.item() if hasattr(fill_value, "item") else fill_value

            result_df[col] = result_df[col].fillna(fill_value)

            columns_imputed.append(
                {
                    "column": str(col),
                    "strategy": strategy,
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage,
                    "fill_value": fill_value_plain,
                    "skew": skew_val,
                }
            )

        elif dtype == "object":
            # categorical column, ≤50 % missing
            mode_series = result_df[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else None

            result_df[col] = result_df[col].fillna(fill_value)

            columns_imputed.append(
                {
                    "column": str(col),
                    "strategy": "mode",
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage,
                    "fill_value": fill_value,
                    "skew": None,
                }
            )

    summary = {
        "columns_imputed": columns_imputed,
        "columns_flagged_high_missing": columns_flagged_high_missing,
        "columns_skipped_no_missing": columns_skipped_no_missing,
    }

    return result_df, summary


@with_agent_logging("outlier_handling")
def handle_outliers(
    df: pd.DataFrame,
    target_column: str | None = None,
    cap_target: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Detect and cap outliers in numeric columns using the IQR method."""
    result_df = df.copy()
    total_rows = len(result_df)

    columns_processed: list[dict] = []
    columns_skipped_low_cardinality: list[dict] = []
    columns_skipped_target_protected: list[dict] = []

    for col in result_df.columns:
        dtype = str(result_df[col].dtype)

        if dtype not in ("int64", "float64"):
            continue

        # Target column protection: skip if target is identified and user hasn't opted in
        if target_column is not None and str(col) == target_column and not cap_target:
            columns_skipped_target_protected.append(
                {
                    "column": str(col),
                    "reason": "target_column_protected_by_default",
                }
            )
            continue

        unique_count = result_df[col].nunique()
        col_data = result_df[col]

        # low cardinality (categorical-in-disguise)
        if unique_count <= 5:
            columns_skipped_low_cardinality.append(
                {"column": str(col), "reason": "low_cardinality"}
            )
            continue

        # bounded count variable (discrete counts with limited range)
        is_non_negative = (col_data >= 0).all()
        is_whole_numbers = (col_data == col_data.round()).all()
        if is_non_negative and is_whole_numbers and unique_count <= 20:
            columns_skipped_low_cardinality.append(
                {"column": str(col), "reason": "bounded_count_variable"}
            )
            continue

        result_df[col] = result_df[col].astype("float64")

        q1 = float(result_df[col].quantile(0.25))
        q3 = float(result_df[col].quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers
        below_mask = result_df[col] < lower_bound
        above_mask = result_df[col] > upper_bound
        outliers_mask = below_mask | above_mask
        outliers_capped_count = int(outliers_mask.sum())

        if outliers_capped_count == 0:
            continue

        outliers_capped_percentage = round(
            (outliers_capped_count / total_rows) * 100, 2
        )

        result_df.loc[below_mask, col] = lower_bound
        result_df.loc[above_mask, col] = upper_bound

        entry: dict = {
            "column": str(col),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers_capped_count": outliers_capped_count,
            "outliers_capped_percentage": outliers_capped_percentage,
        }

        # If the target column was explicitly opted in, note it in the entry
        if target_column is not None and str(col) == target_column and cap_target:
            entry["is_target"] = True

        columns_processed.append(entry)

    summary = {
        "columns_processed": columns_processed,
        "columns_skipped_low_cardinality": columns_skipped_low_cardinality,
        "columns_skipped_target_protected": columns_skipped_target_protected,
    }

    return result_df, summary


@with_agent_logging("cleaning_orchestration")
def clean_dataset_stage_1(
    df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
    target_column: str | None = None,
    cap_target: bool = False,
) -> dict:
    deduped_df, dup_summary = remove_duplicates(df)

    imputed_df, impute_summary = impute_missing_values(deduped_df)

    cleaned_df, outlier_summary = handle_outliers(
        imputed_df,
        target_column=target_column,
        cap_target=cap_target,
    )

    combined_summary = {
        "duplicate_removal": dup_summary,
        "missing_value_imputation": impute_summary,
        "outlier_handling": outlier_summary,
    }

    artifact_path = save_versioned_artifact(
        cleaned_df, "cleaned", version=1, artifacts_dir=artifacts_dir,
    )

    os.makedirs(artifacts_dir, exist_ok=True)
    changelog_name = "cleaned_v1_changelog.json"
    changelog_path = os.path.join(artifacts_dir, changelog_name)
    with open(changelog_path, "w") as f:
        json.dump(combined_summary, f, indent=2, default=str)

    return {
        "artifact_path": artifact_path,
        "changelog_path": os.path.abspath(changelog_path),
        "summary": combined_summary,
    }

