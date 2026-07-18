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


if __name__ == "__main__":
    import tempfile

    print("PART 1: Duplicate removal test")
    # Create a synthetic DataFrame with seeded duplicates.
    # Total duplicates = 3.
    dup_synthetic = pd.DataFrame(
        {
            "id": [1, 2, 1, 3, 1, 4, 5, 3, 6, 7],
            "value": ["a", "b", "a", "c", "a", "d", "e", "c", "f", "g"],
        }
    )

    rows_before = len(dup_synthetic)
    unique_rows = dup_synthetic.drop_duplicates()
    expected_dupes = rows_before - len(unique_rows)
    print(f"Synthetic DataFrame: {rows_before} rows, seeded {expected_dupes} duplicates")

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = clean_dataset_stage_1(dup_synthetic, artifacts_dir=tmp_dir)

        print("\nReturned summary")
        print(json.dumps(result, indent=2, default=str))

        print(f"\nVerifying CSV at: {result['artifact_path']}")
        saved_df = pd.read_csv(result["artifact_path"])
        saved_rows = len(saved_df)
        expected_rows = rows_before - expected_dupes
        print(f"  Saved rows: {saved_rows} (expected {expected_rows})")
        assert saved_rows == expected_rows, (
            f"Expected {expected_rows} rows, got {saved_rows}"
        )
        print("CSV contents match expected row count. PASSED")

        print(f"\nVerifying changelog at: {result['changelog_path']}")
        with open(result["changelog_path"]) as f:
            changelog = json.load(f)
        print(f"  Changelog contents: {json.dumps(changelog, indent=2, default=str)}")
        assert changelog["duplicate_removal"]["rows_before"] == rows_before
        assert changelog["duplicate_removal"]["rows_after"] == expected_rows
        assert changelog["duplicate_removal"]["duplicates_removed"] == expected_dupes
        print("Changelog assertions PASSED")

    print("\nPART 1 passed.\n")

    print("PART 2: Missing-value imputation test")

    impute_synthetic = pd.DataFrame(
        {
            "col_symmetric": [
                10.0, 12.0, 11.0, 13.0, 9.0,
                11.0, 10.0, None, 12.0, 11.0,
                10.0, 13.0, 11.5, 12.5,
            ],
            "col_skewed": [
                1.0, 2.0, 1.0, 3.0, 2.0,
                100.0, 200.0, None, 1.0, 2.0,
                3.0, 1.0, 2.0, None,
            ],
            "col_category": [
                "a", "b", "a", "c", None,
                "a", "b", "c", "a", None,
                "b", "a", "c", "a",
            ],
            "col_high_miss": [
                1.0, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None,
            ],
        }
    )

    total = len(impute_synthetic)
    print(f"\nSynthetic imputation DataFrame: {total} rows")
    print(
        "  col_symmetric  — float64, ~2 NaNs, ~symmetric\n"
        "  col_skewed     — float64, ~2 NaNs,  outliers → skewed\n"
        "  col_category   — object,  ~2 NaNs,  mode='a'\n"
        "  col_high_miss  — float64, ~12 NaNs, ~85.7 % missing  → flagged\n"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        result_stage = clean_dataset_stage_1(impute_synthetic, artifacts_dir=tmp_dir)

        print("Combined summary")
        print(json.dumps(result_stage["summary"], indent=2, default=str))

        imputed_df = pd.read_csv(result_stage["artifact_path"])

        summary = result_stage["summary"]
        impute_summary = summary["missing_value_imputation"]

        # col_symmetric should be imputed with mean (low skew)
        symm_entry = next(
            c for c in impute_summary["columns_imputed"]
            if c["column"] == "col_symmetric"
        )
        assert symm_entry["strategy"] == "mean", (
            f"Expected 'mean' for col_symmetric, got '{symm_entry['strategy']}'"
        )
        assert symm_entry["skew"] is not None
        assert abs(symm_entry["skew"]) <= 1, (
            f"col_symmetric skew should be ≤1, got {symm_entry['skew']}"
        )
        assert imputed_df["col_symmetric"].isna().sum() == 0, (
            "col_symmetric should have no NaNs after imputation"
        )
        print("col_symmetric: imputed with mean — PASSED")

        # col_skewed should be imputed with median (abs(skew) > 1)
        skew_entry = next(
            c for c in impute_summary["columns_imputed"]
            if c["column"] == "col_skewed"
        )
        assert skew_entry["strategy"] == "median", (
            f"Expected 'median' for col_skewed, got '{skew_entry['strategy']}'"
        )
        assert skew_entry["skew"] is not None
        assert abs(skew_entry["skew"]) > 1, (
            f"col_skewed skew should be >1, got {skew_entry['skew']}"
        )
        assert imputed_df["col_skewed"].isna().sum() == 0, (
            "col_skewed should have no NaNs after imputation"
        )
        print("col_skewed: imputed with median — PASSED")

        # col_category should be imputed with mode
        cat_entry = next(
            c for c in impute_summary["columns_imputed"]
            if c["column"] == "col_category"
        )
        assert cat_entry["strategy"] == "mode", (
            f"Expected 'mode' for col_category, got '{cat_entry['strategy']}'"
        )
        assert cat_entry["skew"] is None, (
            "col_category skew should be null (categorical)"
        )
        assert cat_entry["fill_value"] == "a", (
            f"Expected mode 'a' for col_category, got '{cat_entry['fill_value']}'"
        )
        assert imputed_df["col_category"].isna().sum() == 0, (
            "col_category should have no NaNs after imputation"
        )
        print("col_category: imputed with mode — PASSED")

        # col_high_miss should be flagged, NOT imputed
        flagged_entry = next(
            c for c in impute_summary["columns_flagged_high_missing"]
            if c["column"] == "col_high_miss"
        )
        assert flagged_entry["missing_percentage"] > 50, (
            "col_high_miss should be flagged as >50 % missing"
        )
        assert imputed_df["col_high_miss"].isna().sum() > 0, (
            "col_high_miss should still have NaNs (not imputed)"
        )
        print("col_high_miss: flagged and left with NaNs — PASSED")

        print("\nVerifying saved artifact and changelog")

        saved_artifact = pd.read_csv(result_stage["artifact_path"])
        pd.testing.assert_frame_equal(imputed_df, saved_artifact)
        print("Saved artifact CSV matches returned DataFrame")

        with open(result_stage["changelog_path"]) as f:
            saved_changelog = json.load(f)
        assert saved_changelog == result_stage["summary"], (
            "Saved changelog JSON does not match returned summary"
        )
        print("Saved changelog JSON matches returned summary")

    print("\nPART 2 passed.\n")

    print("PART 3: Outlier handling test (revised)")

    import warnings

    outlier_synthetic = pd.DataFrame(
        {
            "col_outliers": [
                15.0, 20.0, 18.0, 22.0, 14.0,
                16.0, 19.0, 21.0, 17.0, 23.0,
                500.0, -200.0, 13.0, 25.0,
            ],
            "is_active": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "col_symmetric": [
                10.0, 12.0, 11.0, 13.0, 9.0,
                11.0, 10.0, 10.0, 12.0, 11.0,
                10.0, 13.0, 11.5, 12.5,
            ],
            "tutoring_sessions": [
                0, 1, 0, 2, 1, 3, 0, 2, 4, 1,
                0, 3, 2, 5,
            ],
        }
    )

    total_rows = len(outlier_synthetic)
    print(f"\nSynthetic outlier DataFrame: {total_rows} rows")
    print(
        "  col_outliers       — float64, values 10-25 range + extreme outliers 500, -200\n"
        "  is_active          — int64,   0/1 flag (low cardinality, should be skipped)\n"
        "  col_symmetric      — float64, no outliers (values in 9-13 range)\n"
        "  tutoring_sessions  — int64,   0-5 range, whole nums <=20 unique (bounded count, should be skipped)\n"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result_stage = clean_dataset_stage_1(outlier_synthetic, artifacts_dir=tmp_dir)

        print("\nCombined summary")
        print(json.dumps(result_stage["summary"], indent=2, default=str))

        cleaned_df = pd.read_csv(result_stage["artifact_path"])

        summary = result_stage["summary"]
        outlier_summary = summary["outlier_handling"]

        # col_outliers (genuine outliers, should be processed)
        outlier_entry = next(
            c for c in outlier_summary["columns_processed"]
            if c["column"] == "col_outliers"
        )
        print(f"\ncol_outliers: q1={outlier_entry['q1']}, q3={outlier_entry['q3']}, "
              f"iqr={outlier_entry['iqr']}")
        print(f"  lower_bound={outlier_entry['lower_bound']}, "
              f"upper_bound={outlier_entry['upper_bound']}")
        print(f"  capped {outlier_entry['outliers_capped_count']} "
              f"({outlier_entry['outliers_capped_percentage']}%) outliers")

        # Check the extreme values are no longer present
        assert 500.0 not in cleaned_df["col_outliers"].values, (
            "500 should have been capped"
        )
        assert -200.0 not in cleaned_df["col_outliers"].values, (
            "-200 should have been capped"
        )
        print("Extreme values 500 and -200 removed via capping — PASSED")

        # Check they were capped to bounds, not dropped (row count preserved)
        assert len(cleaned_df) == total_rows, (
            f"Row count changed: {len(cleaned_df)} vs {total_rows}"
        )
        print(f"Row count preserved ({len(cleaned_df)} rows) — PASSED")

        # Check capped values respect bounds
        col_data = cleaned_df["col_outliers"]
        lb = outlier_entry["lower_bound"]
        ub = outlier_entry["upper_bound"]
        assert col_data.min() >= lb, (
            f"Min value {col_data.min()} is below lower bound {lb}"
        )
        assert col_data.max() <= ub, (
            f"Max value {col_data.max()} is above upper bound {ub}"
        )
        print("All values within [lower_bound, upper_bound] — PASSED")

        # is_active (low cardinality skip) 
        is_active_skip = next(
            c for c in outlier_summary["columns_skipped_low_cardinality"]
            if c["column"] == "is_active"
        )
        assert is_active_skip["reason"] == "low_cardinality", (
            f"Expected reason 'low_cardinality' for is_active, got '{is_active_skip['reason']}'"
        )
        print("is_active correctly skipped with reason='low_cardinality' — PASSED")

        # Verify is_active values were untouched (still exactly 0 and 1)
        actual_values = set(cleaned_df["is_active"].unique())
        assert actual_values == {0, 1}, (
            f"is_active values changed: {actual_values}"
        )
        print("is_active values unchanged (still 0 and 1) — PASSED")

        #tutoring_sessions (bounded count variable skip) 
        tutoring_skip = next(
            c for c in outlier_summary["columns_skipped_low_cardinality"]
            if c["column"] == "tutoring_sessions"
        )
        assert tutoring_skip["reason"] == "bounded_count_variable", (
            f"Expected reason 'bounded_count_variable' for tutoring_sessions, "
            f"got '{tutoring_skip['reason']}'"
        )
        print("tutoring_sessions correctly skipped with reason='bounded_count_variable' — PASSED")

        assert set(cleaned_df["tutoring_sessions"].unique()) == {0, 1, 2, 3, 4, 5}, (
            "tutoring_sessions values should be unchanged"
        )
        print("tutoring_sessions values unchanged — PASSED")

        # col_symmetric (no outliers) 
        # The column should not appear in columns_processed (no outliers capped)
        symm_in_processed = any(
            c["column"] == "col_symmetric"
            for c in outlier_summary["columns_processed"]
        )
        assert not symm_in_processed, (
            "col_symmetric should not appear in columns_processed (no outliers)"
        )
        print("col_symmetric correctly omitted (zero outliers) — PASSED")

        print("\nVerifying saved artifact and changelog")
        saved_artifact = pd.read_csv(result_stage["artifact_path"])
        pd.testing.assert_frame_equal(cleaned_df, saved_artifact)
        print("Saved artifact CSV matches returned DataFrame")

        with open(result_stage["changelog_path"]) as f:
            saved_changelog = json.load(f)
        assert saved_changelog == result_stage["summary"], (
            "Saved changelog JSON does not match returned summary"
        )
        print("Saved changelog JSON matches returned summary")

    print("\nTarget column protection tests\n")

    print("Target protection test (a): target protected by default (cap_target=False)")
    result_a, summary_a = handle_outliers(
        outlier_synthetic,
        target_column="col_outliers",
        cap_target=False,
    )
    print(f"  Summary: {json.dumps(summary_a, indent=2, default=str)}")

    target_protected = next(
        c for c in summary_a["columns_skipped_target_protected"]
        if c["column"] == "col_outliers"
    )
    assert target_protected["reason"] == "target_column_protected_by_default", (
        f"Expected reason 'target_column_protected_by_default', "
        f"got '{target_protected['reason']}'"
    )

    col_in_processed = any(
        c["column"] == "col_outliers" for c in summary_a["columns_processed"]
    )
    assert not col_in_processed, (
        "col_outliers should not be in columns_processed when target protected"
    )

    assert 500.0 in result_a["col_outliers"].values, (
        "500 should still be present when target is protected"
    )
    assert -200.0 in result_a["col_outliers"].values, (
        "-200 should still be present when target is protected"
    )
    print("Target protected (cap_target=False) — PASSED\n")

    print("Target protection test (b): target opted in (cap_target=True)")
    result_b, summary_b = handle_outliers(
        outlier_synthetic,
        target_column="col_outliers",
        cap_target=True,
    )
    print(f"  Summary: {json.dumps(summary_b, indent=2, default=str)}")

    target_opted = next(
        c for c in summary_b["columns_processed"]
        if c["column"] == "col_outliers"
    )
    assert target_opted.get("is_target") is True, (
        "Expected is_target=True in columns_processed entry"
    )

    assert 500.0 not in result_b["col_outliers"].values, (
        "500 should have been capped when target is opted in"
    )
    assert -200.0 not in result_b["col_outliers"].values, (
        "-200 should have been capped when target is opted in"
    )
    print("Target opted in (cap_target=True) — PASSED\n")

    print("Target protection test (c): target_column=None (target-agnostic)")
    result_c, summary_c = handle_outliers(
        outlier_synthetic,
        target_column=None,
    )
    print(f"  Summary: {json.dumps(summary_c, indent=2, default=str)}")

    assert len(summary_c["columns_skipped_target_protected"]) == 0, (
        "columns_skipped_target_protected should be empty when target_column=None"
    )

    col_in_processed_c = any(
        c["column"] == "col_outliers" for c in summary_c["columns_processed"]
    )
    assert col_in_processed_c, (
        "col_outliers should be processed when target_column=None"
    )

    outlier_entry_c = next(
        c for c in summary_c["columns_processed"]
        if c["column"] == "col_outliers"
    )
    assert "is_target" not in outlier_entry_c, (
        "is_target should not be present when target_column=None"
    )

    assert 500.0 not in result_c["col_outliers"].values, (
        "500 should have been capped when target_column=None"
    )
    assert -200.0 not in result_c["col_outliers"].values, (
        "-200 should have been capped when target_column=None"
    )
    print("Target-agnostic (target_column=None) — PASSED")

    print("\nALL TESTS PASSED")
