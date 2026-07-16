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


@with_agent_logging("cleaning_orchestration")
def clean_dataset_stage_1(
    df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> dict:
    deduped_df, dup_summary = remove_duplicates(df)

    cleaned_df, impute_summary = impute_missing_values(deduped_df)

    combined_summary = {
        "duplicate_removal": dup_summary,
        "missing_value_imputation": impute_summary,
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

    print("\nALL TESTS PASSED")
