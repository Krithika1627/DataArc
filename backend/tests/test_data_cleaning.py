from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

from agents.data_cleaning import (
    clean_dataset_stage_1,
    handle_outliers,
    impute_missing_values,
    remove_duplicates,
    save_versioned_artifact,
)

# Duplicate removal tests 

class TestDuplicateRemoval:
    def test_duplicate_removal_counts_correctly(self, dup_synthetic: pd.DataFrame):
        rows_before = len(dup_synthetic)
        unique_rows = dup_synthetic.drop_duplicates()
        expected_dupes = rows_before - len(unique_rows)

        _, summary = remove_duplicates(dup_synthetic)

        assert summary["rows_before"] == rows_before
        assert summary["rows_after"] == rows_before - expected_dupes
        assert summary["duplicates_removed"] == expected_dupes

    def test_duplicate_removal_file_roundtrip(self, tmp_path, dup_synthetic: pd.DataFrame):
        rows_before = len(dup_synthetic)
        unique_rows = dup_synthetic.drop_duplicates()
        expected_dupes = rows_before - len(unique_rows)
        expected_rows = rows_before - expected_dupes

        result = clean_dataset_stage_1(dup_synthetic, artifacts_dir=str(tmp_path))

        saved_df = pd.read_csv(result["artifact_path"])
        assert len(saved_df) == expected_rows

    def test_duplicate_removal_changelog_roundtrip(self, tmp_path, dup_synthetic: pd.DataFrame):
        rows_before = len(dup_synthetic)
        expected_dupes = rows_before - len(dup_synthetic.drop_duplicates())
        expected_rows = rows_before - expected_dupes

        result = clean_dataset_stage_1(dup_synthetic, artifacts_dir=str(tmp_path))

        with open(result["changelog_path"]) as f:
            changelog = json.load(f)

        assert changelog["duplicate_removal"]["rows_before"] == rows_before
        assert changelog["duplicate_removal"]["rows_after"] == expected_rows
        assert changelog["duplicate_removal"]["duplicates_removed"] == expected_dupes

# Missing-value imputation tests 

class TestMissingValueImputation:
    def test_impute_symmetric_with_mean(self, impute_synthetic: pd.DataFrame):
        imputed_df, summary = impute_missing_values(impute_synthetic)
        impute_summary = summary["columns_imputed"]

        entry = next(c for c in impute_summary if c["column"] == "col_symmetric")
        assert entry["strategy"] == "mean"
        assert entry["skew"] is not None
        assert abs(entry["skew"]) <= 1
        assert imputed_df["col_symmetric"].isna().sum() == 0

    def test_impute_skewed_with_median(self, impute_synthetic: pd.DataFrame):
        imputed_df, summary = impute_missing_values(impute_synthetic)
        impute_summary = summary["columns_imputed"]

        entry = next(c for c in impute_summary if c["column"] == "col_skewed")
        assert entry["strategy"] == "median"
        assert entry["skew"] is not None
        assert abs(entry["skew"]) > 1
        assert imputed_df["col_skewed"].isna().sum() == 0

    def test_impute_categorical_with_mode(self, impute_synthetic: pd.DataFrame):
        imputed_df, summary = impute_missing_values(impute_synthetic)
        impute_summary = summary["columns_imputed"]

        entry = next(c for c in impute_summary if c["column"] == "col_category")
        assert entry["strategy"] == "mode"
        assert entry["skew"] is None
        assert entry["fill_value"] == "a"
        assert imputed_df["col_category"].isna().sum() == 0

    def test_impute_high_missing_flagged(self, impute_synthetic: pd.DataFrame):
        imputed_df, summary = impute_missing_values(impute_synthetic)
        flagged = summary["columns_flagged_high_missing"]

        entry = next(c for c in flagged if c["column"] == "col_high_miss")
        assert entry["missing_percentage"] > 50
        assert imputed_df["col_high_miss"].isna().sum() > 0

    def test_impute_file_roundtrip(self, tmp_path, impute_synthetic: pd.DataFrame):
        result = clean_dataset_stage_1(impute_synthetic, artifacts_dir=str(tmp_path))

        saved_artifact = pd.read_csv(result["artifact_path"])
        with open(result["changelog_path"]) as f:
            saved_changelog = json.load(f)

        pd.testing.assert_frame_equal(
            pd.read_csv(result["artifact_path"]), saved_artifact
        )
        assert saved_changelog == result["summary"]

# Outlier handling tests 

class TestOutlierHandling:
    def test_outlier_capping_extreme_values(self, outlier_synthetic: pd.DataFrame):
        cleaned_df, summary = handle_outliers(outlier_synthetic)
        entry = next(
            c for c in summary["columns_processed"]
            if c["column"] == "col_outliers"
        )

        assert 500.0 not in cleaned_df["col_outliers"].values
        assert -200.0 not in cleaned_df["col_outliers"].values

    def test_outlier_row_count_preserved(self, outlier_synthetic: pd.DataFrame):
        total_rows = len(outlier_synthetic)
        cleaned_df, _ = handle_outliers(outlier_synthetic)
        assert len(cleaned_df) == total_rows

    def test_outlier_bounds_respected(self, outlier_synthetic: pd.DataFrame):
        cleaned_df, summary = handle_outliers(outlier_synthetic)
        entry = next(
            c for c in summary["columns_processed"]
            if c["column"] == "col_outliers"
        )

        col_data = cleaned_df["col_outliers"]
        assert col_data.min() >= entry["lower_bound"]
        assert col_data.max() <= entry["upper_bound"]

    def test_outlier_low_cardinality_skip(self, outlier_synthetic: pd.DataFrame):
        _, summary = handle_outliers(outlier_synthetic)

        entry = next(
            c for c in summary["columns_skipped_low_cardinality"]
            if c["column"] == "is_active"
        )
        assert entry["reason"] == "low_cardinality"

    def test_outlier_low_cardinality_values_untouched(self, outlier_synthetic: pd.DataFrame):
        cleaned_df, _ = handle_outliers(outlier_synthetic)
        assert set(cleaned_df["is_active"].unique()) == {0, 1}

    def test_outlier_bounded_count_skip(self, outlier_synthetic: pd.DataFrame):
        _, summary = handle_outliers(outlier_synthetic)

        entry = next(
            c for c in summary["columns_skipped_low_cardinality"]
            if c["column"] == "tutoring_sessions"
        )
        assert entry["reason"] == "bounded_count_variable"

    def test_outlier_bounded_count_values_untouched(self, outlier_synthetic: pd.DataFrame):
        cleaned_df, _ = handle_outliers(outlier_synthetic)
        assert set(cleaned_df["tutoring_sessions"].unique()) == {0, 1, 2, 3, 4, 5}

    def test_outlier_no_outliers_omitted_from_summary(self, outlier_synthetic: pd.DataFrame):
        _, summary = handle_outliers(outlier_synthetic)

        symm_in_processed = any(
            c["column"] == "col_symmetric"
            for c in summary["columns_processed"]
        )
        assert not symm_in_processed

    def test_outlier_file_roundtrip(self, tmp_path, outlier_synthetic: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = clean_dataset_stage_1(outlier_synthetic, artifacts_dir=str(tmp_path))

        saved_artifact = pd.read_csv(result["artifact_path"])
        with open(result["changelog_path"]) as f:
            saved_changelog = json.load(f)

        pd.testing.assert_frame_equal(
            pd.read_csv(result["artifact_path"]), saved_artifact
        )
        assert saved_changelog == result["summary"]

    def test_no_future_warning_on_int_columns(self, outlier_synthetic: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            handle_outliers(outlier_synthetic)

# Target column protection tests 

class TestTargetColumnProtection:
    def test_target_protected_by_default(self, outlier_synthetic: pd.DataFrame):
        result_df, summary = handle_outliers(
            outlier_synthetic,
            target_column="col_outliers",
            cap_target=False,
        )

        entry = next(
            c for c in summary["columns_skipped_target_protected"]
            if c["column"] == "col_outliers"
        )
        assert entry["reason"] == "target_column_protected_by_default"

        col_in_processed = any(
            c["column"] == "col_outliers" for c in summary["columns_processed"]
        )
        assert not col_in_processed

        assert 500.0 in result_df["col_outliers"].values
        assert -200.0 in result_df["col_outliers"].values

    def test_target_opted_in(self, outlier_synthetic: pd.DataFrame):
        result_df, summary = handle_outliers(
            outlier_synthetic,
            target_column="col_outliers",
            cap_target=True,
        )

        entry = next(
            c for c in summary["columns_processed"]
            if c["column"] == "col_outliers"
        )
        assert entry.get("is_target") is True

        assert 500.0 not in result_df["col_outliers"].values
        assert -200.0 not in result_df["col_outliers"].values

    def test_target_none_is_agnostic(self, outlier_synthetic: pd.DataFrame):
        result_df, summary = handle_outliers(
            outlier_synthetic,
            target_column=None,
        )

        assert len(summary["columns_skipped_target_protected"]) == 0

        entry = next(
            c for c in summary["columns_processed"]
            if c["column"] == "col_outliers"
        )
        assert "is_target" not in entry

        assert 500.0 not in result_df["col_outliers"].values
        assert -200.0 not in result_df["col_outliers"].values

# Orchestration tests 

class TestCleaningOrchestration:
    def test_orchestration_returns_expected_keys(self, tmp_path, dup_synthetic: pd.DataFrame):
        result = clean_dataset_stage_1(dup_synthetic, artifacts_dir=str(tmp_path))

        assert "artifact_path" in result
        assert "changelog_path" in result
        assert "summary" in result

        summary = result["summary"]
        assert "duplicate_removal" in summary
        assert "missing_value_imputation" in summary
        assert "outlier_handling" in summary

    def test_save_versioned_artifact(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3]})
        path = save_versioned_artifact(df, "test_stage", version=1, artifacts_dir=str(tmp_path))

        assert path.endswith("test_stage_v1.csv")
        loaded = pd.read_csv(path)
        assert len(loaded) == 3
