from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

from agents.data_cleaning import (
    clean_dataset_stage_1,
    fix_dtypes,
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


class TestDtypeFixing:
    def test_unit_height_column_renamed_to_height_cm(self, height_unit_df: pd.DataFrame):
        """Unit-suffixed column gets detected, converted, and renamed."""
        fixed_df, summary = fix_dtypes(height_unit_df)

        assert len(summary["columns_fixed"]) == 1
        entry = summary["columns_fixed"][0]
        assert entry["column"] == "Height"  # original name before rename
        assert entry["original_dtype"] == "object"
        assert entry["new_dtype"] == "float64"
        assert entry["values_converted"] == 10
        assert entry["values_coerced_to_nan"] == 0
        assert entry["unit_detected"] == "cm"
        assert entry["renamed_from"] == "Height"
        assert entry["renamed_to"] == "Height_cm"

        # Column should be renamed in the DataFrame
        assert "Height_cm" in fixed_df.columns
        assert "Height" not in fixed_df.columns

        # Values should be numeric
        assert fixed_df["Height_cm"].dtype == "float64"
        assert fixed_df["Height_cm"].iloc[0] == 180.0
        assert fixed_df["Height_cm"].iloc[1] == 165.0

        # City column should remain untouched
        assert fixed_df["city"].dtype == "object"

    def test_unit_weight_already_named_not_double_renamed(self, weight_already_named_df: pd.DataFrame):
        """Column already named with unit suffix is NOT double-renamed."""
        fixed_df, summary = fix_dtypes(weight_already_named_df)

        assert len(summary["columns_fixed"]) == 1
        entry = summary["columns_fixed"][0]
        assert entry["column"] == "Weight_kg"
        assert entry["unit_detected"] == "kg"
        # No renamed_from/renamed_to since column already had the unit
        assert "renamed_from" not in entry
        assert "renamed_to" not in entry

        # Column should keep its name, not become "Weight_kg_kg"
        assert "Weight_kg" in fixed_df.columns

        # Values should be numeric
        assert fixed_df["Weight_kg"].dtype == "float64"
        assert fixed_df["Weight_kg"].iloc[0] == 70.0
        assert fixed_df["Weight_kg"].iloc[1] == 65.0

    def test_mixed_units_not_renamed(self, mixed_units_df: pd.DataFrame):
        """Column with mixed/inconsistent units is left as object."""
        fixed_df, summary = fix_dtypes(mixed_units_df)

        # 6 values end in "cm", 4 in "in" -> 60% for "cm", below 95% threshold
        # So no unit should be detected, and the conversion will fail because
        # stripping only artifacts ($, comma, %) leaves the unit suffixes on,
        # so pd.to_numeric won't convert them
        assert len(summary["columns_fixed"]) == 0
        assert "measurement" in summary["columns_left_as_object"]

        # Column should remain object dtype
        assert fixed_df["measurement"].dtype == "object"

    def test_currency_column_converted_correctly(self, currency_df: pd.DataFrame):
        """Currency-formatted column gets correctly detected and converted to float64."""
        fixed_df, summary = fix_dtypes(currency_df)

        assert len(summary["columns_fixed"]) == 1
        entry = summary["columns_fixed"][0]
        assert entry["column"] == "price_str"
        assert entry["original_dtype"] == "object"
        assert entry["new_dtype"] == "float64"
        assert entry["values_converted"] == 10
        assert entry["values_coerced_to_nan"] == 0
        assert "$" in entry["artifacts_stripped"]
        assert "," in entry["artifacts_stripped"]

        # Verify correct numeric values
        assert fixed_df["price_str"].dtype == "float64"
        assert fixed_df["price_str"].iloc[0] == 1200.0
        assert fixed_df["price_str"].iloc[1] == 950.0

    def test_percent_column_converted_correctly(self, percent_df: pd.DataFrame):
        """Percent-formatted column gets correctly detected and converted."""
        fixed_df, summary = fix_dtypes(percent_df)

        assert len(summary["columns_fixed"]) == 1
        entry = summary["columns_fixed"][0]
        assert entry["column"] == "pct_str"
        assert entry["original_dtype"] == "object"
        assert entry["new_dtype"] == "float64"
        assert entry["values_converted"] == 10
        assert entry["values_coerced_to_nan"] == 0
        assert "%" in entry["artifacts_stripped"]

        # Verify correct numeric values (e.g. "45%" -> 45.0)
        assert fixed_df["pct_str"].iloc[0] == 45.0
        assert fixed_df["pct_str"].iloc[1] == 62.0

    def test_categorical_column_left_untouched(self, categorical_df: pd.DataFrame):
        """Genuinely categorical column is left as object and appears in columns_left_as_object."""
        fixed_df, summary = fix_dtypes(categorical_df)

        assert len(summary["columns_fixed"]) == 0
        assert "region" in summary["columns_left_as_object"]
        assert "code" in summary["columns_left_as_object"]

        # Both columns should still be object dtype
        assert fixed_df["region"].dtype == "object"
        assert fixed_df["code"].dtype == "object"

    def test_mixed_conversion_reports_coerced_to_nan(self, mixed_conversion_df: pd.DataFrame):
        """Column with some unconvertible values correctly reports values_coerced_to_nan."""
        fixed_df, summary = fix_dtypes(mixed_conversion_df)

        assert len(summary["columns_fixed"]) == 1
        entry = summary["columns_fixed"][0]
        assert entry["column"] == "amount_str"
        assert entry["values_converted"] == 19  # 19 of 20 values converted
        assert entry["values_coerced_to_nan"] == 1  # "unknown" became NaN

        # The "unknown" value should be NaN (it's the last entry)
        assert pd.isna(fixed_df["amount_str"].iloc[19])

        # Other values should be correct
        assert fixed_df["amount_str"].iloc[0] == 1200.0
        assert fixed_df["amount_str"].iloc[1] == 950.0

    def test_short_unit_no_false_positive_on_word_ending(self, rating_grams_df: pd.DataFrame):
        """A column whose name coincidentally ends in a unit letter (e.g.
        'Rating' ending in 'g') should still get correctly renamed to
        Rating_g, not skipped due to a false 'already has unit' match."""
        fixed_df, summary = fix_dtypes(rating_grams_df)

        entry = summary["columns_fixed"][0]
        assert entry["column"] == "Rating"
        assert entry["unit_detected"] == "g"
        assert entry["renamed_from"] == "Rating"
        assert entry["renamed_to"] == "Rating_g"
        assert "Rating_g" in fixed_df.columns

    def test_all_four_stages_end_to_end(self, tmp_path, full_pipeline_df: pd.DataFrame):
        """Full 4-stage orchestration works end-to-end with a dataset needing all steps."""
        result = clean_dataset_stage_1(full_pipeline_df, artifacts_dir=str(tmp_path))

        summary = result["summary"]
        assert "duplicate_removal" in summary
        assert "dtype_fixing" in summary
        assert "missing_value_imputation" in summary
        assert "outlier_handling" in summary

        saved_artifact = pd.read_csv(result["artifact_path"])
        with open(result["changelog_path"]) as f:
            saved_changelog = json.load(f)

        assert len(saved_artifact) > 0
        assert saved_changelog == summary

        assert saved_artifact["price_str"].dtype == "float64"

        assert len(saved_artifact) == 9
