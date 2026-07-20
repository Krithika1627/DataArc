from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

import unittest.mock

from agents.data_cleaning import (
    clean_dataset,
    fix_dtypes,
    generate_cleaning_explanation,
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

        result = clean_dataset(dup_synthetic, artifacts_dir=str(tmp_path))

        saved_df = pd.read_csv(result["artifact_path"])
        assert len(saved_df) == expected_rows

    def test_duplicate_removal_changelog_roundtrip(self, tmp_path, dup_synthetic: pd.DataFrame):
        rows_before = len(dup_synthetic)
        expected_dupes = rows_before - len(dup_synthetic.drop_duplicates())
        expected_rows = rows_before - expected_dupes

        result = clean_dataset(dup_synthetic, artifacts_dir=str(tmp_path))

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
        result = clean_dataset(impute_synthetic, artifacts_dir=str(tmp_path))

        saved_artifact = pd.read_csv(result["artifact_path"])
        with open(result["changelog_path"]) as f:
            saved_changelog = json.load(f)

        pd.testing.assert_frame_equal(
            pd.read_csv(result["artifact_path"]), saved_artifact
        )
        # Changelog now includes llm_explanation — verify summary matches separately
        assert saved_changelog["llm_explanation"] == result["explanation"]
        for key in result["summary"]:
            assert saved_changelog[key] == result["summary"][key]

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
            result = clean_dataset(outlier_synthetic, artifacts_dir=str(tmp_path))

        saved_artifact = pd.read_csv(result["artifact_path"])
        with open(result["changelog_path"]) as f:
            saved_changelog = json.load(f)

        pd.testing.assert_frame_equal(
            pd.read_csv(result["artifact_path"]), saved_artifact
        )
        # Changelog now includes llm_explanation — verify summary matches separately
        assert saved_changelog["llm_explanation"] == result["explanation"]
        for key in result["summary"]:
            assert saved_changelog[key] == result["summary"][key]

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

# Cleaning explanation tests

class TestCleaningExplanation:
    @pytest.mark.llm
    def test_live_api(self, full_pipeline_df: pd.DataFrame):
        """Live API test: requires a valid GEMINI_API_KEY environment variable."""
        result = clean_dataset(full_pipeline_df, artifacts_dir="/tmp")
        explanation = result["explanation"]

        assert "summary" in explanation
        assert "details" in explanation
        assert isinstance(explanation["details"], list)
        assert len(explanation["details"]) > 0
        assert all(isinstance(d, str) for d in explanation["details"])

    def test_mocked_valid_response(self, monkeypatch):
        """Mocked test: verify correct parsing of a valid mocked JSON response."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"summary": "Cleaned the dataset by removing duplicates and imputing missing values.", '
            '"details": ['
            '"Removed 3 duplicate rows (2.1% of the dataset).", '
            '"Imputed 12 missing values in Age using median because the distribution was skewed."'
            "]}"
        )

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            changelog = {
                "duplicate_removal": {
                    "rows_before": 100,
                    "rows_after": 97,
                    "duplicates_removed": 3,
                    "duplicate_percentage": 3.0,
                },
                "missing_value_imputation": {
                    "columns_imputed": [
                        {
                            "column": "Age",
                            "strategy": "median",
                            "missing_count": 12,
                            "missing_percentage": 12.0,
                            "fill_value": 28.5,
                            "skew": 2.5,
                        }
                    ],
                    "columns_flagged_high_missing": [],
                    "columns_skipped_no_missing": [],
                },
                "outlier_handling": {
                    "columns_processed": [],
                    "columns_skipped_low_cardinality": [],
                    "columns_skipped_target_protected": [],
                },
                "dtype_fixing": {
                    "columns_fixed": [],
                    "columns_left_as_object": [],
                },
            }
            result = generate_cleaning_explanation(changelog)

            assert result["summary"] == (
                "Cleaned the dataset by removing duplicates and imputing missing values."
            )
            assert len(result["details"]) == 2
            assert "duplicate" in result["details"][0].lower()
            assert "median" in result["details"][1].lower()

    def test_fallback_api_failure(self, monkeypatch):
        """Fallback test: simulate API failure and confirm fallback dict."""
        monkeypatch.setenv("GEMINI_API_KEY", "INVALID_KEY_THAT_WILL_FAIL")

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.side_effect = Exception(
                "API call failed: invalid key"
            )
            mock_client_class.return_value = mock_client

            changelog = {
                "duplicate_removal": {"rows_before": 50, "rows_after": 48, "duplicates_removed": 2, "duplicate_percentage": 4.0},
                "missing_value_imputation": {"columns_imputed": [], "columns_flagged_high_missing": [], "columns_skipped_no_missing": []},
                "outlier_handling": {"columns_processed": [], "columns_skipped_low_cardinality": [], "columns_skipped_target_protected": []},
                "dtype_fixing": {"columns_fixed": [], "columns_left_as_object": []},
            }
            result = generate_cleaning_explanation(changelog)

            assert "summary" in result
            assert "details" in result
            assert "LLM explanation unavailable" in result["summary"]
            assert "directly" in result["details"][0]

    def test_fallback_no_key(self, monkeypatch):
        """Fallback when no API key is set."""
        monkeypatch.setenv("GEMINI_API_KEY", "")

        changelog = {
            "duplicate_removal": {"rows_before": 50, "rows_after": 50, "duplicates_removed": 0, "duplicate_percentage": 0.0},
            "missing_value_imputation": {"columns_imputed": [], "columns_flagged_high_missing": [], "columns_skipped_no_missing": []},
            "outlier_handling": {"columns_processed": [], "columns_skipped_low_cardinality": [], "columns_skipped_target_protected": []},
            "dtype_fixing": {"columns_fixed": [], "columns_left_as_object": []},
        }
        result = generate_cleaning_explanation(changelog)

        assert "summary" in result
        assert "details" in result
        assert "GEMINI_API_KEY is not set" in result["summary"]

    def test_empty_changelog_safe(self, monkeypatch):
        """A changelog with mostly 'nothing happened' stages still returns valid output."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"summary": "No cleaning was needed for this dataset.", '
            '"details": ["The dataset was already clean with no issues detected."]}'
        )

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            changelog = {
                "duplicate_removal": {"rows_before": 100, "rows_after": 100, "duplicates_removed": 0, "duplicate_percentage": 0.0},
                "missing_value_imputation": {"columns_imputed": [], "columns_flagged_high_missing": [], "columns_skipped_no_missing": ["col1", "col2"]},
                "outlier_handling": {"columns_processed": [], "columns_skipped_low_cardinality": [{"column": "is_active", "reason": "low_cardinality"}], "columns_skipped_target_protected": []},
                "dtype_fixing": {"columns_fixed": [], "columns_left_as_object": ["col1", "col2"]},
            }
            result = generate_cleaning_explanation(changelog)

            assert "summary" in result
            assert isinstance(result["details"], list)
            # Should still parse successfully even with minimal details

    def test_end_to_end_explanation_key(self, tmp_path, full_pipeline_df: pd.DataFrame, monkeypatch):
        """End-to-end: orchestration returns explanation key and changelog includes llm_explanation."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"summary": "Cleaned the dataset by removing duplicates, fixing dtypes, '
            'imputing missing values, and capping outliers.", '
            '"details": ["Removed duplicate rows.", "Fixed currency columns.", '
            '"Imputed missing scores.", "Capped outlier values."]}'
        )

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = clean_dataset(full_pipeline_df, artifacts_dir=str(tmp_path))

            assert "explanation" in result
            assert result["explanation"]["summary"] != ""
            assert len(result["explanation"]["details"]) > 0

            with open(result["changelog_path"]) as f:
                saved_changelog = json.load(f)

            assert "llm_explanation" in saved_changelog
            assert saved_changelog["llm_explanation"]["summary"] == result["explanation"]["summary"]
            assert saved_changelog["llm_explanation"]["details"] == result["explanation"]["details"]


# Orchestration tests 

class TestCleaningOrchestration:
    def test_orchestration_returns_expected_keys(self, tmp_path, dup_synthetic: pd.DataFrame):
        result = clean_dataset(dup_synthetic, artifacts_dir=str(tmp_path))

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
        result = clean_dataset(full_pipeline_df, artifacts_dir=str(tmp_path))

        summary = result["summary"]
        assert "duplicate_removal" in summary
        assert "dtype_fixing" in summary
        assert "missing_value_imputation" in summary
        assert "outlier_handling" in summary

        saved_artifact = pd.read_csv(result["artifact_path"])
        with open(result["changelog_path"]) as f:
            saved_changelog = json.load(f)

        assert len(saved_artifact) > 0
        assert "llm_explanation" in saved_changelog
        for key in summary:
            assert saved_changelog[key] == summary[key]

        assert saved_artifact["price_str"].dtype == "float64"

        assert len(saved_artifact) == 9
