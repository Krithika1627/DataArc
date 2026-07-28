from __future__ import annotations

import json
import unittest.mock

import numpy as np
import pytest
import pandas as pd

import os

from agents.eda_agent import (
    compute_eda_stats,
    generate_boxplots,
    generate_correlation_heatmap,
    generate_eda_insights,
    generate_histograms,
    generate_target_distribution,
    run_eda,
)


@pytest.fixture
def cleaning_changelog_with_outliers() -> dict:
    """A realistic clean_dataset() changelog with outlier_handling columns_processed.

    Uses ``pclass`` (low cardinality, not ID-like) so that the outlier column is
    *not* also excluded by ``identify_id_columns()`` — this keeps the test focused
    on wiring, not on ID-detection interference.
    """
    return {
        "duplicate_removal": {"rows_before": 14, "rows_after": 14, "duplicates_removed": 0, "duplicate_percentage": 0.0},
        "dtype_fixing": {"columns_fixed": [], "columns_left_as_object": []},
        "missing_value_imputation": {
            "columns_imputed": [],
            "columns_flagged_high_missing": [],
            "columns_skipped_no_missing": [],
        },
        "outlier_handling": {
            "columns_processed": [
                {"column": "pclass", "outliers_capped_count": 0, "lower_bound": 1.0, "upper_bound": 3.0},
            ],
            "columns_skipped_low_cardinality": [],
            "columns_skipped_target_protected": [],
        },
    }


@pytest.fixture
def cleaning_changelog_malformed() -> dict:
    """A changelog that exists but has no outlier_handling key (e.g. a very early abort)."""
    return {"duplicate_removal": {}, "status": "incomplete"}


@pytest.fixture
def titanic_like() -> pd.DataFrame:
    """Module-level fixture accessible by all test classes."""
    return pd.DataFrame(
        {
            "age": [22, 38, 26, 35, 28, 40, 25, 30, 45, 33],
            "fare": [7.25, 71.28, 8.05, 53.10, 15.50, 80.00, 12.00, 22.00, 50.00, 35.00],
            "pclass": [3, 1, 3, 1, 3, 2, 3, 2, 1, 2],
            "survived": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            "sex": ["male", "female", "male", "female", "male", "female", "male", "female", "female", "male"],
            "embarked": ["S", "C", "S", "S", "Q", "C", "S", "S", "C", "Q"],
        }
    )


class TestComputeEdaStatsValidInputs:
    """Tests with a realistic mixed numeric/categorical dataset (Titanic-like)."""

    """Tests with a realistic mixed numeric/categorical dataset (Titanic-like)."""

    @pytest.fixture
    def titanic_like(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [22, 38, 26, 35, 28, 40, 25, 30, 45, 33],
                "fare": [7.25, 71.28, 8.05, 53.10, 15.50, 80.00, 12.00, 22.00, 50.00, 35.00],
                "pclass": [3, 1, 3, 1, 3, 2, 3, 2, 1, 2],
                "survived": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                "sex": ["male", "female", "male", "female", "male", "female", "male", "female", "female", "male"],
                "embarked": ["S", "C", "S", "S", "Q", "C", "S", "S", "C", "Q"],
            }
        )

    def test_returns_expected_keys(self, titanic_like):
        result = compute_eda_stats(titanic_like)
        expected_keys = {"skewness", "correlation_matrix", "flagged_correlations", "distribution_stats", "class_balance"}
        assert set(result.keys()) == expected_keys

    def test_skewness_float_columns(self, titanic_like):
        result = compute_eda_stats(titanic_like)
        skew = result["skewness"]
        # Should have skew for all 4 numeric columns
        assert set(skew.keys()) == {"age", "fare", "pclass", "survived"}
        for col in ("age", "fare", "pclass", "survived"):
            assert isinstance(skew[col], float)

    def test_flagged_correlations_sorted_desc(self, titanic_like):
        result = compute_eda_stats(titanic_like)
        flagged = result["flagged_correlations"]
        # Check sorting: absolute correlation should be descending
        for i in range(len(flagged) - 1):
            assert abs(flagged[i]["correlation"]) >= abs(flagged[i + 1]["correlation"])
        # Each entry has the required keys
        for entry in flagged:
            assert {"col_a", "col_b", "correlation"} == entry.keys()

    def test_correlation_matrix_nested_dict(self, titanic_like):
        result = compute_eda_stats(titanic_like)
        matrix = result["correlation_matrix"]
        numeric_cols = {"age", "fare", "pclass", "survived"}
        assert set(matrix.keys()) == numeric_cols
        for col, row in matrix.items():
            assert set(row.keys()) == numeric_cols
            # Diagonal should be 1.0
            assert abs(row[col] - 1.0) < 1e-4

    def test_distribution_stats_rounded(self, titanic_like):
        result = compute_eda_stats(titanic_like)
        dist = result["distribution_stats"]
        assert set(dist.keys()) == {"age", "fare", "pclass", "survived"}
        for col, stats in dist.items():
            assert set(stats.keys()) == {"mean", "median", "std", "min", "max"}
            for v in stats.values():
                assert isinstance(v, float)

    def test_class_balance_none_without_target(self, titanic_like):
        result = compute_eda_stats(titanic_like)
        assert result["class_balance"] is None

    def test_class_balance_none_with_regression_problem(self, titanic_like):
        result = compute_eda_stats(titanic_like, target_column="fare", problem_type="regression")
        assert result["class_balance"] is None

    def test_class_balance_populated_classification(self, titanic_like):
        result = compute_eda_stats(titanic_like, target_column="survived", problem_type="classification")
        cb = result["class_balance"]
        assert cb is not None
        assert "class_counts" in cb
        assert "class_percentages" in cb
        assert cb["class_counts"] == {"0": 5, "1": 5}
        # Percentages should sum to ~100
        total_pct = sum(cb["class_percentages"].values())
        assert abs(total_pct - 100.0) < 0.05

    def test_class_balance_target_with_text_labels(self, titanic_like):
        df = titanic_like.copy()
        df["category"] = ["low", "med", "high", "low", "med", "high", "low", "med", "high", "low"]
        result = compute_eda_stats(df, target_column="category", problem_type="classification")
        cb = result["class_balance"]
        assert cb is not None
        assert cb["class_counts"] == {"low": 4, "med": 3, "high": 3}


class TestComputeEdaStatsEdgeCases:
    """Edge-case handling: empty, no numeric, single column, stray NaNs."""

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            compute_eda_stats(df)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["foo", "bar", "baz"]})
        result = compute_eda_stats(df)
        assert result["skewness"] == {}
        assert result["correlation_matrix"] == {}
        assert result["flagged_correlations"] == []
        assert result["distribution_stats"] == {}
        assert result["class_balance"] is None

    def test_single_numeric_column(self):
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45]})
        result = compute_eda_stats(df)
        assert result["skewness"] == {"age": pytest.approx(0.0, abs=0.01)}
        assert result["correlation_matrix"] == {"age": {"age": 1.0}}
        assert result["flagged_correlations"] == []  # no pairs to flag
        assert list(result["distribution_stats"].keys()) == ["age"]
        assert result["class_balance"] is None

    def test_target_column_not_in_df_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="target_column 'missing' is not present"):
            compute_eda_stats(df, target_column="missing")

    def test_stray_nan_in_numeric_column(self):
        df = pd.DataFrame(
            {
                "age": [25.0, 30.0, None, 35.0, 40.0],
                "score": [90.0, 85.0, 88.0, None, 95.0],
            }
        )
        # Should not crash; NaN values are naturally handled by pandas
        result = compute_eda_stats(df)
        assert "age" in result["skewness"]
        assert "score" in result["skewness"]
        # NaN for a constant column should be skipped gracefully
        assert isinstance(result["skewness"]["age"], float)
        assert isinstance(result["skewness"]["score"], float)


class TestComputeEdaStatsClassBalanceConditional:
    """Detailed class-balance conditional logic."""

    @pytest.fixture
    def binary_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": range(10),
                "label": [0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
            }
        )

    def test_no_target_none(self, binary_df):
        result = compute_eda_stats(binary_df)
        assert result["class_balance"] is None

    def test_target_no_problem_type_none(self, binary_df):
        result = compute_eda_stats(binary_df, target_column="label")
        assert result["class_balance"] is None

    def test_target_wrong_problem_type_none(self, binary_df):
        result = compute_eda_stats(binary_df, target_column="label", problem_type="regression")
        assert result["class_balance"] is None

    def test_target_classification_populated(self, binary_df):
        result = compute_eda_stats(binary_df, target_column="label", problem_type="classification")
        cb = result["class_balance"]
        assert cb is not None
        assert cb["class_counts"] == {"0": 4, "1": 6}
        assert cb["class_percentages"] == {"0": 40.0, "1": 60.0}

    def test_classification_target_column_none(self, binary_df):
        """problem_type='classification' but no target_column provided => still None."""
        result = compute_eda_stats(binary_df, problem_type="classification")
        assert result["class_balance"] is None


class TestComputeEdaStatsOrchestration:
    """Integration-style tests using the full_pipeline_df from conftest."""

    @pytest.fixture
    def full_df(self, full_pipeline_df) -> pd.DataFrame:
        return full_pipeline_df

    def test_full_pipeline_eda_no_target(self, full_df):
        """After cleaning, the full pipeline df should have numeric columns for EDA."""
        from agents.data_cleaning import clean_dataset
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_result = clean_dataset(full_df, artifacts_dir=tmpdir)
            cleaned_csv_path = cleaned_result["artifact_path"]
            cleaned_df = pd.read_csv(cleaned_csv_path)

        result = compute_eda_stats(cleaned_df)
        # After cleaning, should have numeric columns that dtype fixing produced
        assert "skewness" in result
        assert "correlation_matrix" in result
        assert "distribution_stats" in result
        # At least some numeric columns to analyse
        assert len(result["skewness"]) > 0
        assert len(result["distribution_stats"]) > 0


class TestGenerateHistograms:
    """Tests for generate_histograms()."""

    @pytest.fixture
    def mixed_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [22, 38, 26, 35, 28, 40, 25, 30, 45, 33],
                "fare": [7.25, 71.28, 8.05, 53.10, 15.50, 80.00, 12.00, 22.00, 50.00, 35.00],
                "pclass": [3, 1, 3, 1, 3, 2, 3, 2, 1, 2],
                "passenger_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "sex": ["male", "female", "male", "female", "male", "female", "male", "female", "female", "male"],
            }
        )

    def test_one_histogram_per_numeric_column(self, mixed_df):
        result = generate_histograms(mixed_df)
        # 4 numeric columns: age, fare, pclass, passenger_id
        assert len(result["histograms"]) == 4
        for col in ("age", "fare", "pclass", "passenger_id"):
            assert col in result["histograms"]

    def test_histograms_are_valid_json(self, mixed_df):
        result = generate_histograms(mixed_df)
        for col, json_str in result["histograms"].items():
            # Should be a valid JSON string
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
            # Plotly JSON should have 'data' and 'layout' keys
            assert "data" in parsed
            assert "layout" in parsed

    def test_skipped_columns_empty_when_all_valid(self, mixed_df):
        result = generate_histograms(mixed_df)
        assert result["skipped_columns"] == []

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            generate_histograms(df)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["foo", "bar", "baz"]})
        result = generate_histograms(df)
        assert result["histograms"] == {}
        assert result["skipped_columns"] == []

    def test_exclude_columns_skips_those_columns(self, mixed_df):
        result = generate_histograms(mixed_df, exclude_columns=["passenger_id", "age"])
        hist = result["histograms"]
        assert "passenger_id" not in hist
        assert "age" not in hist
        assert "fare" in hist
        assert "pclass" in hist
        assert len(hist) == 2

    def test_exclude_nonexistent_column_ignored(self, mixed_df):
        result = generate_histograms(mixed_df, exclude_columns=["nonexistent_col"])
        hist = result["histograms"]
        # Nonexistent column is silently ignored, all 4 numeric columns present
        assert len(hist) == 4
        for col in ("age", "fare", "pclass", "passenger_id"):
            assert col in hist

    def test_constant_column_produces_histogram(self):
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5], "other": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = generate_histograms(df)
        assert "constant" in result["histograms"]
        assert "other" in result["histograms"]
        # constant histogram should be valid JSON
        parsed = json.loads(result["histograms"]["constant"])
        assert "data" in parsed
        assert "layout" in parsed

    def test_all_nan_column_skipped(self):
        df = pd.DataFrame(
            {
                "good_col": [1.0, 2.0, 3.0],
                "all_nan": [np.nan, np.nan, np.nan],
            }
        )
        result = generate_histograms(df)
        assert "all_nan" not in result["histograms"]
        assert "good_col" in result["histograms"]
        assert len(result["skipped_columns"]) == 1
        assert result["skipped_columns"][0]["column"] == "all_nan"
        assert "NaN" in result["skipped_columns"][0]["reason"]

    def test_mixed_nan_column_partial_nan_included(self):
        """Column with some NaN values should still produce a histogram (NaN dropped)."""
        df = pd.DataFrame(
            {
                "partially_nan": [1.0, None, 3.0, None, 5.0],
            }
        )
        result = generate_histograms(df)
        assert "partially_nan" in result["histograms"]
        assert result["skipped_columns"] == []
        parsed = json.loads(result["histograms"]["partially_nan"])
        assert "data" in parsed

    def test_large_dataset_does_not_crash(self):
        """10,000+ rows should still produce histograms efficiently."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "col_a": rng.normal(0, 1, 10_000),
                "col_b": rng.uniform(0, 100, 10_000),
            }
        )
        result = generate_histograms(df)
        assert len(result["histograms"]) == 2
        assert result["skipped_columns"] == []


class TestGenerateCorrelationHeatmap:
    """Tests for generate_correlation_heatmap()."""

    @pytest.fixture
    def real_matrix(self, titanic_like) -> dict:
        """Pass a real dataset through compute_eda_stats() to get a real correlation_matrix."""
        stats = compute_eda_stats(titanic_like)
        return stats["correlation_matrix"]

    def test_normal_case_valid_json(self, real_matrix):
        result = generate_correlation_heatmap(real_matrix)
        parsed = json.loads(result)
        assert "data" in parsed
        assert "layout" in parsed
        # Should have 4 columns in the heatmap (age, fare, pclass, survived)
        data = parsed["data"][0]
        assert len(data["x"]) == 4
        assert len(data["y"]) == 4
        # Title should be set
        assert parsed["layout"]["title"]["text"] == "Correlation Heatmap"

    def test_normal_case_preserves_column_order(self, real_matrix):
        """Column order in the heatmap should match the dict's order."""
        result = generate_correlation_heatmap(real_matrix)
        parsed = json.loads(result)
        x_labels = parsed["data"][0]["x"]
        # The matrix keys order from compute_eda_stats: age, fare, pclass, survived
        assert x_labels == ["age", "fare", "pclass", "survived"]

    def test_normal_case_annotation_values(self, real_matrix):
        """Diagonal should show 1.00, off-diagonal should have sensible values."""
        result = generate_correlation_heatmap(real_matrix)
        parsed = json.loads(result)
        text = parsed["data"][0]["text"]
        # Diagonal should be "1.00"
        assert text[0][0] == "1.00"
        assert text[1][1] == "1.00"
        assert text[2][2] == "1.00"
        assert text[3][3] == "1.00"

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="Correlation matrix is empty"):
            generate_correlation_heatmap({})

    def test_single_column_matrix(self):
        matrix = {"age": {"age": 1.0}}
        result = generate_correlation_heatmap(matrix)
        parsed = json.loads(result)
        data = parsed["data"][0]
        assert data["x"] == ["age"]
        assert data["y"] == ["age"]
        assert data["text"] == [["1.00"]]

    def test_malformed_not_nested_dict_raises(self):
        """Value is a scalar instead of a nested dict."""
        with pytest.raises(ValueError, match="malformed"):
            generate_correlation_heatmap({"col_a": 1.0})

    def test_malformed_asymmetric_dict_raises(self):
        """col_a has col_b but not vice versa."""
        matrix = {
            "col_a": {"col_b": 0.8},
            "col_b": {"col_a": 0.8, "col_c": 0.5},
        }
        with pytest.raises(ValueError, match="asymmetric"):
            generate_correlation_heatmap(matrix)

    def test_malformed_non_numeric_value_raises(self):
        """One value is a string instead of a number."""
        matrix = {
            "col_a": {"col_a": 1.0, "col_b": "not_a_number"},
            "col_b": {"col_a": 0.8, "col_b": 1.0},
        }
        with pytest.raises(ValueError, match="non-numeric"):
            generate_correlation_heatmap(matrix)

    def test_rejects_non_dict_input(self):
        with pytest.raises(ValueError, match="not a dict"):
            generate_correlation_heatmap("not_a_dict")  # type: ignore

    def test_diverging_colorscale_centered_at_zero(self, real_matrix):
        """The color scale should be diverging (RdBu_r), centered at 0."""
        result = generate_correlation_heatmap(real_matrix)
        parsed = json.loads(result)
        trace = parsed["data"][0]
        # zmid=0 confirms the scale is centered at the meaningful midpoint
        assert trace["zmid"] == 0
        # zmin=-1 and zmax=1 for proper correlation range
        assert trace["zmin"] == -1
        assert trace["zmax"] == 1
        # colorscale should be expanded from "RdBu_r" into a list of [position, color] pairs
        assert isinstance(trace["colorscale"], list)
        assert len(trace["colorscale"]) > 0
        # First entry should be blue-ish (negative end of RdBu_r)
        first_color = trace["colorscale"][0][1].lower()
        last_color = trace["colorscale"][-1][1].lower()
        assert "rgb" in first_color  # expanded from named scale
        assert "rgb" in last_color


class TestGenerateBoxplots:
    """Tests for generate_boxplots()."""

    @pytest.fixture
    def mixed_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [22, 38, 26, 35, 28, 40, 25, 30, 45, 33],
                "fare": [7.25, 71.28, 8.05, 53.10, 15.50, 80.00, 12.00, 22.00, 50.00, 35.00],
                "pclass": [3, 1, 3, 1, 3, 2, 3, 2, 1, 2],
                "passenger_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "sex": ["male", "female", "male", "female", "male", "female", "male", "female", "female", "male"],
            }
        )

    def test_outlier_columns_none_falls_back_to_all_numeric(self, mixed_df):
        """outlier_columns=None => box plots for all numeric columns."""
        result = generate_boxplots(mixed_df)
        assert len(result["boxplots"]) == 4
        for col in ("age", "fare", "pclass", "passenger_id"):
            assert col in result["boxplots"]

    def test_explicit_outlier_columns_list(self, mixed_df):
        """Only the specified columns appear in output."""
        result = generate_boxplots(mixed_df, outlier_columns=["fare", "age"])
        assert set(result["boxplots"].keys()) == {"fare", "age"}

    def test_outlier_columns_nonexistent_raises(self, mixed_df):
        """Column in outlier_columns that doesn't exist => ValueError naming the bad column."""
        with pytest.raises(ValueError, match="nonexistent_col"):
            generate_boxplots(mixed_df, outlier_columns=["age", "nonexistent_col"])

    def test_outlier_columns_non_numeric_raises(self, mixed_df):
        """Column in outlier_columns that is not numeric => ValueError."""
        with pytest.raises(ValueError, match="sex"):
            generate_boxplots(mixed_df, outlier_columns=["sex"])

    def test_empty_outlier_columns_list_returns_empty(self, mixed_df):
        """outlier_columns=[] (explicit empty) => empty dict, no fallback to all numeric."""
        result = generate_boxplots(mixed_df, outlier_columns=[])
        assert result["boxplots"] == {}

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            generate_boxplots(df)

    def test_exclude_columns_with_none_outlier_columns(self, mixed_df):
        """exclude_columns works with outlier_columns=None."""
        result = generate_boxplots(mixed_df, exclude_columns=["passenger_id", "age"])
        box = result["boxplots"]
        assert "passenger_id" not in box
        assert "age" not in box
        assert "fare" in box
        assert "pclass" in box
        assert len(box) == 2

    def test_exclude_nonexistent_column_ignored(self, mixed_df):
        """Nonexistent exclude column is silently ignored."""
        result = generate_boxplots(mixed_df, exclude_columns=["nonexistent_col"])
        assert len(result["boxplots"]) == 4

    def test_constant_column_produces_boxplot(self):
        """A column with only 1 unique value should still produce a valid box plot."""
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5], "other": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = generate_boxplots(df)
        assert "constant" in result["boxplots"]
        assert "other" in result["boxplots"]
        parsed = json.loads(result["boxplots"]["constant"])
        assert "data" in parsed
        assert "layout" in parsed

    def test_all_nan_column_skipped(self):
        """All-NaN column appears in skipped_columns, not in boxplots."""
        df = pd.DataFrame(
            {
                "good_col": [1.0, 2.0, 3.0],
                "all_nan": [np.nan, np.nan, np.nan],
            }
        )
        result = generate_boxplots(df)
        assert "all_nan" not in result["boxplots"]
        assert "good_col" in result["boxplots"]
        assert len(result["skipped_columns"]) == 1
        assert result["skipped_columns"][0]["column"] == "all_nan"
        assert "NaN" in result["skipped_columns"][0]["reason"]

    def test_boxplots_are_valid_json(self, mixed_df):
        """Every boxplot value should be a valid Plotly JSON string."""
        result = generate_boxplots(mixed_df)
        for col, json_str in result["boxplots"].items():
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
            assert "data" in parsed
            assert "layout" in parsed

    def test_skipped_columns_empty_when_all_valid(self, mixed_df):
        """No all-NaN columns should result in an empty skipped_columns list."""
        result = generate_boxplots(mixed_df)
        assert result["skipped_columns"] == []

    def test_exclude_with_explicit_outlier_columns(self, mixed_df):
        """exclude_columns filters from an explicit outlier_columns list."""
        result = generate_boxplots(
            mixed_df,
            outlier_columns=["age", "fare", "pclass"],
            exclude_columns=["age"],
        )
        assert "age" not in result["boxplots"]
        assert "fare" in result["boxplots"]
        assert "pclass" in result["boxplots"]
        assert len(result["boxplots"]) == 2


class TestGenerateTargetDistribution:
    """Tests for generate_target_distribution()."""

    @pytest.fixture
    def mixed_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [22, 38, 26, 35, 28, 40, 25, 30, 45, 33],
                "fare": [7.25, 71.28, 8.05, 53.10, 15.50, 80.00, 12.00, 22.00, 50.00, 35.00],
                "pclass": [3, 1, 3, 1, 3, 2, 3, 2, 1, 2],
                "survived": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                "sex": ["male", "female", "male", "female", "male", "female", "male", "female", "female", "male"],
            }
        )

    @pytest.fixture
    def many_class_df(self) -> pd.DataFrame:
        """DataFrame with 25 unique classes — exceeds the 20-class limit."""
        return pd.DataFrame({
            "target": list(range(25)),
            "feature": [0] * 25,
        })

    @pytest.fixture
    def single_class_df(self) -> pd.DataFrame:
        """DataFrame with only 1 unique class present."""
        return pd.DataFrame({
            "target": [1, 1, 1, 1, 1],
            "feature": range(5),
        })

    def test_classification_bar_chart(self, mixed_df):
        """Classification target produces a bar chart with text labels."""
        result = generate_target_distribution(mixed_df, target_column="survived", problem_type="classification")
        parsed = json.loads(result)
        assert "data" in parsed
        assert "layout" in parsed
        trace = parsed["data"][0]
        # Bar trace has type "bar"
        assert trace["type"] == "bar"
        # Should have text labels
        assert "text" in trace
        assert trace.get("textposition") == "outside"
        # Title should include the target column name
        assert "Target Distribution: survived" in parsed["layout"]["title"]["text"]

    def test_classification_binary_works(self, mixed_df):
        """Binary classification (2 classes) works normally."""
        result = generate_target_distribution(mixed_df, target_column="survived", problem_type="classification")
        parsed = json.loads(result)
        trace = parsed["data"][0]
        # Survived has 2 classes (0 and 1)
        assert len(trace["x"]) == 2
        assert len(trace["y"]) == 2

    def test_regression_histogram(self, mixed_df):
        """Regression target produces a histogram."""
        result = generate_target_distribution(mixed_df, target_column="age", problem_type="regression")
        parsed = json.loads(result)
        assert "data" in parsed
        assert "layout" in parsed
        trace = parsed["data"][0]
        # Histogram type
        assert trace["type"] == "histogram"
        # Title should include the target column name
        assert "Target Distribution: age" in parsed["layout"]["title"]["text"]

    def test_target_column_not_in_df_raises(self, mixed_df):
        """Nonexistent target column raises ValueError."""
        with pytest.raises(ValueError, match="not_present"):
            generate_target_distribution(mixed_df, target_column="not_present", problem_type="classification")

    def test_target_column_all_nan_raises(self):
        """Entirely NaN target column raises ValueError."""
        df = pd.DataFrame({"target": [np.nan, np.nan, np.nan], "feature": [1, 2, 3]})
        with pytest.raises(ValueError, match="entirely NaN"):
            generate_target_distribution(df, target_column="target", problem_type="classification")

    def test_unrecognized_problem_type_raises(self, mixed_df):
        """problem_type other than classification/regression raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported problem_type"):
            generate_target_distribution(mixed_df, target_column="age", problem_type="clustering")

    def test_unclear_problem_type_raises(self, mixed_df):
        """problem_type='unclear' raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported problem_type"):
            generate_target_distribution(mixed_df, target_column="age", problem_type="unclear")

    def test_classification_too_many_classes_raises(self, many_class_df):
        """Classification target with >20 unique values raises ValueError."""
        with pytest.raises(ValueError, match="exceeds the maximum of 20"):
            generate_target_distribution(many_class_df, target_column="target", problem_type="classification")

    def test_classification_single_class_produces_chart(self, single_class_df):
        """Classification target with only 1 unique value produces a valid single-bar chart."""
        result = generate_target_distribution(single_class_df, target_column="target", problem_type="classification")
        parsed = json.loads(result)
        trace = parsed["data"][0]
        assert trace["type"] == "bar"
        # Verify data exists (handles both plain lists and plotly typed-array dicts)
        assert "x" in trace
        assert "y" in trace

    def test_regression_with_some_nans(self):
        """Regression target with some NaN values produces a valid chart (NaNs excluded)."""
        df = pd.DataFrame({"target": [1.0, 2.0, np.nan, 4.0, np.nan, 6.0], "feature": range(6)})
        result = generate_target_distribution(df, target_column="target", problem_type="regression")
        parsed = json.loads(result)
        assert "data" in parsed
        trace = parsed["data"][0]
        assert trace["type"] == "histogram"

    def test_classification_valid_json(self, mixed_df):
        """Classification result is valid JSON."""
        result = generate_target_distribution(mixed_df, target_column="survived", problem_type="classification")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "data" in parsed
        assert "layout" in parsed

    def test_regression_valid_json(self, mixed_df):
        """Regression result is valid JSON."""
        result = generate_target_distribution(mixed_df, target_column="age", problem_type="regression")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "data" in parsed
        assert "layout" in parsed


# EDA insights tests

@pytest.fixture
def eda_stats_typical() -> dict:
    """A realistic compute_eda_stats() output with numeric columns."""
    return {
        "skewness": {
            "age": 0.253,
            "fare": 1.512,
            "pclass": -0.418,
            "survived": 0.0,
        },
        "correlation_matrix": {
            "age": {"age": 1.0, "fare": 0.082, "pclass": -0.369, "survived": -0.077},
            "fare": {"age": 0.082, "fare": 1.0, "pclass": -0.549, "survived": 0.257},
            "pclass": {"age": -0.369, "fare": -0.549, "pclass": 1.0, "survived": -0.338},
            "survived": {"age": -0.077, "fare": 0.257, "pclass": -0.338, "survived": 1.0},
        },
        "flagged_correlations": [
            {"col_a": "age", "col_b": "pclass", "correlation": -0.369},
            {"col_a": "fare", "col_b": "pclass", "correlation": -0.549},
        ],
        "distribution_stats": {
            "age": {"mean": 32.2, "median": 31.5, "std": 7.1, "min": 22.0, "max": 45.0},
            "fare": {"mean": 35.41, "median": 28.5, "std": 27.5, "min": 7.25, "max": 80.0},
            "pclass": {"mean": 2.1, "median": 2.0, "std": 0.88, "min": 1.0, "max": 3.0},
            "survived": {"mean": 0.5, "median": 0.5, "std": 0.53, "min": 0.0, "max": 1.0},
        },
        "class_balance": {
            "class_counts": {"0": 5, "1": 5},
            "class_percentages": {"0": 50.0, "1": 50.0},
        },
    }


@pytest.fixture
def eda_stats_numeric_only() -> dict:
    """EDA stats with no class_balance (regression or no target supplied)."""
    return {
        "skewness": {
            "age": 0.253,
            "fare": 1.512,
            "pclass": -0.418,
        },
        "correlation_matrix": {
            "age": {"age": 1.0, "fare": 0.082, "pclass": -0.369},
            "fare": {"age": 0.082, "fare": 1.0, "pclass": -0.549},
            "pclass": {"age": -0.369, "fare": -0.549, "pclass": 1.0},
        },
        "flagged_correlations": [
            {"col_a": "fare", "col_b": "pclass", "correlation": -0.549},
        ],
        "distribution_stats": {
            "age": {"mean": 32.2, "median": 31.5, "std": 7.1, "min": 22.0, "max": 45.0},
            "fare": {"mean": 35.41, "median": 28.5, "std": 27.5, "min": 7.25, "max": 80.0},
            "pclass": {"mean": 2.1, "median": 2.0, "std": 0.88, "min": 1.0, "max": 3.0},
        },
        "class_balance": None,
    }


@pytest.fixture
def eda_stats_empty_numeric() -> dict:
    """EDA stats with no numeric columns at all."""
    return {
        "skewness": {},
        "correlation_matrix": {},
        "flagged_correlations": [],
        "distribution_stats": {},
        "class_balance": None,
    }


class TestGenerateEdaInsights:
    """Tests for generate_eda_insights()."""

    def test_mocked_valid_response(self, eda_stats_typical, monkeypatch):
        """A well-formed mocked LLM response is parsed into the expected structure."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"insights": ['
            '"Fare is right-skewed (skewness=1.512), indicating a minority of high-fare passengers.", '
            '"Fare and Pclass show a moderate negative correlation (-0.549), suggesting lower classes pay less.", '
            '"Age is roughly symmetric (skewness=0.253) with a range of 22-45 years.", '
            '"The class balance is perfectly even (50/50), making accuracy a reliable metric."'
            '], '
            '"summary": "The dataset shows moderate correlations between fare and class, '
            'with a skewed fare distribution and a well-balanced target variable."}'
        )

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = generate_eda_insights(eda_stats_typical)

            assert "insights" in result
            assert "summary" in result
            assert len(result["insights"]) == 4
            assert all(isinstance(i, str) for i in result["insights"])
            assert isinstance(result["summary"], str)
            assert "skewness=1.512" in result["insights"][0]
            assert "50/50" in result["insights"][3]

    def test_fallback_no_key(self, eda_stats_typical, monkeypatch):
        """GEMINI_API_KEY not set → returns fallback, no crash, no network call."""
        monkeypatch.setenv("GEMINI_API_KEY", "")

        result = generate_eda_insights(eda_stats_typical)

        assert "insights" in result
        assert "summary" in result
        assert "GEMINI_API_KEY is not set" in result["insights"][0]

    def test_fallback_malformed_response_missing_key(self, eda_stats_typical, monkeypatch):
        """LLM response missing 'insights' key → falls back gracefully."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = '{"summary": "Only a summary, no insights key"}'

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = generate_eda_insights(eda_stats_typical)

            assert "insights" in result
            assert "summary" in result
            assert "LLM insights unavailable" in result["insights"][0]

    def test_exclude_columns_strips_from_prompt(self, eda_stats_typical, monkeypatch):
        """exclude_columns removes specified columns from stats before the prompt."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"insights": ["Fare is right-skewed.", "Age is symmetric."], '
            '"summary": "Two numeric columns remain after exclusion."}'
        )

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Exclude pclass and survived
            result = generate_eda_insights(
                eda_stats_typical,
                exclude_columns=["pclass", "survived"],
            )

            # Verify the LLM was called
            mock_client.models.generate_content.assert_called_once()

            # Extract the prompt that was sent
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            prompt_text = call_kwargs.get("contents", "")

            # Excluded columns should not appear in the prompt
            assert "pclass" not in prompt_text
            assert "survived" not in prompt_text
            # Non-excluded columns should still appear
            assert "age" in prompt_text
            assert "fare" in prompt_text

            # The result itself should still have valid structure
            assert len(result["insights"]) == 2

    def test_empty_stats_llm_not_called(self, eda_stats_empty_numeric, monkeypatch):
        """Empty numeric stats → returns 'not enough data' response, LLM never called."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client_class.return_value = mock_client

            result = generate_eda_insights(eda_stats_empty_numeric)

            # LLM should never be called
            mock_client.models.generate_content.assert_not_called()

            assert "insights" in result
            assert "summary" in result
            assert "no numeric columns" in result["insights"][0].lower()
            assert "LLM was not called" in result["summary"]

    def test_exclude_columns_removes_all_llm_not_called(self, eda_stats_typical, monkeypatch):
        """exclude_columns removing all numeric columns → LLM never called."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client_class.return_value = mock_client

            # Exclude ALL numeric columns
            result = generate_eda_insights(
                eda_stats_typical,
                exclude_columns=["age", "fare", "pclass", "survived"],
            )

            mock_client.models.generate_content.assert_not_called()

            assert "no numeric columns" in result["insights"][0].lower()
            assert "LLM was not called" in result["summary"]

    def test_class_balance_none_prompt_built_ok(self, eda_stats_numeric_only, monkeypatch):
        """class_balance=None → prompt is built without errors, omits class balance section."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"insights": ["Fare is right-skewed.", "Age is symmetric."], '
            '"summary": "Summary without class balance."}'
        )

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = generate_eda_insights(eda_stats_numeric_only)

            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            prompt_text = call_kwargs.get("contents", "")

            # CLASS BALANCE section should NOT appear in the prompt
            assert "CLASS BALANCE" not in prompt_text
            assert "No class balance" not in prompt_text

            # Result structure is correct
            assert len(result["insights"]) == 2
            assert isinstance(result["summary"], str)

    def test_exclude_nonexistent_column(self, eda_stats_typical, monkeypatch):
        """Excluding a nonexistent column is silently ignored."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"insights": ["Fare is right-skewed.", "Age is symmetric."], '
            '"summary": "Summary."}'
        )

        with unittest.mock.patch("google.genai.Client") as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = generate_eda_insights(
                eda_stats_typical,
                exclude_columns=["nonexistent_col"],
            )

            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            prompt_text = call_kwargs.get("contents", "")

            # All original columns should still be present
            assert "age" in prompt_text
            assert "fare" in prompt_text
            assert "pclass" in prompt_text
            assert "survived" in prompt_text
            assert len(result["insights"]) == 2

    @pytest.mark.llm
    def test_live_api(self, eda_stats_typical):
        """Requires a valid GEMINI_API_KEY environment variable."""
        result = generate_eda_insights(eda_stats_typical)

        assert "insights" in result
        assert "summary" in result
        assert len(result["insights"]) > 0
        assert all(isinstance(i, str) for i in result["insights"])
        assert isinstance(result["summary"], str)


# ── run_eda orchestration tests ────────────────────────────────────────────────


class TestRunEda:
    """Tests for run_eda().

    These tests mock ``generate_eda_insights`` because Part 7's job is to verify
    orchestration/wiring, not to re-test Part 6's LLM behaviour (already covered
    in ``TestGenerateEdaInsights``).  Non-LLM parts (stats, histograms, heatmap,
    boxplots, target distribution) run real code.
    """

    @pytest.fixture
    def titanic_with_id(self) -> pd.DataFrame:
        """Titanic-like dataset with a sequential passenger_id column."""
        return pd.DataFrame(
            {
                "passenger_id": range(10),
                "age": [22, 38, 26, 35, 28, 40, 25, 30, 45, 33],
                "fare": [7.25, 71.28, 8.05, 53.10, 15.50, 80.00, 12.00, 22.00, 50.00, 35.00],
                "pclass": [3, 1, 3, 1, 3, 2, 3, 2, 1, 2],
                "survived": [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                "sex": ["male", "female", "male", "female", "male", "female", "male", "female", "female", "male"],
            }
        )

    # ── Full successful runs ──────────────────────────────────────────────────

    def test_full_run_with_target(self, titanic_with_id, tmp_path):
        """All 6 keys populated, errors empty, artifact saved."""
        mock_insights = {"insights": ["Fare is right-skewed."], "summary": "Summary."}
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights", return_value=mock_insights
        ):
            result = run_eda(
                titanic_with_id,
                target_column="survived",
                problem_type="classification",
                artifacts_dir=str(tmp_path),
            )

        assert result["stats"] is not None
        assert result["histograms"] is not None
        assert result["correlation_heatmap"] is not None
        assert result["boxplots"] is not None
        assert result["target_distribution"] is not None
        assert result["insights"] == mock_insights
        assert result["errors"] == {}
        assert os.path.exists(result["artifact_path"])

        # Verify saved file is valid JSON
        with open(result["artifact_path"]) as f:
            saved = json.load(f)
        assert saved["stats"] is not None

    def test_no_target_no_problem_type(self, titanic_with_id, tmp_path):
        """target_distribution is None, all else populated."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(titanic_with_id, artifacts_dir=str(tmp_path))

        assert result["stats"] is not None
        assert result["histograms"] is not None
        assert result["boxplots"] is not None
        assert result["correlation_heatmap"] is not None
        assert result["target_distribution"] is None
        assert result["insights"] is not None

    def test_target_only_no_problem_type(self, titanic_with_id, tmp_path):
        """target_column provided but problem_type=None => target_distribution=None."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(
                titanic_with_id, target_column="survived", artifacts_dir=str(tmp_path)
            )

        assert result["target_distribution"] is None
        assert result["stats"] is not None  # still ran

    def test_problem_type_only_no_target(self, titanic_with_id, tmp_path):
        """problem_type provided but target_column=None => target_distribution=None."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(
                titanic_with_id, problem_type="classification", artifacts_dir=str(tmp_path)
            )

        assert result["target_distribution"] is None
        assert result["stats"] is not None

    # ── Failure isolation ─────────────────────────────────────────────────────

    def test_part_failure_isolation(self, titanic_with_id, tmp_path):
        """One part raises => other parts still run, errors dict populated."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            with unittest.mock.patch(
                "agents.eda_agent.generate_correlation_heatmap",
                side_effect=ValueError("Simulated heatmap failure"),
            ):
                result = run_eda(
                    titanic_with_id,
                    target_column="survived",
                    problem_type="classification",
                    artifacts_dir=str(tmp_path),
                )

        # Other parts populated
        assert result["stats"] is not None
        assert result["histograms"] is not None
        assert result["boxplots"] is not None
        assert result["target_distribution"] is not None
        assert result["insights"] is not None

        # Failed part is None with error recorded
        assert result["correlation_heatmap"] is None
        assert "correlation_heatmap" in result["errors"]
        assert "Simulated heatmap failure" in result["errors"]["correlation_heatmap"]

        # No other errors leaked
        assert set(result["errors"].keys()) == {"correlation_heatmap"}

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_zero_numeric_columns(self, tmp_path):
        """No numeric columns => no crash, empty/None results throughout."""
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["foo", "bar", "baz"]})

        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(df, artifacts_dir=str(tmp_path))

        assert result["stats"] is not None
        # No numeric columns = empty histograms, no correlation matrix, empty boxplots
        assert result["stats"]["skewness"] == {}
        assert result["stats"]["distribution_stats"] == {}
        # Correlation matrix is {} so heatmap code path skipped => None
        assert result["correlation_heatmap"] is None
        assert result["target_distribution"] is None
        assert result["insights"] is not None
        assert isinstance(result["errors"], dict)
        assert os.path.exists(result["artifact_path"])

    # ── Artifact file verification ────────────────────────────────────────────

    def test_artifact_file_valid_json(self, titanic_with_id, tmp_path):
        """Saved artifact matches the returned dict and is valid JSON."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": ["Insight 1"], "summary": "Summary"},
        ):
            result = run_eda(
                titanic_with_id,
                target_column="survived",
                problem_type="classification",
                artifacts_dir=str(tmp_path),
            )

        assert os.path.exists(result["artifact_path"])

        with open(result["artifact_path"]) as f:
            saved = json.load(f)

        assert saved["excluded_columns"] == result["excluded_columns"]
        assert saved["errors"] == result["errors"]
        assert saved["insights"] == result["insights"]

    def test_excluded_columns_populated(self, titanic_with_id, tmp_path):
        """passenger_id is detected as ID-like column and populated in result."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(
                titanic_with_id, target_column="survived", artifacts_dir=str(tmp_path)
            )

        assert "passenger_id" in result["excluded_columns"]

    # ── Cleaning changelog → boxplots ────────────────────────────────────────

    def test_cleaning_changelog_wired_to_boxplots(
        self, titanic_with_id, cleaning_changelog_with_outliers, tmp_path,
    ):
        """Valid changelog with columns_processed → boxplots restricted to those columns."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(
                titanic_with_id,
                target_column="survived",
                problem_type="classification",
                artifacts_dir=str(tmp_path),
                cleaning_changelog=cleaning_changelog_with_outliers,
            )

        assert result["boxplots"] is not None
        box_keys = set(result["boxplots"]["boxplots"].keys())
        # pclass is in the changelog's columns_processed and is not ID-like → should appear
        assert "pclass" in box_keys
        # survived is NOT in the changelog → should NOT appear
        assert "survived" not in box_keys
        # age/fare are ID-like → excluded even if they were in the changelog (they're not)

    def test_cleaning_changelog_none_fallback(
        self, titanic_with_id, tmp_path,
    ):
        """cleaning_changelog=None → default behaviour: all non-excluded numeric columns."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(
                titanic_with_id,
                target_column="survived",
                problem_type="classification",
                artifacts_dir=str(tmp_path),
                cleaning_changelog=None,
            )

        assert result["boxplots"] is not None
        box_keys = set(result["boxplots"]["boxplots"].keys())
        # Numeric columns flagged as ID-like (100% unique in a 10-row dataset):
        # passenger_id, age, fare — all excluded.
        # Non-excluded numeric columns: pclass (3 uniques), survived (2 uniques).
        assert "pclass" in box_keys
        assert "survived" in box_keys
        assert "passenger_id" not in box_keys
        assert "age" not in box_keys
        assert "fare" not in box_keys

    def test_cleaning_changelog_malformed_fallback(
        self, titanic_with_id, cleaning_changelog_malformed, tmp_path,
    ):
        """Malformed changelog (missing outlier_handling) → falls back, doesn't raise."""
        with unittest.mock.patch(
            "agents.eda_agent.generate_eda_insights",
            return_value={"insights": [], "summary": ""},
        ):
            result = run_eda(
                titanic_with_id,
                target_column="survived",
                problem_type="classification",
                artifacts_dir=str(tmp_path),
                cleaning_changelog=cleaning_changelog_malformed,
            )

        # Falls back to default behaviour: all non-excluded numeric columns
        assert result["boxplots"] is not None
        box_keys = set(result["boxplots"]["boxplots"].keys())
        assert "pclass" in box_keys
        assert "survived" in box_keys
        assert "age" not in box_keys
        assert "fare" not in box_keys
        assert result["errors"] == {}