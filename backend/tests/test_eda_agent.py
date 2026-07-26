from __future__ import annotations

import json

import numpy as np
import pytest
import pandas as pd

from agents.eda_agent import compute_eda_stats, generate_histograms, generate_correlation_heatmap


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