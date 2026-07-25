from __future__ import annotations

import pytest
import pandas as pd

from agents.eda_agent import compute_eda_stats


class TestComputeEdaStatsValidInputs:
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
