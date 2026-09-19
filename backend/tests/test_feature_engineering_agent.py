from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from agents.dataset_understanding import identify_id_columns
from agents.feature_engineering_agent import build_feature_pipeline


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic mixed dataset designed so ONLY passenger_id is ID-like.

    Cardinality is deliberately capped well below the 95% ID-detection
    threshold (age=50/120, fare=80/120, category_code=25/120) so the routing
    tests exercise the intended branches rather than ID-detection side effects.
    """
    n = 120
    return pd.DataFrame(
        {
            "passenger_id": range(n),
            "age": [20 + (i % 50) for i in range(n)],
            "fare": [5 + (i % 80) for i in range(n)],
            "sex": ["male", "female"] * (n // 2),
            "embarked": (["S", "C", "Q"] * (n // 3 + 1))[:n],
            "quality": (["bad", "good", "excellent"] * (n // 3 + 1))[:n],
            "category_code": [f"code_{i % 25}" for i in range(n)],
            "survived": [0, 1] * (n // 2),
        }
    )


class TestBuildFeaturePipeline:
    """Unit tests for build_feature_pipeline() routing and output structure."""

    def test_target_never_in_transformed_columns(self, feature_df):
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
        )
        assert "survived" not in result["transformed_df"].columns
        assert result["encoding_map"]["survived"] == "excluded"

    def test_id_like_columns_excluded(self, feature_df):
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
        )
        assert "passenger_id" in result["excluded_columns"]
        assert "passenger_id" not in result["transformed_df"].columns

    def test_ordinal_column_rank_encoding_not_onehot(self, feature_df):
        """3-unique ordinal column (<= threshold) must still be ordinal-encoded,
        NOT one-hot, because it was declared in ordinal_columns."""
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
            ordinal_columns={"quality": ["bad", "good", "excellent"]},
        )
        tdf = result["transformed_df"]

        assert "ordinal__quality" in tdf.columns
        assert result["encoding_map"]["quality"] == "ordinal"
        # bad -> 0, good -> 1, excellent -> 2 (rank order, not alphabetical)
        for value, expected in (("bad", 0), ("good", 1), ("excellent", 2)):
            mask = feature_df["quality"] == value
            assert (tdf.loc[mask, "ordinal__quality"] == expected).all()
        # NOT routed to one-hot despite having only 3 unique values
        assert not any(c.startswith("onehot__quality") for c in tdf.columns)

    def test_ordinal_column_with_unseen_value_raises(self, feature_df):
        df = feature_df.copy()
        df.loc[0, "quality"] = "terrible"  # not in the rank list
        with pytest.raises(ValueError, match="quality"):
            build_feature_pipeline(
                df,
                target_column="survived",
                problem_type="classification",
                ordinal_columns={"quality": ["bad", "good", "excellent"]},
            )

    def test_low_cardinality_non_ordinal_categorical_still_one_hot(self, feature_df):
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
            ordinal_columns={"quality": ["bad", "good", "excellent"]},
        )
        tdf = result["transformed_df"]
        assert result["encoding_map"]["sex"] == "onehot"
        assert "onehot__sex_male" in tdf.columns
        assert "onehot__sex_female" in tdf.columns

    def test_high_cardinality_categorical_label_encoded(self, feature_df):
        """High-cardinality categorical uses OrdinalEncoder (not LabelEncoder)
        and is tagged 'label' in encoding_map."""
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
        )
        tdf = result["transformed_df"]

        assert result["encoding_map"]["category_code"] == "label"
        assert "label__category_code" in tdf.columns
        # 25 distinct categories -> integer mapping, not one-hot explosion
        assert tdf["label__category_code"].nunique() == 25
        assert not any(c.startswith("onehot__category_code") for c in tdf.columns)

        # The actual transformer behind the "label" branch is an OrdinalEncoder,
        # never a LabelEncoder.
        ct = result["pipeline"].named_steps["preprocessing"]
        label_transformer = next(
            t for name, t, _ in ct.transformers_ if name == "label"
        )
        encoder = (
            label_transformer.named_steps["label"]
            if hasattr(label_transformer, "named_steps")
            else label_transformer
        )
        assert isinstance(encoder, OrdinalEncoder)
        assert not isinstance(encoder, LabelEncoder)

    def test_transform_held_out_unseen_category_no_raise(self, feature_df):
        """handle_unknown='use_encoded_value' must actually be wired in: a
        category unseen at fit time encodes to -1 instead of raising."""
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
        )
        held_out = feature_df.copy()
        held_out.loc[0, "category_code"] = "unseen_at_fit_time"

        X = result["pipeline"].transform(held_out)  # must NOT raise
        if hasattr(X, "toarray"):
            X = X.toarray()

        names = list(result["pipeline"].get_feature_names_out())
        label_idx = names.index("label__category_code")
        assert X[0, label_idx] == -1

    def test_numeric_column_scaled(self, feature_df):
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
        )
        vals = result["transformed_df"]["scaled__age"]
        assert result["encoding_map"]["age"] == "scaled"
        # StandardScaler normalises with population std (ddof=0)
        assert vals.mean() == pytest.approx(0.0, abs=1e-6)
        assert vals.std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    def test_collinear_pairs_flagged_but_not_dropped(self):
        """Two columns forced to ~0.99 correlation: flagged in collinear_pairs,
        but both still present in transformed_df."""
        n = 200
        df = pd.DataFrame(
            {
                "x": [i % 50 for i in range(n)],
                "y": [2 * (i % 50) + (i % 7) for i in range(n)],
                "z": [i % 17 for i in range(n)],
                "target": [0, 1] * (n // 2),
            }
        )
        result = build_feature_pipeline(
            df, target_column="target", problem_type="classification"
        )

        pair = next(
            p
            for p in result["collinear_pairs"]
            if {p["col_a"], p["col_b"]} == {"x", "y"}
        )
        assert pair["correlation"] > 0.85

        tdf = result["transformed_df"]
        assert "scaled__x" in tdf.columns
        assert "scaled__y" in tdf.columns

    def test_encoding_map_covers_every_original_column(self, feature_df):
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
            ordinal_columns={"quality": ["bad", "good", "excellent"]},
        )
        expected_columns = {str(c) for c in feature_df.columns}
        assert set(result["encoding_map"].keys()) == expected_columns

        # excluded + target tagged "excluded"; ordinal tagged "ordinal"
        assert result["encoding_map"]["survived"] == "excluded"
        assert result["encoding_map"]["passenger_id"] == "excluded"
        assert result["encoding_map"]["quality"] == "ordinal"
        # every value is one of the allowed tags
        assert set(result["encoding_map"].values()) <= {
            "onehot", "label", "ordinal", "scaled", "excluded",
        }

    def test_ordinal_columns_prioritised_over_cardinality_route(self):
        """A high-cardinality column declared ordinal must take the 'ordinal'
        route (rank list), not the 'label' route."""
        n = 120
        df = pd.DataFrame(
            {
                "quality": [f"level_{i % 20}" for i in range(n)],
                "target": [0, 1] * (n // 2),
            }
        )
        rank_list = [f"level_{i}" for i in range(20)]
        result = build_feature_pipeline(
            df,
            target_column="target",
            problem_type="classification",
            ordinal_columns={"quality": rank_list},
        )
        assert result["encoding_map"]["quality"] == "ordinal"
        assert "ordinal__quality" in result["transformed_df"].columns
        assert "label__quality" not in result["transformed_df"].columns

    def test_missing_target_column_raises(self, feature_df):
        with pytest.raises(ValueError, match="target_column 'nope' is not present"):
            build_feature_pipeline(
                feature_df, target_column="nope", problem_type="classification"
            )

    def test_missing_ordinal_column_raises(self, feature_df):
        with pytest.raises(ValueError, match="ordinal column 'nope'"):
            build_feature_pipeline(
                feature_df,
                target_column="survived",
                problem_type="classification",
                ordinal_columns={"nope": ["a", "b"]},
            )

    def test_pipeline_reusable_single_preprocessing_step(self, feature_df):
        """The returned Pipeline must be directly reusable in Week 5: a model
        step can be appended to the same object."""
        result = build_feature_pipeline(
            feature_df,
            target_column="survived",
            problem_type="classification",
        )
        pipeline = result["pipeline"]
        assert list(pipeline.named_steps.keys()) == ["preprocessing"]
        from sklearn.linear_model import LogisticRegression

        pipeline.steps.append(("model", LogisticRegression(max_iter=200)))
        assert "model" in pipeline.named_steps
        pipeline.fit(
            feature_df, feature_df["survived"]
        )  # refits preprocessor + model end-to-end
        assert hasattr(pipeline, "predict")


class TestBuildFeaturePipelineCrossDataset:
    """Cross-dataset validation (same pattern as Week 3 Part 8)."""

    @pytest.fixture
    def titanic_like(self) -> pd.DataFrame:
        """Titanic-shaped synthetic data. PassengerId and Name are near-unique
        (ID-like); Age/Fare are numeric; Sex/Embarked low-cardinality
        categoricals; Survived is the classification target."""
        n = 120
        return pd.DataFrame(
            {
                "PassengerId": range(1, n + 1),
                "Name": [f"Passenger_{i}" for i in range(n)],
                "Age": [20 + (i % 60) for i in range(n)],
                "Fare": [10 + (i % 90) for i in range(n)],
                "Sex": ["male", "female"] * (n // 2),
                "Embarked": (["S", "C", "Q"] * (n // 3 + 1))[:n],
                "Survived": [0, 1] * (n // 2),
            }
        )

    @pytest.fixture
    def student_like(self) -> pd.DataFrame:
        """Student-performance-shaped synthetic data mirroring
        StudentPerformanceFactors.csv: no ID-like columns, several low-cardinal
        categoricals, an ordinal grade column (letters), and a numeric target."""
        rng = np.random.default_rng(11)
        n = 200
        return pd.DataFrame(
            {
                "Hours_Studied": [i % 24 for i in range(n)],
                "Attendance": [i % 100 for i in range(n)],
                "Sleep_Hours": [i % 12 for i in range(n)],
                "Previous_Scores": [40 + (i % 60) for i in range(n)],
                "Parental_Involvement": rng.choice(["Low", "Medium", "High"], n),
                "Access_to_Resources": rng.choice(["Low", "Medium", "High"], n),
                "Motivation_Level": rng.choice(["Low", "Medium", "High"], n),
                "Family_Income": rng.choice(["Low", "Medium", "High"], n),
                "Teacher_Quality": rng.choice(["Low", "Medium", "High"], n),
                "Peer_Influence": rng.choice(["Negative", "Neutral", "Positive"], n),
                "School_Type": rng.choice(["Public", "Private"], n),
                "Gender": rng.choice(["Male", "Female"], n),
                "Extracurricular_Activities": rng.choice(["Yes", "No"], n),
                "Internet_Access": rng.choice(["Yes", "No"], n),
                "Learning_Disabilities": rng.choice(["Yes", "No"], n),
                "Distance_from_Home": rng.choice(["Near", "Moderate", "Far"], n),
                "Tutoring_Sessions": [i % 9 for i in range(n)],
                "grade": rng.choice(["A", "B", "C", "D", "F"], n),
                "Exam_Score": [30 + (i % 70) for i in range(n)],
            }
        )

    def test_titanic_like_routing(self, titanic_like):
        result = build_feature_pipeline(
            titanic_like,
            target_column="Survived",
            problem_type="classification",
        )
        tdf = result["transformed_df"]
        em = result["encoding_map"]

        # Age/Fare scaled; Sex/Embarked one-hot
        assert "scaled__Age" in tdf.columns
        assert "scaled__Fare" in tdf.columns
        assert em["Age"] == "scaled"
        assert em["Fare"] == "scaled"
        assert em["Sex"] == "onehot"
        assert em["Embarked"] == "onehot"
        assert "onehot__Sex_male" in tdf.columns
        assert "onehot__Sex_female" in tdf.columns
        assert "onehot__Embarked_S" in tdf.columns

        # PassengerId/Name excluded (ID-like), Survived never a feature
        assert "PassengerId" in result["excluded_columns"]
        assert "Name" in result["excluded_columns"]
        assert em["PassengerId"] == "excluded"
        assert em["Name"] == "excluded"
        assert "Survived" not in tdf.columns

    def test_student_identify_id_columns_empty(self, student_like):
        """Re-check the Week 3 open question: identify_id_columns() returns []
        for the student dataset, so nothing is excluded beyond the target."""
        assert identify_id_columns(student_like) == []
        result = build_feature_pipeline(
            student_like,
            target_column="Exam_Score",
            problem_type="regression",
            ordinal_columns={"grade": ["F", "D", "C", "B", "A"]},
        )
        assert result["excluded_columns"] == ["Exam_Score"]

    def test_student_grade_letters_ordinal_not_onehot(self, student_like):
        """Grade letters are genuinely ordinal: declared in ordinal_columns,
        they go through rank-ordered encoding instead of one-hot."""
        result = build_feature_pipeline(
            student_like,
            target_column="Exam_Score",
            problem_type="regression",
            ordinal_columns={"grade": ["F", "D", "C", "B", "A"]},
        )
        tdf = result["transformed_df"]
        assert result["encoding_map"]["grade"] == "ordinal"
        assert "ordinal__grade" in tdf.columns
        assert not any(c.startswith("onehot__grade") for c in tdf.columns)

        # Rank order: F -> 0, D -> 1, C -> 2, B -> 3, A -> 4
        for value, expected in (("F", 0), ("D", 1), ("C", 2), ("B", 3), ("A", 4)):
            mask = student_like["grade"] == value
            assert (tdf.loc[mask, "ordinal__grade"] == expected).all()

    def test_student_low_cardinality_categoricals_onehot(self, student_like):
        """Low-cardinality categoricals default to one-hot when not declared
        ordinal."""
        result = build_feature_pipeline(
            student_like,
            target_column="Exam_Score",
            problem_type="regression",
            ordinal_columns={"grade": ["F", "D", "C", "B", "A"]},
        )
        tdf = result["transformed_df"]
        assert result["encoding_map"]["Gender"] == "onehot"
        assert "onehot__Gender_Male" in tdf.columns
        assert "onehot__Gender_Female" in tdf.columns
        assert result["encoding_map"]["Parental_Involvement"] == "onehot"

    def test_student_numeric_features_scaled(self, student_like):
        result = build_feature_pipeline(
            student_like,
            target_column="Exam_Score",
            problem_type="regression",
            ordinal_columns={"grade": ["F", "D", "C", "B", "A"]},
        )
        tdf = result["transformed_df"]
        assert "scaled__Hours_Studied" in tdf.columns
        assert "scaled__Attendance" in tdf.columns
        assert tdf["scaled__Hours_Studied"].mean() == pytest.approx(0.0, abs=1e-6)


class TestNanLeakageFixAndGuardrails:
    """Tests for NaN-leakage prevention, SimpleImputer integration, and save-time validation."""

    @pytest.fixture
    def titanic_with_cabin(self) -> pd.DataFrame:
        """Titanic dataset with high-missing Cabin and moderate-missing Age."""
        n = 100
        return pd.DataFrame(
            {
                "PassengerId": range(1, n + 1),
                "Name": [f"Passenger_{i}" for i in range(n)],
                "Age": [22.0 + (i % 40) if i % 5 != 0 else np.nan for i in range(n)],
                "Fare": [7.25 + (i % 50) for i in range(n)],
                "Sex": ["male", "female"] * (n // 2),
                "Cabin": [f"C{i}" if i < 20 else np.nan for i in range(n)],  # 80% missing
                "Embarked": (["S", "C", "Q"] * (n // 3 + 1))[:n],
                "Survived": [0, 1] * (n // 2),
            }
        )

    def test_titanic_has_zero_nans_and_excludes_cabin(self, titanic_with_cabin):
        """1. Regression test: Cabin (80% missing) is excluded and output is strictly NaN-free."""
        result = build_feature_pipeline(
            titanic_with_cabin,
            target_column="Survived",
            problem_type="classification",
        )
        tdf = result["transformed_df"]
        em = result["encoding_map"]

        # Assert Cabin is excluded
        assert "Cabin" in result["excluded_columns"]
        assert em["Cabin"] == "excluded"
        assert not any("Cabin" in col for col in tdf.columns)

        # Assert ZERO NaNs across all columns
        assert tdf.isna().sum().sum() == 0

    def test_imputer_handles_residual_nans_across_all_column_types(self):
        """2. SimpleImputer guarantees NaN-free matrix for numeric, categorical, and ordinal paths."""
        df = pd.DataFrame(
            {
                "num_feat": [10.0, 20.0, np.nan, 40.0, 50.0],
                "cat_low": ["A", "B", np.nan, "A", "B"],
                "cat_high": [f"code_{i}" if i != 2 else np.nan for i in range(5)],
                "ord_feat": ["low", "high", np.nan, "medium", "high"],
                "target": [1, 0, 1, 0, 1],
            }
        )
        result = build_feature_pipeline(
            df,
            target_column="target",
            problem_type="classification",
            ordinal_columns={"ord_feat": ["low", "medium", "high"]},
            cardinality_threshold=3,
        )
        tdf = result["transformed_df"]
        assert tdf.isna().sum().sum() == 0
        assert len(tdf) == 5

    def test_save_time_guardrail_raises_on_nans(self, monkeypatch):
        """3. Save-time validation guardrail raises clear error if NaNs exist in output."""
        import sklearn.pipeline
        n = 10
        df = pd.DataFrame({"feat": [1.0, 2.0] * 5, "target": [0, 1] * 5})
        orig_fit_transform = sklearn.pipeline.Pipeline.fit_transform

        def bad_fit_transform(self, X, y=None, **kwargs):
            orig_fit_transform(self, X, y, **kwargs)
            arr = np.ones((n, 1))
            arr[2, 0] = np.nan
            return arr

        monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit_transform", bad_fit_transform)
        with pytest.raises(ValueError, match="Validation failed: transformed_df contains NaN values"):
            build_feature_pipeline(df, target_column="target", problem_type="classification")

    def test_student_performance_dataset_zero_nans(self):
        """4. Verify zero-NaN guarantee holds across student dataset with injected missingness."""
        rng = np.random.default_rng(11)
        n = 100
        df = pd.DataFrame(
            {
                "Hours_Studied": [float(i % 24) for i in range(n)],
                "Attendance": [float(i % 100) for i in range(n)],
                "Gender": rng.choice(["Male", "Female"], n),
                "grade": rng.choice(["F", "D", "C", "B", "A"], n),
                "Exam_Score": [30.0 + (i % 70) for i in range(n)],
            }
        )
        df.loc[0:10, "Hours_Studied"] = np.nan
        df.loc[5:15, "Gender"] = np.nan
        df.loc[10:20, "grade"] = np.nan

        result = build_feature_pipeline(
            df,
            target_column="Exam_Score",
            problem_type="regression",
            ordinal_columns={"grade": ["F", "D", "C", "B", "A"]},
        )
        tdf = result["transformed_df"]
        assert tdf.isna().sum().sum() == 0

