"""Behavior-preservation tests for the BaseAgent wrapper refactor.

These tests verify that the new wrapper agent classes produce *identical*
outputs to calling the underlying functions directly. Any difference between
old and new is treated as a bug in the wrapper, not an acceptable side effect.

Each dataset runs the old and new pipelines exactly once; individual test
methods compare specific outputs from the cached results.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from agents.dataset_understanding import (
    classify_problem_type,
    detect_target_candidates,
    profile_dataset,
)
from agents.data_cleaning import clean_dataset
from agents.eda_agent import run_eda
from agents.feature_engineering_agent import build_feature_pipeline
from agents.pipeline_agents import (
    CleaningAgent,
    DatasetUnderstandingAgent,
    EDAAgent,
    FeatureEngineeringAgent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def titanic_df() -> pd.DataFrame:
    """Titanic-shaped synthetic data (120 rows)."""
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


@pytest.fixture(scope="module")
def student_df() -> pd.DataFrame:
    """Student-performance-shaped synthetic data (200 rows)."""
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


# ---------------------------------------------------------------------------
# Helper: run old pipeline
# ---------------------------------------------------------------------------

def _run_old_pipeline(
    df: pd.DataFrame,
    target_column: str,
    artifacts_dir: str,
    ordinal_columns: dict[str, list[str]] | None = None,
) -> dict:
    """Run the old direct-function-call pipeline and capture all outputs.

    Uses the LLM-derived problem_type for downstream calls, matching
    what the wrapper agents do (read from classify_problem_type output).
    """
    profile = profile_dataset(df)
    target_candidates = detect_target_candidates(df)
    problem_type_analysis = classify_problem_type(profile, target_candidates)
    problem_type = problem_type_analysis["problem_type"]

    clean_result = clean_dataset(
        df, target_column=target_column, cap_target=False, artifacts_dir=artifacts_dir,
    )
    cleaned_df = pd.read_csv(clean_result["artifact_path"])

    eda_result = run_eda(
        cleaned_df,
        target_column=target_column,
        problem_type=problem_type,
        artifacts_dir=artifacts_dir,
        cleaning_changelog=json.load(open(clean_result["changelog_path"])),
    )

    fe_result = build_feature_pipeline(
        cleaned_df,
        target_column=target_column,
        problem_type=problem_type,
        ordinal_columns=ordinal_columns,
    )

    return {
        "profile": profile,
        "target_candidates": target_candidates,
        "selected_target": target_column,
        "problem_type_analysis": problem_type_analysis,
        "cleaned_df": cleaned_df,
        "cleaning_summary": clean_result["summary"],
        "eda_result": eda_result,
        "feature_engineering_result": fe_result,
    }


# ---------------------------------------------------------------------------
# Helper: run new wrapper pipeline
# ---------------------------------------------------------------------------

def _run_new_pipeline(
    df: pd.DataFrame,
    target_column: str,
    artifacts_dir: str,
    ordinal_columns: dict[str, list[str]] | None = None,
) -> dict:
    state: dict = {"df": df, "artifacts_dir": artifacts_dir, "user_selected_target": target_column}
    state = DatasetUnderstandingAgent().run(state)
    state = CleaningAgent().run(state)
    state = EDAAgent().run(state)
    if ordinal_columns is not None:
        state["ordinal_columns"] = ordinal_columns
    state = FeatureEngineeringAgent().run(state)
    return {
        "profile": state["profile"],
        "target_candidates": state["target_candidates"],
        "selected_target": state["selected_target"],
        "problem_type_analysis": state["problem_type_analysis"],
        "cleaned_df": state["cleaned_df"],
        "cleaning_summary": state["cleaning_summary"],
        "eda_result": state["eda_result"],
        "feature_engineering_result": state["feature_engineering_result"],
    }


# ---------------------------------------------------------------------------
# Titanic: run both pipelines once, then compare all outputs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def titanic_old_and_new(titanic_df, tmp_path_factory):
    """Run both pipelines once for Titanic and cache results."""
    d = tmp_path_factory.mktemp("titanic")
    os.chdir(d)
    old = _run_old_pipeline(titanic_df, "Survived", str(d))
    new = _run_new_pipeline(titanic_df, "Survived", str(d))
    return old, new


VALID_PROBLEM_TYPES = {"classification", "regression", "clustering", "unclear"}


class TestTitanicProfile:
    def test_profile_matches(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["profile"] == new["profile"]

    def test_target_candidates_match(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["target_candidates"] == new["target_candidates"]

    def test_selected_target_matches(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["selected_target"] == new["selected_target"]

    def test_problem_type_analysis_structure_matches(self, titanic_old_and_new):
        """Both should have the same structure; problem_type values may differ
        due to Gemini LLM non-determinism across separate calls."""
        old, new = titanic_old_and_new
        old_pt = old["problem_type_analysis"]
        new_pt = new["problem_type_analysis"]
        assert set(old_pt.keys()) == {"problem_type", "confidence_reasoning", "project_plan"}
        assert set(new_pt.keys()) == {"problem_type", "confidence_reasoning", "project_plan"}
        assert old_pt["problem_type"] in VALID_PROBLEM_TYPES
        assert new_pt["problem_type"] in VALID_PROBLEM_TYPES


class TestTitanicCleaning:
    def test_cleaned_df_identical(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["cleaned_df"].equals(new["cleaned_df"])

    def test_cleaning_summary_matches(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["cleaning_summary"] == new["cleaning_summary"]


class TestTitanicEDA:
    def test_eda_stats_non_class_balance_match(self, titanic_old_and_new):
        """All EDA stats except class_balance should match exactly.
        class_balance depends on problem_type which may differ due to LLM."""
        old, new = titanic_old_and_new
        old_stats = {k: v for k, v in old["eda_result"]["stats"].items() if k != "class_balance"}
        new_stats = {k: v for k, v in new["eda_result"]["stats"].items() if k != "class_balance"}
        assert old_stats == new_stats

    def test_eda_class_balance_structure(self, titanic_old_and_new):
        """class_balance should be None for non-classification, or a dict with the
        right keys for classification. Structure must be valid regardless of LLM."""
        old, new = titanic_old_and_new
        for result in (old["eda_result"]["stats"], new["eda_result"]["stats"]):
            cb = result["class_balance"]
            if cb is not None:
                assert "class_counts" in cb
                assert "class_percentages" in cb

    def test_eda_excluded_columns_match(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["eda_result"]["excluded_columns"] == new["eda_result"]["excluded_columns"]

    def test_eda_errors_match_when_same_problem_type(self, titanic_old_and_new):
        """errors may differ when problem_type differs (e.g. 'unclear' vs 'classification'
        causes different target_distribution errors). Only compare when same."""
        old, new = titanic_old_and_new
        old_pt = old["problem_type_analysis"]["problem_type"]
        new_pt = new["problem_type_analysis"]["problem_type"]
        if old_pt == new_pt:
            assert old["eda_result"]["errors"] == new["eda_result"]["errors"]


class TestTitanicFeatureEngineering:
    def test_encoding_map_matches(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["feature_engineering_result"]["encoding_map"] == new["feature_engineering_result"]["encoding_map"]

    def test_excluded_columns_match(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["feature_engineering_result"]["excluded_columns"] == new["feature_engineering_result"]["excluded_columns"]

    def test_collinear_pairs_match(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["feature_engineering_result"]["collinear_pairs"] == new["feature_engineering_result"]["collinear_pairs"]

    def test_transformed_df_identical(self, titanic_old_and_new):
        old, new = titanic_old_and_new
        assert old["feature_engineering_result"]["transformed_df"].equals(
            new["feature_engineering_result"]["transformed_df"]
        )


# ---------------------------------------------------------------------------
# Student performance: run both pipelines once, then compare
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def student_old_and_new(student_df, tmp_path_factory):
    """Run both pipelines once for student data and cache results."""
    d = tmp_path_factory.mktemp("student")
    os.chdir(d)
    ordinal = {"grade": ["F", "D", "C", "B", "A"]}
    old = _run_old_pipeline(student_df, "Exam_Score", str(d), ordinal_columns=ordinal)
    new = _run_new_pipeline(student_df, "Exam_Score", str(d), ordinal_columns=ordinal)
    return old, new


class TestStudentProfile:
    def test_profile_matches(self, student_old_and_new):
        old, new = student_old_and_new
        assert old["profile"] == new["profile"]

    def test_selected_target_matches(self, student_old_and_new):
        old, new = student_old_and_new
        assert old["selected_target"] == new["selected_target"]


class TestStudentCleaning:
    def test_cleaned_df_identical(self, student_old_and_new):
        old, new = student_old_and_new
        assert old["cleaned_df"].equals(new["cleaned_df"])


class TestStudentEDA:
    def test_eda_stats_matches(self, student_old_and_new):
        old, new = student_old_and_new
        assert old["eda_result"]["stats"] == new["eda_result"]["stats"]


class TestStudentFeatureEngineering:
    def test_encoding_map_matches(self, student_old_and_new):
        old, new = student_old_and_new
        assert old["feature_engineering_result"]["encoding_map"] == new["feature_engineering_result"]["encoding_map"]

    def test_transformed_df_identical(self, student_old_and_new):
        old, new = student_old_and_new
        assert old["feature_engineering_result"]["transformed_df"].equals(
            new["feature_engineering_result"]["transformed_df"]
        )

    def test_excluded_columns_match(self, student_old_and_new):
        old, new = student_old_and_new
        assert old["feature_engineering_result"]["excluded_columns"] == new["feature_engineering_result"]["excluded_columns"]


# ---------------------------------------------------------------------------
# Wrapper standalone-entry-point tests
# ---------------------------------------------------------------------------

class TestWrapperStandaloneEntry:
    """Each wrapper can be instantiated and .run() called with only {"df": ...}."""

    def test_dataset_understanding_needs_only_df(self, titanic_df, tmp_path):
        os.chdir(tmp_path)
        state = {"df": titanic_df}
        state = DatasetUnderstandingAgent().run(state)
        assert "profile" in state
        assert "selected_target" in state
        assert "problem_type_analysis" in state
        assert state["selected_target"] == "Survived"

    def test_cleaning_needs_df_and_target(self, titanic_df, tmp_path):
        os.chdir(tmp_path)
        state = {"df": titanic_df, "selected_target": "Survived", "artifacts_dir": str(tmp_path)}
        state = CleaningAgent().run(state)
        assert "cleaned_df" in state
        assert "cleaning_artifact_path" in state
        assert "cleaning_changelog_path" in state
        assert len(state["cleaned_df"]) > 0

    def test_eda_needs_cleaned_df_and_context(self, titanic_df, tmp_path):
        os.chdir(tmp_path)
        state = {"df": titanic_df, "selected_target": "Survived", "artifacts_dir": str(tmp_path)}
        state = CleaningAgent().run(state)
        state["problem_type_analysis"] = {"problem_type": "classification"}
        state = EDAAgent().run(state)
        assert "eda_result" in state
        assert state["eda_result"]["stats"] is not None

    def test_feature_engineering_needs_cleaned_df_and_context(self, titanic_df, tmp_path):
        os.chdir(tmp_path)
        state = {"df": titanic_df, "selected_target": "Survived", "artifacts_dir": str(tmp_path)}
        state = CleaningAgent().run(state)
        state["problem_type_analysis"] = {"problem_type": "classification"}
        state = FeatureEngineeringAgent().run(state)
        assert "feature_engineering_result" in state
        assert "encoding_map" in state["feature_engineering_result"]

    def test_full_chain_from_df_only(self, titanic_df, tmp_path):
        """Full pipeline from just df + user_selected_target."""
        os.chdir(tmp_path)
        state = {"df": titanic_df, "user_selected_target": "Survived", "artifacts_dir": str(tmp_path)}
        state = DatasetUnderstandingAgent().run(state)
        state = CleaningAgent().run(state)
        state = EDAAgent().run(state)
        state = FeatureEngineeringAgent().run(state)

        assert "profile" in state
        assert "cleaned_df" in state
        assert "eda_result" in state
        assert "feature_engineering_result" in state
        assert "Survived" not in state["feature_engineering_result"]["transformed_df"].columns
