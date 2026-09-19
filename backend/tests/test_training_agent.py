from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from agents.data_cleaning import clean_dataset
from agents.dataset_understanding import classify_problem_type, profile_dataset
from agents.eda_agent import run_eda
from agents.feature_engineering_agent import build_feature_pipeline
from agents.pipeline_agents import (
    CleaningAgent,
    DatasetUnderstandingAgent,
    EDAAgent,
    FeatureEngineeringAgent,
    MLPlanningAgent,
    TrainingAgent,
)
from agents.training_agent import MODEL_MAPPING, get_estimator
from agents.versioning_utils import get_latest_version_path, save_artifact


@pytest.fixture
def titanic_fixture() -> pd.DataFrame:
    """Standard Titanic fixture with 120 rows."""
    n = 120
    return pd.DataFrame(
        {
            "PassengerId": range(1, n + 1),
            "Name": [f"Passenger_{i}" for i in range(n)],
            "Age": [20 + (i % 60) for i in range(n)],
            "Fare": [10.0 + (i % 90) for i in range(n)],
            "Sex": ["male", "female"] * (n // 2),
            "Embarked": (["S", "C", "Q"] * (n // 3 + 1))[:n],
            "Survived": [0, 1] * (n // 2),
        }
    )


@pytest.fixture
def student_fixture() -> pd.DataFrame:
    """Standard Student Performance fixture for regression with 150 rows."""
    rng = np.random.default_rng(42)
    n = 150
    return pd.DataFrame(
        {
            "Hours_Studied": [i % 24 for i in range(n)],
            "Attendance": [50 + (i % 50) for i in range(n)],
            "Sleep_Hours": [i % 12 for i in range(n)],
            "Previous_Scores": [40 + (i % 60) for i in range(n)],
            "Parental_Involvement": rng.choice(["Low", "Medium", "High"], n),
            "Exam_Score": [30.0 + float(i % 70) for i in range(n)],
        }
    )


def _setup_pipeline_artifacts(
    df: pd.DataFrame,
    target_column: str,
    artifacts_dir: str,
    problem_type: str = "classification",
    recommended_metric: str = "F1",
    candidate_models: list[dict[str, str]] | None = None,
) -> dict:
    """Helper to run Steps 1-4 and ML plan saving, producing all input artifacts."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Dataset Profile
    profile = profile_dataset(df)
    save_artifact(
        {
            "profile": profile,
            "selected_target": target_column,
            "problem_type_analysis": {"problem_type": problem_type},
        },
        artifacts_dir,
        "dataset_profile",
        "json",
        version=1,
    )

    # 2. Cleaned
    clean_res = clean_dataset(
        df,
        target_column=target_column,
        cap_target=False,
        artifacts_dir=artifacts_dir,
    )

    # 3. EDA
    cleaned_df = pd.read_csv(clean_res["artifact_path"])
    run_eda(
        cleaned_df,
        target_column=target_column,
        problem_type=problem_type,
        artifacts_dir=artifacts_dir,
    )

    # 4. Feature Engineering
    fe_res = build_feature_pipeline(
        cleaned_df,
        target_column=target_column,
        problem_type=problem_type,
    )
    save_artifact(
        fe_res["transformed_df"],
        artifacts_dir,
        "feature_engineered",
        "csv",
        version=1,
    )
    save_artifact(
        {
            "columns": list(fe_res["transformed_df"].columns),
            "target_column": target_column,
            "problem_type": problem_type,
            "encoding_map": fe_res["encoding_map"],
            "excluded_columns": fe_res["excluded_columns"],
        },
        artifacts_dir,
        "feature_engineered",
        "json",
        version=1,
    )

    # 5. ML Plan
    if candidate_models is None:
        if problem_type == "classification":
            candidate_models = [
                {"model_name": "Logistic Regression", "reasoning": "Linear baseline."},
                {"model_name": "Random Forest Classifier", "reasoning": "Ensemble."},
                {"model_name": "XGBoost Classifier", "reasoning": "Gradient boosting."},
                {"model_name": "LightGBM Classifier", "reasoning": "Fast boosting."},
                {"model_name": "SVC", "reasoning": "Support vector."},
            ]
        else:
            candidate_models = [
                {"model_name": "Linear Regression", "reasoning": "Linear baseline."},
                {"model_name": "Random Forest Regressor", "reasoning": "Ensemble."},
                {"model_name": "XGBoost Regressor", "reasoning": "Gradient boosting."},
                {"model_name": "LightGBM Regressor", "reasoning": "Fast boosting."},
            ]

    ml_plan = {
        "problem_type": problem_type,
        "confirmation_reasoning": "Confirmed via dataset profiling.",
        "recommended_metric": recommended_metric,
        "metric_reasoning": "Standard suited metric.",
        "candidate_models": candidate_models,
    }
    save_artifact(ml_plan, artifacts_dir, "ml_plan", "json", version=1)

    return {
        "transformed_df": fe_res["transformed_df"],
        "cleaned_df": cleaned_df,
        "target_column": target_column,
        "ml_plan": ml_plan,
        "artifacts_dir": artifacts_dir,
    }


# ===========================================================================
# 1. Unit Test: Model name-to-class mapping
# ===========================================================================

def test_model_name_to_class_mapping_classification():
    """Assert all confirmed classification model names resolve to exact estimator classes with defaults."""
    # Logistic Regression
    lr = get_estimator("Logistic Regression", "classification")
    assert isinstance(lr, LogisticRegression)
    assert lr.max_iter == 1000
    assert lr.random_state == 42

    # Random Forest Classifier
    rf = get_estimator("Random Forest Classifier", "classification")
    assert isinstance(rf, RandomForestClassifier)
    assert rf.random_state == 42

    # XGBoost Classifier
    xgb = get_estimator("XGBoost Classifier", "classification")
    assert isinstance(xgb, XGBClassifier)
    assert xgb.random_state == 42

    # LightGBM Classifier
    lgb = get_estimator("LightGBM Classifier", "classification")
    assert isinstance(lgb, LGBMClassifier)
    assert lgb.random_state == 42

    # SVC
    svc = get_estimator("SVC", "classification")
    assert isinstance(svc, SVC)
    assert svc.probability is True
    assert svc.random_state == 42


def test_model_name_to_class_mapping_regression():
    """Assert all regression model names resolve to exact estimator classes with defaults."""
    lr = get_estimator("Linear Regression", "regression")
    assert isinstance(lr, LinearRegression)

    rf = get_estimator("Random Forest Regressor", "regression")
    assert isinstance(rf, RandomForestRegressor)
    assert rf.random_state == 42

    xgb = get_estimator("XGBoost Regressor", "regression")
    assert isinstance(xgb, XGBRegressor)
    assert xgb.random_state == 42

    lgb = get_estimator("LightGBM Regressor", "regression")
    assert isinstance(lgb, LGBMRegressor)
    assert lgb.random_state == 42


# ===========================================================================
# 2. Unrecognized model name fallback test (STRENGTHENED)
# ===========================================================================

def test_unrecognized_model_names_fallback_and_distinct_logging(titanic_fixture, tmp_path, monkeypatch):
    """Pass candidate_models containing 3 unsupported names mixed with 2 valid ones.
    Assert all 3 are skipped with status='skipped_unrecognized', 2 valid train normally,
    run completes, and log_agent_run captures all 3 skips distinctly."""
    artifacts_dir = str(tmp_path / "artifacts")
    candidates = [
        {"model_name": "CatBoost Classifier", "reasoning": "Unsupported gradient boosting library."},
        {"model_name": "Logistic Regression", "reasoning": "Valid linear model."},
        {"model_name": "K-Nearest Neighbors", "reasoning": "Unsupported neighbor model."},
        {"model_name": "Random Forest Classifier", "reasoning": "Valid ensemble."},
        {"model_name": "Naive Bayes", "reasoning": "Unsupported probabilistic classifier."},
    ]
    _setup_pipeline_artifacts(
        titanic_fixture,
        target_column="Survived",
        artifacts_dir=artifacts_dir,
        problem_type="classification",
        candidate_models=candidates,
    )

    logged_runs: list[dict[str, Any]] = []

    def mock_log(agent_name, inputs_summary, outputs_summary, duration_seconds, error=None):
        logged_runs.append({
            "agent_name": agent_name,
            "inputs": inputs_summary,
            "outputs": outputs_summary,
            "error": error,
        })

    import agents.training_agent
    monkeypatch.setattr(agents.training_agent, "log_agent_run", mock_log)

    agent = TrainingAgent()
    state = agent.run({"artifacts_dir": artifacts_dir})

    results = state["training_results"]
    assert len(results) == 5

    # Check the 3 unrecognized models
    skipped_results = [r for r in results if r["status"] == "skipped_unrecognized"]
    assert len(skipped_results) == 3
    skipped_names = {r["model_name"] for r in skipped_results}
    assert skipped_names == {"CatBoost Classifier", "K-Nearest Neighbors", "Naive Bayes"}

    for r in skipped_results:
        assert r["score"] is None
        assert r["metrics"] is None
        assert r["training_time_seconds"] == 0.0
        assert r["model_name"] in r["error_message"]
        assert "Unrecognized" in r["error_message"]

    # Check the 2 valid models
    success_results = [r for r in results if r["status"] == "success"]
    assert len(success_results) == 2
    success_names = {r["model_name"] for r in success_results}
    assert success_names == {"Logistic Regression", "Random Forest Classifier"}
    for r in success_results:
        assert r["score"] is not None
        assert r["score"] > 0.0
        assert r["error_message"] is None

    # Check that log_agent_run captured all 3 skip events distinctly
    skip_logs = [
        l for l in logged_runs
        if l["outputs"].get("status") == "skipped_unrecognized"
    ]
    assert len(skip_logs) == 3
    logged_skipped_models = {l["inputs"]["model_name"] for l in skip_logs}
    assert logged_skipped_models == {"CatBoost Classifier", "K-Nearest Neighbors", "Naive Bayes"}


# ===========================================================================
# 3. Failure recovery test (fit-time failure)
# ===========================================================================

def test_fit_failure_recovery(titanic_fixture, tmp_path, monkeypatch):
    """Mock a valid model's .fit() to throw; assert run completes, failed model is recorded,
    other models train normally with real scores, and log_agent_run captures error."""
    artifacts_dir = str(tmp_path / "artifacts")
    candidates = [
        {"model_name": "Logistic Regression", "reasoning": "Valid linear model."},
        {"model_name": "Random Forest Classifier", "reasoning": "Will throw during fit."},
        {"model_name": "SVC", "reasoning": "Valid support vector."},
    ]
    _setup_pipeline_artifacts(
        titanic_fixture,
        target_column="Survived",
        artifacts_dir=artifacts_dir,
        problem_type="classification",
        candidate_models=candidates,
    )

    # Patch RandomForestClassifier.fit to throw an exception
    orig_fit = RandomForestClassifier.fit
    def throwing_fit(self, *args, **kwargs):
        raise RuntimeError("Simulated internal GPU/memory training error")

    monkeypatch.setattr(RandomForestClassifier, "fit", throwing_fit)

    logged_runs = []
    def mock_log(agent_name, inputs_summary, outputs_summary, duration_seconds, error=None):
        logged_runs.append({
            "agent_name": agent_name,
            "inputs": inputs_summary,
            "outputs": outputs_summary,
            "error": error,
        })

    import agents.training_agent
    monkeypatch.setattr(agents.training_agent, "log_agent_run", mock_log)

    agent = TrainingAgent()
    state = agent.run({"artifacts_dir": artifacts_dir})

    results = state["training_results"]
    assert len(results) == 3

    # Failed model
    failed_models = [r for r in results if r["status"] == "failed"]
    assert len(failed_models) == 1
    rf_res = failed_models[0]
    assert rf_res["model_name"] == "Random Forest Classifier"
    assert "Simulated internal GPU/memory training error" in rf_res["error_message"]
    assert rf_res["score"] is None
    assert rf_res["metrics"] is None

    # Other models succeeded
    success_models = [r for r in results if r["status"] == "success"]
    assert len(success_models) == 2
    success_names = {r["model_name"] for r in success_models}
    assert success_names == {"Logistic Regression", "SVC"}
    for m in success_models:
        assert m["score"] is not None
        assert m["error_message"] is None

    # Logging captured error for RandomForest
    rf_logs = [l for l in logged_runs if l["inputs"].get("model_name") == "Random Forest Classifier"]
    assert len(rf_logs) == 1
    assert rf_logs[0]["outputs"]["status"] == "failed"
    assert "Simulated internal GPU/memory training error" in rf_logs[0]["error"]


# ===========================================================================
# 4. Stratification test
# ===========================================================================

def test_stratification_called_for_classification_and_not_regression(
    titanic_fixture, student_fixture, tmp_path, monkeypatch
):
    """Assert train_test_split is called with stratify=y for classification and stratify=None for regression."""
    split_calls: list[dict[str, Any]] = []

    import sklearn.model_selection
    orig_tts = sklearn.model_selection.train_test_split

    def spy_train_test_split(*args, **kwargs):
        split_calls.append(kwargs)
        return orig_tts(*args, **kwargs)

    import agents.training_agent
    monkeypatch.setattr(agents.training_agent, "train_test_split", spy_train_test_split)

    # 1. Classification (Titanic)
    art_cls = str(tmp_path / "artifacts_cls")
    _setup_pipeline_artifacts(
        titanic_fixture,
        target_column="Survived",
        artifacts_dir=art_cls,
        problem_type="classification",
    )
    agent = TrainingAgent()
    agent.run({"artifacts_dir": art_cls})

    assert len(split_calls) == 1
    assert split_calls[0].get("stratify") is not None
    assert isinstance(split_calls[0]["stratify"], (pd.Series, np.ndarray))

    # 2. Regression (Student Performance)
    art_reg = str(tmp_path / "artifacts_reg")
    _setup_pipeline_artifacts(
        student_fixture,
        target_column="Exam_Score",
        artifacts_dir=art_reg,
        problem_type="regression",
        recommended_metric="RMSE",
    )
    agent.run({"artifacts_dir": art_reg})

    assert len(split_calls) == 2
    assert split_calls[1].get("stratify") is None


# ===========================================================================
# 5. Comparison table sorting test
# ===========================================================================

def test_comparison_table_sorting(titanic_fixture, tmp_path, monkeypatch):
    """Assert results are sorted by recommended_metric score descending, with failed
    and skipped_unrecognized placed after all successful ones."""
    artifacts_dir = str(tmp_path / "artifacts")
    candidates = [
        {"model_name": "Unrecognized One", "reasoning": "Skip."},
        {"model_name": "Logistic Regression", "reasoning": "Valid."},
        {"model_name": "Random Forest Classifier", "reasoning": "Valid."},
        {"model_name": "Unrecognized Two", "reasoning": "Skip."},
        {"model_name": "SVC", "reasoning": "Valid."},
    ]
    _setup_pipeline_artifacts(
        titanic_fixture,
        target_column="Survived",
        artifacts_dir=artifacts_dir,
        problem_type="classification",
        candidate_models=candidates,
    )

    # Patch SVC.fit to fail
    def svc_fail(self, *args, **kwargs):
        raise ValueError("SVC convergence failed")

    monkeypatch.setattr(SVC, "fit", svc_fail)

    agent = TrainingAgent()
    state = agent.run({"artifacts_dir": artifacts_dir})
    table = state["training_results"]

    assert len(table) == 5

    # Successful models come first
    assert table[0]["status"] == "success"
    assert table[1]["status"] == "success"
    assert table[0]["score"] >= table[1]["score"]

    # Non-successful models are placed at the bottom
    bottom_statuses = [m["status"] for m in table[2:]]
    assert all(s in ("failed", "skipped_unrecognized") for s in bottom_statuses)
    for m in table[2:]:
        assert m["score"] is None


# ===========================================================================
# 6. Integration test: Full Week 1-5 Sequence on Titanic
# ===========================================================================

def test_full_week1_to_5_sequence_on_titanic(titanic_fixture, tmp_path):
    """Run full sequential pipeline on Titanic fixture and verify training_results_v1.json is produced."""
    artifacts_dir = str(tmp_path / "artifacts")
    state = {
        "df": titanic_fixture,
        "selected_target": "Survived",
        "artifacts_dir": artifacts_dir,
    }

    # Week 1
    state = DatasetUnderstandingAgent().run(state)
    assert os.path.exists(os.path.join(artifacts_dir, "dataset_profile_v1.json"))

    # Week 2
    state = CleaningAgent().run(state)
    assert os.path.exists(os.path.join(artifacts_dir, "cleaned_v1.csv"))

    # Week 3
    state = EDAAgent().run(state)
    assert os.path.exists(os.path.join(artifacts_dir, "eda_bundle_v1.json"))

    # Week 4
    state = FeatureEngineeringAgent().run(state)
    assert os.path.exists(os.path.join(artifacts_dir, "feature_engineered_v1.csv"))
    assert os.path.exists(os.path.join(artifacts_dir, "feature_engineered_v1.json"))

    # Week 5 Part 1 (Mock plan for deterministic testing)
    plan = {
        "problem_type": "classification",
        "confirmation_reasoning": "Binary Survived target.",
        "recommended_metric": "F1",
        "metric_reasoning": "Balanced evaluation.",
        "candidate_models": [
            {"model_name": "Logistic Regression", "reasoning": "Fast linear model."},
            {"model_name": "Random Forest Classifier", "reasoning": "Nonlinear ensemble."},
            {"model_name": "XGBoost Classifier", "reasoning": "Gradient boosted trees."},
            {"model_name": "LightGBM Classifier", "reasoning": "Lightweight gradient boosting."},
            {"model_name": "SVC", "reasoning": "Maximum margin classifier."},
        ],
    }
    save_artifact(plan, artifacts_dir, "ml_plan", "json", version=1)

    # Week 5 Part 2 (TrainingAgent)
    training_state = TrainingAgent().run({"artifacts_dir": artifacts_dir})

    results_file = os.path.join(artifacts_dir, "training_results_v1.json")
    assert os.path.exists(results_file)

    with open(results_file, "r") as f:
        results = json.load(f)

    assert len(results) == 5
    model_names_in_results = {r["model_name"] for r in results}
    assert "Logistic Regression" in model_names_in_results
    assert "Random Forest Classifier" in model_names_in_results

    # All 5 models (including Logistic Regression and SVC) must succeed on clean Titanic data
    assert all(r["status"] == "success" for r in results)
    for r in results:
        assert r["score"] is not None
        assert r["error_message"] is None


# ===========================================================================
# 7. State and Artifact Integrity Test
# ===========================================================================

def test_training_agent_preserves_existing_artifacts(titanic_fixture, tmp_path):
    """Confirm TrainingAgent does not mutate or overwrite any existing artifacts."""
    artifacts_dir = str(tmp_path / "artifacts")
    _setup_pipeline_artifacts(
        titanic_fixture,
        target_column="Survived",
        artifacts_dir=artifacts_dir,
    )

    # Capture initial artifact contents
    initial_files = {}
    for filename in os.listdir(artifacts_dir):
        file_path = os.path.join(artifacts_dir, filename)
        with open(file_path, "rb") as f:
            initial_files[filename] = f.read()

    # Run TrainingAgent
    agent = TrainingAgent()
    agent.run({"artifacts_dir": artifacts_dir})

    # Verify all previous artifacts remain byte-for-byte identical
    for filename, initial_bytes in initial_files.items():
        file_path = os.path.join(artifacts_dir, filename)
        assert os.path.exists(file_path), f"Artifact {filename} disappeared!"
        with open(file_path, "rb") as f:
            current_bytes = f.read()
        assert current_bytes == initial_bytes, f"Artifact {filename} was mutated!"

    # Verify new training_results artifact was created
    results_path = get_latest_version_path(artifacts_dir, "training_results", "json")
    assert os.path.exists(results_path)
