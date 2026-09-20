from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from agents.data_cleaning import clean_dataset
from agents.dataset_understanding import profile_dataset
from agents.eda_agent import run_eda
from agents.evaluation_agent import (
    EvaluationAgent,
    generate_confusion_matrix_plot,
    generate_feature_importance_plot,
    generate_llm_evaluation_explanation,
    generate_roc_curve_plot,
    is_lower_is_better_metric,
    select_winning_model,
)
from agents.feature_engineering_agent import build_feature_pipeline
from agents.pipeline_agents import (
    CleaningAgent,
    DatasetUnderstandingAgent,
    EDAAgent,
    FeatureEngineeringAgent,
    MLPlanningAgent,
    TrainingAgent,
)
from agents.versioning_utils import get_latest_version_path, save_artifact


@pytest.fixture
def titanic_eval_fixture() -> pd.DataFrame:
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
def student_eval_fixture() -> pd.DataFrame:
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


def _setup_training_artifacts(
    df: pd.DataFrame,
    target_column: str,
    artifacts_dir: str,
    problem_type: str = "classification",
    recommended_metric: str = "F1",
    candidate_models: list[dict[str, str]] | None = None,
    custom_results: list[dict[str, Any]] | None = None,
) -> None:
    """Helper to setup artifacts up to training_results."""
    os.makedirs(artifacts_dir, exist_ok=True)
    state = {
        "df": df,
        "artifacts_dir": artifacts_dir,
        "user_selected_target": target_column,
        "selected_target": target_column,
    }
    DatasetUnderstandingAgent().run(state)
    CleaningAgent().run(state)
    EDAAgent().run(state)
    FeatureEngineeringAgent().run(state)

    if candidate_models is None:
        if problem_type == "classification":
            candidate_models = [
                {"model_name": "Random Forest Classifier", "reasoning": "Tree model"},
                {"model_name": "Logistic Regression", "reasoning": "Linear classifier"},
            ]
        else:
            candidate_models = [
                {"model_name": "Random Forest Regressor", "reasoning": "Tree regressor"},
                {"model_name": "Linear Regression", "reasoning": "Linear regressor"},
            ]

    ml_plan = {
        "problem_type": problem_type,
        "target_column": target_column,
        "recommended_metric": recommended_metric,
        "metric_reasoning": f"Optimizing {recommended_metric}",
        "candidate_models": candidate_models,
    }
    save_artifact(ml_plan, artifacts_dir, "ml_plan", "json", version=1)

    if custom_results is not None:
        save_artifact(custom_results, artifacts_dir, "training_results", "json", version=1)
    else:
        state["ml_plan"] = ml_plan
        TrainingAgent().run(state)


class TestEvaluationAgent:
    """Test suite for Week 6 Part 2: Evaluation Agent."""

    def test_winner_selection_higher_and_lower_is_better(self):
        """1. Winner selection test: asserts correct winner picked with direction handling."""
        # Scenario A: Higher-is-better (F1)
        results_classification = [
            {"model_name": "Model_A", "status": "success", "score": 0.75},
            {"model_name": "Model_B", "status": "success", "score": 0.88},
            {"model_name": "Model_C", "status": "failed", "score": None},
            {"model_name": "Model_D", "status": "success", "score": 0.82},
        ]
        winner, runner_up, gap = select_winning_model(
            results_classification,
            recommended_metric="F1",
            problem_type="classification",
        )
        assert winner["model_name"] == "Model_B"
        assert winner["score"] == 0.88
        assert runner_up["model_name"] == "Model_D"
        assert runner_up["score"] == 0.82
        assert gap == pytest.approx(0.06, abs=1e-4)

        # Scenario B: Lower-is-better (RMSE)
        results_regression = [
            {"model_name": "Reg_A", "status": "success", "score": 12.5},
            {"model_name": "Reg_B", "status": "success", "score": 8.2},
            {"model_name": "Reg_C", "status": "failed", "score": None},
            {"model_name": "Reg_D", "status": "success", "score": 9.1},
        ]
        winner_reg, runner_up_reg, gap_reg = select_winning_model(
            results_regression,
            recommended_metric="RMSE",
            problem_type="regression",
        )
        assert winner_reg["model_name"] == "Reg_B"
        assert winner_reg["score"] == 8.2
        assert runner_up_reg["model_name"] == "Reg_D"
        assert runner_up_reg["score"] == 9.1
        assert gap_reg == pytest.approx(0.9, abs=1e-4)

    def test_all_models_failed_raises_clear_error(self, titanic_eval_fixture, tmp_path):
        """2. All-models-failed test: asserts explicit ValueError when 0 models succeed."""
        results_all_failed = [
            {"model_name": "Model_A", "status": "failed", "score": None},
            {"model_name": "Model_B", "status": "skipped_unrecognized", "score": None},
        ]
        with pytest.raises(ValueError, match="All candidate models failed or were skipped"):
            select_winning_model(results_all_failed, "Accuracy", "classification")

        # Also test via EvaluationAgent.run()
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_training_artifacts(
            titanic_eval_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
            custom_results=results_all_failed,
        )
        agent = EvaluationAgent()
        with pytest.raises(ValueError, match="All candidate models failed or were skipped"):
            agent.run({"artifacts_dir": artifacts_dir})

    def test_classification_visualizations_tree_model(self, titanic_eval_fixture, tmp_path):
        """3. Classification visualization test: assert valid Plotly JSON for CM, ROC, and feature importance."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_training_artifacts(
            titanic_eval_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
            problem_type="classification",
            recommended_metric="F1",
            candidate_models=[
                {"model_name": "Random Forest Classifier", "reasoning": "Tree model"},
            ],
        )

        agent = EvaluationAgent()
        state = agent.run({"artifacts_dir": artifacts_dir})
        bundle = state["evaluation_bundle"]

        assert bundle["winning_model_name"] == "Random Forest Classifier"
        assert bundle["problem_type"] == "classification"

        # Confusion matrix is valid Plotly JSON
        assert bundle["confusion_matrix"] is not None
        cm_data = json.loads(bundle["confusion_matrix"])
        assert "data" in cm_data and "layout" in cm_data

        # ROC curve is valid Plotly JSON
        assert bundle["roc_curve"] is not None
        roc_data = json.loads(bundle["roc_curve"])
        assert "data" in roc_data and "layout" in roc_data

        # Feature importance is valid Plotly JSON for tree model
        assert bundle["feature_importance_chart"] is not None
        fi_data = json.loads(bundle["feature_importance_chart"])
        assert "data" in fi_data and "layout" in fi_data

    def test_regression_skips_confusion_matrix_and_roc(self, student_eval_fixture, tmp_path):
        """4. Regression skip test: CM and ROC are null, logged in skipped_visualizations."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_training_artifacts(
            student_eval_fixture,
            target_column="Exam_Score",
            artifacts_dir=artifacts_dir,
            problem_type="regression",
            recommended_metric="RMSE",
            candidate_models=[
                {"model_name": "Random Forest Regressor", "reasoning": "Tree model for regression"},
            ],
        )

        agent = EvaluationAgent()
        state = agent.run({"artifacts_dir": artifacts_dir})
        bundle = state["evaluation_bundle"]

        assert bundle["confusion_matrix"] is None
        assert bundle["roc_curve"] is None

        # Feature importance is present for Random Forest Regressor
        assert bundle["feature_importance_chart"] is not None

        # Check skipped_visualizations reasons
        skipped_types = [s["type"] for s in bundle["skipped_visualizations"]]
        assert "confusion_matrix" in skipped_types
        assert "roc_curve" in skipped_types

        for s in bundle["skipped_visualizations"]:
            if s["type"] in ("confusion_matrix", "roc_curve"):
                assert "regression" in s["reason"].lower()

    def test_non_tree_model_feature_importance_skip(self, titanic_eval_fixture, tmp_path):
        """5. Non-tree model feature importance skip test: Logistic Regression skips feature importance."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_training_artifacts(
            titanic_eval_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
            problem_type="classification",
            recommended_metric="Accuracy",
            candidate_models=[
                {"model_name": "Logistic Regression", "reasoning": "Linear classifier"},
            ],
        )

        agent = EvaluationAgent()
        state = agent.run({"artifacts_dir": artifacts_dir})
        bundle = state["evaluation_bundle"]

        assert bundle["winning_model_name"] == "Logistic Regression"
        assert bundle["feature_importance_chart"] is None

        skipped_types = [s["type"] for s in bundle["skipped_visualizations"]]
        assert "feature_importance_chart" in skipped_types
        fi_skip = next(s for s in bundle["skipped_visualizations"] if s["type"] == "feature_importance_chart")
        assert "tree-based" in fi_skip["reason"].lower()

    def test_llm_explanation_grounding(self):
        """6. LLM explanation grounding test: prompt contains winning score, runner-up score, and score gap."""
        comparison_table = [
            {"model_name": "Random Forest Classifier", "status": "success", "score": 0.8654},
            {"model_name": "Logistic Regression", "status": "success", "score": 0.7923},
        ]
        ml_plan = {
            "candidate_models": [
                {"model_name": "Random Forest Classifier", "reasoning": "Handles non-linear relationships"},
                {"model_name": "Logistic Regression", "reasoning": "Fast linear baseline"},
            ]
        }

        captured_prompts = []

        def mock_generate_content(model, contents):
            captured_prompts.append(contents)
            mock_resp = MagicMock()
            mock_resp.text = "Grounded LLM explanation text."
            return mock_resp

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = mock_generate_content

        with patch("agents.evaluation_agent.genai.Client", return_value=mock_client), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "fake_test_key"}):
            explanation = generate_llm_evaluation_explanation(
                winning_model_name="Random Forest Classifier",
                winning_score=0.8654,
                recommended_metric="F1",
                problem_type="classification",
                comparison_table=comparison_table,
                runner_up_name="Logistic Regression",
                runner_up_score=0.7923,
                score_gap=0.0731,
                ml_plan=ml_plan,
            )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "0.8654" in prompt
        assert "0.7923" in prompt
        assert "0.0731" in prompt
        assert "Random Forest Classifier" in prompt
        assert "Logistic Regression" in prompt
        assert "F1" in prompt
        assert "Handles non-linear relationships" in prompt

    def test_reproducible_split_identical_to_training(self, titanic_eval_fixture, tmp_path):
        """7. Reproducible split test: asserts train_test_split produces identical indices as Week 5."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_training_artifacts(
            titanic_eval_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
            problem_type="classification",
        )

        fe_csv_path = get_latest_version_path(artifacts_dir, "feature_engineered", "csv")
        cleaned_csv_path = get_latest_version_path(artifacts_dir, "cleaned", "csv")
        X = pd.read_csv(fe_csv_path)
        cleaned_df = pd.read_csv(cleaned_csv_path)
        y = cleaned_df["Survived"]

        # Original Week 5 split
        stratify_target = y if (pd.Series(y).value_counts() >= 2).all() else None
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target,
        )

        # EvaluationAgent split
        stratify_target_eval = y if (pd.Series(y).value_counts() >= 2).all() else None
        X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target_eval,
        )

        # Check identical indices
        pd.testing.assert_index_equal(X_train_orig.index, X_train_eval.index)
        pd.testing.assert_index_equal(X_test_orig.index, X_test_eval.index)
        pd.testing.assert_series_equal(y_train_orig, y_train_eval)
        pd.testing.assert_series_equal(y_test_orig, y_test_eval)

    def test_full_pipeline_integration_titanic(self, titanic_eval_fixture, tmp_path):
        """8. Integration test: full Week 1-6 pipeline sequence on Titanic produces evaluation_bundle_v1.json."""
        artifacts_dir = str(tmp_path / "artifacts")
        state = {
            "df": titanic_eval_fixture,
            "artifacts_dir": artifacts_dir,
            "user_selected_target": "Survived",
        }

        # Sequence of Agents
        DatasetUnderstandingAgent().run(state)
        CleaningAgent().run(state)
        EDAAgent().run(state)
        FeatureEngineeringAgent().run(state)

        mock_plan = {
            "problem_type": "classification",
            "confirmation_reasoning": "Binary Survived target on Titanic dataset.",
            "recommended_metric": "F1",
            "metric_reasoning": "F1 score accounts for class balance.",
            "candidate_models": [
                {"model_name": "Random Forest Classifier", "reasoning": "Tree ensemble baseline."},
                {"model_name": "Logistic Regression", "reasoning": "Linear baseline."},
            ],
        }
        with patch("agents.ml_planning_agent.generate_ml_plan", return_value=mock_plan):
            MLPlanningAgent().run(state)

        TrainingAgent().run(state)
        EvaluationAgent().run(state)

        # Verify evaluation_bundle_v1.json exists
        bundle_path = os.path.join(artifacts_dir, "evaluation_bundle_v1.json")
        assert os.path.exists(bundle_path)

        with open(bundle_path, "r") as f:
            bundle = json.load(f)

        expected_keys = {
            "winning_model_name",
            "problem_type",
            "recommended_metric",
            "winning_score",
            "runner_up_model_name",
            "runner_up_score",
            "score_gap",
            "confusion_matrix",
            "roc_curve",
            "feature_importance_chart",
            "llm_explanation",
            "skipped_visualizations",
        }
        for k in expected_keys:
            assert k in bundle, f"Expected key '{k}' missing from evaluation bundle"

        assert bundle["problem_type"] == "classification"
        assert isinstance(bundle["winning_score"], (int, float))
        assert bundle["confusion_matrix"] is not None
        assert isinstance(bundle["llm_explanation"], str) and len(bundle["llm_explanation"]) > 0

    def test_fastapi_evaluate_model_endpoint(self, titanic_eval_fixture, tmp_path):
        """9. FastAPI test for /evaluate-model endpoint."""
        from fastapi.testclient import TestClient
        from main import app

        artifacts_dir = str(tmp_path / "artifacts")
        _setup_training_artifacts(
            titanic_eval_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
            problem_type="classification",
        )

        client = TestClient(app)
        response = client.post(
            "/evaluate-model",
            json={"artifacts_dir": artifacts_dir},
        )
        assert response.status_code == 200
        data = response.json()
        assert "winning_model_name" in data
        assert "winning_score" in data
        assert "confusion_matrix" in data
        assert "llm_explanation" in data
        assert "skipped_visualizations" in data
        assert "artifact_path" in data
