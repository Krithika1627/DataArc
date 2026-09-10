from __future__ import annotations

import json
import unittest.mock
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents.ml_planning_agent import (
    MLPlanningAgent,
    extract_ml_planning_summary,
    parse_and_validate_ml_plan,
)
from main import app

client = TestClient(app)

VALID_CLASSIFICATION_PLAN_JSON = json.dumps({
    "problem_type": "classification",
    "confirmation_reasoning": "The target variable is binary (0/1 survival status).",
    "recommended_metric": "F1-Score",
    "metric_reasoning": "F1-Score provides a balanced measure for moderate class imbalance.",
    "candidate_models": [
        {"model_name": "Logistic Regression", "reasoning": "Fast, interpretable linear baseline."},
        {"model_name": "Random Forest Classifier", "reasoning": "Handles non-linear relationships and tabular interactions well."},
        {"model_name": "XGBoost Classifier", "reasoning": "State-of-the-art gradient boosting for tabular datasets."},
        {"model_name": "LightGBM Classifier", "reasoning": "Fast gradient boosted tree model suited for tabular data."},
    ],
})

VALID_REGRESSION_PLAN_JSON = json.dumps({
    "problem_type": "regression",
    "confirmation_reasoning": "The target variable is a continuous real-valued quantity.",
    "recommended_metric": "RMSE",
    "metric_reasoning": "RMSE penalizes larger prediction errors and maintains original units.",
    "candidate_models": [
        {"model_name": "Linear Regression", "reasoning": "Simple interpretable baseline."},
        {"model_name": "Ridge Regression", "reasoning": "Linear model with L2 regularization against multicollinearity."},
        {"model_name": "Random Forest Regressor", "reasoning": "Captures non-linear dynamics without overfitting."},
        {"model_name": "XGBoost Regressor", "reasoning": "High performance on complex non-linear feature interactions."},
    ],
})


@pytest.fixture
def mock_gemini_client():
    """Helper to mock google.genai.Client and return a custom response."""
    def _create_mock(response_text: str):
        mock_response = unittest.mock.MagicMock()
        mock_response.text = response_text
        mock_client = unittest.mock.MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        return mock_client
    return _create_mock


@pytest.fixture
def sample_state() -> dict:
    """Realistic pipeline state from Weeks 1-4."""
    n = 100
    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Age": [20 + (i % 40) for i in range(n)],
        "Fare": [10.5 + (i % 50) for i in range(n)],
        "Sex": ["male", "female"] * (n // 2),
        "Survived": [0, 1] * (n // 2),
    })
    return {
        "df": df,
        "cleaned_df": df.copy(),
        "selected_target": "Survived",
        "profile": {
            "row_count": n,
            "column_count": 5,
            "columns": [
                {"name": "PassengerId", "dtype": "int64", "missing_count": 0, "missing_percentage": 0.0, "unique_count": n},
                {"name": "Age", "dtype": "int64", "missing_count": 0, "missing_percentage": 0.0, "unique_count": 40},
                {"name": "Fare", "dtype": "float64", "missing_count": 0, "missing_percentage": 0.0, "unique_count": 50},
                {"name": "Sex", "dtype": "object", "missing_count": 0, "missing_percentage": 0.0, "unique_count": 2},
                {"name": "Survived", "dtype": "int64", "missing_count": 0, "missing_percentage": 0.0, "unique_count": 2},
            ],
            "duplicate_row_count": 0,
            "duplicate_row_percentage": 0.0,
        },
        "target_candidates": [
            {"column_name": "Survived", "confidence_score": 90, "reasons": ["Binary target"]}
        ],
        "problem_type_analysis": {
            "problem_type": "classification",
            "confidence_reasoning": "Survived is binary 0/1.",
            "project_plan": "Train binary classification models.",
        },
        "cleaning_summary": {
            "duplicate_removal": {"duplicates_removed": 0},
            "dtype_fixing": {"columns_fixed": []},
            "missing_value_imputation": {"columns_imputed": []},
            "outlier_handling": {"columns_processed": []},
        },
        "eda_result": {
            "stats": {
                "class_balance": {
                    "class_counts": {"0": 50, "1": 50},
                    "class_percentages": {"0": 50.0, "1": 50.0},
                }
            }
        },
        "feature_engineering_result": {
            "encoding_map": {
                "PassengerId": "excluded",
                "Age": "scaled",
                "Fare": "scaled",
                "Sex": "onehot",
                "Survived": "excluded",
            },
            "excluded_columns": ["PassengerId", "Survived"],
        },
    }


class TestMLPlanningAgent:
    """Test suite for MLPlanningAgent (Week 5 Part 1)."""

    def test_mocked_llm_call_populates_ml_plan(self, sample_state, monkeypatch, mock_gemini_client):
        """1. Unit test with a MOCKED LLM call returning a fixed valid JSON string.
        Assert the agent correctly parses it into state['ml_plan'] with all expected keys.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_client = mock_gemini_client(VALID_CLASSIFICATION_PLAN_JSON)

        with unittest.mock.patch("google.genai.Client", return_value=mock_client):
            agent = MLPlanningAgent()
            result_state = agent.run(sample_state)

        assert "ml_plan" in result_state
        plan = result_state["ml_plan"]

        expected_keys = {
            "problem_type",
            "confirmation_reasoning",
            "recommended_metric",
            "metric_reasoning",
            "candidate_models",
        }
        assert expected_keys.issubset(plan.keys())
        assert plan["problem_type"] == "classification"
        assert plan["recommended_metric"] == "F1-Score"
        assert len(plan["candidate_models"]) == 4
        assert plan["candidate_models"][0]["model_name"] == "Logistic Regression"

    def test_schema_validation(self):
        """2. Schema validation test - assert candidate_models is a non-empty list,
        recommended_metric is a non-empty string, problem_type is one of the two allowed values.
        """
        # Valid plan passes
        parsed = parse_and_validate_ml_plan(VALID_CLASSIFICATION_PLAN_JSON)
        assert parsed["problem_type"] in ("classification", "regression")
        assert isinstance(parsed["recommended_metric"], str) and len(parsed["recommended_metric"]) > 0
        assert isinstance(parsed["candidate_models"], list) and len(parsed["candidate_models"]) > 0
        for m in parsed["candidate_models"]:
            assert "model_name" in m and m["model_name"]
            assert "reasoning" in m and m["reasoning"]

        # Invalid problem_type
        bad_type = json.dumps({
            "problem_type": "clustering",
            "confirmation_reasoning": "reason",
            "recommended_metric": "Silhouette",
            "metric_reasoning": "reason",
            "candidate_models": [{"model_name": "K-Means", "reasoning": "fit"}],
        })
        with pytest.raises(ValueError, match="Invalid problem_type"):
            parse_and_validate_ml_plan(bad_type)

        # Empty candidate_models
        empty_models = json.dumps({
            "problem_type": "classification",
            "confirmation_reasoning": "reason",
            "recommended_metric": "Accuracy",
            "metric_reasoning": "reason",
            "candidate_models": [],
        })
        with pytest.raises(ValueError, match="candidate_models must be a non-empty list"):
            parse_and_validate_ml_plan(empty_models)

        # Missing required key
        missing_key = json.dumps({
            "problem_type": "classification",
            "recommended_metric": "Accuracy",
            "candidate_models": [{"model_name": "LR", "reasoning": "fit"}],
        })
        with pytest.raises(ValueError, match="ML plan missing required keys"):
            parse_and_validate_ml_plan(missing_key)

    def test_malformed_json_handling_and_logging(self, sample_state, monkeypatch, mock_gemini_client):
        """3. Malformed-JSON handling test - mock the LLM to return invalid/non-JSON text,
        assert the agent raises a clear, specific error and that this gets logged via log_agent_run().
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_client = mock_gemini_client("This is completely invalid and not JSON at all.")

        with unittest.mock.patch("google.genai.Client", return_value=mock_client):
            with unittest.mock.patch("agents.ml_planning_agent.log_agent_run") as mock_log:
                agent = MLPlanningAgent()
                with pytest.raises(ValueError, match="Malformed JSON"):
                    agent.run(sample_state)

                mock_log.assert_called_once()
                call_kwargs = mock_log.call_args.kwargs
                assert call_kwargs["agent_name"] == "ml_planning"
                assert "error" in call_kwargs
                assert "Malformed JSON" in call_kwargs["error"]

    def test_cross_check_titanic_matches_week1_guess(self, sample_state, monkeypatch, mock_gemini_client):
        """4. Cross-check test using Titanic test fixture - assert the LLM's
        confirmed problem_type matches Week 1's stored guess for that dataset (classification).
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_client = mock_gemini_client(VALID_CLASSIFICATION_PLAN_JSON)

        with unittest.mock.patch("google.genai.Client", return_value=mock_client):
            agent = MLPlanningAgent()
            result_state = agent.run(sample_state)

        week1_guess = sample_state["problem_type_analysis"]["problem_type"]
        confirmed_type = result_state["ml_plan"]["problem_type"]
        assert confirmed_type == week1_guess == "classification"

    def test_state_integrity(self, sample_state, monkeypatch, mock_gemini_client):
        """5. State integrity test - assert all pre-existing keys in state (from Weeks 1-4)
        remain unchanged after this agent runs, and only ml_plan is added.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_client = mock_gemini_client(VALID_CLASSIFICATION_PLAN_JSON)

        original_keys = set(sample_state.keys())
        original_values = {k: sample_state[k] for k in original_keys}

        with unittest.mock.patch("google.genai.Client", return_value=mock_client):
            agent = MLPlanningAgent()
            result_state = agent.run(sample_state)

        # Assert only ml_plan was added
        new_keys = set(result_state.keys())
        assert new_keys - original_keys == {"ml_plan"}

        # Assert all previous keys and their exact references/values are unchanged
        for k in original_keys:
            if isinstance(original_values[k], pd.DataFrame):
                pd.testing.assert_frame_equal(result_state[k], original_values[k])
            else:
                assert result_state[k] == original_values[k]

    def test_strips_markdown_code_fences(self):
        """Verify that markdown code fences (```json ... ```) are stripped defensively."""
        fenced_json = f"```json\n{VALID_CLASSIFICATION_PLAN_JSON}\n```"
        parsed = parse_and_validate_ml_plan(fenced_json)
        assert parsed["problem_type"] == "classification"
        assert len(parsed["candidate_models"]) == 4

    def test_extract_summary_does_not_recompute(self, sample_state):
        """Verify summary extraction reuses existing state fields without recomputing."""
        summary = extract_ml_planning_summary(sample_state)
        assert summary["target_column"] == "Survived"
        assert summary["earlier_problem_type_guess"] == "classification"
        assert summary["row_count"] == 100
        assert summary["column_count"] == 5
        assert summary["class_balance"] == {"class_counts": {"0": 50, "1": 50}, "class_percentages": {"0": 50.0, "1": 50.0}}
        assert summary["encoding_map"]["Sex"] == "onehot"

    def test_fastapi_plan_training_endpoint(self, sample_state, monkeypatch, mock_gemini_client):
        """8. FastAPI endpoint test for /plan-training and /plan-ml."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
        mock_client = mock_gemini_client(VALID_CLASSIFICATION_PLAN_JSON)

        # Prepare JSON-serializable state
        serializable_state = {
            "selected_target": sample_state["selected_target"],
            "profile": sample_state["profile"],
            "problem_type_analysis": sample_state["problem_type_analysis"],
            "eda_result": sample_state["eda_result"],
            "feature_engineering_result": sample_state["feature_engineering_result"],
        }

        with unittest.mock.patch("google.genai.Client", return_value=mock_client):
            response = client.post("/plan-training", json=serializable_state)
            assert response.status_code == 200
            data = response.json()
            assert data["problem_type"] == "classification"
            assert data["recommended_metric"] == "F1-Score"
            assert len(data["candidate_models"]) == 4

            # Test alias endpoint /plan-ml
            response_alias = client.post("/plan-ml", json={"state": serializable_state})
            assert response_alias.status_code == 200
            assert response_alias.json()["problem_type"] == "classification"
