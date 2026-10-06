from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import pytest
from fastapi.testclient import TestClient

from agents.evaluation_agent import EvaluationAgent
from agents.pipeline_agents import (
    CleaningAgent,
    DatasetUnderstandingAgent,
    EDAAgent,
    FeatureEngineeringAgent,
    MLPlanningAgent,
    TrainingAgent,
)
from agents.report_agent import ReportAgent, build_report_html, load_all_pipeline_artifacts
from agents.versioning_utils import (
    get_db_engine,
    _clean_dataset_id,
    get_latest_version_path,
    save_artifact,
)
from sqlalchemy import text
from main import app


@pytest.fixture
def titanic_report_fixture() -> pd.DataFrame:
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
def student_report_fixture() -> pd.DataFrame:
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


def _setup_full_pipeline_artifacts(
    df: pd.DataFrame,
    target_column: str,
    artifacts_dir: str,
    problem_type: str = "classification",
    recommended_metric: str = "F1",
    include_failure: bool = False,
) -> dict:
    """Helper to run Weeks 1-6 Part 2 agents in sequence, producing all required stage artifacts."""
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

    if problem_type == "classification":
        candidates = [
            {"model_name": "Random Forest Classifier", "reasoning": "Strong ensemble baseline."},
            {"model_name": "Logistic Regression", "reasoning": "Fast linear model."},
        ]
        if include_failure:
            candidates.append({"model_name": "Unrecognized Custom Model", "reasoning": "Test failure fallback."})
    else:
        candidates = [
            {"model_name": "Random Forest Regressor", "reasoning": "Tree regressor for nonlinear patterns."},
            {"model_name": "Linear Regression", "reasoning": "Standard linear baseline."},
        ]

    mock_plan = {
        "problem_type": problem_type,
        "target_column": target_column,
        "confirmation_reasoning": f"Optimizing {problem_type} on target {target_column}.",
        "recommended_metric": recommended_metric,
        "metric_reasoning": f"Balanced evaluation using {recommended_metric}.",
        "candidate_models": candidates,
    }
    with patch("agents.ml_planning_agent.generate_ml_plan", return_value=mock_plan):
        MLPlanningAgent().run(state)

    TrainingAgent().run(state)
    EvaluationAgent().run(state)

    return state


class TestReportAgent:
    """Test suite for Week 6 Part 3: Report Agent."""

    def test_missing_artifact_raises_clear_error(self, titanic_report_fixture, tmp_path):
        """1. Missing-artifact test: assert specific FileNotFoundError naming the missing stage."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_full_pipeline_artifacts(
            titanic_report_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
        )

        # Delete evaluation_bundle artifact to simulate missing evaluation stage
        clean_id = _clean_dataset_id(artifacts_dir)
        with get_db_engine().connect() as conn:
            with conn.begin():
                conn.execute(
                    text("DELETE FROM artifacts WHERE dataset_id = :d AND artifact_type = 'evaluation_bundle'"),
                    {"d": clean_id},
                )

        agent = ReportAgent()
        with pytest.raises(FileNotFoundError, match="Model Evaluation"):
            agent.run({"artifacts_dir": artifacts_dir})

    def test_content_completeness_titanic(self, titanic_report_fixture, tmp_path):
        """2. Content-completeness test: asserts target, confidence score, winner, and failed model appear."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_full_pipeline_artifacts(
            titanic_report_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
            problem_type="classification",
            recommended_metric="F1",
            include_failure=True,
        )

        agent = ReportAgent()
        state = agent.run({"artifacts_dir": artifacts_dir})

        html_content = state["report_html"]
        pdf_path = state["report_pdf_path"]
        assert os.path.exists(pdf_path)

        # 1. Target column name
        assert "Survived" in html_content

        # 2. Confidence score and breakdown table
        assert "/ 100" in html_content
        assert "Target Selection Confidence Breakdown" in html_content

        # 3. Winning model name
        assert "Random Forest Classifier" in html_content or "Logistic Regression" in html_content

        # 4. Failed/Skipped model name is NOT dropped and clearly present
        assert "Unrecognized Custom Model" in html_content
        assert "Skipped" in html_content or "Failed" in html_content

        # 5. Extract text from generated PDF via pypdfium2
        pdf = pdfium.PdfDocument(pdf_path)
        full_pdf_text = ""
        for page in pdf:
            textpage = page.get_textpage()
            full_pdf_text += textpage.get_text_range()

        assert "Survived" in full_pdf_text
        assert "DataArc Executive ML Pipeline Report" in full_pdf_text
        assert "Unrecognized Custom Model" in full_pdf_text

    def test_traceability_footer(self, titanic_report_fixture, tmp_path):
        """3. Traceability test: assert footer lists exact artifact version numbers loaded."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_full_pipeline_artifacts(
            titanic_report_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
        )

        agent = ReportAgent()
        state = agent.run({"artifacts_dir": artifacts_dir})
        html_content = state["report_html"]
        versions_used = state["report_versions_used"]

        assert "dataset_profile_v1.json" in html_content
        assert "cleaned_v1_changelog.json" in html_content
        assert "eda_bundle_v1.json" in html_content
        assert "feature_engineered_v1.json" in html_content
        assert "ml_plan_v1.json" in html_content
        assert "training_results_v1.json" in html_content
        assert "evaluation_bundle_v1.json" in html_content

        assert versions_used["dataset_profile"] == "dataset_profile_v1.json"
        assert versions_used["training_results"] == "training_results_v1.json"
        assert versions_used["evaluation_bundle"] == "evaluation_bundle_v1.json"

    def test_pdf_validity_and_magic_bytes(self, titanic_report_fixture, tmp_path):
        """4. PDF validity test: assert %PDF- magic bytes, non-trivial size, and valid structure."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_full_pipeline_artifacts(
            titanic_report_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
        )

        agent = ReportAgent()
        state = agent.run({"artifacts_dir": artifacts_dir})
        pdf_path = state["report_pdf_path"]

        # Check %PDF- magic bytes
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            assert header == b"%PDF-", f"Expected PDF magic bytes %PDF-, got {header!r}"

        # Check file size (> 5KB)
        file_size = os.path.getsize(pdf_path)
        assert file_size > 5000, f"PDF file size is unexpectedly small: {file_size} bytes"

        # Check PDF can be parsed cleanly by pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        assert len(pdf) >= 1

    def test_regression_dataset_report_omits_classification_visualizations(
        self, student_report_fixture, tmp_path
    ):
        """5. Regression-dataset test: assert report omits CM/ROC curve cleanly without broken placeholders."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_full_pipeline_artifacts(
            student_report_fixture,
            target_column="Exam_Score",
            artifacts_dir=artifacts_dir,
            problem_type="regression",
            recommended_metric="RMSE",
        )

        agent = ReportAgent()
        state = agent.run({"artifacts_dir": artifacts_dir})
        html_content = state["report_html"]
        pdf_path = state["report_pdf_path"]

        # Assert no broken confusion matrix / ROC curve image tags
        assert "Confusion Matrix" not in html_content
        assert "Receiver Operating Characteristic" not in html_content

        # Feature importance should still be present for Random Forest Regressor
        assert "Top Feature Importances" in html_content

        # Confirm valid PDF generated
        pdf = pdfium.PdfDocument(pdf_path)
        assert len(pdf) >= 1

    def test_fastapi_generate_report_endpoint(self, titanic_report_fixture, tmp_path):
        """6. FastAPI test for /generate-report endpoint returning FileResponse."""
        artifacts_dir = str(tmp_path / "artifacts")
        _setup_full_pipeline_artifacts(
            titanic_report_fixture,
            target_column="Survived",
            artifacts_dir=artifacts_dir,
        )

        client = TestClient(app)
        response = client.post(
            "/generate-report",
            json={"artifacts_dir": artifacts_dir},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF-")
        assert len(response.content) > 5000
