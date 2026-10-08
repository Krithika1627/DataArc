"""
Dashboard Integration Smoke Test
Tests the full end-to-end endpoint sequence called by the Streamlit dashboard
for both Titanic and Student Performance datasets in the same session,
verifying dataset isolation, artifact persistence, and PDF report delivery.
"""

from __future__ import annotations

import io
import json
import unittest.mock
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from main import app
from agents.versioning_utils import get_db_engine, get_latest_version_path

client = TestClient(app)


def _make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def titanic_df() -> pd.DataFrame:
    n = 60
    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Age": [20 + (i % 30) for i in range(n)],
        "Fare": [10.0 + (i % 50) for i in range(n)],
        "Sex": ["male", "female"] * (n // 2),
        "Survived": [0, 1] * (n // 2),
    })


@pytest.fixture
def student_df() -> pd.DataFrame:
    n = 60
    return pd.DataFrame({
        "StudentID": range(1, n + 1),
        "StudyHours": [1.0 + (i % 10) for i in range(n)],
        "Attendance": [70 + (i % 30) for i in range(n)],
        "ExamScore": [50.0 + float(i % 50) for i in range(n)],
    })


class TestDashboardIntegrationFlow:
    """Simulates the exact API calling pattern executed by the Streamlit dashboard."""

    @pytest.fixture(autouse=True)
    def clean_db(self):
        eng = get_db_engine()
        with eng.connect() as conn:
            with conn.begin():
                conn.execute(text("DELETE FROM artifacts"))
        yield

    def test_dashboard_full_pipeline_sequence_multi_dataset_isolation(
        self, titanic_df: pd.DataFrame, student_df: pd.DataFrame
    ):
        """Run full dashboard sequence for Titanic, then Student, confirming isolation."""

        # =====================================================================
        # RUN 1: Titanic Dataset (Classification)
        # =====================================================================
        titanic_bytes = _make_csv_bytes(titanic_df)

        # 1. Step 1: Upload & Analyze Dataset
        resp_t1 = client.post(
            "/analyze-dataset",
            files={"file": ("titanic.csv", titanic_bytes, "text/csv")},
            data={"user_selected_target": "Survived"},
        )
        assert resp_t1.status_code == 200
        t1_data = resp_t1.json()
        ds_titanic_id = t1_data["dataset_id"]
        assert ds_titanic_id is not None
        assert t1_data["selected_target"] == "Survived"
        assert t1_data["problem_type_analysis"]["problem_type"] == "classification"

        # 2. Step 2: Data Cleaning
        resp_t2 = client.post(
            "/clean-dataset",
            files={"file": ("titanic.csv", titanic_bytes, "text/csv")},
            data={"dataset_id": ds_titanic_id, "target_column": "Survived", "cap_target": "false"},
        )
        assert resp_t2.status_code == 200
        t2_data = resp_t2.json()
        assert t2_data["dataset_id"] == ds_titanic_id
        t_changelog_path = t2_data["changelog_path"]

        # 3. Step 3: EDA
        resp_t3 = client.post(
            "/run-eda",
            files={"file": ("titanic.csv", titanic_bytes, "text/csv")},
            data={
                "dataset_id": ds_titanic_id,
                "target_column": "Survived",
                "problem_type": "classification",
                "cleaning_changelog_path": t_changelog_path,
            },
        )
        assert resp_t3.status_code == 200
        t3_data = resp_t3.json()
        assert t3_data["dataset_id"] == ds_titanic_id
        assert t3_data["correlation_heatmap"] is not None

        # 4. Step 4: Feature Engineering
        resp_t4 = client.post(
            "/run-feature-engineering",
            files={"file": ("titanic.csv", titanic_bytes, "text/csv")},
            data={
                "dataset_id": ds_titanic_id,
                "target_column": "Survived",
                "problem_type": "classification",
                "cleaning_changelog_path": t_changelog_path,
            },
        )
        assert resp_t4.status_code == 200
        t4_data = resp_t4.json()
        assert t4_data["dataset_id"] == ds_titanic_id

        # 5. Step 5: ML Planning (Mock LLM plan response)
        mock_titanic_plan = {
            "problem_type": "classification",
            "confirmation_reasoning": "Binary classification for Survived target.",
            "recommended_metric": "F1",
            "metric_reasoning": "Optimizes balance between precision and recall.",
            "candidate_models": [
                {"model_name": "Random Forest Classifier", "reasoning": "Ensemble tree model"},
                {"model_name": "Logistic Regression", "reasoning": "Linear baseline"},
            ],
        }
        with unittest.mock.patch("agents.ml_planning_agent.generate_ml_plan", return_value=mock_titanic_plan):
            resp_t5 = client.post(
                "/plan-ml",
                json={"dataset_id": ds_titanic_id, "selected_target": "Survived", "problem_type": "classification"},
            )
            assert resp_t5.status_code == 200
            t5_data = resp_t5.json()
            assert t5_data["recommended_metric"] == "F1"

        # 6. Step 6: Model Training
        resp_t6 = client.post("/train-model", json={"dataset_id": ds_titanic_id})
        assert resp_t6.status_code == 200
        t6_data = resp_t6.json()
        assert t6_data["summary"]["succeeded"] >= 1
        assert len(t6_data["comparison_table"]) >= 1



        # 7. Step 7: Model Evaluation
        resp_t7 = client.post("/evaluate-model", json={"dataset_id": ds_titanic_id})
        assert resp_t7.status_code == 200
        t7_data = resp_t7.json()
        assert t7_data["winning_model_name"] in ["Random Forest Classifier", "Logistic Regression"]
        assert t7_data["confusion_matrix"] is not None
        assert t7_data["roc_curve"] is not None

        # 8. Step 8: PDF Report Generation
        resp_t8 = client.post("/generate-report", json={"dataset_id": ds_titanic_id})
        assert resp_t8.status_code == 200
        assert resp_t8.headers.get("content-type") == "application/pdf"
        assert resp_t8.content.startswith(b"%PDF")

        # =====================================================================
        # RUN 2: Student Performance Dataset (Regression)
        # =====================================================================
        student_bytes = _make_csv_bytes(student_df)

        # 1. Step 1: Upload & Analyze Dataset
        resp_s1 = client.post(
            "/analyze-dataset",
            files={"file": ("student.csv", student_bytes, "text/csv")},
            data={"user_selected_target": "ExamScore"},
        )
        assert resp_s1.status_code == 200
        s1_data = resp_s1.json()
        ds_student_id = s1_data["dataset_id"]
        assert ds_student_id is not None
        assert ds_student_id != ds_titanic_id  # Isolated IDs

        # 2. Step 2: Data Cleaning
        resp_s2 = client.post(
            "/clean-dataset",
            files={"file": ("student.csv", student_bytes, "text/csv")},
            data={"dataset_id": ds_student_id, "target_column": "ExamScore", "cap_target": "false"},
        )
        assert resp_s2.status_code == 200
        s_changelog_path = resp_s2.json()["changelog_path"]

        # 3. Step 3: EDA
        resp_s3 = client.post(
            "/run-eda",
            files={"file": ("student.csv", student_bytes, "text/csv")},
            data={
                "dataset_id": ds_student_id,
                "target_column": "ExamScore",
                "problem_type": "regression",
                "cleaning_changelog_path": s_changelog_path,
            },
        )
        assert resp_s3.status_code == 200

        # 4. Step 4: Feature Engineering
        resp_s4 = client.post(
            "/run-feature-engineering",
            files={"file": ("student.csv", student_bytes, "text/csv")},
            data={
                "dataset_id": ds_student_id,
                "target_column": "ExamScore",
                "problem_type": "regression",
                "cleaning_changelog_path": s_changelog_path,
            },
        )
        assert resp_s4.status_code == 200

        # 5. Step 5: ML Planning
        mock_student_plan = {
            "problem_type": "regression",
            "confirmation_reasoning": "Continuous ExamScore regression target.",
            "recommended_metric": "RMSE",
            "metric_reasoning": "Measures root mean squared error on continuous score.",
            "candidate_models": [
                {"model_name": "Random Forest Regressor", "reasoning": "Tree ensemble regressor"},
                {"model_name": "Linear Regression", "reasoning": "Linear regressor"},
            ],
        }
        with unittest.mock.patch("agents.ml_planning_agent.generate_ml_plan", return_value=mock_student_plan):
            resp_s5 = client.post(
                "/plan-ml",
                json={"dataset_id": ds_student_id, "selected_target": "ExamScore", "problem_type": "regression"},
            )
            assert resp_s5.status_code == 200

        # 6. Step 6: Model Training
        resp_s6 = client.post("/train-model", json={"dataset_id": ds_student_id})
        assert resp_s6.status_code == 200

        # 7. Step 7: Model Evaluation
        resp_s7 = client.post("/evaluate-model", json={"dataset_id": ds_student_id})
        assert resp_s7.status_code == 200
        s7_data = resp_s7.json()
        assert s7_data["problem_type"] == "regression"
        # Regression skips confusion matrix & ROC
        assert s7_data["confusion_matrix"] is None
        assert s7_data["roc_curve"] is None

        # 8. Step 8: PDF Report
        resp_s8 = client.post("/generate-report", json={"dataset_id": ds_student_id})
        assert resp_s8.status_code == 200
        assert resp_s8.content.startswith(b"%PDF")

        # =====================================================================
        # VERIFY ZERO CROSS-CONTAMINATION IN DATABASE
        # =====================================================================
        titanic_eval = get_latest_version_path(ds_titanic_id, "evaluation_bundle", "json")
        student_eval = get_latest_version_path(ds_student_id, "evaluation_bundle", "json")

        assert titanic_eval["problem_type"] == "classification"
        assert student_eval["problem_type"] == "regression"
        assert titanic_eval["confusion_matrix"] is not None
        assert student_eval["confusion_matrix"] is None
