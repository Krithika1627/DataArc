"""Comprehensive tests for Dataset-Scoped Artifact Isolation (Week 7 Part 1 & Part 2).

Covers:
1. Dataset isolation across runs in Postgres (two datasets processed in the same session)
2. Missing and non-existent dataset_id validation (HTTP 400 and 404)
3. Report traceability (ReportAgent loads strictly from scoped dataset_id in DB)
4. Independent versioning per dataset_id in DB
"""
from __future__ import annotations

import io
import json
import os
import unittest.mock
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from main import app
from agents.versioning_utils import (
    get_db_engine,
    get_dataset_artifacts_dir,
    get_next_version,
    save_artifact,
    get_latest_artifact,
    get_latest_version_path,
)
from agents.report_agent import ReportAgent, load_all_pipeline_artifacts
from agents.pipeline_agents import (
    DatasetUnderstandingAgent,
    CleaningAgent,
    EDAAgent,
    FeatureEngineeringAgent,
)

client = TestClient(app)


def _make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def titanic_toy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "PassengerId": range(1, 61),
        "Age": [20 + (i % 30) for i in range(60)],
        "Fare": [10.0 + (i % 50) for i in range(60)],
        "Sex": ["male", "female"] * 30,
        "Survived": [0, 1] * 30,
    })


@pytest.fixture
def student_toy_df() -> pd.DataFrame:
    return pd.DataFrame({
        "StudentID": range(1, 61),
        "StudyHours": [1.0 + (i % 10) for i in range(60)],
        "Attendance": [70 + (i % 30) for i in range(60)],
        "ExamScore": [50.0 + (i % 50) for i in range(60)],
    })


class TestDatasetIsolation:
    """Test suite ensuring complete artifact isolation between datasets in Postgres."""

    @pytest.fixture(autouse=True)
    def clean_db(self):
        eng = get_db_engine()
        with eng.connect() as conn:
            with conn.begin():
                conn.execute(text("DELETE FROM artifacts"))
        yield

    def test_independent_versioning_per_dataset(self):
        """Artifact versioning in one dataset never impacts another dataset in Postgres."""
        ds_a = "dataset_alpha"
        ds_b = "dataset_beta"

        # Save v1 in dataset_alpha
        v1_a = save_artifact({"name": "alpha_v1"}, ds_a, "dataset_profile", "json")
        assert v1_a == 1
        assert get_next_version(ds_a, "dataset_profile") == 2

        # Dataset beta should still start at v1
        assert get_next_version(ds_b, "dataset_profile") == 1
        v1_b = save_artifact({"name": "beta_v1"}, ds_b, "dataset_profile", "json")
        assert v1_b == 1

        # Save v2 in dataset_alpha
        v2_a = save_artifact({"name": "alpha_v2"}, ds_a, "dataset_profile", "json")
        assert v2_a == 2

        # Beta still only has v1, next version is 2
        assert get_next_version(ds_b, "dataset_profile") == 2
        latest_b = get_latest_version_path(ds_b, "dataset_profile", "json")
        assert latest_b == {"name": "beta_v1"}
        latest_a = get_latest_version_path(ds_a, "dataset_profile", "json")
        assert latest_a == {"name": "alpha_v2"}

    def test_api_analyze_and_clean_dataset_isolation(
        self, titanic_toy_df, student_toy_df
    ):
        """Process two different datasets via endpoints with separate dataset_ids."""
        # 1. Clean Titanic with dataset_id="ds_titanic"
        resp_t = client.post(
            "/clean-dataset",
            files={"file": ("titanic.csv", _make_csv_bytes(titanic_toy_df), "text/csv")},
            data={"target_column": "Survived", "dataset_id": "ds_titanic"},
        )
        assert resp_t.status_code == 200
        data_t = resp_t.json()
        assert data_t["dataset_id"] == "ds_titanic"

        # 2. Clean Student with dataset_id="ds_student"
        resp_s = client.post(
            "/clean-dataset",
            files={"file": ("student.csv", _make_csv_bytes(student_toy_df), "text/csv")},
            data={"target_column": "ExamScore", "dataset_id": "ds_student"},
        )
        assert resp_s.status_code == 200
        data_s = resp_s.json()
        assert data_s["dataset_id"] == "ds_student"

        # Read back cleaned data from DB for both datasets to verify isolation
        df_t = get_latest_artifact("ds_titanic", "cleaned")
        df_s = get_latest_artifact("ds_student", "cleaned")
        assert isinstance(df_t, pd.DataFrame)
        assert isinstance(df_s, pd.DataFrame)
        assert "Survived" in df_t.columns
        assert "ExamScore" not in df_t.columns
        assert "ExamScore" in df_s.columns
        assert "Survived" not in df_s.columns

    def test_missing_dataset_id_validation_on_downstream_endpoints(self):
        """Downstream endpoints without dataset_id return HTTP 400."""
        # /plan-ml
        resp = client.post("/plan-ml", json={})
        assert resp.status_code == 400
        assert "dataset_id is required" in resp.json()["detail"]

        # /train-model
        resp = client.post("/train-model", json={})
        assert resp.status_code == 400
        assert "dataset_id is required" in resp.json()["detail"]

        # /evaluate-model
        resp = client.post("/evaluate-model", json={})
        assert resp.status_code == 400
        assert "dataset_id is required" in resp.json()["detail"]

        # /generate-report
        resp = client.post("/generate-report", json={})
        assert resp.status_code == 400
        assert "dataset_id is required" in resp.json()["detail"]

    def test_nonexistent_dataset_id_returns_404(self):
        """Calling downstream endpoint with unknown dataset_id returns HTTP 404."""
        resp = client.post("/plan-ml", json={"dataset_id": "nonexistent_dataset_999"})
        assert resp.status_code == 404
        assert "Dataset artifacts not found" in resp.json()["detail"] or "not found" in resp.json()["detail"].lower()

    def test_report_agent_strictly_loads_scoped_dataset_artifacts(self):
        """ReportAgent scoped to dataset_A only discovers dataset_A's artifacts, ignoring dataset_B."""
        ds_a = "run_alpha"
        ds_b = "run_beta"

        # Populate all required artifacts for run_alpha
        save_artifact({"name": "alpha_profile", "selected_target": "TargetA"}, ds_a, "dataset_profile", "json")
        save_artifact({"summary": "alpha cleaned"}, ds_a, "cleaned_changelog", "json")
        save_artifact({"eda": "alpha eda"}, ds_a, "eda_bundle", "json")
        save_artifact({"fe": "alpha fe"}, ds_a, "feature_engineered", "json")
        save_artifact({"plan": "alpha plan"}, ds_a, "ml_plan", "json")
        save_artifact({"results": "alpha results"}, ds_a, "training_results", "json")
        save_artifact({"eval": "alpha eval"}, ds_a, "evaluation_bundle", "json")

        # Also populate some artifacts in run_beta
        save_artifact({"name": "beta_profile", "selected_target": "TargetB"}, ds_b, "dataset_profile", "json")

        # Load artifacts scoped to ds_a
        loaded_a = load_all_pipeline_artifacts(artifacts_dir=ds_a)
        assert loaded_a["artifacts"]["dataset_profile"]["name"] == "alpha_profile"
        assert loaded_a["artifacts"]["dataset_profile"]["selected_target"] == "TargetA"

        # If we load artifacts scoped to ds_b, it should fail with missing stage error because ds_b has only profile
        with pytest.raises(FileNotFoundError, match="Missing required artifact"):
            load_all_pipeline_artifacts(artifacts_dir=ds_b)

