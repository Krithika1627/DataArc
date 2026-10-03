"""Comprehensive tests for Dataset-Scoped Artifact Isolation (Week 7 Part 1).

Covers:
1. Dataset isolation across runs (two datasets processed in the same session)
2. Missing and non-existent dataset_id validation (HTTP 400 and 404)
3. Report traceability (ReportAgent loads strictly from scoped subfolder)
4. Independent versioning per dataset_id
"""
from __future__ import annotations

import io
import json
import os
import unittest.mock
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from agents.versioning_utils import (
    get_dataset_artifacts_dir,
    get_next_version,
    save_artifact,
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
    """Test suite ensuring complete artifact isolation between datasets."""

    def test_independent_versioning_per_dataset(self, tmp_path):
        """Artifact versioning in one dataset subfolder never impacts another dataset."""
        base_dir = str(tmp_path / "artifacts")
        dir_a = get_dataset_artifacts_dir(base_dir, "dataset_alpha")
        dir_b = get_dataset_artifacts_dir(base_dir, "dataset_beta")

        # Save v1 in dataset_alpha
        v1_a = save_artifact({"name": "alpha_v1"}, dir_a, "dataset_profile", "json")
        assert "v1.json" in v1_a
        assert get_next_version(dir_a, "dataset_profile") == 2

        # Dataset beta should still start at v1
        assert get_next_version(dir_b, "dataset_profile") == 1
        v1_b = save_artifact({"name": "beta_v1"}, dir_b, "dataset_profile", "json")
        assert "v1.json" in v1_b

        # Save v2 in dataset_alpha
        v2_a = save_artifact({"name": "alpha_v2"}, dir_a, "dataset_profile", "json")
        assert "v2.json" in v2_a

        # Beta still only has v1, next version is 2
        assert get_next_version(dir_b, "dataset_profile") == 2
        latest_b = get_latest_version_path(dir_b, "dataset_profile", "json")
        assert latest_b == v1_b
        latest_a = get_latest_version_path(dir_a, "dataset_profile", "json")
        assert latest_a == v2_a

    def test_api_analyze_and_clean_dataset_isolation(
        self, tmp_path, monkeypatch, titanic_toy_df, student_toy_df
    ):
        """Process two different datasets via endpoints with separate dataset_ids."""
        monkeypatch.chdir(tmp_path)
        base_dir = tmp_path / "artifacts"

        # 1. Clean Titanic with dataset_id="ds_titanic"
        resp_t = client.post(
            "/clean-dataset",
            files={"file": ("titanic.csv", _make_csv_bytes(titanic_toy_df), "text/csv")},
            data={"target_column": "Survived", "dataset_id": "ds_titanic"},
        )
        assert resp_t.status_code == 200
        data_t = resp_t.json()
        assert data_t["dataset_id"] == "ds_titanic"
        assert "ds_titanic" in data_t["artifact_path"]

        # 2. Clean Student with dataset_id="ds_student"
        resp_s = client.post(
            "/clean-dataset",
            files={"file": ("student.csv", _make_csv_bytes(student_toy_df), "text/csv")},
            data={"target_column": "ExamScore", "dataset_id": "ds_student"},
        )
        assert resp_s.status_code == 200
        data_s = resp_s.json()
        assert data_s["dataset_id"] == "ds_student"
        assert "ds_student" in data_s["artifact_path"]

        # Verify filesystem subdirectories are strictly partitioned
        dir_t = base_dir / "ds_titanic"
        dir_s = base_dir / "ds_student"
        assert dir_t.exists()
        assert dir_s.exists()

        files_t = os.listdir(dir_t)
        files_s = os.listdir(dir_s)

        # Titanic directory has cleaned Titanic artifacts
        assert any("cleaned_v1.csv" in f for f in files_t)
        # Student directory has cleaned Student artifacts
        assert any("cleaned_v1.csv" in f for f in files_s)

        # Read back cleaned data from both to verify no content mix-up
        df_t = pd.read_csv(data_t["artifact_path"])
        df_s = pd.read_csv(data_s["artifact_path"])
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

        # /train-models
        resp = client.post("/train-models", json={})
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

    def test_nonexistent_dataset_id_returns_404(self, tmp_path, monkeypatch):
        """Calling downstream endpoint with unknown dataset_id returns HTTP 404."""
        monkeypatch.chdir(tmp_path)
        resp = client.post("/plan-ml", json={"dataset_id": "nonexistent_dataset_999"})
        assert resp.status_code == 404
        assert "Dataset artifacts directory not found" in resp.json()["detail"]

    def test_report_agent_strictly_loads_scoped_dataset_artifacts(self, tmp_path):
        """ReportAgent scoped to dataset_A only discovers dataset_A's artifacts, ignoring dataset_B."""
        base_dir = str(tmp_path / "artifacts")
        dir_a = get_dataset_artifacts_dir(base_dir, "run_alpha")
        dir_b = get_dataset_artifacts_dir(base_dir, "run_beta")

        # Populate all required artifacts for run_alpha
        save_artifact({"name": "alpha_profile", "selected_target": "TargetA"}, dir_a, "dataset_profile", "json")
        with open(os.path.join(dir_a, "cleaned_v1_changelog.json"), "w") as f:
            json.dump({"summary": "alpha cleaned"}, f)
        save_artifact({"eda": "alpha eda"}, dir_a, "eda_bundle", "json")
        save_artifact({"fe": "alpha fe"}, dir_a, "feature_engineered", "json")
        save_artifact({"plan": "alpha plan"}, dir_a, "ml_plan", "json")
        save_artifact({"results": "alpha results"}, dir_a, "training_results", "json")
        save_artifact({"eval": "alpha eval"}, dir_a, "evaluation_bundle", "json")

        # Also populate some artifacts in run_beta
        save_artifact({"name": "beta_profile", "selected_target": "TargetB"}, dir_b, "dataset_profile", "json")

        # Load artifacts scoped to dir_a
        loaded_a = load_all_pipeline_artifacts(artifacts_dir=dir_a)
        assert loaded_a["artifacts"]["dataset_profile"]["name"] == "alpha_profile"
        assert loaded_a["artifacts"]["dataset_profile"]["selected_target"] == "TargetA"

        # If we load artifacts scoped to dir_b, it should fail with missing stage error because dir_b has only profile
        with pytest.raises(FileNotFoundError, match="Missing required artifact"):
            load_all_pipeline_artifacts(artifacts_dir=dir_b)
