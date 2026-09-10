from __future__ import annotations

import json
import os
import shutil
import pandas as pd
import pytest

from agents.versioning_utils import (
    get_latest_version_path,
    get_next_version,
    save_artifact,
)
from agents.pipeline_agents import (
    CleaningAgent,
    DatasetUnderstandingAgent,
    EDAAgent,
    FeatureEngineeringAgent,
)
from agents.ml_planning_agent import (
    load_planning_inputs,
    _build_planning_summary,
)


class TestVersioningUtils:
    """Unit tests for get_next_version, save_artifact, and get_latest_version_path."""

    def test_get_next_version_empty_or_nonexistent_dir(self, tmp_path):
        empty_dir = str(tmp_path / "empty_artifacts")
        assert get_next_version(empty_dir, "cleaned") == 1

        non_existent = str(tmp_path / "does_not_exist")
        assert get_next_version(non_existent, "dataset_profile") == 1

    def test_get_next_version_increments_existing_files(self, tmp_path):
        artifacts_dir = str(tmp_path / "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Write fake v1 and v2 files
        (tmp_path / "artifacts" / "cleaned_v1.csv").write_text("dummy")
        (tmp_path / "artifacts" / "cleaned_v2.csv").write_text("dummy")

        assert get_next_version(artifacts_dir, "cleaned") == 3

    def test_get_next_version_multiple_extensions_versioned_together(self, tmp_path):
        """cleaned_v1.csv and cleaned_v1_changelog.json count as version 1 of 'cleaned'."""
        artifacts_dir = str(tmp_path / "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        (tmp_path / "artifacts" / "cleaned_v1.csv").write_text("dummy")
        (tmp_path / "artifacts" / "cleaned_v1_changelog.json").write_text("{}")

        # Next version should be 2, not 3
        assert get_next_version(artifacts_dir, "cleaned") == 2

    def test_save_artifact_dataframe_and_dict(self, tmp_path):
        artifacts_dir = str(tmp_path / "artifacts")
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        csv_path = save_artifact(df, artifacts_dir, "cleaned", "csv")
        assert os.path.exists(csv_path)
        assert csv_path.endswith("cleaned_v1.csv")

        payload = {"status": "ok", "count": 3}
        json_path = save_artifact(payload, artifacts_dir, "dataset_profile", "json")
        assert os.path.exists(json_path)
        assert json_path.endswith("dataset_profile_v1.json")

        with open(json_path) as f:
            data = json.load(f)
        assert data["status"] == "ok"

    def test_get_latest_version_path_returns_highest_version(self, tmp_path):
        artifacts_dir = str(tmp_path / "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        (tmp_path / "artifacts" / "dataset_profile_v1.json").write_text("{}")
        (tmp_path / "artifacts" / "dataset_profile_v2.json").write_text("{}")
        (tmp_path / "artifacts" / "dataset_profile_v3.json").write_text("{}")

        latest = get_latest_version_path(artifacts_dir, "dataset_profile", "json")
        assert latest.endswith("dataset_profile_v3.json")

    def test_get_latest_version_path_raises_filenotfound(self, tmp_path):
        artifacts_dir = str(tmp_path / "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="No dataset_profile artifact found"):
            get_latest_version_path(artifacts_dir, "dataset_profile", "json")

        with pytest.raises(FileNotFoundError, match="No cleaned artifact found"):
            get_latest_version_path(str(tmp_path / "non_existent"), "cleaned", "csv")


class TestPipelineArtifactIntegrationAndVersioning:
    """Integration and re-run tests verifying artifact generation across Weeks 1-4."""

    @pytest.fixture
    def titanic_test_data(self) -> pd.DataFrame:
        n = 100
        return pd.DataFrame({
            "PassengerId": range(1, n + 1),
            "Age": [22.0 + (i % 30) for i in range(n)],
            "Fare": [7.25 + (i % 50) for i in range(n)],
            "Sex": ["male", "female"] * (n // 2),
            "Survived": [0, 1] * (n // 2),
        })

    def test_sequential_pipeline_run_produces_v1_artifacts(self, titanic_test_data, tmp_path):
        """3. Integration test: run Week 1 through Week 4 sequentially on Titanic fixture,
        assert all expected artifact files exist on disk with version suffix _v1.
        """
        artifacts_dir = str(tmp_path / "artifacts")
        state = {
            "df": titanic_test_data,
            "artifacts_dir": artifacts_dir,
            "user_selected_target": "Survived",
        }

        state = DatasetUnderstandingAgent().run(state)
        state = CleaningAgent().run(state)
        state = EDAAgent().run(state)
        state = FeatureEngineeringAgent().run(state)

        # Verify all v1 files exist
        assert os.path.exists(os.path.join(artifacts_dir, "dataset_profile_v1.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "cleaned_v1.csv"))
        assert os.path.exists(os.path.join(artifacts_dir, "cleaned_v1_changelog.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "eda_bundle_v1.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "feature_engineered_v1.csv"))
        assert os.path.exists(os.path.join(artifacts_dir, "feature_engineered_v1.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "pipeline_v1.pkl"))

    def test_pipeline_rerun_produces_v2_artifacts_without_overwriting(self, titanic_test_data, tmp_path):
        """4. Re-run test: run the same sequence twice, assert every artifact type
        is now at _v2, and no file was overwritten in place (v1 files still exist and are unchanged).
        """
        artifacts_dir = str(tmp_path / "artifacts")

        # Run 1
        state1 = {
            "df": titanic_test_data,
            "artifacts_dir": artifacts_dir,
            "user_selected_target": "Survived",
        }
        DatasetUnderstandingAgent().run(state1)
        CleaningAgent().run(state1)
        EDAAgent().run(state1)
        FeatureEngineeringAgent().run(state1)

        # Snapshot v1 content
        v1_profile = open(os.path.join(artifacts_dir, "dataset_profile_v1.json")).read()
        v1_cleaning = open(os.path.join(artifacts_dir, "cleaned_v1_changelog.json")).read()
        v1_eda = open(os.path.join(artifacts_dir, "eda_bundle_v1.json")).read()
        v1_fe_csv = open(os.path.join(artifacts_dir, "feature_engineered_v1.csv")).read()

        # Run 2
        state2 = {
            "df": titanic_test_data,
            "artifacts_dir": artifacts_dir,
            "user_selected_target": "Survived",
        }
        DatasetUnderstandingAgent().run(state2)
        CleaningAgent().run(state2)
        EDAAgent().run(state2)
        FeatureEngineeringAgent().run(state2)

        # Verify all v2 files exist
        assert os.path.exists(os.path.join(artifacts_dir, "dataset_profile_v2.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "cleaned_v2.csv"))
        assert os.path.exists(os.path.join(artifacts_dir, "cleaned_v2_changelog.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "eda_bundle_v2.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "feature_engineered_v2.csv"))
        assert os.path.exists(os.path.join(artifacts_dir, "feature_engineered_v2.json"))
        assert os.path.exists(os.path.join(artifacts_dir, "pipeline_v2.pkl"))

        # Verify v1 files are preserved exactly
        assert open(os.path.join(artifacts_dir, "dataset_profile_v1.json")).read() == v1_profile
        assert open(os.path.join(artifacts_dir, "cleaned_v1_changelog.json")).read() == v1_cleaning
        assert open(os.path.join(artifacts_dir, "eda_bundle_v1.json")).read() == v1_eda
        assert open(os.path.join(artifacts_dir, "feature_engineered_v1.csv")).read() == v1_fe_csv

    def test_load_planning_inputs_extracts_clean_summary(self, titanic_test_data, tmp_path):
        """5. Test load_planning_inputs() against the saved artifacts,
        assert the returned summary dict has all expected keys and none of the Plotly chart keys.
        """
        artifacts_dir = str(tmp_path / "artifacts")
        state = {
            "df": titanic_test_data,
            "artifacts_dir": artifacts_dir,
            "user_selected_target": "Survived",
        }
        DatasetUnderstandingAgent().run(state)
        CleaningAgent().run(state)
        EDAAgent().run(state)
        FeatureEngineeringAgent().run(state)

        summary = load_planning_inputs(artifacts_dir)

        # Expected keys
        expected_keys = {
            "target_column",
            "earlier_problem_type_guess",
            "class_balance",
            "skewness",
            "flagged_correlations",
            "eda_insights",
            "eda_summary",
            "cleaning_summary",
            "final_columns",
            "encoding_map",
            "excluded_columns",
            "collinear_pairs",
        }
        for k in expected_keys:
            assert k in summary, f"Expected key '{k}' missing from planning summary"

        # Explicitly ensure Plotly chart keys are NOT present
        plotly_keys = ["histograms", "boxplots", "correlation_heatmap", "target_distribution"]
        for pk in plotly_keys:
            assert pk not in summary, f"Plotly chart key '{pk}' should not be in planning summary"

    def test_load_planning_inputs_raises_error_if_artifacts_missing(self, titanic_test_data, tmp_path):
        """6. Test load_planning_inputs() raises a clear FileNotFoundError if artifacts
        are missing (e.g. Weeks 1-3 ran but Week 4 hasn't).
        """
        artifacts_dir = str(tmp_path / "artifacts")
        state = {
            "df": titanic_test_data,
            "artifacts_dir": artifacts_dir,
            "user_selected_target": "Survived",
        }
        # Run only Weeks 1-3
        DatasetUnderstandingAgent().run(state)
        CleaningAgent().run(state)
        EDAAgent().run(state)

        with pytest.raises(FileNotFoundError, match="feature_engineered"):
            load_planning_inputs(artifacts_dir)
