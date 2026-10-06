from __future__ import annotations

import io
import json
import os
import shutil
import pandas as pd
import pytest
from sqlalchemy import text
from agents.versioning_utils import (
    get_db_engine,
    get_latest_artifact,
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
    """Unit tests for get_next_version, save_artifact, and get_latest_artifact against DB storage."""

    @pytest.fixture(autouse=True)
    def clean_db(self):
        eng = get_db_engine()
        with eng.connect() as conn:
            with conn.begin():
                conn.execute(text("DELETE FROM artifacts"))
        yield

    def test_get_next_version_empty_dataset(self):
        """New dataset starts at version 1."""
        assert get_next_version("dataset_brand_new_123", "cleaned") == 1
        assert get_next_version("dataset_brand_new_123", "dataset_profile") == 1

    def test_get_next_version_increments_existing_records(self):
        """Saving increments version numbers per (dataset_id, artifact_type)."""
        ds_id = "ds_increment_test"
        v1 = save_artifact({"run": 1}, ds_id, "cleaned")
        assert v1 == 1
        assert get_next_version(ds_id, "cleaned") == 2

        v2 = save_artifact({"run": 2}, ds_id, "cleaned")
        assert v2 == 2
        assert get_next_version(ds_id, "cleaned") == 3

    def test_save_artifact_dataframe_and_dict(self):
        """DataFrames and dicts round-trip accurately through save_artifact and get_latest_artifact."""
        ds_id = "ds_roundtrip_test"
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        v_df = save_artifact(df, ds_id, "cleaned")
        assert v_df == 1
        df_retrieved = get_latest_artifact(ds_id, "cleaned")
        assert isinstance(df_retrieved, pd.DataFrame)
        assert df.equals(df_retrieved)

        payload = {"status": "ok", "count": 3}
        v_json = save_artifact(payload, ds_id, "dataset_profile")
        assert v_json == 1
        data = get_latest_artifact(ds_id, "dataset_profile")
        assert isinstance(data, dict)
        assert data["status"] == "ok"
        assert data["count"] == 3

    def test_binary_artifact_round_trip(self):
        """Binary artifacts (e.g. PDF bytes) round-trip with exact byte equality."""
        ds_id = "ds_binary_test"
        pdf_bytes = b"%PDF-1.4 sample binary content with special \x00\xff\xfe bytes"
        v_pdf = save_artifact(pdf_bytes, ds_id, "report", is_binary=True)
        assert v_pdf == 1

        retrieved_bytes = get_latest_artifact(ds_id, "report")
        assert isinstance(retrieved_bytes, bytes)
        assert retrieved_bytes == pdf_bytes

    def test_get_latest_version_path_returns_highest_version(self):
        """get_latest_artifact returns the highest version data."""
        ds_id = "ds_multi_version"
        save_artifact({"v": 1}, ds_id, "dataset_profile")
        save_artifact({"v": 2}, ds_id, "dataset_profile")
        save_artifact({"v": 3}, ds_id, "dataset_profile")

        latest = get_latest_artifact(ds_id, "dataset_profile")
        assert latest["v"] == 3

    def test_get_latest_version_path_raises_filenotfound(self):
        """Missing artifacts raise FileNotFoundError with clear message."""
        with pytest.raises(FileNotFoundError, match="No dataset_profile artifact found"):
            get_latest_artifact("ds_nonexistent", "dataset_profile")


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

    def test_sequential_pipeline_run_produces_v1_artifacts(self, titanic_test_data):
        """Integration test: run Week 1 through Week 4 sequentially on Titanic fixture."""
        ds_id = "ds_sequential_v1_test"
        state = {
            "df": titanic_test_data,
            "dataset_id": ds_id,
            "artifacts_dir": ds_id,
            "user_selected_target": "Survived",
        }

        state = DatasetUnderstandingAgent().run(state)
        state = CleaningAgent().run(state)
        state = EDAAgent().run(state)
        state = FeatureEngineeringAgent().run(state)

        # Verify artifacts exist in DB
        prof = get_latest_artifact(ds_id, "dataset_profile")
        assert prof is not None
        cleaned_df = get_latest_artifact(ds_id, "cleaned")
        assert isinstance(cleaned_df, pd.DataFrame)
        eda = get_latest_artifact(ds_id, "eda_bundle")
        assert eda is not None
        fe = get_latest_artifact(ds_id, "feature_engineered")
        assert fe is not None

    def test_pipeline_rerun_produces_v2_artifacts_without_overwriting(self, titanic_test_data):
        """Re-run test: run the same sequence twice, assert artifacts are now at v2."""
        ds_id = "ds_rerun_v2_test"

        # Run 1
        state1 = {
            "df": titanic_test_data,
            "dataset_id": ds_id,
            "artifacts_dir": ds_id,
            "user_selected_target": "Survived",
        }
        DatasetUnderstandingAgent().run(state1)
        CleaningAgent().run(state1)
        EDAAgent().run(state1)
        FeatureEngineeringAgent().run(state1)

        assert get_next_version(ds_id, "dataset_profile") == 2
        assert get_next_version(ds_id, "cleaned") == 2
        assert get_next_version(ds_id, "eda_bundle") == 2
        assert get_next_version(ds_id, "feature_engineered") == 2

        # Run 2
        state2 = {
            "df": titanic_test_data,
            "dataset_id": ds_id,
            "artifacts_dir": ds_id,
            "user_selected_target": "Survived",
        }
        DatasetUnderstandingAgent().run(state2)
        CleaningAgent().run(state2)
        EDAAgent().run(state2)
        FeatureEngineeringAgent().run(state2)

        assert get_next_version(ds_id, "dataset_profile") == 3
        assert get_next_version(ds_id, "cleaned") == 3
        assert get_next_version(ds_id, "eda_bundle") == 3
        assert get_next_version(ds_id, "feature_engineered") == 3

    def test_load_planning_inputs_extracts_clean_summary(self, titanic_test_data):
        """Test load_planning_inputs() against the saved artifacts."""
        ds_id = "ds_planning_summary_test"
        state = {
            "df": titanic_test_data,
            "dataset_id": ds_id,
            "artifacts_dir": ds_id,
            "user_selected_target": "Survived",
        }
        DatasetUnderstandingAgent().run(state)
        CleaningAgent().run(state)
        EDAAgent().run(state)
        FeatureEngineeringAgent().run(state)

        summary = load_planning_inputs(ds_id)

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

    def test_load_planning_inputs_raises_error_if_artifacts_missing(self, titanic_test_data):
        """Test load_planning_inputs() raises a clear FileNotFoundError if artifacts are missing."""
        ds_id = "ds_incomplete_test"
        state = {
            "df": titanic_test_data,
            "dataset_id": ds_id,
            "artifacts_dir": ds_id,
            "user_selected_target": "Survived",
        }
        # Run only Weeks 1-3
        DatasetUnderstandingAgent().run(state)
        CleaningAgent().run(state)
        EDAAgent().run(state)

        with pytest.raises(FileNotFoundError, match="feature_engineered"):
            load_planning_inputs(ds_id)
