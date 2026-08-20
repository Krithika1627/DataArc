from __future__ import annotations

import io
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app, FeatureEngineeringResponse, CollinearPair

client = TestClient(app)


def _make_csv_buffer(df: pd.DataFrame) -> io.BytesIO:
    """Serialize a DataFrame to an in-memory CSV buffer."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cleaned_csv_path(tmp_path) -> str:
    """Write a basic cleaned CSV (simulating Week 2 output) and return its path."""
    df = pd.DataFrame({
        "passenger_id": range(100),
        "age": [20 + (i % 50) for i in range(100)],
        "fare": [5 + (i % 80) for i in range(100)],
        "sex": ["male", "female"] * 50,
        "embarked": (["S", "C", "Q"] * 34)[:100],
        "survived": [0, 1] * 50,
    })
    path = os.path.join(str(tmp_path), "cleaned_v1.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def collinear_csv_path(tmp_path) -> str:
    """Write a CSV with two forced-highly-correlated numeric columns.

    Uses clean integer-modulo patterns (low cardinality) so neither column
    is flagged as ID-like (>95% unique). x and y are ~0.99 correlated.
    """
    n = 200
    x = [i % 50 for i in range(n)]
    y = [2 * (i % 50) + (i % 7) for i in range(n)]
    df = pd.DataFrame({
        "feature_a": x,
        "feature_b": y,
        "feature_c": [i % 17 for i in range(n)],
        "target": [0, 1] * (n // 2),
    })
    path = os.path.join(str(tmp_path), "cleaned_v1.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def collinear_csv_with_id(tmp_path) -> str:
    """Write a CSV with forced-correlated columns, an ID-like column, and a target.

    Uses clean integer-modulo patterns (low cardinality) so price/value aren't
    flagged as ID-like. The ``id`` column is truly ID-like (100% unique).
    """
    n = 200
    price = [i % 30 for i in range(n)]
    value = [3 * (i % 30) + (i % 5) for i in range(n)]
    df = pd.DataFrame({
        "id": range(n),
        "price": price,
        "value": value,
        "category": (["A", "B", "C"] * 67)[:n],
        "label": [0, 1] * (n // 2),
    })
    path = os.path.join(str(tmp_path), "cleaned_v1.csv")
    df.to_csv(path, index=False)
    return path


def _fe_upload(df: pd.DataFrame, filename: str = "test.csv", **form_data):
    """Helper: build multipart upload args for /run-feature-engineering."""
    buf = _make_csv_buffer(df)
    files = {"file": (filename, buf, "text/csv")}
    return files, form_data


def _run_fe(df: pd.DataFrame, target_column: str, problem_type: str = "classification",
            filename: str = "test.csv", **extra_form):
    """Call /run-feature-engineering with a DataFrame upload and return the response."""
    form = {"target_column": target_column, "problem_type": problem_type}
    form.update(extra_form)
    files = {"file": (filename, _make_csv_buffer(df), "text/csv")}
    return client.post("/run-feature-engineering", files=files, data=form)


# ── /run-feature-engineering tests ────────────────────────────────────────────


class TestRunFeatureEngineering:
    """Tests for POST /run-feature-engineering."""

    def test_valid_request_returns_200_and_response_model(
        self, tmp_path, monkeypatch
    ):
        """Valid request returns 200 and response round-trips through FeatureEngineeringResponse."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "survived": [0, 1, 0, 1, 0],
        })

        response = _run_fe(df, target_column="survived")

        assert response.status_code == 200
        data = response.json()

        # Double-check with Pydantic model
        parsed = FeatureEngineeringResponse(**data)
        assert parsed.encoding_map
        assert isinstance(parsed.collinear_pairs, list)
        assert isinstance(parsed.excluded_columns, list)
        assert parsed.artifact_path
        assert parsed.pipeline_path

    def test_v1_artifact_created_and_target_absent(
        self, tmp_path, monkeypatch
    ):
        """feature_engineered_v1.csv exists on disk and does not contain the target column."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "survived": [0, 1, 0, 1, 0],
        })

        response = _run_fe(df, target_column="survived")

        assert response.status_code == 200
        data = response.json()
        csv_path = data["artifact_path"]

        assert os.path.isfile(csv_path)
        assert "feature_engineered_v1.csv" in csv_path

        result_df = pd.read_csv(csv_path)
        assert "survived" not in result_df.columns

    def test_pipeline_pickle_exists_and_loadable(
        self, tmp_path, monkeypatch
    ):
        """pipeline_v1.pkl exists and can be unpickled into a sklearn Pipeline."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "survived": [0, 1, 0, 1, 0],
        })

        response = _run_fe(df, target_column="survived")

        assert response.status_code == 200
        data = response.json()
        pkl_path = data["pipeline_path"]

        assert os.path.isfile(pkl_path)
        assert "pipeline_v1.pkl" in pkl_path

        with open(pkl_path, "rb") as f:
            pipeline = pickle.load(f)

        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline)

    def test_collinear_pairs_detected_and_both_columns_present(
        self, tmp_path, monkeypatch
    ):
        """When two columns are forced-correlated, collinear_pairs is non-empty
        and both columns are still present in the saved CSV."""
        monkeypatch.chdir(tmp_path)

        n = 200
        x = [i % 50 for i in range(n)]
        y = [2 * (i % 50) + (i % 7) for i in range(n)]
        df = pd.DataFrame({
            "feature_a": x,
            "feature_b": y,
            "feature_c": [i % 17 for i in range(n)],
            "target": [0, 1] * (n // 2),
        })

        response = _run_fe(df, target_column="target")

        assert response.status_code == 200
        data = response.json()

        # Check collinear_pairs is non-empty
        assert len(data["collinear_pairs"]) > 0

        # Find the feature_a / feature_b pair
        pair_found = False
        for p in data["collinear_pairs"]:
            cols = {p["col_a"], p["col_b"]}
            if cols == {"feature_a", "feature_b"}:
                pair_found = True
                assert p["correlation"] > 0.85
                break
        assert pair_found, f"Expected feature_a/feature_b pair, got {data['collinear_pairs']}"

        # Both columns are still present in the saved CSV (flagged, not dropped)
        csv_df = pd.read_csv(data["artifact_path"])
        assert "scaled__feature_a" in csv_df.columns
        assert "scaled__feature_b" in csv_df.columns

    def test_invalid_file_type_returns_400(self, tmp_path, monkeypatch):
        """Non-CSV file returns 400."""
        monkeypatch.chdir(tmp_path)

        buf = io.BytesIO(b"some,data\n1,2")
        files = {"file": ("data.json", buf, "application/json")}
        response = client.post(
            "/run-feature-engineering",
            files=files,
            data={"target_column": "a", "problem_type": "regression"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid file type" in data["detail"]

    def test_invalid_target_column_returns_400(self, tmp_path, monkeypatch):
        """Nonexistent target_column in uploaded CSV returns 400."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        response = _run_fe(df, target_column="nonexistent_col")

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_empty_csv_returns_400(self, tmp_path, monkeypatch):
        """Empty CSV (headers only) returns 400."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")})

        response = _run_fe(df, target_column="a", problem_type="regression")

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "no data rows" in data["detail"].lower()

    def test_with_exclude_columns(self, tmp_path, monkeypatch):
        """exclude_columns are excluded from the transformed output."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "embarked": ["S", "C", "Q", "S", "C"],
            "survived": [0, 1, 0, 1, 0],
        })

        response = _run_fe(
            df, target_column="survived",
            exclude_columns=json.dumps(["embarked"]),
        )

        assert response.status_code == 200
        data = response.json()
        assert "embarked" in data["excluded_columns"]
        assert data["encoding_map"]["embarked"] == "excluded"

    def test_version_increments_on_rerun(self, tmp_path, monkeypatch):
        """Calling /run-feature-engineering twice writes v1 then v2, never overwrites v1."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "survived": [0, 1, 0, 1, 0],
        })

        # First call -> v1
        resp1 = _run_fe(df, target_column="survived")
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert "feature_engineered_v1.csv" in data1["artifact_path"]
        v1_csv = data1["artifact_path"]
        v1_mtime = os.path.getmtime(v1_csv)

        # Second call -> v2
        resp2 = _run_fe(df, target_column="survived")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert "feature_engineered_v2.csv" in data2["artifact_path"]
        v2_csv = data2["artifact_path"]

        # v1 file is untouched
        assert os.path.isfile(v1_csv)
        assert os.path.getmtime(v1_csv) == v1_mtime
        assert v1_csv != v2_csv

    def test_with_ordinal_columns(self, tmp_path, monkeypatch):
        """ordinal_columns parameter is passed through to build_feature_pipeline."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "quality": ["bad", "good", "excellent", "bad", "good"],
            "target": [0, 1, 0, 1, 0],
        })

        response = _run_fe(
            df, target_column="target",
            ordinal_columns=json.dumps({"quality": ["bad", "good", "excellent"]}),
        )

        assert response.status_code == 200
        data = response.json()
        assert data["encoding_map"]["quality"] == "ordinal"

    def test_encoding_map_covers_original_columns(self, tmp_path, monkeypatch):
        """encoding_map has entries for every original column in the uploaded CSV."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "survived": [0, 1, 0, 1, 0],
        })

        response = _run_fe(df, target_column="survived")

        assert response.status_code == 200
        data = response.json()

        for col in df.columns:
            assert col in data["encoding_map"], f"Missing encoding_map entry for '{col}'"

    def test_with_cleaning_changelog_path(self, tmp_path, monkeypatch):
        """cleaning_changelog_path is loaded and additional exclusions are applied."""
        monkeypatch.chdir(tmp_path)

        changelog = {
            "outlier_handling": {
                "columns_skipped_target_protected": [{"column": "embarked"}],
            },
        }
        changelog_path = os.path.join(str(tmp_path), "changelog.json")
        with open(changelog_path, "w") as f:
            json.dump(changelog, f)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "embarked": ["S", "C", "Q", "S", "C"],
            "survived": [0, 1, 0, 1, 0],
        })

        response = _run_fe(
            df, target_column="survived",
            cleaning_changelog_path=changelog_path,
        )

        assert response.status_code == 200
        data = response.json()
        # embarked was flagged in the changelog as target-protected -> should be excluded
        assert "embarked" in data["excluded_columns"]


# ── /apply-collinearity-drop tests ────────────────────────────────────────────


class TestApplyCollinearityDrop:
    """Tests for POST /apply-collinearity-drop."""

    def _make_v1(self, tmp_path, monkeypatch, df, target_column="label"):
        """Helper: run feature engineering to create v1, return response data."""
        monkeypatch.chdir(tmp_path)
        resp = _run_fe(df, target_column=target_column)
        assert resp.status_code == 200
        return resp.json()

    def test_drop_column_creates_v2_and_v1_untouched(
        self, tmp_path, monkeypatch
    ):
        """Dropping a column from v1 creates v2; v1 remains untouched on disk."""
        n = 200
        price = [i % 30 for i in range(n)]
        value = [3 * (i % 30) + (i % 5) for i in range(n)]
        df = pd.DataFrame({
            "id": range(n),
            "price": price,
            "value": value,
            "category": (["A", "B", "C"] * 67)[:n],
            "label": [0, 1] * (n // 2),
        })

        data1 = self._make_v1(tmp_path, monkeypatch, df)
        v1_path = data1["artifact_path"]
        assert "feature_engineered_v1.csv" in v1_path
        v1_mtime = os.path.getmtime(v1_path)

        # Read the v1 CSV to get the actual transformed column names
        v1_df = pd.read_csv(v1_path)
        v1_cols = list(v1_df.columns)

        # Drop the transformed version of "value" (scaled__value)
        drop_col = "scaled__value"
        assert drop_col in v1_cols, f"{drop_col} not in v1 columns: {v1_cols}"

        resp2 = client.post(
            "/apply-collinearity-drop",
            json={
                "artifact_path": v1_path,
                "columns_to_drop": [drop_col],
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()

        assert "feature_engineered_v2.csv" in data2["artifact_path"]
        v2_df = pd.read_csv(data2["artifact_path"])

        # The dropped column should not appear in v2
        assert drop_col not in v2_df.columns

        # v1 is untouched
        assert os.path.isfile(v1_path)
        assert os.path.getmtime(v1_path) == v1_mtime

    def test_drop_nonexistent_column_returns_400(
        self, tmp_path, monkeypatch
    ):
        """Dropping a column that doesn't exist returns 400."""
        n = 200
        df = pd.DataFrame({
            "id": range(n),
            "price": [i % 30 for i in range(n)],
            "value": [3 * (i % 30) + (i % 5) for i in range(n)],
            "category": (["A", "B", "C"] * 67)[:n],
            "label": [0, 1] * (n // 2),
        })

        data1 = self._make_v1(tmp_path, monkeypatch, df)
        v1_path = data1["artifact_path"]

        resp2 = client.post(
            "/apply-collinearity-drop",
            json={
                "artifact_path": v1_path,
                "columns_to_drop": ["nonexistent_column_xyz"],
            },
        )
        assert resp2.status_code == 400
        assert "not found" in resp2.json()["detail"].lower()

    def test_drop_from_nonexistent_artifact_returns_400(self, tmp_path, monkeypatch):
        """Dropping from a nonexistent artifact path returns 400."""
        monkeypatch.chdir(tmp_path)

        response = client.post(
            "/apply-collinearity-drop",
            json={
                "artifact_path": "/nonexistent/feature_engineered_v1.csv",
                "columns_to_drop": ["col_a"],
            },
        )

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_drop_preserves_other_columns(
        self, tmp_path, monkeypatch
    ):
        """Dropping one correlated column preserves the other columns in the output."""
        n = 200
        price = [i % 30 for i in range(n)]
        value = [3 * (i % 30) + (i % 5) for i in range(n)]
        df = pd.DataFrame({
            "id": range(n),
            "price": price,
            "value": value,
            "category": (["A", "B", "C"] * 67)[:n],
            "label": [0, 1] * (n // 2),
        })

        data1 = self._make_v1(tmp_path, monkeypatch, df)
        v1_path = data1["artifact_path"]

        # Drop scaled__value and check scaled__price still present
        resp2 = client.post(
            "/apply-collinearity-drop",
            json={
                "artifact_path": v1_path,
                "columns_to_drop": ["scaled__value"],
            },
        )
        assert resp2.status_code == 200
        v2_df = pd.read_csv(resp2.json()["artifact_path"])

        assert "scaled__value" not in v2_df.columns
        assert "scaled__price" in v2_df.columns

    def test_encoding_map_marks_dropped_columns(
        self, tmp_path, monkeypatch
    ):
        """Dropped columns are marked as 'dropped' in encoding_map."""
        n = 200
        df = pd.DataFrame({
            "id": range(n),
            "price": [i % 30 for i in range(n)],
            "value": [3 * (i % 30) + (i % 5) for i in range(n)],
            "category": (["A", "B", "C"] * 67)[:n],
            "label": [0, 1] * (n // 2),
        })

        data1 = self._make_v1(tmp_path, monkeypatch, df)
        v1_path = data1["artifact_path"]

        resp2 = client.post(
            "/apply-collinearity-drop",
            json={
                "artifact_path": v1_path,
                "columns_to_drop": ["scaled__value"],
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["encoding_map"]["scaled__value"] == "dropped"

    def test_metadata_sidecar_persists_and_is_readable(
        self, tmp_path, monkeypatch
    ):
        """The pipeline metadata JSON sidecar is written during /run-feature-engineering
        and is readable back by /apply-collinearity-drop."""
        monkeypatch.chdir(tmp_path)

        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40],
            "fare": [10.0, 20.0, 30.0, 40.0, 50.0],
            "sex": ["male", "female", "male", "female", "male"],
            "survived": [0, 1, 0, 1, 0],
        })

        resp1 = _run_fe(df, target_column="survived")
        assert resp1.status_code == 200
        data1 = resp1.json()

        # Check metadata sidecar exists
        meta_path = data1["pipeline_path"].replace(".pkl", ".json")
        assert os.path.isfile(meta_path)

        with open(meta_path) as f:
            meta = json.load(f)
        assert "encoding_map" in meta
        assert "collinear_pairs" in meta
        assert "excluded_columns" in meta
