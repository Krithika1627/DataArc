from __future__ import annotations

import io
import json
import os
import unittest.mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from main import EDAResponse

client = TestClient(app)


def _make_csv_buffer(df: pd.DataFrame) -> io.BytesIO:
    """Serialize a DataFrame to an in-memory CSV buffer."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def mock_llm_explanation():
    """Fixture that patches generate_cleaning_explanation to return a known response."""
    fake_explanation = {
        "summary": "Cleaned the dataset by removing duplicates, fixing dtypes, imputing missing values, and capping outliers.",
        "details": [
            "Removed duplicate rows (20.0% of the dataset).",
            "Converted price_str from object to float64 by stripping currency symbols.",
            "Imputed missing values in score using median because the distribution was symmetric.",
            "Capped 2 extreme outliers in outliers_raw at IQR bounds.",
        ],
    }
    with unittest.mock.patch(
        "agents.data_cleaning.generate_cleaning_explanation",
        return_value=fake_explanation,
    ):
        yield


class TestCleanDatasetEndpoint:
    """Tests for the POST /clean-dataset endpoint."""

    def test_valid_csv_no_target(self, mock_llm_explanation, tmp_path, monkeypatch):
        """Valid CSV with no target_column specified returns 200 with expected structure."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
            "score": [10.0, 20.0, 30.0],
        })
        buf = _make_csv_buffer(df)

        response = client.post(
            "/clean-dataset",
            files={"file": ("test.csv", buf, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        assert "artifact_path" in data
        assert "changelog_path" in data
        assert "summary" in data

        summary = data["summary"]
        assert "duplicate_removal" in summary
        assert "missing_value_imputation" in summary
        assert "outlier_handling" in summary
        assert "dtype_fixing" in summary
        assert "llm_explanation" in summary

        assert summary["llm_explanation"]["summary"] != ""
        assert isinstance(summary["llm_explanation"]["details"], list)

    def test_valid_csv_with_target_and_cap(self, mock_llm_explanation, tmp_path, monkeypatch):
        """Valid CSV with target_column and cap_target=True returns 200."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "id": [1, 2, 3, 1],  # one duplicate
            "value": [10.0, 20.0, 30.0, 500.0],  # has an outlier
            "label": ["a", "b", "c", "a"],
        })
        buf = _make_csv_buffer(df)

        response = client.post(
            "/clean-dataset",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": "value", "cap_target": "true"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "summary" in data
        summary = data["summary"]
        assert "duplicate_removal" in summary
        assert "outlier_handling" in summary
        # The value column should appear in outlier_handling since we opted to cap it
        assert summary["outlier_handling"]["columns_processed"] is not None

    def test_invalid_target_column_returns_400(self, mock_llm_explanation):
        """An invalid target_column name returns a 400 error."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buf = _make_csv_buffer(df)

        response = client.post(
            "/clean-dataset",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": "nonexistent_column"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_invalid_file_type_returns_400(self, mock_llm_explanation):
        """A non-CSV file returns a 400 error."""
        buf = io.BytesIO(b"some,data\n1,2")

        response = client.post(
            "/clean-dataset",
            files={"file": ("data.json", buf, "application/json")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid file type" in data["detail"]

    def test_empty_csv_returns_400(self, mock_llm_explanation, tmp_path, monkeypatch):
        """A CSV with only headers and no data rows returns a 400 error."""
        df = pd.DataFrame({"a": pd.Series(dtype="int64"), "b": pd.Series(dtype="float64")})
        buf = _make_csv_buffer(df)

        response = client.post(
            "/clean-dataset",
            files={"file": ("empty.csv", buf, "text/csv")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "no data rows" in data["detail"].lower()

    def test_target_column_empty_string_treated_as_none(self, mock_llm_explanation, tmp_path, monkeypatch):
        """An empty-string target_column is treated as None and should succeed."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buf = _make_csv_buffer(df)

        response = client.post(
            "/clean-dataset",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": ""},
        )

        assert response.status_code == 200


# ── /run-eda endpoint tests ────────────────────────────────────────────────────


@pytest.fixture
def changelog_fixture_path(tmp_path) -> str:
    """Write a simple cleaning changelog to a temp file and return its path.

    Uses ``pclass`` (low-cardinality numeric, not ID-like) so the outlier
    column is *not* also excluded by ``identify_id_columns()``.
    """
    changelog = {
        "outlier_handling": {
            "columns_processed": [{"column": "pclass"}],
            "columns_skipped_low_cardinality": [],
            "columns_skipped_target_protected": [],
        },
    }
    path = os.path.join(tmp_path, "changelog.json")
    with open(path, "w") as f:
        json.dump(changelog, f)
    return path


@pytest.fixture
def mock_eda_insights():
    """Patch generate_eda_insights so endpoint tests don't need a real LLM key."""
    with unittest.mock.patch(
        "agents.eda_agent.generate_eda_insights",
        return_value={"insights": ["Mocked insight."], "summary": "Mocked summary."},
    ):
        yield


class TestRunEdaEndpoint:
    """Tests for the POST /run-eda endpoint."""

    def test_valid_csv_no_target(self, mock_eda_insights, tmp_path, monkeypatch):
        """Valid CSV with no target => 200, target_distribution is null."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "age": [22, 38, 26],
            "fare": [7.25, 71.28, 8.05],
            "survived": [0, 1, 0],
        })
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("test.csv", buf, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["target_distribution"] is None
        assert data["stats"] is not None
        assert data["histograms"] is not None
        assert data["boxplots"] is not None
        assert data["insights"] is not None
        assert "artifact_path" in data
        assert "excluded_columns" in data

    def test_valid_csv_with_target(self, mock_eda_insights, tmp_path, monkeypatch):
        """Valid CSV with target_column and problem_type => 200, target_distribution populated."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "age": [22, 38, 26],
            "fare": [7.25, 71.28, 8.05],
            "survived": [0, 1, 0],
        })
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": "survived", "problem_type": "classification"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["target_distribution"] is not None
        assert "survived" in data["target_distribution"]

    def test_with_cleaning_changelog_path(self, mock_eda_insights, changelog_fixture_path, tmp_path, monkeypatch):
        """cleaning_changelog_path points to a real file => loaded and used."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "age": [22, 38, 26, 35, 28],
            "fare": [7.25, 71.28, 8.05, 53.10, 15.50],
            "pclass": [3, 1, 3, 1, 3],
            "name": ["A", "B", "C", "D", "E"],
        })
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"cleaning_changelog_path": changelog_fixture_path},
        )

        assert response.status_code == 200
        data = response.json()
        # Changelog specifies pclass; age/fare are ID-like (100% unique in 5 rows) → excluded.
        # pclass is low-cardinality → NOT excluded → should appear.
        box_keys = set(data["boxplots"]["boxplots"].keys())
        assert "pclass" in box_keys
        assert "age" not in box_keys  # ID-like
        assert "fare" not in box_keys  # ID-like

    def test_invalid_changelog_path_falls_back(self, mock_eda_insights, tmp_path, monkeypatch):
        """Nonexistent cleaning_changelog_path => fallback, not a 400."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "age": [22, 38, 26],
            "fare": [7.25, 71.28, 8.05],
            "pclass": [1, 1, 3],  # 2 uniques in 3 rows → 66% < 95% → not ID-like
        })
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"cleaning_changelog_path": "/nonexistent/path.json"},
        )

        assert response.status_code == 200
        data = response.json()
        # Invalid path → fallback to None. age/fare are ID-like (100% unique in 3 rows),
        # pclass is low-cardinality → only pclass should appear.
        box_keys = set(data["boxplots"]["boxplots"].keys())
        assert "pclass" in box_keys
        assert "age" not in box_keys
        assert "fare" not in box_keys

    def test_invalid_file_type_returns_400(self, tmp_path, monkeypatch):
        """Non-CSV file => 400."""
        monkeypatch.chdir(tmp_path)
        buf = io.BytesIO(b"some,data\n1,2")

        response = client.post(
            "/run-eda",
            files={"file": ("data.json", buf, "application/json")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Invalid file type" in data["detail"]

    def test_empty_csv_returns_400(self, tmp_path, monkeypatch):
        """Empty CSV => 400."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"a": pd.Series(dtype="int64")})
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("empty.csv", buf, "text/csv")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "no data rows" in data["detail"].lower()

    def test_invalid_target_column_returns_400(self, mock_eda_insights, tmp_path, monkeypatch):
        """Nonexistent target column => 400."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": "nonexistent_col"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_empty_string_target_treated_as_none(self, mock_eda_insights, tmp_path, monkeypatch):
        """Empty-string target_column => treated as None, endpoint succeeds."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": ""},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["target_distribution"] is None

    def test_response_validates_against_edaresponse(self, mock_eda_insights, tmp_path, monkeypatch):
        """Response JSON round-trips through the EDAResponse model without error."""
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "age": [22, 38, 26],
            "fare": [7.25, 71.28, 8.05],
            "survived": [0, 1, 0],
        })
        buf = _make_csv_buffer(df)

        response = client.post(
            "/run-eda",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": "survived", "problem_type": "classification"},
        )

        assert response.status_code == 200
        # Should raise no ValidationError — this double-checks the model wiring
        parsed = EDAResponse(**response.json())
        assert parsed.stats is not None
        assert parsed.target_distribution is not None
        assert parsed.insights is not None
        # age and fare are 100% unique in a 3-row dataset → flagged as ID-like
        assert "age" in parsed.excluded_columns
        assert "fare" in parsed.excluded_columns
        assert "survived" not in parsed.excluded_columns
