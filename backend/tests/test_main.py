from __future__ import annotations

import io
import json
import os
import unittest.mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app


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

    def test_valid_csv_no_target(self, mock_llm_explanation):
        """Valid CSV with no target_column specified returns 200 with expected structure."""
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

    def test_valid_csv_with_target_and_cap(self, mock_llm_explanation):
        """Valid CSV with target_column and cap_target=True returns 200."""
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

    def test_empty_csv_returns_400(self, mock_llm_explanation):
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

    def test_target_column_empty_string_treated_as_none(self, mock_llm_explanation):
        """An empty-string target_column is treated as None and should succeed."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buf = _make_csv_buffer(df)

        response = client.post(
            "/clean-dataset",
            files={"file": ("test.csv", buf, "text/csv")},
            data={"target_column": ""},
        )

        assert response.status_code == 200
