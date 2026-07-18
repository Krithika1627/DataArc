from __future__ import annotations

import os
import unittest.mock

import pandas as pd
import pytest

from agents.dataset_understanding import (
    classify_problem_type,
    detect_target_candidates,
    profile_dataset,
)

# Profile dataset tests 

class TestProfileDataset:
    def test_profile_basic_stats(self, sample_profile_data):
        result = profile_dataset(sample_profile_data)

        assert result["row_count"] == 6
        assert result["column_count"] == 4
        assert result["duplicate_row_count"] == 1
        assert result["duplicate_row_percentage"] == round(100 / 6, 2)

    def test_profile_age_column(self, sample_profile_data):
        result = profile_dataset(sample_profile_data)

        age_col = [c for c in result["columns"] if c["name"] == "age"][0]
        assert age_col["missing_count"] == 1
        assert age_col["missing_percentage"] == round(100 / 6, 2)
        assert age_col["unique_count"] == 4

    def test_profile_name_column(self, sample_profile_data):
        result = profile_dataset(sample_profile_data)

        name_col = [c for c in result["columns"] if c["name"] == "name"][0]
        assert name_col["missing_count"] == 1
        assert name_col["unique_count"] == 4

    def test_profile_salary_column(self, sample_profile_data):
        result = profile_dataset(sample_profile_data)

        salary_col = [c for c in result["columns"] if c["name"] == "salary"][0]
        assert salary_col["missing_count"] == 1
        assert salary_col["unique_count"] == 4

    def test_profile_department_column(self, sample_profile_data):
        result = profile_dataset(sample_profile_data)

        dept_col = [c for c in result["columns"] if c["name"] == "department"][0]
        assert dept_col["missing_count"] == 0
        assert dept_col["unique_count"] == 3

    def test_profile_empty_dataframe(self):
        df = pd.DataFrame()
        result = profile_dataset(df)
        assert result["row_count"] == 0
        assert result["column_count"] == 0
        assert result["columns"] == []
        assert result["duplicate_row_count"] == 0
        assert result["duplicate_row_percentage"] == 0.0

# Target detection tests 

class TestDetectTargetCandidates:
    def test_obvious_target_column(self, obvious_target_df):
        candidates = detect_target_candidates(obvious_target_df)

        assert len(candidates) == 1
        assert candidates[0]["column_name"] == "target"
        assert candidates[0]["confidence_score"] == 85

    def test_id_column_penalized(self, id_and_target_df):
        candidates = detect_target_candidates(id_and_target_df)

        assert len(candidates) == 1
        assert candidates[0]["column_name"] == "label"
        assert candidates[0]["confidence_score"] == 85

    def test_no_clear_target(self, no_target_df):
        candidates = detect_target_candidates(no_target_df)

        assert len(candidates) == 0

    def test_regression_target_wins(self, regression_target_df):
        candidates = detect_target_candidates(regression_target_df)

        assert len(candidates) == 2
        assert candidates[0]["column_name"] == "exam_score"
        assert candidates[0]["confidence_score"] == 55

    def test_returns_all_when_flag_set(self, no_target_df):
        candidates = detect_target_candidates(no_target_df, return_all=True)

        assert len(candidates) == 3
        assert all("column_name" in c for c in candidates)
        assert all("confidence_score" in c for c in candidates)
        for i in range(len(candidates) - 1):
            assert candidates[i]["confidence_score"] >= candidates[i + 1]["confidence_score"]

    def test_id_column_penalized_in_return_all(self, id_and_target_df):
        candidates = detect_target_candidates(id_and_target_df, return_all=True)

        assert len(candidates) == 2
        id_candidate = [c for c in candidates if c["column_name"] == "id"][0]
        label_candidate = [c for c in candidates if c["column_name"] == "label"][0]

        assert "ID" in " ".join(id_candidate["reasons"])
        assert label_candidate["confidence_score"] > id_candidate["confidence_score"]


# Problem type classification tests 

class TestClassifyProblemType:
    def test_classify_structure(self, obvious_target_df, monkeypatch):
        """Classify problem type with mocked Gemini API call (no network needed)."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-mocking")

        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"problem_type": "classification", '
            '"confidence_reasoning": "test", '
            '"project_plan": "test plan"}'
        )

        with unittest.mock.patch(
            "google.genai.Client"
        ) as mock_client_class:
            mock_client = unittest.mock.MagicMock()
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            profile = profile_dataset(obvious_target_df)
            candidates = detect_target_candidates(obvious_target_df)
            result = classify_problem_type(profile, candidates)

            for key in ("problem_type", "confidence_reasoning", "project_plan"):
                assert key in result

            assert result["problem_type"] == "classification"

    @pytest.mark.llm
    def test_classify_live_api(self, obvious_target_df):
        """Requires a valid GEMINI_API_KEY environment variable."""
        profile = profile_dataset(obvious_target_df)
        candidates = detect_target_candidates(obvious_target_df)
        result = classify_problem_type(profile, candidates)

        assert result["problem_type"] in (
            "classification", "regression", "clustering", "unclear"
        )

    def test_classify_garbage_api_key(self, obvious_target_df):
        profile = profile_dataset(obvious_target_df)
        candidates = detect_target_candidates(obvious_target_df)

        real_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "INVALID_KEY_THAT_WILL_FAIL"

        try:
            result = classify_problem_type(profile, candidates)
        finally:
            if real_key:
                os.environ["GEMINI_API_KEY"] = real_key
            else:
                del os.environ["GEMINI_API_KEY"]

        assert result["problem_type"] == "unclear"
        assert "LLM classification unavailable" in result["confidence_reasoning"]
        assert "Unable to generate" in result["project_plan"]

    def test_classify_no_key_fallback(self, no_target_df, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "")

        profile = profile_dataset(no_target_df)
        candidates = detect_target_candidates(no_target_df)
        result = classify_problem_type(profile, candidates)

        assert result["problem_type"] == "unclear"
        assert "GEMINI_API_KEY is not set" in result["confidence_reasoning"]
        assert "Unable to generate" in result["project_plan"]
