from __future__ import annotations

import os
import unittest.mock

import pandas as pd
import pytest

from agents.dataset_understanding import (
    classify_problem_type,
    detect_target_candidates,
    identify_id_columns,
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
        assert candidates[0]["confidence_score"] == 70

    def test_id_column_penalized(self, id_and_target_df):
        candidates = detect_target_candidates(id_and_target_df)

        assert len(candidates) == 1
        assert candidates[0]["column_name"] == "label"
        assert candidates[0]["confidence_score"] == 70

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


# identify_id_columns tests 

class TestIdentifyIdColumns:
    def test_obvious_id_column_detected(self):
        """A sequential unique column is flagged as ID-like."""
        df = pd.DataFrame({
            "passenger_id": range(100),
            "age": [25] * 100,
            "name": [f"person_{i % 5}" for i in range(100)],
        })
        result = identify_id_columns(df)
        assert result == ["passenger_id"]

    def test_no_id_like_columns(self):
        """No column meets the threshold → empty list."""
        df = pd.DataFrame({
            "group": ["A", "B", "C", "D", "A", "B", "C", "D"] * 10,
            "label": [0, 1] * 40,
        })
        result = identify_id_columns(df)
        assert result == []

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame()
        result = identify_id_columns(df)
        assert result == []

    def test_column_order_preserved(self):
        """Result should maintain df.columns order."""
        df = pd.DataFrame({
            "id_a": range(50),
            "normal": [1, 2] * 25,
            "id_b": range(50, 100),
        })
        result = identify_id_columns(df)
        assert result == ["id_a", "id_b"]

    def test_single_row_dataframe_all_columns_flagged(self):
        """Single row → every column has 1 unique == 1 row, so all qualify."""
        df = pd.DataFrame({
            "a": [42],
            "b": ["x"],
        })
        result = identify_id_columns(df)
        assert result == ["a", "b"]

    def test_custom_threshold(self):
        """A lower threshold catches more columns."""
        df = pd.DataFrame({
            "high_card": range(100),
            "med_card": [i % 50 for i in range(100)],
            "low_card": [0] * 100,
        })
        # threshold=0.4: high_card (100/100=1.0) and med_card (50/100=0.5) qualify
        result = identify_id_columns(df, threshold=0.4)
        assert result == ["high_card", "med_card"]

    def test_regression_detect_target_candidates_unchanged(self, id_and_target_df):
        """Refactored detect_target_candidates() produces identical output.

        This pins the exact expected output for a known input to ensure the
        refactor (extracting ID column logic into identify_id_columns) does
        not change behavior in any way.
        """
        candidates = detect_target_candidates(id_and_target_df, return_all=True)

        assert len(candidates) == 2

        # id column: penalised to score 0
        assert candidates[0]["column_name"] == "label"
        assert candidates[0]["confidence_score"] == 70
        assert candidates[0]["reasons"] == [
            "Column name matches common target naming pattern",
            "Column contains boolean-like values, a strong binary-target signal",
            "Column is the last column in the dataset (weak target convention signal)",
        ]

        # id column: penalised to score 0
        assert candidates[1]["column_name"] == "id"
        assert candidates[1]["confidence_score"] == 0
        assert candidates[1]["reasons"] == [
            "Excluded: column has near-unique values per row, "
            "likely an ID column, not a target",
        ]

        # Also confirm the default (non-return_all) output is unchanged
        top = detect_target_candidates(id_and_target_df)
        assert len(top) == 1
        assert top[0]["column_name"] == "label"
        assert top[0]["confidence_score"] == 70


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


class TestComputeConfidenceScore:
    """Test suite for deterministic confidence scoring in Dataset Understanding (Week 6 Part 1)."""

    @pytest.fixture
    def titanic_data(self) -> pd.DataFrame:
        n = 100
        return pd.DataFrame(
            {
                "PassengerId": range(1, n + 1),
                "Name": [f"Passenger_{i}" for i in range(n)],
                "Age": [22.0 + (i % 40) for i in range(n)],
                "Fare": [7.25 + (i % 50) for i in range(n)],
                "Sex": ["male", "female"] * (n // 2),
                "Embarked": (["S", "C", "Q"] * (n // 3 + 1))[:n],
                "Survived": [0, 1] * (n // 2),
            }
        )

    def test_full_marks_titanic(self, titanic_data):
        """1. Full-marks test on Titanic fixture: domain keyword match (+30), boolean-like (+30),
        reasonable class balance (+20), and LLM agreement (+20) -> total_score == 100."""
        from agents.dataset_understanding import (
            compute_confidence_score,
            detect_target_candidates,
            determine_heuristic_problem_type,
        )

        candidates = detect_target_candidates(titanic_data, return_all=True)
        heuristic_type = determine_heuristic_problem_type(titanic_data, "Survived")

        result = compute_confidence_score(
            selected_target="Survived",
            target_candidates=candidates,
            df=titanic_data,
            llm_problem_type="classification",
            heuristic_problem_type=heuristic_type,
        )

        assert result["total_score"] == 100
        assert len(result["breakdown"]) == 4

        breakdown_dict = {b["check"]: b for b in result["breakdown"]}

        # 1. target_name_match
        assert breakdown_dict["target_name_match"]["points_awarded"] == 30
        assert "Survived" in breakdown_dict["target_name_match"]["reason"]

        # 2. cardinality_signal_strength
        assert breakdown_dict["cardinality_signal_strength"]["points_awarded"] == 30
        assert "Survived" in breakdown_dict["cardinality_signal_strength"]["reason"]

        # 3. class_balance_reasonable
        assert breakdown_dict["class_balance_reasonable"]["points_awarded"] == 20
        assert "50.0%" in breakdown_dict["class_balance_reasonable"]["reason"]

        # 4. llm_heuristic_agreement
        assert breakdown_dict["llm_heuristic_agreement"]["points_awarded"] == 20
        assert "agree" in breakdown_dict["llm_heuristic_agreement"]["reason"]

    def test_partial_marks_unmatched_target_name(self, titanic_data):
        """2. Partial-marks test: generic column name without keyword match scores 70/100."""
        from agents.dataset_understanding import (
            compute_confidence_score,
            detect_target_candidates,
            determine_heuristic_problem_type,
        )

        df = titanic_data.rename(columns={"Survived": "col_target_actual"})
        candidates = detect_target_candidates(df, return_all=True)
        heuristic_type = determine_heuristic_problem_type(df, "col_target_actual")

        result = compute_confidence_score(
            selected_target="col_target_actual",
            target_candidates=candidates,
            df=df,
            llm_problem_type="classification",
            heuristic_problem_type=heuristic_type,
        )

        assert result["total_score"] == 70
        breakdown_dict = {b["check"]: b for b in result["breakdown"]}
        assert breakdown_dict["target_name_match"]["points_awarded"] == 0
        assert "No target naming" in breakdown_dict["target_name_match"]["reason"]
        assert breakdown_dict["cardinality_signal_strength"]["points_awarded"] == 30
        assert breakdown_dict["class_balance_reasonable"]["points_awarded"] == 20
        assert breakdown_dict["llm_heuristic_agreement"]["points_awarded"] == 20

    def test_class_imbalance(self, titanic_data):
        """3. Class imbalance test: 95/5 distribution scores 0 on class_balance_reasonable."""
        from agents.dataset_understanding import (
            compute_confidence_score,
            detect_target_candidates,
            determine_heuristic_problem_type,
        )

        df = titanic_data.copy()
        df["Survived"] = [1] * 95 + [0] * 5  # 95% class 1
        candidates = detect_target_candidates(df, return_all=True)
        heuristic_type = determine_heuristic_problem_type(df, "Survived")

        result = compute_confidence_score(
            selected_target="Survived",
            target_candidates=candidates,
            df=df,
            llm_problem_type="classification",
            heuristic_problem_type=heuristic_type,
        )

        assert result["total_score"] == 80
        breakdown_dict = {b["check"]: b for b in result["breakdown"]}
        assert breakdown_dict["class_balance_reasonable"]["points_awarded"] == 0
        assert "95.0%" in breakdown_dict["class_balance_reasonable"]["reason"]

    def test_llm_heuristic_disagreement(self, titanic_data):
        """4. LLM-heuristic disagreement test: LLM=regression, heuristic=classification scores 0."""
        from agents.dataset_understanding import (
            compute_confidence_score,
            detect_target_candidates,
            determine_heuristic_problem_type,
        )

        candidates = detect_target_candidates(titanic_data, return_all=True)
        heuristic_type = determine_heuristic_problem_type(titanic_data, "Survived")

        result = compute_confidence_score(
            selected_target="Survived",
            target_candidates=candidates,
            df=titanic_data,
            llm_problem_type="regression",
            heuristic_problem_type=heuristic_type,
        )

        assert result["total_score"] == 80
        breakdown_dict = {b["check"]: b for b in result["breakdown"]}
        assert breakdown_dict["llm_heuristic_agreement"]["points_awarded"] == 0
        assert "disagree" in breakdown_dict["llm_heuristic_agreement"]["reason"]
        assert "regression" in breakdown_dict["llm_heuristic_agreement"]["reason"]
        assert "classification" in breakdown_dict["llm_heuristic_agreement"]["reason"]

    def test_determinism(self, titanic_data):
        """5. Determinism test: calling compute_confidence_score twice gives byte-for-byte identical output."""
        from agents.dataset_understanding import (
            compute_confidence_score,
            detect_target_candidates,
            determine_heuristic_problem_type,
        )

        candidates = detect_target_candidates(titanic_data, return_all=True)
        heuristic_type = determine_heuristic_problem_type(titanic_data, "Survived")

        res1 = compute_confidence_score("Survived", candidates, titanic_data, "classification", heuristic_type)
        res2 = compute_confidence_score("Survived", candidates, titanic_data, "classification", heuristic_type)

        assert res1 == res2
        import json
        assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)

    def test_regression_target_student_performance(self):
        """6. Regression target test: Student Performance Factors (Exam_Score).
        Assert class balance awards +20 with 'N/A for regression' and regression signal check passes."""
        from agents.dataset_understanding import (
            compute_confidence_score,
            detect_target_candidates,
            determine_heuristic_problem_type,
        )

        n = 100
        df = pd.DataFrame(
            {
                "Hours_Studied": [i % 24 for i in range(n)],
                "Attendance": [50 + (i % 50) for i in range(n)],
                "Exam_Score": [30.0 + (i % 70) for i in range(n)],
            }
        )

        candidates = detect_target_candidates(df, return_all=True)
        heuristic_type = determine_heuristic_problem_type(df, "Exam_Score")
        assert heuristic_type == "regression"

        result = compute_confidence_score(
            selected_target="Exam_Score",
            target_candidates=candidates,
            df=df,
            llm_problem_type="regression",
            heuristic_problem_type=heuristic_type,
        )

        breakdown_dict = {b["check"]: b for b in result["breakdown"]}
        assert breakdown_dict["class_balance_reasonable"]["points_awarded"] == 20
        assert "N/A for regression" in breakdown_dict["class_balance_reasonable"]["reason"]
        assert breakdown_dict["cardinality_signal_strength"]["points_awarded"] == 30
        assert breakdown_dict["llm_heuristic_agreement"]["points_awarded"] == 20

    def test_artifact_integration_includes_confidence_score(self, titanic_data, tmp_path):
        """7. Artifact integration test: dataset_profile_v1.json includes confidence_score structure
        without modifying existing profile fields."""
        import json
        from agents.pipeline_agents import DatasetUnderstandingAgent

        artifacts_dir = str(tmp_path / "artifacts")
        state = {
            "df": titanic_data,
            "artifacts_dir": artifacts_dir,
        }

        result_state = DatasetUnderstandingAgent().run(state)

        assert "confidence_score" in result_state
        cs = result_state["confidence_score"]
        assert "total_score" in cs
        assert "breakdown" in cs
        assert len(cs["breakdown"]) == 4

        artifact_file = result_state["dataset_profile_artifact_path"]
        with open(artifact_file, "r") as f:
            artifact_data = json.load(f)

        assert "confidence_score" in artifact_data
        assert "profile" in artifact_data
        assert "target_candidates" in artifact_data
        assert "selected_target" in artifact_data
        assert "problem_type_analysis" in artifact_data

