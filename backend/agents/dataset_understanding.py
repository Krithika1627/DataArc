from __future__ import annotations

import os
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv

from logging_utils import with_agent_logging

@with_agent_logging("dataset_profiling")
def profile_dataset(df: pd.DataFrame) -> dict:
    if len(df.columns) == 0:
        return {
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "duplicate_row_count": 0,
            "duplicate_row_percentage": 0.0,
        }

    columns_info = []
    for col in df.columns:
        total = len(df)
        missing = int(df[col].isna().sum())
        columns_info.append(
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_percentage": round((missing / total) * 100, 2) if total > 0 else 0.0,
                "unique_count": int(df[col].nunique()),
            }
        )

    total_rows = len(df)
    duplicate_count = int(df.duplicated().sum())

    return {
        "row_count": total_rows,
        "column_count": len(df.columns),
        "columns": columns_info,
        "duplicate_row_count": duplicate_count,
        "duplicate_row_percentage": round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
    }


@with_agent_logging("target_detection")
def detect_target_candidates(df: pd.DataFrame, return_all: bool = False) -> list[dict]:
    target_keywords = {"target", "label", "class", "y", "outcome", "result"}
    outcome_keywords = {"score", "price", "amount", "value", "rating", "revenue", "salary", "cost"}
    row_count = len(df)
    col_count = len(df.columns)
    all_scores = []

    for i, col in enumerate(df.columns):
        col_name = str(col)
        score = 0
        reasons = []

        normalized = col_name.lower().replace("_", "").replace(" ", "")

        # Name match (+40)
        if normalized in target_keywords:
            score += 40
            reasons.append("Column name matches common target naming pattern")

        # Low-to-moderate cardinality (+30, classification signal)
        unique_count = int(df[col].nunique())
        if row_count >= 10 and 2 <= unique_count <= 10:
            score += 30
            reasons.append(
                f"Low cardinality ({unique_count} unique values) suggests a classification target"
            )

        # Combined last-column rule (mutually exclusive a / b)
        is_last = i == col_count - 1
        is_numeric = str(df[col].dtype) in ("int64", "float64")
        high_cardinality = unique_count > 10

        if is_last and is_numeric and high_cardinality:
            score += 35
            reasons.append(
                "Last column, numeric, high cardinality -- strong regression target signal"
            )
        elif is_last:
            score += 15
            reasons.append("Column is the last column in the dataset (common target convention)")

        # Outcome keyword bonus (+20)
        if any(kw in col_name.lower() for kw in outcome_keywords):
            score += 20
            reasons.append("Column name contains a common outcome/target keyword")

        # ID column penalty (-30)
        if unique_count >= 0.95 * row_count:
            score -= 30
            reasons.append(
                "Excluded: column has near-unique values per row, "
                "likely an ID column, not a target"
            )

        score = max(0, min(100, score))

        all_scores.append(
            {
                "column_name": col_name,
                "confidence_score": score,
                "reasons": reasons,
            }
        )

    all_scores.sort(key=lambda c: c["confidence_score"], reverse=True)

    if return_all:
        return all_scores

    return [c for c in all_scores if c["confidence_score"] > 0][:2]

class _ProblemTypeResponse(TypedDict):
    problem_type: str       # "classification" | "regression" | "clustering" | "unclear"
    confidence_reasoning: str
    project_plan: str

def _build_classify_prompt(profile: dict, target_candidates: list[dict]) -> str:
    lines = [
        "You are a data science expert analysing a dataset. "
        "Based on the profile below, classify the problem type and write a brief project plan.",
        "",
        "DATASET PROFILE",
        f"- Rows: {profile['row_count']}",
        f"- Columns: {profile['column_count']}",
    ]

    col_summary = ", ".join(
        f"{c['name']} ({c['dtype']})" for c in profile["columns"]
    )
    lines.append(f"- Columns: {col_summary}")

    lines.append("")
    lines.append("TARGET CANDIDATES")
    if not target_candidates:
        lines.append("- No clear target candidate was detected.")
    else:
        for cand in target_candidates:
            reasons_str = "; ".join(cand["reasons"])
            lines.append(
                f"- {cand['column_name']} (confidence: {cand['confidence_score']}/100)"
            )
            if reasons_str:
                lines.append(f"  Reasons: {reasons_str}")

    lines.append("")
    lines.append(
        'Classify the problem type as one of: "classification", "regression", '
        '"clustering", or "unclear".'
    )

    return "\n".join(lines)


@with_agent_logging("problem_type_classification")
def classify_problem_type(profile: dict, target_candidates: list[dict]) -> dict:
    fallback = {
        "problem_type": "unclear",
        "confidence_reasoning": "LLM classification unavailable: ...",
        "project_plan": (
            "Unable to generate a project plan automatically. "
            "Please review the dataset profile and target candidates manually."
        ),
    }

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        fallback["confidence_reasoning"] = (
            "LLM classification unavailable: GEMINI_API_KEY is not set. "
            "Create a .env file with GEMINI_API_KEY=your_key or export the variable."
        )
        return fallback

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
    except Exception as exc:
        fallback["confidence_reasoning"] = f"LLM classification unavailable: failed to configure Gemini SDK: {exc}"
        return fallback

    prompt = _build_classify_prompt(profile, target_candidates)

    try:
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                response_schema=_ProblemTypeResponse,
            ),
        )

        import json as json_module

        result = json_module.loads(response.text)

        required_keys = {"problem_type", "confidence_reasoning", "project_plan"}
        if not required_keys.issubset(result.keys()):
            raise ValueError(
                f"Response missing keys: {required_keys - result.keys()}"
            )

        valid_types = {"classification", "regression", "clustering", "unclear"}
        if result["problem_type"] not in valid_types:
            raise ValueError(
                f"Invalid problem_type '{result['problem_type']}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}."
            )

        return result

    except Exception as exc:
        fallback["confidence_reasoning"] = (
            f"LLM classification unavailable: {exc}"
        )
        return fallback

if __name__ == "__main__":
    import json

    print("PROFILE_DATASET TEST")

    sample_data = pd.DataFrame(
        {
            "age": [25, 30, 35, None, 40, 25],
            "name": ["Alice", "Bob", "Charlie", "Diana", None, "Alice"],
            "salary": [50000.0, 60000.0, 70000.0, None, 80000.0, 50000.0],
            "department": [
                "Engineering",
                "Marketing",
                "Engineering",
                "Engineering",
                "HR",
                "Engineering",
            ],
        }
    )

    result = profile_dataset(sample_data)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # sanity assertions
    assert result["row_count"] == 6
    assert result["column_count"] == 4
    assert result["duplicate_row_count"] == 1
    assert result["duplicate_row_percentage"] == round(100 / 6, 2)

    age_col = [c for c in result["columns"] if c["name"] == "age"][0]
    assert age_col["missing_count"] == 1
    assert age_col["missing_percentage"] == round(100 / 6, 2)
    assert age_col["unique_count"] == 4

    name_col = [c for c in result["columns"] if c["name"] == "name"][0]
    assert name_col["missing_count"] == 1
    assert name_col["unique_count"] == 4

    salary_col = [c for c in result["columns"] if c["name"] == "salary"][0]
    assert salary_col["missing_count"] == 1
    assert salary_col["unique_count"] == 4

    dept_col = [c for c in result["columns"] if c["name"] == "department"][0]
    assert dept_col["missing_count"] == 0
    assert dept_col["unique_count"] == 3

    print("\nAll profile_dataset sanity checks passed.\n")

    print("DETECT_TARGET_CANDIDATES TESTS")

    print("\nTest (a): Obvious target column")
    df_a = pd.DataFrame(
        {
            "feature_1": range(100),
            "feature_2": [x % 11 for x in range(100)],
            "target": [0, 1] * 50,
        }
    )
    candidates_a = detect_target_candidates(df_a)
    print(json.dumps(candidates_a, indent=2, ensure_ascii=False))
    assert len(candidates_a) == 1
    assert candidates_a[0]["column_name"] == "target"
    assert candidates_a[0]["confidence_score"] == 85
    print("PASSED")

    print("\nTest (b): ID column + real target (ID should be penalised)")
    df_b = pd.DataFrame(
        {
            "id": range(100),
            "label": [0, 1] * 50,
        }
    )
    candidates_b = detect_target_candidates(df_b)
    print(json.dumps(candidates_b, indent=2, ensure_ascii=False))
    assert len(candidates_b) == 1
    assert candidates_b[0]["column_name"] == "label"
    assert candidates_b[0]["confidence_score"] == 85
    print("PASSED")

    print("\nTest (c): No clear target (all continuous / near-unique)")
    df_c = pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": [x % 20 for x in range(100)],
            "feature_c": [f"val_{x}" for x in range(100)],
        }
    )
    candidates_c = detect_target_candidates(df_c)
    print(json.dumps(candidates_c, indent=2, ensure_ascii=False))
    assert len(candidates_c) == 0
    print("PASSED")

    print("\nTest (d): Regression target wins over low-cardinality categoricals")
    df_d = pd.DataFrame(
        {
            "category_1": [f"group_{i % 3}" for i in range(200)],
            "category_2": [f"type_{i % 4}" for i in range(200)],
            "category_3": [f"region_{i % 5}" for i in range(200)],
            "exam_score": [50 + (i % 45) * 0.5 for i in range(200)],
        }
    )
    candidates_d = detect_target_candidates(df_d)
    print("Top 2 (default mode):")
    print(json.dumps(candidates_d, indent=2, ensure_ascii=False))
    assert len(candidates_d) == 2
    assert candidates_d[0]["column_name"] == "exam_score"
    assert candidates_d[0]["confidence_score"] == 55
    print("PASSED")

    print("All detect_target_candidates tests PASSED.\n")

    print("CLASSIFY_PROBLEM_TYPE TESTS")
    print("\nTest 1: classify_problem_type() with sample data")
    profile_a = profile_dataset(df_a)
    classify_result = classify_problem_type(profile_a, candidates_a)
    print(json.dumps(classify_result, indent=2, ensure_ascii=False))

    for key in ("problem_type", "confidence_reasoning", "project_plan"):
        assert key in classify_result, f"Missing key: {key}"

    load_dotenv()
    if os.environ.get("GEMINI_API_KEY"):
        assert classify_result["problem_type"] in (
            "classification", "regression", "clustering", "unclear"
        ), f"Unexpected problem_type: {classify_result['problem_type']}"
        print("  Live API call completed successfully.")
    else:
        assert classify_result["problem_type"] == "unclear"
        print("  No API key set — fallback returned as expected.")
    print("  PASSED")

    print("\nTest 2: Simulated API failure (garbage API key)")
    real_key = os.environ.get("GEMINI_API_KEY", "")
    os.environ["GEMINI_API_KEY"] = "INVALID_KEY_THAT_WILL_FAIL"
    try:
        import google.generativeai as genai
        genai.configure(api_key="INVALID_KEY_THAT_WILL_FAIL")
    except Exception:
        pass  

    fail_result = classify_problem_type(profile_a, candidates_a)
    if real_key:
        os.environ["GEMINI_API_KEY"] = real_key
    else:
        del os.environ["GEMINI_API_KEY"]

    print(json.dumps(fail_result, indent=2, ensure_ascii=False))
    assert fail_result["problem_type"] == "unclear", (
        f"Expected 'unclear' fallback, got '{fail_result['problem_type']}'"
    )
    assert "LLM classification unavailable" in fail_result["confidence_reasoning"], (
        "Fallback should include 'LLM classification unavailable' message"
    )
    assert "Unable to generate" in fail_result["project_plan"]
    print("PASSED")

    profile_c = profile_dataset(df_c)
    classify_result = classify_problem_type(profile_c, candidates_c)
    print(json.dumps(classify_result, indent=2, ensure_ascii=False))
 
    print("ALL TESTS PASSED.")
    
