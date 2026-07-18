from __future__ import annotations

import os
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv

from agents.logging_utils import with_agent_logging

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
        from google import genai

        client = genai.Client(api_key=api_key)
    except Exception as exc:
        fallback["confidence_reasoning"] = f"LLM classification unavailable: failed to configure Gemini SDK: {exc}"
        return fallback

    prompt = _build_classify_prompt(profile, target_candidates)

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _ProblemTypeResponse,
            },
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

