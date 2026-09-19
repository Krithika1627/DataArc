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


def identify_id_columns(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    row_count = len(df)
    if row_count == 0:
        return []

    id_columns = []
    for col in df.columns:
        unique_count = int(df[col].nunique())
        if unique_count >= threshold * row_count:
            id_columns.append(str(col))

    return id_columns


@with_agent_logging("target_detection")
def detect_target_candidates(df: pd.DataFrame, return_all: bool = False) -> list[dict]:
    target_keywords = {"target", "label", "class", "y", "outcome", "result"}
    domain_target_keywords = {
        "survived", "churn", "default", "approved", "fraud", "response",
        "converted", "purchased", "diagnosis", "readmitted", "attrition",
        "loan_status", "status", "success", "failure"
    }
    outcome_keywords = {
        "score", "price", "amount", "value", "rating", "revenue", "salary", "cost",
        "grade", "gpa", "mark", "marks"
    }
    boolean_like_sets = [
        {"0", "1"}, {"true", "false"}, {"yes", "no"}, {"y", "n"}
    ]

    row_count = len(df)
    col_count = len(df.columns)
    all_scores = []

    id_columns = identify_id_columns(df)

    for i, col in enumerate(df.columns):
        col_name = str(col)
        score = 0
        reasons = []

        normalized = col_name.lower().replace("_", "").replace(" ", "")

        if normalized in target_keywords:
            score += 40
            reasons.append("Column name matches common target naming pattern")
        elif any(kw in normalized for kw in domain_target_keywords):
            score += 35
            reasons.append("Column name matches a common domain-specific target pattern")

        unique_count = int(df[col].nunique())
        distinct_vals = set(
            str(v).strip().lower() for v in df[col].dropna().unique()
        )
        is_boolean_like = len(distinct_vals) == 2 and any(distinct_vals == b for b in boolean_like_sets)

        if is_boolean_like:
            score += 25
            reasons.append("Column contains boolean-like values, a strong binary-target signal")
        elif row_count >= 10 and 2 <= unique_count <= 10:
            score += 30
            reasons.append(
                f"Low cardinality ({unique_count} unique values) suggests a classification target"
            )

        is_last = i == col_count - 1
        is_numeric = str(df[col].dtype) in ("int64", "float64")
        high_cardinality = unique_count > 10

        if is_last and is_numeric and high_cardinality:
            score += 35
            reasons.append(
                "Last column, numeric, high cardinality -- strong regression target signal"
            )
        elif is_last:
            score += 5
            reasons.append("Column is the last column in the dataset (weak target convention signal)")

        if any(kw in col_name.lower() for kw in outcome_keywords):
            score += 20
            reasons.append("Column name contains a common outcome/target keyword")

        if col_name in id_columns:
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
                "_is_numeric": is_numeric,      
                "_unique_count": unique_count,  
            }
        )

    all_scores.sort(
        key=lambda c: (c["confidence_score"], c["_is_numeric"], c["_unique_count"]),
        reverse=True,
    )

    for c in all_scores:
        del c["_is_numeric"]
        del c["_unique_count"]

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


def determine_heuristic_problem_type(df: pd.DataFrame, target_column: str | None) -> str:
    """Determine problem type heuristically from dtype and cardinality of target column."""
    if not target_column or target_column not in df.columns:
        return "unclear"

    series = df[target_column].dropna()
    if len(series) == 0:
        return "unclear"

    unique_count = int(series.nunique())
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if unique_count == 2:
        return "classification"
    if not is_numeric:
        return "classification"
    if unique_count <= 10:
        return "classification"
    if is_numeric and unique_count > 10:
        return "regression"

    return "classification"


def compute_confidence_score(
    selected_target: str,
    target_candidates: list[dict],
    df: pd.DataFrame,
    llm_problem_type: str,
    heuristic_problem_type: str,
) -> dict:
    """Compute a deterministic confidence score (0-100) and breakdown for selected target and problem type."""
    cand = next(
        (c for c in target_candidates if c.get("column_name") == selected_target),
        None,
    )
    reasons = cand.get("reasons", []) if cand else []

    # 1. target_name_match (+30/+0)
    has_name_match = any(
        "target naming pattern" in r.lower() or "domain-specific target pattern" in r.lower()
        for r in reasons
    )
    if has_name_match:
        p_name = 30
        r_name = f"Column '{selected_target}' matched target name or domain keyword pattern"
    else:
        p_name = 0
        r_name = f"No target naming pattern matched for '{selected_target}'"

    # 2. cardinality_signal_strength (+30/+0)
    is_regression = heuristic_problem_type.strip().lower() == "regression"
    if is_regression:
        has_reg_signal = any(
            "strong regression target signal" in r.lower()
            or ("regression" in r.lower() and "strong" in r.lower())
            for r in reasons
        )
        if not has_reg_signal and selected_target in df.columns:
            s = df[selected_target].dropna()
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
                has_reg_signal = True

        if has_reg_signal:
            p_card = 30
            r_card = f"Strong numeric high-cardinality signal for regression on '{selected_target}'"
        else:
            p_card = 0
            r_card = f"Weak or absent regression signal on '{selected_target}'"
    else:
        has_cls_signal = any(
            "boolean-like values" in r.lower()
            or "suggests a classification target" in r.lower()
            for r in reasons
        )
        if has_cls_signal:
            p_card = 30
            r_card = f"Strong cardinality signal for classification on '{selected_target}'"
        else:
            p_card = 0
            r_card = f"Weak or absent cardinality signal for classification on '{selected_target}'"

    # 3. class_balance_reasonable (+20/+0)
    if is_regression:
        p_bal = 20
        r_bal = "N/A for regression - check not applicable"
    else:
        if not selected_target or selected_target not in df.columns:
            p_bal = 0
            r_bal = "Target column not found in dataframe"
        elif len(df) < 10:
            p_bal = 0
            r_bal = f"Dataset row count ({len(df)} rows) is too small (<10 rows) to judge class balance meaningfully"
        else:
            vc = df[selected_target].dropna().value_counts(normalize=True)
            if len(vc) == 0:
                p_bal = 0
                r_bal = "Target column has no non-null values"
            else:
                max_prop = float(vc.max())
                if max_prop <= 0.90:
                    p_bal = 20
                    r_bal = f"Reasonable class balance (dominant class represents {round(max_prop * 100, 1)}% of rows <= 90%)"
                else:
                    p_bal = 0
                    r_bal = f"Severe class imbalance: dominant class represents {round(max_prop * 100, 1)}% of rows (> 90%)"

    # 4. llm_heuristic_agreement (+20/+0)
    llm_clean = str(llm_problem_type).strip().lower()
    heur_clean = str(heuristic_problem_type).strip().lower()
    if llm_clean == heur_clean and llm_clean in ("classification", "regression"):
        p_agree = 20
        r_agree = f"LLM said {llm_clean}, heuristic said {heur_clean} - agree"
    else:
        p_agree = 0
        r_agree = f"LLM said {llm_clean}, heuristic said {heur_clean} - disagree"

    total_score = p_name + p_card + p_bal + p_agree

    return {
        "total_score": total_score,
        "breakdown": [
            {
                "check": "target_name_match",
                "points_awarded": p_name,
                "reason": r_name,
            },
            {
                "check": "cardinality_signal_strength",
                "points_awarded": p_card,
                "reason": r_card,
            },
            {
                "check": "class_balance_reasonable",
                "points_awarded": p_bal,
                "reason": r_bal,
            },
            {
                "check": "llm_heuristic_agreement",
                "points_awarded": p_agree,
                "reason": r_agree,
            },
        ],
    }


