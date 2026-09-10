from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

from dotenv import load_dotenv

try:
    from agents.base_agent import BaseAgent
    from agents.logging_utils import log_agent_run, with_agent_logging
    from agents.versioning_utils import (
        get_latest_version_path,
        get_next_version,
        save_artifact,
    )
except ImportError:
    from base_agent import BaseAgent
    from logging_utils import log_agent_run, with_agent_logging
    from versioning_utils import (
        get_latest_version_path,
        get_next_version,
        save_artifact,
    )


class CandidateModel(TypedDict):
    model_name: str
    reasoning: str


class MLPlan(TypedDict):
    problem_type: str  # "classification" | "regression"
    confirmation_reasoning: str
    recommended_metric: str
    metric_reasoning: str
    candidate_models: list[CandidateModel]


def _build_planning_summary(
    profile: dict, cleaning: dict, eda: dict, fe: dict
) -> dict[str, Any]:
    selected_target = (
        profile.get("selected_target")
        or fe.get("target_column")
        or profile.get("target_column")
    )
    problem_type_analysis = profile.get("problem_type_analysis") or {}
    problem_type = (
        problem_type_analysis.get("problem_type")
        or profile.get("problem_type")
        or fe.get("problem_type")
        or "unclear"
    )
    confidence_reasoning = problem_type_analysis.get("confidence_reasoning", "")

    profile_data = (
        profile.get("profile")
        if isinstance(profile.get("profile"), dict)
        else profile
    )
    row_count = profile_data.get("row_count")
    column_count = profile_data.get("column_count")
    columns_info = profile_data.get("columns", [])

    eda_stats = eda.get("stats") or {}
    class_balance = eda_stats.get("class_balance")
    skewness = eda_stats.get("skewness")
    flagged_correlations = eda_stats.get("flagged_correlations")

    eda_insights_obj = eda.get("insights") or {}
    insights_list = eda_insights_obj.get("insights")
    insights_summary = eda_insights_obj.get("summary")

    cleaning_llm = cleaning.get("llm_explanation") or cleaning.get("explanation") or {}
    cleaning_summary_text = (
        cleaning_llm.get("summary")
        if isinstance(cleaning_llm, dict)
        else str(cleaning_llm)
    )

    return {
        "target_column": str(selected_target) if selected_target is not None else None,
        "earlier_problem_type_guess": problem_type,
        "earlier_confidence_reasoning": confidence_reasoning,
        "row_count": row_count,
        "column_count": column_count,
        "columns_info": columns_info,
        "class_balance": class_balance,
        "skewness": skewness,
        "flagged_correlations": flagged_correlations,
        "eda_insights": insights_list,
        "eda_summary": insights_summary,
        "cleaning_summary": cleaning_summary_text,
        "final_columns": fe.get("columns"),
        "encoding_map": fe.get("encoding_map"),
        "excluded_columns": fe.get("excluded_columns"),
        "collinear_pairs": fe.get("collinear_pairs"),
    }


def load_planning_inputs(artifacts_dir: str = "artifacts") -> dict[str, Any]:
    profile_path = get_latest_version_path(artifacts_dir, "dataset_profile", "json")
    cleaning_path = get_latest_version_path(artifacts_dir, "cleaned", "json")  # the changelog
    eda_path = get_latest_version_path(artifacts_dir, "eda_bundle", "json")
    fe_path = get_latest_version_path(artifacts_dir, "feature_engineered", "json")

    with open(profile_path) as f:
        profile = json.load(f)
    with open(cleaning_path) as f:
        cleaning = json.load(f)
    with open(eda_path) as f:
        eda = json.load(f)
    with open(fe_path) as f:
        fe = json.load(f)

    return _build_planning_summary(profile, cleaning, eda, fe)


def extract_ml_planning_summary(state: dict) -> dict[str, Any]:
    if not state.get("profile") and not state.get("selected_target") and not state.get("df"):
        artifacts_dir = state.get("artifacts_dir", "artifacts")
        try:
            return load_planning_inputs(artifacts_dir)
        except FileNotFoundError:
            pass

    target_column = state.get("selected_target") or state.get("target_column")

    problem_type_analysis = state.get("problem_type_analysis") or {}
    earlier_problem_type = (
        problem_type_analysis.get("problem_type")
        or state.get("problem_type")
        or "unclear"
    )
    confidence_reasoning = problem_type_analysis.get("confidence_reasoning", "")

    profile = state.get("profile") or {}
    row_count = profile.get("row_count")
    column_count = profile.get("column_count")
    columns_info = profile.get("columns", [])

    if row_count is None:
        df = state.get("cleaned_df") if state.get("cleaned_df") is not None else state.get("df")
        if df is not None:
            row_count = len(df)
            column_count = len(df.columns)
            if not columns_info:
                columns_info = [
                    {"name": str(c), "dtype": str(df[c].dtype)}
                    for c in df.columns
                ]

    eda_result = state.get("eda_result") or {}
    eda_stats = eda_result.get("stats") or state.get("eda_stats") or {}
    class_balance = eda_stats.get("class_balance") or state.get("class_balance")

    fe_result = state.get("feature_engineering_result") or {}
    encoding_map = fe_result.get("encoding_map")
    excluded_columns = fe_result.get("excluded_columns", [])

    return {
        "target_column": str(target_column) if target_column is not None else None,
        "earlier_problem_type_guess": earlier_problem_type,
        "earlier_confidence_reasoning": confidence_reasoning,
        "row_count": row_count,
        "column_count": column_count,
        "columns_info": columns_info,
        "class_balance": class_balance,
        "encoding_map": encoding_map,
        "excluded_columns": excluded_columns,
    }


def _build_ml_planning_prompt(summary: dict) -> str:
    lines = [
        "You are an expert Machine Learning Engineer creating a dataset-specific ML training plan.",
        "Based on the dataset summary below, confirm the problem type, choose the single best evaluation metric, and recommend 4 to 6 candidate models.",
        "",
        "DATASET SUMMARY:",
        f"- Total rows: {summary.get('row_count', 'unknown')}",
        f"- Total columns: {summary.get('column_count', 'unknown')}",
        f"- Target column: {summary.get('target_column', 'unknown')}",
        f"- Earlier problem type guess (Week 1): {summary.get('earlier_problem_type_guess', 'unclear')}",
    ]
    if summary.get("earlier_confidence_reasoning"):
        lines.append(f"  (Earlier reasoning: {summary['earlier_confidence_reasoning']})")

    if summary.get("columns_info"):
        col_strs = []
        for c in summary["columns_info"]:
            if isinstance(c, dict):
                col_strs.append(f"{c.get('name')} ({c.get('dtype')})")
            else:
                col_strs.append(str(c))
        lines.append(f"- Columns: {', '.join(col_strs[:30])}")
        if len(col_strs) > 30:
            lines.append(f"  ... and {len(col_strs) - 30} more columns")

    if summary.get("class_balance"):
        lines.append(f"- Class balance: {summary['class_balance']}")

    if summary.get("cleaning_summary"):
        lines.append(f"- Data cleaning applied: {summary['cleaning_summary']}")

    if summary.get("eda_summary"):
        lines.append(f"- EDA summary: {summary['eda_summary']}")

    if summary.get("encoding_map"):
        lines.append(f"- Feature encoding applied: {summary['encoding_map']}")

    lines.extend([
        "",
        "INSTRUCTIONS:",
        "1. Confirm problem_type as either 'classification' or 'regression'. Cross-check against the earlier guess rather than blindly trusting either source.",
        "2. Provide 1-2 sentences of confirmation_reasoning explaining why this problem type is appropriate.",
        "3. Recommend ONE primary evaluation metric best suited to this specific dataset (e.g. F1, ROC-AUC, Balanced Accuracy for imbalanced classification; Accuracy for balanced; RMSE, MAE, R2 for regression) with 1-2 sentences of metric_reasoning.",
        "4. Recommend 4 to 6 candidate models appropriate for the confirmed problem type (e.g. Logistic Regression, Random Forest Classifier, XGBoost Classifier, LightGBM Classifier, Gradient Boosting Classifier for classification; Linear Regression, Ridge, Random Forest Regressor, XGBoost Regressor, LightGBM Regressor for regression), each with a one-line reasoning explaining why it fits this dataset.",
        "",
        "STRICT JSON OUTPUT ONLY. Respond with a valid JSON object matching this schema exactly:",
        "{",
        '  "problem_type": "classification" | "regression",',
        '  "confirmation_reasoning": "<1-2 sentences>",',
        '  "recommended_metric": "<metric name>",',
        '  "metric_reasoning": "<1-2 sentences>",',
        '  "candidate_models": [',
        '    {"model_name": "<name>", "reasoning": "<why this fits>"}',
        "  ]",
        "}",
    ])
    return "\n".join(lines)


def parse_and_validate_ml_plan(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except Exception as exc:
        raise ValueError(f"Malformed JSON in ML planning response: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid ML plan response: expected JSON object, got {type(data).__name__}"
        )

    required_keys = {
        "problem_type",
        "confirmation_reasoning",
        "recommended_metric",
        "metric_reasoning",
        "candidate_models",
    }
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"ML plan missing required keys: {sorted(missing)}")

    problem_type = data.get("problem_type")
    if not isinstance(problem_type, str) or problem_type.strip().lower() not in (
        "classification",
        "regression",
    ):
        raise ValueError(
            f"Invalid problem_type '{problem_type}'. Must be 'classification' or 'regression'."
        )
    data["problem_type"] = problem_type.strip().lower()

    if (
        not isinstance(data.get("confirmation_reasoning"), str)
        or not data["confirmation_reasoning"].strip()
    ):
        raise ValueError("confirmation_reasoning must be a non-empty string.")

    if (
        not isinstance(data.get("recommended_metric"), str)
        or not data["recommended_metric"].strip()
    ):
        raise ValueError("recommended_metric must be a non-empty string.")

    if (
        not isinstance(data.get("metric_reasoning"), str)
        or not data["metric_reasoning"].strip()
    ):
        raise ValueError("metric_reasoning must be a non-empty string.")

    candidate_models = data.get("candidate_models")
    if not isinstance(candidate_models, list) or len(candidate_models) == 0:
        raise ValueError("candidate_models must be a non-empty list.")

    for i, model in enumerate(candidate_models):
        if not isinstance(model, dict):
            raise ValueError(f"Candidate model at index {i} must be an object.")
        if (
            "model_name" not in model
            or not isinstance(model["model_name"], str)
            or not model["model_name"].strip()
        ):
            raise ValueError(
                f"Candidate model at index {i} missing valid non-empty 'model_name'."
            )
        if (
            "reasoning" not in model
            or not isinstance(model["reasoning"], str)
            or not model["reasoning"].strip()
        ):
            raise ValueError(
                f"Candidate model at index {i} missing valid non-empty 'reasoning'."
            )

    return data


def generate_ml_plan(summary: dict) -> dict:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Create a .env file with GEMINI_API_KEY=your_key or export the variable."
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to configure Gemini SDK: {exc}") from exc

    prompt = _build_ml_planning_prompt(summary)

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": MLPlan,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    raw_text = getattr(response, "text", "")
    if not raw_text:
        raise ValueError("Empty response received from Gemini LLM.")

    return parse_and_validate_ml_plan(raw_text)


class MLPlanningAgent(BaseAgent):
    """Generates a dataset-specific ML training plan via LLM analysis."""

    def run(self, state: dict) -> dict:
        start_time = time.perf_counter()
        summary = extract_ml_planning_summary(state)

        inputs_summary = {
            "target_column": summary.get("target_column"),
            "problem_type_guess": summary.get("earlier_problem_type_guess"),
            "row_count": summary.get("row_count"),
            "column_count": summary.get("column_count"),
        }

        try:
            plan = generate_ml_plan(summary)
            duration = time.perf_counter() - start_time
            log_agent_run(
                agent_name="ml_planning",
                inputs_summary=inputs_summary,
                outputs_summary={
                    "problem_type": plan.get("problem_type"),
                    "recommended_metric": plan.get("recommended_metric"),
                    "candidate_models_count": len(plan.get("candidate_models", [])),
                },
                duration_seconds=duration,
            )
            state["ml_plan"] = plan
            return state

        except Exception as exc:
            duration = time.perf_counter() - start_time
            log_agent_run(
                agent_name="ml_planning",
                inputs_summary=inputs_summary,
                outputs_summary={},
                duration_seconds=duration,
                error=str(exc),
            )
            raise
