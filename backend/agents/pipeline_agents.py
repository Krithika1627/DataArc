from __future__ import annotations

import json
import os

import pandas as pd

from base_agent import BaseAgent
from dataset_understanding import (
    classify_problem_type,
    detect_target_candidates,
    profile_dataset,
)
from agents.data_cleaning import clean_dataset
from agents.eda_agent import run_eda
from agents.feature_engineering_agent import build_feature_pipeline


class DatasetUnderstandingAgent(BaseAgent):
    """Wraps Week 1 agents: profile, target detection, problem type classification."""

    def run(self, state: dict) -> dict:
        df: pd.DataFrame = state["df"]

        profile = profile_dataset(df)

        target_candidates = detect_target_candidates(df)

        if state.get("user_selected_target"):
            selected_target = state["user_selected_target"]
            target_source = "user_override"
        elif target_candidates:
            selected_target = target_candidates[0]["column_name"]
            target_source = "auto_detected"
        else:
            selected_target = None
            target_source = "none_found"

        if selected_target is not None:
            problem_type_analysis = classify_problem_type(profile, target_candidates)
        else:
            problem_type_analysis = {
                "problem_type": "unclear",
                "confidence_reasoning": "No target candidate was detected.",
                "project_plan": "Cannot generate a project plan without a target.",
            }

        state["profile"] = profile
        state["target_candidates"] = target_candidates
        state["selected_target"] = selected_target
        state["target_source"] = target_source
        state["problem_type_analysis"] = problem_type_analysis

        return state


class CleaningAgent(BaseAgent):
    """Wraps Week 2 agent: data cleaning (dedup, dtype fix, imputation, outliers)."""

    def run(self, state: dict) -> dict:
        df: pd.DataFrame = state["df"]
        target_column: str | None = state.get("selected_target")
        artifacts_dir: str = state.get("artifacts_dir", "artifacts")

        result = clean_dataset(
            df,
            target_column=target_column,
            cap_target=False,
            artifacts_dir=artifacts_dir,
        )

        cleaned_df = pd.read_csv(result["artifact_path"])

        state["cleaned_df"] = cleaned_df
        state["cleaning_artifact_path"] = result["artifact_path"]
        state["cleaning_changelog_path"] = result["changelog_path"]
        state["cleaning_summary"] = result["summary"]

        return state


class EDAAgent(BaseAgent):
    """Wraps Week 3 agent: exploratory data analysis."""

    def run(self, state: dict) -> dict:
        cleaned_df: pd.DataFrame = state["cleaned_df"]
        target_column: str | None = state.get("selected_target")
        problem_type: str | None = state.get("problem_type_analysis", {}).get("problem_type")
        artifacts_dir: str = state.get("artifacts_dir", "artifacts")

        cleaning_changelog = None
        changelog_path = state.get("cleaning_changelog_path")
        if changelog_path and os.path.exists(changelog_path):
            with open(changelog_path, "r") as f:
                cleaning_changelog = json.load(f)

        eda_result = run_eda(
            cleaned_df,
            target_column=target_column,
            problem_type=problem_type,
            artifacts_dir=artifacts_dir,
            cleaning_changelog=cleaning_changelog,
        )

        state["eda_result"] = eda_result

        return state


class FeatureEngineeringAgent(BaseAgent):
    """Wraps Week 4 agent: feature pipeline construction."""

    def run(self, state: dict) -> dict:
        cleaned_df: pd.DataFrame = state["cleaned_df"]
        target_column: str = state["selected_target"]
        problem_type: str = state["problem_type_analysis"]["problem_type"]
        ordinal_columns: dict[str, list[str]] | None = state.get("ordinal_columns")

        fe_result = build_feature_pipeline(
            cleaned_df,
            target_column=target_column,
            problem_type=problem_type,
            ordinal_columns=ordinal_columns,
        )

        state["feature_engineering_result"] = fe_result

        return state
