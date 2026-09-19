from __future__ import annotations

import json
import os

import pandas as pd

try:
    from agents.base_agent import BaseAgent
    from agents.dataset_understanding import (
        classify_problem_type,
        detect_target_candidates,
        profile_dataset,
    )
    from agents.data_cleaning import clean_dataset
    from agents.eda_agent import run_eda
    from agents.feature_engineering_agent import build_feature_pipeline
    from agents.ml_planning_agent import MLPlanningAgent
    from agents.training_agent import TrainingAgent
except ImportError:
    from base_agent import BaseAgent
    from dataset_understanding import (
        classify_problem_type,
        detect_target_candidates,
        profile_dataset,
    )
    from data_cleaning import clean_dataset
    from eda_agent import run_eda
    from feature_engineering_agent import build_feature_pipeline
    from ml_planning_agent import MLPlanningAgent
    from training_agent import TrainingAgent


class DatasetUnderstandingAgent(BaseAgent):
    def run(self, state: dict) -> dict:
        df: pd.DataFrame = state["df"]
        artifacts_dir: str = state.get("artifacts_dir", "artifacts")

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

        saved_payload = {
            "profile": profile,
            "target_candidates": target_candidates,
            "selected_target": selected_target,
            "target_source": target_source,
            "problem_type_analysis": problem_type_analysis,
        }
        try:
            from agents.versioning_utils import save_artifact
        except ImportError:
            from versioning_utils import save_artifact

        artifact_path = save_artifact(
            saved_payload, artifacts_dir, "dataset_profile", "json"
        )

        state["profile"] = profile
        state["target_candidates"] = target_candidates
        state["selected_target"] = selected_target
        state["target_source"] = target_source
        state["problem_type_analysis"] = problem_type_analysis
        state["dataset_profile_artifact_path"] = artifact_path

        return state


class CleaningAgent(BaseAgent):
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
    def run(self, state: dict) -> dict:
        cleaned_df: pd.DataFrame = state["cleaned_df"]
        target_column: str = state["selected_target"]
        problem_type: str = state["problem_type_analysis"]["problem_type"]
        ordinal_columns: dict[str, list[str]] | None = state.get("ordinal_columns")
        artifacts_dir: str = state.get("artifacts_dir", "artifacts")

        high_missing_cols: list[str] = []
        changelog_path = state.get("cleaning_changelog_path")
        if changelog_path and os.path.exists(changelog_path):
            try:
                with open(changelog_path, "r") as f:
                    cl = json.load(f)
                flagged = cl.get("missing_value_imputation", {}).get(
                    "columns_flagged_high_missing", []
                )
                for item in flagged:
                    if isinstance(item, dict) and "column" in item:
                        high_missing_cols.append(item["column"])
                    elif isinstance(item, str):
                        high_missing_cols.append(item)
            except Exception:
                pass

        fe_result = build_feature_pipeline(
            cleaned_df,
            target_column=target_column,
            problem_type=problem_type,
            ordinal_columns=ordinal_columns,
            high_missing_columns=high_missing_cols,
        )

        transformed_df = fe_result["transformed_df"]
        if transformed_df.isna().any().any():
            null_counts = transformed_df.isna().sum()
            nan_cols = null_counts[null_counts > 0].to_dict()
            raise ValueError(
                f"Validation failed: transformed_df contains NaN values in columns: {nan_cols}. "
                f"Feature matrix must be strictly NaN-free."
            )

        try:
            from agents.versioning_utils import get_next_version, save_artifact
        except ImportError:
            from versioning_utils import get_next_version, save_artifact

        version = get_next_version(artifacts_dir, "feature_engineered")
        csv_path = save_artifact(
            transformed_df,
            artifacts_dir,
            "feature_engineered",
            "csv",
            version=version,
        )

        collinear_pairs_dicts = [
            {"col_a": p["col_a"], "col_b": p["col_b"], "correlation": p["correlation"]}
            for p in fe_result["collinear_pairs"]
        ]
        fe_summary = {
            "columns": list(fe_result["transformed_df"].columns),
            "dtypes": {
                str(c): str(t)
                for c, t in fe_result["transformed_df"].dtypes.items()
            },
            "target_column": target_column,
            "problem_type": problem_type,
            "encoding_map": fe_result["encoding_map"],
            "excluded_columns": fe_result["excluded_columns"],
            "collinear_pairs": collinear_pairs_dicts,
        }
        json_path = save_artifact(
            fe_summary,
            artifacts_dir,
            "feature_engineered",
            "json",
            version=version,
        )

        import pickle
        pkl_name = f"pipeline_v{version}.pkl"
        pkl_path = os.path.join(artifacts_dir, pkl_name)
        with open(pkl_path, "wb") as f:
            pickle.dump(fe_result["pipeline"], f)

        state["feature_engineering_result"] = fe_result
        state["feature_engineered_artifact_path"] = csv_path
        state["feature_engineered_json_path"] = json_path
        state["feature_engineered_pipeline_path"] = pkl_path

        return state
