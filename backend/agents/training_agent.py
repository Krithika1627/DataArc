from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from agents.base_agent import BaseAgent
    from agents.logging_utils import log_agent_run
    from agents.versioning_utils import (
        get_latest_version_path,
        get_next_version,
        save_artifact,
    )
except ImportError:
    from base_agent import BaseAgent
    from logging_utils import log_agent_run
    from versioning_utils import (
        get_latest_version_path,
        get_next_version,
        save_artifact,
    )

logger = logging.getLogger("agent_runs")


MODEL_MAPPING: dict[str, dict[str, Any]] = {
    "classification": {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest Classifier": lambda: RandomForestClassifier(
            random_state=42
        ),
        "XGBoost Classifier": lambda: (
            XGBClassifier(random_state=42, eval_metric="logloss")
            if XGBClassifier is not None
            else None
        ),
        "LightGBM Classifier": lambda: (
            LGBMClassifier(random_state=42, verbose=-1)
            if LGBMClassifier is not None
            else None
        ),
        "SVC": lambda: SVC(probability=True, random_state=42),
    },
    "regression": {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest Regressor": lambda: RandomForestRegressor(
            random_state=42
        ),
        "XGBoost Regressor": lambda: (
            XGBRegressor(random_state=42)
            if XGBRegressor is not None
            else None
        ),
        "LightGBM Regressor": lambda: (
            LGBMRegressor(random_state=42, verbose=-1)
            if LGBMRegressor is not None
            else None
        ),
    },
}


def get_estimator(model_name: str, problem_type: str) -> Any | None:
    """Retrieve an instantiated estimator strictly by exact model_name string and problem_type."""
    if not isinstance(model_name, str) or not isinstance(problem_type, str):
        return None

    type_mapping = MODEL_MAPPING.get(problem_type.strip().lower())
    if not type_mapping:
        return None

    factory = type_mapping.get(model_name)
    if factory is None:
        return None

    estimator = factory()
    return estimator


def load_training_inputs(artifacts_dir: str = "artifacts") -> dict[str, Any]:
    """Load latest feature-engineered dataset, metadata, cleaned target, and ML plan from artifacts."""
    fe_csv_path = get_latest_version_path(
        artifacts_dir, "feature_engineered", "csv"
    )
    fe_json_path = get_latest_version_path(
        artifacts_dir, "feature_engineered", "json"
    )
    ml_plan_path = get_latest_version_path(artifacts_dir, "ml_plan", "json")
    cleaned_csv_path = get_latest_version_path(artifacts_dir, "cleaned", "csv")

    X_df = pd.read_csv(fe_csv_path)
    with open(fe_json_path, "r") as f:
        fe_metadata = json.load(f)
    with open(ml_plan_path, "r") as f:
        ml_plan = json.load(f)
    cleaned_df = pd.read_csv(cleaned_csv_path)

    target_column = (
        fe_metadata.get("target_column")
        or ml_plan.get("target_column")
    )
    if not target_column or target_column not in cleaned_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in cleaned dataset '{cleaned_csv_path}'."
        )

    y_series = cleaned_df[target_column]

    return {
        "X": X_df,
        "y": y_series,
        "target_column": target_column,
        "problem_type": ml_plan.get("problem_type")
        or fe_metadata.get("problem_type")
        or "classification",
        "recommended_metric": ml_plan.get("recommended_metric", "Accuracy"),
        "candidate_models": ml_plan.get("candidate_models", []),
        "ml_plan": ml_plan,
        "fe_metadata": fe_metadata,
    }


def compute_metrics(
    problem_type: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard metrics for classification or regression."""
    metrics: dict[str, float] = {}
    pt = problem_type.strip().lower()

    if pt == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        if y_proba is not None:
            try:
                unique_classes = np.unique(y_true)
                if len(unique_classes) == 2:
                    pos_proba = (
                        y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    )
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, pos_proba)
                    )
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="weighted"
                        )
                    )
            except Exception:
                pass

    else:
        try:
            rmse_val = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(rmse_val)
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

    return {k: round(v, 4) for k, v in metrics.items()}


def extract_primary_score(
    recommended_metric: str,
    metrics: dict[str, float],
    problem_type: str,
) -> float:
    """Extract primary score corresponding to recommended_metric."""
    norm_metric = (
        recommended_metric.strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    if "f1" in norm_metric:
        return metrics.get("f1", 0.0)
    if "accuracy" in norm_metric:
        return metrics.get("accuracy", 0.0)
    if "precision" in norm_metric:
        return metrics.get("precision", 0.0)
    if "recall" in norm_metric:
        return metrics.get("recall", 0.0)
    if "rocauc" in norm_metric or "auc" in norm_metric:
        return metrics.get("roc_auc", metrics.get("accuracy", 0.0))
    if "rmse" in norm_metric:
        return metrics.get("rmse", 0.0)
    if "r2" in norm_metric:
        return metrics.get("r2", 0.0)
    if "mae" in norm_metric:
        return metrics.get("mae", 0.0)

    if problem_type == "classification":
        return metrics.get("f1", metrics.get("accuracy", 0.0))
    else:
        return metrics.get("rmse", metrics.get("r2", 0.0))


class TrainingAgent(BaseAgent):
    """Trains candidate ML models with per-model failure isolation and artifact saving."""

    def run(self, state: dict | None = None) -> dict:
        if state is None:
            state = {}

        artifacts_dir = state.get("artifacts_dir", "artifacts")
        start_agent_time = time.perf_counter()

        # Load inputs from state or artifacts
        if (
            "transformed_df" in state
            and "cleaned_df" in state
            and "ml_plan" in state
        ):
            X = state["transformed_df"]
            fe_result = state.get("feature_engineering_result") or {}
            target_column = (
                state.get("selected_target")
                or fe_result.get("target_column")
            )
            cleaned_df = state["cleaned_df"]
            y = cleaned_df[target_column]
            ml_plan = state["ml_plan"]
            problem_type = ml_plan.get("problem_type", "classification")
            recommended_metric = ml_plan.get("recommended_metric", "F1")
            candidate_models = ml_plan.get("candidate_models", [])
        else:
            inputs = load_training_inputs(artifacts_dir)
            X = inputs["X"]
            y = inputs["y"]
            target_column = inputs["target_column"]
            problem_type = inputs["problem_type"]
            recommended_metric = inputs["recommended_metric"]
            candidate_models = inputs["candidate_models"]
            ml_plan = inputs["ml_plan"]

        pt = problem_type.strip().lower()

        # Train/Test Split (80/20, stratify=y for classification, None for regression)
        stratify_target = None
        if pt == "classification":
            class_counts = pd.Series(y).value_counts()
            if (class_counts >= 2).all():
                stratify_target = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_target,
        )

        comparison_table: list[dict[str, Any]] = []
        succeeded_count = 0
        failed_count = 0
        skipped_count = 0

        for candidate in candidate_models:
            model_name = (
                candidate.get("model_name")
                if isinstance(candidate, dict)
                else str(candidate)
            )

            # Step 1: Strict name-to-class lookup
            estimator = get_estimator(model_name, pt)
            if estimator is None:
                # Robust fallback for unrecognized model names
                skipped_count += 1
                error_msg = (
                    f"Unrecognized model name '{model_name}' for problem_type '{pt}'. Skipping."
                )
                log_agent_run(
                    agent_name=f"training_{model_name}",
                    inputs_summary={
                        "model_name": model_name,
                        "problem_type": pt,
                    },
                    outputs_summary={"status": "skipped_unrecognized"},
                    duration_seconds=0.0,
                    error=error_msg,
                )
                comparison_table.append(
                    {
                        "model_name": model_name,
                        "status": "skipped_unrecognized",
                        "recommended_metric": recommended_metric,
                        "score": None,
                        "metrics": None,
                        "training_time_seconds": 0.0,
                        "error_message": error_msg,
                    }
                )
                continue

            # Step 2: Fit, predict, and evaluate inside try/except block
            model_start_time = time.perf_counter()
            try:
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)
                y_proba = None
                if hasattr(estimator, "predict_proba"):
                    try:
                        y_proba = estimator.predict_proba(X_test)
                    except Exception:
                        y_proba = None

                metrics = compute_metrics(pt, y_test, y_pred, y_proba)
                score = extract_primary_score(recommended_metric, metrics, pt)
                model_duration = time.perf_counter() - model_start_time
                succeeded_count += 1

                log_agent_run(
                    agent_name=f"training_{model_name}",
                    inputs_summary={
                        "model_name": model_name,
                        "problem_type": pt,
                        "train_samples": len(X_train),
                    },
                    outputs_summary={
                        "status": "success",
                        "score": score,
                        "metrics": metrics,
                    },
                    duration_seconds=model_duration,
                )

                comparison_table.append(
                    {
                        "model_name": model_name,
                        "status": "success",
                        "recommended_metric": recommended_metric,
                        "score": score,
                        "metrics": metrics,
                        "training_time_seconds": round(model_duration, 4),
                        "error_message": None,
                    }
                )

            except Exception as exc:
                model_duration = time.perf_counter() - model_start_time
                failed_count += 1
                error_msg = str(exc)

                log_agent_run(
                    agent_name=f"training_{model_name}",
                    inputs_summary={
                        "model_name": model_name,
                        "problem_type": pt,
                        "train_samples": len(X_train),
                    },
                    outputs_summary={"status": "failed"},
                    duration_seconds=model_duration,
                    error=error_msg,
                )

                comparison_table.append(
                    {
                        "model_name": model_name,
                        "status": "failed",
                        "recommended_metric": recommended_metric,
                        "score": None,
                        "metrics": None,
                        "training_time_seconds": round(model_duration, 4),
                        "error_message": error_msg,
                    }
                )

        # Step 3: Sort comparison table - successful models sorted by score descending,
        # failed and skipped models placed at the bottom.
        success_models = [
            m for m in comparison_table if m["status"] == "success"
        ]
        success_models.sort(
            key=lambda x: (x["score"] if x["score"] is not None else -float("inf")),
            reverse=True,
        )
        non_success_models = [
            m for m in comparison_table if m["status"] != "success"
        ]
        sorted_comparison_table = success_models + non_success_models

        # Step 4: Save comparison table artifact
        artifact_path = save_artifact(
            sorted_comparison_table,
            artifacts_dir,
            "training_results",
            "json",
        )

        total_duration = time.perf_counter() - start_agent_time
        log_agent_run(
            agent_name="training",
            inputs_summary={
                "candidate_models_count": len(candidate_models),
                "problem_type": pt,
                "target_column": target_column,
            },
            outputs_summary={
                "total_models": len(candidate_models),
                "succeeded": succeeded_count,
                "failed": failed_count,
                "skipped_unrecognized": skipped_count,
                "artifact_path": artifact_path,
            },
            duration_seconds=total_duration,
        )

        state["training_results"] = sorted_comparison_table
        state["training_results_artifact_path"] = artifact_path
        state["training_summary"] = {
            "total_models": len(candidate_models),
            "succeeded": succeeded_count,
            "failed": failed_count,
            "skipped_unrecognized": skipped_count,
        }

        return state
