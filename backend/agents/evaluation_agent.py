from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from agents.base_agent import BaseAgent
    from agents.logging_utils import log_agent_run, with_agent_logging
    from agents.training_agent import get_estimator, load_training_inputs
    from agents.versioning_utils import (
        get_latest_version_path,
        get_next_version,
        save_artifact,
    )
except ImportError:
    from base_agent import BaseAgent
    from logging_utils import log_agent_run, with_agent_logging
    from training_agent import get_estimator, load_training_inputs
    from versioning_utils import (
        get_latest_version_path,
        get_next_version,
        save_artifact,
    )

logger = logging.getLogger("agent_runs")

LOWER_IS_BETTER_METRICS = {
    "rmse",
    "rootmeansquarederror",
    "root_mean_squared_error",
    "mae",
    "meanabsoluteerror",
    "mean_absolute_error",
    "mse",
    "meansquarederror",
    "mean_squared_error",
    "loss",
    "logloss",
}


def is_lower_is_better_metric(metric_name: str | None) -> bool:
    """Return True if lower score is better for the specified metric (e.g. RMSE, MAE)."""
    if not metric_name:
        return False
    norm = (
        str(metric_name)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    return any(m in norm for m in ["rmse", "mae", "mse", "loss"])


def select_winning_model(
    training_results: list[dict[str, Any]],
    recommended_metric: str = "Accuracy",
    problem_type: str = "classification",
) -> tuple[dict[str, Any], dict[str, Any] | None, float | None]:
    """Select the winning model from training results, considering metric direction.

    Returns:
        (winning_model_dict, runner_up_dict, score_gap)

    Raises:
        ValueError: If training_results is empty or no models succeeded.
    """
    if not training_results:
        raise ValueError("training_results list is empty; cannot select a winning model.")

    successful_models = [
        m for m in training_results if m.get("status") == "success" and m.get("score") is not None
    ]

    if not successful_models:
        raise ValueError(
            "All candidate models failed or were skipped during training. "
            "No successful model available for evaluation."
        )

    lower_better = is_lower_is_better_metric(recommended_metric)

    # Sort successful models best-to-worst
    sorted_success = sorted(
        successful_models,
        key=lambda m: float(m["score"]),
        reverse=not lower_better,
    )

    winner = sorted_success[0]
    runner_up = sorted_success[1] if len(sorted_success) > 1 else None

    score_gap = None
    if runner_up is not None and runner_up.get("score") is not None:
        score_gap = round(abs(float(winner["score"]) - float(runner_up["score"])), 4)

    return winner, runner_up, score_gap


def generate_confusion_matrix_plot(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    labels: list[Any] | None = None,
) -> str:
    """Generate a Plotly heatmap representation of the confusion matrix."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if labels is None:
        unique_labels = sorted(
            list(set(y_true_arr.tolist()) | set(y_pred_arr.tolist())),
            key=lambda x: str(x),
        )
    else:
        unique_labels = labels

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_labels)
    str_labels = [str(lbl) for lbl in unique_labels]

    # Create annotated heatmap
    z_text = [[str(val) for val in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred: {lbl}" for lbl in str_labels],
            y=[f"Actual: {lbl}" for lbl in str_labels],
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 14},
            colorscale="Blues",
            showscale=True,
        )
    )

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        yaxis_autorange="reversed",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig.to_json()


def generate_roc_curve_plot(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    problem_type: str = "classification",
) -> tuple[str | None, float | None]:
    """Generate a Plotly ROC curve with calculated AUC.

    Returns:
        (roc_json_str, auc_score)
    """
    if str(problem_type).strip().lower() != "classification" or y_proba is None:
        return None, None

    y_true_arr = np.asarray(y_true)
    unique_classes = np.unique(y_true_arr)

    # Handle binary classification
    if len(unique_classes) == 2:
        pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            # Map classes to 0 and 1 if boolean/string
            pos_label = unique_classes[1]
            fpr, tpr, _ = roc_curve(y_true_arr, pos_proba, pos_label=pos_label)
            auc_val = float(roc_auc_score(y_true_arr, pos_proba))

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC Curve (AUC = {auc_val:.3f})",
                    line=dict(color="#2b5c8f", width=2.5),
                )
            )
            # Diagonal chance line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Chance (AUC = 0.500)",
                    line=dict(color="grey", dash="dash"),
                )
            )

            fig.update_layout(
                title=f"Receiver Operating Characteristic (AUC = {auc_val:.3f})",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0.0, 1.0]),
                yaxis=dict(range=[0.0, 1.05]),
                margin=dict(l=40, r=40, t=50, b=40),
                legend=dict(x=0.6, y=0.1),
            )

            return fig.to_json(), round(auc_val, 4)
        except Exception as exc:
            logger.warning("Failed to compute binary ROC curve: %s", exc)
            return None, None

    # Handle multi-class classification via Macro/One-vs-Rest AUC
    try:
        auc_val = float(
            roc_auc_score(
                y_true_arr, y_proba, multi_class="ovr", average="weighted"
            )
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Chance (AUC = 0.500)",
                line=dict(color="grey", dash="dash"),
            )
        )
        fig.update_layout(
            title=f"Multi-Class ROC (Weighted OVR AUC = {auc_val:.3f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        return fig.to_json(), round(auc_val, 4)
    except Exception as exc:
        logger.warning("Failed to compute multi-class ROC curve: %s", exc)
        return None, None


def generate_feature_importance_plot(
    estimator: Any,
    feature_names: list[str],
    top_n: int = 15,
) -> str | None:
    """Generate a Plotly horizontal bar chart of feature importances for tree-based estimators."""
    if not hasattr(estimator, "feature_importances_"):
        return None

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None or len(importances) == 0:
        return None

    feat_names = list(feature_names)
    if len(feat_names) != len(importances):
        feat_names = [f"feature_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({
        "feature": feat_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=True)

    if len(df_imp) > top_n:
        df_imp = df_imp.tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=df_imp["importance"],
            y=df_imp["feature"],
            orientation="h",
            marker=dict(color="#3b82f6"),
        )
    )

    fig.update_layout(
        title=f"Top {len(df_imp)} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=80, r=40, t=50, b=40),
    )

    return fig.to_json()


def generate_llm_evaluation_explanation(
    winning_model_name: str,
    winning_score: float,
    recommended_metric: str,
    problem_type: str,
    comparison_table: list[dict[str, Any]],
    runner_up_name: str | None = None,
    runner_up_score: float | None = None,
    score_gap: float | None = None,
    ml_plan: dict[str, Any] | None = None,
    confidence_score: dict[str, Any] | None = None,
    eda_summary: str | None = None,
) -> str:
    """Invoke Gemini to generate a grounded plain-English explanation for why the winning model won."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # Format numbers and comparison table cleanly
    table_lines = []
    for item in comparison_table:
        m_name = item.get("model_name", "Unknown")
        m_status = item.get("status", "unknown")
        m_score = item.get("score")
        score_str = f"{m_score:.4f}" if isinstance(m_score, (int, float)) else "N/A"
        table_lines.append(f"- {m_name}: status={m_status}, {recommended_metric}={score_str}")
    table_str = "\n".join(table_lines)

    runner_up_text = (
        f"Runner-up Model: {runner_up_name} with {recommended_metric} = {runner_up_score:.4f}\n"
        f"Numeric Gap: {score_gap:.4f}"
        if runner_up_name and runner_up_score is not None and score_gap is not None
        else "No runner-up model was available."
    )

    ml_plan_reasoning = ""
    if ml_plan and "candidate_models" in ml_plan:
        cands = ml_plan.get("candidate_models", [])
        ml_plan_reasoning = "\n".join(
            f"- {c.get('model_name')}: {c.get('reasoning')}"
            for c in cands
            if isinstance(c, dict) and "model_name" in c
        )

    conf_summary = ""
    if confidence_score and "breakdown" in confidence_score:
        conf_summary = f"Confidence score total: {confidence_score.get('total_score', 'N/A')}/100"

    prompt = f"""You are an expert ML Evaluation Agent. Explain why the winning model outperformed the other candidates based on the actual experimental results below.

EXPERIMENTAL RESULTS:
Problem Type: {problem_type}
Optimized Metric: {recommended_metric}
Winning Model: {winning_model_name}
Winning Score: {winning_score:.4f}
{runner_up_text}

FULL COMPARISON TABLE:
{table_str}

CANDIDATE MODEL ARCHITECTURAL REASONING (from ML Plan):
{ml_plan_reasoning if ml_plan_reasoning else "N/A"}

DATASET CONTEXT:
{conf_summary if conf_summary else "Standard preprocessed dataset"}
{f"EDA Context: {eda_summary}" if eda_summary else ""}

REQUIREMENTS:
1. Ground your explanation directly in the specific numeric scores from the comparison table (e.g. state {winning_score:.4f} vs {runner_up_score if runner_up_score is not None else 'other candidates'}).
2. Explain the architectural advantages of '{winning_model_name}' on this specific problem type and dataset structure that likely caused it to outperform the other models.
3. Keep the tone concise, clear, and professional for an ML practitioner (2-3 paragraphs).
"""

    if not api_key or genai is None:
        gap_phrase = (
            f" surpassing the runner-up {runner_up_name} ({runner_up_score:.4f}) by a margin of {score_gap:.4f}"
            if runner_up_name and runner_up_score is not None and score_gap is not None
            else ""
        )
        return (
            f"The {winning_model_name} achieved the best performance with a {recommended_metric} score of {winning_score:.4f}{gap_phrase}. "
            f"Its model architecture effectively captured feature interactions and patterns present in the {problem_type} dataset, "
            f"outperforming the alternative candidate models in the evaluation comparison."
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text or ""
        return text.strip()
    except Exception as exc:
        logger.warning("Gemini evaluation explanation call failed (%s). Using fallback.", exc)
        gap_phrase = (
            f" surpassing the runner-up {runner_up_name} ({runner_up_score:.4f}) by a margin of {score_gap:.4f}"
            if runner_up_name and runner_up_score is not None and score_gap is not None
            else ""
        )
        return (
            f"The {winning_model_name} achieved the best performance with a {recommended_metric} score of {winning_score:.4f}{gap_phrase}. "
            f"Its model architecture effectively captured feature interactions and patterns present in the {problem_type} dataset, "
            f"outperforming the alternative candidate models in the evaluation comparison."
        )


class EvaluationAgent(BaseAgent):
    """Evaluates the winning ML model, generates Plotly visualizations, and saves an evaluation bundle."""

    def run(self, state: dict | None = None) -> dict:
        if state is None:
            state = {}

        artifacts_dir = state.get("artifacts_dir", "artifacts")
        start_agent_time = time.perf_counter()

        # Step 1: Load inputs from state or artifacts
        if (
            "transformed_df" in state
            and "cleaned_df" in state
            and "training_results" in state
        ):
            X = state["transformed_df"]
            cleaned_df = state["cleaned_df"]
            training_results = state["training_results"]
            fe_result = state.get("feature_engineering_result") or {}
            target_column = state.get("selected_target") or fe_result.get("target_column")
            if not target_column or target_column not in cleaned_df.columns:
                target_column = [c for c in cleaned_df.columns if c not in X.columns][-1]
            y = cleaned_df[target_column]
            ml_plan = state.get("ml_plan") or {}
            problem_type = ml_plan.get("problem_type") or state.get("problem_type") or "classification"
            recommended_metric = ml_plan.get("recommended_metric") or "Accuracy"
            confidence_score = state.get("confidence_score")
            eda_summary = state.get("eda_summary")
        else:
            # Load from artifacts
            tr_path = get_latest_version_path(artifacts_dir, "training_results", "json")
            with open(tr_path, "r") as f:
                training_results = json.load(f)

            train_inputs = load_training_inputs(artifacts_dir)
            X = train_inputs["X"]
            y = train_inputs["y"]
            target_column = train_inputs["target_column"]
            problem_type = train_inputs["problem_type"]
            recommended_metric = train_inputs["recommended_metric"]
            ml_plan = train_inputs.get("ml_plan") or {}

            # Optional dataset profile & confidence score
            confidence_score = None
            try:
                prof_path = get_latest_version_path(artifacts_dir, "dataset_profile", "json")
                with open(prof_path, "r") as f:
                    prof_data = json.load(f)
                    confidence_score = prof_data.get("confidence_score")
            except Exception:
                pass

            eda_summary = None
            try:
                eda_path = get_latest_version_path(artifacts_dir, "eda_bundle", "json")
                with open(eda_path, "r") as f:
                    eda_data = json.load(f)
                    eda_summary = eda_data.get("insights")
            except Exception:
                pass

        pt = problem_type.strip().lower()

        # Step 2: Select winning model
        winner, runner_up, score_gap = select_winning_model(
            training_results=training_results,
            recommended_metric=recommended_metric,
            problem_type=pt,
        )

        winning_model_name = winner["model_name"]
        winning_score = float(winner["score"])
        runner_up_name = runner_up["model_name"] if runner_up else None
        runner_up_score = float(runner_up["score"]) if runner_up and runner_up.get("score") is not None else None

        # Step 3: Re-create identical train/test split from Week 5
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

        # Refit ONLY the winning estimator on (X_train, y_train)
        estimator = get_estimator(winning_model_name, pt)
        if estimator is None:
            raise ValueError(f"Could not instantiate estimator for winning model '{winning_model_name}'.")

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_proba = None
        if hasattr(estimator, "predict_proba"):
            try:
                y_proba = estimator.predict_proba(X_test)
            except Exception as e:
                logger.info("predict_proba failed for %s: %s", winning_model_name, e)
                y_proba = None

        skipped_visualizations: list[dict[str, str]] = []
        confusion_matrix_json = None
        roc_curve_json = None
        feature_importance_json = None

        # Step 4: Classification Visualizations
        if pt == "classification":
            try:
                confusion_matrix_json = generate_confusion_matrix_plot(y_test, y_pred)
            except Exception as exc:
                logger.warning("Failed to generate confusion matrix: %s", exc)
                skipped_visualizations.append({
                    "type": "confusion_matrix",
                    "reason": f"Error generating confusion matrix: {exc}",
                })

            if y_proba is not None:
                try:
                    roc_json, _ = generate_roc_curve_plot(y_test, y_proba, pt)
                    if roc_json:
                        roc_curve_json = roc_json
                    else:
                        skipped_visualizations.append({
                            "type": "roc_curve",
                            "reason": f"ROC curve generation returned empty for model '{winning_model_name}'.",
                        })
                except Exception as exc:
                    skipped_visualizations.append({
                        "type": "roc_curve",
                        "reason": f"Error generating ROC curve: {exc}",
                    })
            else:
                skipped_visualizations.append({
                    "type": "roc_curve",
                    "reason": f"Winning model '{winning_model_name}' does not expose predict_proba.",
                })
        else:
            # Regression problem type
            skipped_visualizations.append({
                "type": "confusion_matrix",
                "reason": "Not applicable for regression problem type.",
            })
            skipped_visualizations.append({
                "type": "roc_curve",
                "reason": "Not applicable for regression problem type.",
            })

        # Step 5: Feature Importance (tree-based models only)
        if hasattr(estimator, "feature_importances_"):
            try:
                feat_names = list(X.columns)
                feature_importance_json = generate_feature_importance_plot(
                    estimator, feat_names, top_n=15
                )
                if not feature_importance_json:
                    skipped_visualizations.append({
                        "type": "feature_importance_chart",
                        "reason": f"Feature importances empty or unavailable on '{winning_model_name}'.",
                    })
            except Exception as exc:
                skipped_visualizations.append({
                    "type": "feature_importance_chart",
                    "reason": f"Error generating feature importance chart: {exc}",
                })
        else:
            skipped_visualizations.append({
                "type": "feature_importance_chart",
                "reason": f"Winning model '{winning_model_name}' does not expose tree-based feature_importances_.",
            })

        # Step 6: LLM-generated explanation
        llm_explanation = generate_llm_evaluation_explanation(
            winning_model_name=winning_model_name,
            winning_score=winning_score,
            recommended_metric=recommended_metric,
            problem_type=pt,
            comparison_table=training_results,
            runner_up_name=runner_up_name,
            runner_up_score=runner_up_score,
            score_gap=score_gap,
            ml_plan=ml_plan,
            confidence_score=confidence_score,
            eda_summary=eda_summary,
        )

        # Step 7: Prepare bundle and save artifact
        bundle = {
            "winning_model_name": winning_model_name,
            "problem_type": pt,
            "recommended_metric": recommended_metric,
            "winning_score": winning_score,
            "runner_up_model_name": runner_up_name,
            "runner_up_score": runner_up_score,
            "score_gap": score_gap,
            "confusion_matrix": confusion_matrix_json,
            "roc_curve": roc_curve_json,
            "feature_importance_chart": feature_importance_json,
            "llm_explanation": llm_explanation,
            "skipped_visualizations": skipped_visualizations,
        }

        artifact_path = save_artifact(
            bundle,
            artifacts_dir,
            "evaluation_bundle",
            "json",
        )
        bundle["artifact_path"] = artifact_path

        # Step 8: Log agent run
        total_duration = time.perf_counter() - start_agent_time
        log_agent_run(
            agent_name="evaluation",
            inputs_summary={
                "winning_model": winning_model_name,
                "problem_type": pt,
                "recommended_metric": recommended_metric,
                "training_models_count": len(training_results),
            },
            outputs_summary={
                "winning_model": winning_model_name,
                "winning_score": winning_score,
                "visualizations_generated": [
                    k for k in ["confusion_matrix", "roc_curve", "feature_importance_chart"]
                    if bundle.get(k) is not None
                ],
                "visualizations_skipped": [s["type"] for s in skipped_visualizations],
            },
            duration_seconds=round(total_duration, 4),
        )

        # Update state
        state["evaluation_bundle"] = bundle
        state["winning_model_name"] = winning_model_name
        state["evaluation_artifact_path"] = artifact_path

        return state
