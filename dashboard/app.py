"""
DataArc - Autonomous Data Science Dashboard
Streamlit-based user interface for dataset analysis, cleaning, EDA, feature engineering,
model training, evaluation, and comprehensive PDF reporting.
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Any, Optional

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st

# Configuration & Backend Client Setup
BACKEND_URL = os.getenv("DATAARC_BACKEND_URL", os.getenv("BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")

st.set_page_config(
    page_title="DataArc - Autonomous Data Scientist",
    layout="wide",
    initial_sidebar_state="expanded",
)


# API Helper Functions
def check_backend_health() -> tuple[bool, str]:
    """Check if the backend server is reachable and database is connected."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            db_status = data.get("database", "connected")
            return True, f"Online (DB: {db_status})"
        return False, f"HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"Offline ({exc.__class__.__name__})"


def api_analyze_dataset(
    file_bytes: bytes, filename: str, user_selected_target: Optional[str] = None
) -> dict[str, Any]:
    """Call POST /analyze-dataset to profile dataset and detect target candidates."""
    files = {"file": (filename, file_bytes, "text/csv")}
    data = {}
    if user_selected_target:
        data["user_selected_target"] = user_selected_target

    resp = requests.post(f"{BACKEND_URL}/analyze-dataset", files=files, data=data, timeout=180.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Dataset Analysis failed ({resp.status_code}): {resp.text}")
    return resp.json()


def api_clean_dataset(
    file_bytes: bytes, filename: str, target_column: Optional[str], dataset_id: str
) -> dict[str, Any]:
    """Call POST /clean-dataset."""
    files = {"file": (filename, file_bytes, "text/csv")}
    data = {
        "dataset_id": dataset_id,
        "cap_target": "false",
    }
    if target_column:
        data["target_column"] = target_column

    resp = requests.post(f"{BACKEND_URL}/clean-dataset", files=files, data=data, timeout=180.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Data Cleaning failed ({resp.status_code}): {resp.text}")
    return resp.json()


def api_run_eda(
    file_bytes: bytes,
    filename: str,
    target_column: Optional[str],
    problem_type: Optional[str],
    cleaning_changelog_path: Optional[str],
    dataset_id: str,
) -> dict[str, Any]:
    """Call POST /run-eda."""
    files = {"file": (filename, file_bytes, "text/csv")}
    data: dict[str, str] = {"dataset_id": dataset_id}
    if target_column:
        data["target_column"] = target_column
    if problem_type:
        data["problem_type"] = problem_type
    if cleaning_changelog_path:
        data["cleaning_changelog_path"] = cleaning_changelog_path

    resp = requests.post(f"{BACKEND_URL}/run-eda", files=files, data=data, timeout=180.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Exploratory Data Analysis failed ({resp.status_code}): {resp.text}")
    return resp.json()


def api_run_feature_engineering(
    file_bytes: bytes,
    filename: str,
    target_column: str,
    problem_type: Optional[str],
    cleaning_changelog_path: Optional[str],
    dataset_id: str,
) -> dict[str, Any]:
    """Call POST /run-feature-engineering."""
    files = {"file": (filename, file_bytes, "text/csv")}
    data: dict[str, str] = {
        "dataset_id": dataset_id,
        "target_column": target_column,
    }
    if problem_type:
        data["problem_type"] = problem_type
    if cleaning_changelog_path:
        data["cleaning_changelog_path"] = cleaning_changelog_path

    resp = requests.post(f"{BACKEND_URL}/run-feature-engineering", files=files, data=data, timeout=180.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Feature Engineering failed ({resp.status_code}): {resp.text}")
    return resp.json()


def api_plan_ml(
    dataset_id: str, selected_target: str, problem_type: Optional[str]
) -> dict[str, Any]:
    """Call POST /plan-ml."""
    payload = {
        "dataset_id": dataset_id,
        "selected_target": selected_target,
    }
    if problem_type:
        payload["problem_type"] = problem_type

    resp = requests.post(f"{BACKEND_URL}/plan-ml", json=payload, timeout=180.0)
    if resp.status_code != 200:
        raise RuntimeError(f"ML Planning failed ({resp.status_code}): {resp.text}")
    return resp.json()


def api_train_model(dataset_id: str) -> dict[str, Any]:
    """Call POST /train-model."""
    payload = {"dataset_id": dataset_id}
    resp = requests.post(f"{BACKEND_URL}/train-model", json=payload, timeout=300.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Model Training failed ({resp.status_code}): {resp.text}")
    return resp.json()


def api_evaluate_model(dataset_id: str) -> dict[str, Any]:
    """Call POST /evaluate-model."""
    payload = {"dataset_id": dataset_id}
    resp = requests.post(f"{BACKEND_URL}/evaluate-model", json=payload, timeout=180.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Model Evaluation failed ({resp.status_code}): {resp.text}")
    return resp.json()


def api_generate_report(dataset_id: str) -> bytes:
    """Call POST /generate-report and return PDF bytes."""
    payload = {"dataset_id": dataset_id}
    resp = requests.post(f"{BACKEND_URL}/generate-report", json=payload, timeout=180.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Report Generation failed ({resp.status_code}): {resp.text}")
    return resp.content


def api_get_history() -> list[dict[str, Any]]:
    """Call GET /history to list all previous runs."""
    try:
        resp = requests.get(f"{BACKEND_URL}/history", timeout=30.0)
        if resp.status_code == 200:
            return resp.json().get("runs", [])
        return []
    except Exception:
        return []


def api_get_dataset_history(dataset_id: str) -> dict[str, Any]:
    """Call GET /history/{dataset_id}."""
    resp = requests.get(f"{BACKEND_URL}/history/{dataset_id}", timeout=60.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to load dataset history ({resp.status_code}): {resp.text}")
    return resp.json()


def api_get_report_pdf(dataset_id: str) -> Optional[bytes]:
    """Call GET /report/{dataset_id}."""
    try:
        resp = requests.get(f"{BACKEND_URL}/report/{dataset_id}", timeout=60.0)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def api_get_cleaned_csv(dataset_id: str) -> Optional[str]:
    """Call GET /artifacts/{dataset_id}/cleaned to fetch the raw cleaned CSV content."""
    try:
        resp = requests.get(f"{BACKEND_URL}/artifacts/{dataset_id}/cleaned", timeout=60.0)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


def api_get_feature_engineered_csv(dataset_id: str) -> Optional[str]:
    """Call GET /artifacts/{dataset_id}/feature-engineered to fetch model-ready CSV content."""
    try:
        resp = requests.get(f"{BACKEND_URL}/artifacts/{dataset_id}/feature-engineered", timeout=60.0)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None


# Session State Initialization & Reset Helper
def init_session_state():
    """Ensure all required session state variables exist."""
    defaults = {
        "dataset_id": None,
        "file_bytes": None,
        "filename": None,
        "file_signature": None,
        "analysis_result": None,
        "selected_target": None,
        "pipeline_running": False,
        "pipeline_completed": False,
        "pipeline_stage": None,
        "pipeline_logs": [],
        "pipeline_results": {},
        "report_pdf_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session_state(clear_file: bool = False):
    """Cleanly resets all pipeline data and results to avoid cross-contamination."""
    st.session_state["dataset_id"] = None
    if clear_file:
        st.session_state["file_bytes"] = None
        st.session_state["filename"] = None
        st.session_state["file_signature"] = None
    st.session_state["analysis_result"] = None
    st.session_state["selected_target"] = None
    st.session_state["pipeline_running"] = False
    st.session_state["pipeline_completed"] = False
    st.session_state["pipeline_stage"] = None
    st.session_state["pipeline_logs"] = []
    st.session_state["pipeline_results"] = {}
    st.session_state["report_pdf_bytes"] = None


# Helper for Plotly JSON rendering
def render_plotly_json(chart_json: Optional[str | dict], key: Optional[str] = None):
    """Safely render a Plotly figure from JSON string or dict."""
    if not chart_json:
        st.info("No visualization available for this metric.")
        return
    try:
        if isinstance(chart_json, str):
            fig_dict = json.loads(chart_json)
        elif isinstance(chart_json, dict):
            fig_dict = chart_json
        else:
            st.info("No visualization available for this metric.")
            return

        if isinstance(fig_dict, dict) and "figure" in fig_dict and isinstance(fig_dict["figure"], (dict, str)):
            fig_dict = json.loads(fig_dict["figure"]) if isinstance(fig_dict["figure"], str) else fig_dict["figure"]

        fig = pio.from_json(json.dumps(fig_dict))
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as exc:
        st.warning(f"Could not render chart: {exc}")


# Reusable Modular Results Walkthrough
def render_pipeline_results(
    results: dict[str, Any],
    dataset_id: str,
    pdf_bytes: Optional[bytes] = None,
    key_prefix: str = "",
):
    """Renders the comprehensive 5-tab walkthrough results for any dataset run."""
    cleaning_raw = results.get("cleaning")
    cleaning = cleaning_raw if isinstance(cleaning_raw, dict) else ({"summary": cleaning_raw} if cleaning_raw else {})

    eda_raw = results.get("eda")
    eda = eda_raw if isinstance(eda_raw, dict) else {}

    fe_raw = results.get("feature_engineering")
    fe = fe_raw if isinstance(fe_raw, dict) else {}

    plan_raw = results.get("ml_plan")
    plan = plan_raw if isinstance(plan_raw, dict) else {}

    training_raw = results.get("training")
    if isinstance(training_raw, dict):
        training = training_raw
    elif isinstance(training_raw, list):
        training = {"comparison_table": training_raw}
    else:
        training = {}

    eval_raw = results.get("evaluation")
    evaluation = eval_raw if isinstance(eval_raw, dict) else {}

    st.subheader("Pipeline Results & Diagnostics")

    # Top champion banner
    winning_model = evaluation.get("winning_model_name") or evaluation.get("winning_model") or "Champion Model"
    winning_score = evaluation.get("winning_score") or 0.0
    metric_name = evaluation.get("recommended_metric") or "Score"
    runner_up = evaluation.get("runner_up_model_name")
    gap = evaluation.get("score_gap")

    banner_col, dl_col = st.columns([3, 1])
    with banner_col:
        gap_text = f" (+{gap:.4f} over {runner_up})" if (runner_up and gap is not None) else ""
        explanation_text = evaluation.get("llm_explanation") or ""
        if isinstance(explanation_text, dict):
            explanation_text = explanation_text.get("summary", "")
        st.success(
            f"### Champion: **{winning_model}** - {metric_name}: `{winning_score:.4f}`{gap_text}\n"
            f"{explanation_text}"
        )
    with dl_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if pdf_bytes:
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"DataArc_Report_{dataset_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key=f"{key_prefix}dl_btn",
            )
        else:
            st.caption("PDF Report not generated for this run.")

    # Tabbed Results Breakdown
    tabs = st.tabs([
        "Model Comparison",
        "Diagnostics & Importances",
        "Exploratory Data Analysis",
        "Data Cleaning",
        "Feature Engineering & Plan",
    ])

    # Tab 1: Model Comparison Table
    with tabs[0]:
        st.markdown("#### Leaderboard & Model Comparison")
        st.caption("All candidate models evaluated on the identical held-out test split. Failures and errors are explicitly surfaced:")

        if isinstance(training, list):
            model_rows = training
        elif isinstance(training, dict):
            model_rows = training.get("comparison_table") or training.get("results") or []
        else:
            model_rows = []

        if not isinstance(model_rows, list):
            model_rows = []

        if model_rows:
            formatted_rows = []
            for r in model_rows:
                if not isinstance(r, dict):
                    continue
                metrics_dict = r.get("metrics") or {}
                if not isinstance(metrics_dict, dict):
                    metrics_dict = {}
                score_val = r.get("score")
                acc_val = metrics_dict.get("accuracy")
                f1_val = metrics_dict.get("f1")
                roc_val = metrics_dict.get("roc_auc")
                rmse_val = metrics_dict.get("rmse")
                r2_val = metrics_dict.get("r2")
                formatted_rows.append({
                    "Model Name": r.get("model_name", "Model"),
                    "Status": "Success" if r.get("status") == "success" else f"Failed: {r.get('status')}",
                    f"Score ({r.get('recommended_metric', 'Metric')})": f"{score_val:.4f}" if score_val is not None else "N/A",
                    "Accuracy": f"{acc_val:.4f}" if acc_val is not None else "-",
                    "F1": f"{f1_val:.4f}" if f1_val is not None else "-",
                    "ROC-AUC": f"{roc_val:.4f}" if roc_val is not None else "-",
                    "RMSE": f"{rmse_val:.4f}" if rmse_val is not None else "-",
                    "R2": f"{r2_val:.4f}" if r2_val is not None else "-",
                    "Training Time (s)": f"{r.get('training_time_seconds', 0):.3f}",
                    "Error Details": r.get("error_message") or "-",
                })
            if formatted_rows:
                comp_df = pd.DataFrame(formatted_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.info("No training results found for this run.")
        else:
            st.info("No training results found for this run.")

    # Tab 2: Diagnostics & Feature Importances
    with tabs[1]:
        st.markdown("#### Evaluation Diagnostics & Interpretability")

        diag_col1, diag_col2 = st.columns([1, 1])

        with diag_col1:
            st.markdown("##### Feature Importance (Winning Model)")
            fi_chart = evaluation.get("feature_importance_chart")
            if fi_chart:
                render_plotly_json(fi_chart, key=f"{key_prefix}fi_chart")
            else:
                st.info("Feature importance is not available or model is non-tree.")

        with diag_col2:
            if evaluation.get("problem_type") == "classification":
                st.markdown("##### Confusion Matrix")
                cm_chart = evaluation.get("confusion_matrix")
                if cm_chart:
                    render_plotly_json(cm_chart, key=f"{key_prefix}cm_chart")

                st.markdown("##### ROC Curve")
                roc_chart = evaluation.get("roc_curve")
                if roc_chart:
                    render_plotly_json(roc_chart, key=f"{key_prefix}roc_chart")
            else:
                st.info("Regression problem type: Confusion Matrix and ROC Curve are omitted.")

        skipped = evaluation.get("skipped_visualizations", [])
        if isinstance(skipped, list) and skipped:
            with st.expander("Skipped Visualizations"):
                for s in skipped:
                    if isinstance(s, dict):
                        st.write(f"- **{s.get('type')}**: {s.get('reason')}")

    # Tab 3: Exploratory Data Analysis (EDA)
    with tabs[2]:
        st.markdown("#### Exploratory Data Analysis")

        insights = eda.get("insights") or {}
        if isinstance(insights, dict) and insights:
            st.info(f"**AI Insights Summary:** {insights.get('summary', '')}")
            i1, i2 = st.columns(2)
            with i1:
                st.markdown("##### Key Patterns")
                for p in insights.get("key_patterns", []):
                    st.write(f"- {p}")
            with i2:
                st.markdown("##### Anomalies & Risks")
                for a in insights.get("anomalies_or_risks", []):
                    st.write(f"- {a}")

        eda_c1, eda_c2 = st.columns(2)
        with eda_c1:
            st.markdown("##### Target Distribution")
            render_plotly_json(eda.get("target_distribution"), key=f"{key_prefix}eda_target_dist")

        with eda_c2:
            st.markdown("##### Correlation Heatmap")
            render_plotly_json(eda.get("correlation_heatmap"), key=f"{key_prefix}eda_corr_heatmap")

        histograms_obj = eda.get("histograms") or {}
        if isinstance(histograms_obj, dict):
            if "histograms" in histograms_obj and isinstance(histograms_obj["histograms"], dict):
                histograms = histograms_obj["histograms"]
            elif "histogram_charts" in histograms_obj and isinstance(histograms_obj["histogram_charts"], dict):
                histograms = histograms_obj["histogram_charts"]
            else:
                histograms = {k: v for k, v in histograms_obj.items() if k != "skipped_columns" and not isinstance(v, list)}
        else:
            histograms = {}

        boxplots_obj = eda.get("boxplots") or {}
        if isinstance(boxplots_obj, dict):
            if "boxplots" in boxplots_obj and isinstance(boxplots_obj["boxplots"], dict):
                boxplots = boxplots_obj["boxplots"]
            elif "boxplot_charts" in boxplots_obj and isinstance(boxplots_obj["boxplot_charts"], dict):
                boxplots = boxplots_obj["boxplot_charts"]
            else:
                boxplots = {k: v for k, v in boxplots_obj.items() if k != "skipped_columns" and not isinstance(v, list)}
        else:
            boxplots = {}

        if isinstance(histograms, dict) and isinstance(boxplots, dict) and (histograms or boxplots):
            with st.expander("View Per-Feature Histograms & Boxplots"):
                feature_options = sorted(list(set(list(histograms.keys()) + list(boxplots.keys()))))
                feature_select = st.selectbox(
                    "Select Feature to inspect:",
                    options=feature_options,
                    key=f"{key_prefix}feature_select",
                )
                if feature_select:
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        if feature_select in histograms:
                            st.markdown(f"**Histogram:** `{feature_select}`")
                            render_plotly_json(histograms[feature_select], key=f"{key_prefix}hist_{feature_select}")
                        else:
                            st.info("No histogram available for this feature.")
                    with hc2:
                        if feature_select in boxplots:
                            st.markdown(f"**Boxplot:** `{feature_select}`")
                            render_plotly_json(boxplots[feature_select], key=f"{key_prefix}box_{feature_select}")
                        else:
                            st.info("No boxplot available for this feature.")

    # Tab 4: Data Cleaning Summary
    with tabs[3]:
        st.markdown("#### Data Cleaning Summary")
        if isinstance(cleaning, dict):
            clean_summary = cleaning.get("summary") or (cleaning if "duplicate_removal" in cleaning else {})
        elif isinstance(cleaning, list):
            clean_summary = {"operations": cleaning}
        else:
            clean_summary = {}

        if isinstance(clean_summary, dict) and clean_summary:
            expl = clean_summary.get("llm_explanation", {})
            expl_str = expl.get("summary", "Automated cleaning applied.") if isinstance(expl, dict) else str(expl)
            st.write(f"**Explanation:** {expl_str}")

            cl_c1, cl_c2, cl_c3, cl_c4 = st.columns(4)
            with cl_c1:
                dup = clean_summary.get("duplicate_removal", {}) if isinstance(clean_summary.get("duplicate_removal"), dict) else {}
                st.metric("Duplicates Removed", dup.get("duplicates_removed", 0))
            with cl_c2:
                imp = clean_summary.get("missing_value_imputation", {}) if isinstance(clean_summary.get("missing_value_imputation"), dict) else {}
                st.metric("Columns Imputed", len(imp.get("columns_imputed", [])))
            with cl_c3:
                out = clean_summary.get("outlier_handling", {}) if isinstance(clean_summary.get("outlier_handling"), dict) else {}
                st.metric("Outlier Columns Handled", len(out.get("columns_processed", [])))
            with cl_c4:
                dt = clean_summary.get("dtype_fixing", {}) if isinstance(clean_summary.get("dtype_fixing"), dict) else {}
                st.metric("Dtype Casts", len(dt.get("columns_fixed", [])))

            with st.expander("Detailed Changelog Operations"):
                st.json(clean_summary)
        else:
            st.info("No cleaning changelog found for this run.")

        st.markdown("---")
        st.markdown("##### Cleaned Dataset Preview & Download")
        cleaned_csv_text = results.get("cleaned_csv_text")
        if not cleaned_csv_text:
            cleaned_csv_text = api_get_cleaned_csv(dataset_id)
            if cleaned_csv_text:
                results["cleaned_csv_text"] = cleaned_csv_text

        if cleaned_csv_text:
            try:
                clean_df = pd.read_csv(io.StringIO(cleaned_csv_text))
                st.caption(f"Showing first 20 rows of cleaned dataset ({len(clean_df):,} total rows, {len(clean_df.columns)} columns):")
                st.dataframe(clean_df.head(20), use_container_width=True)
                st.download_button(
                    label="Download Full Cleaned CSV",
                    data=cleaned_csv_text,
                    file_name=f"{dataset_id}_cleaned.csv",
                    mime="text/csv",
                    key=f"{key_prefix}dl_cleaned_csv",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(f"Could not render cleaned dataset preview: {exc}")
        else:
            st.info("Cleaned CSV is not available for preview.")

    # Tab 5: Feature Engineering & ML Plan
    with tabs[4]:
        st.markdown("#### Feature Pipeline & ML Plan")

        fe_left, fe_right = st.columns(2)
        with fe_left:
            st.markdown("##### Feature Encodings & Scaling")
            enc_map = fe.get("encoding_map", {}) if isinstance(fe, dict) else {}
            if isinstance(enc_map, dict) and enc_map:
                enc_df = pd.DataFrame([{"Feature": k, "Transformation": v} for k, v in enc_map.items()])
                st.dataframe(enc_df, use_container_width=True, hide_index=True)

            collinear = fe.get("collinear_pairs", []) if isinstance(fe, dict) else []
            if isinstance(collinear, list) and collinear:
                st.warning(f"Detected {len(collinear)} highly correlated feature pairs:")
                st.dataframe(pd.DataFrame(collinear), use_container_width=True, hide_index=True)

        with fe_right:
            st.markdown("##### Recommended Metric & Candidates")
            if isinstance(plan, dict) and plan:
                st.write(f"**Optimization Metric:** `{plan.get('recommended_metric')}`")
                st.write(f"**Reasoning:** {plan.get('metric_reasoning')}")
                st.markdown("###### Candidate Models:")
                cand_models = plan.get("candidate_models", [])
                if isinstance(cand_models, list):
                    for m in cand_models:
                        if isinstance(m, dict):
                            st.markdown(f"- **{m.get('model_name', 'Model')}**: {m.get('reasoning', '')}")
                        else:
                            st.markdown(f"- **{m}**")
            else:
                st.info("No ML plan found for this run.")

        st.markdown("---")
        st.markdown("##### Model-Ready Data (Feature-Engineered Preview & Download)")
        st.caption(
            "This section displays the model-ready transformed feature matrix used directly for training. "
            "It looks different from the human-readable cleaned CSV because categorical features have been encoded "
            "(via one-hot or ordinal encoders) and numeric values have been scaled."
        )
        fe_csv_text = results.get("fe_csv_text")
        if not fe_csv_text:
            fe_csv_text = api_get_feature_engineered_csv(dataset_id)
            if fe_csv_text:
                results["fe_csv_text"] = fe_csv_text

        if fe_csv_text:
            try:
                fe_df = pd.read_csv(io.StringIO(fe_csv_text))
                st.caption(f"Showing first 20 rows of model-ready dataset ({len(fe_df):,} total rows, {len(fe_df.columns)} columns):")
                st.dataframe(fe_df.head(20), use_container_width=True)
                st.download_button(
                    label="Download Model-Ready Feature-Engineered CSV",
                    data=fe_csv_text,
                    file_name=f"{dataset_id}_feature_engineered.csv",
                    mime="text/csv",
                    key=f"{key_prefix}dl_fe_csv",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(f"Could not render model-ready dataset preview: {exc}")
        else:
            st.info("Model-ready feature-engineered CSV is not available for preview.")




# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
def main():
    init_session_state()

    # Sidebar
    with st.sidebar:
        st.title("DataArc")
        st.caption("Autonomous Data Science Pipeline")
        st.markdown("---")

        # Health Status
        is_healthy, health_msg = check_backend_health()
        if is_healthy:
            st.success(f"Backend: {health_msg}")
        else:
            st.error(f"Backend: {health_msg}")
            st.caption(f"Connecting to: `{BACKEND_URL}`")

        st.markdown("---")

        # Session Reset Button
        if st.button("Start Over / New Dataset", use_container_width=True):
            reset_session_state(clear_file=True)
            st.rerun()

        st.markdown("---")
        st.markdown("### Pipeline Architecture")
        st.markdown(
            """
            1. **Dataset Profiling**
            2. **Data Cleaning**
            3. **Exploratory Data Analysis**
            4. **Feature Engineering**
            5. **ML Planning**
            6. **Model Training**
            7. **Model Evaluation**
            8. **Executive Report Generation**
            """
        )

        if st.session_state["dataset_id"]:
            st.markdown("---")
            st.caption(f"**Active Dataset ID:** `{st.session_state['dataset_id']}`")

    # Main Area Header
    st.markdown(
        """
        <div style="padding: 1.2rem 0rem 0.8rem 0rem;">
            <h1 style="margin-bottom: 0.2rem;">DataArc Autonomous Data Scientist</h1>
            <p style="color: #666; font-size: 1.1rem; margin-top: 0;">
                Upload a raw CSV dataset to automatically profile, clean, analyze, feature-engineer, train, evaluate, and generate an executive report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top-Level Mode Tabs
    tab_pipeline, tab_history = st.tabs(["Active Pipeline", "Dataset History & Past Results"])

    # TAB 1: ACTIVE PIPELINE EXECUTION
    with tab_pipeline:
        # 1. CSV File Upload Section
        upload_col, info_col = st.columns([2, 1])

        with upload_col:
            uploaded_file = st.file_uploader(
                "Step 1: Upload your Dataset (.csv)",
                type=["csv"],
                help="Upload any tabular CSV file (e.g., Titanic, Student Performance, Customer Churn).",
                disabled=st.session_state["pipeline_running"],
            )

        with info_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "DataArc automatically isolates each dataset run with a unique ID and persists all artifacts permanently."
            )

        # Detect new or changed file upload
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_signature = f"{uploaded_file.name}_{len(file_bytes)}"

            # Check if user uploaded a different CSV -> reset state to prevent bleed
            if st.session_state["file_signature"] != file_signature:
                reset_session_state(clear_file=False)
                st.session_state["file_bytes"] = file_bytes
                st.session_state["filename"] = uploaded_file.name
                st.session_state["file_signature"] = file_signature

                with st.spinner("Profiling dataset, detecting targets, and analyzing structure..."):
                    try:
                        analysis = api_analyze_dataset(file_bytes, uploaded_file.name)
                        st.session_state["analysis_result"] = analysis
                        st.session_state["dataset_id"] = analysis.get("dataset_id")
                        st.session_state["selected_target"] = analysis.get("selected_target")
                        st.toast(f"Dataset analyzed! Assigned ID: {st.session_state['dataset_id']}")
                    except Exception as exc:
                        st.error(f"Failed to analyze dataset: {exc}")

        # If no dataset analyzed yet, show welcome placeholder
        if not st.session_state["analysis_result"]:
            st.markdown("---")
            st.markdown(
                """
                <div style="text-align: center; padding: 3rem 1rem; color: #888; border: 2px dashed #ddd; border-radius: 10px; margin-top: 1rem;">
                    <h3>Upload a CSV file to get started</h3>
                    <p>DataArc will automatically inspect columns, detect problem type, and guide you through the modeling process.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Extract Analysis Response
            analysis = st.session_state["analysis_result"]
            profile = analysis.get("profile", {})
            target_candidates = analysis.get("target_candidates", [])
            problem_type_info = analysis.get("problem_type_analysis", {})
            confidence_score_data = analysis.get("confidence_score") or {}
            columns_list = [col["name"] for col in profile.get("columns", [])]

            # 2. Dataset Overview & Target Column Selection
            st.markdown("---")
            st.subheader("Step 2: Dataset Profile & Target Confirmation")

            # Metrics row
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Total Rows", f"{profile.get('row_count', 0):,}")
            with m2:
                st.metric("Total Columns", profile.get("column_count", 0))
            with m3:
                st.metric("Duplicate Rows", f"{profile.get('duplicate_row_count', 0)} ({profile.get('duplicate_row_percentage', 0.0)}%)")
            with m4:
                st.metric("Problem Type", problem_type_info.get("problem_type", "unclear").capitalize())
            with m5:
                total_conf = confidence_score_data.get("total_score", 0)
                st.metric("Confidence Score", f"{total_conf}/100")

            # Target Column Selection with reasoning
            st.markdown("#### Target Column Selection")
            st.caption("DataArc pre-selects the highest-confidence target candidate. You may override this selection below:")

            candidate_names = [c["column_name"] for c in target_candidates]
            all_col_options = candidate_names + [c for c in columns_list if c not in candidate_names]

            # Determine default index
            default_target = st.session_state.get("selected_target") or (candidate_names[0] if candidate_names else None)
            default_idx = all_col_options.index(default_target) if (default_target and default_target in all_col_options) else 0

            t_col1, t_col2 = st.columns([1, 2])

            with t_col1:
                selected_target = st.selectbox(
                    "Select Target Variable:",
                    options=all_col_options,
                    index=default_idx,
                    disabled=st.session_state["pipeline_running"],
                    help="The variable you want machine learning models to predict.",
                )
                st.session_state["selected_target"] = selected_target

            with t_col2:
                # Show explanation for why target was selected or candidate reasons
                matching_cand = next((c for c in target_candidates if c["column_name"] == selected_target), None)
                if matching_cand:
                    st.success(
                        f"**Auto-Detected Target:** `{selected_target}` (Confidence: {matching_cand.get('confidence_score', 0):.0f}%)\n\n"
                        + " - " + "\n - ".join(matching_cand.get("reasons", []))
                    )
                else:
                    st.info(f"Custom target selected: `{selected_target}`. Pipeline will adapt modeling to this column.")

            # Confidence Score & Plan Expanders
            with st.expander("View Confidence Score Breakdown & Project Plan", expanded=False):
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.markdown("##### Confidence Score Breakdown")
                    breakdown_items = confidence_score_data.get("breakdown", [])
                    if breakdown_items:
                        breakdown_df = pd.DataFrame(breakdown_items)
                        st.dataframe(
                            breakdown_df[["check", "points_awarded", "reason"]],
                            column_config={
                                "check": "Audit Check",
                                "points_awarded": st.column_config.NumberColumn("Points", format="%d pts"),
                                "reason": "Evidence & Justification",
                            },
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.write("No detailed breakdown available.")

                with c_right:
                    st.markdown("##### AI Strategy & Problem Formulation")
                    st.write(f"**Classification Reasoning:** {problem_type_info.get('confidence_reasoning', 'N/A')}")
                    st.write(f"**Project Plan:** {problem_type_info.get('project_plan', 'N/A')}")

            # 3. Pipeline Execution Section
            st.markdown("---")
            st.subheader("Step 3: Execute Autonomous Pipeline")

            btn_col, status_info_col = st.columns([1, 2])

            with btn_col:
                run_button = st.button(
                    "Run Full Pipeline",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state["pipeline_running"],
                )

            with status_info_col:
                if st.session_state["pipeline_completed"]:
                    st.success("Full pipeline executed successfully! Review the walkthrough results below.")
                elif st.session_state["pipeline_running"]:
                    st.info("Pipeline execution in progress...")
                else:
                    st.caption("Clicking will sequentially execute: Cleaning -> EDA -> Feature Engineering -> Planning -> Training -> Evaluation -> PDF Report.")

            # Pipeline Run Trigger
            if run_button:
                st.session_state["pipeline_running"] = True
                st.session_state["pipeline_completed"] = False
                st.session_state["pipeline_logs"] = []
                st.session_state["pipeline_results"] = {}
                st.session_state["report_pdf_bytes"] = None

                dataset_id = st.session_state["dataset_id"]
                file_bytes = st.session_state["file_bytes"]
                filename = st.session_state["filename"]
                target_col = st.session_state["selected_target"]
                problem_type = problem_type_info.get("problem_type", "classification")

                progress_bar = st.progress(0, text="Initializing autonomous pipeline...")
                status_box = st.status("Executing Autonomous Pipeline Stages...", expanded=True)

                try:
                    # Stage 1: Cleaning
                    status_box.write("**Stage 1/7: Data Cleaning** - Handling duplicates, imputing missing values, and validating dtypes...")
                    progress_bar.progress(14, text="Stage 1/7: Data Cleaning...")
                    t0 = time.perf_counter()
                    clean_res = api_clean_dataset(file_bytes, filename, target_col, dataset_id)
                    d1 = round(time.perf_counter() - t0, 3)
                    st.session_state["pipeline_results"]["cleaning"] = clean_res
                    cleaned_csv_text = api_get_cleaned_csv(dataset_id)
                    st.session_state["pipeline_results"]["cleaned_csv_text"] = cleaned_csv_text
                    changelog_path = clean_res.get("changelog_path")
                    st.session_state["pipeline_logs"].append(
                        {"stage": "Data Cleaning", "duration": f"{d1}s", "status": "Success", "summary": f"Cleaned dataset saved (v{clean_res.get('artifact_path', '').split('_v')[-1].replace('.csv','') if '_v' in clean_res.get('artifact_path', '') else '1'})."}
                    )
                    status_box.write(f"Data Cleaning finished in {d1}s")

                    # Stage 2: EDA
                    status_box.write("**Stage 2/7: Exploratory Data Analysis** - Computing stats, correlations, distributions & LLM insights...")
                    progress_bar.progress(28, text="Stage 2/7: Exploratory Data Analysis...")
                    t0 = time.perf_counter()
                    eda_res = api_run_eda(file_bytes, filename, target_col, problem_type, changelog_path, dataset_id)
                    d2 = round(time.perf_counter() - t0, 3)
                    st.session_state["pipeline_results"]["eda"] = eda_res
                    st.session_state["pipeline_logs"].append(
                        {"stage": "Exploratory Data Analysis", "duration": f"{d2}s", "status": "Success", "summary": "Generated statistical profiles, interactive Plotly charts, and AI insights."}
                    )
                    status_box.write(f"EDA completed in {d2}s")

                    # Stage 3: Feature Engineering
                    status_box.write("**Stage 3/7: Feature Engineering** - Building sklearn preprocessing pipeline, encoders & scaling...")
                    progress_bar.progress(42, text="Stage 3/7: Feature Engineering...")
                    t0 = time.perf_counter()
                    fe_res = api_run_feature_engineering(file_bytes, filename, target_col, problem_type, changelog_path, dataset_id)
                    d3 = round(time.perf_counter() - t0, 3)
                    st.session_state["pipeline_results"]["feature_engineering"] = fe_res
                    fe_csv_text = api_get_feature_engineered_csv(dataset_id)
                    st.session_state["pipeline_results"]["fe_csv_text"] = fe_csv_text
                    st.session_state["pipeline_logs"].append(
                        {"stage": "Feature Engineering", "duration": f"{d3}s", "status": "Success", "summary": f"Fitted transformation pipeline ({len(fe_res.get('encoding_map', {}))} feature mappings)."}
                    )
                    status_box.write(f"Feature Engineering completed in {d3}s")


                    # Stage 4: ML Planning
                    status_box.write("**Stage 4/7: ML Planning** - Formulating metric optimization and selecting candidate algorithms...")
                    progress_bar.progress(57, text="Stage 4/7: ML Planning...")
                    t0 = time.perf_counter()
                    plan_res = api_plan_ml(dataset_id, target_col, problem_type)
                    d4 = round(time.perf_counter() - t0, 3)
                    st.session_state["pipeline_results"]["ml_plan"] = plan_res
                    st.session_state["pipeline_logs"].append(
                        {"stage": "ML Planning", "duration": f"{d4}s", "status": "Success", "summary": f"Selected primary metric: {plan_res.get('recommended_metric')} with {len(plan_res.get('candidate_models', []))} models."}
                    )
                    status_box.write(f"ML Planning completed in {d4}s")

                    # Stage 5: Training
                    status_box.write("**Stage 5/7: Model Training** - Training candidate models on cross-validated splits...")
                    progress_bar.progress(71, text="Stage 5/7: Model Training...")
                    t0 = time.perf_counter()
                    train_res = api_train_model(dataset_id)
                    d5 = round(time.perf_counter() - t0, 3)
                    st.session_state["pipeline_results"]["training"] = train_res
                    tr_summary = train_res.get("summary") or {}
                    tr_table = train_res.get("comparison_table") or train_res.get("results") or []
                    total_m = tr_summary.get("total_models", len(tr_table))
                    succ_m = tr_summary.get("succeeded", len([r for r in tr_table if r.get("status") == "success"]))
                    fail_m = tr_summary.get("failed", len([r for r in tr_table if r.get("status") != "success"]))
                    st.session_state["pipeline_logs"].append(
                        {"stage": "Model Training", "duration": f"{d5}s", "status": "Success", "summary": f"Trained {total_m} models ({succ_m} succeeded, {fail_m} failed)."}
                    )
                    status_box.write(f"Model Training completed in {d5}s")

                    # Stage 6: Evaluation
                    status_box.write("**Stage 6/7: Model Evaluation** - Analyzing diagnostics, feature importances, ROC & Confusion Matrix...")
                    progress_bar.progress(85, text="Stage 6/7: Model Evaluation...")
                    t0 = time.perf_counter()
                    eval_res = api_evaluate_model(dataset_id)
                    d6 = round(time.perf_counter() - t0, 3)
                    st.session_state["pipeline_results"]["evaluation"] = eval_res
                    st.session_state["pipeline_logs"].append(
                        {"stage": "Model Evaluation", "duration": f"{d6}s", "status": "Success", "summary": f"Champion Model: {eval_res.get('winning_model_name')} (Score: {eval_res.get('winning_score', 0):.4f})."}
                    )
                    status_box.write(f"Evaluation completed in {d6}s")

                    # Stage 7: PDF Report
                    status_box.write("**Stage 7/7: Executive Report Generation** - Compiling multi-page publication-quality PDF report...")
                    progress_bar.progress(98, text="Stage 7/7: Executive Report Generation...")
                    t0 = time.perf_counter()
                    pdf_bytes = api_generate_report(dataset_id)
                    d7 = round(time.perf_counter() - t0, 3)
                    st.session_state["report_pdf_bytes"] = pdf_bytes
                    st.session_state["pipeline_logs"].append(
                        {"stage": "Report Generation", "duration": f"{d7}s", "status": "Success", "summary": f"Generated {len(pdf_bytes):,} bytes PDF report with complete traceability."}
                    )
                    status_box.write(f"Executive PDF report ready ({d7}s)")

                    progress_bar.progress(100, text="Pipeline Execution Complete!")
                    status_box.update(label="Pipeline Execution Completed Successfully!", state="complete", expanded=False)
                    st.session_state["pipeline_running"] = False
                    st.session_state["pipeline_completed"] = True
                    st.rerun()

                except Exception as exc:
                    st.session_state["pipeline_running"] = False
                    status_box.update(label="Pipeline Execution Failed", state="error", expanded=True)
                    st.error(f"Error during execution: {exc}")
                    st.session_state["pipeline_logs"].append(
                        {"stage": "Error", "duration": "0s", "status": "Failed", "summary": str(exc)}
                    )

            # 4. Live Execution Trace Log
            if st.session_state["pipeline_logs"]:
                with st.expander("View Pipeline Execution Trace & Logs", expanded=not st.session_state["pipeline_completed"]):
                    log_df = pd.DataFrame(st.session_state["pipeline_logs"])
                    st.dataframe(
                        log_df,
                        column_config={
                            "stage": "Agent / Stage",
                            "duration": "Duration",
                            "status": "Status",
                            "summary": "Agent Execution Summary",
                        },
                        use_container_width=True,
                        hide_index=True,
                    )

            # 5. Interactive Results Walkthrough
            if st.session_state["pipeline_completed"] and st.session_state["pipeline_results"]:
                st.markdown("---")
                render_pipeline_results(
                    results=st.session_state["pipeline_results"],
                    dataset_id=st.session_state["dataset_id"],
                    pdf_bytes=st.session_state.get("report_pdf_bytes"),
                    key_prefix="live_",
                )

    # TAB 2: DATASET HISTORY & PAST RESULTS
    with tab_history:
        st.subheader("Historical Dataset Runs & Persistent Results")
        st.caption("All dataset runs stored in the database are listed below. Select any run to inspect complete diagnostic charts, model leaderboards, and download its PDF report.")

        h_col1, h_col2 = st.columns([3, 1])
        with h_col2:
            refresh_history = st.button("Refresh History", use_container_width=True)

        runs = api_get_history()

        if not runs:
            st.info("No saved dataset runs found in the database. Run the pipeline on a dataset to view its history here.")
        else:
            # Format overview table
            run_table_data = []
            for r in runs:
                run_table_data.append({
                    "Dataset ID": r.get("dataset_id"),
                    "File Name": r.get("filename") or "-",
                    "Last Updated": r.get("last_updated", "")[:19].replace("T", " "),
                    "Rows": f"{r.get('row_count'):,}" if r.get("row_count") is not None else "-",
                    "Columns": r.get("column_count") or "-",
                    "Target Column": r.get("target_column") or "-",
                    "Problem Type": (r.get("problem_type") or "-").capitalize(),
                    "Champion Model": r.get("winning_model") or "-",
                    "Score": f"{r.get('winning_score'):.4f}" if r.get("winning_score") is not None else "-",
                    "Metric": r.get("recommended_metric") or "-",
                    "Report Ready": "Yes" if r.get("has_report") else "No",
                })

            overview_df = pd.DataFrame(run_table_data)
            st.dataframe(overview_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Inspect Past Run Details")

            # Selection dropdown with formatted label showing file name
            runs_by_id = {r["dataset_id"]: r for r in runs}
            dataset_options = [r["dataset_id"] for r in runs]

            def format_run_label(d_id: str) -> str:
                item = runs_by_id.get(d_id, {})
                fn = item.get("filename")
                fn_str = f" [{fn}]" if fn else ""
                target = item.get("target_column") or "N/A"
                model = item.get("winning_model") or "N/A"
                score = f"{item.get('winning_score'):.4f}" if item.get("winning_score") is not None else "N/A"
                return f"{d_id}{fn_str} (Target: {target}, Winner: {model}, Score: {score})"

            selected_history_id = st.selectbox(
                "Select a Dataset to load past results:",
                options=dataset_options,
                format_func=format_run_label,
                help="Choose any previously processed dataset to inspect all its stored artifacts.",
            )

            if selected_history_id:
                try:
                    with st.spinner("Loading stored artifacts from database..."):
                        history_bundle = api_get_dataset_history(selected_history_id)
                        hist_pdf_bytes = api_get_report_pdf(selected_history_id)

                    st.caption(f"Loaded results for dataset run: `{selected_history_id}`")

                    # Convert history bundle to pipeline results format
                    hist_results = {
                        "cleaning": history_bundle.get("cleaning") or {},
                        "eda": history_bundle.get("eda") or {},
                        "feature_engineering": history_bundle.get("feature_engineering") or {},
                        "ml_plan": history_bundle.get("ml_plan") or {},
                        "training": history_bundle.get("training") or {},
                        "evaluation": history_bundle.get("evaluation") or {},
                    }

                    render_pipeline_results(
                        results=hist_results,
                        dataset_id=selected_history_id,
                        pdf_bytes=hist_pdf_bytes,
                        key_prefix=f"hist_{selected_history_id}_",
                    )
                except Exception as exc:
                    st.error(f"Failed to load historical run details: {exc}")


if __name__ == "__main__":
    main()
