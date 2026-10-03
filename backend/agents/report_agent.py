from __future__ import annotations

import base64
import datetime
import html
import io
import json
import logging
import os
import re
import sys
import time
from typing import Any

import plotly.io as pio

try:
    import cffi
    orig_dlopen = cffi.FFI.dlopen

    def _patched_dlopen(self, name, flags=0):
        candidates = [name]
        if isinstance(name, str):
            for prefix in ["/opt/homebrew/lib", "/usr/local/lib"]:
                candidates.append(f"{prefix}/{name}")
                if not name.endswith(".dylib"):
                    candidates.append(f"{prefix}/{name}.dylib")
                    candidates.append(f"{prefix}/lib{name}.dylib")
                cleaned = name.replace("-0", "").replace(".0.dylib", ".dylib")
                candidates.append(f"{prefix}/{cleaned}")
                if not cleaned.endswith(".dylib"):
                    candidates.append(f"{prefix}/{cleaned}.dylib")
        for c in candidates:
            try:
                return orig_dlopen(self, c, flags)
            except Exception:
                continue
        return orig_dlopen(self, name, flags)

    cffi.FFI.dlopen = _patched_dlopen
except Exception:
    pass

try:
    import weasyprint
except ImportError:
    weasyprint = None

try:
    from agents.base_agent import BaseAgent
    from agents.logging_utils import log_agent_run
    from agents.versioning_utils import get_latest_version_path, get_next_version, save_artifact
except ImportError:
    from base_agent import BaseAgent
    from logging_utils import log_agent_run
    from versioning_utils import get_latest_version_path, get_next_version, save_artifact

logger = logging.getLogger("agent_runs")


def load_all_pipeline_artifacts(artifacts_dir: str = "artifacts") -> dict[str, Any]:
    required_artifacts = [
        ("dataset_profile", "json", "Dataset Understanding (Week 1)"),
        ("cleaned", "changelog_json", "Data Cleaning (Week 2)"),
        ("eda_bundle", "json", "Exploratory Data Analysis (Week 3)"),
        ("feature_engineered", "json", "Feature Engineering (Week 4)"),
        ("ml_plan", "json", "ML Planning (Week 5 Part 1)"),
        ("training_results", "json", "Model Training (Week 5 Part 2)"),
        ("evaluation_bundle", "json", "Model Evaluation (Week 6 Part 2)"),
    ]

    loaded_data: dict[str, Any] = {}
    artifact_versions: dict[str, str] = {}

    for base_name, ext, stage_desc in required_artifacts:
        try:
            if ext == "changelog_json":
                pattern = re.compile(rf"^{re.escape(base_name)}_v(\d+)_changelog\.json$")
                if not os.path.exists(artifacts_dir):
                    raise FileNotFoundError()
                matched = []
                for fname in os.listdir(artifacts_dir):
                    m = pattern.match(fname)
                    if m:
                        matched.append((int(m.group(1)), fname))
                if not matched:
                    raise FileNotFoundError()
                matched.sort(key=lambda x: (x[0], x[1]))
                fpath = os.path.join(artifacts_dir, matched[-1][1])
            else:
                fpath = get_latest_version_path(artifacts_dir, base_name, ext)

            with open(fpath, "r", encoding="utf-8") as f:
                loaded_data[base_name] = json.load(f)
            artifact_versions[base_name] = os.path.basename(fpath)
        except Exception:
            raise FileNotFoundError(
                f"Missing required artifact for stage '{stage_desc}'. "
                f"Could not find '{base_name}' artifact in '{artifacts_dir}'. "
                f"Please ensure all prior pipeline stages have completed before generating the final report."
            )

    return {
        "artifacts": loaded_data,
        "artifact_versions": artifact_versions,
    }


def plotly_json_to_base64_png(plotly_json_str: str | None) -> str | None:
    """Convert a Plotly JSON string to a base64-encoded PNG data URI."""
    if not plotly_json_str:
        return None
    try:
        fig = pio.from_json(plotly_json_str)
        img_bytes = fig.to_image(format="png", width=700, height=400, scale=2)
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:
        logger.warning("Failed to render Plotly chart to image: %s", exc)
        return None


def build_report_html(
    artifacts: dict[str, Any],
    artifact_versions: dict[str, str],
    generation_time: str | None = None,
) -> str:
    """Construct an executive-ready, semantic HTML report string with embedded styling."""
    if generation_time is None:
        generation_time = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    # 1. Extract Overview Data
    ds_profile = artifacts.get("dataset_profile", {})
    profile_data = ds_profile.get("profile", {})
    row_count = profile_data.get("row_count", "N/A")
    col_count = profile_data.get("column_count", "N/A")
    target_col = (
        ds_profile.get("selected_target")
        or artifacts.get("ml_plan", {}).get("target_column")
        or "Unknown"
    )
    problem_type = (
        artifacts.get("ml_plan", {}).get("problem_type")
        or ds_profile.get("problem_type_analysis", {}).get("problem_type")
        or "classification"
    ).capitalize()

    conf_score_data = ds_profile.get("confidence_score", {})
    total_conf_score = conf_score_data.get("total_score", "N/A")
    conf_breakdown = conf_score_data.get("breakdown", [])

    # 2. Extract Data Cleaning Data
    cleaning_changelog = artifacts.get("cleaned", {})
    cleaning_explanation = cleaning_changelog.get("llm_explanation", {})
    if isinstance(cleaning_explanation, dict):
        cleaning_summary_text = cleaning_explanation.get("summary", "Data cleaning applied successfully.")
    else:
        cleaning_summary_text = str(cleaning_explanation)

    dup_summary = cleaning_changelog.get("duplicate_removal", {})
    duplicates_removed = dup_summary.get("duplicates_removed", 0)

    impute_summary = cleaning_changelog.get("missing_value_imputation", {})
    cols_imputed = impute_summary.get("columns_imputed", [])

    outlier_summary = cleaning_changelog.get("outlier_handling", {})
    cols_outliers = outlier_summary.get("columns_processed", [])

    # 3. Extract EDA Insights
    eda_bundle = artifacts.get("eda_bundle", {})
    eda_insights_data = eda_bundle.get("insights", {})
    if isinstance(eda_insights_data, dict):
        eda_summary_text = eda_insights_data.get("summary", "Exploratory data analysis completed.")
        eda_bullets = eda_insights_data.get("insights", [])
    else:
        eda_summary_text = str(eda_insights_data)
        eda_bullets = []

    # 4. Extract Model Comparison Data
    training_results = artifacts.get("training_results", [])
    recommended_metric = artifacts.get("ml_plan", {}).get("recommended_metric", "Score")

    # 5. Extract Best Model & Evaluation Data
    eval_bundle = artifacts.get("evaluation_bundle", {})
    winning_model_name = eval_bundle.get("winning_model_name", "N/A")
    winning_score = eval_bundle.get("winning_score", "N/A")
    runner_up_name = eval_bundle.get("runner_up_model_name")
    runner_up_score = eval_bundle.get("runner_up_score")
    score_gap = eval_bundle.get("score_gap")
    llm_eval_explanation = eval_bundle.get("llm_explanation", "")

    # Convert evaluation charts to base64 images
    cm_img = plotly_json_to_base64_png(eval_bundle.get("confusion_matrix"))
    roc_img = plotly_json_to_base64_png(eval_bundle.get("roc_curve"))
    fi_img = plotly_json_to_base64_png(eval_bundle.get("feature_importance_chart"))

    # Render Confidence Breakdown Table
    conf_rows_html = ""
    for item in conf_breakdown:
        check_name = html.escape(str(item.get("check", "")).replace("_", " ").title())
        pts = item.get("points_awarded", 0)
        pts_badge = f'<span class="badge badge-success">+{pts} pts</span>' if pts > 0 else '<span class="badge badge-neutral">0 pts</span>'
        reason = html.escape(str(item.get("reason", "")))
        conf_rows_html += f"""
        <tr>
            <td><strong>{check_name}</strong></td>
            <td>{pts_badge}</td>
            <td>{reason}</td>
        </tr>
        """

    # Render Comparison Table Rows
    comparison_rows_html = ""
    for item in training_results:
        m_name = html.escape(str(item.get("model_name", "Unknown")))
        m_status = str(item.get("status", "unknown")).lower()
        m_score = item.get("score")
        m_time = item.get("training_time_seconds", 0.0)
        m_err = item.get("error_message")

        is_winner = (m_name == winning_model_name and m_status == "success")

        if m_status == "success":
            status_html = '<span class="badge badge-success">Success</span>'
            row_class = "row-winner" if is_winner else ""
            score_display = f"{m_score:.4f}" if isinstance(m_score, (int, float)) else "N/A"
            notes_display = "⭐ Winning Model" if is_winner else "Trained & Evaluated"
        elif m_status == "skipped_unrecognized":
            status_html = '<span class="badge badge-warning">Skipped</span>'
            row_class = "row-failed"
            score_display = "—"
            notes_display = html.escape(m_err or "Unrecognized model name")
        else:
            status_html = '<span class="badge badge-danger">Failed</span>'
            row_class = "row-failed"
            score_display = "—"
            notes_display = html.escape(m_err or "Model training failed")

        time_display = f"{m_time:.3f}s" if isinstance(m_time, (int, float)) else "—"

        comparison_rows_html += f"""
        <tr class="{row_class}">
            <td><strong>{m_name}</strong></td>
            <td>{status_html}</td>
            <td><strong>{score_display}</strong></td>
            <td>{time_display}</td>
            <td class="small-text">{notes_display}</td>
        </tr>
        """

    # Render Cleaning Bullet Points
    imputed_text = (
        ", ".join(f"<code>{html.escape(c['column'])}</code> ({c['strategy']})" for c in cols_imputed)
        if cols_imputed
        else "No missing value imputation needed."
    )
    outlier_text = (
        ", ".join(f"<code>{html.escape(c['column'])}</code> ({c.get('outliers_detected', 0)} capped)" for c in cols_outliers)
        if cols_outliers
        else "No outlier capping required."
    )

    # Render EDA Bullets
    eda_bullets_html = ""
    for b in eda_bullets:
        eda_bullets_html += f"<li>{html.escape(str(b))}</li>"
    if not eda_bullets_html:
        eda_bullets_html = "<li>Dataset distributions and correlations evaluated cleanly.</li>"

    # Traceability list
    versions_list_str = ", ".join(
        f"<code>{html.escape(v)}</code>" for v in artifact_versions.values()
    )

    # Winner vs Runner-up text
    winner_score_str = f"{winning_score:.4f}" if isinstance(winning_score, (int, float)) else str(winning_score)
    runner_up_html = ""
    if runner_up_name and runner_up_score is not None:
        gap_str = f"{score_gap:.4f}" if isinstance(score_gap, (int, float)) else "N/A"
        runner_up_html = f"""
        <p class="runner-up-note">
            Outperformed runner-up <strong>{html.escape(runner_up_name)}</strong> ({runner_up_score:.4f})
            by a margin of <strong>+{gap_str}</strong> on <em>{html.escape(recommended_metric)}</em>.
        </p>
        """

    # Evaluation Charts HTML
    eval_charts_html = ""
    if cm_img:
        eval_charts_html += f"""
        <div class="chart-container">
            <h4>Confusion Matrix</h4>
            <img src="{cm_img}" alt="Confusion Matrix" class="report-img" />
        </div>
        """
    if roc_img:
        eval_charts_html += f"""
        <div class="chart-container">
            <h4>Receiver Operating Characteristic (ROC)</h4>
            <img src="{roc_img}" alt="ROC Curve" class="report-img" />
        </div>
        """
    if fi_img:
        eval_charts_html += f"""
        <div class="chart-container">
            <h4>Top Feature Importances</h4>
            <img src="{fi_img}" alt="Feature Importance" class="report-img" />
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DataArc Executive ML Pipeline Report</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 16mm 20mm 16mm;
        @bottom-right {{
            content: "Page " counter(page) " of " counter(pages);
            font-size: 8pt;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #64748b;
        }}
    }}
    
    * {{
        box-sizing: border-box;
    }}
    
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background-color: #ffffff;
        font-size: 9.5pt;
        line-height: 1.5;
        margin: 0;
        padding: 0;
    }}
    
    code, pre {{
        font-family: ui-monospace, "Cascadia Code", "Source Code Pro", Menlo, Monaco, Consolas, "Courier New", monospace;
        font-size: 8.5pt;
    }}
    .header-banner {{
        border-bottom: 2px solid #2563eb;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }}
    .header-title {{
        font-size: 20pt;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }}
    .header-subtitle {{
        font-size: 10pt;
        color: #64748b;
        margin-top: 4px;
        margin-bottom: 0;
    }}
    h2 {{
        font-size: 12pt;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 22px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    h3 {{
        font-size: 10.5pt;
        font-weight: 600;
        color: #1e293b;
        margin-top: 14px;
        margin-bottom: 6px;
    }}
    h4 {{
        font-size: 9.5pt;
        font-weight: 600;
        color: #334155;
        margin-top: 8px;
        margin-bottom: 4px;
        text-align: center;
    }}
    .grid-2 {{
        display: flex;
        gap: 15px;
        margin-bottom: 12px;
    }}
    .card {{
        flex: 1;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 10px 14px;
    }}
    .metric-value {{
        font-size: 16pt;
        font-weight: 700;
        color: #2563eb;
        margin-top: 2px;
        margin-bottom: 0;
    }}
    .metric-label {{
        font-size: 8pt;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.5px;
        margin: 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
        margin-bottom: 12px;
        font-size: 8.5pt;
    }}
    th {{
        background-color: #f1f5f9;
        color: #334155;
        text-align: left;
        padding: 6px 10px;
        font-weight: 600;
        border-bottom: 1px solid #cbd5e1;
    }}
    td {{
        padding: 6px 10px;
        border-bottom: 1px solid #e2e8f0;
        vertical-align: middle;
    }}
    .row-winner {{
        background-color: #eff6ff;
        font-weight: 500;
    }}
    .row-failed {{
        background-color: #fef2f2;
        color: #991b1b;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 7.5pt;
        font-weight: 600;
    }}
    .badge-success {{ background-color: #dcfce7; color: #166534; }}
    .badge-danger {{ background-color: #fee2e2; color: #991b1b; }}
    .badge-warning {{ background-color: #fef3c7; color: #92400e; }}
    .badge-neutral {{ background-color: #f1f5f9; color: #475569; }}
    .callout {{
        background-color: #f8fafc;
        border-left: 3px solid #2563eb;
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        margin: 10px 0;
        font-size: 9pt;
        color: #334155;
    }}
    .runner-up-note {{
        font-size: 8.5pt;
        color: #475569;
        margin-top: 4px;
        margin-bottom: 10px;
    }}
    .chart-container {{
        text-align: center;
        margin-top: 10px;
        margin-bottom: 14px;
        page-break-inside: avoid;
    }}
    .report-img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
    }}
    ul {{
        margin-top: 4px;
        margin-bottom: 8px;
        padding-left: 18px;
    }}
    li {{
        margin-bottom: 3px;
        font-size: 9pt;
    }}
    code {{
        background-color: #f1f5f9;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 8pt;
        font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    .footer {{
        border-top: 1px solid #e2e8f0;
        margin-top: 24px;
        padding-top: 8px;
        font-size: 7.5pt;
        color: #64748b;
        line-height: 1.4;
    }}
    .small-text {{
        font-size: 8pt;
        color: #64748b;
    }}
</style>
</head>
<body>

<div class="header-banner">
    <h1 class="header-title">DataArc Executive ML Pipeline Report</h1>
    <p class="header-subtitle">Comprehensive Automated Dataset Analysis, Model Exploration, and Evaluation Summary</p>
</div>

<!-- SECTION A: DATASET OVERVIEW -->
<h2>1. Dataset & Problem Overview</h2>
<div class="grid-2">
    <div class="card">
        <p class="metric-label">Dataset Dimensions</p>
        <p class="metric-value">{row_count} <span style="font-size: 10pt; font-weight: normal; color: #64748b;">rows</span> &times; {col_count} <span style="font-size: 10pt; font-weight: normal; color: #64748b;">cols</span></p>
    </div>
    <div class="card">
        <p class="metric-label">Target & Task Type</p>
        <p class="metric-value"><code>{html.escape(target_col)}</code> <span style="font-size: 10pt; font-weight: 500; color: #0284c7;">({html.escape(problem_type)})</span></p>
    </div>
    <div class="card">
        <p class="metric-label">Target Confidence Score</p>
        <p class="metric-value">{total_conf_score} <span style="font-size: 10pt; font-weight: normal; color: #64748b;">/ 100</span></p>
    </div>
</div>

<h3>Target Selection Confidence Breakdown</h3>
<table>
    <thead>
        <tr>
            <th style="width: 25%;">Verification Check</th>
            <th style="width: 15%;">Points</th>
            <th style="width: 60%;">Reason & Grounding</th>
        </tr>
    </thead>
    <tbody>
        {conf_rows_html}
    </tbody>
</table>

<!-- SECTION B: DATA CLEANING -->
<h2>2. Data Cleaning & Integrity</h2>
<div class="callout">
    <strong>Cleaning Summary:</strong> {html.escape(cleaning_summary_text)}
</div>
<ul>
    <li><strong>Duplicates Removed:</strong> {duplicates_removed} duplicate rows cleaned.</li>
    <li><strong>Missing Value Imputations:</strong> {imputed_text}</li>
    <li><strong>Outlier Treatment:</strong> {outlier_text}</li>
</ul>

<!-- SECTION C: EDA INSIGHTS -->
<h2>3. Exploratory Data Insights</h2>
<p style="margin-bottom: 4px;">{html.escape(eda_summary_text)}</p>
<ul>
    {eda_bullets_html}
</ul>

<!-- SECTION D: MODEL COMPARISON -->
<h2>4. Candidate Model Comparison</h2>
<p style="font-size: 8.5pt; color: #64748b; margin-bottom: 6px;">
    Optimized primary evaluation metric: <strong>{html.escape(recommended_metric)}</strong>. Failed and skipped models are highlighted explicitly below.
</p>
<table>
    <thead>
        <tr>
            <th style="width: 32%;">Model Name</th>
            <th style="width: 14%;">Status</th>
            <th style="width: 18%;">{html.escape(recommended_metric)}</th>
            <th style="width: 16%;">Training Time</th>
            <th style="width: 20%;">Outcome Notes</th>
        </tr>
    </thead>
    <tbody>
        {comparison_rows_html}
    </tbody>
</table>

<!-- SECTION E: WINNING MODEL & EVALUATION -->
<h2>5. Best Model & Evaluation</h2>
<div class="card" style="margin-bottom: 12px; background-color: #f0fdf4; border-color: #bbf7d0;">
    <p class="metric-label" style="color: #166534;">🏆 Recommended Best Model</p>
    <p class="metric-value" style="color: #15803d; font-size: 17pt;">
        {html.escape(winning_model_name)}
        <span style="font-size: 11pt; color: #166534; font-weight: 600;">({html.escape(recommended_metric)} = {winner_score_str})</span>
    </p>
    {runner_up_html}
</div>

<div class="callout">
    <strong>Architectural Rationale & Grounded Explanation:</strong><br/>
    {html.escape(llm_eval_explanation)}
</div>

{eval_charts_html}

<!-- SECTION F: TRACEABILITY FOOTER -->
<div class="footer">
    <strong>Report Traceability:</strong> Generated on {html.escape(generation_time)}.<br/>
    <strong>Artifact Versions Consumed:</strong> {versions_list_str}.
</div>

</body>
</html>
"""
    return html_content


class ReportAgent(BaseAgent):
    """Compiles all pipeline stage artifacts into an executive-ready PDF report."""

    def run(self, state: dict | None = None) -> dict:
        if state is None:
            state = {}

        artifacts_dir = state.get("artifacts_dir", "artifacts")
        start_agent_time = time.perf_counter()

        # Step 1: Load all pipeline stage artifacts
        loaded_info = load_all_pipeline_artifacts(artifacts_dir)
        artifacts = loaded_info["artifacts"]
        artifact_versions = loaded_info["artifact_versions"]

        # Step 2: Build structured HTML content
        html_content = build_report_html(
            artifacts=artifacts,
            artifact_versions=artifact_versions,
        )

        # Step 3: Render to PDF via WeasyPrint
        if weasyprint is None:
            raise RuntimeError("WeasyPrint is not installed or available in the environment.")

        try:
            pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        except Exception as exc:
            logger.error("WeasyPrint PDF rendering failed: %s", exc)
            raise RuntimeError(f"WeasyPrint PDF rendering failed: {exc}") from exc

        # Step 4: Save report_v{N}.pdf versioned artifact
        report_artifact_path = save_artifact(
            pdf_bytes,
            artifacts_dir,
            "report",
            "pdf",
        )

        total_duration = time.perf_counter() - start_agent_time

        # Step 5: Log run
        log_agent_run(
            agent_name="reporting",
            inputs_summary={
                "artifacts_dir": artifacts_dir,
                "artifacts_loaded_count": len(artifacts),
                "dataset_rows": artifacts.get("dataset_profile", {}).get("profile", {}).get("row_count", 0),
            },
            outputs_summary={
                "pdf_path": report_artifact_path,
                "pdf_size_bytes": len(pdf_bytes),
                "versions_used": list(artifact_versions.values()),
            },
            duration_seconds=round(total_duration, 4),
        )

        # Step 6: Update state
        state["report_pdf_path"] = report_artifact_path
        state["report_html"] = html_content
        state["report_pdf_bytes_length"] = len(pdf_bytes)
        state["report_versions_used"] = artifact_versions

        return state
