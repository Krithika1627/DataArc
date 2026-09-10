from __future__ import annotations
import io
import json
import logging
import os
import pickle
import time
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from pydantic import BaseModel, Field
from agents.dataset_understanding import (
    classify_problem_type,
    detect_target_candidates,
    profile_dataset,
)
from agents.data_cleaning import clean_dataset
from agents.eda_agent import run_eda
from agents.feature_engineering_agent import build_feature_pipeline
from agents.ml_planning_agent import MLPlanningAgent
from agents.logging_utils import log_agent_run
from agents.versioning_utils import get_next_version, save_artifact, get_latest_version_path


class CleaningSummary(BaseModel):
    duplicate_removal: dict[str, Any] = Field(
        description="Duplicate removal results: rows_before, rows_after, duplicates_removed, duplicate_percentage"
    )
    missing_value_imputation: dict[str, Any] = Field(
        description="Missing value imputation results: columns_imputed, columns_flagged_high_missing, columns_skipped_no_missing"
    )
    outlier_handling: dict[str, Any] = Field(
        description="Outlier handling results: columns_processed, columns_skipped_low_cardinality, columns_skipped_target_protected"
    )
    dtype_fixing: dict[str, Any] = Field(
        description="Dtype fixing results: columns_fixed, columns_left_as_object"
    )
    llm_explanation: dict[str, Any] = Field(
        description="LLM-generated plain-English explanation of the cleaning steps"
    )


class CleanDatasetResponse(BaseModel):
    artifact_path: str = Field(
        description="Absolute path to the cleaned CSV artifact"
    )
    changelog_path: str = Field(
        description="Absolute path to the changelog JSON file"
    )
    summary: CleaningSummary = Field(
        description="Structured summary of all cleaning operations"
    )


class ColumnProfile(BaseModel):
    name: str = Field(description="Column name")
    dtype: str = Field(description="Pandas dtype string, e.g. 'int64'")
    missing_count: int = Field(description="Number of missing (NaN) values")
    missing_percentage: float = Field(description="Percentage of missing values, rounded to 2 decimals")
    unique_count: int = Field(description="Number of unique values")


class DatasetProfile(BaseModel):
    row_count: int = Field(description="Total number of rows")
    column_count: int = Field(description="Total number of columns")
    columns: list[ColumnProfile] = Field(description="Per-column statistics")
    duplicate_row_count: int = Field(description="Number of duplicate rows")
    duplicate_row_percentage: float = Field(description="Percentage of duplicate rows, rounded to 2 decimals")


class TargetCandidateResponse(BaseModel):
    column_name: str = Field(description="Name of the candidate column")
    confidence_score: float = Field(description="Heuristic confidence score (0–100)")
    reasons: list[str] = Field(description="Plain-English reasons the column was flagged or excluded")


class ProblemTypeAnalysis(BaseModel):
    problem_type: str = Field(
        description="One of: classification, regression, clustering, unclear"
    )
    confidence_reasoning: str = Field(
        description="Explanation of why this problem type was chosen"
    )
    project_plan: str = Field(
        description="2–4 sentence plain-English project plan"
    )


class AnalyzeDatasetResponse(BaseModel):
    profile: DatasetProfile = Field(description="Dataset profiling results")
    target_candidates: list[TargetCandidateResponse] = Field(
        description="Heuristic target candidates (always included for user review)"
    )
    target_source: str = Field(
        description="How the final target was determined: 'user_provided' or 'auto_detected'"
    )
    selected_target: Optional[str] = Field(
        description="The final target column name, or None if no candidates found and no user selection"
    )
    problem_type_analysis: ProblemTypeAnalysis = Field(
        description="LLM-generated problem-type classification and project plan"
    )


class EDAStats(BaseModel):
    skewness: dict[str, float] = Field(
        description="Skewness per numeric column"
    )
    correlation_matrix: dict[str, dict[str, float]] = Field(
        description="Full Pearson correlation matrix, nested by column"
    )
    flagged_correlations: list[dict[str, Any]] = Field(
        description="Column pairs with |correlation| > 0.7, sorted descending"
    )
    distribution_stats: dict[str, dict[str, float]] = Field(
        description="Mean/median/std/min/max per numeric column"
    )
    class_balance: Optional[dict[str, Any]] = Field(
        default=None,
        description="Class counts/percentages if classification target provided, else None",
    )


class HistogramsResult(BaseModel):
    histograms: dict[str, str] = Field(
        description="Column name to Plotly JSON string"
    )
    skipped_columns: list[dict[str, Any]] = Field(
        description="Columns skipped (e.g. all-NaN), with reasons"
    )


class BoxplotsResult(BaseModel):
    boxplots: dict[str, str] = Field(
        description="Column name to Plotly JSON string"
    )
    skipped_columns: list[dict[str, Any]] = Field(
        description="Columns skipped (e.g. all-NaN), with reasons"
    )


class EDAInsights(BaseModel):
    insights: list[str] = Field(
        description="Specific, quantitative LLM-generated insights"
    )
    summary: str = Field(
        description="1-2 sentence overview tying insights together"
    )


class CollinearPair(BaseModel):
    col_a: str
    col_b: str
    correlation: float = Field(description="Pearson correlation between col_a and col_b")


class FeatureEngineeringResponse(BaseModel):
    """Response from /run-feature-engineering and /apply-collinearity-drop."""
    encoding_map: dict[str, str] = Field(description="Column name to transformation applied: onehot, label, ordinal, scaled, or excluded")
    collinear_pairs: list[CollinearPair] = Field(description="Column pairs with |correlation| > threshold; flagged only, not auto-dropped")
    excluded_columns: list[str] = Field(description="ID-like columns, target column, and caller-supplied exclusions")
    artifact_path: str = Field(description="Path to the saved feature-engineered CSV for this version")
    pipeline_path: str = Field(description="Path to the pickled sklearn Pipeline object, reusable in Week 5")


class ApplyCollinearityDropRequest(BaseModel):
    artifact_path: str = Field(description="Path to an existing feature_engineered_vN.csv")
    columns_to_drop: list[str] = Field(description="Columns to remove from the feature matrix")


class EDAResponse(BaseModel):
    """Response from the run-eda endpoint."""
    stats: Optional[EDAStats] = Field(
        default=None,
        description="Computed statistical summary, or None if computation failed",
    )
    histograms: Optional[HistogramsResult] = Field(
        default=None,
        description="Histogram charts, or None if generation failed",
    )
    correlation_heatmap: Optional[str] = Field(
        default=None,
        description="Correlation heatmap as Plotly JSON, or None if not generated or generation failed",
    )
    boxplots: Optional[BoxplotsResult] = Field(
        default=None,
        description="Box plot charts, or None if generation failed",
    )
    target_distribution: Optional[str] = Field(
        default=None,
        description="Target distribution chart as Plotly JSON, or None if target/problem_type not both provided, or generation failed",
    )
    insights: Optional[EDAInsights] = Field(
        default=None,
        description="LLM-generated insights, or None if stats unavailable or generation failed",
    )
    excluded_columns: list[str] = Field(
        description="ID-like columns excluded from EDA"
    )
    errors: dict[str, str] = Field(
        description="Part name to error message, only populated for parts that failed",
    )
    artifact_path: str = Field(
        description="Path to the saved EDA bundle JSON",
    )


app = FastAPI(
    title="DataArc API",
    description="Autonomous data-scientist pipeline – dataset analysis",
    version="0.2.0",
)

@app.post("/clean-dataset", response_model=CleanDatasetResponse)
async def clean_dataset_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    cap_target: bool = Form(False),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.filename}'. Only .csv files are accepted.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {exc}",
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV has headers but no data rows. Please upload a dataset with at least one row.",
        )

    if target_column is not None and target_column.strip() == "":
        target_column = None

    if target_column is not None:
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' was not found "
                    f"in the dataset. Available columns: {sorted(str(c) for c in df.columns)}"
                ),
            )

    result = clean_dataset(
        df,
        target_column=target_column,
        cap_target=cap_target,
        artifacts_dir="artifacts",
    )

    return CleanDatasetResponse(
        artifact_path=result["artifact_path"],
        changelog_path=result["changelog_path"],
        summary=CleaningSummary(
            duplicate_removal=result["summary"]["duplicate_removal"],
            missing_value_imputation=result["summary"]["missing_value_imputation"],
            outlier_handling=result["summary"]["outlier_handling"],
            dtype_fixing=result["summary"]["dtype_fixing"],
            llm_explanation=result["explanation"],
        ),
    )


@app.post("/profile-dataset", response_model=DatasetProfile)
async def profile_dataset_endpoint(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.filename}'. Only .csv files are accepted.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {exc}",
        )

    result = profile_dataset(df)

    if result["row_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV has headers but no data rows. Please upload a dataset with at least one row.",
        )

    return DatasetProfile(**result)


@app.post("/analyze-dataset", response_model=AnalyzeDatasetResponse)
async def analyze_dataset(
    file: UploadFile = File(...),
    user_selected_target: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.filename}'. Only .csv files are accepted.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {exc}",
        )

    profile = profile_dataset(df)

    if profile["row_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV has headers but no data rows. Please upload a dataset with at least one row.",
        )

    target_candidates = detect_target_candidates(df)

    if user_selected_target is not None and user_selected_target.strip() == "":
        user_selected_target = None

    if user_selected_target is not None:
        if user_selected_target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Selected target column '{user_selected_target}' was not found "
                    f"in the dataset. Available columns: {sorted(str(c) for c in df.columns)}"
                ),
            )
        selected_target = user_selected_target
        target_source = "user_provided"
        llm_candidates = [
            {
                "column_name": selected_target,
                "confidence_score": 100,
                "reasons": ["User-selected target column"],
            }
        ]
    else:
        selected_target = (
            target_candidates[0]["column_name"] if target_candidates else None
        )
        target_source = "auto_detected"
        llm_candidates = target_candidates

    problem_type_analysis = classify_problem_type(profile, llm_candidates)

    saved_profile_payload = {
        "profile": profile,
        "target_candidates": target_candidates,
        "target_source": target_source,
        "selected_target": selected_target,
        "problem_type_analysis": problem_type_analysis,
    }
    save_artifact(saved_profile_payload, "artifacts", "dataset_profile", "json")

    return AnalyzeDatasetResponse(
        profile=DatasetProfile(**profile),
        target_candidates=[TargetCandidateResponse(**c) for c in target_candidates],
        target_source=target_source,
        selected_target=selected_target,
        problem_type_analysis=ProblemTypeAnalysis(**problem_type_analysis),
    )


@app.post("/run-eda", response_model=EDAResponse)
async def run_eda_endpoint(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    problem_type: Optional[str] = Form(None),
    cleaning_changelog_path: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.filename}'. Only .csv files are accepted.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {exc}",
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV has headers but no data rows. Please upload a dataset with at least one row.",
        )

    if target_column is not None and target_column.strip() == "":
        target_column = None

    if problem_type is not None and problem_type.strip() == "":
        problem_type = None

    if target_column is not None:
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column '{target_column}' was not found "
                    f"in the dataset. Available columns: {sorted(str(c) for c in df.columns)}"
                ),
            )

    cleaning_changelog: dict | None = None
    if cleaning_changelog_path is not None and cleaning_changelog_path.strip():
        try:
            with open(cleaning_changelog_path) as f:
                cleaning_changelog = json.load(f)
        except Exception as exc:
            logging.warning(
                "Failed to load cleaning_changelog_path=%s: %s",
                cleaning_changelog_path, exc,
            )

    result = run_eda(
        df,
        target_column=target_column,
        problem_type=problem_type,
        cleaning_changelog=cleaning_changelog,
    )

    return EDAResponse(**result)


def _next_artifact_version(artifacts_dir: str, prefix: str) -> int:
    """Return the next unused version number for a given prefix in artifacts_dir."""
    os.makedirs(artifacts_dir, exist_ok=True)
    existing = [f for f in os.listdir(artifacts_dir) if f.startswith(prefix) and f.endswith(".csv")]
    versions = []
    for f in existing:
        try:
            v = int(f.replace(prefix, "").replace(".csv", "").lstrip("v"))
            versions.append(v)
        except (ValueError, IndexError):
            continue
    return max(versions, default=0) + 1


def _next_csv_version_from_path(artifact_path: str) -> int:
    """Determine the next version number based on an existing artifact_path."""
    basename = os.path.basename(artifact_path)
    for prefix in ("feature_engineered_v",):
        if basename.startswith(prefix) and basename.endswith(".csv"):
            try:
                v = int(basename.replace(prefix, "").replace(".csv", ""))
                return v + 1
            except (ValueError, IndexError):
                continue
    return 1


@app.post("/run-feature-engineering", response_model=FeatureEngineeringResponse)
async def run_feature_engineering_endpoint(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    problem_type: Optional[str] = Form(...),
    exclude_columns: Optional[str] = Form(None),
    ordinal_columns: Optional[str] = Form(None),
    cleaning_changelog_path: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.filename}'. Only .csv files are accepted.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {exc}",
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV has headers but no data rows. Please upload a dataset with at least one row.",
        )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column '{target_column}' not found in cleaned dataset. "
                f"Available columns: {sorted(str(c) for c in df.columns)}"
            ),
        )

    exclude_list: list[str] | None = None
    if exclude_columns and exclude_columns.strip():
        try:
            exclude_list = json.loads(exclude_columns)
            if not isinstance(exclude_list, list):
                raise ValueError("exclude_columns must be a JSON list")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid exclude_columns JSON: {exc}",
            )

    ordinal_map: dict[str, list[str]] | None = None
    if ordinal_columns and ordinal_columns.strip():
        try:
            ordinal_map = json.loads(ordinal_columns)
            if not isinstance(ordinal_map, dict):
                raise ValueError("ordinal_columns must be a JSON dict")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid ordinal_columns JSON: {exc}",
            )

    cleaning_changelog: dict | None = None
    if cleaning_changelog_path and cleaning_changelog_path.strip():
        try:
            with open(cleaning_changelog_path) as f:
                cleaning_changelog = json.load(f)
        except Exception as exc:
            logging.warning(
                "Failed to load cleaning_changelog_path=%s: %s",
                cleaning_changelog_path, exc,
            )

    additional_excludes = list(exclude_list or [])
    if cleaning_changelog:
        outlier_skipped = cleaning_changelog.get("outlier_handling", {}).get(
            "columns_skipped_target_protected", []
        )
        for item in outlier_skipped:
            if isinstance(item, dict) and "column" in item:
                additional_excludes.append(item["column"])
            elif isinstance(item, str):
                additional_excludes.append(item)

    additional_excludes = list(dict.fromkeys(additional_excludes))

    start = time.perf_counter()
    try:
        result = build_feature_pipeline(
            df,
            target_column=target_column,
            problem_type=problem_type,
            exclude_columns=additional_excludes,
            ordinal_columns=ordinal_map,
        )
    except Exception as exc:
        duration = time.perf_counter() - start
        log_agent_run(
            agent_name="feature_engineering",
            inputs_summary={
                "target_column": target_column,
                "problem_type": problem_type,
            },
            outputs_summary={},
            duration_seconds=duration,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc))

    duration = time.perf_counter() - start

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    version = get_next_version(artifacts_dir, "feature_engineered")
    csv_path = os.path.join(artifacts_dir, f"feature_engineered_v{version}.csv")
    json_path = os.path.join(artifacts_dir, f"feature_engineered_v{version}.json")
    pkl_path = os.path.join(artifacts_dir, f"pipeline_v{version}.pkl")
    meta_json_path = os.path.join(artifacts_dir, f"pipeline_v{version}.json")

    result["transformed_df"].to_csv(csv_path, index=False)
    with open(pkl_path, "wb") as f:
        pickle.dump(result["pipeline"], f)

    collinear_pairs_dicts = [
        {"col_a": p["col_a"], "col_b": p["col_b"], "correlation": p["correlation"]}
        for p in result["collinear_pairs"]
    ]
    fe_summary = {
        "columns": list(result["transformed_df"].columns),
        "dtypes": {str(c): str(t) for c, t in result["transformed_df"].dtypes.items()},
        "target_column": target_column,
        "problem_type": problem_type,
        "encoding_map": result["encoding_map"],
        "collinear_pairs": collinear_pairs_dicts,
        "excluded_columns": result["excluded_columns"],
    }
    with open(json_path, "w") as f:
        json.dump(fe_summary, f, indent=2, default=str)
    with open(meta_json_path, "w") as f:
        json.dump(fe_summary, f, indent=2, default=str)

    log_agent_run(
        agent_name="feature_engineering",
        inputs_summary={
            "target_column": target_column,
            "problem_type": problem_type,
            "rows": len(df),
            "columns": len(df.columns),
        },
        outputs_summary={
            "encoding_map": result["encoding_map"],
            "collinear_pairs_count": len(result["collinear_pairs"]),
            "excluded_columns": result["excluded_columns"],
            "artifact_path": csv_path,
            "pipeline_path": pkl_path,
        },
        duration_seconds=duration,
    )

    return FeatureEngineeringResponse(
        encoding_map=result["encoding_map"],
        collinear_pairs=[CollinearPair(**p) for p in result["collinear_pairs"]],
        excluded_columns=result["excluded_columns"],
        artifact_path=csv_path,
        pipeline_path=pkl_path,
    )


@app.post("/apply-collinearity-drop", response_model=FeatureEngineeringResponse)
async def apply_collinearity_drop_endpoint(
    request: ApplyCollinearityDropRequest = Body(...),
):
    start = time.perf_counter()

    if not os.path.isfile(request.artifact_path):
        raise HTTPException(
            status_code=400,
            detail=f"Feature-engineered artifact not found: '{request.artifact_path}'",
        )

    try:
        df = pd.read_csv(request.artifact_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse feature-engineered CSV: {exc}",
        )

    missing_cols = [c for c in request.columns_to_drop if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in artifact: {missing_cols}",
        )

    next_version = _next_csv_version_from_path(request.artifact_path)
    artifacts_dir = os.path.dirname(request.artifact_path) or "artifacts"
    csv_path = os.path.join(artifacts_dir, f"feature_engineered_v{next_version}.csv")
    pkl_path = os.path.join(artifacts_dir, f"pipeline_v{next_version}.pkl")

    df_dropped = df.drop(columns=request.columns_to_drop)
    df_dropped.to_csv(csv_path, index=False)

    orig_pipeline_version = None
    basename = os.path.basename(request.artifact_path)
    for prefix in ("feature_engineered_v",):
        if basename.startswith(prefix) and basename.endswith(".csv"):
            try:
                orig_pipeline_version = int(basename.replace(prefix, "").replace(".csv", ""))
            except (ValueError, IndexError):
                pass

    if orig_pipeline_version is not None:
        orig_pkl_path = os.path.join(artifacts_dir, f"pipeline_v{orig_pipeline_version}.pkl")
        if os.path.isfile(orig_pkl_path):
            with open(orig_pkl_path, "rb") as f:
                orig_pipeline = pickle.load(f)
            with open(pkl_path, "wb") as f:
                pickle.dump(orig_pipeline, f)

    meta_json_path = os.path.join(artifacts_dir, f"pipeline_v{next_version}.json")
    fe_json_path = os.path.join(artifacts_dir, f"feature_engineered_v{next_version}.json")
    meta_path_for_orig = os.path.join(artifacts_dir, f"pipeline_v{orig_pipeline_version}.json") if orig_pipeline_version else None

    orig_meta = {}
    if meta_path_for_orig and os.path.isfile(meta_path_for_orig):
        with open(meta_path_for_orig) as f:
            orig_meta = json.load(f)

    new_encoding_map = dict(orig_meta.get("encoding_map", {}))
    for col in request.columns_to_drop:
        new_encoding_map[col] = "dropped"

    new_meta = {
        "columns": list(df_dropped.columns),
        "dtypes": {str(c): str(t) for c, t in df_dropped.dtypes.items()},
        "target_column": orig_meta.get("target_column"),
        "problem_type": orig_meta.get("problem_type"),
        "encoding_map": new_encoding_map,
        "collinear_pairs": orig_meta.get("collinear_pairs", []),
        "excluded_columns": orig_meta.get("excluded_columns", []),
    }
    with open(meta_json_path, "w") as f:
        json.dump(new_meta, f)
    with open(fe_json_path, "w") as f:
        json.dump(new_meta, f)

    collinear_pairs = [
        CollinearPair(**p) for p in orig_meta.get("collinear_pairs", [])
    ]
    excluded_columns = list(dict.fromkeys(orig_meta.get("excluded_columns", [])))

    duration = time.perf_counter() - start
    log_agent_run(
        agent_name="collinearity_drop",
        inputs_summary={
            "artifact_path": request.artifact_path,
            "columns_to_drop": request.columns_to_drop,
        },
        outputs_summary={
            "artifact_path": csv_path,
            "columns_dropped": request.columns_to_drop,
        },
        duration_seconds=duration,
    )

    return FeatureEngineeringResponse(
        encoding_map=new_encoding_map,
        collinear_pairs=collinear_pairs,
        excluded_columns=excluded_columns,
        artifact_path=csv_path,
        pipeline_path=pkl_path if os.path.isfile(pkl_path) else "",
    )


class CandidateModelResponse(BaseModel):
    model_name: str = Field(description="Name of candidate ML model")
    reasoning: str = Field(description="Why this model fits the dataset")


class MLPlanResponse(BaseModel):
    problem_type: str = Field(
        description="Confirmed problem type: classification or regression"
    )
    confirmation_reasoning: str = Field(
        description="1-2 sentence justification for the problem type"
    )
    recommended_metric: str = Field(
        description="Primary evaluation metric suited for the dataset"
    )
    metric_reasoning: str = Field(
        description="1-2 sentence justification for the selected metric"
    )
    candidate_models: list[CandidateModelResponse] = Field(
        description="4-6 candidate models with reasoning"
    )


class PlanTrainingRequest(BaseModel):
    state: Optional[dict[str, Any]] = Field(
        default=None, description="Shared pipeline state dictionary from earlier stages"
    )
    target_column: Optional[str] = Field(
        default=None, description="Target column name"
    )
    problem_type: Optional[str] = Field(
        default=None, description="Initial problem type guess"
    )
    profile: Optional[dict[str, Any]] = Field(
        default=None, description="Dataset profile dictionary"
    )
    class_balance: Optional[dict[str, Any]] = Field(
        default=None, description="Class balance dictionary"
    )
    feature_engineering_summary: Optional[dict[str, Any]] = Field(
        default=None, description="Feature engineering metadata"
    )


@app.post("/plan-training", response_model=MLPlanResponse)
@app.post("/plan-ml", response_model=MLPlanResponse)
async def plan_training_endpoint(
    request: dict[str, Any] = Body(...),
):
    try:
        if "state" in request and isinstance(request["state"], dict):
            state_dict = dict(request["state"])
        else:
            state_dict = dict(request)

        agent = MLPlanningAgent()
        result_state = agent.run(state_dict)
        plan = result_state.get("ml_plan")
        if not plan:
            raise ValueError("MLPlanningAgent did not produce an 'ml_plan' in state.")

        return MLPlanResponse(
            problem_type=plan["problem_type"],
            confirmation_reasoning=plan["confirmation_reasoning"],
            recommended_metric=plan["recommended_metric"],
            metric_reasoning=plan["metric_reasoning"],
            candidate_models=[
                CandidateModelResponse(**m) for m in plan["candidate_models"]
            ],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

