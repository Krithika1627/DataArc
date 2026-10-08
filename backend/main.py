import io
import json
import logging
import os
import pickle
import time
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body, Response, Request
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from agents.dataset_understanding import (
    classify_problem_type,
    compute_confidence_score,
    detect_target_candidates,
    determine_heuristic_problem_type,
    profile_dataset,
)
from agents.data_cleaning import clean_dataset
from agents.eda_agent import run_eda
from agents.feature_engineering_agent import build_feature_pipeline
from agents.ml_planning_agent import MLPlanningAgent
from agents.training_agent import TrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.report_agent import ReportAgent
from agents.logging_utils import log_agent_run
from agents.versioning_utils import (
    get_dataset_artifacts_dir,
    get_next_version,
    resolve_dataset_artifacts_dir,
    save_artifact,
)
import uuid

BASE_ARTIFACTS_DIR = os.getenv("DATAARC_ARTIFACTS_DIR", "artifacts")


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
    dataset_id: Optional[str] = Field(
        default=None, description="Dedicated dataset ID"
    )
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
    dataset_id: Optional[str] = Field(
        default=None, description="Dedicated dataset ID"
    )
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


class ConfidenceBreakdownItem(BaseModel):
    check: str = Field(description="Name of the confidence check")
    points_awarded: int = Field(description="Points awarded for this check (0, 20, 30)")
    reason: str = Field(description="Evidence or justification for points awarded")


class ConfidenceScoreResponse(BaseModel):
    total_score: int = Field(description="Total confidence score (0-100)")
    breakdown: list[ConfidenceBreakdownItem] = Field(description="Per-check points and reasons breakdown")


class AnalyzeDatasetResponse(BaseModel):
    dataset_id: Optional[str] = Field(
        default=None, description="Dedicated dataset ID"
    )
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
    confidence_score: Optional[ConfidenceScoreResponse] = Field(
        default=None, description="Deterministic confidence score breakdown"
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
    dataset_id: Optional[str] = Field(default=None, description="Dedicated dataset ID")
    encoding_map: dict[str, str] = Field(description="Column name to transformation applied: onehot, label, ordinal, scaled, or excluded")
    collinear_pairs: list[CollinearPair] = Field(description="Column pairs with |correlation| > threshold; flagged only, not auto-dropped")
    excluded_columns: list[str] = Field(description="ID-like columns, target column, and caller-supplied exclusions")
    artifact_path: str = Field(description="Path to the saved feature-engineered CSV for this version")
    pipeline_path: str = Field(description="Path to the pickled sklearn Pipeline object, reusable in Week 5")


class ApplyCollinearityDropRequest(BaseModel):
    dataset_id: Optional[str] = Field(default=None, description="Dedicated dataset ID")
    artifact_path: str = Field(description="Path to an existing feature_engineered_vN.csv")
    columns_to_drop: list[str] = Field(description="Columns to remove from the feature matrix")


class EDAResponse(BaseModel):
    """Response from the run-eda endpoint."""
    dataset_id: Optional[str] = Field(default=None, description="Dedicated dataset ID")
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


limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="DataArc API",
    description="Autonomous data-scientist pipeline – dataset analysis",
    version="0.2.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.on_event("startup")
async def on_startup():
    try:
        from agents.versioning_utils import get_db_engine
        eng = get_db_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Successfully connected to Neon / Postgres database.")
    except Exception as exc:
        logging.warning("Database connection check on startup: %s", exc)


@app.get("/health")
async def health_endpoint():
    """Startup and liveness health check testing database connectivity."""
    try:
        from agents.versioning_utils import get_db_engine
        eng = get_db_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as exc:
        logging.error("Health check database failure: %s", exc)
        return {"status": "unhealthy", "database": f"connection failed: {exc}"}


@app.post("/clean-dataset", response_model=CleanDatasetResponse)
@limiter.limit("10/minute")
async def clean_dataset_endpoint(
    request: Request,
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    cap_target: bool = Form(False),
    dataset_id: Optional[str] = Form(None),
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

    clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
        dataset_id=dataset_id, create_if_missing=True
    )

    result = clean_dataset(
        df,
        target_column=target_column,
        cap_target=cap_target,
        artifacts_dir=artifacts_dir,
    )

    return CleanDatasetResponse(
        dataset_id=clean_id,
        artifact_path=str(result["artifact_path"]),
        changelog_path=str(result["changelog_path"]),
        summary=CleaningSummary(
            duplicate_removal=result["summary"]["duplicate_removal"],
            missing_value_imputation=result["summary"]["missing_value_imputation"],
            outlier_handling=result["summary"]["outlier_handling"],
            dtype_fixing=result["summary"]["dtype_fixing"],
            llm_explanation=result["explanation"],
        ),
    )


@app.post("/profile-dataset", response_model=DatasetProfile)
async def profile_dataset_endpoint(
    file: UploadFile = File(...),
    dataset_id: Optional[str] = Form(None),
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

    result = profile_dataset(df)

    if result["row_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="CSV has headers but no data rows. Please upload a dataset with at least one row.",
        )

    clean_id, _ = resolve_dataset_artifacts_dir(
        dataset_id=dataset_id, create_if_missing=True
    )
    result["dataset_id"] = clean_id

    return DatasetProfile(**result)


@app.post("/analyze-dataset", response_model=AnalyzeDatasetResponse)
@limiter.limit("10/minute")
async def analyze_dataset(
    request: Request,
    file: UploadFile = File(...),
    user_selected_target: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None),
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
    all_candidates = detect_target_candidates(df, return_all=True)

    if selected_target:
        heuristic_type = determine_heuristic_problem_type(df, selected_target)
        confidence_score = compute_confidence_score(
            selected_target=selected_target,
            target_candidates=all_candidates,
            df=df,
            llm_problem_type=problem_type_analysis.get("problem_type", "unclear"),
            heuristic_problem_type=heuristic_type,
        )
    else:
        confidence_score = compute_confidence_score(
            selected_target="",
            target_candidates=[],
            df=df,
            llm_problem_type="unclear",
            heuristic_problem_type="unclear",
        )

    clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
        dataset_id=dataset_id, create_if_missing=True
    )
    profile["dataset_id"] = clean_id
    profile["filename"] = file.filename

    saved_profile_payload = {
        "dataset_id": clean_id,
        "filename": file.filename,
        "profile": profile,
        "target_candidates": target_candidates,
        "target_source": target_source,
        "selected_target": selected_target,
        "problem_type_analysis": problem_type_analysis,
        "confidence_score": confidence_score,
    }
    save_artifact(saved_profile_payload, artifacts_dir, "dataset_profile", "json")

    return AnalyzeDatasetResponse(
        dataset_id=clean_id,
        profile=DatasetProfile(**profile),
        target_candidates=[TargetCandidateResponse(**c) for c in target_candidates],
        target_source=target_source,
        selected_target=selected_target,
        problem_type_analysis=ProblemTypeAnalysis(**problem_type_analysis),
        confidence_score=ConfidenceScoreResponse(
            total_score=confidence_score["total_score"],
            breakdown=[ConfidenceBreakdownItem(**b) for b in confidence_score["breakdown"]],
        ),
    )


@app.post("/run-eda", response_model=EDAResponse)
@limiter.limit("10/minute")
async def run_eda_endpoint(
    request: Request,
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    problem_type: Optional[str] = Form(None),
    cleaning_changelog_path: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None),
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

    clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
        dataset_id=dataset_id, create_if_missing=True
    )

    result = run_eda(
        df,
        target_column=target_column,
        problem_type=problem_type,
        cleaning_changelog=cleaning_changelog,
        artifacts_dir=artifacts_dir,
    )
    result["dataset_id"] = clean_id

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
@limiter.limit("10/minute")
async def run_feature_engineering_endpoint(
    request: Request,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    problem_type: Optional[str] = Form(...),
    exclude_columns: Optional[str] = Form(None),
    ordinal_columns: Optional[str] = Form(None),
    cleaning_changelog_path: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None),
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

    clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
        dataset_id=dataset_id, create_if_missing=True
    )

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
        "dataset_id": clean_id,
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

    save_artifact(fe_summary, artifacts_dir, "feature_engineered", "json", version=version)
    save_artifact(result["transformed_df"], artifacts_dir, "feature_engineered", "csv", version=version)

    log_agent_run(
        agent_name="feature_engineering",
        inputs_summary={
            "dataset_id": clean_id,
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
        dataset_id=clean_id,
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

    explicit_artifacts_dir = os.path.dirname(request.artifact_path)
    clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
        dataset_id=request.dataset_id,
        create_if_missing=False,
        explicit_artifacts_dir=explicit_artifacts_dir if not request.dataset_id else None,
    )

    next_version = _next_csv_version_from_path(request.artifact_path)
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
        "dataset_id": clean_id,
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

    save_artifact(new_meta, artifacts_dir, "feature_engineered", "json", version=next_version)
    save_artifact(df_dropped, artifacts_dir, "feature_engineered", "csv", version=next_version)


    collinear_pairs = [
        CollinearPair(**p) for p in orig_meta.get("collinear_pairs", [])
    ]
    excluded_columns = list(dict.fromkeys(orig_meta.get("excluded_columns", [])))

    duration = time.perf_counter() - start
    log_agent_run(
        agent_name="collinearity_drop",
        inputs_summary={
            "dataset_id": clean_id,
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
        dataset_id=clean_id,
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
    dataset_id: Optional[str] = Field(
        default=None, description="Dedicated dataset ID"
    )
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
    artifact_path: Optional[str] = Field(
        default=None, description="Path to the saved ml_plan_v{N}.json artifact"
    )


class PlanTrainingRequest(BaseModel):
    dataset_id: Optional[str] = Field(
        default=None, description="Dedicated dataset ID"
    )
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


@app.post("/plan-ml", response_model=MLPlanResponse)
@limiter.limit("10/minute")
async def plan_training_endpoint(
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
):
    try:
        if "state" in payload and isinstance(payload["state"], dict):
            state_dict = dict(payload["state"])
        else:
            state_dict = dict(payload)

        dataset_id = state_dict.get("dataset_id") or payload.get("dataset_id")
        explicit_artifacts_dir = state_dict.get("artifacts_dir") or payload.get("artifacts_dir")
        clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
            dataset_id=dataset_id,
            create_if_missing=False,
            explicit_artifacts_dir=explicit_artifacts_dir,
        )
        state_dict["dataset_id"] = clean_id
        state_dict["artifacts_dir"] = artifacts_dir

        agent = MLPlanningAgent()
        result_state = agent.run(state_dict)
        plan = result_state.get("ml_plan")
        if not plan:
            raise ValueError("MLPlanningAgent did not produce an 'ml_plan' in state.")

        return MLPlanResponse(
            dataset_id=clean_id,
            problem_type=plan["problem_type"],
            confirmation_reasoning=plan["confirmation_reasoning"],
            recommended_metric=plan["recommended_metric"],
            metric_reasoning=plan["metric_reasoning"],
            candidate_models=[
                CandidateModelResponse(**m) for m in plan["candidate_models"]
            ],
            artifact_path=str(result_state.get("ml_plan_artifact_path")) if result_state.get("ml_plan_artifact_path") is not None else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


class ModelTrainingResultResponse(BaseModel):
    model_name: str = Field(description="Name of the candidate model")
    status: str = Field(description="One of 'success', 'failed', 'skipped_unrecognized'")
    recommended_metric: str = Field(description="Name of the recommended metric")
    score: Optional[float] = Field(default=None, description="Primary score for recommended metric")
    metrics: Optional[dict[str, float]] = Field(default=None, description="Computed metrics dictionary")
    training_time_seconds: float = Field(description="Training duration in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if model failed or was skipped")


class TrainingSummaryResponse(BaseModel):
    total_models: int = Field(description="Total candidate models processed")
    succeeded: int = Field(description="Number of models trained successfully")
    failed: int = Field(description="Number of models that failed during fit/evaluation")
    skipped_unrecognized: int = Field(description="Number of models skipped as unrecognized")


class TrainModelsResponse(BaseModel):
    dataset_id: Optional[str] = Field(default=None, description="Dedicated dataset ID")
    comparison_table: list[ModelTrainingResultResponse] = Field(description="Ranked list of model training results")
    artifact_path: str = Field(description="Path to saved training results JSON artifact")
    summary: TrainingSummaryResponse = Field(description="Overall training run summary")

@app.post("/train-model", response_model=TrainModelsResponse)
@limiter.limit("10/minute")
async def train_models_endpoint(
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
):
    try:
        if "state" in payload and isinstance(payload["state"], dict):
            state_dict = dict(payload["state"])
        else:
            state_dict = dict(payload)

        dataset_id = state_dict.get("dataset_id") or payload.get("dataset_id")
        explicit_artifacts_dir = state_dict.get("artifacts_dir") or payload.get("artifacts_dir")
        clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
            dataset_id=dataset_id,
            create_if_missing=False,
            explicit_artifacts_dir=explicit_artifacts_dir,
        )
        state_dict["dataset_id"] = clean_id
        state_dict["artifacts_dir"] = artifacts_dir

        agent = TrainingAgent()
        result_state = agent.run(state_dict)
        results = result_state.get("training_results", [])
        artifact_path = result_state.get("training_results_artifact_path", "")
        summary = result_state.get("training_summary", {})

        return TrainModelsResponse(
            dataset_id=clean_id,
            comparison_table=[ModelTrainingResultResponse(**r) for r in results],
            artifact_path=str(artifact_path),
            summary=TrainingSummaryResponse(**summary),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


class SkippedVisualizationItem(BaseModel):
    type: str = Field(description="Name/type of visualization skipped")
    reason: str = Field(description="Reason for skipping the visualization")


class EvaluationResponse(BaseModel):
    dataset_id: Optional[str] = Field(default=None, description="Dedicated dataset ID")
    winning_model_name: str = Field(description="Name of the winning model")
    problem_type: str = Field(description="Problem type (classification or regression)")
    recommended_metric: str = Field(description="Metric used for evaluation")
    winning_score: float = Field(description="Primary metric score achieved by the winning model")
    runner_up_model_name: Optional[str] = Field(default=None, description="Name of the runner-up model if available")
    runner_up_score: Optional[float] = Field(default=None, description="Score of the runner-up model if available")
    score_gap: Optional[float] = Field(default=None, description="Absolute difference between winner and runner-up scores")
    confusion_matrix: Optional[str] = Field(default=None, description="Plotly JSON string of confusion matrix")
    roc_curve: Optional[str] = Field(default=None, description="Plotly JSON string of ROC curve")
    feature_importance_chart: Optional[str] = Field(default=None, description="Plotly JSON string of feature importances")
    llm_explanation: str = Field(description="LLM-generated explanation of the evaluation results")
    skipped_visualizations: list[SkippedVisualizationItem] = Field(default_factory=list, description="List of skipped visualizations with reasons")
    artifact_path: str = Field(description="Path to saved evaluation bundle JSON artifact")


@app.post("/evaluate-model", response_model=EvaluationResponse)
@limiter.limit("10/minute")
async def evaluate_model_endpoint(
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
):
    try:
        if "state" in payload and isinstance(payload["state"], dict):
            state_dict = dict(payload["state"])
        else:
            state_dict = dict(payload)

        dataset_id = state_dict.get("dataset_id") or payload.get("dataset_id")
        explicit_artifacts_dir = state_dict.get("artifacts_dir") or payload.get("artifacts_dir")
        clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
            dataset_id=dataset_id,
            create_if_missing=False,
            explicit_artifacts_dir=explicit_artifacts_dir,
        )
        state_dict["dataset_id"] = clean_id
        state_dict["artifacts_dir"] = artifacts_dir

        agent = EvaluationAgent()
        result_state = agent.run(state_dict)
        bundle = result_state.get("evaluation_bundle")
        if not bundle:
            raise ValueError("EvaluationAgent did not produce an 'evaluation_bundle' in state.")

        bundle["dataset_id"] = clean_id
        return EvaluationResponse(**bundle)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.post("/generate-report")
@limiter.limit("10/minute")
async def generate_report_endpoint(
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
):
    try:
        if "state" in payload and isinstance(payload["state"], dict):
            state_dict = dict(payload["state"])
        else:
            state_dict = dict(payload)

        dataset_id = state_dict.get("dataset_id") or payload.get("dataset_id")
        explicit_artifacts_dir = state_dict.get("artifacts_dir") or payload.get("artifacts_dir")
        clean_id, artifacts_dir = resolve_dataset_artifacts_dir(
            dataset_id=dataset_id,
            create_if_missing=False,
            explicit_artifacts_dir=explicit_artifacts_dir,
        )
        state_dict["dataset_id"] = clean_id
        state_dict["artifacts_dir"] = artifacts_dir

        agent = ReportAgent()
        result_state = agent.run(state_dict)
        
        pdf_bytes = result_state.get("report_pdf_bytes")
        if not pdf_bytes:
            from agents.versioning_utils import get_latest_artifact
            try:
                pdf_bytes = get_latest_artifact(clean_id, "report")
            except Exception:
                pdf_bytes = None

        if not pdf_bytes:
            report_path = result_state.get("report_pdf_path")
            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()

        if not pdf_bytes:
            raise ValueError("ReportAgent did not produce a valid PDF report.")

        filename = f"report_{clean_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.get("/history")
async def list_history_endpoint():
    """Lists all previous dataset runs saved in the database with their summaries."""
    from agents.versioning_utils import list_all_dataset_runs
    try:
        runs = list_all_dataset_runs()
        return {"runs": runs, "total": len(runs)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {exc}")


@app.get("/history/{dataset_id}")
async def get_history_endpoint(dataset_id: str):
    """Retrieves all stored artifacts for a given dataset_id."""
    from agents.versioning_utils import get_dataset_history
    try:
        data = get_dataset_history(dataset_id)
        return data
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Failed to load dataset run: {exc}")


@app.get("/report/{dataset_id}")
async def get_report_pdf_endpoint(dataset_id: str):
    """Downloads the stored PDF report for a given dataset_id."""
    from agents.versioning_utils import get_latest_artifact
    try:
        pdf_bytes = get_latest_artifact(dataset_id, "report", extension="pdf")
        if not pdf_bytes:
            raise FileNotFoundError("PDF report not found")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=report_{dataset_id}.pdf"},
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Report not found for dataset {dataset_id}: {exc}")


@app.get("/artifacts/{dataset_id}/cleaned")
async def get_cleaned_csv_endpoint(dataset_id: str):
    """Downloads the cleaned CSV dataset for a given dataset_id."""
    from agents.versioning_utils import get_latest_artifact
    try:
        data = get_latest_artifact(dataset_id, "cleaned", extension="csv")
        if isinstance(data, pd.DataFrame):
            csv_content = data.to_csv(index=False)
        elif isinstance(data, bytes):
            csv_content = data.decode("utf-8")
        elif isinstance(data, str):
            csv_content = data
        else:
            raise FileNotFoundError("Cleaned dataset not found as tabular data")
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_id}_cleaned.csv"},
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Cleaned dataset not found for dataset {dataset_id}: {exc}")


@app.get("/artifacts/{dataset_id}/feature-engineered")
async def get_feature_engineered_csv_endpoint(dataset_id: str):
    """Downloads the model-ready feature-engineered CSV dataset for a given dataset_id."""
    from agents.versioning_utils import get_latest_artifact
    try:
        data = get_latest_artifact(dataset_id, "feature_engineered", extension="csv")
        if isinstance(data, pd.DataFrame):
            csv_content = data.to_csv(index=False)
        elif isinstance(data, bytes):
            csv_content = data.decode("utf-8")
        elif isinstance(data, str):
            csv_content = data
        else:
            raise FileNotFoundError("Feature-engineered dataset not found as tabular data")
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_id}_feature_engineered.csv"},
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Feature-engineered dataset not found for dataset {dataset_id}: {exc}")






