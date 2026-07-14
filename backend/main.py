from __future__ import annotations
import io
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from agents.dataset_understanding import (
    classify_problem_type,
    detect_target_candidates,
    profile_dataset,
)

class ColumnProfile(BaseModel):
    """Statistics for a single column."""
    name: str = Field(description="Column name")
    dtype: str = Field(description="Pandas dtype string, e.g. 'int64'")
    missing_count: int = Field(description="Number of missing (NaN) values")
    missing_percentage: float = Field(description="Percentage of missing values, rounded to 2 decimals")
    unique_count: int = Field(description="Number of unique values")


class DatasetProfile(BaseModel):
    """Profiling result for an uploaded dataset."""
    row_count: int = Field(description="Total number of rows")
    column_count: int = Field(description="Total number of columns")
    columns: list[ColumnProfile] = Field(description="Per-column statistics")
    duplicate_row_count: int = Field(description="Number of duplicate rows")
    duplicate_row_percentage: float = Field(description="Percentage of duplicate rows, rounded to 2 decimals")


class TargetCandidateResponse(BaseModel):
    """A single target column candidate with heuristic scoring."""
    column_name: str = Field(description="Name of the candidate column")
    confidence_score: float = Field(description="Heuristic confidence score (0–100)")
    reasons: list[str] = Field(description="Plain-English reasons the column was flagged or excluded")


class ProblemTypeAnalysis(BaseModel):
    """LLM-generated problem-type classification and project plan."""
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
    """Combined response from the /analyze-dataset endpoint."""
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


app = FastAPI(
    title="DataArc API",
    description="Autonomous data-scientist pipeline – dataset analysis",
    version="0.2.0",
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
        # User-provided target
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
        # Auto-detection
        selected_target = (
            target_candidates[0]["column_name"] if target_candidates else None
        )
        target_source = "auto_detected"
        llm_candidates = target_candidates

    problem_type_analysis = classify_problem_type(profile, llm_candidates)

    return AnalyzeDatasetResponse(
        profile=DatasetProfile(**profile),
        target_candidates=[TargetCandidateResponse(**c) for c in target_candidates],
        target_source=target_source,
        selected_target=selected_target,
        problem_type_analysis=ProblemTypeAnalysis(**problem_type_analysis),
    )
