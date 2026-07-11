from __future__ import annotations
import io
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from agents.dataset_understanding import profile_dataset

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

app = FastAPI(
    title="DataArc API",
    description="Autonomous data-scientist pipeline – dataset profiling",
    version="0.1.0",
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
