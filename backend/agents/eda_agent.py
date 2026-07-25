from __future__ import annotations

from typing import Any

import pandas as pd

from logging_utils import with_agent_logging


@with_agent_logging("eda_compute_stats")
def compute_eda_stats(
    df: pd.DataFrame,
    target_column: str | None = None,
    problem_type: str | None = None,
) -> dict[str, Any]:
    """Compute exploratory data analysis statistics for a DataFrame."""
    if len(df) == 0:
        raise ValueError("DataFrame is empty. Cannot compute EDA stats on zero rows.")

    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' is not present in the DataFrame. "
            f"Available columns: {sorted(str(c) for c in df.columns)}"
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_df = df[numeric_cols]

    # Skewness per numeric column
    skewness: dict[str, float] = {}
    if not numeric_df.empty:
        skew_vals = numeric_df.skew(numeric_only=False)
        skewness = {
            str(col): round(float(val), 3)
            for col, val in skew_vals.items()
            if pd.notna(val)
        }

    # Correlation matrix + flagged pairs
    correlation_matrix: dict[str, dict[str, float]] = {}
    flagged_correlations: list[dict[str, Any]] = []

    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        correlation_matrix = {
            str(col): {str(c): round(float(v), 4) for c, v in row.items()}
            for col, row in corr.iterrows()
        }

        pairs: list[dict[str, Any]] = []
        cols_list = numeric_df.columns.tolist()
        for i, col_a in enumerate(cols_list):
            for j, col_b in enumerate(cols_list):
                if j <= i:
                    continue  
                val = corr.loc[col_a, col_b]
                if pd.notna(val) and abs(val) > 0.7:
                    pairs.append(
                        {
                            "col_a": str(col_a),
                            "col_b": str(col_b),
                            "correlation": round(float(val), 4),
                        }
                    )
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        flagged_correlations = pairs

    # Basic distribution stats
    distribution_stats: dict[str, dict[str, float]] = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().T
        for col in numeric_df.columns:
            col_str = str(col)
            row_data = desc.loc[col]
            distribution_stats[col_str] = {
                "mean": round(float(row_data["mean"]), 3),
                "median": round(float(numeric_df[col].median()), 3),
                "std": round(float(row_data["std"]), 3),
                "min": round(float(row_data["min"]), 3),
                "max": round(float(row_data["max"]), 3),
            }

    # Class balance (conditional)
    class_balance: dict[str, Any] | None = None
    if (
        target_column is not None
        and problem_type == "classification"
    ):
        counts = df[target_column].value_counts()
        percentages = df[target_column].value_counts(normalize=True) * 100
        class_balance = {
            "class_counts": {
                str(k): int(v) for k, v in counts.items()
            },
            "class_percentages": {
                str(k): round(float(v), 2) for k, v in percentages.items()
            },
        }

    return {
        "skewness": skewness,
        "correlation_matrix": correlation_matrix,
        "flagged_correlations": flagged_correlations,
        "distribution_stats": distribution_stats,
        "class_balance": class_balance,
    }

