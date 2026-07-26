from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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


@with_agent_logging("eda_histogram_generation")
def generate_histograms(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Generate Plotly histograms for each numeric column in the DataFrame."""
    if len(df) == 0:
        raise ValueError(
            "DataFrame is empty. Cannot generate histograms on zero rows."
        )

    exclude = set(exclude_columns or [])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    histograms: dict[str, str] = {}
    skipped: list[dict[str, str]] = []

    for col in numeric_cols:
        col_str = str(col)
        if col_str in exclude:
            continue

        series = df[col].dropna()

        if len(series) == 0:
            skipped.append({"column": col_str, "reason": "Column is entirely NaN"})
            continue

        unique_vals = series.nunique()

        if unique_vals <= 1:
            # Constant column: single bar
            fig = px.histogram(
                series,
                nbins=1,
                title=col_str,
            )
        elif unique_vals < 10:
            # Few unique values: reduce bins to match
            fig = px.histogram(
                series,
                nbins=unique_vals,
                title=col_str,
            )
        else:
            fig = px.histogram(
                series,
                title=col_str,
            )

        fig.update_layout(
            xaxis_title=col_str,
            yaxis_title="Count",
            showlegend=False,
        )

        histograms[col_str] = fig.to_json()

    return {
        "histograms": histograms,
        "skipped_columns": skipped,
    }


@with_agent_logging("eda_correlation_heatmap")
def generate_correlation_heatmap(correlation_matrix: dict) -> str:
    """Convert a correlation matrix (nested dict) into a Plotly heatmap JSON string."""
    if not isinstance(correlation_matrix, dict) or len(correlation_matrix) == 0:
        raise ValueError(
            "Correlation matrix is empty or not a dict. "
            "Cannot generate a heatmap without numeric columns to correlate."
        )

    columns = list(correlation_matrix.keys())
    for col_a in columns:
        row = correlation_matrix[col_a]
        if not isinstance(row, dict):
            raise ValueError(
                f"Correlation matrix is malformed: value for column '{col_a}' "
                f"is {type(row).__name__}, expected a dict."
            )
        for col_b in columns:
            if col_b not in row:
                raise ValueError(
                    f"Correlation matrix is asymmetric: column '{col_a}' has "
                    f"{len(row)} entries but column '{col_b}' is missing."
                )
            val = row[col_b]
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Correlation matrix has non-numeric value at "
                    f"[{col_a}][{col_b}]: {val!r}"
                )

    z = [
        [correlation_matrix[col_a][col_b] for col_b in columns]
        for col_a in columns
    ]
    text = [
        [f"{correlation_matrix[col_a][col_b]:.2f}" for col_b in columns]
        for col_a in columns
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=columns,
            y=columns,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="%{x} vs %{y}: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="",
        yaxis_title="",
        width=600,
        height=600,
        xaxis=dict(side="bottom"),
    )

    return fig.to_json()


@with_agent_logging("eda_target_distribution")
def generate_target_distribution(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
) -> str:
    """Generate a Plotly chart for the target variable distribution."""
    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' is not present in the DataFrame. "
            f"Available columns: {sorted(str(c) for c in df.columns)}"
        )

    if df[target_column].isna().all():
        raise ValueError(
            f"target_column '{target_column}' is entirely NaN. "
            "Cannot generate a distribution chart with no valid data."
        )

    if problem_type == "classification":
        series = df[target_column].dropna()
        unique_vals = series.nunique()

        if unique_vals > 20:
            raise ValueError(
                f"target_column '{target_column}' has {unique_vals} unique values, "
                "which exceeds the maximum of 20 for classification. "
                "This column may not be a sensible classification target, or "
                f"the problem_type may be mislabeled (try '{unique_vals} unique values' for regression?)."
            )

        counts = series.value_counts().sort_index()
        x_vals = list(counts.index.astype(str))
        y_vals = [int(v) for v in counts.values]
        fig = px.bar(
            x=x_vals,
            y=y_vals,
            title=f"Target Distribution: {target_column}",
            labels={"x": target_column, "y": "Count"},
            text=y_vals,
        )
        fig.update_traces(
            textposition="outside",
            textfont_size=11,
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Count",
            xaxis_title=target_column,
        )

    elif problem_type == "regression":
        series = df[target_column].dropna()
        unique_vals = series.nunique()

        if unique_vals <= 1:
            fig = px.histogram(series, nbins=1, title=f"Target Distribution: {target_column}")
        elif unique_vals < 10:
            fig = px.histogram(series, nbins=unique_vals, title=f"Target Distribution: {target_column}")
        else:
            fig = px.histogram(series, title=f"Target Distribution: {target_column}")

        fig.update_layout(
            xaxis_title=target_column,
            yaxis_title="Count",
            showlegend=False,
        )

    else:
        raise ValueError(
            f"Unsupported problem_type '{problem_type}'. "
            "Target distribution charts are only supported for "
            "'classification' and 'regression'."
        )

    return fig.to_json()


@with_agent_logging("eda_boxplot_generation")
def generate_boxplots(
    df: pd.DataFrame,
    outlier_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Generate Plotly box plots for specified numeric columns."""
    if len(df) == 0:
        raise ValueError(
            "DataFrame is empty. Cannot generate box plots on zero rows."
        )

    exclude = set(exclude_columns or [])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if outlier_columns is not None:
        if len(outlier_columns) == 0:
            return {"boxplots": {}, "skipped_columns": []}
        for col in outlier_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in outlier_columns does not exist in the DataFrame. "
                    f"Available columns: {sorted(str(c) for c in df.columns)}"
                )
            if col not in numeric_cols:
                raise ValueError(
                    f"Column '{col}' in outlier_columns is not numeric. "
                    f"Numeric columns: {sorted(numeric_cols)}"
                )
        plot_cols = outlier_columns
    else:
        plot_cols = numeric_cols

    boxplots: dict[str, str] = {}
    skipped: list[dict[str, str]] = []

    for col in plot_cols:
        col_str = str(col)
        if col_str in exclude:
            continue

        series = df[col].dropna()

        if len(series) == 0:
            skipped.append({"column": col_str, "reason": "Column is entirely NaN"})
            continue

        fig = px.box(series, title=col_str)
        fig.update_layout(
            yaxis_title=col_str,
            showlegend=False,
        )

        boxplots[col_str] = fig.to_json()

    return {
        "boxplots": boxplots,
        "skipped_columns": skipped,
    }
