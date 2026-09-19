from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from agents.dataset_understanding import identify_id_columns
from agents.eda_agent import compute_correlation_pairs

logger = logging.getLogger("agent_runs")


def build_feature_pipeline(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    exclude_columns: list[str] = None,
    ordinal_columns: dict[str, list[str]] = None,
    cardinality_threshold: int = 10,
    collinearity_threshold: float = 0.85,
    high_missing_columns: list[str] = None,
    high_missing_threshold: float = 0.5,
) -> dict:
    """Build (and fit) a reusable feature-engineering pipeline for a DataFrame.

    Guarantees a strictly NaN-free transformed matrix by:
    1. Dropping / excluding columns flagged with high missingness (> 50%).
    2. Inserting SimpleImputer in every transformer branch (median for numeric, most_frequent for categoricals).
    3. Validating the transformed dataframe before returning.
    """
    if len(df) == 0:
        raise ValueError(
            "DataFrame is empty. Cannot build a feature pipeline on zero rows."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' is not present in the DataFrame. "
            f"Available columns: {sorted(str(c) for c in df.columns)}"
        )

    all_columns = [str(c) for c in df.columns]

    # Clean df copy so that None in object columns is treated as np.nan
    df_clean = df.copy()
    for c in df_clean.columns:
        if not pd.api.types.is_numeric_dtype(df_clean[c]):
            df_clean[c] = df_clean[c].astype(object).where(pd.notna(df_clean[c]), np.nan)

    # 1. High-missing column detection (> 50% missing or explicitly flagged)
    auto_high_missing = [
        str(c)
        for c in all_columns
        if c != str(target_column) and df[c].isna().mean() > high_missing_threshold
    ]
    flagged_high_missing = set(auto_high_missing)
    if high_missing_columns:
        flagged_high_missing.update(str(c) for c in high_missing_columns)

    id_columns = identify_id_columns(df)
    excluded = set(id_columns)
    excluded.update(str(c) for c in (exclude_columns or []))
    excluded.update(flagged_high_missing)
    excluded.add(str(target_column))
    excluded_columns = [c for c in all_columns if c in excluded]

    ordinal_map = {str(k): list(vals) for k, vals in (ordinal_columns or {}).items()}
    for col in ordinal_map:
        if col not in all_columns:
            raise ValueError(
                f"ordinal column '{col}' is not present in the DataFrame. "
                f"Available columns: {sorted(all_columns)}"
            )

    ordinal_cols = [c for c in ordinal_map if c not in excluded]
    for col in ordinal_cols:
        unique_vals = df[col].dropna().unique()
        rank_set = set(ordinal_map[col])
        unseen = [v for v in unique_vals if v not in rank_set]
        if unseen:
            raise ValueError(
                f"ordinal column '{col}' contains values not present in its "
                f"ranked category list: {unseen}. "
                f"Rank list supplied: {ordinal_map[col]}"
            )

    remaining = [c for c in all_columns if c not in excluded and c not in ordinal_map]

    numeric_cols = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in remaining if not pd.api.types.is_numeric_dtype(df[c])]

    low_cardinality_cols = [
        c for c in categorical_cols if df[c].nunique() <= cardinality_threshold
    ]
    high_cardinality_cols = [
        c for c in categorical_cols if df[c].nunique() > cardinality_threshold
    ]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaled", StandardScaler()),
        ])
        transformers.append(("scaled", num_pipeline, numeric_cols))

    if low_cardinality_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("onehot", cat_pipeline, low_cardinality_cols))

    if ordinal_cols:
        ord_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[ordinal_map[c] for c in ordinal_cols])),
        ])
        transformers.append(("ordinal", ord_pipeline, ordinal_cols))

    if high_cardinality_cols:
        high_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("label", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("label", high_pipeline, high_cardinality_cols))

    preprocessor = ColumnTransformer(transformers)
    pipeline = Pipeline([("preprocessing", preprocessor)])

    X = pipeline.fit_transform(df_clean)
    if hasattr(X, "toarray"):
        X = X.toarray()
    feature_names = list(pipeline.get_feature_names_out())
    transformed_df = pd.DataFrame(X, columns=feature_names, index=df.index)

    # Post-transformation NaN validation guardrail
    if transformed_df.isna().any().any():
        null_counts = transformed_df.isna().sum()
        nan_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Validation failed: transformed_df contains NaN values in columns: {nan_cols}. "
            f"Feature matrix must be strictly NaN-free."
        )

    _, collinear_pairs = compute_correlation_pairs(
        df[remaining], threshold=collinearity_threshold
    )

    encoding_map: dict[str, str] = {}
    for col in all_columns:
        if col in excluded:
            encoding_map[col] = "excluded"
        elif col in ordinal_map:
            encoding_map[col] = "ordinal"
        elif col in high_cardinality_cols:
            encoding_map[col] = "label"
        elif col in low_cardinality_cols:
            encoding_map[col] = "onehot"
        elif col in numeric_cols:
            encoding_map[col] = "scaled"

    return {
        "transformed_df": transformed_df,
        "pipeline": pipeline,
        "collinear_pairs": collinear_pairs,
        "encoding_map": encoding_map,
        "excluded_columns": excluded_columns,
    }
