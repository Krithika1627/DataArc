from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from agents.dataset_understanding import identify_id_columns
from agents.eda_agent import compute_correlation_pairs


def build_feature_pipeline(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    exclude_columns: list[str] = None,
    ordinal_columns: dict[str, list[str]] = None,
    cardinality_threshold: int = 10,
    collinearity_threshold: float = 0.85,
) -> dict:
    """Build (and fit) a reusable feature-engineering pipeline for a DataFrame."""
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

    id_columns = identify_id_columns(df)
    excluded = set(id_columns)
    excluded.update(str(c) for c in (exclude_columns or []))
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
        transformers.append(("scaled", StandardScaler(), numeric_cols))
    if low_cardinality_cols:
        transformers.append(
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                low_cardinality_cols,
            )
        )
    if ordinal_cols:
        transformers.append(
            (
                "ordinal",
                OrdinalEncoder(categories=[ordinal_map[c] for c in ordinal_cols]),
                ordinal_cols,
            )
        )
    if high_cardinality_cols:
        transformers.append(
            (
                "label",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                high_cardinality_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers)
    pipeline = Pipeline([("preprocessing", preprocessor)])

    X = pipeline.fit_transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    feature_names = list(pipeline.get_feature_names_out())
    transformed_df = pd.DataFrame(X, columns=feature_names, index=df.index)

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
