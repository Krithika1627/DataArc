from __future__ import annotations

import pytest
import pandas as pd


@pytest.fixture
def dup_synthetic() -> pd.DataFrame:
    """DataFrame with 3 seeded duplicates (rows with id=1 and id=3 are duplicated)."""
    return pd.DataFrame(
        {
            "id": [1, 2, 1, 3, 1, 4, 5, 3, 6, 7],
            "value": ["a", "b", "a", "c", "a", "d", "e", "c", "f", "g"],
        }
    )


@pytest.fixture
def impute_synthetic() -> pd.DataFrame:
    """DataFrame with missing values covering all imputation scenarios.

    - col_symmetric: float64, ~2 NaNs, roughly symmetric distribution (should impute with mean)
    - col_skewed: float64, ~2 NaNs, outliers pulling skew >1 (should impute with median)
    - col_category: object, ~2 NaNs, mode='a' (should impute with mode)
    - col_high_miss: float64, ~12 NaNs, ~85.7% missing (should be flagged, not imputed)
    """
    return pd.DataFrame(
        {
            "col_symmetric": [
                10.0, 12.0, 11.0, 13.0, 9.0,
                11.0, 10.0, None, 12.0, 11.0,
                10.0, 13.0, 11.5, 12.5,
            ],
            "col_skewed": [
                1.0, 2.0, 1.0, 3.0, 2.0,
                100.0, 200.0, None, 1.0, 2.0,
                3.0, 1.0, 2.0, None,
            ],
            "col_category": [
                "a", "b", "a", "c", None,
                "a", "b", "c", "a", None,
                "b", "a", "c", "a",
            ],
            "col_high_miss": [
                1.0, None, None, None, None,
                None, None, None, None, None,
                None, None, None, None,
            ],
        }
    )


@pytest.fixture
def outlier_synthetic() -> pd.DataFrame:
    """DataFrame covering all outlier-handling scenarios.

    - col_outliers: float64, values 10-25 range with extreme outliers 500 and -200
    - is_active: int64, 0/1 flag (low cardinality, should be skipped)
    - col_symmetric: float64, no outliers (values in 9-13 range)
    - tutoring_sessions: int64, 0-5 range, whole numbers, <=20 unique (bounded count, should be skipped)
    """
    return pd.DataFrame(
        {
            "col_outliers": [
                15.0, 20.0, 18.0, 22.0, 14.0,
                16.0, 19.0, 21.0, 17.0, 23.0,
                500.0, -200.0, 13.0, 25.0,
            ],
            "is_active": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "col_symmetric": [
                10.0, 12.0, 11.0, 13.0, 9.0,
                11.0, 10.0, 10.0, 12.0, 11.0,
                10.0, 13.0, 11.5, 12.5,
            ],
            "tutoring_sessions": [
                0, 1, 0, 2, 1, 3, 0, 2, 4, 1,
                0, 3, 2, 5,
            ],
        }
    )


@pytest.fixture
def sample_profile_data() -> pd.DataFrame:
    """Standard 6-row dataset for profile_dataset tests."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, None, 40, 25],
            "name": ["Alice", "Bob", "Charlie", "Diana", None, "Alice"],
            "salary": [50000.0, 60000.0, 70000.0, None, 80000.0, 50000.0],
            "department": [
                "Engineering",
                "Marketing",
                "Engineering",
                "Engineering",
                "HR",
                "Engineering",
            ],
        }
    )


@pytest.fixture
def obvious_target_df() -> pd.DataFrame:
    """DataFrame where 'target' is the obvious classification target."""
    return pd.DataFrame(
        {
            "feature_1": range(100),
            "feature_2": [x % 11 for x in range(100)],
            "target": [0, 1] * 50,
        }
    )


@pytest.fixture
def id_and_target_df() -> pd.DataFrame:
    """DataFrame with an ID column (should be penalised) and a real target."""
    return pd.DataFrame(
        {
            "id": range(100),
            "label": [0, 1] * 50,
        }
    )


@pytest.fixture
def no_target_df() -> pd.DataFrame:
    """DataFrame with no clear target (all continuous or near-unique columns)."""
    return pd.DataFrame(
        {
            "feature_a": range(100),
            "feature_b": [x % 20 for x in range(100)],
            "feature_c": [f"val_{x}" for x in range(100)],
        }
    )


@pytest.fixture
def regression_target_df() -> pd.DataFrame:
    """DataFrame where 'exam_score' (last column, numeric, high cardinality) wins."""
    return pd.DataFrame(
        {
            "category_1": [f"group_{i % 3}" for i in range(200)],
            "category_2": [f"type_{i % 4}" for i in range(200)],
            "category_3": [f"region_{i % 5}" for i in range(200)],
            "exam_score": [50 + (i % 45) * 0.5 for i in range(200)],
        }
    )
