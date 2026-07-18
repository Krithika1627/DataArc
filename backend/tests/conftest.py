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


@pytest.fixture
def currency_df() -> pd.DataFrame:
    """DataFrame with a numeric column stored as strings with currency symbols.

    - price_str: object dtype, values like "$1,200", "$950", etc.
    - city: genuinely categorical object column (should NOT be converted)
    """
    return pd.DataFrame(
        {
            "price_str": [
                "$1,200", "$950", "$1,050", "$2,000", "$750",
                "$1,500", "$800", "$1,100", "$3,000", "$600",
            ],
            "city": [
                "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
            ],
        }
    )


@pytest.fixture
def percent_df() -> pd.DataFrame:
    """DataFrame with a numeric column stored as strings with percent signs.

    - pct_str: object dtype, values like "45%", "62%", etc.
    - label: genuinely categorical object column (should NOT be converted)
    """
    return pd.DataFrame(
        {
            "pct_str": [
                "45%", "62%", "78%", "33%", "91%",
                "55%", "20%", "87%", "42%", "69%",
            ],
            "label": [
                "low", "medium", "high", "low", "very high",
                "medium", "low", "high", "medium", "high",
            ],
        }
    )


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    """DataFrame with a genuinely categorical object column that should NOT be converted.

    - region: object dtype with city/region names (no numeric pattern)
    - code: object dtype with short alphanumeric codes
    """
    return pd.DataFrame(
        {
            "region": [
                "North", "South", "East", "West", "Central",
                "North", "South", "East", "West", "Central",
            ],
            "code": [
                "AB-01", "CD-02", "EF-03", "GH-04", "IJ-05",
                "KL-06", "MN-07", "OP-08", "QR-09", "ST-10",
            ],
        }
    )


@pytest.fixture
def mixed_conversion_df() -> pd.DataFrame:
    """DataFrame with a column that has some convertible and some unconvertible values.

    - amount_str: object dtype, 19 convertible currency-formatted numbers and
      1 unconvertible value ("unknown"), so conversion rate = 19/20 = 95%,
      just meeting the >= 95% threshold
    - id: genuinely categorical object column (should NOT be converted)
    """
    amounts = [f"${v:,}" for v in [1200, 950, 1050, 2000, 750, 1500, 800, 1100, 3000, 600]]
    amounts += [f"${v:,}" for v in [2500, 1800, 700, 2200, 1350, 900, 1600, 4200, 2100]]
    amounts.append("unknown")  # the one unconvertible value (1/20 = 5%)
    ids = [f"A{i+1:03d}" for i in range(20)]
    return pd.DataFrame({"amount_str": amounts, "id": ids})


@pytest.fixture
def full_pipeline_df() -> pd.DataFrame:
    """DataFrame that needs all four cleaning stages.

    - id: int64, has exact duplicate rows (row 0 and row 8 are identical)
    - price_str: object dtype with currency symbols (needs dtype fixing)
    - score: float64, has some missing values (needs imputation)
    - outliers_raw: object dtype with currency + extreme values (needs dtype
      fixing first, then outlier capping)
    - is_active: int64, 0/1 flag (should be skipped by outlier handling)
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 1, 1, 8],
            "price_str": [
                "$1,200", "$950", "$2,000", "$750",
                "$1,500", "$1,100", "$800", "$1,200",  # row 7 is identical to row 0
                "$1,200", "$600",
            ],
            "score": [
                85.0, 92.0, 78.0, 95.0,
                88.0, 90.0, 76.0, 85.0,  # row 7 identical to row 0
                None, None,
            ],
            "outliers_raw": [
                "$100", "$150", "$200", "$130",
                "$500,000", "$140", "$110", "$100",  # row 7 identical to row 0
                "$500,000", "$160",
            ],
            "is_active": [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
        }
    )


@pytest.fixture
def height_unit_df() -> pd.DataFrame:
    """DataFrame with a unit-suffixed numeric column and a plain categorical.

    - Height: object dtype, values like "180cm", "165cm", etc. (should be
      detected as unit-suffixed numeric, renamed to "Height_cm")
    - city: genuinely categorical object column (should NOT be converted)
    """
    return pd.DataFrame(
        {
            "Height": [
                "180cm", "165cm", "172cm", "158cm", "190cm",
                "175cm", "168cm", "182cm", "160cm", "185cm",
            ],
            "city": [
                "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
            ],
        }
    )


@pytest.fixture
def weight_already_named_df() -> pd.DataFrame:
    """DataFrame where the column name already includes the unit.

    - Weight_kg: object dtype, values like "70kg", "65kg", etc. (should be
      converted to numeric but NOT double-renamed to "Weight_kg_kg")
    - label: genuinely categorical object column (should NOT be converted)
    """
    return pd.DataFrame(
        {
            "Weight_kg": [
                "70kg", "65kg", "80kg", "55kg", "90kg",
                "75kg", "68kg", "85kg", "60kg", "95kg",
            ],
            "label": [
                "A", "B", "C", "D", "E",
                "F", "G", "H", "I", "J",
            ],
        }
    )


@pytest.fixture
def mixed_units_df() -> pd.DataFrame:
    """DataFrame with mixed/inconsistent units that should NOT trigger renaming.

    - measurement: object dtype, some values end in "cm", some in "in", so
      no single unit reaches the 95% consistency threshold
    - desc: genuinely categorical object column
    """
    return pd.DataFrame(
        {
            "measurement": [
                "180cm", "165cm", "172cm", "70in", "68in",
                "175cm", "168cm", "182cm", "72in", "66in",
            ],
            "desc": [
                "tall", "short", "medium", "short", "short",
                "medium", "short", "tall", "medium", "short",
            ],
        }
    )

@pytest.fixture
def rating_grams_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Rating": [
                "120g", "183g", "651g", "812 g"
            ]
        }
    )