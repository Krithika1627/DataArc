import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    if len(df.columns) == 0:
        return {
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "duplicate_row_count": 0,
            "duplicate_row_percentage": 0.0,
        }

    columns_info = []
    for col in df.columns:
        total = len(df)
        missing = int(df[col].isna().sum())
        columns_info.append(
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "missing_count": missing,
                "missing_percentage": round((missing / total) * 100, 2) if total > 0 else 0.0,
                "unique_count": int(df[col].nunique()),
            }
        )

    total_rows = len(df)
    duplicate_count = int(df.duplicated().sum())

    return {
        "row_count": total_rows,
        "column_count": len(df.columns),
        "columns": columns_info,
        "duplicate_row_count": duplicate_count,
        "duplicate_row_percentage": round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0,
    }


def detect_target_candidates(df: pd.DataFrame, return_all: bool = False) -> list[dict]:
    target_keywords = {"target", "label", "class", "y", "outcome", "result"}
    outcome_keywords = {"score", "price", "amount", "value", "rating", "revenue", "salary", "cost"}
    row_count = len(df)
    col_count = len(df.columns)
    all_scores = []

    for i, col in enumerate(df.columns):
        col_name = str(col)
        score = 0
        reasons = []

        normalized = col_name.lower().replace("_", "").replace(" ", "")

        # Name match (+40)
        if normalized in target_keywords:
            score += 40
            reasons.append("Column name matches common target naming pattern")

        # Low-to-moderate cardinality (+30, classification signal)
        unique_count = int(df[col].nunique())
        if row_count >= 10 and 2 <= unique_count <= 10:
            score += 30
            reasons.append(
                f"Low cardinality ({unique_count} unique values) suggests a classification target"
            )

        # Combined last-column rule (mutually exclusive a / b)
        is_last = i == col_count - 1
        is_numeric = str(df[col].dtype) in ("int64", "float64")
        high_cardinality = unique_count > 10  
        if is_last and is_numeric and high_cardinality:
            # regression-target signal
            score += 35
            reasons.append(
                "Last column, numeric, high cardinality -- strong regression target signal"
            )
        elif is_last:
            # plain last-column bonus
            score += 15
            reasons.append("Column is the last column in the dataset (common target convention)")

        # Outcome keyword bonus (+20)
        if any(kw in col_name.lower() for kw in outcome_keywords):
            score += 20
            reasons.append("Column name contains a common outcome/target keyword")

        # ID column penalty (-30)
        if unique_count >= 0.95 * row_count:
            score -= 30
            reasons.append(
                "Excluded: column has near-unique values per row, "
                "likely an ID column, not a target"
            )

        score = max(0, min(100, score))

        all_scores.append(
            {
                "column_name": col_name,
                "confidence_score": score,
                "reasons": reasons,
            }
        )

    all_scores.sort(key=lambda c: c["confidence_score"], reverse=True)

    if return_all:
        return all_scores

    return [c for c in all_scores if c["confidence_score"] > 0][:2]

if __name__ == "__main__":
    import json

    print("PROFILE_DATASET TEST")
    
    sample_data = pd.DataFrame(
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

    result = profile_dataset(sample_data)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # sanity assertions
    assert result["row_count"] == 6
    assert result["column_count"] == 4
    assert result["duplicate_row_count"] == 1
    assert result["duplicate_row_percentage"] == round(100 / 6, 2)

    age_col = [c for c in result["columns"] if c["name"] == "age"][0]
    assert age_col["missing_count"] == 1
    assert age_col["missing_percentage"] == round(100 / 6, 2)
    assert age_col["unique_count"] == 4

    name_col = [c for c in result["columns"] if c["name"] == "name"][0]
    assert name_col["missing_count"] == 1
    assert name_col["unique_count"] == 4

    salary_col = [c for c in result["columns"] if c["name"] == "salary"][0]
    assert salary_col["missing_count"] == 1
    assert salary_col["unique_count"] == 4

    dept_col = [c for c in result["columns"] if c["name"] == "department"][0]
    assert dept_col["missing_count"] == 0
    assert dept_col["unique_count"] == 3

    print("\nAll profile_dataset sanity checks passed.\n")

    print("DETECT_TARGET_CANDIDATES TESTS")
    
    print("\nTest (a): Obvious target column")
    df_a = pd.DataFrame(
        {
            "feature_1": range(100),  # 100 unique → near-unique penalty → 0
            "feature_2": [x % 11 for x in range(100)],  # 11 unique (> 10, no bonus) → 0
            "target": [0, 1] * 50,  # 2 unique, int64, name match (+40), low card (+30),
                                     # last col + numeric but 2 unique NOT high card → (b) +15
                                     # score = 85
        }
    )
    candidates_a = detect_target_candidates(df_a)
    print(json.dumps(candidates_a, indent=2, ensure_ascii=False))
    assert len(candidates_a) == 1, f"Expected 1 candidate, got {len(candidates_a)}"
    assert candidates_a[0]["column_name"] == "target"
    assert candidates_a[0]["confidence_score"] == 85, (
        f"Expected score 85, got {candidates_a[0]['confidence_score']}"
    )
    print("PASSED")

    print("\nTest (b): ID column + real target (ID should be penalised)")
    df_b = pd.DataFrame(
        {
            "id": range(100),  # 100 unique → near-unique penalty → 0
            "label": [0, 1] * 50,  # 2 unique, int64, name match (+40), low card (+30),
                                    # last col + numeric but 2 unique NOT high card → (b) +15
                                    # score = 85
        }
    )
    candidates_b = detect_target_candidates(df_b)
    print(json.dumps(candidates_b, indent=2, ensure_ascii=False))
    assert len(candidates_b) == 1, f"Expected 1 candidate, got {len(candidates_b)}"
    assert candidates_b[0]["column_name"] == "label"
    assert candidates_b[0]["confidence_score"] == 85, (
        f"Expected score 85, got {candidates_b[0]['confidence_score']}"
    )
    print("PASSED")

    print("\nTest (c): No clear target (all continuous / near-unique)")
    df_c = pd.DataFrame(
        {
            "feature_a": range(100),  # 100 unique, near-unique penalty → 0
            "feature_b": [x % 20 for x in range(100)],  # 20 unique, no bonuses → 0
            # Non-numeric dtype so last-column rule uses sub-rule (b): +15
            # 100 unique >= 95% → -30 penalty → score 0
            "feature_c": [f"val_{x}" for x in range(100)],
        }
    )
    candidates_c = detect_target_candidates(df_c)
    print(json.dumps(candidates_c, indent=2, ensure_ascii=False))
    assert len(candidates_c) == 0, f"Expected 0 candidates, got {len(candidates_c)}"
    print("  PASSED")

    print("\nTest (d): Regression target wins over low-cardinality categoricals")
    df_d = pd.DataFrame(
        {
            # Low-cardinality categorical feature columns (would dominate under old rules)
            "category_1": [f"group_{i % 3}" for i in range(200)],   # 3 unique, object, low card (+30)
            "category_2": [f"type_{i % 4}" for i in range(200)],    # 4 unique, object, low card (+30)
            "category_3": [f"region_{i % 5}" for i in range(200)],  # 5 unique, object, low card (+30)
            # Final numeric column with "score" keyword and high cardinality (45 unique)
            # keyword (+20) + last+numeric+highcard (+35) = 55
            # Must NOT trigger ID penalty (< 95% of 200 = 190 unique)
            "exam_score": [50 + (i % 45) * 0.5 for i in range(200)],  # 45 unique, float64
        }
    )
    candidates_d = detect_target_candidates(df_d)
    print("Top 2 (default mode):")
    print(json.dumps(candidates_d, indent=2, ensure_ascii=False))
    assert len(candidates_d) == 2, f"Expected 2 candidates, got {len(candidates_d)}"
    assert candidates_d[0]["column_name"] == "exam_score", (
        f"Expected exam_score as top candidate, got {candidates_d[0]['column_name']}"
    )
    assert candidates_d[0]["confidence_score"] == 55, (
        f"Expected score 55, got {candidates_d[0]['confidence_score']}"
    )
    print("PASSED")

    print("\nFull ranking (return_all=True):")
    all_d = detect_target_candidates(df_d, return_all=True)
    print(json.dumps(all_d, indent=2, ensure_ascii=False))

    print("ALL TESTS PASSED.")