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


if __name__ == "__main__":
    # Inline test 
    import json

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
    print(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    # sanity assertions
    assert result["row_count"] == 6
    assert result["column_count"] == 4
    assert result["duplicate_row_count"] == 1  # row 0 and row 5
    assert result["duplicate_row_percentage"] == round(100 / 6, 2)

    age_col = [c for c in result["columns"] if c["name"] == "age"][0]
    assert age_col["missing_count"] == 1
    assert age_col["missing_percentage"] == round(100 / 6, 2)
    assert age_col["unique_count"] == 4  # 25, 30, 35, 40 (None excluded)

    name_col = [c for c in result["columns"] if c["name"] == "name"][0]
    assert name_col["missing_count"] == 1
    assert name_col["unique_count"] == 4  # Alice, Bob, Charlie, Diana

    salary_col = [c for c in result["columns"] if c["name"] == "salary"][0]
    assert salary_col["missing_count"] == 1
    assert salary_col["unique_count"] == 4

    dept_col = [c for c in result["columns"] if c["name"] == "department"][0]
    assert dept_col["missing_count"] == 0
    assert dept_col["unique_count"] == 3  # Engineering, Marketing, HR

    print("\nAll sanity checks passed.")
