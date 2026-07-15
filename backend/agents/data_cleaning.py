from __future__ import annotations

import json
import os

import pandas as pd

from logging_utils import with_agent_logging


@with_agent_logging("duplicate_removal")
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows from the DataFrame"""
    rows_before = len(df)
    cleaned = df.drop_duplicates()
    rows_after = len(cleaned)
    duplicates_removed = rows_before - rows_after
    duplicate_percentage = (
        round((duplicates_removed / rows_before) * 100, 2)
        if rows_before > 0
        else 0.0
    )

    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "duplicate_percentage": duplicate_percentage,
    }

    return cleaned, summary


def save_versioned_artifact(
    df: pd.DataFrame,
    stage_name: str,
    version: int = 1,
    artifacts_dir: str = "artifacts",
) -> str:
    """Save a DataFrame as a versioned CSV file inside the artifacts directory"""
    os.makedirs(artifacts_dir, exist_ok=True)
    file_name = f"{stage_name}_v{version}.csv"
    file_path = os.path.join(artifacts_dir, file_name)
    df.to_csv(file_path, index=False)
    return os.path.abspath(file_path)


@with_agent_logging("cleaning_orchestration")
def clean_duplicates_and_save(
    df: pd.DataFrame,
    artifacts_dir: str = "artifacts",
) -> dict:
    """Remove duplicate rows, save the cleaned result, and record a changelog"""
    cleaned_df, dup_summary = remove_duplicates(df)

    artifact_path = save_versioned_artifact(
        cleaned_df, "cleaned", version=1, artifacts_dir=artifacts_dir,
    )

    os.makedirs(artifacts_dir, exist_ok=True)
    changelog_name = "cleaned_v1_changelog.json"
    changelog_path = os.path.join(artifacts_dir, changelog_name)
    with open(changelog_path, "w") as f:
        json.dump(dup_summary, f, indent=2)

    return {
        "artifact_path": artifact_path,
        "changelog_path": os.path.abspath(changelog_path),
        "summary": dup_summary,
    }


if __name__ == "__main__":
    import tempfile

    # Create a synthetic DataFrame with seeded duplicates 
    # Total duplicates = 3.
    synthetic = pd.DataFrame(
        {
            "id": [1, 2, 1, 3, 1, 4, 5, 3, 6, 7],
            "value": ["a", "b", "a", "c", "a", "d", "e", "c", "f", "g"],
        }
    )

    rows_before = len(synthetic)
    unique_rows = synthetic.drop_duplicates()
    expected_dupes = rows_before - len(unique_rows)
    print(f"Synthetic DataFrame: {rows_before} rows, seeded {expected_dupes} duplicates")

    # Use a temporary directory so we don't pollute the real artifacts folder
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = clean_duplicates_and_save(synthetic, artifacts_dir=tmp_dir)

        print("\nReturned summary")
        print(json.dumps(result, indent=2, default=str))

        print(f"\nVerifying CSV at: {result['artifact_path']}")
        saved_df = pd.read_csv(result["artifact_path"])
        saved_rows = len(saved_df)
        expected_rows = rows_before - expected_dupes
        print(f"  Saved rows: {saved_rows} (expected {expected_rows})")
        assert saved_rows == expected_rows, (
            f"Expected {expected_rows} rows, got {saved_rows}"
        )
        print("CSV contents match expected row count. PASSED")

        print(f"\nVerifying changelog at: {result['changelog_path']}")
        with open(result["changelog_path"]) as f:
            changelog = json.load(f)
        print(f"  Changelog contents: {json.dumps(changelog, indent=2)}")
        assert changelog["rows_before"] == rows_before
        assert changelog["rows_after"] == expected_rows
        assert changelog["duplicates_removed"] == expected_dupes
        print("  Changelog assertions PASSED")

    print("\nAll tests passed")
