from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd


def get_next_version(artifacts_dir: str, base_filename: str) -> int:
    if not os.path.exists(artifacts_dir):
        return 1

    pattern = re.compile(rf"^{re.escape(base_filename)}_v(\d+)(?:_.*|\..*)?$")
    versions: list[int] = []

    for filename in os.listdir(artifacts_dir):
        match = pattern.match(filename)
        if match:
            try:
                versions.append(int(match.group(1)))
            except (ValueError, IndexError):
                continue

    return max(versions, default=0) + 1


def save_artifact(
    data: Any,
    artifacts_dir: str,
    base_filename: str,
    extension: str,
    version: int | None = None,
) -> str:
    os.makedirs(artifacts_dir, exist_ok=True)
    if version is None:
        version = get_next_version(artifacts_dir, base_filename)

    ext = extension.lstrip(".")
    filename = f"{base_filename}_v{version}.{ext}"
    file_path = os.path.join(artifacts_dir, filename)

    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=False)
    elif isinstance(data, (dict, list)):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif isinstance(data, bytes):
        with open(file_path, "wb") as f:
            f.write(data)
    elif isinstance(data, str):
        with open(file_path, "w") as f:
            f.write(data)
    else:
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            with open(file_path, "w") as f:
                f.write(str(data))

    return os.path.abspath(file_path)


def get_latest_version_path(
    artifacts_dir: str, base_filename: str, extension: str
) -> str:
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(
            f"No {base_filename} artifact found in '{artifacts_dir}' - artifacts directory does not exist. Has the corresponding pipeline step been run?"
        )

    ext = extension.lstrip(".")
    pattern = re.compile(
        rf"^{re.escape(base_filename)}_v(\d+)(?:_.*)?\.{re.escape(ext)}$"
    )

    matched_files: list[tuple[int, str]] = []
    for filename in os.listdir(artifacts_dir):
        match = pattern.match(filename)
        if match:
            try:
                v = int(match.group(1))
                matched_files.append((v, filename))
            except (ValueError, IndexError):
                continue

    if not matched_files:
        raise FileNotFoundError(
            f"No {base_filename} artifact found with extension '.{ext}' in '{artifacts_dir}' - has the corresponding pipeline step been run?"
        )

    matched_files.sort(key=lambda x: (x[0], x[1]))
    latest_filename = matched_files[-1][1]
    return os.path.abspath(os.path.join(artifacts_dir, latest_filename))
