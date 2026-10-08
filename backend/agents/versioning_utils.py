from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

load_dotenv()

_ENGINE: Optional[Engine] = None
_DB_INITIALIZED: bool = False


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url or not url.strip():
        # Fallback to local SQLite for offline/test environments
        return "sqlite:///artifacts.db"
    url = url.strip()
    # Normalize postgres:// to postgresql+psycopg2:// or postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


def get_db_engine() -> Engine:
    """Returns a singleton SQLAlchemy engine with connection pooling."""
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    url = _get_database_url()
    if url.startswith("sqlite"):
        if ":memory:" in url:
            _ENGINE = create_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            _ENGINE = create_engine(
                url,
                connect_args={"check_same_thread": False},
            )
    else:
        # Postgres / Neon configuration
        _ENGINE = create_engine(
            url,
            pool_size=5,
            max_overflow=2,
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={"connect_timeout": 10},
        )

    init_db(_ENGINE)
    return _ENGINE


def get_db_connection():
    """Returns an active connection from the database engine pool."""
    return get_db_engine().connect()


def init_db(engine: Optional[Engine] = None) -> None:
    """Creates the artifacts table and lookup index if they do not exist."""
    global _DB_INITIALIZED
    eng = engine or get_db_engine()

    dialect_name = eng.dialect.name
    with eng.connect() as conn:
        with conn.begin():
            if dialect_name == "postgresql":
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS artifacts (
                            id SERIAL PRIMARY KEY,
                            dataset_id TEXT NOT NULL,
                            artifact_type TEXT NOT NULL,
                            version INTEGER NOT NULL,
                            content_json JSONB,
                            content_binary BYTEA,
                            content_type TEXT NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            UNIQUE (dataset_id, artifact_type, version)
                        );
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS idx_artifacts_lookup 
                        ON artifacts (dataset_id, artifact_type, version DESC);
                        """
                    )
                )
            else:
                conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS artifacts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dataset_id TEXT NOT NULL,
                            artifact_type TEXT NOT NULL,
                            version INTEGER NOT NULL,
                            content_json JSON,
                            content_binary BLOB,
                            content_type TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE (dataset_id, artifact_type, version)
                        );
                        """
                    )
                )
                conn.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS idx_artifacts_lookup 
                        ON artifacts (dataset_id, artifact_type, version DESC);
                        """
                    )
                )
    _DB_INITIALIZED = True


def _clean_dataset_id(dataset_id: str) -> str:
    """Extracts a clean dataset_id from a string or legacy path."""
    if not dataset_id or not str(dataset_id).strip():
        raise ValueError("dataset_id cannot be empty or None.")
    s = str(dataset_id).strip().rstrip("/\\")
    if "/" in s or "\\" in s:
        base = os.path.basename(s)
        if base == "artifacts":
            parent = os.path.basename(os.path.dirname(s))
            if parent:
                return parent
        return base
    return s


def get_dataset_artifacts_dir(base_dir: str, dataset_id: str) -> str:
    """Legacy helper returning dataset_id."""
    return _clean_dataset_id(dataset_id)


def _normalize_artifact_type(
    artifact_type: str, extension: str | None = None, is_df: bool = False
) -> str:
    norm = artifact_type.strip()
    if norm == "feature_engineered" and (extension == "csv" or is_df):
        return "feature_engineered_csv"
    if norm in ("cleaned", "cleaned_dataset") and (extension == "csv" or is_df):
        return "cleaned"
    if norm == "cleaned" and extension in ("json", "changelog"):
        return "cleaned_changelog"
    if "changelog" in norm:
        return "cleaned_changelog"
    return norm


def get_next_version(dataset_id: str, artifact_type: str) -> int:
    """Computes the next version number for (dataset_id, artifact_type)."""
    clean_id = _clean_dataset_id(dataset_id)
    eng = get_db_engine()

    norm_type = _normalize_artifact_type(artifact_type)
    
    with eng.connect() as conn:
        if norm_type in ("feature_engineered", "feature_engineered_csv"):
            query = text(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM artifacts "
                "WHERE dataset_id = :dataset_id AND artifact_type IN ('feature_engineered', 'feature_engineered_csv')"
            )
            val = conn.execute(query, {"dataset_id": clean_id}).scalar()
        elif norm_type in ("cleaned", "cleaned_changelog"):
            query = text(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM artifacts "
                "WHERE dataset_id = :dataset_id AND artifact_type IN ('cleaned', 'cleaned_changelog')"
            )
            val = conn.execute(query, {"dataset_id": clean_id}).scalar()
        else:
            query = text(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM artifacts "
                "WHERE dataset_id = :dataset_id AND artifact_type = :artifact_type"
            )
            val = conn.execute(
                query, {"dataset_id": clean_id, "artifact_type": norm_type}
            ).scalar()
        return int(val or 1)


import math
import numpy as np


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NaN, Inf, and numpy types to JSON-safe primitives (null, int, float)."""
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in obj]
    return obj


def save_artifact(
    data: Any,
    dataset_id: str,
    artifact_type: str,
    extension: str = "json",
    version: int | None = None,
    is_binary: bool = False,
) -> Any:
    """
    Saves an artifact row in the database.
    JSON objects/dicts/DataFrames are stored in content_json.
    Binary data (e.g. PDF bytes) are stored in content_binary.
    Returns the version number written.
    """
    clean_id = _clean_dataset_id(dataset_id)
    eng = get_db_engine()

    is_df = isinstance(data, pd.DataFrame)
    target_type = _normalize_artifact_type(artifact_type, extension=extension, is_df=is_df)

    if version is None:
        version = get_next_version(clean_id, artifact_type)

    content_json: Any = None
    content_binary: bytes | None = None
    content_type: str = "json"

    if is_binary or isinstance(data, (bytes, bytearray)):
        content_type = "binary"
        content_binary = bytes(data)
    elif isinstance(data, pd.DataFrame):
        content_type = "json"
        content_json = {
            "__type__": "dataframe",
            "json": data.to_json(orient="split"),
        }
    elif isinstance(data, (dict, list)):
        content_type = "json"
        content_json = _sanitize_for_json(data)
    elif isinstance(data, str):
        try:
            parsed = json.loads(data)
            content_json = _sanitize_for_json(parsed)
            content_type = "json"
        except Exception:
            content_type = "binary"
            content_binary = data.encode("utf-8")
    else:
        try:
            parsed = json.loads(json.dumps(data, default=str))
            content_json = _sanitize_for_json(parsed)
            content_type = "json"
        except Exception:
            content_type = "binary"
            content_binary = str(data).encode("utf-8")

    with eng.connect() as conn:
        with conn.begin():
            if eng.dialect.name == "postgresql":
                json_val = json.dumps(_sanitize_for_json(content_json), default=str) if content_json is not None else None
                stmt = text(
                    """
                    INSERT INTO artifacts (dataset_id, artifact_type, version, content_json, content_binary, content_type)
                    VALUES (:dataset_id, :artifact_type, :version, CAST(:content_json AS JSONB), :content_binary, :content_type)
                    ON CONFLICT (dataset_id, artifact_type, version) 
                    DO UPDATE SET content_json = EXCLUDED.content_json, content_binary = EXCLUDED.content_binary, content_type = EXCLUDED.content_type
                    """
                )
                conn.execute(
                    stmt,
                    {
                        "dataset_id": clean_id,
                        "artifact_type": target_type,
                        "version": version,
                        "content_json": json_val,
                        "content_binary": content_binary,
                        "content_type": content_type,
                    },
                )
            else:
                json_val = json.dumps(_sanitize_for_json(content_json), default=str) if content_json is not None else None
                stmt = text(
                    """
                    INSERT OR REPLACE INTO artifacts (dataset_id, artifact_type, version, content_json, content_binary, content_type)
                    VALUES (:dataset_id, :artifact_type, :version, :content_json, :content_binary, :content_type)
                    """
                )
                conn.execute(
                    stmt,
                    {
                        "dataset_id": clean_id,
                        "artifact_type": target_type,
                        "version": version,
                        "content_json": json_val,
                        "content_binary": content_binary,
                        "content_type": content_type,
                    },
                )


    # Optional local file system mirroring for backwards compatibility with legacy file assertions
    try:
        if isinstance(dataset_id, str) and (
            "/" in dataset_id or "\\" in dataset_id or os.path.exists(dataset_id)
        ):
            os.makedirs(dataset_id, exist_ok=True)
            if is_binary or content_type == "binary":
                fpath = os.path.join(dataset_id, f"{artifact_type}_v{version}.{extension}")
                with open(fpath, "wb") as f:
                    f.write(content_binary or b"")
            elif isinstance(data, pd.DataFrame):
                fpath = os.path.join(dataset_id, f"{artifact_type}_v{version}.csv")
                data.to_csv(fpath, index=False)
            else:
                fpath = os.path.join(dataset_id, f"{artifact_type}_v{version}.{extension}")
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(content_json, f, indent=2, default=str)
    except Exception:
        pass

    return version


def get_latest_artifact(
    dataset_id: str, artifact_type: str, extension: str | None = None
) -> Any:
    """
    Retrieves the latest version of an artifact for dataset_id.
    Returns parsed JSON dict/list, reconstructed pd.DataFrame, or bytes.
    Raises FileNotFoundError if not found.
    """
    clean_id = _clean_dataset_id(dataset_id)
    eng = get_db_engine()

    lookup_types = []
    if extension:
        norm = _normalize_artifact_type(artifact_type, extension=extension)
        lookup_types.append(norm)
    if artifact_type not in lookup_types:
        lookup_types.append(artifact_type)
    if artifact_type == "cleaned_changelog" and "cleaned" not in lookup_types:
        lookup_types.append("cleaned")
    elif artifact_type == "cleaned" and "cleaned_changelog" not in lookup_types:
        lookup_types.append("cleaned_changelog")
    elif artifact_type == "feature_engineered" and "feature_engineered_csv" not in lookup_types:
        lookup_types.append("feature_engineered_csv")

    with eng.connect() as conn:
        row = None
        for l_type in lookup_types:
            query = text(
                """
                SELECT content_json, content_binary, content_type, version 
                FROM artifacts 
                WHERE dataset_id = :dataset_id AND artifact_type = :artifact_type 
                ORDER BY version DESC 
                LIMIT 1
                """
            )
            row = conn.execute(
                query, {"dataset_id": clean_id, "artifact_type": l_type}
            ).fetchone()
            if row:
                break

        if not row:
            raise FileNotFoundError(
                f"No {artifact_type} artifact found for dataset '{clean_id}'. "
                f"Has the corresponding pipeline step been run?"
            )

        content_json, content_binary, content_type, version = row

        if content_type == "binary":
            return bytes(content_binary) if content_binary is not None else b""

        if content_json is not None:
            data = content_json
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass

            if isinstance(data, dict) and data.get("__type__") == "dataframe":
                return pd.read_json(io.StringIO(data["json"]), orient="split")

            return data

        return {}


def get_latest_version_path(
    dataset_id: str, artifact_type: str, extension: str = "json"
) -> Any:
    """Wrapper around get_latest_artifact for compatibility with previous call sites."""
    return get_latest_artifact(dataset_id, artifact_type, extension=extension)


def list_all_dataset_runs() -> list[dict[str, Any]]:
    """Returns a list of all distinct dataset runs saved in the database with summary info."""
    eng = get_db_engine()
    runs_map: dict[str, dict[str, Any]] = {}

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT dataset_id, artifact_type, version, created_at,
                       CASE WHEN artifact_type IN ('dataset_profile', 'profile', 'evaluation_summary', 'evaluation_bundle', 'feature_engineered', 'feature_engineering', 'ml_plan', 'report') 
                            THEN content_json ELSE NULL END AS content_json,
                       content_type
                FROM artifacts
                ORDER BY created_at DESC
                """
            )
        ).fetchall()

        for row in rows:
            d_id, art_type, ver, created, content_json, c_type = row
            if d_id not in runs_map:
                runs_map[d_id] = {
                    "dataset_id": d_id,
                    "filename": None,
                    "first_created": str(created),
                    "last_updated": str(created),
                    "artifacts": set(),
                    "row_count": None,
                    "column_count": None,
                    "target_column": None,
                    "problem_type": None,
                    "winning_model": None,
                    "winning_score": None,
                    "recommended_metric": None,
                    "has_report": False,
                }

            run = runs_map[d_id]
            run["artifacts"].add(art_type)
            run["last_updated"] = str(created)

            # Parse metadata
            if art_type == "report":
                run["has_report"] = True

            parsed_json = content_json
            if isinstance(parsed_json, str):
                try:
                    parsed_json = json.loads(parsed_json)
                except Exception:
                    parsed_json = None

            if isinstance(parsed_json, dict):
                if not run["filename"]:
                    run["filename"] = parsed_json.get("filename")
                if art_type in ("dataset_profile", "profile"):
                    profile_obj = parsed_json.get("profile") or parsed_json
                    if isinstance(profile_obj, dict):
                        run["row_count"] = profile_obj.get("row_count")
                        run["column_count"] = profile_obj.get("column_count")
                        if not run["filename"]:
                            run["filename"] = profile_obj.get("filename")
                    if not run["target_column"]:
                        run["target_column"] = parsed_json.get("selected_target") or parsed_json.get("target_column")
                    if not run["problem_type"]:
                        analysis = parsed_json.get("problem_type_analysis")
                        if isinstance(analysis, dict):
                            run["problem_type"] = analysis.get("problem_type")
                elif art_type in ("evaluation_summary", "evaluation_bundle"):
                    run["winning_model"] = parsed_json.get("winning_model_name")
                    run["winning_score"] = parsed_json.get("winning_score")
                    run["recommended_metric"] = parsed_json.get("recommended_metric")
                    if not run["problem_type"]:
                        run["problem_type"] = parsed_json.get("problem_type")
                    if not run["target_column"]:
                        run["target_column"] = parsed_json.get("target_column")
                elif art_type in ("feature_engineered", "feature_engineering"):
                    if not run["target_column"]:
                        run["target_column"] = parsed_json.get("target_column")
                    if not run["problem_type"]:
                        run["problem_type"] = parsed_json.get("problem_type")
                elif art_type == "ml_plan":
                    if not run["target_column"]:
                        run["target_column"] = parsed_json.get("selected_target") or parsed_json.get("target_column")
                    if not run["problem_type"]:
                        run["problem_type"] = parsed_json.get("problem_type")
                elif art_type == "report":
                    meta = parsed_json.get("metadata") or {}
                    if isinstance(meta, dict):
                        if not run["target_column"]:
                            run["target_column"] = meta.get("target_column")
                        if not run["problem_type"]:
                            run["problem_type"] = meta.get("problem_type")
                        if not run["filename"]:
                            run["filename"] = meta.get("filename")

    result = []
    for d_id, data in runs_map.items():
        data["artifacts"] = sorted(list(data["artifacts"]))
        result.append(data)

    return result


def get_dataset_history(dataset_id: str) -> dict[str, Any]:
    """Retrieves all available stage artifacts for a given dataset_id."""
    clean_id = _clean_dataset_id(dataset_id)
    history: dict[str, Any] = {"dataset_id": clean_id}

    stage_types = {
        "profile": [("dataset_profile", "json"), ("profile", "json")],
        "cleaning": [("cleaned_changelog", "json"), ("cleaned", "json")],
        "eda": [("eda_bundle", "json"), ("eda_insights", "json")],
        "feature_engineering": [("feature_pipeline", "json"), ("feature_engineered", "json")],
        "ml_plan": [("ml_plan", "json")],
        "training": [("training_results", "json"), ("training_summary", "json")],
        "evaluation": [("evaluation_bundle", "json"), ("evaluation_summary", "json")],
    }

    for key, type_candidates in stage_types.items():
        val = None
        for art_type, ext in type_candidates:
            try:
                val = get_latest_artifact(clean_id, art_type, extension=ext)
                if val:
                    break
            except Exception:
                continue
        history[key] = val

    try:
        pdf_bytes = get_latest_artifact(clean_id, "report", extension="pdf")
        history["has_report"] = bool(pdf_bytes)
    except Exception:
        history["has_report"] = False

    return history


