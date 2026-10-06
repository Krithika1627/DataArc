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
        content_json = data
    elif isinstance(data, str):
        try:
            content_json = json.loads(data)
            content_type = "json"
        except Exception:
            content_type = "binary"
            content_binary = data.encode("utf-8")
    else:
        try:
            content_json = json.loads(json.dumps(data, default=str))
            content_type = "json"
        except Exception:
            content_type = "binary"
            content_binary = str(data).encode("utf-8")

    with eng.connect() as conn:
        with conn.begin():
            if eng.dialect.name == "postgresql":
                json_val = json.dumps(content_json, default=str) if content_json is not None else None
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
                json_val = json.dumps(content_json, default=str) if content_json is not None else None
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
