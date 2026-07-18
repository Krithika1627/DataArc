from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import os
import time
from datetime import datetime, timezone

import pandas as pd

_LOGGER_NAME = "agent_runs"
logger = logging.getLogger(_LOGGER_NAME)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    os.makedirs("logs", exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        "logs/agent_runs.log",
        maxBytes=5 * 1024 * 1024,  # ~5 MB
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

def log_agent_run(
    agent_name: str,
    inputs_summary: dict,
    outputs_summary: dict,
    duration_seconds: float,
    error: str | None = None,
) -> None:
    
    entry = {
        "agent_name": agent_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "inputs_summary": inputs_summary,
        "outputs_summary": outputs_summary,
    }

    if error is not None:
        entry["error"] = error
        logger.error(json.dumps(entry))
    else:
        logger.info(json.dumps(entry))

def _summarise_arg(value: object) -> dict:
    if isinstance(value, pd.DataFrame):
        return {"type": "DataFrame", "shape": [int(s) for s in value.shape]}
    if isinstance(value, dict):
        scalar_keys = {
            k: v
            for k, v in value.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        return {"type": "dict", "total_keys": len(value), "scalar_values": scalar_keys}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__, "repr": repr(value)[:120]}


def _summarise_output(value: object) -> dict:
    if isinstance(value, dict):
        scalar_values = {
            k: v
            for k, v in value.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        return {
            "type": "dict",
            "total_keys": len(value),
            "scalar_values": scalar_values,
        }
    if isinstance(value, list):
        preview = None
        if value:
            preview = str(value[0])[:150]
        return {"type": "list", "length": len(value), "first_item_preview": preview}
    if isinstance(value, pd.DataFrame):
        return {"type": "DataFrame", "shape": [int(s) for s in value.shape]}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "length": len(value),
            "elements": [_summarise_output(item) for item in value],
        }
    return {"type": type(value).__name__, "repr": repr(value)[:120]}

def with_agent_logging(agent_name: str):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            inputs = {
                f"arg_{i}": _summarise_arg(a)
                for i, a in enumerate(args)
            }
            inputs.update(
                {k: _summarise_arg(v) for k, v in kwargs.items()}
            )

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                duration = time.perf_counter() - start
                log_agent_run(
                    agent_name, inputs, {}, duration,
                    error="Agent function raised an exception",
                )
                raise

            duration = time.perf_counter() - start
            log_agent_run(agent_name, inputs, _summarise_output(result), duration)
            return result

        return wrapper

    return decorator
