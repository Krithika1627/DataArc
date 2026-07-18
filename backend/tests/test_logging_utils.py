from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

import pytest

from agents.logging_utils import with_agent_logging


class TestLoggingUtils:
    def test_successful_run_logs_correctly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "agent_runs.log")

            with mock.patch("agents.logging_utils.logger") as mock_logger:
                @with_agent_logging("dummy_agent")
                def dummy_success(value: int) -> dict:
                    return {"value": value, "squared": value * value}

                result = dummy_success(42)
                assert result == {"value": 42, "squared": 1764}

                mock_logger.info.assert_called_once()
                log_call_args = mock_logger.info.call_args[0][0]
                log_entry = json.loads(log_call_args)

                assert log_entry["agent_name"] == "dummy_agent"
                assert "duration_seconds" in log_entry
                assert "timestamp" in log_entry
                assert "inputs_summary" in log_entry
                assert "outputs_summary" in log_entry
                assert "error" not in log_entry

    def test_failing_run_logs_error(self):
        with mock.patch("agents.logging_utils.logger") as mock_logger:
            @with_agent_logging("dummy_agent")
            def dummy_fail(value: int) -> dict:
                raise RuntimeError(f"Simulated failure for input {value}")

            with pytest.raises(RuntimeError, match="Simulated failure for input 99"):
                dummy_fail(99)

            mock_logger.error.assert_called_once()
            log_call_args = mock_logger.error.call_args[0][0]
            log_entry = json.loads(log_call_args)

            assert log_entry["agent_name"] == "dummy_agent"
            assert log_entry["error"] == "Agent function raised an exception"
            assert "duration_seconds" in log_entry
