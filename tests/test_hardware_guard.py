"""Unit tests for hardware safety process priority, thread caps, and memory watchdog."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import psutil
import pytest

from src.utils import (
    HardwareWatchdog,
    MemoryLimitExceededError,
    check_memory_ceiling,
    enforce_single_thread_env,
    set_process_safety,
)


def test_set_process_safety_enforces_priority_and_single_thread_vars() -> None:
    """Verify below-normal priority, threads=1, n_jobs=1, and RSS status."""
    status = set_process_safety(threads=1, n_jobs=1, max_memory_mb=1024.0)
    assert status["threads"] == 1
    assert status["n_jobs"] == 1
    assert status["rss_mb"] > 0.0
    assert status["max_memory_mb"] == 1024.0

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "DUCKDB_THREADS",
        "AFC_THREADS",
        "AFC_N_JOBS",
        "LOKY_MAX_CPU_COUNT",
    ):
        assert os.environ[var] == "1"

    proc = psutil.Process()
    if os.name == "nt":
        assert status["priority_applied"] == "BELOW_NORMAL_PRIORITY_CLASS"
        assert proc.nice() == psutil.BELOW_NORMAL_PRIORITY_CLASS
    else:
        assert status["priority_applied"] == "nice_10"


def test_memory_ceiling_terminates_and_raises_on_breach() -> None:
    """Verify RSS > 1024 MB invokes terminator and raises MemoryLimitExceededError."""
    fake_proc = MagicMock()
    fake_proc.pid = 42424
    fake_proc.memory_info.return_value = SimpleNamespace(rss=1150 * 1024 * 1024)

    terminated_pids: list[int] = []

    with patch("src.utils.hardware_guard.psutil.Process", return_value=fake_proc):
        with pytest.raises(MemoryLimitExceededError) as exc_info:
            check_memory_ceiling(
                max_memory_mb=1024.0,
                terminate_on_breach=True,
                terminator=lambda pid: terminated_pids.append(pid),
            )
    assert exc_info.value.rss_mb == 1150.0
    assert exc_info.value.max_memory_mb == 1024.0
    assert terminated_pids == [42424]

    with patch("src.utils.hardware_guard.psutil.Process", return_value=fake_proc):
        with pytest.raises(MemoryLimitExceededError):
            check_memory_ceiling(max_memory_mb=1024.0, terminate_on_breach=True)
    fake_proc.terminate.assert_called_once()


def test_hardware_watchdog_context_and_background_breach_termination() -> None:
    """Verify HardwareWatchdog background thread monitors RSS and stops cleanly."""
    with HardwareWatchdog(max_memory_mb=1024.0, poll_interval_seconds=0.05) as guard:
        assert guard.is_running is True
        assert guard.check_now() <= 1024.0
    assert guard.is_running is False

    mem_readings = [
        SimpleNamespace(rss=200 * 1024 * 1024),
        SimpleNamespace(rss=1200 * 1024 * 1024),
        SimpleNamespace(rss=1200 * 1024 * 1024),
    ]
    call_idx = {"i": 0}

    def _next_mem() -> Any:
        idx = min(call_idx["i"], len(mem_readings) - 1)
        call_idx["i"] += 1
        return mem_readings[idx]

    fake_proc = MagicMock()
    fake_proc.pid = 7777
    fake_proc.memory_info.side_effect = _next_mem
    terminated: list[int] = []
    breaches: list[MemoryLimitExceededError] = []

    with patch("src.utils.hardware_guard.psutil.Process", return_value=fake_proc):
        watchdog = HardwareWatchdog(
            max_memory_mb=1024.0,
            poll_interval_seconds=0.02,
            terminate_on_breach=True,
            terminator=lambda pid: terminated.append(pid),
            on_breach=lambda err: breaches.append(err),
        )
        watchdog.start()
        assert watchdog.breach_event.wait(timeout=2.0) is True
        watchdog.stop()

    assert terminated == [7777]
    assert len(breaches) == 1
    assert breaches[0].rss_mb == 1200.0
    assert watchdog.is_running is False


def test_validation_booleans_non_finite_and_callback_exception_resilience() -> None:
    """Reject bool/non-finite inputs and isolate callback errors and active exceptions."""
    with pytest.raises(ValueError, match="threads must be an integer >= 1"):
        enforce_single_thread_env(threads=0)
    with pytest.raises(ValueError, match="threads must be an integer >= 1"):
        enforce_single_thread_env(threads=True)
    with pytest.raises(ValueError, match="n_jobs must be an integer >= 1"):
        enforce_single_thread_env(n_jobs=0)
    with pytest.raises(ValueError, match="n_jobs must be an integer >= 1"):
        enforce_single_thread_env(n_jobs=False)

    for bad_float in (True, False, -5.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_memory_mb"):
            check_memory_ceiling(max_memory_mb=bad_float)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="poll_interval_seconds"):
            HardwareWatchdog(poll_interval_seconds=bad_float)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="threads must be an integer >= 1"):
        HardwareWatchdog(threads=0)
    with pytest.raises(ValueError, match="n_jobs must be an integer >= 1"):
        HardwareWatchdog(n_jobs=-1)

    fake_proc = MagicMock()
    fake_proc.pid = 999
    fake_proc.nice.side_effect = psutil.AccessDenied(pid=999)
    fake_proc.memory_info.return_value = SimpleNamespace(rss=128 * 1024 * 1024)
    with patch("src.utils.hardware_guard.psutil.Process", return_value=fake_proc):
        status = set_process_safety()
    assert status["priority_applied"] == "unchanged"
    assert status["rss_mb"] == 128.0

    # Verify on_breach raising + calling stop() inside callback does not deadlock or crash
    readings = [
        SimpleNamespace(rss=100 * 1024 * 1024),
        SimpleNamespace(rss=1300 * 1024 * 1024),
    ]
    idx_state = {"i": 0}

    def _seq_mem() -> Any:
        i = min(idx_state["i"], len(readings) - 1)
        idx_state["i"] += 1
        return readings[i]

    cb_proc = MagicMock()
    cb_proc.pid = 8888
    cb_proc.memory_info.side_effect = _seq_mem

    with patch("src.utils.hardware_guard.psutil.Process", return_value=cb_proc):
        wd_box: dict[str, HardwareWatchdog] = {}

        def _failing_cb(_err: MemoryLimitExceededError) -> None:
            wd_box["wd"].stop()
            raise RuntimeError("callback failure")

        wd = HardwareWatchdog(
            max_memory_mb=1024.0,
            poll_interval_seconds=0.02,
            terminate_on_breach=False,
            on_breach=_failing_cb,
        )
        wd_box["wd"] = wd
        wd.start()
        assert wd.breach_event.wait(timeout=2.0) is True
        wd.stop()
        assert wd.is_running is False

    # Verify __exit__ preserves an active user exception without running check_now()
    err_readings = [
        SimpleNamespace(rss=100 * 1024 * 1024),
        SimpleNamespace(rss=1500 * 1024 * 1024),
    ]
    err_idx = {"i": 0}

    def _err_mem() -> Any:
        i = min(err_idx["i"], len(err_readings) - 1)
        err_idx["i"] += 1
        return err_readings[i]

    exc_proc = MagicMock()
    exc_proc.pid = 5555
    exc_proc.memory_info.side_effect = _err_mem
    with patch("src.utils.hardware_guard.psutil.Process", return_value=exc_proc):
        with pytest.raises(RuntimeError, match="user boom"):
            with HardwareWatchdog(max_memory_mb=1024.0, poll_interval_seconds=5.0):
                raise RuntimeError("user boom")
