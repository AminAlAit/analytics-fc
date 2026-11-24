"""Cross-platform hardware safety manager and process memory watchdog.

Enforces below-normal OS priority, single-threaded execution variables
(threads=1, n_jobs=1), and deterministic termination or exception guards when
process resident set size (RSS) exceeds the 1024 MB memory ceiling.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable
from typing import Any

import psutil

logger = logging.getLogger("analytics_fc.utils.hardware_guard")

THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "DUCKDB_THREADS",
    "AFC_THREADS",
)


class MemoryLimitExceededError(MemoryError):
    """Raised when process RSS exceeds the configured hardware memory cap."""

    def __init__(self, rss_mb: float, max_memory_mb: float) -> None:
        self.rss_mb = rss_mb
        self.max_memory_mb = max_memory_mb
        super().__init__(
            f"Process RSS {rss_mb:.2f} MB exceeded memory ceiling {max_memory_mb:.2f} MB."
        )


def _validate_positive_int(name: str, value: Any) -> int:
    """Validate an integer >= 1 while explicitly rejecting booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    """Validate a finite positive number while explicitly rejecting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive float.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive float.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive float.")
    return parsed


def enforce_single_thread_env(threads: int = 1, n_jobs: int = 1) -> dict[str, str]:
    """Bind math, DuckDB, and parallel runtime env variables to single-thread limits."""
    safe_threads = _validate_positive_int("threads", threads)
    safe_jobs = _validate_positive_int("n_jobs", n_jobs)

    thread_str = str(safe_threads)
    jobs_str = str(safe_jobs)
    updated: dict[str, str] = {}
    for key in THREAD_ENV_KEYS:
        os.environ[key] = thread_str
        updated[key] = thread_str

    os.environ["AFC_N_JOBS"] = jobs_str
    os.environ["LOKY_MAX_CPU_COUNT"] = jobs_str
    updated["AFC_N_JOBS"] = jobs_str
    updated["LOKY_MAX_CPU_COUNT"] = jobs_str
    return updated


def check_memory_ceiling(
    max_memory_mb: float = 1024.0,
    terminate_on_breach: bool = False,
    terminator: Callable[[int], None] | None = None,
) -> float:
    """Inspect process RSS in MiB and enforce the upper memory ceiling."""
    safe_ceiling = _validate_positive_float("max_memory_mb", max_memory_mb)

    proc = psutil.Process()
    rss_mb = float(round(float(proc.memory_info().rss) / (1024.0 * 1024.0), 2))
    if rss_mb > safe_ceiling:
        logger.error(
            "Memory ceiling breach detected: PID=%d RSS=%.2f MB (limit=%.2f MB)",
            proc.pid,
            rss_mb,
            safe_ceiling,
        )
        if terminate_on_breach:
            if terminator is not None:
                terminator(proc.pid)
            else:
                proc.terminate()
        raise MemoryLimitExceededError(rss_mb=rss_mb, max_memory_mb=safe_ceiling)
    return rss_mb


def set_process_safety(
    threads: int = 1,
    n_jobs: int = 1,
    max_memory_mb: float = 1024.0,
    terminate_on_breach: bool = False,
    terminator: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Apply below-normal priority, thread caps, and verify the 1GB RSS ceiling."""
    safe_threads = _validate_positive_int("threads", threads)
    safe_jobs = _validate_positive_int("n_jobs", n_jobs)
    safe_ceiling = _validate_positive_float("max_memory_mb", max_memory_mb)

    proc = psutil.Process()
    priority_applied = "unchanged"

    below_normal = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None)
    try:
        if os.name == "nt" and below_normal is not None:
            proc.nice(below_normal)
            priority_applied = "BELOW_NORMAL_PRIORITY_CLASS"
        else:
            proc.nice(10)
            priority_applied = "nice_10"
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError) as exc:
        logger.warning("Unable to update OS process priority cleanly: %s", exc)

    env_vars = enforce_single_thread_env(threads=safe_threads, n_jobs=safe_jobs)
    rss_mb = check_memory_ceiling(
        max_memory_mb=safe_ceiling,
        terminate_on_breach=terminate_on_breach,
        terminator=terminator,
    )

    return {
        "pid": proc.pid,
        "priority_applied": priority_applied,
        "threads": safe_threads,
        "n_jobs": safe_jobs,
        "env_vars": env_vars,
        "rss_mb": rss_mb,
        "max_memory_mb": safe_ceiling,
    }


class HardwareWatchdog:
    """Context manager and background memory watcher terminating on 1GB RSS breach."""

    def __init__(
        self,
        max_memory_mb: float = 1024.0,
        poll_interval_seconds: float = 0.5,
        threads: int = 1,
        n_jobs: int = 1,
        terminate_on_breach: bool = True,
        terminator: Callable[[int], None] | None = None,
        on_breach: Callable[[MemoryLimitExceededError], None] | None = None,
    ) -> None:
        self.max_memory_mb = _validate_positive_float("max_memory_mb", max_memory_mb)
        self.poll_interval_seconds = _validate_positive_float(
            "poll_interval_seconds", poll_interval_seconds
        )
        self.threads = _validate_positive_int("threads", threads)
        self.n_jobs = _validate_positive_int("n_jobs", n_jobs)
        self.terminate_on_breach = bool(terminate_on_breach)
        self._terminator = terminator
        self._on_breach = on_breach

        self._lock = threading.Lock()
        self._active_stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self.breach_event = threading.Event()
        self.last_error: MemoryLimitExceededError | None = None
        self.last_status: dict[str, Any] | None = None

    @property
    def is_running(self) -> bool:
        """Return True while the background RSS polling thread is alive."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def check_now(self) -> float:
        """Run an immediate RSS ceiling check and return current memory in MiB."""
        return check_memory_ceiling(
            max_memory_mb=self.max_memory_mb,
            terminate_on_breach=self.terminate_on_breach,
            terminator=self._terminator,
        )

    def _watch_loop(self, stop_event: threading.Event) -> None:
        """Poll RSS until stop_event is signaled or memory ceiling is breached."""
        while not stop_event.wait(self.poll_interval_seconds):
            try:
                self.check_now()
            except MemoryLimitExceededError as exc:
                self.last_error = exc
                self.breach_event.set()
                if self._on_breach is not None:
                    try:
                        self._on_breach(exc)
                    except Exception:
                        logger.exception("HardwareWatchdog on_breach callback raised an error.")
                return

    def start(self) -> dict[str, Any]:
        """Configure process safety limits and launch background memory watcher."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.last_status or {}

            self.breach_event.clear()
            self.last_error = None

            status = set_process_safety(
                threads=self.threads,
                n_jobs=self.n_jobs,
                max_memory_mb=self.max_memory_mb,
                terminate_on_breach=self.terminate_on_breach,
                terminator=self._terminator,
            )
            self.last_status = status

            stop_event = threading.Event()
            watcher = threading.Thread(
                target=self._watch_loop,
                args=(stop_event,),
                name="analytics-fc-hardware-watchdog",
                daemon=True,
            )
            self._active_stop_event = stop_event
            self._thread = watcher
            watcher.start()
            return status

    def stop(self) -> None:
        """Signal active watcher thread termination and join deterministically."""
        with self._lock:
            thread = self._thread
            stop_event = self._active_stop_event
            self._thread = None
            self._active_stop_event = None
            if stop_event is not None:
                stop_event.set()

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.poll_interval_seconds * 4.0))

    def __enter__(self) -> HardwareWatchdog:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if exc_type is None:
                self.check_now()
        finally:
            self.stop()
