"""Cross-cutting hardware governance and system safety utilities."""

from __future__ import annotations

from src.utils.hardware_guard import (
    HardwareWatchdog,
    MemoryLimitExceededError,
    check_memory_ceiling,
    enforce_single_thread_env,
    set_process_safety,
)

__all__ = [
    "HardwareWatchdog",
    "MemoryLimitExceededError",
    "check_memory_ceiling",
    "enforce_single_thread_env",
    "set_process_safety",
]
