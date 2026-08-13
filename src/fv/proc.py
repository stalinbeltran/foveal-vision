"""Is the process that owns a piece of durable state still alive?

The inherited trap (herencia.md §4): a job whose process died — a crash, an API
restart, this machine hibernating overnight — leaves its state file saying
"running" forever, because the cooperative-stop signal is only ever read by a
live runner. The cure is an explicit owner: whoever marks a sweep/run "running"
records its PID, and a reader can ask whether that PID is still there.

Cross-process by design: the reader (the API) and the owner (an API job thread
OR a separate `fv-sweep` CLI process) need not be the same process. This errs
SAFE — an owner that is genuinely alive is never reported dead, so a running
sweep is never wrongly reconciled; the only risk (a recycled PID reported alive)
leaves stale state untouched, exactly as today.
"""

from __future__ import annotations

import os
import sys


def pid_alive(pid: int | None) -> bool:
    """True if a process with this PID currently exists. None/≤0 → False."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False  # no such process (same-user query does not get denied)
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE  # an exited process is not alive
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True
