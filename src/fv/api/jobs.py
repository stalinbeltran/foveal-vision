"""X — the in-process job queue: worker limit 1 (on CPU torch already uses
every core; N trainings at once each go ~N x slower and multiply RAM),
cooperative cancellation (the callable receives a stop function), metadata
and polling. The durable state (runs, sweeps) lives on disk in its own
domain; this registry is execution only.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor


class JobQueue:
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, target, describe: dict | None = None,
               on_cancel=None) -> dict:
        job_id = uuid.uuid4().hex[:12]
        info = {"id": job_id, "kind": kind, "status": "queued",
                "created_at": time.time(), "describe": describe or {},
                "error": None, "result": None}
        stop_event = threading.Event()
        with self._lock:
            self._jobs[job_id] = {"info": info, "stop": stop_event,
                                  "on_cancel": on_cancel}

        def _run():
            with self._lock:
                info["status"] = "running"
                info["started_at"] = time.time()
            try:
                result = target(stop_event.is_set)
                with self._lock:
                    info["status"] = "cancelled" if stop_event.is_set() else "done"
                    info["result"] = result
            except Exception as e:
                with self._lock:
                    info["status"] = "error"
                    info["error"] = {"code": getattr(e, "code", "error"),
                                     "message": str(e),
                                     "hint": getattr(e, "hint", ""),
                                     "trace": traceback.format_exc()}
            finally:
                info["finished_at"] = time.time()

        self._executor.submit(_run)
        return dict(info)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j["info"]) if j else None

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(j["info"]) for j in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(job_id)
        if not j:
            return False
        j["stop"].set()
        if j["on_cancel"]:
            j["on_cancel"]()
        return True
