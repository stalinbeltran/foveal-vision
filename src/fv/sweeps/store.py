"""Sweep state on disk: sweeps/<name>/spec.json + state.json + the child runs.

Durable by construction: a sweep survives a restart (this machine hibernates
overnight; the GPU server redeploys) and resumes by counting finished runs.
"""

from __future__ import annotations

import time
from pathlib import Path

from fv import settings
from fv.ioutils import read_json_retrying, write_json_atomic
from fv.proc import pid_alive


class SweepStoreError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


class SweepStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.sweeps_root()

    def path(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return (self.path(name) / "spec.json").exists()

    def create(self, name: str, spec: dict) -> Path:
        d = self.path(name)
        if d.exists():
            raise SweepStoreError("sweep_exists",
                                  f"ya existe un recorrido llamado '{name}'",
                                  "elige otro nombre: no se sobrescribe nunca")
        d.mkdir(parents=True)
        write_json_atomic(d / "spec.json", spec)
        self.set_state(name, "queued")
        return d

    def spec(self, name: str) -> dict:
        p = self.path(name) / "spec.json"
        if not p.exists():
            raise SweepStoreError("sweep_not_found",
                                  f"no existe el recorrido '{name}'",
                                  "mira la lista en /sweeps")
        return read_json_retrying(p)

    def set_state(self, name: str, status: str, **extra) -> None:
        payload = {"status": status, "updated_at": time.time()}
        payload.update(extra)
        write_json_atomic(self.path(name) / "state.json", payload)

    def state(self, name: str) -> dict:
        p = self.path(name) / "state.json"
        return read_json_retrying(p) if p.exists() else {"status": "unknown"}

    def reconcile(self, name: str) -> dict:
        """Heal a stale 'running': if the owner process that wrote it is gone
        (crash / API restart / this machine hibernating), no runner will ever
        read the stop file, so the state would say 'running' forever (the
        inherited trap). Mark it 'interrupted' — a terminal state that is
        resumable and deletable. Errs safe: a live or unknown owner is left be.
        Returns the (possibly updated) state."""
        st = self.state(name)
        if st.get("status") != "running":
            return st
        pid = st.get("pid")
        if pid is None or pid_alive(pid):
            return st  # legacy sweep without an owner, or genuinely still running
        self.set_state(name, "interrupted", done=st.get("done", 0),
                       total=st.get("total"),
                       reason="el proceso que lo ejecutaba ya no existe "
                              "(caida/reinicio/hibernacion); reanudalo para seguir")
        return self.state(name)

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "spec.json").exists():
                spec = read_json_retrying(d / "spec.json")
                st = self.reconcile(d.name)
                out.append({"name": d.name, "spec": spec, "state": st})
        return out

    def request_stop(self, name: str) -> None:
        if not self.exists(name):
            raise SweepStoreError("sweep_not_found",
                                  f"no existe el recorrido '{name}'", "")
        write_json_atomic(self.path(name) / "stop.json", {"requested_at": time.time()})

    def delete(self, name: str) -> None:
        """Remove the sweep dir ONLY. The child runs are the caller's problem —
        deleting a sweep without its runs would orphan them, so the orchestration
        (guards + cascade) lives in runner.delete_sweep, never here."""
        d = self.path(name)
        if not self.exists(name):
            raise SweepStoreError("sweep_not_found",
                                  f"no existe el recorrido '{name}'", "nada que borrar")
        for f in sorted(d.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        d.rmdir()

    def clear_stop(self, name: str) -> None:
        (self.path(name) / "stop.json").unlink(missing_ok=True)

    def stop_requested(self, name: str) -> bool:
        return (self.path(name) / "stop.json").exists()
