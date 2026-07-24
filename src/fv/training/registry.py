"""E — the run store. Explicit state, no silent overwrite, provenance complete.

The three measured traps this makes impossible (herencia.md §4): state deduced
from which files exist (a crash stays 'running' forever) -> status.json;
silent overwrite (mkdir exist_ok + truncate) -> create() refuses; no
environment capture -> provenance carries python/torch/platform/device.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from fv import settings
from fv.ioutils import read_json_retrying, read_text_retrying, write_json_atomic


class RunError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
        return "unknown: not a git repository"
    except Exception as e:  # the reason, never a silent null (formatos.md §2)
        return f"unknown: {e}"


def environment(device: str) -> dict:
    try:
        import torch
        torch_v = torch.__version__
    except ImportError:
        torch_v = "not installed"
    return {"python": sys.version.split()[0], "torch": torch_v,
            "platform": sys.platform, "device": device}


class RunStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.runs_root()

    def path(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return (self.path(name) / "config.json").exists()

    def create(self, name: str, config: dict) -> Path:
        d = self.path(name)
        if d.exists():
            raise RunError("run_exists",
                           f"ya existe un run llamado '{name}'",
                           "elige otro nombre, o borra ese run primero: no se "
                           "sobrescribe nunca")
        d.mkdir(parents=True)
        write_json_atomic(d / "config.json", config)
        self.set_status(name, "queued")
        return d

    def set_status(self, name: str, status: str, **extra) -> None:
        payload = {"status": status, "updated_at": time.time()}
        payload.update(extra)
        write_json_atomic(self.path(name) / "status.json", payload)

    def status(self, name: str) -> dict:
        p = self.path(name) / "status.json"
        if not p.exists():
            return {"status": "unknown"}
        return read_json_retrying(p)

    def config(self, name: str) -> dict:
        p = self.path(name) / "config.json"
        if not p.exists():
            raise RunError("run_not_found", f"no existe el run '{name}'",
                           "mira la lista en /runs")
        return read_json_retrying(p)

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        # newest first: the run you just trained is the one you want to look at,
        # and it keeps the default selection on a current (loadable) checkpoint
        # instead of the alphabetically-first, possibly-stale one.
        dirs = sorted(self.root.iterdir(),
                      key=lambda d: d.stat().st_mtime, reverse=True)
        for d in dirs:
            if not (d / "config.json").exists():
                continue
            cfg = read_json_retrying(d / "config.json")
            st = self.status(d.name)
            summary = {}
            sp = d / "summary.json"
            if sp.exists():
                summary = read_json_retrying(sp)
            prov = cfg.get("provenance", {})
            out.append({
                "name": d.name,
                "status": st.get("status"),
                "epoch": st.get("epoch"),
                "window_dataset": prov.get("window_dataset", {}).get("name"),
                "network": prov.get("network", {}).get("name"),
                "recipe": prov.get("recipe", {}).get("name"),
                "sweep": prov.get("sweep"),
                "best": summary.get("best"),
                "monitor": summary.get("monitor"),
                "epochs_run": summary.get("epochs_run"),
                "seconds_per_epoch": summary.get("seconds_per_epoch"),
            })
        return out

    def metrics_since(self, name: str, since: int = 0) -> dict:
        p = self.path(name) / "metrics.jsonl"
        records = []
        if p.exists():
            lines = read_text_retrying(p).splitlines()
            for line in lines[since:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a live run: the last line may be mid-write — normal
            return {"records": records, "next": since + len(records)}
        return {"records": [], "next": since}

    def request_stop(self, name: str, reason: str = "user") -> None:
        if not self.exists(name):
            raise RunError("run_not_found", f"no existe el run '{name}'", "")
        write_json_atomic(self.path(name) / "stop.json",
                          {"requested_at": time.time(), "reason": reason})

    def stop_requested(self, name: str) -> bool:
        return (self.path(name) / "stop.json").exists()

    def used_by_dataset(self, dataset_name: str) -> list[str]:
        out = []
        for r in self.list():
            if r.get("window_dataset") == dataset_name:
                out.append(r["name"])
        return out

    def used_by_sweep(self, sweep_name: str) -> list[str]:
        out = []
        for r in self.list():
            if r.get("sweep") == sweep_name:
                out.append(r["name"])
        return out

    def rename(self, name: str, new_name: str) -> None:
        if self.status(name).get("status") == "running":
            raise RunError("run_is_running", f"'{name}' esta corriendo",
                           "para el run antes de renombrarlo")
        if self.exists(new_name):
            raise RunError("run_exists", f"ya existe '{new_name}'", "elige otro nombre")
        if not self.exists(name):
            raise RunError("run_not_found", f"no existe el run '{name}'", "")
        self.path(name).rename(self.path(new_name))

    def delete(self, name: str) -> None:
        if not self.exists(name):
            raise RunError("run_not_found", f"no existe el run '{name}'", "nada que borrar")
        if self.status(name).get("status") == "running":
            raise RunError("run_is_running", f"'{name}' esta corriendo",
                           "para el run antes de borrarlo")
        d = self.path(name)
        for f in sorted(d.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        d.rmdir()
