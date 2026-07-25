"""Study state on disk: studies/<name>/plan.json (committed) + progress.json
(live state). No engine of its own — a study generates sweeps (H) and reads
their rankings; its store is these two files (formatos.md §4.7).
"""

from __future__ import annotations

from pathlib import Path

from fv import settings
from fv.ioutils import read_json_retrying, write_json_atomic


class StudyStoreError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


class StudyStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.studies_root()

    def path(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return (self.path(name) / "plan.json").exists()

    def create(self, name: str, plan: dict, progress: dict) -> Path:
        d = self.path(name)
        if d.exists():
            raise StudyStoreError("study_exists",
                                  f"ya existe un estudio llamado '{name}'",
                                  "elige otro nombre: no se sobrescribe nunca")
        d.mkdir(parents=True)
        write_json_atomic(d / "plan.json", plan)      # committed (description)
        write_json_atomic(d / "progress.json", progress)  # live state
        return d

    def plan(self, name: str) -> dict:
        p = self.path(name) / "plan.json"
        if not p.exists():
            raise StudyStoreError("study_not_found",
                                  f"no existe el estudio '{name}'",
                                  "mira la lista en /studies")
        return read_json_retrying(p)

    def progress(self, name: str) -> dict:
        p = self.path(name) / "progress.json"
        if p.exists():
            return read_json_retrying(p)
        # progress.json is regenerable live state (gitignored); a committed
        # plan.json with no progress is a fresh clone / cleaned tree, not a
        # missing study. Reconstruct the step-0 progress from the plan and
        # persist it (self-healing, like SweepStore/RunStore.reconcile).
        if not self.exists(name):
            raise StudyStoreError("study_not_found",
                                  f"no existe el estudio '{name}'",
                                  "mira la lista en /studies")
        from fv.studies.driver import initial_progress
        progress = initial_progress(read_json_retrying(self.path(name) / "plan.json"))
        self.set_progress(name, progress)
        return progress

    def set_progress(self, name: str, progress: dict) -> None:
        write_json_atomic(self.path(name) / "progress.json", progress)

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "plan.json").exists():
                out.append({"name": d.name,
                            "plan": read_json_retrying(d / "plan.json"),
                            "progress": self.progress(d.name)})
        return out

    def delete(self, name: str) -> None:
        d = self.path(name)
        if not self.exists(name):
            raise StudyStoreError("study_not_found",
                                  f"no existe el estudio '{name}'", "nada que borrar")
        for f in sorted(d.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        d.rmdir()
