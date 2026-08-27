"""Study state on disk: studies/<name>/plan.json (committed) + progress.json
(live state). No engine of its own — a study generates sweeps (H) and reads
their rankings; its store is these two files (formatos.md §4.7).
"""

from __future__ import annotations

from pathlib import Path

from fv import datarepo, settings
from fv.ioutils import read_json_retrying, write_json_atomic


class StudyStoreError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


class StudyStore:
    # `root` given = a flat directory (tests, and any caller pinning a layout).
    # `root` omitted = the data repository, filed by month (fv.datarepo). The
    # study is what CHOOSES the month: it is picked once here, at create(), and
    # every sweep and run of that study inherits it (see fv.datarepo).
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else None

    def path(self, name: str, month: str | None = None) -> Path:
        if self.root is not None:
            return self.root / name
        return datarepo.resolve("studies", name, month)

    def _dirs(self) -> list[Path]:
        if self.root is None:
            return datarepo.iter_dirs("studies")
        return sorted(self.root.iterdir()) if self.root.exists() else []

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
        out = []
        for d in self._dirs():
            if (d / "plan.json").exists():
                out.append({"name": d.name,
                            "plan": read_json_retrying(d / "plan.json"),
                            "progress": self.progress(d.name)})
        return out

    def _plans(self):
        """(name, plan) for every study, reading plan.json only — no progress
        self-heal side effect (this is called from delete-guards, a read)."""
        for d in self._dirs():
            p = d / "plan.json"
            if p.exists():
                yield d.name, read_json_retrying(p)

    def used_by_dataset(self, dataset_name: str) -> list[str]:
        """Studies that FIX this B (plan.window_dataset). A study retrains on it
        by name at every advance, so deleting B would break the study later."""
        return [n for n, plan in self._plans()
                if plan.get("window_dataset") == dataset_name]

    def used_by_recipe(self, recipe_name: str) -> list[str]:
        """Studies that carry this D as base_recipe. advance re-resolves it by
        name (generate_sweep), so deleting D would break the study later."""
        return [n for n, plan in self._plans()
                if plan.get("base_recipe") == recipe_name]

    def delete(self, name: str) -> None:
        d = self.path(name)
        if not self.exists(name):
            raise StudyStoreError("study_not_found",
                                  f"no existe el estudio '{name}'", "nada que borrar")
        for f in sorted(d.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        d.rmdir()
