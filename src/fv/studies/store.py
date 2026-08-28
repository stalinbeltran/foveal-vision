"""Study state on disk: studies/<name>/plan.json (committed) + progress.json
(live state). No engine of its own — a study generates sweeps (H) and reads
their rankings; its store is these two files (formatos.md §4.7).
"""

from __future__ import annotations

from pathlib import Path

from fv import artefactos, settings
from fv.ioutils import read_json_retrying, write_json_atomic


class StudyStoreError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


class StudyStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.studies_root()

    def path(self, name: str) -> Path:
        """Donde ESTA el estudio: plano -> archivo fechado -> legado.

        La forma PLANA, no `destino()`: ver `SweepStore.path`.
        """
        return artefactos.resolver("studies", name, self.root / name)

    def destino(self, name: str) -> Path:
        """Donde se ESCRIBE. Un estudio ESTRENA su carpeta de mes, y todo lo
        suyo -- sus recorridos y los runs de estos -- la hereda.

        Salvo que sus recorridos ya hayan estrenado una: por script el recorrido
        se crea antes que el estudio, asi que quien llegue segundo hereda del
        primero en vez de abrir un mes nuevo."""
        d = artefactos.destino_agrupado("studies", name)
        return d if d is not None else self.root / name

    def exists(self, name: str) -> bool:
        return (self.path(name) / "plan.json").exists()

    def create(self, name: str, plan: dict, progress: dict) -> Path:
        # el DESTINO, no `path()`: si ya estuviera archivado, `path()` devolveria
        # el archivo y esto escribiria dentro de el (misma razon que RunStore)
        d = self.destino(name)
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
        # los tres sitios, sin repetir: lo nuevo, lo archivado y lo legado
        out = []
        for nombre in artefactos.nombres("studies", self.root):
            d = self.path(nombre)
            if (d / "plan.json").exists():
                out.append({"name": nombre,
                            "plan": read_json_retrying(d / "plan.json"),
                            "progress": self.progress(nombre)})
        return out

    def _plans(self):
        """(name, plan) for every study, reading plan.json only — no progress
        self-heal side effect (this is called from delete-guards, a read)."""
        for nombre in artefactos.nombres("studies", self.root):
            p = self.path(nombre) / "plan.json"
            if p.exists():
                yield nombre, read_json_retrying(p)

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
