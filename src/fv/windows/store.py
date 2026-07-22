"""Access to window datasets on disk (B)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fv import settings
from fv.ioutils import read_json_retrying


class WindowStoreError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


class WindowDatasetStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.window_datasets_root()

    def path(self, name: str) -> Path:
        return self.root / name

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "manifest.json").exists():
                m = read_json_retrying(d / "manifest.json")
                m["name"] = d.name
                out.append(m)
        return out

    def manifest(self, name: str) -> dict:
        p = self.path(name) / "manifest.json"
        if not p.exists():
            raise WindowStoreError("window_dataset_missing",
                                   f"no existe el dataset de ventanas '{name}'",
                                   "construyelo con fv-extract o POST /window-datasets")
        m = read_json_retrying(p)
        m["name"] = name
        return m

    def arrays(self, name: str) -> dict:
        p = self.path(name) / "windows.npz"
        if not p.exists():
            raise WindowStoreError("window_dataset_missing",
                                   f"'{name}' no tiene windows.npz",
                                   "reconstruye el dataset")
        data = np.load(p)
        return {k: data[k] for k in data.files}

    def split_map(self, name: str) -> dict:
        p = self.path(name) / "split.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def delete(self, name: str, used_by: list[str]) -> None:
        if used_by:
            raise WindowStoreError(
                "window_dataset_in_use",
                f"'{name}' lo referencian los runs: {', '.join(used_by)}",
                "borra esos runs primero, o deja el dataset")
        d = self.path(name)
        if not d.exists():
            raise WindowStoreError("window_dataset_missing",
                                   f"no existe '{name}'", "nada que borrar")
        for f in sorted(d.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        d.rmdir()
