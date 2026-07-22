"""Named network configs (C): configs/networks/*.yaml — source, versioned in git."""

from __future__ import annotations

from pathlib import Path

import yaml

from fv import settings


class NetworkStoreError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


class NetworkStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.networks_root()

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for f in sorted(self.root.glob("*.yaml")):
            cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            cfg["name"] = f.stem
            out.append(cfg)
        return out

    def get(self, name: str) -> dict:
        f = self.root / f"{name}.yaml"
        if not f.exists():
            known = ", ".join(x.stem for x in self.root.glob("*.yaml")) \
                if self.root.exists() else ""
            raise NetworkStoreError("network_not_found",
                                    f"no existe la red '{name}'",
                                    f"las redes disponibles son: {known or '(ninguna)'}")
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        cfg.pop("name", None)
        cfg.pop("format_version", None)  # file-level, never frozen into a run
        return cfg

    def save(self, name: str, cfg: dict, overwrite: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        f = self.root / f"{name}.yaml"
        if f.exists() and not overwrite:
            raise NetworkStoreError("network_exists",
                                    f"ya existe una red llamada '{name}'",
                                    "elige otro nombre, o edita esa")
        body = {"format_version": 1}
        body.update({k: v for k, v in cfg.items() if k not in ("name",)})
        f.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")

    def delete(self, name: str) -> None:
        f = self.root / f"{name}.yaml"
        if not f.exists():
            raise NetworkStoreError("network_not_found",
                                    f"no existe la red '{name}'", "nada que borrar")
        f.unlink()
