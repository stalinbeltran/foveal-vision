"""Project roots. Overridable by environment for tests and the GPU server."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(os.environ.get("FV_ROOT", Path(__file__).resolve().parents[2]))


def external_datasets_root() -> Path | None:
    v = os.environ.get("FV_DATASETS_ROOT")
    if v:
        return Path(v)
    sibling = project_root().parent / "image-text-sample-generator" / "data" / "datasets"
    return sibling if sibling.exists() else None


def local_sources_root() -> Path:
    return project_root() / "data" / "sources"


def window_datasets_root() -> Path:
    return project_root() / "data" / "window-datasets"


def networks_root() -> Path:
    return project_root() / "configs" / "networks"


def recipes_root() -> Path:
    return project_root() / "configs" / "recipes"


def runs_root() -> Path:
    return project_root() / "runs"


def sweeps_root() -> Path:
    return project_root() / "sweeps"


def studies_root() -> Path:
    return project_root() / "studies"


def cache_root() -> Path:
    return project_root() / "data" / "cache"


def ui_state_path() -> Path:
    # Remembered UI defaults (filters + form values). Committable so it travels
    # with the repo to the GPU server. NOT a domain artifact: an opaque blob of
    # conveniences, never a source of truth for A-H.
    return project_root() / "state" / "ui-state.json"
