"""Project roots. Overridable by environment for tests and the GPU server.

Los artefactos de ESTUDIO (runs, recorridos, estudios) viven en un repo aparte,
`foveal-vision-data`: este repo es el codigo que mide, aquel es lo medido. Aqui
esta la unica indireccion que lo decide -- `data_root()` -- y por eso separar de
verdad se hace tocando ESTE fichero y no cada script.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(os.environ.get("FV_ROOT", Path(__file__).resolve().parents[2]))


def data_root() -> Path:
    """Donde se ESCRIBEN los artefactos de estudio.

    Orden: `FV_DATA_ROOT` > el repo hermano `foveal-vision-data` si esta clonado
    > este repo. El ultimo caso es deliberado y NO es un fallo: sin el repo de
    datos clonado, todo sigue funcionando exactamente como antes. Una separacion
    que rompe al que no ha clonado nada es una separacion que nadie adopta.
    """
    v = os.environ.get("FV_DATA_ROOT")
    if v:
        return Path(v)
    hermano = project_root().parent / "foveal-vision-data"
    return hermano if hermano.exists() else project_root()


def data_archive_root() -> Path | None:
    """El archivo FECHADO de `foveal-vision-data` (`<anio>/<mes>/...`), o None.

    Lo escribio la migracion y NO tiene la forma plana `runs/<name>/` que usan
    los almacenes: un run vive dentro de su recorrido y dentro de su mes. Por eso
    se lee aparte, con `index.json` como mapa, en vez de intentar que una sola
    raiz sirva para las dos formas.
    """
    d = data_root() / "index.json"
    return data_root() if d.exists() else None


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
    return data_root() / "runs"


def sweeps_root() -> Path:
    return data_root() / "sweeps"


def studies_root() -> Path:
    return data_root() / "studies"


def cache_root() -> Path:
    return project_root() / "data" / "cache"


def ui_state_path() -> Path:
    # Remembered UI defaults (filters + form values). Committable so it travels
    # with the repo to the GPU server. NOT a domain artifact: an opaque blob of
    # conveniences, never a source of truth for A-H.
    return project_root() / "state" / "ui-state.json"
