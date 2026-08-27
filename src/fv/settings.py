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


# OBSOLETAS desde 2026-08-27: E/H/I ya no viven en este repo, sino en el de
# datos, agrupados por mes (ver `data_root()` abajo y `fv.datarepo`). Se dejan
# porque describen el layout PLANO que siguen usando quien inyecta `root=` a
# mano y los artefactos historicos, pero ningun store las llama ya: resolver un
# nombre es trabajo de `fv.datarepo`, en un solo sitio.


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


# --- El repositorio de datos (foveal-vision-data) -----------------------------
#
# Desde 2026-08-27 los artefactos de estudio (E runs, H recorridos, I estudios)
# NO viven en este repo: viven en el repositorio hermano `foveal-vision-data`.
# Este repo es el codigo que mide; aquel es lo medido (ver CLAUDE.md).
#
# Dentro del repo de datos cada estudio ocupa una carpeta de mes,
# `<anio>/<NN>-<mes>/`, elegida UNA VEZ al crearse y heredada por todo lo que
# cuelgue de el. El mes separa estudios para poder leerlos; NO es una linea
# temporal de cada run. Por eso un recorrido lanzado el dia 1 del mes siguiente
# se queda con su estudio: lo contrario dispersaria un mismo estudio en dos
# carpetas por el mero paso de la medianoche.

MESES_ES = {
    1: "01-enero", 2: "02-febrero", 3: "03-marzo", 4: "04-abril",
    5: "05-mayo", 6: "06-junio", 7: "07-julio", 8: "08-agosto",
    9: "09-septiembre", 10: "10-octubre", 11: "11-noviembre", 12: "12-diciembre",
}


def data_root() -> Path:
    """Raiz del repositorio de datos.

    `FV_DATA_ROOT` manda; si no, el repo hermano `../foveal-vision-data`.
    Mismo patron que `external_datasets_root()` para el generador: funciona sin
    configurar nada en una maquina donde los dos repos son hermanos, y se
    redirige con una variable en la flota o en los tests.
    """
    v = os.environ.get("FV_DATA_ROOT")
    if v:
        return Path(v)
    return project_root().parent / "foveal-vision-data"


def month_dir(when: "dt.datetime | None" = None) -> str:
    """`2026/08-agosto` para la fecha dada (por defecto, ahora en UTC)."""
    import datetime as dt
    d = when or dt.datetime.now(dt.UTC)
    return f"{d.year}/{MESES_ES[d.month]}"
