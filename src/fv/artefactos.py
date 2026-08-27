"""Resolucion de rutas de artefactos de estudio: plano -> archivo fechado -> legado.

Por que hacen falta TRES sitios y no uno
----------------------------------------
Los artefactos de estudio se escriben ahora en `foveal-vision-data` (settings.
`data_root`), pero lo que ya existe esta repartido en dos formas distintas:

  1. **plano**, `<data>/runs/<run>/` -- lo que se escribe DE AHORA EN ADELANTE.
     Es la forma que `RunStore.path()` ha usado siempre.
  2. **archivo fechado**, `<data>/<anio>/<mes>/sweeps/<recorrido>/runs/<run>/` --
     lo que dejo la migracion. Un run vive dentro de su recorrido y de su mes, y
     esa relacion es estructura de directorios a proposito. `index.json` es el
     mapa: sin el no se puede encontrar un run sin saber su mes.
  3. **legado**, `<foveal-vision>/runs/<run>/` -- lo que este repo todavia tiene
     y la migracion copio, mas lo que se escribio DESPUES de aquella copia.

Se busca en ese orden y se ESCRIBE siempre en (1). El orden importa: si (3) se
mirara antes, un run migrado se leeria de la copia vieja y no de la buena.

⚠ Esto es una escalera para poder migrar sin parar el mundo, no un diseño
permanente. Cuando (3) se vacie, `legado()` se borra y queda una escalera de dos.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fv import settings


@lru_cache(maxsize=1)
def _indice() -> dict:
    """El mapa del archivo fechado. Cacheado: se lee por cada `path()` y son 851
    entradas. Si falta o esta roto se devuelve vacio -- el archivo pasa a no
    existir para el resolutor, que es degradar, no romper."""
    raiz = settings.data_archive_root()
    if raiz is None:
        return {}
    try:
        return json.loads((raiz / "index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _archivado(clase: str, nombre: str) -> Path | None:
    entrada = (_indice().get(clase) or {}).get(nombre)
    if not entrada:
        return None
    raiz = settings.data_archive_root()
    return (raiz / entrada["path"]) if raiz else None


def resolver(clase: str, nombre: str, plano: Path) -> Path:
    """La ruta donde ESTA `nombre`, o `plano` si no esta en ningun sitio.

    `plano` es tambien la respuesta cuando no existe todavia: crear siempre
    ocurre en la forma nueva.
    """
    if plano.exists():
        return plano
    arch = _archivado(clase, nombre)
    if arch is not None and arch.exists():
        return arch
    legado = settings.project_root() / clase / nombre
    if legado.exists():
        return legado
    return plano


def nombres(clase: str, plano: Path) -> list[str]:
    """Todos los nombres visibles de esa clase, sin repetir, en el orden de
    prioridad de `resolver`."""
    vistos: dict[str, None] = {}
    for d in (plano, settings.project_root() / clase):
        if d.exists():
            for x in sorted(d.iterdir()):
                if x.is_dir():
                    vistos.setdefault(x.name, None)
    for n in (_indice().get(clase) or {}):
        vistos.setdefault(n, None)
    return list(vistos)
