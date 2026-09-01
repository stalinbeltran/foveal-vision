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
    """Donde viven los datasets de ventanas -- CON su `windows.npz`.

    Desde 2026-08-27 en el repo de DATOS, y el npz se COMMITEA. El motivo es que
    lo contrario esta medido y salio mal: el `.gitignore` de `foveal-vision-data`
    excluia `*.npz` como "artefacto regenerable", y `repro-chk` demostro el
    2026-08-26 que un `windows.npz` reconstruido NO es el mismo dato (curvas
    distintas con el mismo punto, la misma semilla y la misma CPU). Un dato que
    no se puede re-derivar y no se guarda es un dato que se pierde: se perdio el
    `r20260824`, y con el la comparabilidad de 20 runs ya pagados.

    Sin repo de datos clonado cae a `<codigo>/data/window-datasets`, que es
    donde el payload de la flota desempaqueta los datasets en las maquinas
    alquiladas. Alli NO hay repo de datos y no debe haberlo: reciben el dato
    hecho, no lo buscan (ver `estudio_flota.construir_payload`).
    """
    d = data_root()
    return (d / "window-datasets") if d != project_root() \
        else project_root() / "data" / "window-datasets"


def published_sources_root() -> Path:
    """Fuentes PUBLICADAS: las que viajan por git en el repo de datos.

    Existe porque `/data/sources/` esta en el `.gitignore` de este repo (son
    renders regenerables, 234 MB la grande), asi que una maquina recien hecha se
    queda SIN una sola fuente -- y sin fuente no se puede mirar una imagen ni
    medir la metrica de tarea, que se puntua contra los parrafos de A.

    Lo que se publica es la fuente REDUCIDA (80x60): medido el 2026-08-29, sus
    1000 PNG ocupan 2,01 MB (2,0 KB de media). La grande NO se publica, que sigue
    siendo cache regenerable.

    ⚠ Mismo prefijo `local/` que `local_sources_root()`, a proposito: el id de
    una fuente esta escrito en el `source_id` de cada manifest de ventanas (18
    hoy). Publicarla bajo otro prefijo la haria invisible para todos ellos.

    Sin repo de datos clonado cae a `local_sources_root()`, y entonces las dos
    raices son la misma y `_roots()` la cuenta una sola vez.
    """
    d = data_root()
    return (d / "sources") if d != project_root() else local_sources_root()


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


def errores_root() -> Path:
    """Donde vive el LOG DE ERRORES: lo que se rompio y nadie estaba mirando.

    En el repo de DATOS y no en el de codigo (R7): lo produce el sistema al
    correr, no el que lo escribio, y **un error que ya ocurrio no se puede
    re-derivar** (R9) -- si se pierde con la maquina, la unica forma de volver a
    tenerlo es que vuelva a fallar.

    Es historial puro: solo se anade, un fichero por mes, nunca se reescribe
    (R8). El estado --que esta roto AHORA-- lo contesta el autochequeo de
    arranque, que es otra cosa y vive en otro sitio.

    ⚠ El repo es PRIVADO desde el 2026-09-01, y esta ruta lo da por hecho: aqui
    caen mensajes y trazas, que llevan rutas, nombres y valores. Aun asi se
    redacta por patron antes de escribir, porque "privado" es un permiso, no un
    borrado: git no olvida y un secreto que se cuela hay que ROTARLO.
    """
    return data_root() / "errores"


def reviews_root() -> Path:
    """Donde vive la REVISION A OJO: que rangos ya se miraron y que imagenes
    quedaron marcadas para volver.

    En el repo de DATOS, como runs/sweeps/studies, y por el mismo motivo (R7): lo
    produce quien mira, no quien transporta, y el juicio de "esta salio mal" no
    se puede re-derivar de nada -- si se pierde, se vuelve a mirar a mano.

    Sin repo de datos clonado cae al repo de codigo, igual que `data_root()`:
    degrada en vez de romper (R2), pero ahi NO se commitea nada. Quien lo lea
    debe decirlo -- ver `review.donde_se_guarda`.
    """
    return data_root() / "reviews"


def cache_root() -> Path:
    return project_root() / "data" / "cache"


def inference_staging_root() -> Path:
    """El ANTESALA de los pesos: donde aterrizan mientras se entrena.

    Va en el repo de CODIGO y fuera de git (`/data/inferencia/` en el
    .gitignore), no en el repo de datos, y esa es toda la decision:

      - git guarda TODAS las versiones que se commitean. Un `last.pt` son 2,0 MB
        y una sonda cada pocas epocas serian decenas de versiones por run: el
        `.gitignore` del repo de datos ya avisa de que eso son gigabytes por
        barrido. Lo de en medio no puede tocar el repo de datos.
      - y aun asi hay que poder MIRAR el modelo con el entrenamiento en marcha,
        que es justo para lo que `entrenar_vast.py` se trae los pesos en cada
        sonda. La antesala existe para eso.

    ⚠ NO es `cache_root()`, aunque se le parezca. La cache se puede borrar sin
    perder nada; esto guarda los unicos pesos de un entrenamiento en curso, y
    borrarlo a mitad pierde horas de maquina. Un directorio con otra regla de
    borrado es otro directorio.

    Lo definitivo vive en el repo de DATOS y solo llega ahi por una promocion
    explicita (`fv.inference.catalogo`).
    """
    return project_root() / "data" / "inferencia"


def ui_state_path() -> Path:
    # Remembered UI defaults (filters + form values). Committable so it travels
    # with the repo to the GPU server. NOT a domain artifact: an opaque blob of
    # conveniences, never a source of truth for A-H.
    return project_root() / "state" / "ui-state.json"
