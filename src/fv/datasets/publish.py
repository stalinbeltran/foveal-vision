"""Publicar una fuente en el repo de DATOS, para que viaje por git.

Por que hace falta
------------------
`/data/sources/` esta en el `.gitignore` de este repo, asi que una maquina
recien hecha se queda con CERO fuentes -- medido el 2026-08-29 en este dev:
`discover_sources()` devolvia 0 y con ello no se puede mirar una imagen, ni
puntuar la metrica de tarea (que se mide contra los parrafos de A), ni revisar a
ojo lo que detecta la red.

Lo que se publica es la fuente REDUCIDA (80x60): sus 1000 PNG son 2,01 MB
(medido el 2026-08-29 sobre `dirty1000-80px-16px-r20260827`, 2,0 KB de media).
La copia grande de 640x480 NO se publica: son 234 MB de cache regenerable, y ya
esta escrito en el README que es desechable.

⚠ EL GUARD, que es la razon de que esto sea un modulo y no un `cp -r`
-----------------------------------------------------------------------
Esta MEDIDO (2026-08-27, `bench_dataset.py`) que **renderizar de nuevo no da el
mismo dato**: el motor de render cambio (`cdn.playwright.dev` da 403 desde nyc1,
asi que se rinde con Google Chrome) y la comprobacion de huella abortaba sola.

Publicar un re-render seria entonces el peor fallo posible de esta pieza: las
imagenes que se revisan a ojo NO serian las que el modelo miro, y no habria un
solo error por el camino -- se veria como "el modelo detecta raro".

Y se puede comprobar EXACTO, sin huellas ni fechas, porque `windows.npz` guarda
las imagenes verbatim (`extract.py:131`, `images[si] = s.load_image()`) y ese
npz SI esta commiteado. Asi que antes de publicar se compara pixel a pixel
contra todos los datasets de ventanas que salieron de esa fuente, y si no
coinciden se ABORTA. Es la R9 aplicada: el dato que no se puede re-derivar se
guarda, pero el no-determinismo sigue vivo y aqui muerde.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from fv import settings
from fv.datasets.loader import SourceDataset, resolve_source


# 50 MB: muy por encima de una fuente reducida (2 MB medidos) y muy por debajo
# de la grande (234 MB). No pretende ser un limite fino, sino separar las dos
# cosas que de verdad se confunden al teclear.
MAX_PUBLICABLE = 50_000_000


class PublishError(ValueError):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def window_datasets_from(source_id: str) -> list[str]:
    """Los datasets de ventanas que salieron de esta fuente."""
    root = settings.window_datasets_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        m = d / "manifest.json"
        if not m.exists():
            continue
        try:
            if json.loads(m.read_text(encoding="utf-8")).get("source_id") == source_id:
                out.append(d.name)
        except json.JSONDecodeError:
            continue
    return out


def verify_against_windows(source_id: str) -> dict:
    """¿Las imagenes de esta fuente son las que el npz commiteado guarda?

    Devuelve un informe; NO lanza. Un dataset sin `windows.npz` (solo manifest)
    no puede comprobar nada y se dice, en vez de contarse como que cuadra --
    "no se pudo mirar" y "mire y cuadra" no se pueden leer igual.
    """
    ds = SourceDataset(source_id)
    comprobados, sin_npz, discrepan = [], [], []
    for nombre in window_datasets_from(source_id):
        npz = settings.window_datasets_root() / nombre / "windows.npz"
        if not npz.exists():
            sin_npz.append(nombre)
            continue
        z = np.load(npz)
        if "images" not in z.files:
            sin_npz.append(nombre)
            continue
        imgs, idx = z["images"], z["images_sample_idx"]
        malas = []
        for fila, sidx in enumerate(idx.tolist()):
            propia = np.asarray(ds.sample_at(int(sidx)).load_image(), dtype=np.uint8)
            if propia.shape != imgs[fila].shape or not np.array_equal(propia, imgs[fila]):
                malas.append(int(sidx))
                if len(malas) >= 5:
                    break
        (discrepan if malas else comprobados).append(
            {"dataset": nombre, "distintas": malas} if malas else nombre)
    return {"comprobados": comprobados, "sin_npz": sin_npz, "discrepan": discrepan,
            "concluyente": bool(comprobados) and not discrepan}


def publish_source(source_id: str, *, verify: bool = True,
                   force: bool = False) -> dict:
    """Copia la fuente al repo de datos. Devuelve el informe."""
    origen = resolve_source(source_id)
    destino_root = settings.published_sources_root()
    if destino_root.resolve() == settings.local_sources_root().resolve():
        raise PublishError(
            "no_data_repo",
            "no hay repo de datos clonado: publicar aqui seria copiar una fuente "
            "sobre si misma",
            "clona foveal-vision-data al lado, o pon FV_DATA_ROOT")

    nombre = origen.name
    destino = destino_root / nombre
    if destino.exists() and destino.resolve() == origen.resolve():
        raise PublishError(
            "already_published",
            f"'{source_id}' YA se resuelve desde el repo de datos: no hay nada que copiar",
            "es la que ya viaja por git")

    informe = {"source": source_id, "name": nombre, "from": str(origen),
               "to": str(destino), "verified": None}
    if verify:
        v = verify_against_windows(source_id)
        informe["verify"] = v
        if v["discrepan"] and not force:
            raise PublishError(
                "source_does_not_match_windows",
                f"las imagenes de '{source_id}' NO son las que guarda el "
                f"windows.npz de {[d['dataset'] for d in v['discrepan']]}: es otro "
                f"render, y publicarlo haria revisar imagenes que el modelo nunca vio",
                "no la publiques: recupera la fuente con la que se extrajo ese "
                "dataset, o publicala con force=True sabiendo que no cuadra")
        informe["verified"] = v["concluyente"]

    # ⚠ El tope NO es una preferencia de estilo: esto acaba en git, y en git lo
    # grande no se puede quitar (queda en el historial para siempre). La fuente
    # reducida son 2 MB; la grande de 640x480 son 234 MB de cache regenerable que
    # el README ya declara desechable. Publicar la equivocada es un error de una
    # tecla que no tiene deshacer.
    peso = sum(f.stat().st_size for f in origen.rglob("*") if f.is_file())
    informe["source_bytes"] = peso
    if peso > MAX_PUBLICABLE and not force:
        raise PublishError(
            "source_too_big",
            f"'{source_id}' ocupa {_humano(peso)} y el tope para publicar es "
            f"{_humano(MAX_PUBLICABLE)}: esto va a git y de git no se quita",
            "publica la fuente REDUCIDA (fv-resize --width 80), que es la que se "
            "entrena; la grande es cache regenerable")

    if destino.exists():
        if not force:
            raise PublishError(
                "destination_exists", f"ya existe {destino}",
                "borralo a mano si de verdad quieres reemplazarlo, o usa force")
        shutil.rmtree(destino)

    (destino / "images").mkdir(parents=True)
    n, total = 0, 0
    for rec in _records(origen):
        rel = rec["image"]
        src = origen / rel
        dst = destino / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += dst.stat().st_size
        n += 1
    shutil.copy2(origen / "labels.jsonl", destino / "labels.jsonl")
    if (origen / "dataset.json").exists():
        shutil.copy2(origen / "dataset.json", destino / "dataset.json")
    informe.update({"images": n, "bytes": total})
    return informe


def _humano(n: int) -> str:
    """`0 MB` en un mensaje que BLOQUEA a alguien no dice nada: la unidad se
    elige por el numero, no por costumbre."""
    return f"{n/1e6:.1f} MB" if n >= 1e6 else f"{n/1e3:.0f} KB"


def _records(root: Path) -> list[dict]:
    lineas = (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lineas if ln.strip()]
