#!/usr/bin/env python3
"""Un PNG por dataset con LAS MISMAS 10 ventanas — leidas DEL dataset construido.

    python nn/muestra.py                 # los tres
    python nn/muestra.py --brazo 1k5

QUE ENSENA, Y POR QUE ESO
   Una fila por ventana: a la izquierda la vista 20x20 que entro, a la derecha el
   mapa que el dataset guarda de verdad. O sea que la imagen no ilustra lo que el
   preproceso deberia hacer: ensena lo que hay ESCRITO EN EL FICHERO.

⚠⚠ SE LEE DEL `.npz` CONSTRUIDO, NO SE RECALCULA
   Es la diferencia entre una figura y una comprobacion. Si `construir_datasets.py`
   tuviera un fallo --el lookup de imagenes, el orden de las filas, la escala-- una
   figura recalculada saldria perfecta y el dataset seguiria roto. Leyendo del
   fichero, el PNG es evidencia de su contenido.

LAS MISMAS 10 EN LOS TRES, Y ESO NO ES CASUALIDAD
   Los indices salen de `comun/set-visualizacion.json`, el set congelado que
   comparten los siete gemelos (elegido una vez con semilla 2026 sobre el split de
   validacion). Por eso las tres imagenes se pueden poner una al lado de otra: es
   la misma ventana en la misma fila. Si cada dataset eligiera su muestra, comparar
   las figuras no significaria nada.

DONDE SE ESCRIBE, Y POR QUE EN DOS SITIOS
   · `muestras/<brazo>.png` en el experimento -> **va a git**. Es lo que sobrevive a
     rehacer la maquina, que es lo unico que cuenta aqui (~40 KB).
   · `data/preprocesado/<dataset>/muestra.png` -> junto al dataset, para mirarlo sin
     salir de su carpeta. FUERA de git, como el resto de esa carpeta.
   Se escriben en la MISMA pasada desde el MISMO array, asi que no pueden divergir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
COMUN = EXP.parent / "comun"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(COMUN))
sys.path.insert(0, str(AQUI))

import aplicar_kernels as ak                                     # noqa: E402
from construir_datasets import DESTINO, RED_GEOMETRIA            # noqa: E402
from preproceso import CARPETAS, cargar_kernel                   # noqa: E402

MUESTRAS = EXP / "muestras"


def _indices() -> list[int]:
    """Los 10 del set congelado que comparten los siete gemelos."""
    d = json.loads((COMUN / "set-visualizacion.json").read_text())
    return [int(v["indice"]) for v in d["ventanas"]]


def una(brazo: str) -> int:
    carpetas = sorted(DESTINO.glob(f"{brazo}-*"))
    if not carpetas:
        print(f"  {brazo}: NO ESTA construido — nn/construir_datasets.py --brazo {brazo}")
        return 1
    carpeta = carpetas[0]
    man = json.loads((carpeta / "manifiesto.json").read_text())
    z = np.load(carpeta / "preprocesado.npz")
    x = z["x"]                                    # (N, C, H, W) — TAL CUAL en disco

    idx = _indices()
    mapas = x[idx]                                # (10, C, H, W)

    # La vista 20x20 de esas mismas ventanas, para la columna de la izquierda.
    # Sale de `comun/`, o sea de la misma funcion que uso el entrenamiento.
    ak.RED = REPO / "configs" / "networks" / f"{RED_GEOMETRIA}.yaml"
    _x, _e, vistas = ak.entradas(ak.set_visualizacion(10, 2026))

    kern = cargar_kernel(brazo, pesos=man["pesos"])
    lado = mapas.shape[-1]
    titulo = (f"{brazo}: dataset preprocesado {lado}x{lado} "
              f"(kernel {kern.k}x{kern.k} congelado, sin relleno, {man['activacion']})")
    pie = (f"leido de {carpeta.name}/preprocesado.npz · {x.shape[0]} ventanas · "
           f"las 10 del set congelado (semilla 2026) · ceros por la ReLU: "
           f"{100.0 * float((mapas == 0).mean()):.0f} %")

    MUESTRAS.mkdir(parents=True, exist_ok=True)
    destino = MUESTRAS / f"{brazo}.png"
    # ⚠ el pie por defecto dice «sin activar, con signo», que vale para los gemelos
    #   y aqui seria FALSO: estos mapas vienen de `aplicaKernel`, o sea activados.
    ak.montaje(vistas, mapas, destino, titulo, pie_extra=pie,
               nota_escala=f"ACTIVADO ({man['activacion']}): no hay negativos")
    # la copia de al lado del dataset, del MISMO array y en la misma pasada
    shutil.copyfile(destino, carpeta / "muestra.png")
    print(f"  {brazo}: {destino.relative_to(REPO)}  ({mapas.shape[0]} ventanas de "
          f"{lado}x{lado}, min {mapas.min():.3f} max {mapas.max():.3f})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brazo", choices=sorted(CARPETAS))
    a = p.parse_args()
    malo = 0
    for b in ([a.brazo] if a.brazo else sorted(CARPETAS)):
        malo |= una(b)
    return malo


if __name__ == "__main__":
    raise SystemExit(main())
