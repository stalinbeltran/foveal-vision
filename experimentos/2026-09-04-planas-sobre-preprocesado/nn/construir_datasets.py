#!/usr/bin/env python3
"""Construye los TRES datasets preprocesados — ANTES de entrenar, no al vuelo.

    python nn/construir_datasets.py --plan            # que haria, sin escribir nada
    python nn/construir_datasets.py --brazo 1k5       # construye uno
    python nn/construir_datasets.py --todos           # los tres
    python nn/construir_datasets.py --comprobar       # ¿casan con su manifiesto?

⚠ ESTADO (2026-09-04): los tres CONSTRUIDOS y commiteados en el repo de datos.
   `--comprobar` los valida contra su manifiesto. 1 min 20 s cada uno, 0 $.

POR QUE EXISTE ESTE FICHERO, Y QUE ARREGLA
   El experimento anterior (`2026-09-04-preproceso-kernel-congelado`) metió el kernel
   congelado DENTRO del modelo como capa 0. Numéricamente daba el mismo tensor, pero
   cambiaba lo que se medía:

     - la entrada de la red seguía siendo la vista `(2,20,20)` de siempre;
     - la red pasaba a tener DOS convoluciones, o sea que dejaba de ser «plana»;
     - y el preproceso dejaba de ser un PASO PREVIO para ser parte de la red.

   El encargo pedía tres **datasets preprocesados** y tres **CNN planas**. Aquí el
   preproceso se materializa antes, y lo que entrena después es una plana de verdad:
   UNA convolución y su cabeza.

QUE SE GUARDA, EXACTAMENTE
   Por cada ventana del dataset origen se construye su vista foveada `(2,20,20)`
   —exactamente como la construye el entrenamiento de siempre, `build_view` +
   `input_stack`— y se le aplica `aplicaKernel_1k<kf>` sin relleno:

       (2, 20, 20)  ->  (1, 20-kf+1, 20-kf+1)

       kf=3 -> 18x18      kf=5 -> 16x16      kf=7 -> 14x14

   ⚠ La vista se construye con la COBERTURA REAL del dataset, no con el defecto
   `relleno=0` que pone `aplicaKernel` sobre una imagen suelta. Es la diferencia entre
   reproducir lo que vio la L1 original y aproximarlo; equivocarse ahí no falla, sale
   otro número.

LA NO-LINEALIDAD LA APLICA `aplicaKernel`, NO ESTE FICHERO
   Orden del dueno (2026-09-04): «el dataset debe ser generado con las funciones que
   aplican kernel, y esas funciones ya deben aplicar la no-linearidad». Asi que aqui
   NO hay ningun flag de activacion: se llama a `aplicaKernel` y punto, y lo que
   salga es lo que el preprocesador define como su salida.

   Hoy eso es `relu` (`preproceso.ACTIVACION_POR_DEFECTO`), y queda escrito en el
   manifiesto de cada dataset -- no porque se pueda elegir desde aqui, sino porque un
   dataset construido con otra activacion no seria comparable y el nombre del fichero
   no lo diria.

   ⚠ Por que importa: sin no-linealidad el preproceso es una operacion LINEAL, asi que
   la plana que entrene encima haria `conv(conv(x))` sin nada en medio -- una sola
   convolucion de tamano `k+2` con los pesos atados-- y cada brazo seria un subconjunto
   estricto de un gemelo ya corrido, capaz solo de empatar o perder. Con la ReLU dentro
   del preproceso eso no pasa. El detalle esta en el docstring de `aplicaKernel`.

   ⚠ Y su precio, medido el 2026-09-04 sobre las 10 ventanas del set congelado: la ReLU
   tira el 15 % de las celdas en 1k3, el 13 % en 1k5 y el 7 % en 1k7.

⚠ EL CANAL DE RELLENO SE PIERDE, y está medido que vale
   El kernel consume `(vista, relleno)` y devuelve UN mapa, así que aguas abajo la red ya
   no puede pesar el relleno por su cuenta. El reporte #19 midió que ese canal sube el
   recall del último píxel de 0,608 a 0,974. `--con-relleno` lo conserva como segundo
   canal del dataset; por defecto NO se conserva, que es la lectura literal del encargo.

DONDE SE GUARDA: EN EL REPO DE DATOS, Y COMMITEADO
   Orden del dueno (2026-09-04): «has push de los datasets». Van a
   `foveal-vision-data/preprocesado/<brazo>-<activacion>/`, con su manifiesto y su
   muestra al lado.

   ⚠ Lo propuesto era lo contrario --enlazarlos en vez de guardarlos, por la regla 4 de
   `experimentos/README.md`: son re-derivables exactamente en 1 min 20 s cada uno-- y el
   dueno decidio guardarlos. Queda anotado aqui con su motivo, no dado por inaplicable.

   ⚠⚠ Y CABEN SOLO PORQUE SE COMPRIMEN. GitHub rechaza ficheros de mas de 100 MB y en
   crudo miden 180/144/112 MB. Comprimidos son ~56/44/34 MB (medido: el 1k7 pasa de
   109,8 a 33,9 MB, 3,2x, en 3 s). Si alguien vuelve a `np.savez` a secas, el push se
   rechaza -- por eso esta escrito aqui y no solo en el commit.

   ⚠ `*.npz` esta ignorado en el repo de datos; hace falta la excepcion
   `!/preprocesado/*/preprocesado.npz`, como la que ya existe para `window-datasets`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(EXP.parent / "comun"))

from fv import settings                                          # noqa: E402
from fv.fovea import build_view, derive_dims, input_stack        # noqa: E402
from fv.models.builder import full_config                        # noqa: E402
from preproceso import (ACTIVACION_POR_DEFECTO, CARPETAS,        # noqa: E402
                        aplicaKernel, cargar_kernel)

# Los MISMOS que los siete gemelos: si el dataset origen cambia, esto no es
# comparable con nada de la serie.
DATASET = "dirty1000-80px-16px-r20260827"
RED_GEOMETRIA = "plana-20-1k3"      # de aquí sale la vista 20x20 (fovea 16 + borde 8/4)
PESOS = "best"                      # de un preprocesador se quiere el mejor estado

# EN el repo de DATOS y COMMITEADOS, por orden del dueno (2026-09-04: «has push de
# los datasets»). Ver la cabecera: no era el plan y el plan estaba razonado, pero es
# su decision y aqui esta lo que hace falta para que quepa.
DESTINO = Path(settings.data_root()) / "preprocesado"


def _huella(kern, con_relleno: bool, origen_fp: str) -> str:
    """Identidad del dataset: si cambia cualquier ingrediente, cambia la huella.

    Incluye los PESOS del kernel (no su nombre): un `best.pt` reentrenado daría otro
    dataset con el mismo nombre, y eso es exactamente lo que no puede pasar
    inadvertido.
    """
    h = hashlib.sha256()
    h.update(kern.peso.numpy().tobytes())
    h.update(kern.sesgo.numpy().tobytes())
    h.update(f"{ACTIVACION_POR_DEFECTO}|{con_relleno}|{origen_fp}|{DATASET}".encode())
    return h.hexdigest()


def _origen():
    ruta = Path(settings.window_datasets_root()) / DATASET
    z = np.load(ruta / "windows.npz")
    fp = ""
    man = ruta / "manifest.json"
    if man.exists():
        fp = json.loads(man.read_text()).get("fingerprint", "")
    return z, fp


def plan(brazos, con_relleno: bool) -> int:
    """Qué se construiría y cuánto ocupa. NO escribe nada."""
    z, fp = _origen()
    n = int(z["y"].shape[0])
    cfg = full_config(yaml.safe_load(
        (REPO / "configs" / "networks" / f"{RED_GEOMETRIA}.yaml").read_text()))
    d = derive_dims(cfg)
    print(f"origen   : {DATASET}  ({n} ventanas · vista {d.N}x{d.N} de una ventana "
          f"de {d.original_size}x{d.original_size} px)")
    print(f"huella   : {fp[:24] or '(sin manifest)'}...")
    print(f"activacion: {ACTIVACION_POR_DEFECTO} (la aplica aplicaKernel)   ·   canal de relleno: "
          f"{'SI (2 canales)' if con_relleno else 'no (1 canal)'}")
    print(f"destino  : {DESTINO}   (repo de DATOS, commiteado)\n")
    total = 0
    print(f"{'brazo':6} {'kernel':8} {'salida':>12} {'canales':>8} {'MB (f32)':>9}")
    for b in brazos:
        kern = cargar_kernel(b, pesos=PESOS)
        lado = d.N - kern.k + 1
        canales = 2 if con_relleno else 1
        mb = n * canales * lado * lado * 4 / 1e6
        total += mb
        print(f"{b:6} {kern.k}x{kern.k:<6} {f'{lado}x{lado}':>12} {canales:>8} {mb:>9.1f}")
    print(f"{'':6} {'':8} {'':>12} {'TOTAL':>8} {total:>9.1f}")
    print(f"\n⚠ En crudo son {total:.0f} MB; se escriben COMPRIMIDOS (~3,2x) porque "
          f"GitHub rechaza ficheros de mas de 100 MB.")
    return 0


def construir(brazo: str, con_relleno: bool) -> int:
    z, fp = _origen()
    cfg = full_config(yaml.safe_load(
        (REPO / "configs" / "networks" / f"{RED_GEOMETRIA}.yaml").read_text()))
    d = derive_dims(cfg)
    kern = cargar_kernel(brazo, pesos=PESOS)

    y, sample_idx = z["y"], z["sample_idx"]
    window_xy, split, images = z["window_xy"], z["split"], z["images"]
    # `sample_idx` NO indexa `images`: hay que pasar por `images_sample_idx`. Es el
    # mismo lookup que hace `FoveatedWindowDataset.__init__`, y saltárselo daría
    # ventanas de OTRA imagen sin fallar.
    lookup = {int(a): i for i, a in enumerate(z["images_sample_idx"])}

    n = int(y.shape[0])
    lado = d.N - kern.k + 1
    canales = 2 if con_relleno else 1
    salida = np.empty((n, canales, lado, lado), dtype=np.float32)

    for i in range(n):
        img = images[lookup[int(sample_idx[i])]]
        wx0, wy0 = int(window_xy[i, 0]), int(window_xy[i, 1])
        # EXACTAMENTE como lo hace el entrenamiento de siempre: misma vista, misma
        # cobertura. Si esto se desvía, el dataset deja de ser «el mismo de los
        # experimentos anteriores» y la comparación se rompe en silencio.
        vista, cov = build_view(img, wx0, wy0, d,
                                pool_mode=cfg["pool_mode"], pad_mode=cfg["pad_mode"])
        x = torch.from_numpy(input_stack(vista, cov, cfg["mask_channel"]))
        # SIN tocar la activacion: la aplica `aplicaKernel`, que es el encargo.
        mapa = aplicaKernel(x, kern, escala="0-1")               # (1, lado, lado)
        if con_relleno:
            # el relleno RECORTADO al centro valido, para que case con el mapa
            r = kern.k // 2
            relleno = x[1:2, r:d.N - r, r:d.N - r]
            mapa = torch.cat([mapa, relleno], dim=0)
        salida[i] = mapa.numpy()
        if i % 20000 == 0:
            print(f"  {i}/{n}", flush=True)

    dest = DESTINO / f"{brazo}-{ACTIVACION_POR_DEFECTO}{'-relleno' if con_relleno else ''}"
    dest.mkdir(parents=True, exist_ok=True)
    # COMPRIMIDO, y no es por ahorrar disco: GitHub RECHAZA cualquier fichero de mas
    # de 100 MB, y en crudo los tres miden 180/144/112 MB. Medido el 2026-09-04 sobre
    # el 1k7: 109,8 -> 33,9 MB, o sea 3,2x, y en 3 s. Comprime tanto porque el 12-15 %
    # de las celdas son CERO exacto (la ReLU) y el resto es suave.
    # ⚠ float32, NO float16: la mitad de tamano sonaba bien y habria cambiado el dato
    #   de entrada del estudio (perdida maxima medida 1,2e-04). Comprimir no pierde
    #   un solo bit y ya basta para caber.
    np.savez_compressed(dest / "preprocesado.npz", x=salida, y=y, split=split)
    manifiesto = {
        "brazo": brazo, "kernel": CARPETAS[brazo], "pesos": PESOS,
        "k": kern.k, "activacion": ACTIVACION_POR_DEFECTO,
        "con_relleno": con_relleno,
        "dataset_origen": DATASET, "fingerprint_origen": fp,
        "forma": list(salida.shape), "huella": _huella(kern, con_relleno, fp),
        "construido_por": "nn/construir_datasets.py",
    }
    (dest / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2))
    print(f"escrito {dest}  {salida.shape}  ({salida.nbytes/1e6:.1f} MB en RAM)")
    return 0


def comprobar(brazos) -> int:
    """¿Está, y es el que dice ser? Distingue «no está» de «está y no casa»."""
    _z, fp = _origen()
    malo = False
    for b in brazos:
        kern = cargar_kernel(b, pesos=PESOS)
        encontrados = list(DESTINO.glob(f"{b}-*/manifiesto.json"))
        if not encontrados:
            print(f"  {b}: NO ESTA — reconstruye con --brazo {b}")
            malo = True
            continue
        for m in encontrados:
            d = json.loads(m.read_text())
            esperada = _huella(kern, d["con_relleno"], fp)
            casa = esperada == d.get("huella")
            malo |= not casa
            print(f"  {b} [{d['activacion']}"
                  f"{'+relleno' if d['con_relleno'] else ''}]: "
                  f"{'✓ casa' if casa else '✗ NO CASA con su manifiesto (kernel o dataset origen cambiaron)'}")
    return 1 if malo else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brazo", choices=sorted(CARPETAS))
    p.add_argument("--todos", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--con-relleno", action="store_true",
                   help="conserva el canal de relleno recortado como 2º canal")
    a = p.parse_args()
    brazos = [a.brazo] if a.brazo else sorted(CARPETAS)
    if a.plan:
        return plan(brazos, a.con_relleno)
    if a.comprobar:
        return comprobar(brazos)
    if a.brazo or a.todos:
        for b in brazos:
            construir(b, a.con_relleno)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
