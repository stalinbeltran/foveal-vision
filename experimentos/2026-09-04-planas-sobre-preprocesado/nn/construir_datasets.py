#!/usr/bin/env python3
"""Construye los TRES datasets preprocesados — ANTES de entrenar, no al vuelo.

    python nn/construir_datasets.py --plan            # que haria, sin escribir nada
    python nn/construir_datasets.py --brazo 1k5       # construye uno
    python nn/construir_datasets.py --todos           # los tres
    python nn/construir_datasets.py --comprobar       # ¿casan con su manifiesto?

⚠⚠ NO EJECUTADO TODAVIA (2026-09-04). El dueño pidió parar y no correr nada más, así
   que este fichero está escrito pero **no se ha lanzado ni una vez**. No lo des por
   bueno: `--plan` y `--comprobar` existen precisamente para mirarlo antes de gastar.

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

⚠⚠ LA DECISION ABIERTA: `--activacion`, Y CAMBIA SI EL ESTUDIO ES DEGENERADO
   `aplicaKernel` devuelve el mapa SIN ACTIVAR y con signo (es la capa L1 de esas redes,
   cuya última capa no lleva ReLU a propósito). Guardarlo así o pasarle una ReLU no es un
   detalle de formato:

   · `--activacion ninguna` (lo literal): el dataset guarda el mapa con signo. Entonces
     la plana que entrene encima hace `conv(conv(x))` SIN no-linealidad en medio, y eso
     es **una sola convolución** de tamaño `kf+k2-1` con los pesos atados. O sea que cada
     brazo sería un SUBCONJUNTO ESTRICTO de un gemelo ya corrido:

         1k3 + plana k=3  ==  una 5x5 atada  ->  gemelo libre `1k5 crudo`, f1 0,642
         1k5 + plana k=3  ==  una 7x7 atada  ->  gemelo libre `1k7 crudo`, f1 0,618
         1k7 + plana k=3  ==  una 9x9 atada  ->  no existe gemelo

     Sólo puede EMPATAR O PERDER contra un número ya pagado. Sigue siendo una pregunta
     legítima («¿cuánto cuesta congelar y factorizar?») pero no puede salir a favor.

   · `--activacion relu`: el dataset guarda `max(0, mapa)`. Deja de colapsar y el
     preproceso pasa a ser un extractor de rasgos de verdad. Es lo que hacía el
     experimento anterior, y por eso sus tres brazos sirven de comparación.

   **No se elige aquí por defecto**: es una decisión del dueño y va escrita en el
   manifiesto de cada dataset, porque dos datasets construidos con distinta activación no
   son comparables y el fichero no lo diría por su nombre.

⚠ EL CANAL DE RELLENO SE PIERDE, y está medido que vale
   El kernel consume `(vista, relleno)` y devuelve UN mapa, así que aguas abajo la red ya
   no puede pesar el relleno por su cuenta. El reporte #19 midió que ese canal sube el
   recall del último píxel de 0,608 a 0,974. `--con-relleno` lo conserva como segundo
   canal del dataset; por defecto NO se conserva, que es la lectura literal del encargo.

DONDE SE GUARDA, Y POR QUE NO EN GIT
   ~435 MB en float32 los tres, contra un repo de datos de 197 MiB. Y son
   **re-derivables exactamente** de (kernel commiteado + dataset origen commiteado +
   opciones), así que la regla 4 de `experimentos/README.md` dice que se enlazan, no se
   guardan. Lo que SI se commitea es este script y el `manifiesto.json` de cada uno: con
   eso, una máquina nueva los reconstruye en ~1-2 min.
   ⚠ Este server es efímero. Reconstruir NO es opcional al rehacerlo: es un paso del
   arranque, y por eso `--comprobar` distingue «no está» de «está y no casa».
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
from preproceso import CARPETAS, aplicaKernel, cargar_kernel     # noqa: E402

# Los MISMOS que los siete gemelos: si el dataset origen cambia, esto no es
# comparable con nada de la serie.
DATASET = "dirty1000-80px-16px-r20260827"
RED_GEOMETRIA = "plana-20-1k3"      # de aquí sale la vista 20x20 (fovea 16 + borde 8/4)
PESOS = "best"                      # de un preprocesador se quiere el mejor estado

# Fuera de git a propósito (ver cabecera). Bajo el repo de CODIGO y no el de datos
# para que un `git status` del repo de datos no se llene de 435 MB ignorados.
DESTINO = REPO / "data" / "preprocesado"


def _huella(kern, activacion: str, con_relleno: bool, origen_fp: str) -> str:
    """Identidad del dataset: si cambia cualquier ingrediente, cambia la huella.

    Incluye los PESOS del kernel (no su nombre): un `best.pt` reentrenado daría otro
    dataset con el mismo nombre, y eso es exactamente lo que no puede pasar
    inadvertido.
    """
    h = hashlib.sha256()
    h.update(kern.peso.numpy().tobytes())
    h.update(kern.sesgo.numpy().tobytes())
    h.update(f"{activacion}|{con_relleno}|{origen_fp}|{DATASET}".encode())
    return h.hexdigest()


def _origen():
    ruta = Path(settings.window_datasets_root()) / DATASET
    z = np.load(ruta / "windows.npz")
    fp = ""
    man = ruta / "manifest.json"
    if man.exists():
        fp = json.loads(man.read_text()).get("fingerprint", "")
    return z, fp


def plan(brazos, activacion: str, con_relleno: bool) -> int:
    """Qué se construiría y cuánto ocupa. NO escribe nada."""
    z, fp = _origen()
    n = int(z["y"].shape[0])
    cfg = full_config(yaml.safe_load(
        (REPO / "configs" / "networks" / f"{RED_GEOMETRIA}.yaml").read_text()))
    d = derive_dims(cfg)
    print(f"origen   : {DATASET}  ({n} ventanas · vista {d.N}x{d.N} de una ventana "
          f"de {d.original_size}x{d.original_size} px)")
    print(f"huella   : {fp[:24] or '(sin manifest)'}...")
    print(f"activacion: {activacion}   ·   canal de relleno: "
          f"{'SI (2 canales)' if con_relleno else 'no (1 canal)'}")
    print(f"destino  : {DESTINO}   (FUERA de git)\n")
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
    print(f"\n⚠ NO commiteado: se reconstruye con este script. Lo que va a git es el "
          f"script y el manifiesto.")
    return 0


def construir(brazo: str, activacion: str, con_relleno: bool) -> int:
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
        mapa = aplicaKernel(x, kern, escala="0-1")               # (1, lado, lado)
        if activacion == "relu":
            mapa = torch.relu(mapa)
        elif activacion != "ninguna":
            raise SystemExit(f"--activacion '{activacion}': usa 'ninguna' o 'relu'")
        if con_relleno:
            # el relleno RECORTADO al centro valido, para que case con el mapa
            r = kern.k // 2
            relleno = x[1:2, r:d.N - r, r:d.N - r]
            mapa = torch.cat([mapa, relleno], dim=0)
        salida[i] = mapa.numpy()
        if i % 20000 == 0:
            print(f"  {i}/{n}", flush=True)

    dest = DESTINO / f"{brazo}-{activacion}{'-relleno' if con_relleno else ''}"
    dest.mkdir(parents=True, exist_ok=True)
    # SIN comprimir a propósito: son floats casi incompresibles, comprimir cuesta
    # minutos y no va a git de todas formas (73 GB libres, medido 2026-08-28). Si
    # algún día hubiera que moverlo a otra máquina, se comprime al enviarlo.
    np.savez(dest / "preprocesado.npz", x=salida, y=y, split=split)
    manifiesto = {
        "brazo": brazo, "kernel": CARPETAS[brazo], "pesos": PESOS,
        "k": kern.k, "activacion": activacion, "con_relleno": con_relleno,
        "dataset_origen": DATASET, "fingerprint_origen": fp,
        "forma": list(salida.shape), "huella": _huella(kern, activacion, con_relleno, fp),
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
            esperada = _huella(kern, d["activacion"], d["con_relleno"], fp)
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
    p.add_argument("--activacion", default="ninguna", choices=("ninguna", "relu"),
                   help="ver la cabecera: cambia si el estudio es degenerado")
    p.add_argument("--con-relleno", action="store_true",
                   help="conserva el canal de relleno recortado como 2º canal")
    a = p.parse_args()
    brazos = [a.brazo] if a.brazo else sorted(CARPETAS)
    if a.plan:
        return plan(brazos, a.activacion, a.con_relleno)
    if a.comprobar:
        return comprobar(brazos)
    if a.brazo or a.todos:
        for b in brazos:
            construir(b, a.activacion, a.con_relleno)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
