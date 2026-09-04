#!/usr/bin/env python3
"""`aplicaKernel`: la PRIMERA CAPA de las tres planas de un kernel, suelta.

    python experimentos/comun/preproceso.py --comprobar   # contra los mapas.npy guardados

Encargo del dueno (2026-09-04): «de estos 3 experimentos toma sus kernels, y
para cada uno crea una funcion aplicaKernel que tome una entrada cualquiera
(imagen, como las que empleamos en nuestros entrenamientos) y le aplique este
kernel sin padding. La salida de esta funcion sera luego empleada (opcionalmente)
como pre-procesador de las imagenes de entrada».

    aplicaKernel_1k3(x)   kernel 3x3 de `2026-09-04-cnn-plana-1k3-sinpadding`
    aplicaKernel_1k5(x)   kernel 5x5 de `2026-09-04-cnn-plana-1k5-sinpadding`
    aplicaKernel_1k7(x)   kernel 7x7 de `2026-09-04-cnn-plana-1k7-sinpadding`

QUE ES EXACTAMENTE, Y COMO SE COMPRUEBA QUE LO ES
   Las tres redes son `regions: single` y `n_layers: 1`, asi que su unica
   convolucion ve la entrada ENTERA sin mascara (`builder.py:_branches`) y su
   mapa se queda SIN activar --la ultima capa no lleva ReLU a proposito
   (`builder.py:197`)--. O sea que `aplicaKernel` no es «algo parecido a lo que
   hacia la red»: es literalmente su capa L1.

   Y eso no se afirma, se comprueba: `--comprobar` pasa las MISMAS 10 ventanas
   del set congelado y contrasta contra el `mapas.npy` que cada experimento dejo
   escrito en su `stop-04`, calculado en su momento con el modelo vivo. Si no
   casan, este fichero miente. Es la misma vara que usa `cargar_pesos.py` con las
   normas L2, aplicada a la salida en vez de a los pesos.

POR QUE VIVE EN `comun/` Y NO DENTRO DE UN EXPERIMENTO
   El mismo motivo que `aplicar_kernels.py`: los tres son GEMELOS y lo unico que
   cambia entre ellos es `k`. Si la salida de los tres va a compararse como
   preprocesador, la operacion tiene que ser LA MISMA; tres copias derivarian y
   la comparacion seria una ilusion sin que nada fallara.
   ⚠ Choca de refilon con el antipatron de la R1 de `reglas-de-diseno.md`
   («`comun/`, `utils/`, `core/`: acaban siendo el sitio donde va lo que no se
   sabe donde poner»). Se anota en vez de darlo por inaplicable: aqui la carpeta
   tiene un criterio comprobable --lo que los gemelos COMPARTEN para poder
   compararse, y `--comprobar` lo verifica-- y no un nombre de capa.

NO DEPENDE DE `fv` NI DE `red_local.py` (R3: una pieza que no se puede usar sola
no es una pieza). Se lee el tensor del `.pt` y la geometria del `config.json` que
esta a su lado; nada mas. Se puede copiar este fichero y la carpeta `nn/pesos/`
de un experimento a otro sitio y sigue funcionando.

⚠ ES OPCIONAL, y por eso no toca produccion. `src/fv/` sigue intacto: la
instruccion del dueno (2026-09-03) es que un experimento no cambia el codigo de
produccion hasta que el numero lo respalde. Esto es material para decidirlo.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent          # experimentos/comun
EXPERIMENTOS = AQUI.parent

# El acoplamiento se DECLARA (R4): la carpeta de cada kernel esta escrita aqui,
# no se deduce buscando por el disco. `cargar_kernel` acepta ademas una ruta
# suelta, que es lo que permite usar esto con un experimento que aun no este en
# esta tabla.
CARPETAS = {
    "1k3": "2026-09-04-cnn-plana-1k3-sinpadding",
    "1k5": "2026-09-04-cnn-plana-1k5-sinpadding",
    "1k7": "2026-09-04-cnn-plana-1k7-sinpadding",
}

# El maximo que puede traer una imagen ya normalizada. 1.0 con holgura para el
# error de redondeo de un /255 hecho en float32.
MAX_NORMALIZADA = 1.5

__all__ = ["KernelEntrenado", "cargar_kernel", "aplicaKernel",
           "aplicaKernel_1k3", "aplicaKernel_1k5", "aplicaKernel_1k7",
           "CARPETAS"]


# La no-linealidad va DENTRO del preproceso, por orden del dueno (2026-09-04).
# El porque -- y por que 'ninguna' sigue existiendo -- en el docstring de
# `aplicaKernel`. Es `relu` y no otra porque es la que usa la red en todas partes
# (`builder.py:_branch_forward` entre capas, y `F.relu(feat)` antes de la cabeza):
# un preproceso que activara distinto que la red no seria «su capa L1 activada».
ACTIVACION_POR_DEFECTO = "relu"
ACTIVACIONES = ("relu", "ninguna")


class PreprocesoError(ValueError):
    """Lo que esta funcion se niega a adivinar. Ver R2: o defecto declarado, o
    fallar ANTES de empezar; nunca a mitad."""


def _activar(y: torch.Tensor, activacion: str) -> torch.Tensor:
    if activacion == "relu":
        return torch.relu(y)
    if activacion == "ninguna":
        return y
    raise PreprocesoError(
        f"activacion '{activacion}': usa {' o '.join(repr(a) for a in ACTIVACIONES)}")


@dataclass(frozen=True)
class KernelEntrenado:
    """El kernel de un experimento, con lo que hace falta para volver a aplicarlo.

    `peso` es (K, C, k, k) y `sesgo` (K,) --K=1 en los tres--, tal cual salieron
    de `center_convs.0`. `canales` es C: **2** en los tres (la vista y el
    RELLENO), y saberlo importa porque una imagen suelta trae uno solo.
    """

    nombre: str
    carpeta: Path
    checkpoint: Path
    peso: torch.Tensor
    sesgo: torch.Tensor
    stride: int
    epoca: object
    dataset: str | None

    @property
    def k(self) -> int:
        return int(self.peso.shape[-1])

    @property
    def canales(self) -> int:
        return int(self.peso.shape[1])

    @property
    def n_kernels(self) -> int:
        return int(self.peso.shape[0])

    def __str__(self) -> str:
        return (f"{self.nombre}: {self.n_kernels}x{self.canales}x{self.k}x{self.k}"
                f" · stride {self.stride} · {self.checkpoint.name} epoca {self.epoca}")


def cargar_kernel(cual: str | Path, pesos: str = "best") -> KernelEntrenado:
    """El kernel de `cual` ('1k3'/'1k5'/'1k7', o la carpeta de un experimento).

    `pesos`: 'best' (el mejor por `val_loss`, que es lo que quieres de un
    preprocesador) o 'last' (la ultima epoca, que es con lo que se pintaron los
    stops -- por eso `--comprobar` usa 'last').

    ⚠ Se lee el `state_dict` a pelo y NO se construye la red. Es lo que hace que
    esto no dependa de `fv.models.builder` ni de `red_local.py`, o sea lo que
    permite usarlo dentro de un ano con la carpeta del experimento y poco mas.
    Que el tensor sea el mismo que el de la red construida lo fija un test
    (`tests/test_preproceso.py::test_el_kernel_es_el_de_la_red`).
    """
    carpeta = Path(CARPETAS[cual]) if cual in CARPETAS else Path(cual)
    if not carpeta.is_absolute():
        carpeta = EXPERIMENTOS / carpeta
    ck = carpeta / "nn" / "pesos" / f"{pesos}.pt"
    if not ck.exists():
        raise PreprocesoError(
            f"no esta {ck}\n"
            f"  los pesos de un experimento van en su `nn/pesos/` (regla 3 de "
            f"experimentos/README.md).\n"
            f"  si el run existe en foveal-vision-data, copialos ahi y commitealos.")
    e = torch.load(ck, map_location="cpu", weights_only=False)
    estado = e.get("model", e.get("state_dict", e))
    try:
        w = estado["center_convs.0.weight"].detach().clone().float()
        b = estado["center_convs.0.bias"].detach().clone().float()
    except KeyError as exc:                       # pragma: no cover - defensivo
        raise PreprocesoError(
            f"{ck} no tiene `center_convs.0`: no es una de estas redes planas "
            f"({sorted(estado)[:4]}...)") from exc

    stride, dataset = 1, None
    cfg_f = carpeta / "nn" / "pesos" / "config.json"
    if cfg_f.exists():
        cfg = json.loads(cfg_f.read_text())
        red = cfg.get("network", {})
        stride = int(red.get("s_center", 1))
        if int(red.get("n_layers", 1)) != 1:
            raise PreprocesoError(
                f"{cfg_f} dice n_layers={red['n_layers']}: `aplicaKernel` es la "
                f"PRIMERA capa, y con mas de una la salida de la red no es esto.")
        dataset = cfg.get("provenance", {}).get("window_dataset", {}).get("name")
    # Sin config.json el stride se queda en 1 -- el DEFECTO DECLARADO de la R2, y
    # es el valor de las tres configs (`s_center: 1`). Se anota en vez de callarlo.
    return KernelEntrenado(
        nombre=cual if isinstance(cual, str) and cual in CARPETAS else carpeta.name,
        carpeta=carpeta, checkpoint=ck, peso=w, sesgo=b, stride=stride,
        epoca=e.get("epoch", "?"), dataset=dataset)


def _lote(entrada, canales: int, relleno: float, escala: str) -> tuple[torch.Tensor, bool, bool]:
    """(B, C, H, W) float32, y si hay que deshacer el lote / devolver numpy."""
    era_np = isinstance(entrada, np.ndarray)
    if era_np:
        x = torch.from_numpy(np.ascontiguousarray(entrada))
    elif torch.is_tensor(entrada):
        x = entrada.detach()
    else:
        x = torch.as_tensor(np.asarray(entrada))
        era_np = True

    entero = not torch.is_floating_point(x)
    x = x.float()

    if x.ndim == 2:                                # (H, W): una imagen en gris
        x, lote = x[None, None], False
    elif x.ndim == 3:
        # ⚠ (C,H,W) y (B,H,W) son indistinguibles por la forma, y elegir mal
        # cambia la salida sin fallar. Se decide por C y si no cuadra se NIEGA:
        # el fallo ruidoso antes que el silencioso.
        if x.shape[0] not in (1, canales):
            raise PreprocesoError(
                f"entrada 3-D con {x.shape[0]} canales: solo se aceptan 1 (la "
                f"vista) o {canales} (la vista y el relleno).\n"
                f"  si es un LOTE de {x.shape[0]} imagenes en gris, dale forma "
                f"(B, 1, H, W): `x[:, None]`.")
        x, lote = x[None], False
    elif x.ndim == 4:
        if x.shape[1] not in (1, canales):
            raise PreprocesoError(
                f"lote con {x.shape[1]} canales: se esperaba 1 o {canales}")
        lote = True
    else:
        raise PreprocesoError(f"entrada de {x.ndim} dimensiones: se esperaba "
                              f"(H,W), (C,H,W) o (B,C,H,W)")

    if escala == "0-255" or (escala == "auto" and entero):
        x = x / 255.0
    elif escala not in ("auto", "0-1"):
        raise PreprocesoError(f"escala '{escala}': usa 'auto', '0-1' o '0-255'")
    if escala == "auto" and not entero:
        # ⚠ Un float en 0..255 es EL fallo caro de un preprocesador: la salida
        # sale 255x y no falla nada. `build_view` entrega [0,1] y el kernel se
        # entreno sobre eso, asi que fuera de rango se niega en vez de suponer.
        mx = float(x.max()) if x.numel() else 0.0
        if mx > MAX_NORMALIZADA:
            raise PreprocesoError(
                f"la entrada es float y llega a {mx:.3f}: el kernel se entreno "
                f"sobre vistas en [0,1] (`build_view` divide por 255).\n"
                f"  · si son niveles 0..255: escala='0-255'\n"
                f"  · si de verdad esta normalizada y se sale del rango: escala='0-1'")

    if x.shape[1] == 1 and canales == 2:
        # DEFECTO DECLARADO (R2): una imagen suelta no trae el canal de RELLENO,
        # y `relleno=0` significa «todo pixel es real», que es lo que vale en el
        # interior de una pagina. `input_stack` define ese canal como
        # `1 - coverage`: 0 = real, 1 = inventado por el borde.
        pad = torch.full_like(x, float(relleno))
        x = torch.cat([x, pad], dim=1)
    return x.contiguous(), lote, era_np


def aplicaKernel(entrada, kernel: KernelEntrenado | str | Path = "1k5", *,
                 con_sesgo: bool = True, relleno: float = 0.0,
                 escala: str = "auto", pesos: str = "best",
                 activacion: str = ACTIVACION_POR_DEFECTO):
    """Aplica el kernel entrenado a `entrada` SIN RELLENO (`padding=0`) y ACTIVA.

    entrada
        `(H,W)` en gris · `(C,H,W)` · `(B,C,H,W)`. numpy, torch o lista.
        Con C=1 se anade el canal de relleno a `relleno` (ver abajo).
        `uint8` se divide por 255; un float que se salga de [0,1] se NIEGA.
    kernel
        '1k3' / '1k5' / '1k7', la carpeta de un experimento, o un
        `KernelEntrenado` ya cargado (lo suyo si vas a llamar en bucle).
    con_sesgo
        el sesgo de la convolucion. `True` reproduce exactamente lo que veia la
        cabeza de la red; `False` deja solo la respuesta del kernel, que es lo
        que quieres si detras vas a normalizar.
    relleno
        el valor del segundo canal cuando la entrada trae uno solo. 0.0 = «todo
        pixel es real», que es el caso normal fuera del borde de la pagina.
    activacion
        'relu' (POR DEFECTO) o 'ninguna'. Ver abajo: no es un detalle de formato.

    Devuelve `(K, H-k+1, W-k+1)` --o `(B, K, ...)` si entraste con lote--, en
    numpy o torch segun como entraste. K es 1 en los tres experimentos.

    ⚠⚠ LA NO-LINEALIDAD VA AQUI DENTRO, Y ES UNA DECISION DEL DUENO (2026-09-04):
    «el dataset debe ser generado con las funciones que aplican kernel, y esas
    funciones ya deben aplicar la no-linearidad».

    No es cosmetico, y este es el motivo. Un preprocesador SIN activar es una
    operacion LINEAL, asi que una red plana entrenada encima hace `conv(conv(x))`
    sin nada en medio -- y eso es **una sola convolucion** de tamano `k1+k2-1` con
    los pesos atados. El estudio entero seria degenerado: cada brazo un
    subconjunto estricto de un gemelo ya corrido, capaz solo de empatar o perder.
    Con la ReLU dentro, el preproceso es un extractor de rasgos de verdad y la
    pregunta deja de tener respuesta conocida de antemano.

    ⚠ EL PRECIO, dicho: `relu` tira la parte negativa del mapa, que es informacion
    real (la respuesta del kernel viene con signo). Por eso `activacion='ninguna'`
    sigue existiendo y es lo que usa `--comprobar` para demostrar que este kernel
    es LITERALMENTE la capa L1 de su red: esa comprobacion se hace contra los
    `mapas.npy` guardados, que estan SIN activar. Las dos cosas tienen que poder
    convivir -- la identidad se prueba sin activar, y el preproceso se usa activado.

    ⚠ Sin relleno la salida ENCOGE `k-1` px por lado: 20x20 -> 18x18 (k=3),
    16x16 (k=5), 14x14 (k=7). Encadenar dos preprocesos encoge dos veces.
    """
    if not isinstance(kernel, KernelEntrenado):
        kernel = cargar_kernel(kernel, pesos=pesos)
    x, lote, era_np = _lote(entrada, kernel.canales, relleno, escala)
    k = kernel.k
    if x.shape[-2] < k or x.shape[-1] < k:
        raise PreprocesoError(
            f"la entrada es {tuple(x.shape[-2:])} y el kernel {k}x{k}: sin "
            f"relleno no cabe ni una posicion. Hacen falta al menos {k}x{k}.")
    with torch.no_grad():
        y = F.conv2d(x, kernel.peso, kernel.sesgo if con_sesgo else None,
                     stride=kernel.stride, padding=0)
        y = _activar(y, activacion)
    if not lote:
        y = y[0]
    return y.numpy() if era_np else y


def _liga(nombre: str):
    """Una `aplicaKernel` por experimento, con su kernel ya puesto.

    El kernel se carga la PRIMERA vez y se guarda: llamar a esto en un bucle de
    entrenamiento no puede pagar un `torch.load` por imagen.
    """
    cache: dict[str, KernelEntrenado] = {}

    def aplica(entrada, *, con_sesgo: bool = True, relleno: float = 0.0,
               escala: str = "auto", pesos: str = "best",
               activacion: str = ACTIVACION_POR_DEFECTO):
        if pesos not in cache:
            cache[pesos] = cargar_kernel(nombre, pesos=pesos)
        return aplicaKernel(entrada, cache[pesos], con_sesgo=con_sesgo,
                            relleno=relleno, escala=escala, activacion=activacion)

    aplica.__name__ = f"aplicaKernel_{nombre}"
    aplica.__doc__ = (f"`aplicaKernel` con el kernel de `{CARPETAS[nombre]}` "
                      f"ya puesto. Misma firma menos `kernel`.")
    return aplica


aplicaKernel_1k3 = _liga("1k3")
aplicaKernel_1k5 = _liga("1k5")
aplicaKernel_1k7 = _liga("1k7")


# --------------------------------------------------------------- comprobacion
def _comprobar() -> int:
    """Contra el `mapas.npy` de cada `stop-04`, que se calculo con el modelo vivo.

    ⚠ Necesita `fv` y el dataset, porque hay que reconstruir las 10 entradas de
    DOS canales (los PNG guardados son la vista estirada para verse, no el dato).
    Es la comprobacion, no la funcion: `aplicaKernel` sigue sin depender de nada.
    """
    import sys
    sys.path.insert(0, str(EXPERIMENTOS.parent / "src"))
    sys.path.insert(0, str(AQUI))
    import aplicar_kernels as ak          # ⚠ import, NO runpy: `run_path` devuelve
    ok = True                             #   una COPIA del namespace y fijar `RED`
    for nombre in CARPETAS:               #   ahi no llega al modulo (medido)
        kern = cargar_kernel(nombre, pesos="last")     # los stops son de `last`
        ak.RED = (EXPERIMENTOS.parent / "configs" / "networks"
                  / f"plana-20-{nombre}.yaml")
        x, _, _ = ak.entradas(ak.set_visualizacion(10, 2026))
        esperado = np.load(kern.carpeta / "evaluacion" / "stop-04-37epocas" / "mapas.npy")
        # ⚠ SIN activar: los `mapas.npy` se guardaron pre-activacion, que es lo que
        # lee la cabeza de esas redes. La identidad con la L1 se prueba aqui; el
        # preproceso se USA activado (ver el docstring de `aplicaKernel`).
        salida = aplicaKernel(x.numpy(), kern, activacion="ninguna")
        casa = (salida.shape == esperado.shape
                and bool(np.allclose(salida, esperado, atol=1e-6)))
        dif = float(np.abs(salida - esperado).max()) if salida.shape == esperado.shape else float("nan")
        print(f"  {kern}\n      -> {tuple(salida.shape[1:])} desde "
              f"{tuple(x.shape[2:])} · {'✓ casa' if casa else '✗ NO CASA'} con "
              f"stop-04 (dif max {dif:.2e})")
        ok &= casa

        # ...y que el DEFECTO de hoy es la version activada de eso mismo. Sin este
        # segundo paso, cambiar el defecto por accidente no rompe nada visible.
        act = aplicaKernel(x.numpy(), kern)
        bien = bool(np.allclose(act, np.maximum(esperado, 0.0), atol=1e-6))
        negativos = float((esperado < 0).mean()) * 100
        ok &= bien
        print(f"      -> por defecto ACTIVA ({ACTIVACION_POR_DEFECTO}): "
              f"{'✓' if bien else '✗ NO'} es max(0, ese mapa) · "
              f"la ReLU tira el {negativos:.0f} % de las celdas")
    print("los tres reproducen la capa L1 de su red, y por defecto la ACTIVAN."
          if ok else "⚠ alguno NO reproduce lo que su experimento dejo escrito")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true",
                   help="contrasta los tres contra el mapas.npy de su stop-04")
    a = p.parse_args()
    if a.comprobar:
        print("comprobando los tres kernels contra lo que su experimento dejo escrito:")
        return _comprobar()
    for nombre in CARPETAS:
        print(f"  {cargar_kernel(nombre)}")
    print("\n  uso: from preproceso import aplicaKernel_1k5; aplicaKernel_1k5(imagen)")
    print("  comprobacion: python experimentos/comun/preproceso.py --comprobar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
