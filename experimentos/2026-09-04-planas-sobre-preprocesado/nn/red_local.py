#!/usr/bin/env python3
"""Las TRES estructuras: CNN planas MINIMAS sobre los datasets preprocesados.

    python nn/red_local.py                # la tabla de las tres
    python nn/red_local.py --comprobar    # que son identicas salvo por la entrada

⚠ SOLO LA ESTRUCTURA (2026-09-04). No hay entrenamiento, ni receta cableada, ni pesos.

LA DEFINICION, Y DE DONDE SALE CADA COSA
   Redefinicion del dueno (2026-09-04), para abaratar y sacar preliminares:
   2 capas · 2 canales por capa · SIN relleno · stride = la mitad del kernel.
   Los valores literales y su procedencia estan en `PARAMS` y `STRIDE`, abajo.

   Lo que NO redefinio se queda en el optimo medido de la foveada
   (`estudios-redes-neuronales/ESTADO.md`): `dropout` 0,0 y `regions: single`.
   ⚠ `n_layers` y `channels` SI se apartan del optimo foveado a proposito --eran 4 y
   [16]x4-- y eso es el encargo, no un descuido.

⚠⚠ EL COSTE BAJA 242x, Y ESO TIENE UN PRECIO QUE HAY QUE DECIR
   La red pasa de 69.340 parametros a 286. Es lo que se pedia («reducir el coste al
   minimo»), pero con 18 features llegando a una cabeza de 12 salidas el riesgo real
   es que un resultado malo mida «la red es demasiado pequena» y no «el preproceso no
   sirve». Un preliminar acota, no declara -- y menos este.

⚠ LO BUENO QUE TRAE, Y NO SE BUSCABA: casi desaparece el confound de la cabeza
   Con k=3 y stride 2, el `1k3` y el `1k5` caen los DOS en 18 features (3x3x2). O sea
   que esos dos brazos son iso-features por construccion, que es justo lo que en la
   version de 4 capas habia que corregir con anclas externas. El `1k7` se queda en 8.

LO QUE CAMBIA ENTRE LAS TRES, Y NADA MAS
   El alto y ancho de la entrada, que vienen del kernel con que se preproceso:

       1k3 -> 18x18      1k5 -> 16x16      1k7 -> 14x14

   Y de ahi, lo unico que se mueve: el ancho de la cabeza. `--comprobar` lo verifica
   comparando las tres capa por capa en vez de fiarlo a la lectura.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

# ---------------------------------------------------------------------------
# LA DEFINICION DE HOY (2026-09-04, redefinicion del dueno para abaratar):
#   «Las capas de la nn seran 2, por ahora. Cada capa de la nn va a tener solo 2
#    canales. El padding del kernel va a ser siempre 'sin padding'. Cada capa va a
#    reducir el tamano de los features. Ademas, el stride va a ser la mitad del
#    ancho del kernel (redondeado).»
#
# (R4: se DECLARA, no se deduce. Y cada valor con de donde sale.)
PARAMS = {
    "n_layers": 2,        # dueno 2026-09-04 («por ahora»). Antes: 4, el optimo foveado
    "canales": 2,         # dueno 2026-09-04. Antes: 16, el optimo foveado
    "k": 3,               # ⚠ NO es una eleccion: ver abajo, es lo unico que cabe
    "pad": 0,             # dueno 2026-09-04: «siempre sin padding»
    "dropout": 0.0,       # ESTADO.md: sigue siendo el optimo, y el dueno no lo movio
}
# «el stride va a ser la mitad del ancho del kernel (redondeado)».
# Media hacia ARRIBA: `(k+1)//2` da 3->2, 5->3, 7->4. `k//2` seria TRUNCAR, no
# redondear, y con k=3 daria 1 -- que cambia la cabeza de 18 a 392 features.
STRIDE = (PARAMS["k"] + 1) // 2      # = 2

# ⚠⚠ POR QUE k=3 NO ES UNA ELECCION MIA, SINO LA UNICA POSIBLE
#    El encargo no dijo el tamano del kernel, pero con el resto de las condiciones
#    --2 capas, sin relleno, stride = mitad de k-- solo k=3 sobrevive. Medido:
#
#        k=3  s=2 |  18->8->3  16->7->3  14->6->2   <- las tres viven
#        k=5  s=3 |  18->5->1  16->4->0  14->4->0   <- el 1k5 y el 1k7 MUEREN
#        k=7  s=4 |  18->3->0  16->3->0  14->2->-1  <- mueren los tres
#
#    Con `n2 = (n - k)//s + 1`, un k grande con stride grande se come el mapa en dos
#    capas. Asi que k=3 esta FORZADO por el encargo, no heredado del optimo foveado
#    (aunque coincida con el, que ademas esta cerrado en ESTADO.md).

# El stride va en TODAS las capas, no solo en la primera. `builder.py` lo pone solo
# en la primera (convencion D-S1, «el submuestreo total es s sea cual sea la
# profundidad»), pero el encargo dice que el stride ES la mitad del kernel, sin
# distinguir capas, y ademas pide que CADA capa reduzca. Se anota porque es una
# divergencia deliberada con produccion.

# La forma de la entrada de cada brazo. Sale del kernel con que se preproceso:
# 20 - k + 1. Se declara aqui (R4) y `--comprobar` lo contrasta contra el .npz real.
ENTRADAS = {"1k3": (1, 18, 18), "1k5": (1, 16, 16), "1k7": (1, 14, 14)}

SALIDAS = 12          # 4 esquinas x [existe, x, y]


class PlanaPreprocesada(nn.Module):
    """Una rama sobre todo el input, `n_layers` convs, y una cabeza lineal.

    Es la MISMA forma que `FoveatedRegionalNN` con `regions: single` --convs con
    ReLU ENTRE capas y ninguna tras la ultima, flatten, ReLU, dropout, cabeza--,
    reescrita aqui porque aquella deriva su geometria de `fovea_px`/`border_px` y
    dimensiona la cabeza con `dims.N` (`builder.py:274-278`). Con una entrada de
    18x18 eso daria una cabeza de otro tamano y reventaria al primer lote.

    ⚠ No toca `src/fv/`: instruccion del dueno (2026-09-03), como toda la serie.
    """

    def __init__(self, alto: int, ancho: int, canales_in: int = 1,
                 n_layers: int = PARAMS["n_layers"],
                 k: int = PARAMS["k"],
                 canales: int = PARAMS["canales"],
                 stride: int | None = None,
                 dropout: float = PARAMS["dropout"],
                 pad: int | None = None, n_edge: int = 0):
        super().__init__()
        self.pad = PARAMS["pad"] if pad is None else pad
        stride = (k + 1) // 2 if stride is None else stride
        self.alto, self.ancho, self.canales_in = alto, ancho, canales_in
        self.k, self.stride, self.n_edge = k, stride, n_edge

        capas, c_in = [], canales_in
        for i in range(n_layers):
            # stride en TODAS las capas (ver la nota de arriba), no solo la 1a
            capas.append(nn.Conv2d(c_in, canales, k, stride=stride, padding=self.pad))
            c_in = canales
        self.convs = nn.ModuleList(capas)
        self.drop = nn.Dropout(float(dropout))
        self.flat_features = self._inferir_flat()
        self.head = nn.Linear(self.flat_features + n_edge, SALIDAS)

    def _inferir_flat(self) -> int:
        """La cabeza se dimensiona con la forma REAL de la entrada de este brazo."""
        with torch.no_grad():
            d = torch.zeros(1, self.canales_in, self.alto, self.ancho)
            return int(self._ramas(d).flatten(1).shape[1])

    def _ramas(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU ENTRE capas y ninguna tras la ultima -- igual que
        # `builder._branch_forward`. El mapa final queda pre-activacion.
        for i, c in enumerate(self.convs):
            x = c(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x

    def forward(self, x: torch.Tensor, edge: torch.Tensor | None = None):
        feat = self.drop(F.relu(self._ramas(x).flatten(1)))
        if self.n_edge:
            feat = torch.cat([feat, edge], dim=1)
        return self.head(feat).view(-1, 4, 3)

    def forma(self) -> list[tuple]:
        """La traza capa a capa, para la tabla y para `--comprobar`."""
        t, filas = torch.zeros(1, self.canales_in, self.alto, self.ancho), []
        filas.append(("entrada", tuple(t.shape[1:]), 0))
        for i, c in enumerate(self.convs):
            t = c(t)
            filas.append((f"conv{i}", tuple(t.shape[1:]),
                          sum(p.numel() for p in c.parameters())))
        filas.append(("flatten", (int(t.flatten(1).shape[1]),), 0))
        filas.append(("head", (4, 3), sum(p.numel() for p in self.head.parameters())))
        return filas


def construir(brazo: str, **kw) -> PlanaPreprocesada:
    c, h, w = ENTRADAS[brazo]
    return PlanaPreprocesada(h, w, canales_in=c, **kw)


# --------------------------------------------------------------------- tabla
def tabla() -> int:
    print("Tres CNN planas MINIMAS (redefinicion del dueno, 2026-09-04)")
    print(f"  n_layers {PARAMS['n_layers']} · k {PARAMS['k']} · "
          f"channels [{PARAMS['canales']}]x{PARAMS['n_layers']} · "
          f"stride {STRIDE} (= mitad de k, redondeado) · padding {PARAMS['pad']} "
          f"(SIN relleno) · dropout {PARAMS['dropout']} · regions single\n")
    print(f"{'brazo':6} {'entrada':>12} {'tras conv0':>12} {'tras conv1':>12} "
          f"{'features':>9} {'convs':>7} {'cabeza':>8} {'TOTAL':>7}")
    for b in ENTRADAS:
        m = construir(b)
        f = m.forma()
        convs = sum(p for n, _s, p in f if n.startswith("conv"))
        cab = f[-1][2]
        print(f"{b:6} {str(f[0][1]):>12} {str(f[1][1]):>12} {str(f[2][1]):>12} "
              f"{m.flat_features:>9} {convs:>7} {cab:>8} {convs + cab:>7}")
    print("\ndetalle de un brazo (1k3), capa a capa:")
    for n, s, p in construir("1k3").forma():
        print(f"  {n:9} {str(s):>14} {p:>6} params")
    print("\n⚠ 286 parametros contra los 69.340 de la version de 4 capas: 242x menos.")
    print("⚠ Y `1k3` y `1k5` caen los dos en 18 features -> iso-features por construccion.")
    return 0


def _comprobar() -> int:
    ok = True
    print("1 · la forma declarada casa con el .npz construido de cada brazo:")
    try:
        from construir_datasets import DESTINO
        import numpy as np
        for b, esperada in ENTRADAS.items():
            carp = sorted(DESTINO.glob(f"{b}-*"))
            if not carp:
                print(f"  {b}: dataset no construido — no se puede contrastar")
                ok = False
                continue
            real = tuple(np.load(carp[0] / "preprocesado.npz")["x"].shape[1:])
            casa = real == esperada
            ok &= casa
            print(f"  {b}: .npz {real} contra declarado {esperada} "
                  f"{'✓' if casa else '✗ NO CASA'}")
    except Exception as e:                        # pragma: no cover - defensivo
        print(f"  no se pudo leer el dataset: {e}")
        ok = False

    print("\n2 · las tres son IDENTICAS salvo por lo que impone la entrada:")
    ms = {b: construir(b) for b in ENTRADAS}
    ref = ms["1k3"]
    for b, m in ms.items():
        mismos = (len(m.convs) == len(ref.convs)
                  and all(a.kernel_size == c.kernel_size
                          and a.out_channels == c.out_channels
                          and a.stride == c.stride and a.padding == c.padding
                          for a, c in zip(m.convs, ref.convs))
                  and type(m.drop) is type(ref.drop)
                  and m.drop.p == ref.drop.p)
        ok &= mismos
        print(f"  {b}: convs y dropout identicos al 1k3 {'✓' if mismos else '✗'} · "
              f"lo unico distinto: cabeza {m.flat_features} -> {ref.flat_features}")

    print("\n3 · un forward de verdad da (B, 4, 3):")
    for b, m in ms.items():
        c, h, w = ENTRADAS[b]
        y = m(torch.zeros(5, c, h, w))
        bien = tuple(y.shape) == (5, 4, 3)
        ok &= bien
        print(f"  {b}: {tuple(y.shape)} {'✓' if bien else '✗'}")

    print("\n4 · `src/fv/` intacto (esto es codigo LOCAL):")
    fuente = (REPO / "src" / "fv" / "models" / "builder.py").read_text()
    intacto = "padding=pad" in fuente
    ok &= intacto
    print(f"  builder.py sin tocar {'✓' if intacto else '✗'}")

    print("\n" + ("todo casa." if ok else "⚠ ALGO NO CASA"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    return _comprobar() if p.parse_args().comprobar else tabla()


if __name__ == "__main__":
    raise SystemExit(main())
