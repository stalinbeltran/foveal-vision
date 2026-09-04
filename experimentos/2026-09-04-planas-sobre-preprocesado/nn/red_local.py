#!/usr/bin/env python3
"""Las TRES estructuras: CNN planas con los parametros OPTIMOS de la foveada.

    python nn/red_local.py                # la tabla de las tres
    python nn/red_local.py --comprobar    # que son identicas salvo por la entrada

⚠ SOLO LA ESTRUCTURA (2026-09-04). Encargo del dueno: «crea solo las estructuras de
   las nn q vamos a entrenar». No hay entrenamiento, ni receta cableada, ni pesos.

DE DONDE SALEN LOS PARAMETROS, Y POR QUE DE AHI
   De `estudios-redes-neuronales/ESTADO.md`, seccion «Red foveada (ws16-p2-d2-L4)».
   ⚠ El encargo decia `reportes/README.md`: ese fichero es el HISTORIAL cronologico y
   el propio README avisa de que «el estado no vive aqui: en que quedo cada parametro
   esta en ../ESTADO.md». Se toma de ESTADO.md, que es la fuente que el otro senala.

   ⚠⚠ Y ESA TABLA TIENE DOS COLUMNAS QUE NO SIEMPRE COINCIDEN: «vigente» y «optimo
   medido». El encargo pide los OPTIMOS, asi que donde difieren se toma el optimo y se
   dice cual era el vigente:

       n_layers      4      cerrado (2 -> 0,9066 · 3 -> 0,9246 · 4 -> 0,9341 · 5 -> 0,9136)
       k_center      3      cerrado (5 y 7 son peores Y mas caros)
       channels    [16]x4   cerrado 20/20 (24 y 32 no aportan; 8 hace dano)
       s_center      1      NO BARRIBLE: un solo valor legal con esta geometria
       dropout     0,0      tanteo; 0,1 es el PEOR de los cuatro
       merge/pool_mode      concat / avg -- medidos, ninguno mueve el vigente

   Los que NO se heredan, y por que -- son justo «los parametros afectados por los
   datasets de entrada» que menciona el encargo:

       fovea_px, border_px, border_reduce, overlap_fovea_px, overlap_border_px
           Describen como se construye la VISTA foveada a partir de la pagina. Aqui la
           entrada YA es un mapa preprocesado de tamano fijo (18/16/14), asi que esos
           mandos ya se aplicaron al construir el dataset y no vuelven a aplicarse.
           ⚠ El dataset se construyo con la geometria de `plana-20-1k3.yaml`, que YA
           lleva los optimos medidos `border_px: 8` y `overlap_fovea_px: 7`.
       regions
           `single` por definicion: una plana es una rama sobre todo el input, sin
           mascara y sin periferia. Es lo que significa «plana» en esta serie
           (`plana-24-single.yaml`), no «una sola capa».
       k_periph, s_periph, mask_channel
           No aplican: no hay rama periferica, y el canal de relleno ya lo consumio el
           kernel congelado al construir el dataset.

⚠⚠ EL RELLENO DE LA CONVOLUCION ES `k//2`, COMO EN LA FOVEADA -- Y NO ES LO MISMO QUE
   HACIAN LOS SIETE GEMELOS
   `builder.py:145` calcula `pad = k_center // 2` y no es un dato: es una expresion. O
   sea que «los parametros optimos de la foveada» traen relleno `same`, y con el la
   resolucion NO cae por las capas: 18x18 sigue siendo 18x18 tras las cuatro.

   Los siete gemelos de la serie plana usan `padding=0` (por eso se llaman
   `-sinpadding`), pero eso era el EJE de aquellos experimentos, no un optimo medido:
   ESTADO.md no tiene ninguna fila que diga que 0 gane. Aqui se hereda lo de la
   foveada, que es lo que pide el encargo, y se deja anotado que la otra opcion existe
   y que cambia el ancho de la cabeza 3,2x (5.184 contra 1.600 en el brazo 1k3).

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
# Los optimos de la foveada, en un solo sitio y con su procedencia al lado.
# (R4: se DECLARA, no se deduce.)
OPTIMOS_FOVEADA = {
    "n_layers": 4,        # ESTADO.md: cerrado. 4 -> 0,9341 contra 3 -> 0,9246 y 5 -> 0,9136
    "k": 3,               # ESTADO.md `k_center`: cerrado. 5 y 7 son peores y mas caros
    "canales": 16,        # ESTADO.md `channels`: cerrado 20/20. 16 es el suelo util
    "stride": 1,          # ESTADO.md `s_center`: no barrible, un solo valor legal
    "dropout": 0.0,       # ESTADO.md: tanteo, y 0,1 es el PEOR de los cuatro
}
# La forma de la entrada de cada brazo. Sale del kernel con que se preproceso:
# 20 - k + 1. Se declara aqui y `--comprobar` lo contrasta contra el .npz real.
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
                 n_layers: int = OPTIMOS_FOVEADA["n_layers"],
                 k: int = OPTIMOS_FOVEADA["k"],
                 canales: int = OPTIMOS_FOVEADA["canales"],
                 stride: int = OPTIMOS_FOVEADA["stride"],
                 dropout: float = OPTIMOS_FOVEADA["dropout"],
                 pad: int | None = None, n_edge: int = 0):
        super().__init__()
        # `k//2` = el relleno de la foveada (`builder.py:145`), no `0`. Ver cabecera.
        self.pad = k // 2 if pad is None else pad
        self.alto, self.ancho, self.canales_in = alto, ancho, canales_in
        self.k, self.stride, self.n_edge = k, stride, n_edge

        capas, c_in = [], canales_in
        for i in range(n_layers):
            capas.append(nn.Conv2d(c_in, canales, k,
                                   stride=stride if i == 0 else 1, padding=self.pad))
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
    print("Tres CNN planas · parametros OPTIMOS de la foveada (ESTADO.md)")
    print(f"  n_layers {OPTIMOS_FOVEADA['n_layers']} · k {OPTIMOS_FOVEADA['k']} · "
          f"channels [{OPTIMOS_FOVEADA['canales']}]x{OPTIMOS_FOVEADA['n_layers']} · "
          f"stride {OPTIMOS_FOVEADA['stride']} · dropout {OPTIMOS_FOVEADA['dropout']} · "
          f"regions single · padding k//2\n")
    print(f"{'brazo':6} {'entrada':>12} {'tras las 4 convs':>18} {'features':>9} "
          f"{'convs':>7} {'cabeza':>8} {'TOTAL':>8}")
    for b in ENTRADAS:
        m = construir(b)
        f = m.forma()
        convs = sum(p for _n, _s, p in f if _n.startswith("conv"))
        cab = f[-1][2]
        print(f"{b:6} {str(f[0][1]):>12} {str(f[-3][1]):>18} {m.flat_features:>9} "
              f"{convs:>7} {cab:>8} {convs + cab:>8}")
    print("\ndetalle de un brazo (1k3), capa a capa:")
    for n, s, p in construir("1k3").forma():
        print(f"  {n:9} {str(s):>16} {p:>8} params")
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
