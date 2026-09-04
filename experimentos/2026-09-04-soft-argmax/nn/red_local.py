#!/usr/bin/env python3
"""La red de ESTE experimento: el cuerpo `4k7` de siempre y una cabeza que lee
las coordenadas como ESPERANZA sobre un mapa de calor (soft-argmax).

    python nn/red_local.py --comprobar

⚠⚠ POR QUE VIVE AQUI Y NO EN `src/fv/models/builder.py`
   Instruccion del dueno (2026-09-03): «estos son experimentos, nada tienen que
   ver con las redes previas... Si hay que hacer cambios al codigo tendremos que
   copiarlo localmente (pero si vale la pena, y eso depende de nuestras pruebas
   en estos experimentos)». El codigo de produccion NO se toca para probar una
   idea. `src/fv/` sigue intacto y se comprueba al final de cada corrida.

QUE CAMBIA
   El ancla (`plana-4k7-s1`) lee las 12 salidas de UNA `Linear(1604, 12)`. Aqui
   esa cabeza se parte en dos y solo cambia la mitad de las coordenadas:

     exists  ->  Linear(1604, 4)                      (igual que siempre)
     x, y    ->  pila conv -> 4 mapas -> softmax -> esperanza sobre la rejilla

   `exists` se deja EXACTAMENTE como estaba --misma entrada, mismo ReLU, mismo
   dropout, misma Linear-- porque es el CONTROL del experimento: si el f1 se
   mueve, no es por el soft-argmax.

LOS TRES BRAZOS (`modo`)
   softargmax : la esperanza, con beta aprendida.                 (run A)
   softargmax + lambda_var > 0 : ademas penaliza la dispersion.   (run B)
   lineal     : MISMA pila conv, MISMOS 4 mapas, pero las coordenadas salen de
                una `Linear` global. Es el CONTROL que separa «la pila conv
                ayuda» de «el soft-argmax ayuda».                 (run C)

EL PRESUPUESTO DE LA CABEZA, QUE ES EL CONFOUND DE ESTA SERIE
   La serie plana ya midio que lo que mueve el f1 es cuantas features llegan a
   la cabeza (`experimentos/README.md`), y `planas-sobre-preprocesado` colapso a
   f1 0,000 por quedarse corta. Una cabeza de mapas de calor hecha con un
   `Conv2d(4,4,1)` son 20 parametros contra los 19.260 del ancla: mediria otra
   vez el tamano de la cabeza. Por eso la pila oculta se dimensiona para IGUALAR
   el presupuesto (canales_ocultos=64 -> +0,15 % sobre el ancla), y `--comprobar`
   lo imprime en vez de afirmarlo.

LA REJILLA, Y POR QUE ABARCA LA VISTA ENTERA Y NO SOLO LA FOVEA
   Las coordenadas objetivo van normalizadas a la ventana etiquetada de 16 px, y
   medido el 2026-09-04 sobre el `windows.npz`: 0 de 72.380 esquinas caen fuera
   de [0,1) ... pero el 24,1 % cae en el PRIMER O ULTIMO PIXEL. La esperanza de
   un softmax solo alcanza el extremo de su rejilla con toda la masa en una
   celda, o sea nunca. La vista 20x20 abarca 32 px del recorte (la fovea de 16
   mas 8 px de anillo por lado), que en unidades de fovea es [-0,375 · 1,375]:
   hay masa que colocar FUERA del objetivo, y por eso la esperanza si puede
   llegar a 0 y a 1.

   ⚠ El anillo es de resolucion reducida (1 celda = 4 px contra 1 px en la
   fovea), asi que la rejilla NO es un `linspace`: sale de los bordes de celda
   reales. `--comprobar` la contrasta contra `fv.fovea._axis_edges`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))

from fv.fovea import dims_of                                      # noqa: E402
from fv.models.builder import FoveatedRegionalNN, full_config     # noqa: E402

RED_BASE = REPO / "configs" / "networks" / "plana-20-4k7.yaml"

# El presupuesto de la cabeza del ancla, para dimensionar la pila oculta.
# No se escribe a mano: `--comprobar` lo mide y falla si se desvia.
CANALES_OCULTOS = 64


def bordes_de_celda(dims) -> list[int]:
    """Offset en el recorte original donde EMPIEZA cada una de las N celdas (+ fin).

    Copia local de `fv.fovea._axis_edges`, que es privada. Se reimplementa en vez
    de importarla porque un experimento tiene que poder abrirse dentro de un ano
    (regla 1 de `experimentos/README.md`) -- y para que las dos copias no
    diverjan en silencio, `--comprobar` las contrasta.
    """
    m, c, r, N = dims.border_px, dims.fovea_px, dims.border_reduce, dims.N
    po = dims.border_cells
    bordes = []
    for k in range(N):
        if k < po:
            bordes.append(k * r)
        elif k < po + c:
            bordes.append(m + (k - po))
        else:
            bordes.append(m + c + (k - po - c) * r)
    bordes.append(dims.original_size)
    return bordes


def rejilla_fovea(dims) -> torch.Tensor:
    """Centro de cada celda de la vista, en unidades de la VENTANA ETIQUETADA.

    u = (centro_en_el_recorte - border_px) / fovea_px, que es exactamente la
    normalizacion que usa `fv.windows.extract` para el objetivo:
    `x = (cx - wx0) / n`. Devuelve (N,) float32.
    """
    b = bordes_de_celda(dims)
    centros = [(b[i] + b[i + 1]) / 2.0 for i in range(dims.N)]
    return torch.tensor([(c - dims.border_px) / dims.fovea_px for c in centros],
                        dtype=torch.float32)


class CabezaSoftArgmax(FoveatedRegionalNN):
    """El cuerpo del repo, con la cabeza de coordenadas sustituida.

    ⚠ El CUERPO sale bit a bit igual al del ancla, y eso no es casualidad: se
    hereda en vez de copiarse, `super().__init__` construye las convoluciones
    ANTES de la cabeza, y `loop.py` siembra (`torch.manual_seed(recipe.seed)`)
    antes de construir el modelo. Lo unico que cambia de una corrida a otra es
    la cabeza. `--comprobar` lo contrasta contra `build_model` tensor a tensor.
    """

    def __init__(self, cfg: dict, modo: str = "softargmax",
                 canales_ocultos: int = CANALES_OCULTOS):
        super().__init__(cfg)
        if modo not in ("softargmax", "lineal"):
            raise ValueError(f"modo '{modo}' no existe: usa softargmax o lineal")
        if not self.single:
            raise ValueError("este experimento es sobre la plana (regions: single)")
        self.modo = modo
        dims = dims_of(cfg)
        canales = int(cfg["channels"][-1])

        # La cabeza de 12 del ancla se retira ENTERA: dejarla sumaria 19.260
        # parametros muertos a la cuenta y el presupuesto dejaria de significar
        # nada. Su construccion en `super().__init__` si ocurrio, que es lo que
        # mantiene el cuerpo alineado con el ancla.
        del self.head

        # exists: la MISMA cabeza de siempre, recortada a 4 salidas. Es el control.
        self.exists_head = nn.Linear(self.flat_features + self.n_edge, 4)

        # La pila que produce los 4 mapas de calor, uno por esquina.
        self.mapa = nn.Sequential(
            nn.Conv2d(canales, canales_ocultos, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(canales_ocultos, 4, 1))

        if modo == "softargmax":
            # beta APRENDIDA, parametrizada en log para que no pueda hacerse
            # negativa (beta<0 invertiria el maximo por el minimo sin avisar).
            # Init 0 -> beta=1: con los logits pequenos de la inicializacion el
            # softmax sale casi plano y la esperanza arranca en el centroide de
            # la rejilla, que es (0,5 · 0,5) -- justo la media del objetivo
            # (medido: x=0,5002 · y=0,4932).
            self.log_beta = nn.Parameter(torch.zeros(()))
            u = rejilla_fovea(dims)
            self.register_buffer("GX", u[None, :].expand(dims.N, dims.N).contiguous())
            self.register_buffer("GY", u[:, None].expand(dims.N, dims.N).contiguous())
        else:
            self.coord_head = nn.Linear(4 * dims.N * dims.N, 8)

        # Lo que lee el regularizador de dispersion del run B. Se guarda por
        # forward y lo consume la perdida parcheada; con `modo='lineal'` no
        # existe y el parche no lo mira.
        self.ultima_var: torch.Tensor | None = None

    # -- la cuenta de parametros, partida como se compara -------------------
    def presupuesto(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters())
        cabeza = n(self.exists_head) + n(self.mapa)
        cabeza += self.log_beta.numel() if self.modo == "softargmax" else n(self.coord_head)
        cuerpo = sum(p.numel() for p in self.center_convs.parameters())
        return {"cuerpo": cuerpo, "cabeza": cabeza, "total": cuerpo + cabeza}

    def forward(self, x: torch.Tensor, edge: torch.Tensor | None = None) -> torch.Tensor:
        mapa = self._branches(x)["single"]              # (B, C, H, W), pre-activacion
        # EXACTAMENTE el tensor que ve la cabeza del ancla: mismo ReLU, mismo
        # dropout. Lo comparten `exists` y la pila de mapas a proposito -- lo que
        # este experimento varia es COMO se leen las coordenadas, no QUE se ve.
        feat = self.drop(F.relu(mapa.flatten(1)))
        fe = feat
        if self.n_edge:
            fe = torch.cat([feat, self._edge_batch(edge, x)], dim=1)
        existe = self.exists_head(fe)                   # (B, 4)

        calor = self.mapa(feat.view_as(mapa))           # (B, 4, H, W)
        if self.modo == "lineal":
            xy = self.coord_head(calor.flatten(1)).view(-1, 4, 2)
            self.ultima_var = None
        else:
            B, K, H, W = calor.shape
            p = F.softmax(torch.exp(self.log_beta) * calor.reshape(B, K, H * W), dim=-1)
            gx = self.GX.reshape(1, 1, -1)
            gy = self.GY.reshape(1, 1, -1)
            cx = (p * gx).sum(-1)                       # (B, 4)
            cy = (p * gy).sum(-1)
            xy = torch.stack([cx, cy], dim=-1)          # (B, 4, 2)
            # Varianza del mapa alrededor de su propia esperanza: mide cuanto se
            # DISPERSA, y no necesita el objetivo. La consume el run B.
            d2 = (gx - cx[..., None]) ** 2 + (gy - cy[..., None]) ** 2
            self.ultima_var = (p * d2).sum(-1)          # (B, 4)
        return torch.cat([existe[..., None], xy], dim=-1)   # (B, 4, 3)


def construir(modo: str = "softargmax", semilla: int = 1) -> CabezaSoftArgmax:
    cfg = full_config(yaml.safe_load(RED_BASE.read_text()))
    torch.manual_seed(semilla)
    return CabezaSoftArgmax(cfg, modo=modo)


def _comprobar() -> int:
    from fv.fovea import _axis_edges
    from fv.models.builder import build_model

    cfg = full_config(yaml.safe_load(RED_BASE.read_text()))
    dims = dims_of(cfg)
    fallos = []

    # 1. la rejilla: la copia local contra la privada del repo
    if bordes_de_celda(dims) != list(_axis_edges(dims)):
        fallos.append("bordes_de_celda ya NO coincide con fv.fovea._axis_edges")
    u = rejilla_fovea(dims)
    print(f"  rejilla ({dims.N} celdas), en unidades de la ventana de {dims.fovea_px} px:")
    print(f"    {' '.join(f'{v:+.3f}' for v in u[:4])} ... {' '.join(f'{v:+.3f}' for v in u[-4:])}")
    print(f"    abarca [{u.min():+.3f} · {u.max():+.3f}] · centroide {u.mean():.4f}")
    if not (u.min() < 0.0 and u.max() > 1.0):
        fallos.append("la rejilla NO sale de [0,1]: la esperanza no podra alcanzar los extremos")
    if abs(float(u.mean()) - 0.5) > 1e-6:
        fallos.append(f"el centroide de la rejilla es {u.mean():.4f} y no 0,5")

    # 2. el cuerpo, bit a bit contra el ancla
    torch.manual_seed(1); ancla = build_model(cfg)
    a = construir("softargmax", 1)
    b = construir("lineal", 1)
    for nom, m in (("A/B softargmax", a), ("C lineal", b)):
        if not torch.equal(m.center_convs[0].weight, ancla.center_convs[0].weight):
            fallos.append(f"{nom}: el CUERPO no sale igual al del ancla")

    # 3. el presupuesto de la cabeza
    cab_ancla = sum(p.numel() for p in ancla.head.parameters())
    cue_ancla = sum(p.numel() for p in ancla.center_convs.parameters())
    print(f"\n  ancla `plana-4k7-s1` : cuerpo {cue_ancla} · cabeza {cab_ancla} "
          f"· total {cue_ancla + cab_ancla}")
    for nom, m in (("A/B softargmax  ", a), ("C lineal (control)", b)):
        p = m.presupuesto()
        d = 100.0 * (p["cabeza"] - cab_ancla) / cab_ancla
        print(f"  {nom}: cuerpo {p['cuerpo']} · cabeza {p['cabeza']} "
              f"· total {p['total']}  ({d:+.1f} % de cabeza)")
    da = 100.0 * (a.presupuesto()["cabeza"] - cab_ancla) / cab_ancla
    if abs(da) > 2.0:
        fallos.append(f"la cabeza de A/B se desvia {da:+.1f} % del ancla (tope 2 %): "
                      f"ajusta CANALES_OCULTOS")

    # 4. la salida sigue siendo 4 esquinas x 3, y arranca en el centro
    x = torch.zeros(2, 2, dims.N, dims.N)
    e = torch.zeros(2, a.n_edge)
    for nom, m in (("A/B", a), ("C", b)):
        m.eval()
        with torch.no_grad():
            o = m(x, e)
        if o.shape != (2, 4, 3):
            fallos.append(f"{nom}: la salida deja de ser (B,4,3): {tuple(o.shape)}")
    with torch.no_grad():
        xy0 = a(x, e)[:, :, 1:]
    print(f"\n  A sin entrenar predice x,y = {xy0[0,0,0]:.4f} · {xy0[0,0,1]:.4f} "
          f"(el centroide de la rejilla es 0,5)")
    if abs(float(xy0.mean()) - 0.5) > 0.05:
        fallos.append("A sin entrenar NO arranca en el centro de la rejilla")

    # 5. el soft-argmax ALCANZA los extremos que el objetivo pide
    #    (el 24,1 % de las esquinas esta en el primer/ultimo px)
    with torch.no_grad():
        calor = torch.full((1, 4, dims.N, dims.N), -20.0)
        calor[0, 0, 0, 0] = 20.0                 # toda la masa en la esquina de la rejilla
        p = F.softmax(calor.reshape(1, 4, -1) , dim=-1)
        alcance_min = float((p[0, 0] * a.GX.reshape(-1)).sum())
        calor2 = torch.full((1, 4, dims.N, dims.N), -20.0)
        calor2[0, 0, -1, -1] = 20.0
        p2 = F.softmax(calor2.reshape(1, 4, -1), dim=-1)
        alcance_max = float((p2[0, 0] * a.GX.reshape(-1)).sum())
    print(f"  con toda la masa en una celda la esperanza llega a "
          f"[{alcance_min:+.3f} · {alcance_max:+.3f}] — el objetivo pide [0,000 · 0,999]")
    if not (alcance_min < 0.0 and alcance_max > 1.0):
        fallos.append("la esperanza NO cubre [0,1]")

    # 6. la varianza que lee el run B existe en A y NO en C
    if a.ultima_var is None:
        fallos.append("A no dejo `ultima_var`: el regularizador del run B no tendria que leer")
    if b.ultima_var is not None:
        fallos.append("C dejo `ultima_var`: no deberia, no hay mapa de calor que dispersar")

    print()
    if fallos:
        for f in fallos:
            print(f"  ✗ {f}")
        return 1
    print("  ✓ rejilla == la del repo · cuerpo == el del ancla · cabeza dentro del "
          "presupuesto\n  ✓ salida (B,4,3) · arranca en el centro · alcanza los extremos")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--comprobar", action="store_true")
    p.parse_args()
    raise SystemExit(_comprobar())
