#!/usr/bin/env python3
"""La figura que SOLO este experimento puede producir: los mapas de calor.

    python nn/mapas_de_calor.py [--brazo A]

Una fila por ventana; en cada fila, la vista y los cuatro mapas de probabilidad
--TL, TR, BR, BL-- que el softmax pone sobre la rejilla, con la esquina
VERDADERA (circulo) y la PREDICHA (cruz) encima.

⚠ POR QUE ESTA FIGURA Y NO LA DE `comun/aplicar_kernels.py`
   Aquella pinta los KERNELS de L1, que es lo que varia en la serie plana. Aqui
   el cuerpo es el del ancla y lo que varia es la cabeza, asi que mirar los
   kernels no diria nada de lo que este experimento cambia. Lo que hay que ver es
   DONDE pone la masa el softmax: es la unica representacion en la que se ve el
   sesgo de contraccion --si existe-- sin tener que deducirlo de un promedio.

⚠ LAS 10 VENTANAS SI SON LAS COMPARTIDAS. Salen de `comun/set-visualizacion.json`
   y las entradas las construye `comun/aplicar_kernels.py:entradas`, no una copia
   de aqui: si esta figura se pusiera al lado de la de otro experimento, tendrian
   que ser las mismas ventanas o la comparacion seria una ilusion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
COMUN = REPO / "experimentos" / "comun"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

from fv.fovea import dims_of                                       # noqa: E402
from fv.models.builder import full_config                          # noqa: E402
from fv.training.registry import RunStore                          # noqa: E402
from entrenar_local import BRAZOS, RED                             # noqa: E402
from red_local import CabezaSoftArgmax                             # noqa: E402

RED_YAML = REPO / "configs" / "networks" / f"{RED}.yaml"
ESQUINAS = ("TL", "TR", "BR", "BL")
CEL = 96


def _comun():
    """El evaluador compartido, cargado como modulo para reusar `entradas`."""
    spec = importlib.util.spec_from_file_location("ak", COMUN / "aplicar_kernels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.RED = str(RED_YAML)
    return m


def _gris(a: np.ndarray) -> Image.Image:
    t = np.clip((a - a.min()) / max(a.max() - a.min(), 1e-9), 0, 1)
    v = (t * 255).astype(np.uint8)
    return Image.fromarray(np.stack([v, v, v], -1), "RGB")


def _calor(p: np.ndarray) -> Image.Image:
    """Blanco (0) -> rojo (el maximo DE ESE MAPA). La escala es por mapa porque
    lo que se mira es la FORMA de la distribucion, no cuanto vale su pico: un
    mapa plano y uno concentrado con el mismo maximo tienen que verse distintos.
    """
    t = np.clip(p / max(p.max(), 1e-12), 0.0, 1.0)
    rgb = np.stack([255 - t * 40, 255 - t * 215, 255 - t * 235], -1)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")


def _marca(d: ImageDraw.ImageDraw, ox: int, oy: int, u: float, v: float,
           dims, forma: str, color) -> None:
    """Pinta una coordenada en unidades de FOVEA sobre una celda de CEL px.

    La rejilla dibujada es la VISTA (N celdas), asi que hay que llevar u de
    unidades de fovea a celdas: la fovea ocupa las celdas [border_cells,
    border_cells + fovea_px) y cada una vale 1/fovea_px.
    """
    cx = ox + (dims.border_cells + u * dims.fovea_px) * CEL / dims.N
    cy = oy + (dims.border_cells + v * dims.fovea_px) * CEL / dims.N
    r = 4.5
    if forma == "circulo":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
    else:
        d.line([cx - r, cy - r, cx + r, cy + r], fill=color, width=2)
        d.line([cx - r, cy + r, cx + r, cy - r], fill=color, width=2)


def figura(brazo: str, dest: Path) -> Path:
    ak = _comun()
    cfg = full_config(yaml.safe_load(RED_YAML.read_text()))
    dims = dims_of(cfg)
    vent = json.loads((COMUN / "set-visualizacion.json").read_text())["ventanas"]
    x, e, vistas = ak.entradas(vent)

    d = BRAZOS[brazo]
    if d["modo"] != "softargmax":
        raise SystemExit(f"el brazo {brazo} no tiene mapa de calor (modo {d['modo']})")
    m = CabezaSoftArgmax(cfg, modo=d["modo"])
    st = torch.load(RunStore().path(d["run"]) / "best.pt", map_location="cpu",
                    weights_only=False)
    m.load_state_dict(st["model"] if "model" in st else st["state_dict"])
    m.eval()

    with torch.no_grad():
        salida = m(x, e)                                   # (10, 4, 3)
        # el mismo tensor que ve la cabeza, calculado UNA vez
        mapa = m._branches(x)["single"]
        calor = m.mapa(m.drop(F.relu(mapa.flatten(1))).view_as(mapa))   # (10, 4, N, N)
        B, K, H, W = calor.shape
        p = F.softmax(torch.exp(m.log_beta) * calor.reshape(B, K, H * W), dim=-1)
        p = p.reshape(B, K, H, W).numpy()

    # La esquina VERDADERA sale del mismo `windows.npz`, por el indice que el set
    # congelado guarda. Sin esto la figura ensena donde CREE la red que esta la
    # esquina y no contra que, que es justo lo que hay que poder juzgar de un ojo.
    from fv import settings
    z = np.load(settings.window_datasets_root() / ak.DATASET / "windows.npz")
    verdad = np.stack([z["y"][v["indice"]] for v in vent])          # (10, 4, 3)

    gap, cab, pie = 8, 30, 46
    ancho = 5 * CEL + 6 * gap
    alto = cab + len(vent) * (CEL + gap) + pie
    lienzo = Image.new("RGB", (ancho, alto), "white")
    dib = ImageDraw.Draw(lienzo)
    fuente = ak._font(13)
    for j, t in enumerate(("vista",) + ESQUINAS):
        dib.text((gap + j * (CEL + gap), 8), t, fill="black", font=fuente)

    for i in range(len(vent)):
        oy = cab + i * (CEL + gap)
        lienzo.paste(_gris(vistas[i]).resize((CEL, CEL), Image.NEAREST), (gap, oy))
        for c in range(4):
            ox = gap + (c + 1) * (CEL + gap)
            lienzo.paste(_calor(p[i, c]).resize((CEL, CEL), Image.NEAREST), (ox, oy))
            dib.rectangle([ox, oy, ox + CEL - 1, oy + CEL - 1], outline=(210, 210, 210))
            # la fovea, para no confundir la rejilla con la ventana etiquetada
            f0 = ox + dims.border_cells * CEL / dims.N
            f1 = ox + (dims.border_cells + dims.fovea_px) * CEL / dims.N
            dib.rectangle([f0, oy + (f0 - ox), f1 - 1, oy + (f1 - ox) - 1],
                          outline=(120, 160, 255))
            px, py = float(salida[i, c, 1]), float(salida[i, c, 2])
            _marca(dib, ox, oy, px, py, dims, "cruz", (0, 90, 200))
            if verdad is not None and verdad[i, c, 0] >= 0.5:
                _marca(dib, ox, oy, float(verdad[i, c, 1]), float(verdad[i, c, 2]),
                       dims, "circulo", (0, 150, 0))

    beta = float(torch.exp(m.log_beta))
    dib.text((gap, alto - pie + 4),
             f"brazo {brazo} ({d['run']}) · beta aprendida = {beta:.3f} · "
             f"escala por mapa (blanco 0 -> rojo el maximo DE ESE mapa)",
             fill="black", font=fuente)
    dib.text((gap, alto - pie + 22),
             "azul: el recuadro es la ventana ETIQUETADA (16 px); la cruz, lo predicho. "
             "verde: la esquina verdadera. El resto de la rejilla es periferia (1 celda = 4 px).",
             fill="black", font=fuente)

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"mapas-de-calor-{brazo}.png"
    lienzo.save(out)
    print(f"  {out.relative_to(REPO)}  ({len(vent)} ventanas · beta {beta:.3f})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brazo", default=None, help="A o B; por defecto los dos")
    a = ap.parse_args()
    brazos = [a.brazo] if a.brazo else ["A", "B"]
    for b in brazos:
        figura(b, EXP / "resultados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
