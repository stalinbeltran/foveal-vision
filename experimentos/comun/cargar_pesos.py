#!/usr/bin/env python3
"""Carga los pesos guardados de cualquiera de las CNN planas, y los COMPRUEBA.

    python experimentos/comun/cargar_pesos.py                 # comprueba los cuatro
    python experimentos/comun/cargar_pesos.py --exp <carpeta> # uno solo

⚠ POR QUE EXISTE ESTA COMPROBACION Y NO SOLO EL FICHERO
   Un `.pt` guardado que no carga es peor que no tenerlo: ocupa sitio, parece un
   respaldo y no lo es. Aqui se carga cada uno de verdad y se contrasta la norma
   L2 de sus kernels contra la que quedo escrita en el `resumen.json` del ultimo
   stop -- que se calculo en su momento, con el modelo vivo. Si no cuadran, el
   fichero no es el que dice ser.

⚠ Y EL EXPERIMENTO «SIN PADDING» NECESITA SU PROPIA CLASE
   Sus pesos NO se pueden cargar con `build_model` del repo: la cabeza es de 392
   features en vez de 800, porque la convolucion sin relleno encoge el mapa de
   20x20 a 14x14. Se carga con `nn/red_local.py` de ese experimento, y por eso
   ese fichero es parte del respaldo tanto como el `.pt`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parents[1]
sys.path.insert(0, str(REPO / "src"))

from fv.models.builder import build_model, full_config      # noqa: E402

# (carpeta del experimento, config de red, si necesita su clase local)
EXPERIMENTOS = [
    ("2026-09-03-cnn-plana-4k7", "plana-20-4k7", False),
    ("2026-09-03-cnn-plana-2k7", "plana-20-2k7", False),
    ("2026-09-03-cnn-plana-4k7-replicate", "plana-20-4k7-rep", False),
    ("2026-09-03-cnn-plana-2k7-sinpadding", "plana-20-2k7", True),
]


def cargar(exp: str, red: str, local: bool = False, cual: str = "best"):
    """Devuelve el modelo con los pesos guardados EN EL EXPERIMENTO."""
    carpeta = REPO / "experimentos" / exp
    ck = carpeta / "nn" / "pesos" / f"{cual}.pt"
    if not ck.exists():
        raise SystemExit(f"no esta {ck}")
    cfg = full_config(yaml.safe_load(
        (REPO / "configs" / "networks" / f"{red}.yaml").read_text()))
    if local:
        sys.path.insert(0, str(carpeta / "nn"))
        from red_local import PlanaSinPadding          # noqa: E402
        m = PlanaSinPadding(cfg)
    else:
        m = build_model(cfg)
    e = torch.load(ck, map_location="cpu", weights_only=False)
    m.load_state_dict(e["model"] if "model" in e else e["state_dict"])
    m.eval()
    return m, e


def _comprobar(exp: str, red: str, local: bool) -> bool:
    carpeta = REPO / "experimentos" / exp
    m, e = cargar(exp, red, local, "last")
    W = m.center_convs[0].weight.detach()
    normas = [round(float(v), 5) for v in W.flatten(1).norm(dim=1)]
    stops = sorted((carpeta / "evaluacion").glob("stop-*/resumen.json"))
    guardadas = json.loads(stops[-1].read_text())["kernels"]["norma_l2"]
    ok = all(abs(a - b) < 1e-4 for a, b in zip(normas, guardadas))
    epoca = e.get("epoch", "?")
    print(f"  {exp.replace('2026-09-03-','') :<34} last.pt · epoca {epoca:<3} "
          f"· L1 {tuple(W.shape)} · {'✓ casa con ' + stops[-1].parent.name if ok else '✗ NO CASA'}")
    if not ok:
        print(f"      guardadas {guardadas}\n      del .pt   {normas}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default=None)
    a = p.parse_args()
    todo = [x for x in EXPERIMENTOS if a.exp in (None, x[0])]
    if not todo:
        raise SystemExit(f"'{a.exp}' no esta en la lista")
    print(f"comprobando {len(todo)} juego(s) de pesos:")
    ok = all(_comprobar(*x) for x in todo)
    print("todos cargan y coinciden con lo medido." if ok
          else "⚠ HAY PESOS QUE NO CASAN con lo que dice su ultimo stop")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
