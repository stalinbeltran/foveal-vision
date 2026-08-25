#!/usr/bin/env python3
r"""Crea los recorridos de la CNN PLANA (docs/plan-cnn-plana.md), en dos fases.

Que red es, y por que estos numeros
------------------------------------
La "cnn tipica" contra la que se compara la foveada tiene que cumplir DOS cosas a
la vez, y ninguna es negociable si la comparacion va a significar algo:

1. **La misma entrada.** No el mismo tensor: la misma INFORMACION. La foveada
   (fovea 16px + borde 4px comprimido x2) tiene un tensor de 20x20, pero el area
   original que cubre es **24x24**. MEDIDO: `dims_of(...).original_size == 24`.
   Asi que la plana lleva el MISMO borde de 4px sin comprimir
   (border_px=4, border_reduce=1 => 24x24 de entrada) y la fovea sigue siendo la
   ventana de 16 (contrato ①a).
2. **Aproximadamente los mismos parametros.** Con channels=[16]x4 la plana sale a
   117.724 contra 167.852 de la foveada: **0,70x**. La cabeza es el 92 % del
   modelo, y una rama sobre 24x24 da 9.216 features planas contra las 12.800 de
   dos ramas sobre 20x20. Ensanchando a 22: **165.430, o sea 0,99x**. MEDIDO.

⚠ `d` NO se barre aqui. Con regions=single, subir `d` agranda el AREA ORIGINAL
(d=2 -> 32x32, medido), asi que dejaria de ser "la misma entrada" y romperia la
premisa. Es el unico eje del estudio foveado que no se replica, y esta es la
razon.

Las dos fases, y para que sirve cada una
-----------------------------------------
**tanteo** -- 2 semillas, rangos ANCHOS (16x de span), para ACOTAR. Su unico
trabajo es decir en que vecindario mirar: un optimo de otra arquitectura no se
hereda -- es literalmente el motivo por el que existio `plan-lr-L4` -- asi que
partir de los rangos de la foveada seria suponer la respuesta.

⚠ Con 2 semillas la permutacion exacta da 2 arreglos: el tanteo **no puede
declarar ningun ganador** y no lo intenta. Es la misma regla que plan-40h.md §2
escribio para su cribado de 1 semilla. Lo que si distingue es una zona donde el
entrenamiento va bien de otra donde diverge, y eso basta para acotar.

**final** -- 5 semillas sobre los rangos que el tanteo acoto, con una maquina por
recorrido x semilla, igual que el estudio foveado. Los rangos se pasan por la
linea de ordenes A PROPOSITO: asi la decision que se toma al ver el tanteo queda
escrita en el comando y en el documento, no escondida en una constante.

    .venv/bin/python scripts/estudio_plana.py --fase tanteo --dataset <B>
    .venv/bin/python scripts/estudio_plana.py --fase final --dataset <B> \
        --lr 0.0007,0.0014,0.0028 --batch-size 43,85,170 --n-layers 3,4,5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fv.sweeps.generate import generate_sweep       # noqa: E402
from fv.sweeps.spec import SweepError               # noqa: E402
from fv.sweeps.store import SweepStore              # noqa: E402

RECIPE = "plan40"
EPOCHS_CAP = 150           # alto A PROPOSITO: `patience` tiene que ser quien pare
OBJECTIVE = "f1"
BORDER_PX = 4              # los mismos 4 px de contexto, sin comprimir
BASE = {"regions": "single", "border_reduce": 1, "n_layers": 4,
        "channels": [22] * 4}

# Rangos del TANTEO: anchos y log-espaciados, centrados en el optimo de la
# FOVEADA -- que aqui es solo un punto de partida razonable, no una prediccion.
# El orden pone ese centro primero: si algo se corta, lo que sobrevive es el ancla.
TANTEO = [
    {"name": "pl-t-lr", "axis": "lr",
     "range": [0.0014, 0.0007, 0.0028, 0.00035, 0.0056]},
    {"name": "pl-t-bs", "axis": "batch_size",
     "range": [85, 43, 170, 24, 340]},
    {"name": "pl-t-nl", "axis": "n_layers",
     "range": [4, 3, 5, 2, 6]},
]
FINAL = {"lr": "pl5-lr", "batch_size": "pl5-bs", "n_layers": "pl5-nl"}


def crear(nombre: str, dataset: str, axis: str, rango: list, seeds: int,
          store: SweepStore) -> str:
    if store.exists(nombre):
        spec = store.spec(nombre)
        if spec.get("window_dataset") != dataset:
            print(f"ERROR: '{nombre}' ya existe sobre otro dataset "
                  f"('{spec['window_dataset']}').", file=sys.stderr)
            raise SystemExit(2)
        print(f"'{nombre}' ya existe ({len(spec['points'])} puntos) -- se reutiliza")
        return nombre
    try:
        e = generate_sweep(nombre, dataset, axis, rango, base_recipe=RECIPE,
                           objective=OBJECTIVE, budget={"epochs": EPOCHS_CAP},
                           seeds=seeds, overrides=BASE, border_px=BORDER_PX,
                           device="cpu", sstore=store)
    except SweepError as ex:
        print(f"NO se pudo crear '{nombre}': [{ex.code}] {ex.message} -> {ex.hint}",
              file=sys.stderr)
        raise SystemExit(2)
    desc = e.get("discarded") or []
    ch = e["base_network_value"]["channels"]
    print(f"'{nombre}' creado: base {e['base_label']} (N={e['base_network_value']['N']}, "
          f"1 rama, channels {ch}), eje {axis} {rango} x {seeds} semillas = "
          f"{len(e['points'])} puntos"
          + (f"  ⚠ {len(desc)} descartados: "
             f"{json.dumps([d['point'] for d in desc])}" if desc else ""))
    return nombre


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fase", choices=("tanteo", "final"), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--semillas", type=int, default=0,
                    help="por defecto 2 en tanteo y 5 en final")
    ap.add_argument("--lr", default="", help="rango del eje lr (fase final)")
    ap.add_argument("--batch-size", default="", help="rango de batch_size (final)")
    ap.add_argument("--n-layers", default="", help="rango de n_layers (final)")
    args = ap.parse_args()

    store = SweepStore()
    seeds = args.semillas or (2 if args.fase == "tanteo" else 5)
    creados = []
    if args.fase == "tanteo":
        for eje in TANTEO:
            creados.append(crear(eje["name"], args.dataset, eje["axis"],
                                 eje["range"], seeds, store))
    else:
        pedidos = {"lr": args.lr, "batch_size": args.batch_size,
                   "n_layers": args.n_layers}
        if not any(pedidos.values()):
            print("ERROR: la fase final necesita al menos un rango "
                  "(--lr / --batch-size / --n-layers).\n"
                  "  Se pasan a mano a proposito: la decision que se toma al ver "
                  "el tanteo queda escrita en el comando.", file=sys.stderr)
            return 2
        for eje, texto in pedidos.items():
            if not texto:
                continue
            conv = float if eje == "lr" else int
            rango = [conv(x) for x in texto.split(",") if x.strip()]
            creados.append(crear(FINAL[eje], args.dataset, eje, rango, seeds, store))

    print(f"\n{len(creados)} recorridos listos: {', '.join(creados)}")
    print("Lanzalos con UNA sola flota:")
    print("  .venv/bin/python scripts/estudio_flota.py "
          + " ".join(f"--sweep {c}" for c in creados)
          + " --reparto seed --cpu 'E5-26' --criba 3 --git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
