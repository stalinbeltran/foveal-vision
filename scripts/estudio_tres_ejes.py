#!/usr/bin/env python3
r"""Crea los tres recorridos de docs/plan-tres-ejes.md. No entrena nada.

Las constantes de abajo son las del documento, copiadas AQUI y en ningun otro
sitio (como `plan_lr_L4.py`). El criterio de lectura vive en el documento y este
script no lo re-implementa: solo declara el espacio y llama al generador del
proyecto, que valida cada punto con el MISMO `check_run` que cualquier otra
puerta que entrena.

    .venv/bin/python scripts/estudio_tres_ejes.py --dataset dirty1000-80px-16px-rXXXX
    .venv/bin/python scripts/estudio_tres_ejes.py --dataset ... --estimar

El ORDEN de cada rango es la mitigacion (plan-40h.md §7.3, plan-lr-alto §1): con
`--reparto seed` cada maquina entrena los valores EN ESTE ORDEN, asi que si algo
se corta lo que sobrevive es el vigente y su vecindario, y lo que falta son los
extremos. El ranking agrega por valor, no por orden: **no cambia ningun
resultado**, solo que se pierde si algo se corta.
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

# --- constantes de docs/plan-tres-ejes.md §1 ---------------------------------
RECIPE = "plan40"          # el vigente: lr 0,0014 · adam · batch 85 · patience 10
EPOCHS_CAP = 150           # alto A PROPOSITO: `patience` tiene que ser quien pare
SEEDS = 5
OBJECTIVE = "f1"
# la base vigente, la misma de `lr-alto-L4` (base_label ws16-p2-d2-L4). `channels`
# viaja con `n_layers` porque la profundidad vive en los dos sitios.
BASE_NETWORK = {"n_layers": 4, "channels": [16, 16, 16, 16]}

EJES = [
    {
        "name": "bs5-L4", "axis": "batch_size",
        # log-espaciado factor 1,5 alrededor del vigente (85). Orden: vigente,
        # vecinos, extremos. §2.1 del documento dice por que estos y no otros.
        "range": [85, 57, 128, 38, 192],
        "vigente": 85,
    },
    {
        "name": "nl5-L4", "axis": "n_layers",
        # el mismo rango que `p40-confirm-n_layers`, que lo dejo acotado por los
        # dos lados (gano 4, con 2 y 5 por debajo). §2.2.
        "range": [4, 3, 5, 2],
        "vigente": 4,
    },
    {
        "name": "d5-L4", "axis": "d",
        # el rango auto es [1..6]; se recortan 5 y 6 porque `proxy-c-d` los midio
        # por debajo del 2 y el optimo cae a la izquierda. §2.3.
        "range": [2, 1, 3, 4],
        "vigente": 2,
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True,
                    help="dataset de ventanas (B) sobre el que se entrena")
    ap.add_argument("--sufijo", default="",
                    help="se anade al nombre de cada recorrido")
    ap.add_argument("--estimar", action="store_true",
                    help="ademas, imprime el tiempo y el coste estimados")
    ap.add_argument("--reparto", choices=("seed", "run"), default="seed")
    args = ap.parse_args()

    store = SweepStore()
    creados = []
    for eje in EJES:
        nombre = eje["name"] + (f"-{args.sufijo}" if args.sufijo else "")
        if store.exists(nombre):
            spec = store.spec(nombre)
            if spec.get("window_dataset") != args.dataset:
                print(f"ERROR: '{nombre}' ya existe pero sobre el dataset "
                      f"'{spec['window_dataset']}', no '{args.dataset}'.\n"
                      f"  Dos recorridos con el mismo nombre y distinto dato es "
                      f"la forma de que la no-comparabilidad no se note nunca.\n"
                      f"  Usa --sufijo para darle un nombre propio.", file=sys.stderr)
                return 2
            print(f"'{nombre}' ya existe ({len(spec['points'])} puntos) -- se reutiliza")
            creados.append(nombre)
            continue
        try:
            e = generate_sweep(nombre, args.dataset, eje["axis"], eje["range"],
                               base_recipe=RECIPE, objective=OBJECTIVE,
                               budget={"epochs": EPOCHS_CAP}, seeds=SEEDS,
                               overrides=BASE_NETWORK, device="cpu", sstore=store)
        except SweepError as ex:
            print(f"NO se pudo crear '{nombre}': [{ex.code}] {ex.message} -> {ex.hint}",
                  file=sys.stderr)
            return 2
        desc = e.get("discarded") or []
        print(f"'{nombre}' creado: base {e['base_label']}, eje {eje['axis']} "
              f"{eje['range']} x {SEEDS} semillas = {len(e['points'])} puntos, "
              f"tope {EPOCHS_CAP} epocas"
              + (f"  ⚠ {len(desc)} descartados: "
                 f"{json.dumps([d['point'] for d in desc])}" if desc else ""))
        creados.append(nombre)

    print(f"\n{len(creados)} recorridos listos: {', '.join(creados)}")
    print("Lanzalos con UNA sola flota (comparten el pozo de maquinas):")
    print("  .venv/bin/python scripts/estudio_flota.py "
          + " ".join(f"--sweep {c}" for c in creados)
          + f" --reparto {args.reparto} --cpu 'E5-26' --criba 4 --git")

    if args.estimar:
        import estudio_estimar as E
        sys.argv = ["estudio_estimar"] + [x for c in creados for x in ("--sweep", c)] \
            + ["--reparto", args.reparto]
        E.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
