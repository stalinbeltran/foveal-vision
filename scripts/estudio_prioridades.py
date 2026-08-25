#!/usr/bin/env python3
r"""Crea los recorridos de prioridad 1 y 2, con el criterio ya escrito.

Que es esto
-----------
`docs/plan-prioridades-2026-08-25.md` fija QUE se mide y COMO se lee, antes de
mirar nada. Este script solo lo traduce a recorridos en `sweeps/`, para que la
lista de estudios y la lista de barridos no puedan divergir: un estudio que
esta en el plan y no aqui no se corre, y uno que esta aqui y no en el plan no
tiene criterio.

No alquila nada y no entrena nada. Lo que se hace despues con los recorridos es
`scripts/estudio_flota.py`, que es quien gasta.

    .venv/bin/python scripts/estudio_prioridades.py --dataset dirty1000-80px-16px-r20260824
    .venv/bin/python scripts/estudio_prioridades.py --dataset <B> --solo borde-ancho

La ATADURA de E1, que es la razon de que este script exista
-----------------------------------------------------------
`borde-ancho` barre `border_px` manteniendo el anillo en 2 celdas, o sea con
`border_reduce` = `border_px`/2. Eso es una DIAGONAL, no un producto cartesiano:
con los dos como ejes libres saldrian tambien puntos como (border_px=8,
border_reduce=8), que es OTRA red y que `aggregate_seeds` agruparia bajo el
mismo valor del eje. Se declara con `couple` (ver `fv/sweeps/spec.py`,
`expand_points`) y por eso el rango de la atadura se escribe aqui al lado del
del eje: si uno se toca, el otro salta por `couple_length_mismatch`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.sweeps.generate import generate_sweep       # noqa: E402
from fv.sweeps.spec import SweepError               # noqa: E402
from fv.sweeps.store import SweepStore              # noqa: E402

RECIPE = "plan40"
OBJECTIVE = "f1"
EPOCHS = 150                    # el de bs5-L4 / nl5-L4 / d5-L4

# La foveada vigente: ws16-p2-d2-L4.
FOVEADA = {"n_layers": 4, "channels": [16] * 4}

ESTUDIOS = [
    # ---------------------------------------------------------------- prioridad 1
    {
        "name": "borde-ancho",
        "que": "E1 - mas contexto A COSTE CONSTANTE: N=20 en los cinco puntos",
        "axis": "border_px",
        "range": [4, 8, 10, 12, 16],
        # el anillo se queda en 2 celdas -> N no se mueve -> mismos parametros.
        # Sin esto el eje mediria "mas area Y mas parametros", que son dos cosas.
        "couple": {"border_reduce": {"axis": "border_px",
                                     "values": [2, 4, 5, 6, 8]}},
        "base": FOVEADA,
        "border_px": 4,
        "epochs": EPOCHS,
    },
    # ---------------------------------------------------------------- prioridad 2
    {
        "name": "pw-fov",
        "que": "E5 - pos_weight: el mando que ataca el cuello de botella de DETECCION",
        "axis": "pos_weight",
        "range": [1.0, 2.0, 4.0, 8.0],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS,
    },
    {
        "name": "mon-fov",
        "que": "E6 - monitor: hoy el checkpoint se elige con una metrica y se rankea con otra",
        "axis": "monitor",
        "range": ["val_loss", "val_f1"],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS,
    },
    {
        "name": "sch-fov",
        "que": "E7 - scheduler: tope 100 y NO 150, para que cosine llegue a bajar",
        "axis": "scheduler",
        "range": ["none", "cosine"],
        "base": FOVEADA, "border_px": 4,
        # cosine planifica sobre T_max = recipe.epochs, o sea el TOPE. Con 150 y
        # parada real entre las epocas 32 y 81, el lr solo bajaria a ~0,75 y esto
        # mediria "cosine casi sin aplicar". Con 100 la bajada es real y el tope
        # sigue por encima de la parada mas tardia observada: manda patience (R1).
        "epochs": 100,
    },
    {
        "name": "ch-fov",
        "que": "E8 - channels: y el interes puede estar HACIA ABAJO (si 8 empata, sobra la mitad)",
        "axis": "channels",
        "range": [[8] * 4, [16] * 4, [24] * 4, [32] * 4],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS,
    },
    {
        "name": "kc-fov",
        "que": "E9 - k_center: el unico eje donde proxy y tarea se contradicen EN EL SIGNO",
        "axis": "k_center",
        "range": [3, 5, 7],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS,
    },
    {
        "name": "ov-fov",
        "que": "E10 - overlap_fovea_px: el 0 (ramas disjuntas) no era expresable hasta el 25-ago",
        "axis": "overlap_fovea_px",
        "range": [0, 1, 2, 4],
        "base": FOVEADA, "border_px": 4, "epochs": EPOCHS,
    },
    {
        "name": "red-fov",
        "que": "E10b - border_reduce a border_px FIJO. NO es cost-neutral: N = 20/24/32",
        "axis": "border_reduce",
        "range": [4, 2, 1],
        "base": FOVEADA,
        "border_px": 8,          # el area se queda quieta; cambia como se condensa
        "epochs": EPOCHS,
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--solo", action="append",
                    help="crea solo estos recorridos (repetible)")
    ap.add_argument("--rehacer", action="store_true",
                    help="borra y rehace el que ya exista (pierde sus runs)")
    args = ap.parse_args()

    store = SweepStore()
    existentes = set(store.list_names() if hasattr(store, "list_names")
                     else [p.name for p in (ROOT / "sweeps").iterdir() if p.is_dir()])
    pedidos = set(args.solo or [])
    creados, saltados, fallidos = [], [], []

    for est in ESTUDIOS:
        name = est["name"]
        if pedidos and name not in pedidos:
            continue
        if name in existentes and not args.rehacer:
            print(f"  = {name:14s} ya existe, se deja (--rehacer para borrarlo)")
            saltados.append(name)
            continue
        try:
            spec = generate_sweep(
                name, args.dataset, est["axis"], est["range"],
                base_recipe=RECIPE, objective=OBJECTIVE,
                budget={"epochs": est["epochs"]},
                seeds=args.semillas, device="cpu",
                overrides=est["base"], border_px=est.get("border_px"),
                couple=est.get("couple"),
                study="prioridades-2026-08-25", sstore=store,
            )
        except SweepError as e:
            print(f"  ! {name:14s} {e.code}: {e.message}\n      {e.hint}")
            fallidos.append(name)
            continue
        n = len(spec["points"])
        desc = spec["descarte"] if "descarte" in spec else spec.get("discarded") or []
        print(f"  + {name:14s} {len(spec['space'][est['axis']])} valores x "
              f"{args.semillas} semillas = {n:3d} runs   [{est['que']}]")
        for d in desc:
            print(f"      descartado {d['point']}: {d['problems'][0]['message']}")
        creados.append((name, n))

    print(f"\n{len(creados)} recorridos creados, {len(saltados)} ya estaban, "
          f"{len(fallidos)} fallaron.")
    if creados:
        print(f"Runs nuevos: {sum(n for _, n in creados)}")
        print("\nSiguiente paso (esto SI gasta):")
        print("  .venv/bin/python scripts/estudio_flota.py \\")
        for name, _ in creados:
            print(f"      --sweep {name} \\")
        print("      --cpu E5-26 --max-price 0.12 --criba 2 --git --estimar")
    return 1 if fallidos else 0


if __name__ == "__main__":
    raise SystemExit(main())
