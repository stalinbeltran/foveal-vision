#!/usr/bin/env python3
r"""Cerrar `batch_size` por ARRIBA, en las dos redes. Tanteo primero.

Por que existe
--------------
`bs5-L4` (docs/plan-tres-ejes.md §7.1) dejo el eje **sin acotar por la derecha**:
el ganador nominal fue **192, que es el extremo del rango**, y la regla escrita
antes dice que eso no es un optimo sino el final de la regla. El eje ademas salio
PLANO entre 57 y 192 (0,9302 a 0,9351), asi que no se sabe si sigue plano, si
sube, o donde cae.

Y no es una curiosidad: subir el batch **abarata** el reloj (192 va 1,08x mas
rapido por epoca que 85, medido), asi que saber hasta donde se puede subir sin
perder calidad es dinero y tiempo directamente.

Dos fases, y la primera es barata a proposito
----------------------------------------------
**tanteo** (2 semillas): rangos que suben x2 hasta 1536, o sea 18x el vigente.
Su unico trabajo es encontrar **donde cae el eje**. Con 2 semillas la permutacion
da 2 arreglos: no declara ganador y no lo intenta.

**final** (5 semillas): sobre el rango que el tanteo acote, con las reglas R1-R6
de plan-tres-ejes.md §5.

⚠ EL TOPE DE EPOCAS SUBE A 300 EN EL TANTEO, y hay que decir por que. Un batch
grande da menos actualizaciones por epoca, asi que necesita mas epocas para el
mismo trabajo. Con el tope de 150 de `bs5-L4`, los puntos altos pararian POR EL
TOPE y no por `patience` -- que es exactamente el defecto (R1) que invalido los
tres estudios de `batch_size` de julio, y seria repetirlo justo en la zona que se
quiere medir. Las epocas altas son ademas baratas ahi: a batch 1536 son 55 pasos
por epoca contra 989 a batch 85.

    .venv/bin/python scripts/estudio_bs_alto.py --dataset <B>
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
EPOCHS_CAP = 300            # ver el aviso del docstring

# La FOVEADA: su vigente es 85 y su ganador nominal fue 192 (el extremo). El
# rango arranca EN 192 -- que asi deja de ser extremo y pasa a ser ancla -- y
# sube x2 hasta 1536.
FOVEADA = {"name": "bs-alto-fov", "base": {"n_layers": 4, "channels": [16] * 4},
           "border_px": None, "range": [192, 384, 768, 1536]}

# La PLANA (docs/plan-cnn-plana.md §6): su tanteo llego solo a 340. Mismo
# tratamiento, arrancando en 170 para solapar con lo ya medido.
PLANA = {"name": "bs-alto-pl",
         "base": {"regions": "single", "border_reduce": 1, "n_layers": 4,
                  "channels": [22] * 4},
         "border_px": 4, "range": [170, 340, 680, 1360]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--semillas", type=int, default=2)
    ap.add_argument("--solo", choices=("fov", "pl"), default="",
                    help="crear solo uno de los dos")
    args = ap.parse_args()

    store = SweepStore()
    creados = []
    quienes = [FOVEADA, PLANA]
    if args.solo == "fov":
        quienes = [FOVEADA]
    elif args.solo == "pl":
        quienes = [PLANA]
    for q in quienes:
        if store.exists(q["name"]):
            print(f"'{q['name']}' ya existe -- se reutiliza")
            creados.append(q["name"])
            continue
        kw = {"border_px": q["border_px"]} if q["border_px"] is not None else {}
        try:
            e = generate_sweep(q["name"], args.dataset, "batch_size", q["range"],
                               base_recipe=RECIPE, objective=OBJECTIVE,
                               budget={"epochs": EPOCHS_CAP}, seeds=args.semillas,
                               overrides=q["base"], device="cpu", sstore=store, **kw)
        except SweepError as ex:
            print(f"NO se pudo crear '{q['name']}': [{ex.code}] {ex.message}",
                  file=sys.stderr)
            return 2
        print(f"'{q['name']}' creado: base {e['base_label']} "
              f"(N={e['base_network_value']['N']}, channels "
              f"{e['base_network_value']['channels']}), batch_size {q['range']} "
              f"x {args.semillas} semillas = {len(e['points'])} puntos, "
              f"tope {EPOCHS_CAP} epocas")
        creados.append(q["name"])
    print(f"\n{len(creados)} recorridos: {', '.join(creados)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
