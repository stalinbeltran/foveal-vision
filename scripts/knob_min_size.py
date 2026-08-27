#!/usr/bin/env python3
r"""`min_size`: el ultimo knob de inferencia (F) que seguia SIN MEDIR.

Por que aparte de `knobs_f.py`
------------------------------
Aquel barre `threshold`, `stride` y `nms_radius` -- los tres que ya tenian
optimo publicado -- y `min_size` NO entra en su rejilla. Por eso la tabla del
coordinador lo lleva como «sin medir» desde julio: no es que diera un resultado
flojo, es que nadie lo barrio.

Es el mas barato del inventario entero: es **post-hoc**, o sea que se aplica
sobre pesos que ya existen. Cero alquileres, cero entrenamientos.

    .venv/bin/python scripts/knob_min_size.py --run <run con best.pt>

Se mide DOS veces a proposito, y las dos hacen falta:
  1. con los otros knobs en su DEFAULT vigente (0,5 · n/2 · n/2), que es la
     configuracion con la que estan publicados todos los numeros del proyecto;
  2. con los otros knobs en el optimo medido, por si `min_size` interactua --
     un minimo de tamano filtra lo que la deteccion deja pasar, asi que su
     efecto puede depender de cuanto deje pasar el `threshold`.
Medir solo (1) diria «no hace nada» aunque en (2) si lo hiciera, y al reves.

⚠ Esto NO aplica nada: la decision F15 (del usuario, 2026-07-26) sigue CERRADA
en NO, porque cambiar los defaults re-escala todos los numeros publicados. Aqui
solo se rellena la casilla que faltaba.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fv.task import task_score                    # noqa: E402
from fv.training.registry import RunStore         # noqa: E402
# `f1_de` y `DEFAULTS` se IMPORTAN de knobs_f.py en vez de repetirse: la metrica
# de tarea es `macro.f1` y los defaults vigentes son los que aquel publica. Un
# numero definido dos veces es un numero que acaba divergiendo, y aqui ademas
# haria incomparables los dos estudios de knobs.
from knobs_f import DEFAULTS, f1_de               # noqa: E402

# El default es 4,0. Se barre a los dos lados y hasta un valor que TIENE que
# hacer dano (32 px sobre imagenes de 60x80 descarta casi cualquier parrafo):
# un rango cuyo extremo no empeora no ha acotado nada.
MIN_SIZES = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
# El optimo de F medido en el #12 (2026-08-26). Se cita como lo que es: medido
# sobre OTROS pesos y otro dataset, asi que aqui es un punto de trabajo, no una
# prediccion.
OPTIMO = {"threshold": 0.3, "stride": 2, "nms_radius": 16.0}


def barre(run: str, split: str, knobs: dict, etiqueta: str) -> list:
    print(f"\n### {etiqueta}  (threshold={knobs['threshold']} "
          f"stride={knobs['stride']} nms_radius={knobs['nms_radius']})")
    print("\n| `min_size` | f1 de tarea | Δ vs default (4,0) |")
    print("|---:|---:|---:|")
    filas, base = [], None
    for ms in MIN_SIZES:
        try:
            r = task_score(run, split, min_size=ms, **knobs)
        except Exception as e:
            print(f"| {ms} | ERROR {type(e).__name__}: {e} | - |")
            continue
        v = f1_de(r)
        if ms == 4.0:
            base = v
        filas.append({"min_size": ms, "f1": v})
        print(f"| {ms:g} | {v:.4f} | {'' if base is None else f'{v-base:+.4f}'} |"
              .replace(".", ","))
    if filas:
        mejor = max(filas, key=lambda f: f["f1"])
        interior = 0 < MIN_SIZES.index(mejor["min_size"]) < len(MIN_SIZES) - 1
        print(f"\n**Óptimo: `min_size` = {mejor['min_size']:g}** "
              f"(f1 {mejor['f1']:.4f})".replace(".", ",")
              + (" — **interior**, o sea acotado por los dos lados."
                 if interior else " — ⚠ **en el borde del rango**: sin acotar."))
        if base is not None:
            print(f"El default 4,0 deja {mejor['f1']-base:+.4f}".replace(".", ",")
                  + " sobre la mesa.")
    return filas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    if not (RunStore().path(args.run) / "best.pt").exists():
        print(f"'{args.run}' no tiene best.pt: sin pesos no hay inferencia que medir.")
        return 1

    print(f"# `min_size` sobre `{args.run}` (split {args.split})")
    a = barre(args.run, args.split, DEFAULTS, "Con los knobs en su DEFAULT vigente")
    b = barre(args.run, args.split, OPTIMO, "Con los otros knobs en el óptimo del #12")

    if a and b:
        ma = max(a, key=lambda f: f["f1"])["min_size"]
        mb = max(b, key=lambda f: f["f1"])["min_size"]
        print(f"\n**¿Interactúa?** Óptimo con defaults = {ma:g}, con los knobs "
              f"buenos = {mb:g} → "
              + ("**no cambia**: `min_size` se puede fijar sin mirar los otros."
                 if ma == mb else
                 "⚠ **cambia**, así que `min_size` NO es independiente de los "
                 "otros knobs y no se puede fijar por separado."))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"run": args.run, "split": args.split, "rejilla": MIN_SIZES,
             "con_defaults": a, "con_optimo": b,
             "knobs_defaults": DEFAULTS, "knobs_optimo": OPTIMO}, indent=1),
            encoding="utf-8")
        print(f"\nJSON en {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
