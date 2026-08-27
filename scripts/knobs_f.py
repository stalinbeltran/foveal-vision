#!/usr/bin/env python3
r"""E4 — los knobs de INFERENCIA (dominio F), re-medidos sobre los modelos de hoy.

Por que se re-mide algo que ya estaba medido
---------------------------------------------
La medida que sostiene la decision **F15** es del **2026-07-26**, y se hizo sobre
runs de la red **L2** y del dataset **anterior**. Desde entonces cambiaron las dos
cosas: la red vigente es L4 y el dataset es `dirty1000-80px-16px-r20260824`, que
resulto ser **mas facil** (+0,0095 de escala). Un numero que decide algo tan caro
como re-escalar todo lo publicado no deberia venir de memoria ni de otra red.

Esto **no cambia ningun default**. F15 esta CERRADA en `docs/decisiones.md` con un
NO del usuario (2026-07-26), y este script no la reabre: solo deja el numero de hoy
al lado del de julio para que, si se reabre, se reabra con datos y no de memoria.

Que mide, y por que estos tres runs y no uno
---------------------------------------------
La pregunta que hace util este barrido no es «cual es el optimo» sino **«es el
MISMO optimo para modelos de calidad distinta?»**. Si lo es, los knobs son un
ajuste GLOBAL y comparar runs con knobs fijos no sesga el orden. Si no lo es, los
knobs son parte de la identidad de cada run y entonces toda comparacion publicada
esta mezclando dos cosas. Por eso se cogen tres runs de calidad deliberadamente
distinta -- el mejor, uno mediano y el peor de los que tengan checkpoint.

⚠ Y el hallazgo de julio que importa mas que el propio optimo, y que hay que
volver a comprobar: las ganancias eran DESIGUALES (el run malo ganaba 4x mas que
el bueno), asi que los knobs buenos **comprimen** la separacion entre modelos
mientras el ruido se queda igual. Mejoran el numero absoluto y empeoran su poder
de distinguir, que es justo para lo que se usa la metrica.

    .venv/bin/python scripts/knobs_f.py --json reportes/knobs-f-20260825.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv import datarepo
from fv.task import task_score                      # noqa: E402
from fv.training.registry import RunError, RunStore  # noqa: E402

# La rejilla YA ACOTADA de metrica-de-tarea.md §9.2: los tres optimos salieron
# INTERIORES tras extenderla, asi que se reusa tal cual. Extenderla otra vez seria
# volver a acotar algo ya acotado; recortarla, medir con una rejilla que no acota.
THRESHOLDS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
STRIDES = [2, 4, 8]
NMS = [4, 8, 12, 16, 20]
DEFAULTS = {"threshold": 0.5, "stride": 8, "nms_radius": 8}   # n=16 -> n/2, n/2

# Suelo para que un run cuente como "modelo malo" y no como "entrenamiento
# MUERTO". No es lo mismo y la diferencia rompe el estudio: el proyecto tiene
# runs con f1 EXACTAMENTE 0,0000 -- el fallo bimodal que plan-plana.md §6.1
# documenta, y que en `pl-t-lr` dejo una semilla colapsada en lr=0,0028. Ese run
# no es un modelo de baja calidad, es una red que no aprendio nada: su "optimo
# de knobs" es el que mejor ordena ruido, y meterlo en la comparacion contesta
# la pregunta con un punto que no significa nada.
# Medido el 2026-08-26: sin este suelo, el "malo" elegido era justamente ese
# colapsado, con optimo 0,0437 en threshold=0,1 -- o sea el knob que mas detecta,
# porque cualquier cosa es mejor que nada.
F1_MINIMO_UTIL = 0.05


def runs_con_checkpoint(store: RunStore) -> list:
    fuera = []
    for d in sorted(datarepo.iter_dirs("runs"), key=lambda d: d.name):
        if not d.is_dir() or not (d / "best.pt").exists():
            continue
        try:
            st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if st.get("status") != "done":
            continue
        fuera.append(d.name)
    return fuera


# El f1 de tarea vive en `macro.f1` (parrafo por imagen, promediado por imagen),
# que es el que reporta metrica-de-tarea.md y el que trae su `sem`. NO hay un
# "f1" en la raiz: leerlo de ahi da None en silencio para TODOS los runs, y el
# barrido concluye "no hay runs puntuables" sin que nada haya fallado.
def f1_de(r: dict) -> float:
    return r["macro"]["f1"]


def f1_por_defecto(run: str, split: str) -> tuple:
    """(f1, None) o (None, motivo). El motivo VIAJA: un run que no se puede
    puntuar y se salta en silencio es un hueco que luego se lee como cero."""
    try:
        return f1_de(task_score(run, split, **DEFAULTS)), None
    except RunError as e:
        return None, f"{e.code}: {e.message}"
    except (KeyError, OSError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="val")
    ap.add_argument("--runs", type=int, default=3,
                    help="cuantos runs de calidad distinta (bueno/medio/malo)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    store = RunStore()
    candidatos = runs_con_checkpoint(store)
    if not candidatos:
        print("No hay ningun run con best.pt y status done.")
        return 1
    print(f"{len(candidatos)} runs con checkpoint. Puntuando con los defaults "
          f"para elegir {args.runs} de calidad distinta...", flush=True)

    t0 = time.time()
    base, motivos = [], {}
    for r in candidatos:
        v, motivo = f1_por_defecto(r, args.split)
        if v is None:
            motivos[r] = motivo
        else:
            base.append((v, r))
    base.sort()
    muertos = [(v, r) for v, r in base if v < F1_MINIMO_UTIL]
    base = [(v, r) for v, r in base if v >= F1_MINIMO_UTIL]
    if muertos:
        print(f"  ({len(muertos)} runs con f1 < {F1_MINIMO_UTIL} FUERA: son "
              f"entrenamientos colapsados, no modelos malos. p.ej. "
              f"{muertos[0][1]} con f1={muertos[0][0]:.4f})")
    if len(base) < args.runs:
        print(f"solo {len(base)} runs puntuables de {len(candidatos)}; hacen "
              f"falta {args.runs}. Las razones, que es lo que hace falta para "
              f"arreglarlo:")
        for r, m in list(motivos.items())[:8]:
            print(f"    {r}: {m}")
        return 1
    if motivos:
        print(f"  ({len(motivos)} runs no puntuables, se saltan: "
              f"{sorted(motivos.values())[0]} ...)")
    # el peor, el mediano y el mejor: la calidad tiene que ser DISTINTA a proposito
    elegidos = [base[0], base[len(base) // 2], base[-1]]
    etiquetas = ["malo", "medio", "bueno"]
    print(f"  elegidos en {time.time() - t0:.0f} s:")
    for (v, r), et in zip(elegidos, etiquetas):
        print(f"    {et:6s} {r}  f1_defecto={v:.4f}")

    filas = []
    for (v_def, run), et in zip(elegidos, etiquetas):
        print(f"\n=== {et}: {run} ===", flush=True)
        mejor = None
        n = 0
        for th in THRESHOLDS:
            for st in STRIDES:
                for nm in NMS:
                    try:
                        r = task_score(run, args.split, threshold=th, stride=st,
                                       nms_radius=nm)
                    except (RunError, KeyError, OSError, ValueError) as e:
                        print(f"    (combo th={th} st={st} nms={nm} fallo: {e})")
                        continue
                    n += 1
                    v = f1_de(r)
                    if mejor is None or v > mejor["f1"]:
                        mejor = {"f1": v, "threshold": th, "stride": st,
                                 "nms_radius": nm}
        if mejor is None:
            print("  no se pudo puntuar")
            continue
        # el PUESTO del default entre todas las combinaciones, que es lo que dice
        # si el default es una eleccion mala o solo no-optima
        print(f"  {n} combinaciones · defecto {v_def:.4f} -> optimo "
              f"{mejor['f1']:.4f}  (+{mejor['f1'] - v_def:.4f})")
        print(f"  optimo en threshold={mejor['threshold']} stride={mejor['stride']} "
              f"nms_radius={mejor['nms_radius']}")
        filas.append({"etiqueta": et, "run": run, "f1_defecto": v_def,
                      "f1_optimo": mejor["f1"], "ganancia": mejor["f1"] - v_def,
                      "optimo": mejor, "combinaciones": n})

    print(f"\n--- resumen ({time.time() - t0:.0f} s) ---")
    print("run                                    calidad  defecto   optimo  ganancia")
    for f in filas:
        print(f"{f['run'][:38]:38s} {f['etiqueta']:7s} {f['f1_defecto']:7.4f} "
              f"{f['f1_optimo']:8.4f}  {f['ganancia']:+.4f}")
    if len(filas) >= 2:
        # LA pregunta de F11/F15: ¿los knobs buenos comprimen la separacion?
        d_def = max(f["f1_defecto"] for f in filas) - min(f["f1_defecto"] for f in filas)
        d_opt = max(f["f1_optimo"] for f in filas) - min(f["f1_optimo"] for f in filas)
        print(f"\nSeparacion mejor<->peor: {d_def:.4f} con los defaults -> "
              f"{d_opt:.4f} con el optimo de cada uno.")
        print("  " + ("SE COMPRIME: los knobs buenos suben el numero y bajan el "
                      "poder de distinguir entre modelos (es el hallazgo de julio)."
                      if d_opt < d_def else
                      "NO se comprime esta vez, al reves que en julio 2026."))
        opts = {(f["optimo"]["threshold"], f["optimo"]["stride"],
                 f["optimo"]["nms_radius"]) for f in filas}
        print(f"\n¿Mismo optimo para los {len(filas)}? "
              + ("SI -> es un ajuste GLOBAL y comparar con knobs fijos no sesga "
                 "el orden." if len(opts) == 1 else
                 f"NO ({len(opts)} distintos) -> los knobs son parte de la "
                 f"identidad de cada run, y eso es mas grave que el propio optimo."))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(
            {"cuando": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "split": args.split, "defaults": DEFAULTS,
             "rejilla": {"threshold": THRESHOLDS, "stride": STRIDES,
                         "nms_radius": NMS},
             "filas": filas}, indent=2), encoding="utf-8")
        print(f"\nJSON en {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
