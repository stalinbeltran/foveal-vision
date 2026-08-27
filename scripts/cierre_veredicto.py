#!/usr/bin/env python3
r"""El veredicto del plan de cierre, aplicando el criterio que se escribio antes.

`docs/plan-cierre-2026-08-26.md` fija COMO se lee cada bloque **antes de mirar**.
Esto solo lo aplica. No inventa criterio y no redefine ningun numero: usa
`sweep_trials`, `es_medida`, `aggregate_seeds`, `tie_delta` y `permutation_test`
del propio proyecto, igual que `estudio_informe.py`.

    .venv/bin/python scripts/cierre_veredicto.py            # todo
    .venv/bin/python scripts/cierre_veredicto.py --bloque A

LO QUE HACE QUE ESTO NO SEA `estudio_informe.py` REPETIDO
---------------------------------------------------------
Dos cosas que aquel no puede hacer porque mira UN recorrido:

1. **COMBINA `ov-r26` y `ov-sig26`.** El contraste decisivo del eje del solape es
   4 contra 2 con **10 semillas por punto**, y esas 10 viven en dos recorridos:
   las 1-5 en `ov-r26` y las 6-10 en `ov-sig26`. Combinar es legitimo aqui y NO
   en general: los dos recorridos comparten dataset, red base, receta y tope de
   epocas, y eso se COMPRUEBA abajo antes de juntar nada -- no se supone.

2. **Aplica la regla de ascenso del bloque B** (§2 del plan), que decide que
   tanteo merece 5 semillas. Un tanteo de 2 semillas no puede declarar ganador
   (el `p` minimo alcanzable es 0,333), asi que la unica lectura legitima es
   "¿hay senal suficiente para gastar mas?".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.metrics import permutation_test               # noqa: E402
from fv.sweeps.runner import es_medida, sweep_trials  # noqa: E402
from fv.sweeps.store import SweepStore                # noqa: E402
from fv.sweeps.winner import aggregate_seeds, hashable, tie_delta  # noqa: E402
from fv.training.registry import RunStore             # noqa: E402

# Lo que tienen que compartir dos recorridos para poder juntar sus semillas.
# `base_recipe_value` se compara SIN `seed`, que es justo lo que los distingue.
COMPARABLE = ("window_dataset", "base_network_value", "base_recipe", "budget",
              "objective")

BLOQUE_B = [("wd-t", "weight_decay", 0.0), ("opt-t", "optimizer", "adam"),
            ("lp-t", "lambda_pos", 1.0), ("sb-t", "smooth_l1_beta", 0.08),
            ("pat-t", "patience", 10), ("mrg-t", "merge", "concat"),
            ("pool-t", "pool_mode", "avg"), ("pad-t", "pad_mode", "edge"),
            ("ovb-t", "overlap_border_px", 0)]
BLOQUE_C = [("bp-r26", "border_px", 4), ("pl-f2-bs", "batch_size", 85),
            ("pl-f2-nl", "n_layers", 4)]

UMBRAL_AMPLITUD = 0.010      # §2 criterio 1: el doble del ruido tipico entre semillas
UMBRAL_MRG = 0.010           # §2 criterio 3: sum "no pierde mas de" esto


def n(v, d=4) -> str:
    return "-" if v is None else f"{v:.{d}f}".replace(".", ",")


def medidos(nombre, store, runs):
    """Los trials que SON medida, con su valor del eje y su semilla."""
    try:
        t = sweep_trials(nombre, store=store, run_store=runs)
    except Exception as e:                       # recorrido sin runs todavia
        return None, f"{type(e).__name__}: {e}"
    buenos = [x for x in t["trials"] if es_medida(x)]
    t["scored"] = buenos
    t["pendientes"] = sum(1 for x in t["trials"] if x["value"] is None)
    t["a_medias"] = sum(1 for x in t["trials"]
                        if x["value"] is not None and not es_medida(x))
    return t, None


def compatibles(a: dict, b: dict) -> list:
    """Que impide juntar las semillas de dos recorridos. Vacio = nada."""
    malos = []
    for c in COMPARABLE:
        if a.get(c) != b.get(c):
            malos.append(c)
    ra = {k: v for k, v in a["base_recipe_value"].items() if k != "seed"}
    rb = {k: v for k, v in b["base_recipe_value"].items() if k != "seed"}
    if ra != rb:
        malos.append("base_recipe_value(sin seed)")
    return malos


def tabla(scored, eje, direction, vigente):
    grupos = aggregate_seeds(scored, direction, "seconds_per_epoch")
    delta, fuente = tie_delta(grupos)
    vk = hashable(vigente)
    base = [t["value"] for t in scored if hashable(t["point"].get(eje)) == vk]
    filas = []
    for g in sorted(grupos, key=lambda g: -g["value"]):
        v = hashable(g["point"].get(eje))
        otros = [t["value"] for t in scored if hashable(t["point"].get(eje)) == v]
        pt = None if v == vk else permutation_test(otros, base)
        filas.append({"valor": g["point"].get(eje), "media": g["value"],
                      "n": g["n_seeds"], "sem": g.get("value_sem"),
                      "s_epoca": g.get("seconds_per_epoch"),
                      "p": (pt or {}).get("p"), "diff": (pt or {}).get("diff"),
                      "es_vigente": v == vk})
    return filas, delta, fuente


def pinta(titulo, filas, delta, fuente, nota=""):
    print(f"\n### {titulo}")
    if nota:
        print(f"{nota}")
    print(f"δ = {n(delta)} ({fuente})" if delta is not None else "δ = -")
    print(f"\n| valor | media f1 | semillas | sem | s/época | Δ vs vigente | p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for f in filas:
        m = " **(vigente)**" if f["es_vigente"] else ""
        print(f"| `{f['valor']}`{m} | {n(f['media'])} | {f['n']} | {n(f['sem'])} | "
              f"{n(f['s_epoca'],1)} | {'-' if f['diff'] is None else n(f['diff'])} | "
              f"{'-' if f['p'] is None else n(f['p'],3)} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bloque", action="append", choices=["A", "B", "C"])
    args = ap.parse_args()
    bloques = set(args.bloque or ["A", "B", "C"])
    store, runs = SweepStore(), RunStore()

    # ------------------------------------------------------------------ bloque A
    if "A" in bloques:
        print("\n## Bloque A — `overlap_fovea_px`")
        a, err_a = medidos("ov-r26", store, runs)
        b, err_b = medidos("ov-sig26", store, runs)
        if err_a:
            print(f"  ov-r26: {err_a}")
        else:
            sa, sb = store.spec("ov-r26"), store.spec("ov-sig26")
            malos = compatibles(sa, sb)
            print(f"\n`ov-r26`: {len(a['scored'])}/{len(a['trials'])} medidos "
                  f"({a['pendientes']} sin empezar, {a['a_medias']} a medias)")
            if not err_b:
                print(f"`ov-sig26`: {len(b['scored'])}/{len(b['trials'])} medidos "
                      f"({b['pendientes']} sin empezar, {b['a_medias']} a medias)")
            # A-1: el eje entero con las semillas 1-5 de ov-r26
            filas, delta, fuente = tabla(a["scored"], "overlap_fovea_px",
                                         a["direction"], 2)
            pinta("A-1 · el eje entero (`ov-r26`, semillas 1–5)", filas, delta, fuente)
            vals = [f["valor"] for f in filas]
            if vals:
                mejor = filas[0]["valor"]
                arriba = [f for f in filas if isinstance(f["valor"], (int, float))
                          and f["valor"] > 4]
                peor_arriba = [f for f in arriba if f["media"] < max(
                    x["media"] for x in filas if x["valor"] == mejor)]
                print(f"\n**A-1**: mejor punto = `{mejor}`. "
                      + ("El eje queda **CERRADO por los dos lados**: hay al menos un "
                         "punto por encima que es peor."
                         if peor_arriba and mejor != 7 else
                         "⚠ El maximo cae en **7**, que es la **pared legal** "
                         "(`overlap_fovea_range(16)` = [0..7]): el eje NO queda "
                         "cerrado por evidencia sino por la geometria."
                         if mejor == 7 else
                         "⚠ Sin punto peor por arriba entre los medidos."))
            # A-2: 10 contra 10, combinando
            if err_b:
                print(f"\n**A-2**: `ov-sig26` aun no tiene medidas ({err_b}).")
            elif malos:
                print(f"\n**A-2**: NO se combinan: difieren en {malos}.")
            else:
                junto = a["scored"] + b["scored"]
                filas2, delta2, fuente2 = tabla(
                    [t for t in junto
                     if hashable(t["point"].get("overlap_fovea_px")) in (2, 4)],
                    "overlap_fovea_px", a["direction"], 2)
                # El titulo dice las semillas QUE HAY, no las que se pidieron:
                # "10 contra 10" con 3 medidas es una afirmacion falsa, y este
                # informe se lee cuando el recorrido aun no ha terminado.
                ns = "+".join(str(f["n"]) for f in filas2)
                pinta(f"A-2 · el contraste decisivo, semillas {ns} "
                      "(`ov-r26` + `ov-sig26`)", filas2, delta2, fuente2,
                      nota="Combinados tras comprobar que comparten dataset, red "
                           "base, receta, tope de épocas y objetivo.")
                f4 = next((f for f in filas2 if f["valor"] == 4), None)
                if f4 and f4["p"] is not None:
                    mueve = f4["p"] < 0.05 and (f4["diff"] or 0) > (delta2 or 0)
                    print(f"\n**A-2**: `p` = {n(f4['p'],3)}, Δ = {n(f4['diff'])}, "
                          f"δ = {n(delta2)} → el vigente "
                          f"**{'PASA A 4' if mueve else 'SE QUEDA EN 2'}**.")
                    if not mueve:
                        print("   Con 10 contra 10 el suelo del test es 1,08·10⁻⁵, "
                              "así que un `p` alto aquí **es una medida, no una "
                              "limitación de semillas**.")

    # ------------------------------------------------------------------ bloque B
    if "B" in bloques:
        print("\n\n## Bloque B — tanteo (2 semillas). ACOTA, no declara ganador")
        print("\nRegla de ascenso a 5 semillas (§2 del plan): amplitud > "
              f"{UMBRAL_AMPLITUD}, **o** mejor punto por encima del vigente en más "
              "de 1 SE, **o** (`mrg-t`) `sum` no pierde más de "
              f"{UMBRAL_MRG}.")
        for nombre, eje, vig in BLOQUE_B:
            t, err = medidos(nombre, store, runs)
            if err or not t["scored"]:
                print(f"\n### `{nombre}` ({eje}) — sin medidas todavía"
                      f"{': ' + err if err else ''}")
                continue
            filas, delta, fuente = tabla(t["scored"], eje, t["direction"], vig)
            pinta(f"`{nombre}` · {eje} — {len(t['scored'])}/{len(t['trials'])} medidos",
                  filas, delta, fuente)
            medias = [f["media"] for f in filas]
            amplitud = max(medias) - min(medias)
            vf = next((f for f in filas if f["es_vigente"]), None)
            mejor = filas[0]
            gana_1se = (vf is not None and not mejor["es_vigente"]
                        and mejor["sem"] is not None
                        and mejor["media"] - vf["media"] > mejor["sem"])
            razones = []
            if amplitud > UMBRAL_AMPLITUD:
                razones.append(f"amplitud {n(amplitud)} > {UMBRAL_AMPLITUD}")
            if gana_1se:
                razones.append("el mejor supera al vigente en más de 1 SE")
            if nombre == "mrg-t":
                fs = next((f for f in filas if f["valor"] == "sum"), None)
                fc = next((f for f in filas if f["valor"] == "concat"), None)
                if fs and fc and (fc["media"] - fs["media"]) <= UMBRAL_MRG:
                    razones.append(f"`sum` pierde sólo {n(fc['media']-fs['media'])} "
                                   f"con 0,54× de parámetros")
            print(f"\n**{nombre}**: amplitud {n(amplitud)}. "
                  + (f"→ **ASCIENDE a 5 semillas** ({'; '.join(razones)})."
                     if razones else "→ **tanteado, sin señal**: se cierra aquí."))

    # ------------------------------------------------------------------ bloque C
    if "C" in bloques:
        print("\n\n## Bloque C — verificación de 5/10 semillas")
        for nombre, eje, vig in BLOQUE_C:
            t, err = medidos(nombre, store, runs)
            if err or not t["scored"]:
                print(f"\n### `{nombre}` ({eje}) — sin medidas todavía"
                      f"{': ' + err if err else ''}")
                continue
            filas, delta, fuente = tabla(t["scored"], eje, t["direction"], vig)
            pinta(f"`{nombre}` · {eje} — {len(t['scored'])}/{len(t['trials'])} medidos",
                  filas, delta, fuente)
            for f in filas:
                if f["es_vigente"] or f["p"] is None:
                    continue
                mueve = f["p"] < 0.05 and (f["diff"] or 0) > (delta or 0)
                if mueve:
                    print(f"  → `{f['valor']}` MUEVE el vigente "
                          f"(p = {n(f['p'],3)}, Δ = {n(f['diff'])} > δ = {n(delta)})")
            ceros = [x for x in t["scored"] if x["value"] == 0.0]
            if ceros:
                print(f"\n⚠ **{len(ceros)} run(s) con f1 = 0,0000** — colapso de "
                      f"entrenamiento. La media de un punto con un cero es el "
                      f"promedio de una moneda y **no debe citarse**: "
                      f"{[x['run'] for x in ceros]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
