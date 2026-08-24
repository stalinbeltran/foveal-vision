#!/usr/bin/env python3
"""Cuanto va a tardar y cuanto va a costar un recorrido, ANTES de alquilar nada.

Por que existe
--------------
Un estudio de este proyecto son horas de maquinas alquiladas por segundo. La
decision de lanzarlo -o de recortarlo- se toma bien solo si el numero esta antes;
despues ya es una factura. plan-40h.md §4 tenia una guarda de presupuesto por
esto mismo, pero costeaba al TOPE de epocas y sobreestimaba ~2x (§7.2).

Que hay dentro, y de donde sale cada numero
-------------------------------------------
El modelo es deliberadamente pequeno y **todos sus coeficientes estan medidos**;
ninguno es de memoria. Se declara la procedencia de cada uno porque un numero sin
procedencia se lee siempre como medido (CLAUDE.md, regla 2).

1. `S_EPOCA_REF` -- 40 s/epoca para L4 con batch 85 en una maquina de Vast con 8
   hilos. MEDIDO el 2026-08-23 en 12 maquinas (docs/plan-lr-alto.md §6.3 y §7.1):
   36,3 · 50,5 · 53,3 en la corrida A y 37,7 · 45,2 · 37,7 entre otras en la B.
   40 es la mediana redondeada. ⚠ El rango real es 36-53: un factor 1,47 entre
   maquinas del mismo catalogo, que es EXACTAMENTE lo que la criba de velocidad
   existe para recortar (scripts/sonda_velocidad.py).

2. `FACTOR_PROFUNDIDAD` -- coste relativo por `n_layers`. MEDIDO en
   `p40-confirm-n_layers` (2026-08-07, 20 runs en una sola maquina, asi que el
   COCIENTE es limpio aunque el absoluto sea de otra maquina): 59,8 · 82,2 ·
   105,7 · 128,8 s/epoca para L2 · L3 · L4 · L5.

3. `FACTOR_BATCH` -- coste relativo por `batch_size`, ajustado a
   `s_epoca = K + A/b` con K = 42,2 s y A = 2060 s·muestra. MEDIDO en
   `d1000-batch_size-1` (2026-08-05, L2): 40 -> 93,7 y 100 -> 62,8 s/epoca.
   ⚠ Se EXCLUYEN dos puntos de esa misma tanda, y hay que decir por que: batch 25
   dio 252,3 s/epoca y batch 85 dio 131,4, con sus vecinos en 93,7 y 62,8. Esa
   maquina entrenaba con otra cosa encima (el aviso de CLAUDE.md 2026-08-08: "el
   micro-benchmark de costo miente bajo carga"). Incluirlos deformaria el ajuste.
   Comprobacion del ajuste contra un punto que NO se uso: batch 55 predice 79,7 y
   midio 80,1.
   ⚠⚠ El ajuste es de una red L2 y se aplica a L4. En L4 el trabajo por paso es
   mayor, asi que la parte fija pesa menos y el efecto del batch deberia ser algo
   MAS fuerte de lo que dice este factor. O sea que para batch grande el modelo
   es CONSERVADOR (predice mas lento de lo que ira).

4. `EPOCAS_REF` -- 52 epocas hasta que `patience=10` corta, para L4 / lr 0,0014.
   MEDIDO en `lr-alto-L4` (2026-08-23): 36 · 54 · 66, media 52. Por profundidad,
   MEDIDO en `p40-confirm-n_layers`: 55,4 · 44,6 · 47,4 · 49,0.

5. `EXP_EPOCAS_BATCH` = 0,25 -- **ESTIMADO, NO MEDIDO**. Un batch mayor da menos
   actualizaciones por epoca, asi que a `lr` fijo deberia hacer falta mas epocas.
   Nadie lo ha medido en este proyecto con `patience` decidiendo: los estudios de
   `batch_size` anteriores toparon los 20 y no llegaron a converger. Va marcado
   como estimado en la salida y es la mayor fuente de error de esta prediccion.

6. `PEAJE_MIN` = 8,4 min por maquina (arranque + subida + instalacion de torch).
   MEDIDO: 10,9 min entre 3 maquinas y 31,3 entre 9 (plan-lr-alto §7.1) -> 3,5
   min/maquina de peaje puro; se suman 5 min de margen de arranque, que es lo que
   tardaron de media las instancias en dar SSH.

7. `USD_HORA` = 0,06 -- MEDIDO: 0,0449 $/h con 6 maquinas y 0,0596 con 12
   (plan-lr-alto §7.1). ⚠ El precio medio SUBE con el numero de maquinas
   distintas (+33 % al pasar de 6 a 12): pedir mas obliga a bajar en la lista
   ordenada por precio. Con 24 maquinas el modelo aplica un recargo declarado.

Lo que esta prediccion NO puede hacer
-------------------------------------
Acertar. Lo que puede es acotar: imprime un rango optimista/pesimista construido
con el mejor y el peor `s/epoca` MEDIDOS (36,3 y 53,3), no un numero solo. Si el
resultado real cae fuera de ese rango, el modelo esta mal y hay que arreglarlo
aqui -- no reinterpretarlo.

    python3 scripts/estudio_estimar.py --sweep bs-L4 --sweep nl-L4 --reparto seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# --- constantes medidas (procedencia en el docstring, una por una) -----------
S_EPOCA_REF = 40.0          # L4, batch 85, Vast 8 hilos (mediana medida)
S_EPOCA_MEJOR = 36.3        # la maquina mas rapida medida
S_EPOCA_PEOR = 53.3         # la mas lenta medida
FACTOR_PROFUNDIDAD = {1: 0.35, 2: 0.566, 3: 0.778, 4: 1.0, 5: 1.219, 6: 1.44}
BATCH_K, BATCH_A = 42.2, 2060.0
EPOCAS_REF = 52.0
EPOCAS_PROFUNDIDAD = {2: 55.4, 3: 44.6, 4: 47.4, 5: 49.0}
EXP_EPOCAS_BATCH = 0.25     # ESTIMADO, no medido
PEAJE_MIN = 8.4
USD_HORA = 0.06
RECARGO_POR_MAQUINA = 0.011  # +33 % al pasar de 6 a 12 maquinas -> ~1,1 %/maquina


def factor_batch(b: float) -> float:
    """Coste relativo de una epoca a batch `b` frente a batch 85."""
    return (BATCH_K + BATCH_A / b) / (BATCH_K + BATCH_A / 85.0)


def s_por_epoca(net: dict, batch_size: float, s_epoca_ref: float = S_EPOCA_REF) -> float:
    prof = FACTOR_PROFUNDIDAD.get(int(net.get("n_layers", 4)), 1.0)
    return s_epoca_ref * prof * factor_batch(float(batch_size))


def epocas(net: dict, batch_size: float) -> float:
    """Cuantas epocas hasta que `patience` corte. Medido por profundidad; el
    ajuste por batch es ESTIMADO (ver §5 del docstring)."""
    base = EPOCAS_PROFUNDIDAD.get(int(net.get("n_layers", 4)), EPOCAS_REF)
    return base * (float(batch_size) / 85.0) ** EXP_EPOCAS_BATCH


def estimar_sweep(nombre: str, reparto: str, pendientes: set | None = None) -> dict:
    """Coste por punto y por lote de un recorrido ya creado."""
    import dataclasses

    from fv.sweeps.spec import expand_points
    from fv.sweeps.store import SweepStore
    from fv.training.recipe import Recipe

    spec = SweepStore().spec(nombre)
    valid, _ = expand_points(spec, spec["base_network_value"])
    base_recipe = Recipe(**spec["base_recipe_value"])
    tope = int((spec.get("budget") or {}).get("epochs", 0) or 0)

    puntos = []
    for i, p in enumerate(valid):
        if pendientes is not None and i not in pendientes:
            continue
        receta = dataclasses.replace(base_recipe, **p["recipe_overrides"])
        ep = min(epocas(p["network"], receta.batch_size), tope or 1e9)
        fila = {"i": i, "overrides": p["overrides"], "epocas": round(ep, 1)}
        for etiqueta, ref in (("min", S_EPOCA_MEJOR), ("med", S_EPOCA_REF),
                              ("max", S_EPOCA_PEOR)):
            spe = s_por_epoca(p["network"], receta.batch_size, ref)
            fila[f"s_epoca_{etiqueta}"] = round(spe, 1)
            fila[f"min_{etiqueta}"] = round(ep * spe / 60.0, 1)
        puntos.append(fila)

    if reparto == "run":
        lotes = [[p] for p in puntos]
    else:
        semillas = sorted({p["overrides"].get("seed") for p in puntos})
        lotes = [[p for p in puntos if p["overrides"].get("seed") == s] for s in semillas]
        lotes = [l for l in lotes if l]
    return {"sweep": nombre, "puntos": puntos, "lotes": lotes,
            "tope_epocas": tope, "eje": [k for k in spec["space"] if k != "seed"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="append", required=True,
                    help="recorrido ya creado; repetible")
    ap.add_argument("--reparto", choices=("seed", "run"), default="seed")
    ap.add_argument("--json", default="", help="escribe el detalle aqui")
    args = ap.parse_args()

    todos = [estimar_sweep(s, args.reparto) for s in args.sweep]
    maquinas = sum(len(e["lotes"]) for e in todos)
    recargo = 1.0 + RECARGO_POR_MAQUINA * max(0, maquinas - 6)

    print(f"\nEstimacion para {len(todos)} recorrido(s), reparto '{args.reparto}'\n")
    print("Modelo: s/epoca medido en Vast (36,3 - 53,3, mediana 40) escalado por")
    print("profundidad (medido) y por batch (medido, ajuste K+A/b). Las epocas por")
    print("profundidad estan MEDIDAS; el ajuste de epocas por batch es ESTIMADO.\n")

    reloj_min = reloj_med = reloj_max = 0.0
    maq_min_total = maq_med_total = maq_max_total = 0.0
    for e in todos:
        eje = ", ".join(e["eje"]) or "?"
        print(f"### {e['sweep']}   eje {eje}   tope {e['tope_epocas']} epocas")
        print(f"{'punto':>28} {'epocas':>7} {'s/epoca':>18} {'minutos por run':>22}")
        for p in e["puntos"]:
            ov = json.dumps(p["overrides"], separators=(",", ":"))
            print(f"{ov:>28} {p['epocas']:7.0f} "
                  f"{p['s_epoca_min']:5.0f}-{p['s_epoca_max']:<5.0f}({p['s_epoca_med']:>4.0f}) "
                  f"{p['min_min']:8.0f}-{p['min_max']:<6.0f}({p['min_med']:>5.0f})")
        # el reloj de un recorrido es el del LOTE mas largo: los lotes van a la vez
        for etiqueta in ("min", "med", "max"):
            largos = [sum(p[f"min_{etiqueta}"] for p in lote) for lote in e["lotes"]]
            suma = sum(largos)
            if etiqueta == "min":
                reloj_min = max(reloj_min, max(largos)); maq_min_total += suma
            elif etiqueta == "med":
                reloj_med = max(reloj_med, max(largos)); maq_med_total += suma
            else:
                reloj_max = max(reloj_max, max(largos)); maq_max_total += suma
        n = len(e["lotes"])
        largos_med = [sum(p["min_med"] for p in lote) for lote in e["lotes"]]
        print(f"  -> {len(e['puntos'])} runs en {n} maquina(s); la mas cargada "
              f"~{max(largos_med):.0f} min de entrenamiento\n")

    peaje_total = maquinas * PEAJE_MIN
    print("=" * 72)
    print(f"  MAQUINAS: {maquinas}   (una por " +
          ("punto" if args.reparto == "run" else "recorrido x semilla") + ")")
    for etiqueta, reloj, maq in (("optimista", reloj_min, maq_min_total),
                                 ("central  ", reloj_med, maq_med_total),
                                 ("pesimista", reloj_max, maq_max_total)):
        reloj_t = reloj + PEAJE_MIN
        maq_t = maq + peaje_total
        usd = maq_t / 60.0 * USD_HORA * recargo
        print(f"  {etiqueta}: RELOJ {reloj_t / 60:5.1f} h  ·  "
              f"maquina-horas {maq_t / 60:6.1f}  ·  {usd:6.2f} $")
    print(f"  (peaje incluido: {PEAJE_MIN:.1f} min x {maquinas} maquinas = "
          f"{peaje_total:.0f} min. Recargo por catalogo: x{recargo:.2f})")
    print("=" * 72)
    print("  ⚠ El reloj es el de la maquina mas cargada, porque van a la vez.")
    print("  ⚠ La franja sale del s/epoca mejor y peor MEDIDOS en Vast. La criba")
    print("    de velocidad existe para acercar el resultado al extremo optimista.")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"reparto": args.reparto, "maquinas": maquinas, "recargo": recargo,
             "reloj_h": {"min": (reloj_min + PEAJE_MIN) / 60,
                         "med": (reloj_med + PEAJE_MIN) / 60,
                         "max": (reloj_max + PEAJE_MIN) / 60},
             "recorridos": todos}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"  Detalle en {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
