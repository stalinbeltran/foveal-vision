#!/usr/bin/env python3
"""Compara dos corridas del MISMO recorrido: lo que costo cada reparto, y si
dieron la misma respuesta.

Son dos preguntas distintas y las dos hacen falta:

1. **Coste y reloj.** Repartir mas fino gana reloj pero paga el PEAJE (arranque
   + subida + instalacion) una vez POR MAQUINA. El trabajo -las epocas- es el
   mismo se reparta como se reparta. Comparar dos repartos es comparar esas dos
   columnas, no el precio total a secas.

2. **¿La misma respuesta?** Esto es lo que de verdad decide si el reparto fino
   se puede usar. `--reparto run` da una maquina distinta a cada run, asi que el
   efecto de la maquina deja de cancelarse y pasa a ser ruido repartido al azar.
   Si las dos corridas coinciden en el ganador y las medias caen dentro de sus
   bandas, ese ruido es pequeno para este trabajo. Si no coinciden, el reparto
   fino esta comprando tiempo con precision, y hay que decirlo.

⚠ Lo que esta comparacion NO es: un experimento controlado. Las dos corridas
usaron maquinas distintas, en momentos distintos y a precios distintos. Da una
IDEA del orden de magnitud, no una medida de la diferencia entre repartos.

    python3 scripts/estudio_comparar.py lr-alto-L4 lr-alto-L4-b --eje lr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv import datarepo
from fv.sweeps.runner import sweep_trials         # noqa: E402
from fv.sweeps.store import SweepStore            # noqa: E402
from fv.sweeps.winner import aggregate_seeds      # noqa: E402
from fv.training.registry import RunStore         # noqa: E402


def coma(v, d=4) -> str:
    return "-" if v is None else f"{v:.{d}f}".replace(".", ",")


def leer_flota(sweep: str) -> dict:
    """Normaliza los dos formatos de `flota.json`.

    El primero se escribio antes de que existiera el desglose de peaje, asi que
    no lo trae; ahi el peaje se DEDUCE como (vivida - entrenamiento), que ademas
    incluye la recogida de los runs. Se marca como deducido en vez de mezclarlo
    con el medido: un numero deducido y uno medido no son el mismo numero.
    """
    d = json.loads((datarepo.resolve("sweeps", sweep) / "flota.json").read_text(encoding="utf-8"))
    maquinas = d.get("lotes") or d.get("semillas") or []
    vividas = sum(float(m.get("segundos_vivida") or 0) for m in maquinas)
    trabajo = sum(float(m.get("entrenamiento_s") or 0) for m in maquinas)
    medido = all(m.get("peaje_s") is not None for m in maquinas) and bool(maquinas)
    peaje = (sum(float(m["peaje_s"]) for m in maquinas) if medido
             else vividas - trabajo)
    return {
        "sweep": sweep,
        "reparto": d.get("reparto", "seed"),
        "maquinas": len(maquinas),
        "reloj_min": d.get("reloj_min"),
        "usd": d.get("usd"),
        "maquina_min": vividas / 60,
        "trabajo_min": trabajo / 60,
        "peaje_min": peaje / 60,
        "peaje_medido": medido,
        "epocas": sum(int(m.get("epocas") or 0) for m in maquinas),
        "usd_hora_medio": (sum(float(m.get("usd_hora") or 0) for m in maquinas)
                           / len(maquinas)) if maquinas else 0.0,
        "detalle": maquinas,
    }


def grupos_de(sweep: str, eje: str) -> dict:
    store, runs = SweepStore(), RunStore()
    tabla = sweep_trials(sweep, store=store, run_store=runs)
    scored = [t for t in tabla["trials"] if t["value"] is not None]
    gs = aggregate_seeds(scored, tabla["direction"], "seconds_per_epoch")
    return {g["point"].get(eje): g for g in gs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--eje", default="lr")
    args = ap.parse_args()

    A, B = leer_flota(args.a), leer_flota(args.b)
    out = [f"## `{args.a}` (reparto {A['reparto']}) contra "
           f"`{args.b}` (reparto {B['reparto']})", ""]

    # --- 1. coste y reloj
    out += ["### Lo que costo cada reparto", "",
            f"| | `{A['reparto']}` ({A['maquinas']} maq.) | "
            f"`{B['reparto']}` ({B['maquinas']} maq.) | diferencia |",
            "|---|---:|---:|---:|"]

    def fila(nombre, ka, kb, d=1, sufijo="", inv=False):
        va, vb = A[ka], B[kb]
        if not va:
            return
        rel = (vb - va) / va * 100
        flecha = "" if abs(rel) < 0.5 else (" ↓" if (rel < 0) != inv else " ↑")
        out.append(f"| {nombre} | {coma(va, d)}{sufijo} | {coma(vb, d)}{sufijo} | "
                   f"{rel:+.0f} %{flecha} |")

    fila("**reloj** (lo que esperas)", "reloj_min", "reloj_min", 1, " min")
    fila("**coste**", "usd", "usd", 4, " $")
    fila("maquina-minutos (lo que se factura)", "maquina_min", "maquina_min", 1, " min")
    fila("  · de eso, trabajo (epocas)", "trabajo_min", "trabajo_min", 1, " min")
    fila("  · de eso, PEAJE (arranque+subida+instalacion)", "peaje_min", "peaje_min",
         1, " min")
    fila("epocas entrenadas", "epocas", "epocas", 0, "")
    fila("precio medio de maquina", "usd_hora_medio", "usd_hora_medio", 4, " $/h")
    out += ["",
            f"El peaje de `{args.a}` esta "
            + ("MEDIDO" if A["peaje_medido"] else
               "**deducido** (vivida - entrenamiento; incluye tambien la recogida "
               "de los runs, asi que sobreestima un poco)")
            + f"; el de `{args.b}`, "
            + ("MEDIDO." if B["peaje_medido"] else "deducido."),
            ""]
    if A["maquinas"] and B["maquinas"]:
        pa, pb = A["peaje_min"] / A["maquinas"], B["peaje_min"] / B["maquinas"]
        out.append(f"Peaje por maquina: {coma(pa, 1)} min contra {coma(pb, 1)} min. "
                   f"Es casi constante, y ese es el punto: se paga UNA VEZ POR "
                   f"MAQUINA, asi que multiplicar maquinas por "
                   f"{B['maquinas'] / A['maquinas']:.0f} multiplica el peaje por "
                   f"{B['peaje_min'] / A['peaje_min']:.1f} mientras el trabajo se "
                   f"queda igual.")

    # --- 2. ¿la misma respuesta?
    ga, gb = grupos_de(args.a, args.eje), grupos_de(args.b, args.eje)
    out += ["", "### ¿Dieron la misma respuesta?", "",
            f"| {args.eje} | `{args.a}` | sem | `{args.b}` | sem | diferencia | "
            f"¿dentro de las bandas? |", "|---:|---:|---:|---:|---:|---:|---|"]
    comunes = [v for v in ga if v in gb]
    orden = sorted(comunes, key=lambda v: -ga[v]["value"])
    for v in orden:
        a, b = ga[v], gb[v]
        dif = b["value"] - a["value"]
        # "dentro de las bandas" = la diferencia cabe en la suma de los dos
        # errores estandar. Es un criterio flojo a proposito: con 3 semillas
        # pedir mas seria pedirle a 3 replicas lo que no pueden dar.
        banda = (a.get("value_sem") or 0) + (b.get("value_sem") or 0)
        cabe = abs(dif) <= banda if banda else None
        out.append(f"| {str(v).replace('.', ',')} | {coma(a['value'])} | "
                   f"{coma(a.get('value_sem'))} | {coma(b['value'])} | "
                   f"{coma(b.get('value_sem'))} | {dif:+.4f} | "
                   + ("sí" if cabe else "**NO**" if cabe is False else "?") + " |")
    gan_a = max(ga.items(), key=lambda kv: kv[1]["value"])[0]
    gan_b = max(gb.items(), key=lambda kv: kv[1]["value"])[0]
    out += ["", f"**Ganador: `{gan_a}` contra `{gan_b}` — "
                + ("EL MISMO." if gan_a == gan_b else "**DISTINTO**.") + "**"]
    ord_a = [str(v) for v in sorted(ga, key=lambda v: -ga[v]["value"])]
    ord_b = [str(v) for v in sorted(gb, key=lambda v: -gb[v]["value"])]
    out.append(f"Orden completo: {' > '.join(ord_a)}  contra  {' > '.join(ord_b)}"
               + ("  (igual)" if ord_a == ord_b else "  — **el orden de los "
                  "perdedores NO se repite**"))

    texto = "\n".join(out)
    print(texto)
    destino = datarepo.resolve("sweeps", args.b) / "comparacion.json"
    destino.write_text(json.dumps(
        {"a": {k: v for k, v in A.items() if k != "detalle"},
         "b": {k: v for k, v in B.items() if k != "detalle"},
         "ganador_a": gan_a, "ganador_b": gan_b,
         "orden_a": ord_a, "orden_b": ord_b,
         "grupos_a": {str(k): v for k, v in ga.items()},
         "grupos_b": {str(k): v for k, v in gb.items()}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # el destino vive en el REPO DE DATOS, fuera de ROOT: `relative_to` reventaria
    # y tiraria una comparacion ya escrita (misma leccion que estudio_stride_informe.py)
    try:
        donde = destino.relative_to(ROOT)
    except ValueError:
        donde = destino
    print(f"\n<!-- tambien en {donde} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
