#!/usr/bin/env python3
"""El informe de un recorrido: la tabla, el veredicto y las reglas aplicadas.

No inventa criterio: aplica el que el documento del estudio dejo escrito ANTES
(R1..R4) y usa las funciones del proyecto -- `sweep_trials` para el ranking,
`aggregate_seeds` para la media por valor, `suggest_winner` para el ganador con
su banda, `permutation_test` para el contraste. Un numero definido dos veces es
un numero que acaba divergiendo.

    python3 scripts/estudio_informe.py --sweep lr-alto-L4 --vigente 0.0014

Sale por stdout en markdown para pegarlo en el documento del estudio, y deja el
mismo contenido en JSON al lado del recorrido.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.metrics import permutation_test            # noqa: E402
from fv.sweeps.runner import es_medida, sweep_trials  # noqa: E402
from fv.sweeps.store import SweepStore             # noqa: E402
from fv.sweeps.winner import (aggregate_seeds, hashable, suggest_winner,  # noqa: E402
                              tie_delta)
from fv.training.registry import RunStore          # noqa: E402


def num(v, d=4) -> str:
    return "-" if v is None else f"{v:.{d}f}".replace(".", ",")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--eje", default=None,
                    help="por defecto se DERIVA del spec (el unico eje que no es "
                         "'seed'); pasalo solo para releer un recorrido por otro campo")
    ap.add_argument("--vigente", default=None,
                    help="valor vigente del eje, para el contraste de R4. Se lee "
                         "como JSON, asi que vale para los ejes que NO son "
                         "numeros: 4 / 0.0014 / '\"val_loss\"' / '[16,16,16,16]'")
    args = ap.parse_args()

    # El eje no siempre es un numero: `monitor` y `scheduler` son cadenas y
    # `channels` es una LISTA. Se lee como JSON y se cae a la cadena tal cual, que
    # es lo que deja escribir --vigente val_loss sin comillas raras en Telegram.
    if args.vigente is not None:
        try:
            args.vigente = json.loads(args.vigente)
        except json.JSONDecodeError:
            pass                                  # una cadena suelta: vale asi

    store, runs = SweepStore(), RunStore()
    spec = store.spec(args.sweep)
    # El eje se DERIVA del espacio del recorrido. Tenia default "lr" cableado, y
    # eso no fallaba: producia una tabla entera con el eje a `None` y un ganador
    # llamado `None` -- creible y falsa, que es peor que un error. Es el mismo
    # dato en dos sitios (el spec lo sabe; la bandera lo repetia).
    if args.eje is None:
        ejes = [k for k in (spec.get("space") or {}) if k != "seed"]
        if len(ejes) != 1:
            print(f"No puedo derivar el eje de '{args.sweep}': su espacio declara "
                  f"{sorted(ejes) or 'ninguno'}.\n"
                  f"  -> pasalo con --eje <campo>")
            return 1
        args.eje = ejes[0]
    tabla = sweep_trials(args.sweep, store=store, run_store=runs)
    # Un run a medias NO es una medida: trae valor (lo mejor hasta donde llego)
    # pero no ha convergido, asi que hunde su punto. Ver ESTADOS_MEDIBLES.
    scored = [t for t in tabla["trials"] if es_medida(t)]
    pendientes = [t for t in tabla["trials"] if t["value"] is None]
    a_medias = [t for t in tabla["trials"]
                if t["value"] is not None and not es_medida(t)]
    if not scored:
        print(f"El recorrido '{args.sweep}' no tiene ningun punto medido todavia.")
        return 1

    tope = int((spec.get("budget") or {}).get("epochs", 0) or 0)

    # --- R1: validez. Un punto que para POR EL TOPE mide presupuesto, no calidad.
    detalle, truncados = [], []
    for t in scored:
        s = {}
        p = runs.path(t["run"]) / "summary.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
        fila = {"run": t["run"], "punto": t["point"], "valor": t["value"],
                "epocas": s.get("epochs_run"), "mejor_epoca": s.get("best_epoch"),
                "paro_patience": s.get("stopped_early"),
                "s_por_epoca": s.get("seconds_per_epoch")}
        if s.get("stopped_early") is False or (
                tope and (s.get("epochs_run") or 0) >= tope):
            truncados.append(fila)
        detalle.append(fila)

    grupos = aggregate_seeds(scored, tabla["direction"], "seconds_per_epoch")
    delta, fuente_delta = tie_delta(grupos)
    ganador = suggest_winner(args.sweep, store=store, run_store=runs)

    # --- R4: cada valor contra el vigente, con permutacion exacta.
    # clave HASHABLE: `channels` es una lista y una lista no puede ser clave de
    # dict. Se usa la misma normalizacion que aggregate_seeds (winner.hashable),
    # no otra, o dos sitios agruparian distinto.
    por_valor = {hashable(g["point"].get(args.eje)): g for g in grupos}
    vigente_k = hashable(args.vigente) if args.vigente is not None else None
    contrastes = []
    if vigente_k is not None and vigente_k in por_valor:
        base = [t["value"] for t in scored
                if hashable(t["point"].get(args.eje)) == vigente_k]
        for v, g in por_valor.items():
            if v == vigente_k:
                continue
            otros = [t["value"] for t in scored
                     if hashable(t["point"].get(args.eje)) == v]
            pt = permutation_test(otros, base)
            if pt:
                contrastes.append({"valor": v, **pt})

    linea = "|---:|" + "---:|" * 6
    out = [
        f"## Resultado de `{args.sweep}`",
        "",
        f"{len(scored)}/{len(tabla['trials'])} puntos medidos"
        + (f" ({len(pendientes)} sin empezar)" if pendientes else "")
        + (f", **{len(a_medias)} A MEDIAS y EXCLUIDOS** (la maquina murio antes "
           f"de que convergieran: tienen valor pero no es una medida)"
           if a_medias else "")
        + f", dataset `{spec['window_dataset']}`, objetivo `{tabla['objective']}` "
          f"del checkpoint.",
        "",
        f"| {args.eje} | {tabla['objective']} (media) | sem | min | max | épocas | s/época |",
        linea,
    ]
    for g in grupos:
        v = g["point"].get(args.eje)
        eps = [d["epocas"] for d in detalle
               if hashable(d["punto"].get(args.eje)) == hashable(v)]
        out.append(
            f"| {str(v).replace('.', ',')} | **{num(g['value'])}** | "
            f"{num(g.get('value_sem'))} | {num(g['value_min'])} | "
            f"{num(g['value_max'])} | "
            f"{' · '.join(str(e) for e in sorted(x for x in eps if x))} | "
            f"{num(g.get('seconds_per_epoch'), 1)} |")

    out += ["", "### Las reglas, aplicadas", ""]
    if truncados:
        out.append(f"**R1 ❌ — {len(truncados)} runs pararon POR EL TOPE de {tope} "
                   f"épocas**, no por `patience`: miden presupuesto, no calidad. "
                   f"No se declara ganador sobre ellos: "
                   + ", ".join(t["run"] for t in truncados[:4]))
    else:
        eps = [d["epocas"] for d in detalle if d["epocas"]]
        out.append(f"**R1 ✅ — el recorrido es válido.** Los {len(scored)} runs "
                   f"pararon por `patience` (`stopped_early` en {len(scored)}/"
                   f"{len(scored)}), entre {min(eps)} y {max(eps)} épocas, "
                   f"ninguno cerca del tope de {tope}.")
    mejor = grupos[0]
    out.append("")
    out.append(f"**R2 — el ganador por media.** `{args.eje} = "
               f"{str(mejor['point'].get(args.eje)).replace('.', ',')}` con "
               f"{num(mejor['value'])} de media sobre {mejor['n_seeds']} semillas. "
               f"δ = {num(delta)} ({fuente_delta}). "
               f"`suggest_winner` sugiere "
               f"`{str(ganador['suggested']['point'].get(args.eje)).replace('.', ',')}` "
               f"(la más barata dentro de δ).")
    if contrastes:
        out += ["", "**R4 — contra el vigente "
                f"`{str(args.vigente).replace('.', ',')}`, permutación exacta:**", "",
                f"| {args.eje} | diferencia | p | arreglos |", "|---:|---:|---:|---:|"]
        for c in contrastes:
            out.append(f"| {str(c['valor']).replace('.', ',')} | "
                       f"{num(c['diff'])} | {num(c['p'], 3)} | {c['arrangements']} |")
        pmin = min(c["p"] for c in contrastes)
        arr = contrastes[0]["arrangements"]
        alcanzable = 2 / arr
        out.append("")
        if alcanzable > 0.05:
            # El aviso SOLO cuando es cierto: con pocas semillas el contraste no
            # puede bajar del 5 % ni siendo la diferencia enorme, y eso hay que
            # decirlo ANTES de que alguien lea "p = 0,10" como "casi significativo".
            out.append(f"⚠ Con {mejor['n_seeds']} semillas por punto solo hay **{arr} "
                       f"arreglos**, asi que el p mas pequeno ALCANZABLE es "
                       f"{num(alcanzable, 3)}: con este tamano **R4 no puede declarar "
                       f"significacion al 5 %** aunque la diferencia sea grande, y el "
                       f"vigente se queda pase lo que pase. El p mas bajo observado es "
                       f"{num(pmin, 3)}.")
        else:
            hay = [c for c in contrastes if c["p"] <= 0.05]
            out.append(f"Con {mejor['n_seeds']} semillas hay {arr} arreglos "
                       f"(p minimo alcanzable {num(alcanzable, 3)}), asi que el "
                       f"contraste SI puede declarar significacion al 5 %. "
                       + (f"La declaran {len(hay)} valor(es): "
                          + ", ".join(str(c['valor']) for c in hay) + "."
                          if hay else
                          f"Ninguno la alcanza (p mas bajo: {num(pmin, 3)}), asi que "
                          f"el vigente se queda."))

    texto = "\n".join(out)
    print(texto)
    destino = ROOT / "sweeps" / args.sweep / "informe.json"
    destino.write_text(json.dumps(
        {"recorrido": args.sweep, "eje": args.eje, "objetivo": tabla["objective"],
         "dataset": spec["window_dataset"], "delta": delta, "delta_fuente": fuente_delta,
         "grupos": grupos, "detalle": detalle, "truncados": truncados,
         "contrastes": contrastes, "pendientes": [t["run"] for t in pendientes],
         "sugerido": ganador["suggested"]["point"]},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n<!-- tambien en {destino.relative_to(ROOT)} -->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
