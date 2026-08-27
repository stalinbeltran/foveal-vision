#!/usr/bin/env python3
r"""El veredicto del barrido de stride de extraccion: la tabla y R1..R4.

No inventa criterio: aplica el que `docs/plan-stride-2026-08-27.md` 3 dejo
escrito ANTES de mirar, y usa las funciones del proyecto -- `sweep_trials` para
leer cada brazo, `aggregate_seeds` para la media por valor con su banda,
`tie_delta` para delta, `select_winner` para la frontera y `permutation_test`
para el contraste. Un numero definido dos veces es un numero que acaba
divergiendo (misma razon por la que existe `estudio_informe.py`).

    .venv/bin/python scripts/estudio_stride_informe.py --estudio stride-2026-08-27

Por que hace falta uno propio y no vale `estudio_informe.py`
------------------------------------------------------------
El eje de este estudio NO vive en `space`: vive en el DATASET, y por eso hay un
recorrido por valor (docs/barrido-stride.md 1). `estudio_informe.py --eje` lee
un recorrido y agrupa por un campo del punto; aqui hay que juntar N recorridos y
agrupar por el valor que cada uno representa.

La costura es exacta y no se reimplementa nada: `aggregate_seeds` agrupa por el
punto SIN `seed`, asi que dandole `point = {"stride": s, "seed": k}` agrupa por
stride sin tocar una linea de `winner.py`.

LO QUE HAY QUE RESPETAR SI SE TOCA ESTO
---------------------------------------
1. **Se NIEGA si los brazos no comparten rejilla de evaluacion.** Seria la
   trampa de barrido-stride.md 2.1 disfrazada de tabla: cada brazo examinado de
   otra cosa, y el f1 comparado como si fuera el mismo numero.
2. **Se NIEGA si los brazos no comparten presupuesto** (`windows_per_epoch`).
   Entonces la tabla mide el presupuesto, no la densidad.
3. **R4 es un CONTROL, no un resultado.** Si `seconds_per_epoch` se desvia entre
   brazos, el igualado de presupuesto fallo y no se declara nada. Un control que
   se reporta como un resultado mas es un control que nadie mira.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.metrics import VAL_METRICS, permutation_test    # noqa: E402
from fv.sweeps.runner import es_medida, sweep_trials    # noqa: E402
from fv.sweeps.store import SweepStore                  # noqa: E402
from fv.sweeps.winner import (aggregate_seeds, select_winner,  # noqa: E402
                              tie_delta, tie_reason)
from fv.training.registry import RunStore               # noqa: E402

ESTUDIO = "stride-2026-08-27"
DESVIO_MAX_R4 = 0.15      # plan 3, R4: 15 % sobre la mediana de s/epoca


def num(v, d=4) -> str:
    return "-" if v is None else f"{v:.{d}f}".replace(".", ",")


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\nERROR: {msg}\n", file=sys.stderr, flush=True)
    raise SystemExit(2)


def brazos_del_estudio(store: SweepStore, estudio: str, nombres: list) -> list:
    """(nombre, spec) por brazo, ordenados por el valor del stride."""
    if nombres:
        faltan = [n for n in nombres if not store.exists(n)]
        if faltan:
            die(f"no existen estos recorridos: {', '.join(faltan)}")
        candidatos = list(nombres)
    else:
        candidatos = store.used_by_study(estudio)
    if not candidatos:
        die(f"no hay ningun recorrido del estudio '{estudio}'.\n"
            f"  Crealos con: .venv/bin/python scripts/estudio_stride.py")

    brazos = []
    for n in candidatos:
        spec = store.spec(n)
        eje = spec.get("eje_dataset") or {}
        if eje.get("campo") != "stride":
            die(f"'{n}' no declara `eje_dataset.campo = stride`.\n"
                f"  Un recorrido sin esa etiqueta no dice que valor representa, y\n"
                f"  meterlo en la tabla seria inventarselo.")
        brazos.append((n, spec, int(eje["valor"])))
    brazos.sort(key=lambda b: b[2])

    # Las dos cosas que harian incomparables los brazos. Se falla, no se avisa.
    rejillas = {b[1]["eje_dataset"].get("eval_stride") for b in brazos}
    if len(rejillas) > 1:
        die(f"los brazos NO comparten rejilla de evaluacion: {sorted(rejillas)}.\n"
            f"  Cada uno se ha examinado de un conjunto distinto, asi que sus f1\n"
            f"  no son el mismo numero. Ver docs/barrido-stride.md 2.1.")
    presupuestos = {(b[1].get("base_recipe_value") or {}).get("windows_per_epoch", 0)
                    for b in brazos}
    if len(presupuestos) > 1:
        die(f"los brazos NO comparten presupuesto de ventanas por epoca: "
            f"{sorted(presupuestos)}.\n"
            f"  La tabla mediria el presupuesto y no la densidad de la rejilla.\n"
            f"  Ver docs/barrido-stride.md 2.2.")
    return brazos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--estudio", default=ESTUDIO)
    ap.add_argument("--sweep", action="append", default=[],
                    help="brazos explicitos (por defecto, los del estudio)")
    ap.add_argument("--objetivo", default=None,
                    help="re-lee la tabla por otra metrica de val (no toca los specs)")
    ap.add_argument("--json", default=None, help="donde dejar el JSON")
    args = ap.parse_args()

    store, run_store = SweepStore(), RunStore()
    brazos = brazos_del_estudio(store, args.estudio, args.sweep)
    objetivo = args.objetivo or brazos[0][1].get("objective", "f1")
    if objetivo not in VAL_METRICS:
        die(f"objetivo '{objetivo}' no existe; usa uno de {sorted(VAL_METRICS)}")
    direccion = VAL_METRICS[objetivo]

    scored, pendientes, por_brazo = [], [], []
    for nombre, spec, stride in brazos:
        tabla = sweep_trials(nombre, store, run_store, objective=objetivo)
        medidas = [t for t in tabla["trials"] if es_medida(t)]
        faltan = [t for t in tabla["trials"] if not es_medida(t)]
        pendientes += [(nombre, t["run"], t.get("status")) for t in faltan]
        por_brazo.append({"sweep": nombre, "stride": stride,
                          "dataset": spec["window_dataset"],
                          "medidas": len(medidas),
                          "puntos": len(tabla["trials"])})
        for t in medidas:
            fila = dict(t)
            # el punto que ve `aggregate_seeds`: el eje es el stride, la replica
            # sigue siendo la semilla
            fila["point"] = {"stride": stride,
                             "seed": t["point"].get("seed", spec.get("seed", 1))}
            scored.append(fila)

    if not scored:
        die("ningun brazo tiene medidas todavia. Mira como va con:\n"
            "  .venv/bin/python scripts/estudio_progreso.py "
            + " ".join(f"--sweep {b[0]}" for b in brazos))

    scored.sort(key=lambda r: r["value"], reverse=(direccion == "max"))
    grupos = aggregate_seeds(scored, direccion, "seconds_per_epoch")
    delta, razon_delta = tie_delta(grupos)
    mejor, sugerido, frontera = select_winner(grupos, direccion, delta,
                                              "seconds_per_epoch")

    por_stride = {g["point"]["stride"]: g for g in grupos}
    strides = sorted(por_stride)
    un_solo_brazo = len(strides) < 2

    # ---- R1: saturacion = el stride MAS GRANDE dentro de delta del mejor
    en_frontera = sorted(g["point"]["stride"] for g in frontera)
    saturacion = max(en_frontera)
    cerrado_por_arriba = saturacion != max(strides)

    # ---- R2: mejor brazo contra el mas disperso
    mas_disperso = max(strides)
    a = [t["value"] for t in scored if t["point"]["stride"] == mejor["point"]["stride"]]
    b = [t["value"] for t in scored if t["point"]["stride"] == mas_disperso]
    contraste = permutation_test(a, b) if mejor["point"]["stride"] != mas_disperso else None

    # ---- R3: monotonia al bajar el stride
    orden_denso = sorted(strides, reverse=True)     # de disperso a denso
    rupturas = []
    for x, y in zip(orden_denso, orden_denso[1:]):
        v0, v1 = por_stride[x]["value"], por_stride[y]["value"]
        peor = (v1 < v0 - delta) if direccion == "max" else (v1 > v0 + delta)
        if peor:
            rupturas.append({"de": x, "a": y, "valor_de": v0, "valor_a": v1})

    # ---- R4: el CONTROL de presupuesto
    #
    # MIDE LOS PASOS, NO LOS SEGUNDOS. La primera version comparaba `s/epoca`
    # entre brazos con un tope del 15 %, y eso es el proxy equivocado: cada run
    # corre en una maquina alquilada distinta, y la dispersion entre maquinas del
    # catalogo es mayor que cualquier efecto del eje.
    #
    # MEDIDO el 2026-08-27 en el estudio completo (25 runs): DENTRO de un mismo
    # brazo -- misma config, cinco maquinas -- el cociente max/min de s/epoca
    # llego a **2,50**, mientras que ENTRE brazos las medias solo se separaban
    # 1,53. O sea que el control marcaba FALLO por ruido de maquina, que es
    # justo lo que no mide. Un control que da falsa alarma se acaba ignorando, y
    # entonces no hay control.
    #
    # Lo que R4 quiere saber es si los brazos hicieron el MISMO trabajo por
    # epoca, y eso se lee directo del `config.json` que cada maquina devolvio:
    # pasos = ceil(windows_per_epoch / batch_size). Es exacto y no tiene ruido.
    pasos_por_run, sin_config = {}, []
    for t_ in scored:
        try:
            cfg = run_store.config(t_["run"]).get("recipe", {})
            w = int(cfg.get("windows_per_epoch", 0) or 0)
            b = int(cfg.get("batch_size", 0) or 0)
            pasos_por_run[t_["run"]] = -(-w // b) if (w and b) else None
        except Exception:                                      # noqa: BLE001
            sin_config.append(t_["run"])
    distintos = sorted({v for v in pasos_por_run.values() if v})
    con_pool_entero = [r for r, v in pasos_por_run.items() if v is None]

    segundos = {s: por_stride[s].get("seconds_per_epoch") for s in strides}
    medidos = [v for v in segundos.values() if v]
    mediana = statistics.median(medidos) if medidos else None
    # la dispersion DENTRO de un brazo es la vara de medir del ruido de maquina
    dentro = {}
    for s in strides:
        v = [t_["seconds_per_epoch"] for t_ in scored
             if t_["point"]["stride"] == s and t_.get("seconds_per_epoch")]
        if len(v) > 1:
            dentro[s] = max(v) / min(v)
    desviados = []

    # ------------------------------------------------------------------ salida
    p = print
    p(f"# Barrido de stride de extraccion — {args.estudio}")
    p("")
    p(f"Objetivo **{objetivo}** ({'mayor' if direccion == 'max' else 'menor'} es "
      f"mejor) · rejilla de evaluacion FIJA "
      f"(`eval_stride` = {brazos[0][1]['eje_dataset'].get('eval_stride')}) · "
      f"presupuesto igualado "
      f"(`windows_per_epoch` = "
      f"{(brazos[0][1].get('base_recipe_value') or {}).get('windows_per_epoch', 0)})")
    p("")
    p("| stride | " + objetivo + " (media) | banda min–max | SEM | semillas | s/epoca | dataset |")
    p("|---|---|---|---|---|---|---|")
    for s in strides:
        g = por_stride[s]
        arm = next(x for x in por_brazo if x["stride"] == s)
        p(f"| {s} | {num(g['value'])} | {num(g['value_min'])}–{num(g['value_max'])} "
          f"| {num(g.get('value_sem'))} | {g['n_seeds']} "
          f"| {num(g.get('seconds_per_epoch'), 1)} | `{arm['dataset']}` |")
    p("")
    if pendientes:
        p(f"⚠ **{len(pendientes)} punto(s) sin medida** — la tabla es parcial:")
        for sw, run, st in pendientes[:10]:
            p(f"  - `{run}` ({sw}): {st or 'sin status'}")
        if len(pendientes) > 10:
            p(f"  - … (+{len(pendientes) - 10})")
        p("")

    p(f"**δ = {num(delta)}** — {razon_delta}")
    # Sin replicas no hay banda de ruido, y entonces CUALQUIER diferencia -por
    # pequena que sea- pasa el filtro de delta: R1 corona un ganador y R3 marca
    # ruptura, los dos por ruido. Se dice una vez, arriba, en vez de dejar que el
    # lector lo deduzca de un "delta = 0,0000" que parece precision y es ausencia.
    sin_banda = delta == 0 and all(g["n_seeds"] < 2 for g in grupos)
    if sin_banda:
        p("")
        p("⚠ **Ningún brazo tiene réplicas, así que δ = 0 y no hay banda de ruido "
          "medida.** Con eso, cualquier diferencia «supera» δ: R1 corona un "
          "ganador y R3 marca rupturas **por construcción**. Nada de lo de abajo "
          "declara nada; para eso hacen falta las semillas del plan §2.3.")
    p("")
    # Con un solo brazo medido NO hay eje que leer, y hay que decirlo asi.
    # Antes se aplicaban R1 y R3 igual, y como min(strides) == max(strides) se
    # imprimian los DOS avisos a la vez: «la densidad no compra nada» y «el eje no
    # queda cerrado por arriba», que se contradicen. Un informe que dice dos cosas
    # opuestas es peor que uno que calla. Sale al leer un estudio a medias, que es
    # el caso normal mientras la flota corre.
    if un_solo_brazo:
        p(f"**R1 · Saturación.** ⚠ **No evaluable**: sólo hay medidas de un brazo "
          f"(stride {strides[0]}, {num(por_stride[strides[0]]['value'])}). La "
          f"saturación es una comparación y necesita al menos dos brazos.")
    else:
        p(f"**R1 · Saturación.** El mejor brazo es **stride "
          f"{mejor['point']['stride']}** ({num(mejor['value'])}). Dentro de δ quedan "
          f"{en_frontera}, así que el **punto de saturación es stride "
          f"{saturacion}**: es el dato más barato que no pierde calidad.")
        if saturacion == max(strides):
            p("")
            p(f"  ⚠ La saturación cae en el extremo más disperso del eje "
              f"({saturacion}): la densidad no compra nada en este rango.")
        elif saturacion == min(strides):
            p("")
            p(f"  ⚠ La saturación cae en el extremo más denso ({saturacion}): el eje "
              f"**no queda cerrado por arriba**. La frase correcta es «gana el "
              f"extremo», no «satura en {saturacion}».")
    p("")
    if contraste is None:
        p(f"**R2 · Significación.** El mejor brazo ES el más disperso "
          f"(stride {mas_disperso}): no hay contraste que hacer.")
    elif contraste is None or not contraste:
        p(f"**R2 · Significación.** No se pudo contrastar (hacen falta ≥2 "
          f"semillas en cada grupo).")
    else:
        pv = contraste.get("p")
        dif = por_stride[mejor['point']['stride']]["value"] - por_stride[mas_disperso]["value"]
        p(f"**R2 · Significación.** stride {mejor['point']['stride']} contra "
          f"stride {mas_disperso}: diferencia **{num(abs(dif))}**, "
          f"`p` = **{num(pv, 5)}** ({contraste.get('n')} vs "
          f"{contraste.get('m')} semillas, "
          f"{'exacto' if contraste.get('exact') else 'aproximado'} sobre "
          f"{contraste.get('arrangements')} reordenaciones).")
        if pv is not None and pv < 0.05 and abs(dif) > delta:
            p("")
            p("  → **la densidad de la rejilla mueve la calidad de predicción.**")
        else:
            p("")
            p("  → **con estas semillas la densidad no separa.** No es «da igual»: "
              "es que el efecto, si lo hay, cabe dentro del ruido de "
              "reinicialización de este dataset.")
    p("")
    if un_solo_brazo:
        p("**R3 · Monotonía.** ⚠ **No evaluable** con un solo brazo.")
    elif sin_banda:
        p(f"**R3 · Monotonía.** ⚠ **No evaluable sin banda de ruido**: con δ = 0 "
          f"saldrían {len(rupturas)} «ruptura(s)», pero cualquier diferencia lo "
          f"sería.")
    elif rupturas:
        p(f"**R3 · Monotonía.** ⚠ **{len(rupturas)} ruptura(s)** mayores que δ:")
        for r in rupturas:
            p(f"  - de stride {r['de']} ({num(r['valor_de'])}) a stride "
              f"{r['a']} ({num(r['valor_a'])}): baja al hacerse más denso")
        p("")
        p("  No se suaviza ni se explica a posteriori: a igual cómputo no hay "
          "mecanismo obvio para esto, así que **abre sospecha sobre R4**.")
    else:
        p("**R3 · Monotonía.** Sin rupturas mayores que δ: el objetivo no empeora "
          "al hacer la rejilla más densa.")
    p("")
    if sin_config or not pasos_por_run:
        p(f"**R4 · Control de presupuesto.** ⚠ **No se pudo aplicar**: falta el "
          f"`config.json` de {len(sin_config)} run(s), así que no se puede leer "
          f"qué trabajo hizo cada época.")
    elif con_pool_entero:
        p(f"**R4 · Control de presupuesto.** ❌ **FALLA.** {len(con_pool_entero)} "
          f"run(s) entrenaron sobre el **pool entero** (`windows_per_epoch` = 0), "
          f"así que su época no es la de los demás.")
        p("")
        p("  **El estudio NO declara nada hasta explicarlo.**")
    elif len(distintos) > 1:
        p(f"**R4 · Control de presupuesto.** ❌ **FALLA.** Los runs no hicieron los "
          f"mismos pasos por época: {distintos}.")
        p("")
        p("  **El estudio NO declara nada hasta explicarlo**: la tabla estaría "
          "midiendo cuánto entrenó cada brazo, no la densidad de su rejilla.")
    else:
        p(f"**R4 · Control de presupuesto.** ✅ **Pasa.** Los {len(pasos_por_run)} "
          f"runs hicieron **{distintos[0]} pasos de gradiente por época**, el mismo "
          f"número, leído del `config.json` que devolvió cada máquina.")
        if mediana:
            p("")
            p(f"  *Información, no control*: la mediana fue {num(mediana, 1)} "
              f"s/época. Ese número **no** sirve de control porque cada run corrió "
              f"en una máquina alquilada distinta: la dispersión **dentro** de un "
              f"mismo brazo llega a "
              f"{num(max(dentro.values()), 2) if dentro else '-'}× "
              f"(misma config, máquinas distintas), o sea del orden de la que hay "
              f"entre brazos. Es ruido de catálogo, y es lo que la criba de "
              f"velocidad existe para recortar.")
    if not un_solo_brazo:
        p("")
        p(f"*{tie_reason(frontera, delta)}*")

    salida = {
        "estudio": args.estudio, "objetivo": objetivo, "direccion": direccion,
        "delta": delta, "razon_delta": razon_delta,
        "brazos": por_brazo,
        "tabla": [{"stride": s, "valor": por_stride[s]["value"],
                   "min": por_stride[s]["value_min"],
                   "max": por_stride[s]["value_max"],
                   "sem": por_stride[s].get("value_sem"),
                   "n_seeds": por_stride[s]["n_seeds"],
                   "seconds_per_epoch": por_stride[s].get("seconds_per_epoch")}
                  for s in strides],
        "R1_saturacion": ({"evaluable": False,
                           "motivo": "un solo brazo con medidas"} if un_solo_brazo else
                          {"evaluable": True, "stride": saturacion,
                           "frontera": en_frontera,
                           "mejor": mejor["point"]["stride"],
                           "cerrado_por_arriba": cerrado_por_arriba}),
        "R2_contraste": contraste,
        "R3_rupturas": rupturas, "sin_banda_de_ruido": sin_banda,
        "R4_control": {"pasos_por_epoca": distintos,
                       "runs_sin_config": sin_config,
                       "runs_con_pool_entero": con_pool_entero,
                       "mediana_s_por_epoca": mediana,
                       "dispersion_dentro_de_brazo": dentro,
                       "pasa": bool(pasos_por_run) and not sin_config
                               and not con_pool_entero and len(distintos) == 1},
        "pendientes": [{"sweep": s, "run": r, "status": st}
                       for s, r, st in pendientes],
    }
    destino = Path(args.json) if args.json else ROOT / "data" / f"{args.estudio}-informe.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    p("")
    # `relative_to` LANZA si el destino esta fuera del repo, y estaba en la ultima
    # linea: el informe salia entero y el proceso moria con codigo 1 justo despues.
    # Un fallo que deja el trabajo hecho y devuelve error hace que quien llama
    # descarte un resultado bueno. Encontrado por su propio test, que pasaba
    # --json /tmp/... (2026-08-27).
    try:
        donde = destino.relative_to(ROOT)
    except ValueError:
        donde = destino
    p(f"JSON: {donde}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
