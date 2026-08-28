#!/usr/bin/env python3
r"""Los recorridos del estudio de `dropout`, con el criterio ya escrito.

Que es esto
-----------
`docs/plan-dropout-2026-08-28.md` fija QUE se mide y COMO se lee, ANTES de mirar
nada. Este script solo lo traduce a recorridos en `sweeps/`, igual que
`estudio_cierre.py` y `estudio_prioridades.py` hacen con los suyos: aqui no se
alquila ni se entrena nada. Lo que gasta es `estudio_flota.py`.

    .venv/bin/python scripts/estudio_dropout.py --dataset <B> --fase tanteo
    .venv/bin/python scripts/estudio_dropout.py --dataset <B> --fase completo

LAS TRES COSAS QUE HAY QUE RESPETAR SI SE TOCA ESTO
---------------------------------------------------
1. **El rango de la fase 2 lo elige TABLA_PICO, no quien lanza.** El plan (§5)
   escribio, antes de tener un solo numero, que rango se barre para CADA
   resultado posible del tanteo. Esa tabla esta aqui abajo en codigo y el plan
   la repite en prosa: si divergen, manda esta -- pero lo correcto es que no
   diverjan, y por eso el script IMPRIME el rango con su motivo al crearlo.

   Si el rango se eligiera despues de ver los numeros, "el rango lo dijo el
   tanteo" no significaria nada: estaria ajustado al resultado.

2. **`p*` se DERIVA del disco, no se teclea.** `--fase completo` lee los runs de
   `do-t` con las mismas funciones que el informe (`sweep_trials` +
   `aggregate_seeds`), asi que el pico que usa el rango es el mismo que sale en
   la tabla del informe. Un numero definido dos veces acaba divergiendo.
   `--pico` existe para forzarlo a mano y AVISA EN VOZ ALTA cuando se usa.

3. **La fase 2 corre sus 5 semillas ENTERAS, sin sumar las 2 del tanteo.**
   Sumarlas seria legitimo (mismo dato, misma red, misma receta) y es lo que
   hizo el bloque D del cierre, pero `estudio_informe.py` trabaja sobre UN
   recorrido: la tabla de 5 semillas habria que componerla a mano. Cuesta ~0,43 $
   de mas y compra un veredicto que sale entero de una herramienta.
   Por eso `seed0` NO se usa aqui, a proposito.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.sweeps.generate import generate_sweep       # noqa: E402
from fv.sweeps.runner import es_medida, sweep_trials  # noqa: E402
from fv.sweeps.spec import SweepError               # noqa: E402
from fv.sweeps.store import SweepStore              # noqa: E402
from fv.sweeps.winner import aggregate_seeds        # noqa: E402
from fv.training.recipe import RecipeStore          # noqa: E402
from fv.training.registry import RunStore           # noqa: E402

RUNS = RunStore()
SWEEPS = SweepStore()

RECIPE = "plan40"
OBJECTIVE = "f1"
EPOCHS = 150                    # el de los recorridos con los que se compara
ESTUDIO = "dropout-2026-08-28"
EJE = "dropout"

# La foveada VIGENTE: ws16-p2-d2-L4, 167.852 parametros. La misma base que todos
# los estudios de la tabla -- no la "mejor conocida" (#14 dejo border_px 8 y
# overlap_fovea_px 7 medidos y SIN APLICAR). Ver la reserva 2 del plan §6.
FOVEADA = {"n_layers": 4, "channels": [16] * 4}
BORDER_PX = 4

TANTEO = "do-t"
COMPLETO = "do-v"

# El tanteo: 4 valores x 2 semillas = 8 runs. ACOTA, no declara (p minimo
# alcanzable con 2 contra 2 = 0,333).
RANGO_TANTEO = [0.0, 0.1, 0.25, 0.5]

# --------------------------------------------------------------------------
# LA TABLA DEL §5, EN CODIGO. Escrita antes de tener un solo numero: para cada
# resultado posible del tanteo, que rango barre la fase 2 y por que.
#
# La clave es el pico `p*` del tanteo (mejor f1 medio). Cubre los cuatro
# valores de RANGO_TANTEO y ninguno mas: un `p*` que no este aqui es un fallo
# de coherencia, no un caso a improvisar, y el script se niega.
TABLA_PICO = {
    0.0:  ([0.0, 0.05, 0.1, 0.2],
           "el tanteo dice que el dropout solo hace dano. El paso mas pequeno de "
           "aquella rejilla era 0,1, y que 0,1 ya duela NO descarta una ganancia "
           "en 0,05: cerrar 'no ayuda' exige haber mirado ahi"),
    0.1:  ([0.0, 0.05, 0.1, 0.2],
           "gana 0,1: se encierra por los dos lados (0,05 por debajo, 0,2 por "
           "encima) para que el optimo quede INTERIOR y no en un borde"),
    0.25: ([0.0, 0.1, 0.25, 0.4],
           "gana 0,25: se encierra por los dos lados (0,1 y 0,4)"),
    0.5:  ([0.0, 0.25, 0.5, 0.7],
           "gana el EXTREMO del tanteo, asi que el eje no esta acotado por la "
           "derecha: se extiende mas alla (0,7, legal en [0,1)), como hizo "
           "`borde-ancho` cuando su ganador salio en el borde"),
}


def pico_del_tanteo(store: SweepStore) -> tuple[float, int, float]:
    """El mejor valor del eje en `do-t`, leido del disco con las MISMAS
    funciones que usa `estudio_informe.py`.

    Devuelve (pico, n_puntos_medidos, f1_medio_del_pico). Falla en vez de
    adivinar: un tanteo a medias no puede elegir el rango de un estudio que
    cuesta 20 runs, y un rango elegido sobre 3 de 8 runs es peor que ninguno
    porque no lo dice nadie.
    """
    try:
        tabla = sweep_trials(TANTEO, store=store, run_store=RUNS)
    except Exception as e:                       # noqa: BLE001 - el motivo importa
        raise SystemExit(
            f"No puedo leer el tanteo '{TANTEO}': {e}\n"
            f"  -> crealo y correlo primero:\n"
            f"     .venv/bin/python scripts/estudio_dropout.py --dataset <B> --fase tanteo")
    medidos = [t for t in tabla["trials"] if es_medida(t)]
    total = len(tabla["trials"])
    if not medidos:
        raise SystemExit(
            f"El tanteo '{TANTEO}' no tiene ningun punto medido todavia "
            f"(0/{total}).\n"
            f"  -> mira como va:  .venv/bin/python scripts/estudio_progreso.py "
            f"--sweep {TANTEO} --tabla")
    if len(medidos) < total:
        raise SystemExit(
            f"El tanteo '{TANTEO}' esta a medias: {len(medidos)}/{total} runs "
            f"medidos.\n"
            f"  El rango de la fase 2 (20 runs) se elige con el pico del tanteo; "
            f"elegirlo sobre un tanteo incompleto es elegirlo sobre el subconjunto "
            f"de puntos que dio la casualidad de terminar primero.\n"
            f"  -> termina el tanteo, o forzalo a mano con --pico <valor> si de "
            f"verdad es lo que quieres.")
    grupos = aggregate_seeds(medidos, tabla["direction"], "seconds_per_epoch")
    mejor = grupos[0]
    return float(mejor["point"][EJE]), len(medidos), float(mejor["value"])


def crear(name: str, rango: list, semillas: int, dataset: str, receta: dict,
          store: SweepStore, motivo: str) -> bool:
    # `store.exists`, y no un listado de `store.root`: la raiz plana puede no
    # existir todavia (en un clon nuevo del repo de datos no existe), y ademas
    # un recorrido puede vivir en el archivo fechado `<anio>/<mes>/`, donde un
    # `iterdir` de la raiz NO lo ve. Un "no existe" falso aqui crea un recorrido
    # duplicado con el mismo nombre.
    if store.exists(name):
        print(f"  = {name:6s} ya existe, se deja. Borralo a mano si de verdad "
              f"quieres rehacerlo (sus runs son dinero ya gastado).")
        return False
    try:
        spec = generate_sweep(
            name, dataset, EJE, list(rango),
            base_recipe=RECIPE, base_recipe_value=receta,
            objective=OBJECTIVE, budget={"epochs": EPOCHS},
            seeds=semillas, device="cpu",
            overrides=FOVEADA, border_px=BORDER_PX,
            study=ESTUDIO, sstore=store,
        )
    except SweepError as e:
        print(f"  ! {name:6s} {e.code}: {e.message}\n      {e.hint}")
        return False
    n = len(spec["points"])
    sem = spec["space"].get("seed", [receta["seed"]])
    print(f"  + {name:6s} {len(spec['space'][EJE])} valores x semillas {sem} "
          f"= {n:3d} runs")
    print(f"      rango: {rango}")
    print(f"      motivo: {motivo}")
    for d in spec.get("discarded") or []:
        print(f"      descartado {d['point']}: {d['problems'][0]['message']}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fase", required=True, choices=["tanteo", "completo"])
    ap.add_argument("--pico", type=float, default=None,
                    help="fuerza el pico del tanteo en vez de derivarlo del disco. "
                         "Solo para --fase completo, y avisa en voz alta")
    args = ap.parse_args()

    store = SweepStore()
    receta = dict(RecipeStore().get(RECIPE).as_dict())

    if args.fase == "tanteo":
        print(f"Fase 1 -- TANTEO. 2 semillas: ACOTA, no declara (p minimo "
              f"alcanzable 0,333).")
        ok = crear(TANTEO, RANGO_TANTEO, 2, args.dataset, receta, store,
                   "0,0 es el ancla (el vigente); 0,1 y 0,25 son los valores que "
                   "propuso el inventario; 0,5 es el default clasico Y acota el "
                   "eje por la derecha")
        siguiente = TANTEO
    else:
        if args.pico is not None:
            pico = float(args.pico)
            print(f"⚠ PICO FORZADO A MANO: {pico}. NO se ha derivado del tanteo, "
                  f"asi que el rango de abajo no lo respalda el disco.")
        else:
            pico, n, f1 = pico_del_tanteo(store)
            print(f"Pico derivado de '{TANTEO}' ({n} runs medidos): "
                  f"{EJE} = {pico} con f1 = {f1:.4f} de media.")
        if pico not in TABLA_PICO:
            raise SystemExit(
                f"El pico {pico} no esta en la tabla del plan §5, que cubre "
                f"{sorted(TABLA_PICO)}.\n"
                f"  Eso significa que el tanteo midio otro rango del que el plan "
                f"escribio, o que --pico trae un valor inventado. Improvisar aqui "
                f"un rango es justo lo que el plan existe para impedir.\n"
                f"  -> revisa `docs/plan-dropout-2026-08-28.md` §5 y RANGO_TANTEO.")
        rango, motivo = TABLA_PICO[pico]
        print(f"Fase 2 -- ESTUDIO COMPLETO. 5 semillas: con 5 contra 5 el p minimo "
              f"alcanzable es 0,0079, o sea que R4 SI puede declarar al 5 %.")
        ok = crear(COMPLETO, rango, 5, args.dataset, receta, store, motivo)
        siguiente = COMPLETO

    if ok:
        print("\nSiguiente paso (esto SI gasta -- ALQUILA MAQUINAS QUE FACTURAN):")
        print(f"  .venv/bin/python scripts/estudio_estimar.py --sweep {siguiente}")
        print(f"  scripts/desacoplar.sh .venv/bin/python scripts/estudio_flota.py \\")
        print(f"      --sweep {siguiente} --cpu E5-26 --criba 2 --git "
              f"--horas-max {6 if args.fase == 'tanteo' else 8} --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
