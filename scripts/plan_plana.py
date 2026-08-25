r"""Ejecutor de los estudios de la CNN plana (docs/plan-plana.md).

Encadenado DETRAS del recorrido `p40-lr-L4`: no entrena nada mientras ese siga
vivo. Lleva la red plana de control (`regions: single`) a su optimo con dos
estudios -- cribado de 1 semilla y confirmacion de 5 -- y NO compara nada con la
foveada: eso lo decide el usuario despues, con estos datos.

    .\.venv\Scripts\python.exe scripts\plan_plana.py [--force] [--max-hours H]

Reanudable por diseno: cada fase se salta si ya esta hecha, y un estudio a medias
continua por donde iba. El watchdog lo relanza sin argumentos, asi que todo lo
que decida este script tiene que sobrevivir en disco por si solo.

`--force` se salta la guarda del recorrido anterior (para probar el encadenado,
no para adelantarlo: dos entrenamientos peleandose por los nucleos no aceleran
nada, y el coste medido queda mintiendo -- plan-40h.md §7).

Salida ASCII (consola cp1252).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.studies.driver import (advance, confirm, create_study,  # noqa: E402
                               status)
from fv.studies.store import StudyStore                          # noqa: E402
from fv.sweeps.runner import run_sweep, sweep_trials             # noqa: E402
from fv.sweeps.spec import SweepError                            # noqa: E402
from fv.sweeps.store import SweepStore                           # noqa: E402
from fv.sweeps.winner import suggest_winner                      # noqa: E402
from fv.training.registry import RunStore                        # noqa: E402

# --- constantes de docs/plan-plana.md, copiadas AQUI y en ningun otro sitio.
GATE_SWEEP = "p40-lr-L4"      # nada arranca hasta que este cierre
DATASET = "dirty1000-80px-16px"
RECIPE = "plan40"
EPOCHS_CAP = 150              # alto A PROPOSITO: tiene que parar `patience`
OBJECTIVE = "f1"

# la base plana: control C de plan-cnn-plana.md §3. `border_px=0` + regions
# single => entrada 16x16, sin anillo.
# ⚠ NO pongas aqui ningun campo que sea EJE. `derive_base` aplica los overrides
# DESPUES de los winners, asi que un campo fijado aqui anula el arrastre del
# estudio EN SILENCIO. Fijar `n_layers: 4` "como punto de partida" hizo que el
# paso de `lr` se midiera a L4 aunque el cribado habia coronado L5 (§6.4).
BASE_NETWORK = {"regions": "single", "border_reduce": 1}
BORDER_PX = 0

SCREEN = "plana-screen"       # 1 semilla: descarta barato, NO concluye
CONFIRM = "plana-confirm"     # 5 semillas: lo unico que entra en una tabla
N_LAYERS_RANGE = [2, 3, 4, 5, 6, 8]
LR_RANGE = [0.00035, 0.0009, 0.0014, 0.0028, 0.005]

# §2.1: por encima de este techo el recorrido lleva guardas. Aqui la guarda es
# parar y decirlo, en vez de quemar dias en silencio.
MAX_PROJECTED_HOURS = 40.0

LOG = ROOT / "plan-plana.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def log_once(msg: str) -> None:
    """Como `log`, pero calla si el mensaje anterior decia lo mismo. El watchdog
    despierta cada 10 minutos durante la espera (~200 veces mientras corre el
    recorrido anterior): sin esto el log seria 200 lineas identicas y la que
    importa -- el cambio de estado -- no se veria."""
    prev = ""
    if LOG.exists():
        try:
            tail = LOG.read_text(encoding="utf-8").rstrip("\n").rsplit("\n", 1)[-1]
            prev = tail[21:] if len(tail) > 21 else tail   # sin el sello de tiempo
        except OSError:
            pass
    if prev.strip() != msg.strip():
        log(msg)


def gate_is_open(sstore: SweepStore) -> tuple[bool, str]:
    """El recorrido anterior ha terminado? El estado lo dice EL PROPIO recorrido
    (su state.json), no un fichero aparte: una segunda copia del mismo dato es
    como se rompen las cosas en este proyecto."""
    if not sstore.exists(GATE_SWEEP):
        return True, f"'{GATE_SWEEP}' no existe: nada que esperar"
    st = sstore.state(GATE_SWEEP)
    s = st.get("status")
    if s == "done":
        return True, f"'{GATE_SWEEP}' terminado ({st.get('done')}/{st.get('total')})"
    return False, (f"'{GATE_SWEEP}' esta {s} ({st.get('done', 0)}/{st.get('total', '?')} "
                   f"puntos): no se arranca nada")


def _plan(name: str, axes: list[dict], seeds: int) -> dict:
    return {
        "window_dataset": DATASET, "base_recipe": RECIPE, "objective": OBJECTIVE,
        "seeds": seeds, "budget": {"epochs": EPOCHS_CAP},
        # la red base del estudio: la plana. Sin esto el estudio derivaria la
        # foveada del window_size y estaria optimizando la red equivocada.
        "base_network": dict(BASE_NETWORK), "border_px": BORDER_PX,
        "axes": axes,
    }


def confirmed_winners(name: str, store: StudyStore) -> dict:
    """{campo: valor} de lo que se confirmo en cada paso del estudio.

    NO se lee de `progress['winners']`, que parece lo obvio y son otros datos:
    ahi solo entran los campos de C (`winner_overrides` filtra por
    NETWORK_PARAMS, porque su trabajo es alimentar la base derivada del paso
    siguiente), asi que `lr` -- que es de D -- NUNCA aparece; y los que si
    entran vienen envueltos en {'value', 'from'}. El punto crudo, que es lo que
    hace falta para estrechar el rango, lo guarda cada paso en su `winner`.
    Medido el 2026-08-09 en el ensayo de la cadena: leer el sitio equivocado
    daba rango [{'value': 2, ...}] para n_layers y [None] para lr.
    """
    out: dict = {}
    for step in status(name, store)["steps"]:
        if step.get("confirmed") and step.get("winner"):
            out.update(step["winner"])
    return out


def neighbours(value, ordered: list) -> list:
    """El ganador y sus dos vecinos DEL RANGO CRIBADO. Confirmar solo el ganador
    no acota nada; con los vecinos, o el optimo queda dentro, o se ve que sigue
    pegado a un borde -- y eso ultimo se publica (R3 de plan-lr-L4)."""
    if value not in ordered:
        return [value]
    i = ordered.index(value)
    return ordered[max(0, i - 1):i + 2]


def cost_so_far(sweep: str, sstore: SweepStore, rstore: RunStore) -> tuple[float, int]:
    """(segundos por epoca medianos, epocas medianas) de lo ya entrenado. Sirve
    para rehacer la proyeccion con numeros MEDIDOS: la estimacion aritmetica de
    §4 es eso, una estimacion."""
    try:
        trials = sweep_trials(sweep, sstore, rstore)
    except SweepError:
        return 0.0, 0
    spe = sorted(t["seconds_per_epoch"] for t in trials["trials"]
                 if t.get("seconds_per_epoch"))
    eps = sorted(t["epochs"] for t in trials["trials"] if t.get("epochs"))
    if not spe or not eps:
        return 0.0, 0
    return spe[len(spe) // 2], eps[len(eps) // 2]


def run_study(name: str, plan: dict, store: StudyStore, sstore: SweepStore,
              rstore: RunStore, max_hours: float, budget: dict) -> bool:
    """Corre (o continua) un estudio hasta el final. True si quedo completo.

    Es el bucle de `fv-study --auto`, con log a fichero y con la guarda de
    presupuesto entre puntos -- que es donde se puede parar sin tirar computo.
    """
    if not store.exists(name):
        create_study(name, plan, store)
        log(f"estudio '{name}' creado: {len(plan['axes'])} ejes, "
            f"{plan['seeds']} semilla(s), base plana")
    else:
        log(f"estudio '{name}' ya existe -- se reanuda (salta lo hecho)")

    while True:
        st = status(name, store)
        if st["done"]:
            log(f"estudio '{name}' COMPLETO. ganadores: {json.dumps(st['winners'])}")
            return True
        out = advance(name, store, sstore)
        step = out["step"]
        sweep = step["sweep"]
        log(f"  paso {step['step']}: eje {step['axis']} sobre base "
            f"{step['base_label']} ({step['points']} puntos, "
            f"{step['discarded']} descartados) -> recorrido '{sweep}'")

        stopped = {"yes": False}

        def progress(done: int, total: int, run: str) -> None:
            log(f"    punto {done}/{total} terminado: {run}")
            spe, eps = cost_so_far(sweep, sstore, rstore)
            if not spe:
                return
            # la proyeccion se rehace con lo MEDIDO, no con lo estimado
            left = budget["runs_left"] - budget["runs_done"] - done
            proj = budget["hours_spent"] + (left * spe * eps) / 3600.0
            if done == 1:
                log(f"    coste medido: {spe:.1f} s/epoca, {eps} epocas -> "
                    f"proyeccion de la cadena: {proj:.1f} h")
            if proj > max_hours and not stopped["yes"]:
                stopped["yes"] = True
                log(f"    PARADA POR PRESUPUESTO: la proyeccion ({proj:.1f} h) supera "
                    f"el techo de {max_hours:.0f} h. Se pide parada; el siguiente "
                    f"punto no arranca. Todo queda reanudable.")
                sstore.request_stop(sweep)

        run_sweep(sweep, sstore, rstore, progress=progress)
        state = sstore.state(sweep)
        if state.get("status") != "done":
            log(f"  recorrido '{sweep}' quedo {state.get('status')} "
                f"({state.get('done')}/{state.get('total')}): no se confirma nada")
            return False

        spe, eps = cost_so_far(sweep, sstore, rstore)
        budget["runs_done"] += step["points"]
        budget["hours_spent"] += (step["points"] * spe * eps) / 3600.0

        sug = suggest_winner(sweep, store=sstore, run_store=rstore)
        best = sug["best"]
        band = (f"  banda {best['value_min']:.4f}-{best['value_max']:.4f} "
                f"n={best['n_seeds']}" if best["n_seeds"] > 1 else "  (1 semilla)")
        log(f"    mejor: {json.dumps(best['point'])} = {best['value']:.4f}{band}")
        log(f"    delta={sug['delta']:.4f} ({sug['delta_source']})")
        log(f"    {sug['tie_reason']}")
        point = sug["suggested"]["point"]
        log(f"    sugerido: {json.dumps(point)} -> confirmado")
        confirm(name, point, store)


def main() -> int:
    ap = argparse.ArgumentParser(description="Estudios de la CNN plana (docs/plan-plana.md)")
    ap.add_argument("--force", action="store_true",
                    help="ignora la guarda del recorrido anterior (para probar el "
                         "encadenado, no para adelantarlo)")
    ap.add_argument("--max-hours", type=float, default=MAX_PROJECTED_HOURS,
                    help="techo de la proyeccion; al superarlo para y lo dice")
    args = ap.parse_args()

    store, sstore, rstore = StudyStore(), SweepStore(), RunStore()

    # La guarda que faltaba (§6.4): un campo fijado en BASE_NETWORK gana a los
    # winners del estudio, asi que fijar un EJE mata el arrastre sin decir nada.
    # Se comprueba ANTES de entrenar 41 runs, no despues de mirarlos.
    ejes = {"n_layers", "lr"}
    chocan = sorted(ejes & set(BASE_NETWORK))
    if chocan:
        log(f"NO se arranca: BASE_NETWORK fija {chocan}, que tambien son ejes. "
            f"`derive_base` aplica los overrides DESPUES de los winners, asi que "
            f"eso anularia el arrastre del ganador en silencio (docs/plan-plana.md "
            f"§6.4). Quitalos de BASE_NETWORK.")
        return 2

    open_, why = gate_is_open(sstore)
    if not open_ and not args.force:
        log_once(f"esperando: {why}")
        return 0
    log(f"guarda abierta: {why}" if open_ else f"guarda IGNORADA (--force): {why}")

    if store.exists(CONFIRM) and status(CONFIRM, store)["done"]:
        log("la cadena ya esta completa: no hay nada que hacer")
        return 0

    # presupuesto de la cadena entera, en runs, para proyectar con lo medido
    budget = {"runs_done": 0, "hours_spent": 0.0,
              "runs_left": len(N_LAYERS_RANGE) + len(LR_RANGE) + 3 * 5 + 3 * 5}

    # --- fase 1: cribado, 1 semilla. Descarta barato; NO concluye nada.
    screen_plan = _plan(SCREEN, [{"axis": "n_layers", "range": N_LAYERS_RANGE},
                                 {"axis": "lr", "range": LR_RANGE}], seeds=1)
    if not run_study(SCREEN, screen_plan, store, sstore, rstore, args.max_hours, budget):
        log("el cribado no quedo completo: se reanuda en el siguiente arranque")
        return 3

    # --- fase 2: confirmacion, 5 semillas, alrededor del ganador cribado
    w = confirmed_winners(SCREEN, store)
    if "n_layers" not in w or "lr" not in w:
        log(f"el cribado no dejo ganador de los dos ejes ({json.dumps(w)}): "
            f"no se estrecha nada a ciegas")
        return 3
    nl = neighbours(w["n_layers"], N_LAYERS_RANGE)
    lr = neighbours(w["lr"], LR_RANGE)
    for label, won, rng, full in (("n_layers", w["n_layers"], nl, N_LAYERS_RANGE),
                                  ("lr", w["lr"], lr, LR_RANGE)):
        if won in (full[0], full[-1]):
            log(f"AVISO: el cribado gano en un EXTREMO de {label} ({won}); la "
                f"confirmacion {rng} no lo acota por ese lado, y eso se publica tal cual")
    log(f"confirmacion: n_layers={nl}, lr={lr} (ganadores del cribado: {json.dumps(w)})")

    confirm_plan = _plan(CONFIRM, [{"axis": "n_layers", "range": nl},
                                   {"axis": "lr", "range": lr}], seeds=5)
    if not run_study(CONFIRM, confirm_plan, store, sstore, rstore, args.max_hours, budget):
        log("la confirmacion no quedo completa: se reanuda en el siguiente arranque")
        return 3

    final = confirmed_winners(CONFIRM, store)
    log("=" * 60)
    log(f"CADENA COMPLETA. la plana en su optimo: {json.dumps(final)}")
    log("NO se ha comparado nada con la foveada: eso es el paso siguiente, "
        "y lo decide el usuario (docs/plan-plana.md §3.5)")
    log("puedes quitar el watchdog: Unregister-ScheduledTask "
        "-TaskName 'fv-plana-watchdog' -Confirm:$false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
