r"""Ejecutor del recorrido `p40-lr-L4` (docs/plan-lr-L4.md).

Re-barre `lr` sobre la red L4 que gano el plan de 40 h. El criterio esta escrito
ANTES, en el documento, y este script NO lo re-implementa: crea el recorrido si
falta, lo corre, y al cerrar imprime el veredicto que calcula `suggest_winner`.

Reanudable por diseno (§4 del documento): `run_sweep` rehace todo punto que no
este `done`/`cancelled`, asi que relanzarlo continua donde se quedo. El equipo se
apaga por falta de energia y hiberna; el watchdog se apoya en eso.

    .\.venv\Scripts\python.exe scripts\plan_lr_L4.py [--stop-after N] [--clear-stop]

`--stop-after N` para limpiamente DESPUES de N puntos terminados: la parada se
pide en cuanto el punto N cierra, y el runner la comprueba antes de arrancar el
siguiente, asi que no se tira computo a medias. Es como se corre la sonda de §2
(`--stop-after 1`), que es el primer punto del recorrido de verdad, no un
experimento aparte.

Una parada pedida SOBREVIVE en disco y este script NO la borra solo: el watchdog
relanza sin argumentos, asi que borrarla de oficio haria que una parada pedida
por el usuario se perdiera en el siguiente relanzamiento. Para reanudar tras la
sonda hay que escribir `--clear-stop`, que es una decision, no un efecto.

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

from fv.sweeps.generate import generate_sweep          # noqa: E402
from fv.sweeps.runner import run_sweep, sweep_trials   # noqa: E402
from fv.sweeps.spec import SweepError                  # noqa: E402
from fv.sweeps.store import SweepStore                 # noqa: E402
from fv.sweeps.winner import suggest_winner            # noqa: E402

# --- constantes de docs/plan-lr-L4.md §1, copiadas AQUI y en ningun otro sitio.
NAME = "p40-lr-L4"
DATASET = "dirty1000-80px-16px"
AXIS = "lr"
# el ORDEN es la mitigacion (§1): si el presupuesto se corta, lo que falta son
# los interiores. Primero el extremo caro e incierto, despues el vigente.
RANGE = [0.00035, 0.0014, 0.0006, 0.0009]
SEEDS = 5
EPOCHS_CAP = 150       # alto A PROPOSITO: `patience` tiene que ser quien pare
RECIPE = "plan40"
# la base es el ganador L4 exacto; `channels` viaja con `n_layers` porque la
# profundidad vive en los dos sitios (plan-40h.md §7, «el mismo dato en dos
# sitios»): pasar uno sin el otro deja una base que `check_run` rechaza.
BASE_NETWORK = {"n_layers": 4, "channels": [16, 16, 16, 16]}

LOG = ROOT / "plan-lr-L4.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Recorrido p40-lr-L4 (docs/plan-lr-L4.md)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="para limpiamente tras N puntos terminados; 0 = todos")
    # APAGADO por defecto, y a proposito: una parada pedida sobrevive en disco, y
    # el watchdog relanza este script sin argumentos. Si borrara la parada solo,
    # una parada pedida por el usuario (desde la UI, o la del --stop-after de la
    # sonda) se perderia en el siguiente relanzamiento y el recorrido seguiria
    # corriendo contra su voluntad. Limpiarla es una decision, no un efecto.
    ap.add_argument("--clear-stop", action="store_true",
                    help="olvida una parada pedida antes (hace falta para "
                         "reanudar tras un --stop-after)")
    args = ap.parse_args()

    store = SweepStore()
    if not store.exists(NAME):
        try:
            e = generate_sweep(NAME, DATASET, AXIS, RANGE, base_recipe=RECIPE,
                               objective="f1", budget={"epochs": EPOCHS_CAP},
                               seeds=SEEDS, overrides=BASE_NETWORK, device="cpu",
                               sstore=store)
        except SweepError as ex:
            log(f"NO se pudo crear: [{ex.code}] {ex.message} -> {ex.hint}")
            return 2
        log(f"recorrido '{NAME}' creado: base {e['base_label']}, "
            f"{len(e['points'])} puntos ({len(RANGE)} valores x {SEEDS} semillas), "
            f"tope {EPOCHS_CAP} epocas")
    else:
        log(f"recorrido '{NAME}' ya existe -- se reanuda (salta lo hecho)")

    if args.clear_stop:
        store.clear_stop(NAME)
        log("parada anterior olvidada (--clear-stop)")
    elif store.stop_requested(NAME):
        # decirlo, y no entrenar: un recorrido que para en el primer punto sin
        # explicar por que parece un cuelgue (formatos.md §2, la razon viaja)
        log("hay una parada pedida: no se entrena nada. Reanuda con --clear-stop")
        return 3

    def progress(done: int, total: int, run: str) -> None:
        log(f"  punto {done}/{total} terminado: {run}")
        if args.stop_after and done >= args.stop_after:
            log(f"  --stop-after {args.stop_after} alcanzado: pido parada "
                f"(el siguiente punto no arranca)")
            store.request_stop(NAME)

    state = run_sweep(NAME, store, progress=progress)
    log(f"estado final: {state.get('status')} ({state.get('done')}/{state.get('total')})")

    trials = sweep_trials(NAME, store)
    log(f"ranking por {trials['objective']} ({trials['direction']}) en el checkpoint:")
    for t in trials["trials"]:
        ep = f"ep {t['epoch']}/{t['epochs']}" if t["epoch"] else "sin checkpoint"
        log(f"  {t['run']}: {t['value']}  [{ep}]  {json.dumps(t['point'])}")
    try:
        sug = suggest_winner(NAME, store=store)
        log(f"veredicto: {'EMPATE' if sug['tie'] else 'ganador distinguible'} "
            f"(delta={sug['delta']:.4f}, {sug['delta_source']})")
        log(f"  {sug['tie_reason']}")
        log(f"  sugerido: {json.dumps(sug['suggested']['point'])}")
    except SweepError as ex:
        log(f"sin veredicto todavia: {ex.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
