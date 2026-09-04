#!/usr/bin/env python3
"""Entrena UN brazo reusando el bucle del repo, sin tocarlo.

    python nn/entrenar_local.py 1k5 crear  --epochs 3 --patience 0
    python nn/entrenar_local.py 1k5 seguir --more 8

⚠⚠ COMO SE HACE «SIN TOCAR PRODUCCION», Y POR QUE ASI
   Instruccion del dueno (2026-09-03): el codigo de produccion no se cambia para probar
   una idea; si hace falta, se copia localmente.

   `fv.training.loop` tiene todo lo demas --dataloaders, perdida, metricas,
   checkpoints, REANUDACION, `patience`-- y es identico para cualquier `nn.Module` y
   cualquier `Dataset`. Asi que aqui se sustituyen DOS simbolos en el modulo ya
   importado, en memoria, y se llama al entrenamiento de siempre.

   ⚠ DOS, no uno, y esa es la diferencia con los experimentos anteriores de la serie:

       loop.build_model            -> la plana de `red_local.py`
       loop.FoveatedWindowDataset  -> el `Dataset` de `dataset.py`

   El segundo hace falta porque el bucle construye la vista foveada desde las imagenes
   (`loop.py:250-258`) y aqui el dato YA viene preprocesado. Sin ese parche entrenaria
   sobre la vista 20x20 de siempre --que es exactamente el fallo por el que se detuvo
   el experimento anterior-- y no fallaria nada.

   ⚠ El parche vive SOLO en este proceso: nada se escribe en `src/fv/`. Comprobado al
   final de cada corrida (`_comprobar_intacto`).

⚠ HAY QUE REANUDAR POR AQUI, NO CON `fv-continue` A PELO
   Los dos parches se montan aqui. Con `fv-continue` suelto se construiria la red de la
   config --una plana normal sobre la vista 20x20-- y el `state_dict` no casaria. Falla
   ruidosamente, que es lo que se quiere, pero el camino bueno es este fichero.

⚠ EL DATASET ORIGEN SIGUE SIENDO EL DE SIEMPRE, y no es un adorno
   `--window-dataset` apunta a `dirty1000-80px-16px-r20260827` aunque el dato salga del
   `.npz` preprocesado. Es lo que mantiene VIVO el guard de `reanudar`
   (`loop.py:213`), que compara la huella del manifest contra la del `config.json` del
   run: si alguien cambia el dataset origen bajo los pies, la reanudacion se niega. Con
   un dataset inventado ese guard pasaria siempre y no protegeria de nada.
"""

from __future__ import annotations

import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

from fv.models import builder as _builder                # noqa: E402
from fv.training import cli as _cli                      # noqa: E402
from fv.training import loop as _loop                    # noqa: E402
from dataset import como_produccion                      # noqa: E402
from red_local import ENTRADAS, construir                # noqa: E402

# Los MISMOS que los siete gemelos y que el experimento detenido. Que coincidan es lo
# que permite hablar de la serie sin explicar nada.
DATASET = "dirty1000-80px-16px-r20260827"
RED = "plana-20-1k3"      # solo por la geometria/provenance; la red la pone el parche
RECETA = "plan40"         # dueno 2026-09-04: lr 0,0014 · batch_size 85 · semilla 1

BRAZOS = tuple(ENTRADAS)

_ORIGINAL_MODEL = _loop.build_model
_ORIGINAL_DS = _loop.FoveatedWindowDataset


def run_de(brazo: str) -> str:
    return f"plana-pp{brazo}-s1"          # `pp` = plana sobre preprocesado


def _comprobar_intacto() -> None:
    """El fichero de produccion sigue igual y los dos simbolos, restaurados."""
    fuente = (REPO / "src" / "fv" / "models" / "builder.py").read_text()
    assert "padding=pad" in fuente, "builder.py cambio: el parche NO puede tocarlo"
    assert _builder.build_model is _ORIGINAL_MODEL, "se piso build_model globalmente"
    assert _loop.FoveatedWindowDataset is _ORIGINAL_DS, "se piso el Dataset globalmente"


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] not in BRAZOS:
        print(__doc__)
        print(f"  brazos: {' · '.join(BRAZOS)}")
        return 2
    brazo, accion, resto = args[0], args[1], args[2:]
    run = run_de(brazo)

    # Se reusa la CLI ENTERA del repo --`fv-train` / `fv-continue`-- para que el
    # `--epochs`, el `--patience`, el formato del progreso y el resumen final sean
    # literalmente los mismos que en el resto de la serie.
    if accion == "crear":
        sys.argv = ["fv-train", "--name", run, "--window-dataset", DATASET,
                    "--network", RED, "--recipe", RECETA] + resto
        entrada = _cli.main
    elif accion == "seguir":
        sys.argv = ["fv-continue", "--name", run] + resto
        entrada = _cli.main_continue
    else:
        print("uso: entrenar_local.py <brazo> crear --epochs N | seguir --more N")
        return 2

    c, _h, _w = ENTRADAS[brazo]
    _loop.build_model = lambda net: construir(brazo)      # noqa: ARG005
    _loop.FoveatedWindowDataset = como_produccion(brazo)
    try:
        codigo = entrada()
    finally:
        _loop.build_model = _ORIGINAL_MODEL
        _loop.FoveatedWindowDataset = _ORIGINAL_DS
    _comprobar_intacto()
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
