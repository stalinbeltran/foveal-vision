#!/usr/bin/env python3
"""Entrena la red SIN RELLENO reusando el bucle del repo, sin tocarlo.

    python nn/entrenar_local.py crear  --epochs 1       # crea el run
    python nn/entrenar_local.py seguir --more 2        # anade epocas

⚠⚠ COMO SE HACE «SIN TOCAR PRODUCCION», Y POR QUE ASI
   Instruccion del dueno (2026-09-03): el codigo de produccion no se cambia para
   probar una idea; si hace falta, se copia localmente.

   `fv.training.loop` construye el modelo en UN sitio --`build_model(net)`,
   linea 277-- y todo lo demas (dataloaders, perdida, metricas, checkpoints,
   reanudacion, `patience`) es identico para cualquier `nn.Module`. Asi que aqui
   se sustituye ESE UNICO simbolo en el modulo ya importado, en memoria, y se
   llama al entrenamiento de siempre.

   Se hace asi y no copiando el bucle porque copiarlo son ~200 lineas que
   tendrian que dar EXACTAMENTE los mismos numeros que los gemelos --misma
   perdida, mismo barajado, mismo criterio de `best`-- y dos copias de eso
   divergen. Lo que se quiere variar es el relleno; todo lo demas tiene que ser
   literalmente el mismo codigo.

   ⚠ El parche vive SOLO en este proceso: nada se escribe en `src/fv/`.
   Comprobado al final de cada corrida (`_comprobar_intacto`).
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
from red_local import PlanaSinPadding                    # noqa: E402

RUN = "plana-2k7sp-s1"
DATASET = "dirty1000-80px-16px-r20260827"
RED = "plana-20-2k7"      # la MISMA config que el gemelo; el relleno lo quita el parche
RECETA = "plan40"

_ORIGINAL = _loop.build_model


def _comprobar_intacto() -> None:
    """El fichero de produccion sigue igual: el parche es SOLO en memoria."""
    fuente = (REPO / "src" / "fv" / "models" / "builder.py").read_text()
    assert "padding=pad" in fuente, "builder.py cambio: el parche NO puede tocarlo"
    assert _builder.build_model is _ORIGINAL, "se ha pisado build_model globalmente"


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    # Se reusa la CLI ENTERA del repo --`fv-train` / `fv-continue`-- para que el
    # `--epochs`, el `--patience`, el formato del progreso y el resumen final
    # sean literalmente los mismos que en los gemelos.
    if args[0] == "crear":
        sys.argv = ["fv-train", "--name", RUN, "--window-dataset", DATASET,
                    "--network", RED, "--recipe", RECETA] + args[1:]
        entrada = _cli.main
    elif args[0] == "seguir":
        sys.argv = ["fv-continue", "--name", RUN] + args[1:]
        entrada = _cli.main_continue
    else:
        print("uso: entrenar_local.py crear --epochs N | seguir --more N")
        return 2

    _loop.build_model = _build_sin_relleno       # el parche, solo en este proceso
    try:
        codigo = entrada()
    finally:
        _loop.build_model = _ORIGINAL
    _comprobar_intacto()
    return codigo


def _build_sin_relleno(net: dict):
    """Lo que `loop.py` llamara en vez de `build_model`."""
    return PlanaSinPadding(net)


if __name__ == "__main__":
    raise SystemExit(main())
