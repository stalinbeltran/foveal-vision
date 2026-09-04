#!/usr/bin/env python3
"""El evaluador COMPARTIDO, pero construyendo la red local sin relleno.

    python nn/evaluar_local.py --stop 01-3epocas

⚠ Se parchea el mismo unico simbolo que en el entrenamiento y se llama al
evaluador de `experimentos/comun/`: mismas 10 ventanas, mismas figuras, mismo
`resumen.json`. Reescribirlo aqui haria que los stops de este experimento no
fueran comparables con los de los gemelos, que es lo unico que se quiere.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

from fv.models import builder as _builder            # noqa: E402
from red_local import PlanaSinPadding                # noqa: E402

_ORIGINAL = _builder.build_model
_builder.build_model = lambda net: PlanaSinPadding(net)

sys.argv = ["aplicar_kernels.py", "--exp", str(EXP), "--red", "plana-20-1k3"] + sys.argv[1:]
if "--run" not in sys.argv and "--stop" in sys.argv:
    i = sys.argv.index("--stop")
    if sys.argv[i + 1] != "00-sin-entrenar":
        sys.argv += ["--run", "plana-1k3sp-s1"]
try:
    runpy.run_path(str(REPO / "experimentos" / "comun" / "aplicar_kernels.py"),
                   run_name="__main__")
finally:
    _builder.build_model = _ORIGINAL
