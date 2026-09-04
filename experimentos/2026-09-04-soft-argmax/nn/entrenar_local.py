#!/usr/bin/env python3
"""Entrena un brazo de este experimento reusando el bucle del repo, sin tocarlo.

    python nn/entrenar_local.py A --epochs 37      # soft-argmax
    python nn/entrenar_local.py B --epochs 37      # soft-argmax + dispersion
    python nn/entrenar_local.py C --epochs 37      # control: misma pila, lectura lineal
    python nn/entrenar_local.py A seguir --more 5  # anadir epocas a uno ya creado

⚠⚠ COMO SE HACE «SIN TOCAR PRODUCCION», Y POR QUE ASI
   Mismo mecanismo que los seis experimentos de la serie plana: `fv.training.loop`
   construye el modelo en UN sitio --`build_model(net)`-- y todo lo demas
   (dataloaders, metricas, checkpoints, `patience`) es identico para cualquier
   `nn.Module`. Se sustituye ESE simbolo en el modulo ya importado, en memoria.
   `src/fv/` no se escribe; se comprueba al final de cada corrida.

   El brazo B necesita ademas un termino de perdida, asi que parchea tambien
   `loop.corner_loss` -- el mismo mecanismo, el mismo alcance (este proceso).

⚠⚠ EL REGULARIZADOR VA SOLO EN ENTRENAMIENTO, NUNCA EN VALIDACION
   `plan40` trae `monitor: val_loss`, o sea que el `val_loss` decide QUE epoca
   guarda `best.pt`. Si el termino de dispersion entrara tambien en validacion,
   el `best.pt` del brazo B se elegiria con un criterio distinto al de A, al de C
   y al del ancla, y los cuatro `val_loss` dejarian de ser comparables entre si
   -- sin que nada fallara. Se distingue por `model.training`, que el bucle ya
   conmuta (`model.train()` / `evaluate` hace `model.eval()`).

LOS TRES BRAZOS
   A  soft-argmax, beta aprendida.                   Cabeza +0,2 % sobre el ancla.
   B  A + penalizacion de la dispersion del mapa.    Misma cabeza que A.
   C  MISMA pila conv, MISMOS 4 mapas, coordenadas por `Linear` global.
      ⚠ C tiene 66,6 % MAS cabeza que el ancla, y se deja a proposito: es un
      control GENEROSO. Si A le gana con 19.289 parametros contra 32.096, la
      ventaja no puede ser de tamano. Igualarlo habria pedido encoger la pila, y
      entonces A y C ya no compartirian el trozo que se quiere tener fijo.
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
from red_local import CabezaSoftArgmax                   # noqa: E402

DATASET = "dirty1000-80px-16px-r20260827"   # el NO preprocesado, el de los otros seis
RED = "plana-20-4k7"                        # el mismo cuerpo que el ancla `plana-4k7-s1`
RECETA = "plan40"

# El peso del termino de dispersion del brazo B. MEDIDO, no elegido a ojo: con el
# mapa plano de la inicializacion la varianza vale 0,3639 y el termino de posicion
# 0,4362 (error tipico de arranque, 5,84 px sobre una ventana de 16). 0,1 deja el
# regularizador en el 8,3 % de la posicion: un empujon, no un segundo objetivo.
# ⚠ NO esta medido que este sea un buen valor; esta medido que sea un empujon.
LAMBDA_VAR = 0.1

BRAZOS = {
    "A": {"run": "sargmax-a-s1", "modo": "softargmax", "lambda_var": 0.0},
    "B": {"run": "sargmax-b-s1", "modo": "softargmax", "lambda_var": LAMBDA_VAR},
    "C": {"run": "sargmax-c-s1", "modo": "lineal",     "lambda_var": 0.0},
}

_BUILD_ORIGINAL = _loop.build_model
_LOSS_ORIGINAL = _loop.corner_loss


def _comprobar_intacto() -> None:
    """El fichero de produccion sigue igual: el parche es SOLO en memoria."""
    fuente = (REPO / "src" / "fv" / "models" / "builder.py").read_text()
    assert "self.head = nn.Linear(flat + self.n_edge, 12)" in fuente, \
        "builder.py cambio: el parche NO puede tocarlo"
    assert _builder.build_model is _BUILD_ORIGINAL, "se ha pisado build_model globalmente"
    perdida = (REPO / "src" / "fv" / "training" / "losses.py").read_text()
    assert "def corner_loss(" in perdida, "losses.py cambio"


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] not in BRAZOS:
        print(__doc__)
        return 2
    brazo = BRAZOS[args[0]]
    resto = args[1:]

    if resto and resto[0] == "seguir":
        sys.argv = ["fv-continue", "--name", brazo["run"]] + resto[1:]
        entrada = _cli.main_continue
    else:
        sys.argv = ["fv-train", "--name", brazo["run"], "--window-dataset", DATASET,
                    "--network", RED, "--recipe", RECETA] + resto
        entrada = _cli.main

    creado: dict = {}

    def _build(net: dict):
        m = CabezaSoftArgmax(net, modo=brazo["modo"])
        creado["m"] = m
        return m

    lam = float(brazo["lambda_var"])

    def _loss(logits, target, lambda_pos, pos_weight, smooth_l1_beta):
        base = _LOSS_ORIGINAL(logits, target, lambda_pos, pos_weight, smooth_l1_beta)
        m = creado.get("m")
        # `m.training` es la puerta: en validacion el bucle hace `model.eval()`,
        # asi que el val_loss que se registra y que elige `best.pt` es el de
        # siempre en los tres brazos. Ver el encabezado.
        if lam > 0.0 and m is not None and m.training and m.ultima_var is not None:
            ex = target[:, :, 0]
            disp = (m.ultima_var * ex).sum() / ex.sum().clamp(min=1.0)
            return base + lam * disp
        return base

    _loop.build_model = _build
    _loop.corner_loss = _loss
    try:
        codigo = entrada()
    finally:
        _loop.build_model = _BUILD_ORIGINAL
        _loop.corner_loss = _LOSS_ORIGINAL
    _comprobar_intacto()
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
