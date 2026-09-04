#!/usr/bin/env python3
"""Entrena UN brazo reusando el bucle del repo, sin tocarlo.

    python nn/entrenar_local.py 1k5 crear  --epochs 1     # crea el run
    python nn/entrenar_local.py 1k5 seguir --more 2       # anade epocas

⚠⚠ COMO SE HACE «SIN TOCAR PRODUCCION», Y POR QUE ASI
   Instruccion del dueno (2026-09-03): el codigo de produccion no se cambia para
   probar una idea; si hace falta, se copia localmente.

   `fv.training.loop` construye el modelo en UN sitio --`build_model(net)`-- y
   todo lo demas (dataloaders, perdida, metricas, checkpoints, reanudacion,
   `patience`) es identico para cualquier `nn.Module`. Asi que aqui se sustituye
   ESE UNICO simbolo en el modulo ya importado, en memoria, y se llama al
   entrenamiento de siempre.

   Se hace asi y no copiando el bucle porque copiarlo son ~200 lineas que
   tendrian que dar EXACTAMENTE los mismos numeros que los gemelos --misma
   perdida, mismo barajado, mismo criterio de `best`-- y dos copias divergen.

   ⚠ El parche vive SOLO en este proceso: nada se escribe en `src/fv/`.
   Comprobado al final de cada corrida (`_comprobar_intacto`).

⚠⚠ HAY QUE REANUDAR POR AQUI, NO CON `fv-continue` A PELO
   El brazo congelado se monta en el parche. Con `fv-continue` suelto se
   construiria la red de la config --una plana normal, cabeza de 324-- y el
   `state_dict` guardado no casaria. Falla ruidosamente, que es lo que se quiere,
   pero el camino bueno es este fichero. Lo mismo vale para los gemelos.

⚠ Y LA CAPA CONGELADA SE COMPRUEBA AL TERMINAR, no se da por hecho.
   Un `requires_grad=False` que alguien deshaga sin querer convierte esto en otro
   experimento --uno donde L1 tambien entrena-- y los numeros seguirian saliendo.
   `_comprobar_congelado` contrasta el kernel del checkpoint contra el que dice
   `comun/preproceso.py`, que es la unica fuente de verdad de cual era.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(EXP.parent / "comun"))

from fv.models import builder as _builder                # noqa: E402
from fv.training import cli as _cli                      # noqa: E402
from fv.training import loop as _loop                    # noqa: E402
from preproceso import cargar_kernel                     # noqa: E402
from red_local import PESOS, construir                   # noqa: E402

# Los MISMOS que los seis gemelos. Que coincidan es lo que permite ponerlos uno
# al lado del otro sin tener que explicar nada.
DATASET = "dirty1000-80px-16px-r20260827"
RED = "plana-20-1k3"      # la geometria; el kernel congelado lo pone red_local
RECETA = "plan40"

BRAZOS = ("1k3", "1k5", "1k7")

_ORIGINAL = _loop.build_model


def run_de(brazo: str) -> str:
    """El nombre del run. `pre` para que no se confunda con el gemelo libre."""
    return f"plana-pre{brazo}-s1"


def _comprobar_intacto() -> None:
    """El fichero de produccion sigue igual: el parche es SOLO en memoria."""
    fuente = (REPO / "src" / "fv" / "models" / "builder.py").read_text()
    assert "padding=pad" in fuente, "builder.py cambio: el parche NO puede tocarlo"
    assert _builder.build_model is _ORIGINAL, "se ha pisado build_model globalmente"


def _comprobar_congelado(brazo: str) -> None:
    """El kernel del checkpoint sigue siendo el que se congelo, bit a bit.

    ⚠ La ruta se le pregunta a `RunStore.path`, NO se compone a mano. Los runs no
    estan en `<datos>/runs/<run>`: `fv.artefactos` los agrupa por mes
    (`<datos>/2026/09-septiembre/runs/<run>`). La primera version de esto componia
    la ruta plana, no encontraba el fichero y **se saltaba la comprobacion sin
    decir nada** -- justo el fallo silencioso que esta funcion existe para evitar.
    """
    from fv.training.registry import RunStore
    ck = RunStore().path(run_de(brazo)) / "last.pt"
    if not ck.exists():
        print(f"  [congelado] ⚠ no hay {ck}: NO comprobado")
        return
    estado = torch.load(ck, map_location="cpu", weights_only=False)
    estado = estado.get("model", estado)
    kern = cargar_kernel(brazo, pesos=PESOS)
    guardado = estado["center_convs.0.weight"].float()
    if not torch.equal(guardado, kern.peso):
        raise SystemExit(
            f"⚠⚠ el kernel congelado de {brazo} HA CAMBIADO en {ck}.\n"
            f"  esto ya no es «L1 congelada»: es otro experimento, y sus numeros\n"
            f"  no se pueden comparar con los gemelos. No sigas: mira "
            f"`red_local.construir`.")
    print(f"  [congelado] el kernel de {brazo} sigue intacto en last.pt")


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2 or args[0] not in BRAZOS:
        print(__doc__)
        print(f"  brazos: {' · '.join(BRAZOS)}")
        return 2
    brazo, accion, resto = args[0], args[1], args[2:]
    kf = int(brazo[-1])
    run = run_de(brazo)

    # Se reusa la CLI ENTERA del repo --`fv-train` / `fv-continue`-- para que el
    # `--epochs`, el `--patience`, el formato del progreso y el resumen final
    # sean literalmente los mismos que en los gemelos.
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

    def _build(net: dict):
        """Lo que `loop.py` llamara en vez de `build_model`."""
        return construir(kf, net)

    _loop.build_model = _build                   # el parche, solo en este proceso
    try:
        codigo = entrada()
    finally:
        _loop.build_model = _ORIGINAL
    _comprobar_intacto()
    _comprobar_congelado(brazo)
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
