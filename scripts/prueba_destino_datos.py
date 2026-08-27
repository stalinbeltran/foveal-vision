#!/usr/bin/env python3
r"""¿Dónde acaban los datos de un estudio: en este repo o en `foveal-vision-data`?

Por qué existe
--------------
La separación de datos (ver CLAUDE.md) se hizo en dos mitades: se COPIARON los
artefactos a `foveal-vision-data`, y el código siguió leyendo y escribiendo aquí.
Esas dos mitades se leen igual desde fuera --el repo de datos existe y tiene los
ficheros-- pero significan cosas opuestas, y la diferencia sólo se ve corriendo
algo y mirando dónde cae.

Esto corre **un estudio de verdad, muy corto y LOCAL**: un recorrido real, un
punto, dos épocas, con el mismo `run_sweep` que usa la flota. No alquila nada.
Luego dice, fichero a fichero, en qué repo aterrizó cada artefacto.

    .venv/bin/python scripts/prueba_destino_datos.py            # crea, corre y comprueba
    .venv/bin/python scripts/prueba_destino_datos.py --limpiar  # borra lo que dejó

Código de salida: 0 si los datos van a `foveal-vision-data` · 1 si siguen aquí.

⚠ Corto pero REALISTA a propósito: 2 épocas de la red vigente sobre el dataset
vigente. Un mock no valdría -- lo que se comprueba es la ruta que elige el
código de verdad, y un doble de prueba elegiría la suya.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.sweeps.generate import generate_sweep      # noqa: E402
from fv.sweeps.runner import run_sweep             # noqa: E402
from fv.sweeps.store import SweepStore             # noqa: E402
from fv.training.recipe import RecipeStore         # noqa: E402
from fv.training.registry import RunStore          # noqa: E402

NOMBRE = "data-destino-chk"
DATASET = "dirty1000-80px-16px-r20260826"
DATA_REPO = ROOT.parent / "foveal-vision-data"


def limpiar() -> None:
    st = SweepStore()
    if st.exists(NOMBRE):
        st.delete(NOMBRE)
        print(f"  borrado sweeps/{NOMBRE}")
    for d in sorted((ROOT / "runs").glob(f"{NOMBRE}-*")):
        shutil.rmtree(d, ignore_errors=True)
        print(f"  borrado runs/{d.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limpiar", action="store_true")
    args = ap.parse_args()
    if args.limpiar:
        limpiar()
        return 0

    print(f"# ¿Dónde caen los datos de un estudio?\n")
    print(f"repo de código : {ROOT}")
    print(f"repo de datos  : {DATA_REPO}"
          f"{'' if DATA_REPO.exists() else '   ⚠ NO EXISTE (no está clonado)'}\n")

    limpiar()
    receta = dict(RecipeStore().get("plan40").as_dict())
    receta["patience"] = 0                      # que pare en la 2, no antes
    print("1/3  creando un recorrido real (1 punto, 2 épocas)...")
    spec = generate_sweep(
        NOMBRE, DATASET, "overlap_fovea_px", [2],
        base_recipe="plan40", base_recipe_value=receta, objective="f1",
        budget={"epochs": 2}, seeds=1, device="cpu",
        overrides={"n_layers": 4, "channels": [16] * 4}, border_px=4,
        study="prueba-destino-datos")
    print(f"     {len(spec['points'])} punto(s)\n")

    print("2/3  entrenando (el MISMO run_sweep que usa la flota)...")
    run_sweep(NOMBRE)
    print()

    print("3/3  dónde aterrizó cada artefacto:\n")
    aqui, alla = [], []
    for etiqueta, patron in (("recorrido", f"sweeps/{NOMBRE}/*"),
                             ("runs", f"runs/{NOMBRE}-*/*")):
        for f in sorted(ROOT.glob(patron)):
            if f.is_file():
                aqui.append(f.relative_to(ROOT))
        if DATA_REPO.exists():
            for f in sorted(DATA_REPO.rglob(f"*{NOMBRE}*")):
                if f.is_file():
                    alla.append(f.relative_to(DATA_REPO))

    print(f"  en foveal-vision (código): {len(aqui)} fichero(s)")
    for f in aqui[:8]:
        print(f"      {f}")
    if len(aqui) > 8:
        print(f"      ...y {len(aqui) - 8} más")
    print(f"\n  en foveal-vision-data    : {len(alla)} fichero(s)")
    for f in alla[:8]:
        print(f"      {f}")

    print()
    if alla and not aqui:
        print("🟢 Los datos van al REPO DE DATOS. La separación está aplicada.")
        return 0
    if alla and aqui:
        print("🟡 Caen en LOS DOS SITIOS. Es lo peor de ambos: dos copias que "
              "divergen sin avisar.")
        return 1
    print("🔴 Los datos siguen cayendo en ESTE repo. La separación NO está "
          "aplicada en el código:")
    print("   los artefactos los escriben `RunStore`/`SweepStore`, que resuelven "
          "contra\n   `settings.py` (`runs_root`, `sweeps_root`) y ése sigue "
          "apuntando a FV_ROOT.")
    print("   Ver CLAUDE.md § «Los datos de los estudios van a foveal-vision-data».")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
