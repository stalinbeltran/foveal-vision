#!/usr/bin/env python3
"""Los tres brazos, en cadena y REANUDABLE.

    python nn/cadena.py [--epocas 37] [--solo A]

⚠⚠ POR QUE ES REANUDABLE Y NO UN GUION DE TRES LINEAS
   Esto lo lanza `desacoplar-persistente.sh`, o sea una unidad de systemd con
   `Restart=on-failure`. Un guion que empezara siempre por el principio se
   estrellaria contra el `fv-train` del brazo A --«ese run ya existe»-- y la
   unidad entraria en el bucle que este proyecto ya ha pagado dos veces. Asi que
   mira el estado y elige, por brazo: SALTAR (ya tiene sus epocas) · CONTINUAR
   (tiene menos) · CREAR (no existe). Es la misma regla que
   `entrenar_para_inferencia.sh`.

⚠ El aviso NO decide el codigo de salida. Lo pone `cadena.sh`, con `|| true`, y
   despues del trabajo: un `notify.mjs` que falla no puede marcar como fallidas
   37 epocas que terminaron bien (medido el 2026-09-04, 62 reinicios).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

from fv.training.registry import RunStore              # noqa: E402
from entrenar_local import BRAZOS                      # noqa: E402

PY = str(REPO / ".venv" / "bin" / "python")
ENTRENAR = str(AQUI / "entrenar_local.py")


def epocas_hechas(run: str) -> int:
    p = RunStore().path(run) / "metrics.jsonl"
    if not p.exists():
        return 0
    return sum(1 for line in p.read_text().splitlines() if line.strip())


def correr(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"fallo ({r.returncode}): {' '.join(cmd)}")


def guardar(brazo: str, run: str) -> None:
    """Los pesos y el registro, dentro del experimento (regla 3 de la carpeta).

    Se copian `best.pt`, `last.pt`, `config.json`, `metrics.jsonl` y
    `summary.json`: el registro completo, no solo el tensor. Estas redes son de
    ~20-33 k parametros, asi que la razon de la regla general --2,3 GB de 862
    runs-- no aplica aqui.
    """
    origen = RunStore().path(run)
    destino = AQUI / "pesos" / brazo
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in ("best.pt", "last.pt", "config.json", "metrics.jsonl", "summary.json"):
        f = origen / nombre
        if f.exists():
            shutil.copy2(f, destino / nombre)
        else:
            print(f"  ⚠ {run}: no hay {nombre}")
    print(f"  guardado en {destino.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=37,
                    help="las mismas 37 que los seis de la serie plana")
    ap.add_argument("--solo", default=None, help="un solo brazo: A, B o C")
    a = ap.parse_args()

    brazos = [a.solo] if a.solo else list(BRAZOS)
    for brazo in brazos:
        run = BRAZOS[brazo]["run"]
        hechas = epocas_hechas(run)
        print(f"\n=== brazo {brazo} ({run}) · {hechas}/{a.epocas} epocas ===", flush=True)
        if hechas >= a.epocas:
            print("  ya esta: se salta")
        elif hechas == 0:
            correr([PY, ENTRENAR, brazo, "--epochs", str(a.epocas)])
        else:
            correr([PY, ENTRENAR, brazo, "seguir", "--more", str(a.epocas - hechas)])
        guardar(brazo, run)

    print("\nCADENA COMPLETA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
