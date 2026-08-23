#!/usr/bin/env python3
"""Corre los puntos de UNA semilla de un recorrido. Es lo que se ejecuta DENTRO
de cada maquina alquilada.

Una maquina por semilla, y no por valor del eje, a proposito: la semilla es el
eje replica, asi que si una maquina resulta ser mas lenta o mas rara que las
otras, esa rareza entra por igual en TODOS los valores que se comparan. Repartir
por valor de `lr` haria lo contrario -- confundir la maquina con la respuesta.

Los nombres de los runs los pone `point_run_name` con el indice GLOBAL del punto
dentro del recorrido, asi que los runs de las N maquinas se juntan despues en un
solo directorio `runs/` y forman el recorrido entero, sin renombrar nada.

    python3 scripts/estudio_semilla.py --sweep lr-alto-L4 --seed 2

Escribe el progreso por stdout (una linea por punto terminado) y deja el codigo
de salida en el fichero que se le diga con --rc, para que el que vigila desde
fuera pueda distinguir "sigue corriendo" de "termino" sin mantener viva una
sesion de SSH.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.sweeps.runner import run_sweep          # noqa: E402
from fv.sweeps.spec import SweepError           # noqa: E402
from fv.sweeps.store import SweepStore          # noqa: E402


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rc", default="", help="fichero donde dejar el codigo de salida")
    args = ap.parse_args()

    t0 = time.monotonic()
    store = SweepStore()
    if not store.exists(args.sweep):
        log(f"ERROR: no existe el recorrido '{args.sweep}'")
        return 2

    log(f"semilla {args.seed} del recorrido '{args.sweep}'")

    def progreso(hechos: int, total: int, run: str) -> None:
        log(f"  punto {hechos}/{total} terminado: {run} "
            f"({(time.monotonic() - t0) / 60:.1f} min acumulados)")

    try:
        estado = run_sweep(args.sweep, progress=progreso, only_seed=args.seed)
    except SweepError as exc:
        log(f"ERROR [{exc.code}] {exc.message} -> {exc.hint}")
        return 2
    log(f"terminado: {json.dumps(estado)} en {(time.monotonic() - t0) / 60:.1f} min")
    # `run_sweep` no lanza cuando un punto falla: lo apunta en last_error y sigue.
    # Sin esta comprobacion, una semilla con puntos perdidos volveria como buena
    # y el hueco solo se veria al juntar los runs y contar.
    return 0 if estado.get("done") == estado.get("total") else 1


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    finally:
        rc = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--rc=")]
        if not rc and "--rc" in sys.argv:
            i = sys.argv.index("--rc")
            rc = sys.argv[i + 1:i + 2]
        if rc:
            Path(rc[0]).write_text(str(code), encoding="utf-8")
    raise SystemExit(code)
