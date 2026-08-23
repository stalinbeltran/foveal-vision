#!/usr/bin/env python3
"""Corre un SUBCONJUNTO de los puntos de un recorrido. Es lo que se ejecuta
DENTRO de cada maquina alquilada.

Recibe indices GLOBALES de punto (`--puntos 0,3,6`), no un criterio: quien
reparte es `estudio_flota.py`, y este script solo obedece. Asi la politica de
reparto -y sus consecuencias estadisticas- vive en un solo sitio, el que alquila
las maquinas, en vez de repetida aqui.

Los nombres de los runs los pone `point_run_name` con el indice global, asi que
los runs de las N maquinas se juntan despues en un solo directorio `runs/` y
forman el recorrido entero, sin renombrar nada.

    python3 scripts/estudio_lote.py --sweep lr-alto-L4 --puntos 1,4,7

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
    ap.add_argument("--puntos", required=True,
                    help="indices globales de punto, separados por coma")
    ap.add_argument("--que", default="", help="como llamar a este lote en el log")
    ap.add_argument("--rc", default="", help="fichero donde dejar el codigo de salida")
    args = ap.parse_args()

    t0 = time.monotonic()
    store = SweepStore()
    if not store.exists(args.sweep):
        log(f"ERROR: no existe el recorrido '{args.sweep}'")
        return 2
    try:
        puntos = [int(x) for x in args.puntos.split(",") if x.strip()]
    except ValueError:
        log(f"ERROR: --puntos quiere numeros separados por coma, no {args.puntos!r}")
        return 2

    log(f"{args.que or 'lote'}: puntos {puntos} del recorrido '{args.sweep}'")

    def progreso(hechos: int, total: int, run: str) -> None:
        log(f"  punto {hechos}/{total} terminado: {run} "
            f"({(time.monotonic() - t0) / 60:.1f} min acumulados)")

    try:
        estado = run_sweep(args.sweep, progress=progreso, solo=puntos)
    except SweepError as exc:
        log(f"ERROR [{exc.code}] {exc.message} -> {exc.hint}")
        return 2
    log(f"terminado: {json.dumps(estado)} en {(time.monotonic() - t0) / 60:.1f} min")
    # `run_sweep` no lanza cuando un punto falla: lo apunta en last_error y sigue.
    # Sin esta comprobacion, un lote con puntos perdidos volveria como bueno y el
    # hueco solo se veria al juntar los runs y contar.
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
