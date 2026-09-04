#!/usr/bin/env python3
"""Lleva LOS TRES brazos a la misma época — y se niega si se han desincronizado.

    python nn/avanzar.py --hasta 3          # crea si hace falta y avanza hasta la ép. 3
    python nn/avanzar.py --hasta 11         # el siguiente tramo
    python nn/avanzar.py --comprobar        # ¿están los tres donde deben?

⚠ Los brazos cuestan ~7 s/época (medido 2026-09-04, este droplet), asi que los tres
   stops enteros (37 épocas x 3) son ~13 min. La version de 4 capas costaba 44-46 s.

⚠⚠ POR QUE ESTO ES UN SCRIPT Y NO UN `.sh` CON COMENTARIOS
   La invariante de este experimento es «los brazos se leen a la MISMA época». En
   los gemelos eso lo sostenía la prosa de `avances.sh`, que encadena
   `seguir --more 8` a ojo. Aquí no basta, por una razón medida en el codigo:

       `loop.py:376`  if recipe.patience and no_improve >= recipe.patience: break

   ...y `no_improve` se RESTAURA de `last.pt`. O sea que un brazo que ya tocó
   `patience` corre UNA época y para, aunque le pidas trece. `summary.json` lo
   dice (`stopped_early: true`), pero nadie lo estaba leyendo: el desfase sólo se
   veía abriendo los tres `metrics.jsonl` a mano. Un brazo comparado a la época
   24 contra otros a la 37 no es una tabla, es un error silencioso.

   Por eso aquí `patience` se pone a **0** al CREAR (queda congelado en el
   `config.json` del run) y esto comprueba el resultado en vez de confiar en el.

⚠ `patience` se decide al crear y NO se pasa al reanudar. `reanudar` lo acepta y
   lo pisa por llamada (`loop.py:203-204`), así que dos brazos podrían avanzar con
   paciencias distintas sin una sola queja.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

from entrenar_local import BRAZOS, run_de                        # noqa: E402

PY = REPO / ".venv" / "bin" / "python"
ENTRENAR = AQUI / "entrenar_local.py"

# Los MISMOS stops que los seis gemelos. Que caigan en las mismas épocas es lo
# que permite ponerlos uno al lado del otro sin explicar nada. El 3 es el que
# pidió el dueño para el primer feedback.
STOPS = (3, 11, 24, 37)


def estado(brazo: str) -> dict | None:
    """Lo que dice el `summary.json` del run, o None si no existe todavía."""
    from fv.training.registry import RunStore
    s = RunStore().path(run_de(brazo)) / "summary.json"
    if not s.exists():
        return None
    return json.loads(s.read_text())


def _epocas(brazo: str) -> int:
    e = estado(brazo)
    return int(e["epochs_run"]) if e else 0


def comprobar(esperado: int | None = None) -> int:
    """¿Están los tres en la misma época, y ninguno paró antes de tiempo?"""
    filas, malo = [], False
    for b in BRAZOS:
        e = estado(b)
        if e is None:
            filas.append((b, 0, False, "sin crear"))
            malo = True
            continue
        ep, corto = int(e["epochs_run"]), bool(e.get("stopped_early"))
        malo |= corto
        filas.append((b, ep, corto, "⚠ PARO ANTES (early-stop)" if corto else "ok"))

    epocas = {f[1] for f in filas}
    desfase = len(epocas) > 1
    malo |= desfase
    if esperado is not None and epocas != {esperado}:
        malo = True

    for b, ep, _c, nota in filas:
        print(f"  {b:5} época {ep:>3}   {nota}")
    if desfase:
        print(f"  ⚠⚠ LOS BRAZOS NO ESTAN EN LA MISMA EPOCA: {sorted(epocas)}\n"
              f"     Una tabla que compare estos brazos entre sí NO es válida.\n"
              f"     Llévalos al mismo sitio: nn/avanzar.py --hasta {max(epocas)}")
    elif esperado is not None and epocas != {esperado}:
        print(f"  ⚠ los tres están en {epocas.pop()}, se esperaba {esperado}")
    elif not malo:
        print(f"  ✓ los tres en la época {epocas.pop()}: la comparación es válida")
    return 1 if malo else 0


def avanzar(hasta: int) -> int:
    for b in BRAZOS:
        ya = _epocas(b)
        if ya == 0:
            # `--patience 0` = sin early-stop. Ver la cabecera: es lo que
            # garantiza que «misma época» sea cierto por construcción.
            cmd = [str(PY), str(ENTRENAR), b, "crear",
                   "--epochs", str(hasta), "--patience", "0"]
        elif ya < hasta:
            cmd = [str(PY), str(ENTRENAR), b, "seguir", "--more", str(hasta - ya)]
        else:
            print(f"[{b}] ya está en la época {ya}: nada que hacer")
            continue
        print(f"\n[{b}] {' '.join(cmd[2:])}")
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"⚠ el brazo {b} falló (código {r.returncode}). Se para aquí: "
                  f"avanzar los demás dejaría la tabla desincronizada.")
            return r.returncode
    print(f"\nlos tres brazos en la época {hasta}:")
    return comprobar(hasta)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hasta", type=int, help=f"época objetivo (stops: {STOPS})")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    if a.hasta:
        return avanzar(a.hasta)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
