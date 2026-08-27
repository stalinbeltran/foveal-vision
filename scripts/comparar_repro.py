#!/usr/bin/env python3
r"""¿Reproduce el dato? Compara `repro-chk` epoca a epoca con el run original.

Un `windows.npz` reconstruido puede tener otra huella y la MISMA informacion
(compresion distinta) o informacion distinta (otro rasterizado). La huella no
distingue esos dos casos y las dos posibilidades llevan a decisiones opuestas,
asi que se decide entrenando: mismo punto, misma semilla, misma familia de CPU
--donde esta MEDIDO que el entrenamiento sale identico bit a bit-- y se comparan
las curvas.

    .venv/bin/python scripts/comparar_repro.py

Lectura, escrita antes de mirar:
  identico bit a bit  -> el dato reproduce; los recorridos que SUMAN semillas a
                         runs viejos (ov-sig, bp-sig, pl-f2-*) son validos.
  distinto            -> es otro dataset: se le pone nombre nuevo y esos tres
                         recorridos hay que rehacerlos enteros sobre el nuevo.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv import datarepo  # noqa: E402  (needs the sys.path above)

PARES = [("repro-chk-0000-overlap_fovea_px2_seed1",
          "ov-fov-0010-overlap_fovea_px2_seed1"),
         ("repro-chk-0001-overlap_fovea_px2_seed2",
          "ov-fov-0011-overlap_fovea_px2_seed2")]
CAMPOS = [("train_loss", lambda d: d["train_loss"]),
          ("val_loss",   lambda d: d["val"]["loss"]),
          ("val_f1",     lambda d: d["val"]["f1"]),
          ("pos_err_px", lambda d: d["val"]["pos_err_px"])]


def curva(run: str) -> list:
    f = datarepo.resolve("runs", run) / "metrics.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def main() -> int:
    veredicto = []
    for nuevo, viejo in PARES:
        a, b = curva(nuevo), curva(viejo)
        print(f"\n=== {nuevo}\n    contra {viejo}")
        if not a:
            print("    (aun no hay metrics.jsonl del nuevo)"); continue
        n = min(len(a), len(b))
        iguales = True
        for e in range(n):
            for nombre, get in CAMPOS:
                x, y = get(a[e]), get(b[e])
                marca = "IGUAL" if x == y else f"DISTINTO  delta={x-y:+.3e}"
                if x != y:
                    iguales = False
                print(f"    e{a[e]['epoch']} {nombre:11s} {x!r:24s} vs {y!r:24s}  {marca}")
        veredicto.append(iguales)
        print(f"    -> {'BIT A BIT IGUAL' if iguales else 'NO REPRODUCE'} en {n} epocas")

    if not veredicto:
        print("\nSin datos todavia."); return 2
    if all(veredicto):
        print("\nVEREDICTO: el dato REPRODUCE. Los recorridos que suman semillas son validos.")
        return 0
    print("\nVEREDICTO: el dato NO reproduce. Es OTRO dataset: renombrar y rehacer\n"
          "  ov-sig, bp-sig y pl-f2-* enteros sobre el nuevo (no se pueden sumar).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
