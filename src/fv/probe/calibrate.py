"""Calibrate lambda PER CELL, instead of sweeping it as an axis.

WHY (owner's review, 2026-09-02)
--------------------------------
`lambda` does not mean the same thing in every cell. The measured map
lambda -> activation comes from ONE point (k=7, K=16); at k=9 with K=32 the same
lambda lands somewhere else, because both the atom size and the overcompleteness
change. Sweeping a fixed lambda grid AND then filtering by activation band (E2)
can leave cells -- or a whole `k` -- with no admissible combination, and the axis
that carries the premise of the experiment is the one that ends up with holes.

So: bisect lambda in each cell until the activation hits the target, record the
resulting lambda as a DATUM of the run, and let `lambda` stop being an axis. The
sparsity is then constant across cells and the sweep measures what it claims to.
That is 2 values (`0` = control, `calibrada`) instead of 4 or 5.

THREE THINGS THAT MAKE THIS HONEST
----------------------------------
1. **It bisects in log(lambda)**, because the effect is multiplicative: measured
   on 2026-09-02, going from 0.03 to 3.0 (100x) moved the activation from 44.9 %
   to 22.3 %, and the useful range spans two decades. A linear bisection would
   spend every evaluation in the flat part.
2. **It reports whether it LANDED IN BAND**, and the caller stores that. A
   calibration that ran out of evaluations and returns its best attempt must not
   be indistinguishable from one that converged -- that difference is exactly
   what would make a cell incomparable without anyone noticing.
3. **The activation is measured on the SAME quantity the criterion reads**
   (fraction of positions with z > 0 over validation), not on a proxy.
"""

from __future__ import annotations

import math
import time

import torch

from fv.probe.run import _val_pass, fit

OBJETIVO = 0.10          # 10 % de activacion...
TOLERANCIA = 0.03        # ...+-3, la banda que pidio el dueno

# Si un x8 en lambda mueve la activacion MENOS que esto, la celda ha tocado su
# suelo: seguir subiendo solo destruye la reconstruccion. Medido en k=3/K=8.
SATURA = 0.01
# Dos lambdas cuya activacion difiere menos que esto empatan, y entre las que
# empatan gana la menor.
EMPATE = 0.01


def _activacion(data: dict, channels: int, k: int, lam: float, seed: int,
                epochs: int, batch: int, lr: float, val_log: int) -> float:
    m, _, _ = fit(data, channels, k, lam, seed, epochs, batch, lr,
                  out_dir=None, val_log=val_log, verbose=False)
    va = torch.from_numpy(data["val"])
    _, act = _val_pass(m, va[:val_log], data["var"], batch)
    return act


def calibrate_lambda(data: dict, channels: int, k: int, *, seed: int = 1,
                     objetivo: float = OBJETIVO, tolerancia: float = TOLERANCIA,
                     epochs: int = 2, batch: int = 256, lr: float = 3e-3,
                     val_log: int = 4096, max_evals: int = 7,
                     lam0: float = 10.0, verbose: bool = True) -> dict:
    """Bisect lambda until activation is `objetivo` +- `tolerancia`.

    `lam0` = 10.0 is not arbitrary: it is where the measured scan put ~12.9 %
    activation at k=7/K=16, so the search starts one step from the answer in the
    only cell where the answer is known.
    """
    t0 = time.time()
    evals: list[dict] = []

    def prueba(lam: float) -> float:
        a = _activacion(data, channels, k, lam, seed, epochs, batch, lr, val_log)
        evals.append({"lambda": lam, "activa": a})
        if verbose:
            print(f"    [calibrar k={k} K={channels}] λ={lam:<9.4g} -> activa {a*100:.1f} %")
        return a

    # 1. acotar: la activacion BAJA con lambda, asi que se expande en la
    #    direccion que haga falta hasta tener un intervalo que contenga el
    #    objetivo. Sin esto, una celda cuyo lambda util caiga fuera del intervalo
    #    inicial devolveria un extremo sin decir que no convergio.
    #
    # ⚠ Y se para si SATURA. Medido el 2026-09-02 en k=3/K=8: la activacion se
    #    queda en 14,4 % y no baja mas, asi que la expansion siguio multiplicando
    #    hasta λ=2,6e6 -- un valor absurdo, que aplasta la reconstruccion, por
    #    0,6 puntos de activacion frente a λ=640. Hay celdas con SUELO de
    #    activacion, y una busqueda que no lo detecta convierte "esta celda no
    #    admite esa esparsidad" en "esta celda usa un λ de dos millones".
    lo = hi = lam0
    a = prueba(lam0)
    a_lo = a_hi = a
    saturado = False
    while len(evals) < max_evals and not (abs(a - objetivo) <= tolerancia):
        previa = a
        if a > objetivo:                     # demasiada activacion -> subir lambda
            lo, a_lo = hi, a_hi
            hi *= 8.0
            a = a_hi = prueba(hi)
        else:                                # demasiado poca -> bajar
            hi, a_hi = lo, a_lo
            lo /= 8.0
            a = a_lo = prueba(lo)
        if abs(a - previa) < SATURA:
            saturado = True
            break
        if (a_lo - objetivo) * (a_hi - objetivo) <= 0:
            break

    # 2. biseccion en log(lambda)
    while (not saturado and len(evals) < max_evals
           and abs(a - objetivo) > tolerancia and hi > lo):
        med = math.sqrt(lo * hi)
        a = prueba(med)
        if a > objetivo:
            lo = med
        else:
            hi = med

    # 3. de entre las que empatan en activacion, gana la lambda MAS PEQUENA. El
    #    objetivo es una esparsidad dada; entre dos lambdas que la consiguen, la
    #    menor distorsiona menos la reconstruccion. Sin esto, k=3/K=8 devolvia
    #    2,6e6 en vez de 640 por 0,6 puntos.
    cerca = min(abs(e["activa"] - objetivo) for e in evals)
    mejor = min((e for e in evals if abs(e["activa"] - objetivo) <= cerca + EMPATE),
                key=lambda e: e["lambda"])
    return {
        "lambda": mejor["lambda"],
        "activa_calibrada": mejor["activa"],
        "en_banda": abs(mejor["activa"] - objetivo) <= tolerancia,
        "saturado": saturado,
        "evaluaciones": evals,
        "n_evaluaciones": len(evals),
        "objetivo": objetivo,
        "tolerancia": tolerancia,
        "segundos_calibracion": round(time.time() - t0, 1),
    }
