#!/usr/bin/env python3
"""Cuanto tarda ESTA maquina en dar un paso del entrenamiento que va a correr.

Por que existe
--------------
Alquilar por precio y numero de nucleos elige mal. MEDIDO el 2026-08-23
(docs/plan-lr-alto.md §6.3): la maquina de 16 vCPU fue la MAS LENTA (53,3
s/epoca) y la de 9,3 vCPU la mas rapida (36,3) -- un factor 1,47 que el catalogo
no anuncia y que se paga entero en reloj y en dinero. La conclusion de aquel
informe fue literal: *"la eleccion de oferta filtra hoy por numero de nucleos y
precio, que resulta ser un mal criterio para este trabajo"*.

La forma barata de arreglarlo es no adivinar: alquilar de mas, PREGUNTARLE a
cada maquina cuanto tarda -- unos segundos-- y quedarse con las rapidas.

Que mide, y por que asi
-----------------------
Pasos de entrenamiento DE VERDAD: el modelo del recorrido, la receta del
recorrido, el `batch_size` del recorrido y el `windows.npz` del recorrido. No un
micro-benchmark de matrices. Un proxy sintetico mide otra cosa (BLAS puro, sin
dataloader ni construccion del compuesto foveado, que aqui es una parte real del
coste: plan-40h.md midio ~35 s fijos de dataloader por epoca) y elegiria por el
motivo equivocado.

Se mide en dos tramos y se reportan los dos:

- `calentamiento`: los primeros pasos, que incluyen la asignacion de memoria y
  el primer toque de cada tensor. Se DESCARTAN del numero final.
- `medido`: los siguientes. La mediana, no la media: en una maquina compartida
  un paso puede irse al triple por una interrupcion ajena, y la media se lo come
  entero.

⚠ Lo que este numero NO es: una prediccion exacta de `seconds_per_epoch`. La
epoca tambien incluye la validacion (28.000 ventanas) y el barrido del
dataloader. `s_epoca_estimada` extrapola con los pasos de entrenamiento y el
tramo fijo que se le pase; sirve para ORDENAR maquinas, que es para lo que se
usa, no para presupuestar.

    .venv/bin/python scripts/sonda_velocidad.py --sweep bs-L4 --punto 0
    .venv/bin/python scripts/sonda_velocidad.py --sweep bs-L4 --punto 0 --segundos 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--punto", type=int, default=0,
                    help="indice global del punto con cuya config se mide")
    ap.add_argument("--segundos", type=float, default=8.0,
                    help="cuanto medir COMO MUCHO (el corte real es --pasos)")
    ap.add_argument("--pasos", type=int, default=12, help="pasos a cronometrar")
    ap.add_argument("--calentamiento", type=int, default=3,
                    help="pasos que se tiran antes de cronometrar")
    args = ap.parse_args()

    t_arranque = time.monotonic()
    import dataclasses

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from fv.fovea import dims_of
    from fv.models.builder import build_model
    from fv.sweeps.spec import expand_points
    from fv.sweeps.store import SweepStore
    from fv.training.loop import make_optimizer
    from fv.training.losses import corner_loss
    from fv.training.recipe import Recipe
    from fv.windows.dataset import FoveatedWindowDataset
    from fv.windows.store import WindowDatasetStore

    store = SweepStore()
    spec = store.spec(args.sweep)
    valid, _ = expand_points(spec, spec["base_network_value"])
    if not 0 <= args.punto < len(valid):
        print(json.dumps({"ok": False, "error":
                          f"el recorrido tiene {len(valid)} puntos y se pidio {args.punto}"}))
        return 2
    punto = valid[args.punto]
    receta = dataclasses.replace(Recipe(**spec["base_recipe_value"]),
                                 **punto["recipe_overrides"])

    wstore = WindowDatasetStore()
    manifest = wstore.manifest(spec["window_dataset"])
    arrays = wstore.arrays(spec["window_dataset"])
    net = punto["network"]
    ds = FoveatedWindowDataset(arrays, dims_of(net), split=0,
                               pool_mode=net["pool_mode"], pad_mode=net["pad_mode"])
    g = torch.Generator()
    g.manual_seed(int(receta.seed))
    loader = DataLoader(ds, batch_size=receta.batch_size, shuffle=True,
                        num_workers=0, generator=g)

    torch.manual_seed(int(receta.seed))
    model = build_model(net)
    opt = make_optimizer(model, receta)
    model.train()

    tiempos: list[float] = []
    t_inicio = time.monotonic()
    for i, (x, y) in enumerate(loader):
        t0 = time.monotonic()
        opt.zero_grad()
        loss = corner_loss(model(x), y, receta.lambda_pos, receta.pos_weight,
                           receta.smooth_l1_beta)
        loss.backward()
        opt.step()
        dt = time.monotonic() - t0
        if i >= args.calentamiento:
            tiempos.append(dt)
        # el corte por tiempo es una RED, no el criterio: una maquina tan lenta
        # que no da ni los pasos pedidos tiene que devolver algo igualmente, y
        # ese "algo" es justo lo que la descarta
        if len(tiempos) >= args.pasos or (time.monotonic() - t_inicio) > args.segundos:
            break

    if not tiempos:
        print(json.dumps({"ok": False, "error": "no dio ni un paso cronometrable",
                          "segundos": round(time.monotonic() - t_inicio, 2)}))
        return 1

    s_paso = statistics.median(tiempos)
    pasos_epoca = int(np.ceil(len(ds) / receta.batch_size))
    salida = {
        "ok": True,
        "sweep": args.sweep, "punto": args.punto,
        "batch_size": receta.batch_size, "n_layers": net["n_layers"],
        "pasos_medidos": len(tiempos),
        "s_paso": round(s_paso, 5),
        "s_paso_min": round(min(tiempos), 5),
        "s_paso_max": round(max(tiempos), 5),
        "pasos_s": round(1.0 / s_paso, 3) if s_paso else None,
        "pasos_epoca": pasos_epoca,
        # extrapolacion DECLARADA: solo el tramo de entrenamiento, sin validacion
        "s_epoca_estimada": round(s_paso * pasos_epoca, 1),
        "hilos_torch": torch.get_num_threads(),
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "cpu": _cpu(),
        "nucleos": os.cpu_count(),
        "carga": _carga(),
        "arranque_s": round(t_inicio - t_arranque, 1),
        "ventanas_train": len(ds),
        "fingerprint": manifest.get("fingerprint"),
    }
    print(json.dumps(salida, ensure_ascii=False))
    return 0


def _cpu() -> str:
    try:
        for linea in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if linea.lower().startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def _carga():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
