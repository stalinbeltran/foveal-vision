#!/usr/bin/env python3
"""¿Dónde falla la red respecto al BORDE de la imagen? Recall y error por tramo.

Por qué existe
--------------
`pad_mode: edge` replica la fila del borde cuando el recorte se sale de la
imagen, y esa réplica es POR CONSTRUCCIÓN indistinguible de imagen real que
sigue. La consecuencia se ve sólo si se desglosa: el f1 global la esconde,
porque el caso es el 3 % de las esquinas.

*Medido el 2026-09-01 con `demo-fov16-optimo` sobre el val de
`dirty1000-80px-16px-r20260827`: recall 0,608 a ≤1 px del borde contra 0,939 a
más de 8 px.* Ése es el número que motivó `mask_channel`.

⚠ Vivía en /tmp y produce el veredicto de un plan. Lo que queda en /tmp se
pierde con la máquina, así que está aquí.

    .venv/bin/python scripts/error_por_borde.py demo-fov16-optimo fov16-mask-p20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv import settings                                        # noqa: E402
from fv.fovea import build_view, edge_features, input_stack     # noqa: E402
from fv.inference import catalogo                               # noqa: E402
from fv.inference.checkpoint import MODEL_CACHE                 # noqa: E402
from fv.metrics import corner_scores, detection_counts          # noqa: E402
from fv.training.registry import RunStore                       # noqa: E402
from fv.windows.store import WindowDatasetStore                 # noqa: E402

SPLITS = {"train": 0, "val": 1, "test": 2}
TRAMOS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 10_000))


def analizar(nombre: str, ds: str, split: str, runs) -> dict:
    ck, _ = catalogo.checkpoint_de(nombre, runs)
    if ck is None:
        raise SystemExit(f"'{nombre}' no tiene pesos utilizables "
                         f"(ni antesala ni catálogo aprobado)")
    model = MODEL_CACHE.get(ck)
    dims, cfg = model.dims, model.cfg
    n = dims.fovea_px
    arr = WindowDatasetStore(settings.window_datasets_root()).arrays(ds)
    y, wxy, imgs = arr["y"], arr["window_xy"], arr["images"]
    fila = {int(a): i for i, a in enumerate(arr["images_sample_idx"])}
    sel = np.where(arr["split"] == SPLITS[split])[0]
    H, W = imgs.shape[1:]

    entradas, bordes = [], []
    for j in sel:
        wx0, wy0 = int(wxy[j, 0]), int(wxy[j, 1])
        im = imgs[fila[int(arr["sample_idx"][j])]]
        v, cov = build_view(im, wx0, wy0, dims, pool_mode=cfg["pool_mode"],
                            pad_mode=cfg["pad_mode"])
        # ⚠ por `input_stack` y NO apilando a mano: es la MISMA función que usan
        # el dataloader y `predict_image` (contrato (5)). La primera versión de
        # este script armaba la entrada por su cuenta y reventó en cuanto llegó
        # el segundo canal -- el fallo ruidoso, por suerte.
        entradas.append(input_stack(v, cov, cfg.get("mask_channel", "off")))
        bordes.append(edge_features(im.shape, wx0, wy0, dims, cfg["edge_inputs"]))
    x = torch.from_numpy(np.stack(entradas))
    e = torch.from_numpy(np.stack(bordes))
    with torch.no_grad():
        out = np.concatenate([model(x[i:i + 2048], e[i:i + 2048]).numpy()
                              for i in range(0, len(x), 2048)])

    sc, ex = corner_scores(out), y[sel][:, :, 0]
    err = np.sqrt((((out[:, :, 1:] - y[sel][:, :, 1:]) * n) ** 2).sum(-1))
    ax = wxy[sel][:, None, 0] + y[sel][:, :, 1] * n
    ay = wxy[sel][:, None, 1] + y[sel][:, :, 2] * n
    dist = np.minimum.reduce([ax, ay, W - ax, H - ay])
    m = dims.border_px
    conPad = ((wxy[sel][:, 0] < m) | (wxy[sel][:, 1] < m)
              | (wxy[sel][:, 0] + n > W - m) | (wxy[sel][:, 1] + n > H - m))

    tot = detection_counts(sc, ex)
    r = {"run": nombre, "edge_inputs": cfg["edge_inputs"],
         "mask_channel": cfg.get("mask_channel", "off"),
         "global": {"f1": tot["f1"], "recall": tot["recall"],
                    "err_px": float(np.nanmean(np.where(ex >= 0.5, err, np.nan)))},
         "por_relleno": {}, "por_distancia": {}}
    for etq, mv in (("sin_relleno", ~conPad), ("con_relleno", conPad)):
        d = detection_counts(sc[mv], ex[mv])
        r["por_relleno"][etq] = {
            "ventanas": int(mv.sum()), "f1": d["f1"], "recall": d["recall"],
            "err_px": float(np.nanmean(np.where(ex[mv] >= 0.5, err[mv], np.nan)))}
    for lo, hi in TRAMOS:
        k = (ex >= 0.5) & (dist > (lo if lo else -1)) & (dist <= hi)
        if not k.any():
            continue
        r["por_distancia"][f"{lo}-{hi}"] = {
            "esquinas": int(k.sum()), "recall": float((sc[k] >= 0.5).mean()),
            "err_px": float(np.nanmean(err[k]))}
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dataset", default="dirty1000-80px-16px-r20260827")
    ap.add_argument("--split", default="val", choices=sorted(SPLITS))
    args = ap.parse_args()
    store = RunStore(settings.runs_root())

    filas = [analizar(n, args.dataset, args.split, store) for n in args.runs]
    print(f"\n{args.dataset} · split {args.split}\n")
    cab = f"{'run':22s} {'edge':5s} {'mask':9s} {'f1':>7s} {'err':>6s}"
    tramos = list(filas[0]["por_distancia"])
    cab += "".join(f"{('rec ' + t):>10s}" for t in tramos)
    print(cab)
    print("-" * len(cab))
    for r in filas:
        linea = (f"{r['run']:22s} {r['edge_inputs']:5s} {r['mask_channel']:9s} "
                 f"{r['global']['f1']:7.4f} {r['global']['err_px']:6.3f}")
        linea += "".join(f"{r['por_distancia'][t]['recall']:10.4f}" for t in tramos)
        print(linea)
    print(f"\n  (esquinas por tramo: "
          + " · ".join(f"{t}px n={filas[0]['por_distancia'][t]['esquinas']}"
                       for t in tramos) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
