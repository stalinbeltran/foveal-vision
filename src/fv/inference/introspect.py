"""V1/V2/F0: kernels and feature maps PER BRANCH, and the input view channel by
channel with its coverage mask (the fundamental debugging view here: a shifted
or transposed mask does not fail — it trains fine with mislabelled context).

Payloads via fv.matrixview; the colour work is DECLARED by the producer:
weights are signed -> diverging centred on 0; post-ReLU activations ->
sequential (conv2 output is pre-activation here -> diverging).
"""

from __future__ import annotations

import numpy as np
import torch

from fv.fovea import EDGE_SIDES, build_view, edge_features
from fv.matrixview import map_payload, maps_payload


def kernels_payload(model) -> dict:
    ks = model.kernels()
    return {
        "branches": {
            branch: maps_payload(stack, color="diverging",
                                 labels=[f"{branch}[{i}]" for i in range(stack.shape[0])])
            for branch, stack in ks.items()
        }
    }


def feature_maps_payload(model, view: np.ndarray) -> dict:
    x = torch.from_numpy(view).float()[None, None]
    fm = model.feature_maps(x)
    out = {}
    for branch, layers in fm.items():
        out[branch] = [
            maps_payload(layers[0], color="sequential",   # post-ReLU: magnitude
                         labels=[f"L1[{i}]" for i in range(layers[0].shape[0])]),
            maps_payload(layers[1], color="diverging",    # pre-activation: signed
                         labels=[f"L2[{i}]" for i in range(layers[1].shape[0])]),
        ]
    return {"branches": out}


def input_view_payload(model, image: np.ndarray, wx0: int, wy0: int) -> dict:
    """F0: the composite the net actually receives + branch masks + coverage."""
    dims = model.dims
    view, coverage = build_view(image, wx0, wy0, dims,
                                pool_mode=model.cfg["pool_mode"],
                                pad_mode=model.cfg["pad_mode"])
    channels = [map_payload(view, "sequential", "vista compuesta (lo que ve la red)")]
    # a 'single' net has NO masks — it convolves the whole input. Drawing a mask
    # there would show a division that does not happen (the flat control's whole
    # point), so the channel is absent, not faked with an all-ones square.
    if not model.single:
        channels.append(map_payload(model.center_mask[0, 0].cpu().numpy(),
                                    "sequential", "mascara rama central"))
        channels.append(map_payload(model.periph_mask[0, 0].cpu().numpy(),
                                    "sequential", "mascara rama periferica"))
    channels.append(map_payload(coverage, "sequential",
                                "cobertura (fraccion real por celda)"))
    return {
        "dims": dims.as_dict(),
        "regions": model.cfg["regions"],
        "channels": channels,
        "coverage_min": float(coverage.min()),
        "coverage_mean": float(coverage.mean()),
        # Lo que esta ventana le dice a la CABEZA sobre el borde de la imagen, si
        # esta red pide que se lo digan. No es un canal: son 4 escalares y no
        # tienen forma espacial. Va aqui porque F0 es "que recibe exactamente la
        # red", y desde que hay una entrada que no se ve en la imagen, una F0 que
        # solo ensene pixeles ensena media entrada.
        "edge_inputs": model.cfg["edge_inputs"],
        "edge": None if not model.n_edge else dict(zip(
            EDGE_SIDES,
            (round(float(v), 4) for v in edge_features(
                image.shape, wx0, wy0, dims, model.cfg["edge_inputs"])))),
    }
