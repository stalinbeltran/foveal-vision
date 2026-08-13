"""Contract (4): the checkpoint describes itself — load_model rebuilds the net
(foveated geometry included) without any YAML or dataset."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from fv.models.builder import FoveatedRegionalNN, build_model


class CheckpointError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def load_model(ckpt_path: Path, device: str = "cpu") -> FoveatedRegionalNN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"])
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        # a checkpoint from a previous builder (e.g. the fixed two-layer conv1/
        # conv2 before the parametric builder) no longer fits — no weight-compat
        # code is written on purpose (D-C2 §13). Fail with the reason, never a 500.
        raise CheckpointError(
            "checkpoint_incompatible",
            "este checkpoint es de un builder anterior y sus pesos ya no encajan "
            "en la red parametrica",
            "reentrena el run (fv-train / un recorrido): no se migra state_dict "
            "(barrido-por-ejes.md §13)") from e
    model.to(device)
    model.eval()
    return model


class ModelCache:
    """Keyed by (path, device, mtime): a live run rewrites best.pt every epoch
    it improves — without the mtime you would serve the first epoch forever."""

    def __init__(self):
        self._cache: dict = {}

    def get(self, ckpt_path: Path, device: str = "cpu") -> FoveatedRegionalNN:
        p = Path(ckpt_path)
        key = (str(p), device, p.stat().st_mtime_ns)
        if key not in self._cache:
            self._cache.clear()  # keep at most one: models are MBs
            self._cache[key] = load_model(p, device)
        return self._cache[key]


MODEL_CACHE = ModelCache()
