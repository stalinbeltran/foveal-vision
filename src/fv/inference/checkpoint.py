"""Contract (4): the checkpoint describes itself — load_model rebuilds the net
(foveated geometry included) without any YAML or dataset."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from fv.models.builder import FoveatedRegionalNN, build_model


def load_model(ckpt_path: Path, device: str = "cpu") -> FoveatedRegionalNN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
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
