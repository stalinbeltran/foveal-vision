"""C — the foveated regional NN (instructionsNewNN.md §6, head per C9).

Two independent conv branches (centre / periphery) over the composite N x N
input; masks are applied to the INPUT (option A — masking after convolution
was rejected: reconstructing masks at output resolution is fragile). In the
penetration band both masks are 1, so both branches contribute.

The head is the corner head (C9), NOT the reference classifier of the spec:
4 corners x [exists, x, y] over the flattened branch features — the reference
adaptive_avg_pool2d(feat, 1) destroys the "where" a position head predicts.
merge: 'concat' flattens both branches and concatenates (tolerates different
strides); 'sum' adds aligned feature maps first (validator enforces equal
strides).

Only imports fv.fovea (contract (7)): the net does not know A exists.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fv.fovea import build_masks, derive_dims

NETWORK_DEFAULTS = {
    "N": 20, "c_frac": 0.8, "d": 2, "pen_frac": 0.1, "n_layers": 2,
    "k_center": 3, "k_periph": 3, "s_center": 1, "s_periph": 1,
    "ch1": 16, "ch2": 32, "merge": "concat", "pool_mode": "avg",
    "pad_mode": "edge",
}


def full_config(cfg: dict) -> dict:
    out = dict(NETWORK_DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k in NETWORK_DEFAULTS})
    return out


class FoveatedRegionalNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        cfg = full_config(cfg)
        self.cfg = cfg
        dims = derive_dims(cfg["N"], cfg["c_frac"], cfg["d"], cfg["pen_frac"])
        self.dims = dims
        kc, kp = cfg["k_center"], cfg["k_periph"]
        pc, pp = kc // 2, kp // 2
        ch1, ch2 = cfg["ch1"], cfg["ch2"]

        self.center_conv1 = nn.Conv2d(1, ch1, kc, stride=cfg["s_center"], padding=pc)
        self.center_conv2 = nn.Conv2d(ch1, ch2, kc, stride=1, padding=pc)
        self.periph_conv1 = nn.Conv2d(1, ch1, kp, stride=cfg["s_periph"], padding=pp)
        self.periph_conv2 = nn.Conv2d(ch1, ch2, kp, stride=1, padding=pp)

        cm, pm = build_masks(dims)
        self.register_buffer("center_mask", torch.from_numpy(cm)[None, None])
        self.register_buffer("periph_mask", torch.from_numpy(pm)[None, None])

        flat = self._infer_flat_features()
        self.flat_features = flat
        self.head = nn.Linear(flat, 12)  # 4 corners x [exists, x, y]

    def _branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # option A: mask the input, then convolve — strides act on data already
        # separated by region, and masks stay N x N.
        xc = x * self.center_mask
        xp = x * self.periph_mask
        c = self.center_conv2(F.relu(self.center_conv1(xc)))
        p = self.periph_conv2(F.relu(self.periph_conv1(xp)))
        return c, p

    def _infer_flat_features(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.dims.N, self.dims.N)
            c, p = self._branches(dummy)
            if self.cfg["merge"] == "sum":
                return int((c + p).flatten(1).shape[1])
            return int(c.flatten(1).shape[1] + p.flatten(1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, p = self._branches(x)
        if self.cfg["merge"] == "sum":
            feat = (c + p).flatten(1)
        else:
            feat = torch.cat([c.flatten(1), p.flatten(1)], dim=1)
        out = self.head(F.relu(feat))
        return out.view(-1, 4, 3)

    # ------------------------------------------------------------------
    # introspection (V1/V2): per-branch, in_channels=1 per branch means the
    # first-layer kernels are exact and interpretable in both.

    def kernels(self) -> dict:
        return {
            "center": self.center_conv1.weight.detach().cpu().numpy()[:, 0],
            "periph": self.periph_conv1.weight.detach().cpu().numpy()[:, 0],
        }

    def feature_maps(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            xc = x * self.center_mask
            xp = x * self.periph_mask
            c1 = F.relu(self.center_conv1(xc))
            c2 = self.center_conv2(c1)
            p1 = F.relu(self.periph_conv1(xp))
            p2 = self.periph_conv2(p1)
        return {"center": [c1[0].cpu().numpy(), c2[0].cpu().numpy()],
                "periph": [p1[0].cpu().numpy(), p2[0].cpu().numpy()]}


def build_model(cfg: dict) -> FoveatedRegionalNN:
    return FoveatedRegionalNN(cfg)


def network_trace(cfg: dict) -> dict:
    """Derived dims, per-branch spatial trace and param count — no weights
    needed; feeds POST /networks/validate and the Redes screen live."""
    cfg = full_config(cfg)
    model = build_model(cfg)
    dims = model.dims
    with torch.no_grad():
        dummy = torch.zeros(1, 1, dims.N, dims.N)
        c, p = model._branches(dummy)
    n_params = sum(int(np.prod(t.shape)) for t in model.state_dict().values())
    return {
        "dims": dims.as_dict(),
        "branch_out": {"center": list(c.shape[2:]), "periph": list(p.shape[2:])},
        "flat_features": model.flat_features,
        "num_params": n_params,
    }
