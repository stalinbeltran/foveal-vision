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

DEFAULT_CHANNEL = 16  # D-C2: a derived net defaults to [16]*n_layers (constant 16)

NETWORK_DEFAULTS = {
    "N": 20, "c_frac": 0.8, "d": 2, "pen_frac": 0.1, "n_layers": 2,
    "k_center": 3, "k_periph": 3, "s_center": 1, "s_periph": 1,
    "channels": None, "merge": "concat", "pool_mode": "avg",
    "pad_mode": "edge",
}


def resolve_channels(cfg: dict, n_layers: int) -> list[int]:
    """The per-layer channel vector (D-C3). Precedence: an explicit `channels`
    list wins; else the legacy scalar `ch1/ch2` maps to `[ch1, ch2]` (read old,
    write channels); else the default `[16]*n_layers` (D-C2)."""
    if cfg.get("channels") is not None:
        return [int(c) for c in cfg["channels"]]
    if "ch1" in cfg or "ch2" in cfg:
        return [int(cfg.get("ch1", DEFAULT_CHANNEL)),
                int(cfg.get("ch2", DEFAULT_CHANNEL))]
    return [DEFAULT_CHANNEL] * int(n_layers)


def full_config(cfg: dict) -> dict:
    out = dict(NETWORK_DEFAULTS)
    out.update({k: v for k, v in cfg.items() if k in NETWORK_DEFAULTS})
    out["channels"] = resolve_channels(cfg, out["n_layers"])  # always a list
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
        channels = cfg["channels"]  # length == n_layers (D-C3)

        # L conv layers per branch, same kernel in all of them. The branch stride
        # goes on the FIRST layer only; the rest are stride 1 (D-S1), so the total
        # subsampling is `s` regardless of depth -> n_layers stays out of stride_range.
        self.center_convs = self._make_branch(channels, kc, pc, cfg["s_center"])
        self.periph_convs = self._make_branch(channels, kp, pp, cfg["s_periph"])

        cm, pm = build_masks(dims)
        self.register_buffer("center_mask", torch.from_numpy(cm)[None, None])
        self.register_buffer("periph_mask", torch.from_numpy(pm)[None, None])

        flat = self._infer_flat_features()
        self.flat_features = flat
        self.head = nn.Linear(flat, 12)  # 4 corners x [exists, x, y]

    @staticmethod
    def _make_branch(channels: list[int], k: int, pad: int, stride: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        in_ch = 1  # the masked composite image is one channel per branch
        for i, out_ch in enumerate(channels):
            s = stride if i == 0 else 1
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad))
            in_ch = out_ch
        return layers

    @staticmethod
    def _branch_forward(convs: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        # ReLU BETWEEN layers, none after the last (the last map stays
        # pre-activation, as conv2 did — introspection reads it signed, V2).
        for i, conv in enumerate(convs):
            x = conv(x)
            if i < len(convs) - 1:
                x = F.relu(x)
        return x

    def _branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # option A: mask the input, then convolve — strides act on data already
        # separated by region, and masks stay N x N.
        c = self._branch_forward(self.center_convs, x * self.center_mask)
        p = self._branch_forward(self.periph_convs, x * self.periph_mask)
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
        # first-layer kernels: in_channels=1 per branch keeps them exact (V1).
        return {
            "center": self.center_convs[0].weight.detach().cpu().numpy()[:, 0],
            "periph": self.periph_convs[0].weight.detach().cpu().numpy()[:, 0],
        }

    def _branch_maps(self, convs: nn.ModuleList, x: torch.Tensor) -> list:
        # V1/V2 for the first and last conv (layers >1 are optional this phase,
        # barrido-por-ejes.md §3.3): [L1 post-ReLU, last pre-activation]. Always
        # two maps, so the payload never breaks — for n_layers=2 it is bit-identical.
        first = None
        for i, conv in enumerate(convs):
            x = conv(x)
            if i == 0:
                first = F.relu(x)[0].cpu().numpy()
            if i < len(convs) - 1:
                x = F.relu(x)
        last = x[0].cpu().numpy()
        return [first, last]

    def feature_maps(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            return {"center": self._branch_maps(self.center_convs, x * self.center_mask),
                    "periph": self._branch_maps(self.periph_convs, x * self.periph_mask)}


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
