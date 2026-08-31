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

`dropout` (C, 0.0 = off) sits on the flattened features just before the head —
regularisation from inside the net, the sibling of D's `weight_decay`.

Only imports fv.fovea (contract (7)): the net does not know A exists.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fv.fovea import (EDGE_MODES, EDGE_SIDES, GEOMETRY_FIELDS, REGIONS,
                      FoveaError, build_masks, dims_of, is_single_region,
                      n_edge_features, normalize_geometry)

DEFAULT_CHANNEL = 16  # D-C2: a derived net defaults to [16]*n_layers (constant 16)

# REGIONS ('split' | 'single') is defined ONCE, in fv.fovea, and re-exported here
# because that is where C's vocabulary is read from. 'split' is the foveated net
# (two masked branches); 'single' is ONE unmasked branch over the whole N x N
# input — the flat CNN of protocolo.md §6, built as a declared degeneration of C
# and not as a separate architecture (F12, docs/plan-cnn-plana.md §2). The default
# keeps every artefact already on disk meaning exactly what it meant.
__all__ = ["REGIONS", "EDGE_MODES", "EDGE_SIDES", "NETWORK_DEFAULTS",
           "full_config", "build_model", "network_trace", "resolve_channels",
           "FoveatedRegionalNN"]

# The geometry is stated in REAL PIXELS (fv.fovea): the fovea and the border are
# independent lengths, and `border_reduce` is the reduction method's factor. The
# defaults reproduce EXACTLY the pre-2026-08-25 base (N=20, c_frac=0.8, d=2,
# pen_frac=0.1): fovea 16 px, border 2 cells of 2 px = 4 px, 2 px of overlap.
NETWORK_DEFAULTS = {
    "fovea_px": 16, "border_px": 4, "border_reduce": 2,
    "overlap_fovea_px": 2, "overlap_border_px": 0,
    "n_layers": 2,
    "k_center": 3, "k_periph": 3, "s_center": 1, "s_periph": 1,
    "channels": None, "merge": "concat", "pool_mode": "avg",
    "pad_mode": "edge", "regions": "split",
    # Regularisation inside C (the sibling of D's weight_decay). 0.0 = OFF, and
    # OFF is the default because every artefact on disk was trained without it:
    # a non-zero default would silently change what every stored config means.
    # nn.Dropout with p=0.0 is the identity in both train and eval, so a net
    # built with the default is bit-identical to one built before this field
    # existed -- the module has no parameters, so checkpoints keep loading
    # strict (tested).
    "dropout": 0.0,
    # Extra head inputs about the IMAGE edge (fv.fovea.EDGE_MODES). They skip the
    # conv branches entirely and are concatenated to the flattened features right
    # before the Linear -- see `forward` for why that is the only place they can
    # go. 'off' is the default for the same reason `dropout` is 0.0: the net
    # built from a stored config has to keep meaning what it meant, and with 0
    # extra inputs the head has exactly the shape it had (checkpoints load
    # strict, the forward is bit-identical -- tested).
    "edge_inputs": "off",
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
    """Every field C needs, with defaults filled in — and the geometry in the
    canonical spelling whatever spelling `cfg` arrived in.

    The normalisation happens BEFORE the defaults are applied, and that order is
    the whole point: a legacy config carries N/c_frac/pen_frac, which are not
    keys of NETWORK_DEFAULTS any more, so applying the defaults first would drop
    its geometry and silently substitute the default one. Same fact, two places,
    resolved by precedence instead of detection — the failure this project keeps
    paying for. `fv.fovea.normalize_geometry` refuses ambiguity instead.
    """
    out = dict(NETWORK_DEFAULTS)
    geom = normalize_geometry(cfg)
    rest = {k: v for k, v in cfg.items()
            if k in NETWORK_DEFAULTS and k not in GEOMETRY_FIELDS}
    out.update(rest)
    out.update(geom)
    out["channels"] = resolve_channels(cfg, out["n_layers"])  # always a list
    return out


class FoveatedRegionalNN(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        cfg = full_config(cfg)
        self.cfg = cfg
        self.single = is_single_region(cfg)
        dims = dims_of(cfg)
        self.dims = dims
        kc, kp = cfg["k_center"], cfg["k_periph"]
        pc, pp = kc // 2, kp // 2
        channels = cfg["channels"]  # length == n_layers (D-C3)

        # L conv layers per branch, same kernel in all of them. The branch stride
        # goes on the FIRST layer only; the rest are stride 1 (D-S1), so the total
        # subsampling is `s` regardless of depth -> n_layers stays out of stride_range.
        self.center_convs = self._make_branch(channels, kc, pc, cfg["s_center"])
        # 'single': no second branch and NO MASKS AT ALL — build_masks is never
        # called, so the geometry every other domain imports stays untouched. The
        # module names of 'split' are unchanged, so existing checkpoints load.
        if not self.single:
            self.periph_convs = self._make_branch(channels, kp, pp, cfg["s_periph"])
            cm, pm = build_masks(dims)
            self.register_buffer("center_mask", torch.from_numpy(cm)[None, None])
            self.register_buffer("periph_mask", torch.from_numpy(pm)[None, None])

        flat = self._infer_flat_features()
        self.flat_features = flat
        # Dropout goes on the FLATTENED features, right before the head: that is
        # where 97% of the parameters live (measured, plan-40h.md §2), so it is
        # the only place regularisation has anything to bite on. It is NOT put
        # between conv layers: those hold ~3% of the parameters, and dropping
        # whole activations of a small spatial map would mostly add noise to a
        # position head that has to say WHERE a corner is.
        #
        # It is a real module and not F.dropout(self.training) so that .eval()
        # governs it through the one switch the training loop already flips,
        # and so `p` is visible in repr() and in the module tree.
        self.drop = nn.Dropout(float(cfg["dropout"]))
        # The image-edge inputs widen the head and NOTHING else: they are not a
        # channel, they never reach a convolution. Two reasons, and the second is
        # the one that matters:
        #  - they are not spatial. "there is no more image above this window" is
        #    one number about the whole view; as a channel it would be the same
        #    value painted on N*N cells, and the branches would spend kernels
        #    re-deriving a constant.
        #  - a conv branch is MASKED by region. The centre branch cannot see the
        #    ring at all, so an edge signal entering through the input would be
        #    invisible to exactly the branch that predicts the corners.
        # `flat + 0` when off, so the head is bit-identical to the old one.
        self.edge_inputs = str(cfg["edge_inputs"])
        self.n_edge = n_edge_features(self.edge_inputs)
        self.head = nn.Linear(flat + self.n_edge, 12)  # 4 corners x [exists, x, y]

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

    def _branches(self, x: torch.Tensor) -> dict:
        """Each branch's output map, keyed by branch name. A DICT and not a pair:
        'single' has one branch, and every reader (introspection, the trace, the
        UI) iterates the keys it is given instead of assuming there are two."""
        if self.single:
            # the whole input, unmasked: the flat-CNN control
            return {"single": self._branch_forward(self.center_convs, x)}
        # option A: mask the input, then convolve — strides act on data already
        # separated by region, and masks stay N x N.
        return {"center": self._branch_forward(self.center_convs, x * self.center_mask),
                "periph": self._branch_forward(self.periph_convs, x * self.periph_mask)}

    def _merge(self, outs: dict) -> torch.Tensor:
        if self.single:
            return outs["single"].flatten(1)
        if self.cfg["merge"] == "sum":
            return (outs["center"] + outs["periph"]).flatten(1)
        return torch.cat([outs["center"].flatten(1), outs["periph"].flatten(1)], dim=1)

    def _infer_flat_features(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.dims.N, self.dims.N)
            return int(self._merge(self._branches(dummy)).shape[1])

    def forward(self, x: torch.Tensor, edge: torch.Tensor | None = None) -> torch.Tensor:
        feat = self._merge(self._branches(x))
        # after the ReLU, so what is zeroed are the activations the head reads;
        # with dropout=0.0 (the default) this is the identity in both modes and
        # the forward is bit-identical to the pre-dropout one (tested).
        feat = self.drop(F.relu(feat))
        if self.n_edge:
            feat = torch.cat([feat, self._edge_batch(edge, x)], dim=1)
        out = self.head(feat)
        return out.view(-1, 4, 3)

    def _edge_batch(self, edge: "torch.Tensor | None", x: torch.Tensor) -> torch.Tensor:
        """The edge vector, checked. It joins AFTER the ReLU and AFTER the
        dropout, on purpose:
          - not through the ReLU, because the head should read the number as it
            is. It happens to be >= 0 today, so the ReLU would be the identity --
            and a mode added later that signs the sides would silently lose half
            its range against a clamp nobody remembered was there.
          - not through the dropout, because it is a MEASUREMENT, not a learned
            activation. Zeroing it at random does not regularise 4 inputs, it
            tells the head "no edge here" on a window that has one -- noise with
            the shape of a fact.
        """
        if edge is None:
            raise FoveaError(
                "edge_inputs_missing",
                f"la red se construyo con edge_inputs='{self.edge_inputs}' y el "
                f"forward no recibio el vector de borde",
                "pasa model(x, edge) con fv.fovea.edge_features(...); un vector "
                "de ceros NO es el defecto: significaria 'no hay borde por "
                "ningun lado', que es falso justo en las ventanas que lo tienen")
        if edge.ndim != 2 or edge.shape[0] != x.shape[0] or edge.shape[1] != self.n_edge:
            raise FoveaError(
                "edge_inputs_shape",
                f"el vector de borde es {tuple(edge.shape)} y la red espera "
                f"({x.shape[0]}, {self.n_edge})",
                f"un valor por lado {list(EDGE_SIDES)} y por muestra del lote")
        # the dataloader hands float32; `x` carries the dtype/device the branches
        # ran in, so the concat never silently upcasts or crosses devices
        return edge.to(device=x.device, dtype=x.dtype)

    # ------------------------------------------------------------------
    # introspection (V1/V2): per-branch, in_channels=1 per branch means the
    # first-layer kernels are exact and interpretable in both.

    def kernels(self) -> dict:
        # first-layer kernels: in_channels=1 per branch keeps them exact (V1).
        if self.single:
            return {"single": self.center_convs[0].weight.detach().cpu().numpy()[:, 0]}
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
            if self.single:
                return {"single": self._branch_maps(self.center_convs, x)}
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
        outs = model._branches(dummy)
    n_params = sum(int(np.prod(t.shape)) for t in model.state_dict().values())
    return {
        "dims": dims.as_dict(),
        "regions": cfg["regions"],
        # keyed by the branches this net ACTUALLY has: 'single' brings one
        "branch_out": {b: list(t.shape[2:]) for b, t in outs.items()},
        "flat_features": model.flat_features,
        # what reaches the head, split into where it came from: the branches
        # produce `flat_features`, the image edge adds `edge_features` straight
        # in. Reported apart because "the head grew" and "the net saw more
        # pixels" are different facts and the param count alone conflates them.
        "edge_inputs": cfg["edge_inputs"],
        "edge_features": model.n_edge,
        "head_inputs": model.flat_features + model.n_edge,
        "num_params": n_params,
    }
