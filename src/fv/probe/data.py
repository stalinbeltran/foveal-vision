"""The input to the probe: the SAME 20x20 view production sees, contrast-normalised.

THE TWO DECISIONS THAT WOULD BREAK SILENTLY
-------------------------------------------
1. **The view is built by `fv.fovea.build_view`, not rebuilt here.** That is the
   one import this package takes from `fv.fovea` (see `fv.probe.__init__`): it
   is the same function the dataloader and inference use (contract (5)). If the
   view were assembled again here, the probe would measure a datum the network
   never sees and every conclusion would be about a different image.

2. **Views are PRECOMPUTED once, never built per item.** `build_view` costs
   163 us/window *(measured 2026-09-02 on this droplet, 2 vCPU)*; at 84,000
   windows x 30 epochs x 48 runs that is ~4 h of pure Python re-assembling the
   same view over and over. Precomputed: 14 s and 134 MB.

`var(x)` is a FIXED constant of the train split, computed once and stored in the
summary -- never the batch variance. If it changed per batch, lambda would mean
something different at every step and the sweep over lambda, which is the whole
point of the loss normaliser, would stop being comparable.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fv import settings
from fv.fovea import build_view, dims_of   # the window loader: the one exception

SPLIT_TRAIN, SPLIT_VAL = 0, 1


# ---------------------------------------------------------- contrast normalisation

def _gauss1d(sigma: float, radius: int) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _blur(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Separable gaussian with REPLICATED padding, matching `pad_mode: edge`.

    With zero padding the local mean at the border would collapse and the probe
    would see a ring of contrast the image does not have.
    """
    r = (g.numel() - 1) // 2
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, g.view(1, 1, -1, 1))


def local_contrast_norm(x: torch.Tensor, sigma: float, eps: float) -> torch.Tensor:
    """(x - local mean) / (local sd + eps), sigma in pixels of the view.

    MANDATORY, and not cosmetic: without it the highest-variance component is
    the mean intensity level, and the loss spends its first degrees of freedom
    there. That is literally what produced the duplicated k5/k7 pair of
    `fov16-mask-p20`, both pure negative DC.
    """
    g = _gauss1d(sigma, max(1, int(round(3 * sigma))))
    mu = _blur(x, g)
    var = _blur((x - mu) ** 2, g).clamp_min(0.0)
    return (x - mu) / (var.sqrt() + eps)


# ---------------------------------------------------------------- the views

def views_for_split(arrays: dict, split: int, dims) -> np.ndarray:
    sel = arrays["split"] == split
    wxy = arrays["window_xy"][sel]
    sidx = arrays["sample_idx"][sel]
    imgs = arrays["images"]
    lookup = {int(a): i for i, a in enumerate(arrays["images_sample_idx"])}
    row = np.asarray([lookup[int(s)] for s in sidx], dtype=np.int32)
    out = np.empty((len(row), dims.N, dims.N), dtype=np.float32)
    for i in range(len(row)):
        v, _ = build_view(imgs[row[i]], int(wxy[i, 0]), int(wxy[i, 1]), dims,
                          pool_mode="avg", pad_mode="edge")
        out[i] = v
    return out


def geometry_dims(net: dict):
    """Resolve the production geometry the caller hands in.

    `net` travels IN as a plain dict so this package never imports `fv.models`:
    the single source of truth stays in `builder.NETWORK_DEFAULTS` and the
    script does the wiring.
    """
    return dims_of(dict(net))


def prepare(dataset: str, net: dict, sigma: float, eps: float | None,
            cache_dir: Path, limit: int | None, verbose: bool = True) -> dict:
    """Train and val views, already normalised. Computed ONCE, then cached."""
    dims = geometry_dims(net)
    key = hashlib.sha256(
        json.dumps({"N": dims.N, "orig": dims.original_size, "b": dims.border_px},
                   sort_keys=True).encode()).hexdigest()[:8]
    cache_dir = Path(cache_dir)
    fich = cache_dir / f"{dataset}-{key}-s{sigma}-lim{limit or 0}.npz"
    if fich.exists():
        z = np.load(fich)
        d = {k: z[k] for k in z.files}
        d["eps"] = float(d["eps"])
        d["var"] = float(d["var"])
        if verbose:
            print(f"[datos] cache {fich.name}: train {d['train'].shape} val {d['val'].shape}")
        return d

    npz = settings.window_datasets_root() / dataset / "windows.npz"
    if not npz.exists():
        raise SystemExit(f"no esta {npz} -- ¿esta clonado foveal-vision-data?")
    z = np.load(npz)
    arrays = {k: z[k] for k in z.files}
    t = time.time()
    tr = torch.from_numpy(views_for_split(arrays, SPLIT_TRAIN, dims))[:, None]
    va = torch.from_numpy(views_for_split(arrays, SPLIT_VAL, dims))[:, None]
    if verbose:
        print(f"[datos] vistas construidas en {time.time()-t:.1f} s: "
              f"train {tuple(tr.shape)} val {tuple(va.shape)}")

    # eps is MEASURED from the train split and stored: it is the median local sd,
    # i.e. the typical contrast. An eps typed by hand silently decides how much
    # noise gets amplified in the blank areas, which are most of a text view.
    if eps is None:
        g = _gauss1d(sigma, max(1, int(round(3 * sigma))))
        mu = _blur(tr[:4000], g)
        sd = _blur((tr[:4000] - mu) ** 2, g).clamp_min(0).sqrt()
        eps = float(sd.median())
        if verbose:
            print(f"[datos] eps medido = {eps:.4f} (mediana de la sd local del train)")

    tr = local_contrast_norm(tr, sigma, eps)
    va = local_contrast_norm(va, sigma, eps)
    if limit:
        g = torch.Generator().manual_seed(0)
        tr = tr[torch.randperm(tr.shape[0], generator=g)[:limit]]

    var = float(tr.var())          # FIXED constant of the train split
    if verbose:
        print(f"[datos] var(x) del train = {var:.4f}  ->  denominador fijo de la perdida")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(fich, train=tr.numpy(), val=va.numpy(),
             eps=np.float32(eps), var=np.float32(var))
    return {"train": tr.numpy(), "val": va.numpy(), "eps": eps, "var": var}
