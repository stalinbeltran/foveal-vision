"""Metric 4 of the brief, THE MAIN ONE: how well does a 2-D Gabor fit each kernel?

WHY THIS IS READ AS A DIFFERENCE AND NEVER AS AN ABSOLUTE VALUE
---------------------------------------------------------------
A Gabor has many free parameters and fits noise better than one expects. On a
3x3 kernel it has 7 free parameters for 9 numbers: it fits ANYTHING. So the same
metric is computed on random kernels of the same size and what gets read is
`gabor_r2 - gabor_r2_base`. The absolute value alone is meaningless, and the
brief says so explicitly.

THE THREE CHOICES THAT DECIDE WHAT THE NUMBER MEANS
---------------------------------------------------
1. **Amplitude is solved in closed form, not optimised.** For a fixed Gabor
   SHAPE `s`, the best-scaling residual is minimised by `A = <w,s>/<s,s>`, and
   then `R2 = cos^2(w, s)`. That removes one nonlinear parameter (7 instead of
   8), bounds the metric in [0, 1] by construction, and makes it exactly
   "fraction of the kernel's energy explained by the best-scaled Gabor".

2. **R2 is against the kernel's ENERGY, not its variance around the mean.**
   `SS_tot = ||w||^2` with `w` unit-normalised, so `SS_tot = 1`. Variance around
   the mean is the textbook definition, but it blows up for a near-constant (DC)
   kernel -- and DC kernels are precisely what `fov16-mask-p20` produced, i.e.
   the case this probe exists to look at. Energy is well defined for every
   kernel, and it is the same normaliser metric 5 (classic subspace) uses, so
   the two metrics stay on one footing. The random baseline uses the identical
   definition, so the DIFFERENCE is unaffected by this choice.

3. **The fit is multi-start and deterministic.** A fixed grid of 32 starts
   (4 orientations x 4 frequencies x 2 phases); no randomness anywhere, so
   re-running gives the same number. Only the BASELINE kernels are random, and
   they come from a fixed seed written below.
"""

from __future__ import annotations

import math

import torch

# The baseline kernels are drawn from THIS seed and no other, so the null is the
# same across runs with the same (k, n) and the comparison is not moved by it.
BASELINE_SEED = 20260902

_N_THETA, _N_FREQ, _N_PHASE = 4, 4, 2
N_STARTS = _N_THETA * _N_FREQ * _N_PHASE


def _grid(k: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    y, x = torch.meshgrid(a, a, indexing="ij")
    return x.reshape(-1), y.reshape(-1)


def _shape(p: torch.Tensor, x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """Unit-amplitude Gabor for a batch of parameter vectors. p: (B, 7)."""
    half = k / 2.0
    x0 = half * torch.tanh(p[:, 0:1])          # centre stays inside the kernel
    y0 = half * torch.tanh(p[:, 1:2])
    theta = p[:, 2:3]
    # sigma >= 0.35 px: below that the gaussian is a delta between samples and
    # the fit degenerates into "one pixel", which is not a Gabor.
    su = 0.35 + torch.nn.functional.softplus(p[:, 3:4])
    sv = 0.35 + torch.nn.functional.softplus(p[:, 4:5])
    freq = 0.5 * torch.sigmoid(p[:, 5:6])      # 0 .. Nyquist (cycles/px)
    phase = p[:, 6:7]

    dx, dy = x[None, :] - x0, y[None, :] - y0
    c, s = torch.cos(theta), torch.sin(theta)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    env = torch.exp(-(u ** 2 / (2 * su ** 2) + v ** 2 / (2 * sv ** 2)))
    return env * torch.cos(2 * math.pi * freq * u + phase)


def _starts(k: int, device, dtype) -> torch.Tensor:
    rows = []
    sigma0 = max(0.6, k / 5.0)
    raw_sigma = math.log(math.expm1(max(sigma0 - 0.35, 1e-3)))     # inverse softplus
    for i in range(_N_THETA):
        for f in (0.02, 0.12, 0.25, 0.40):
            # inverse of 0.5*sigmoid
            raw_f = math.log(f / (0.5 - f)) if 0 < f < 0.5 else 0.0
            for ph in (0.0, math.pi / 2):
                rows.append([0.0, 0.0, i * math.pi / _N_THETA,
                             raw_sigma, raw_sigma, raw_f, ph])
    return torch.tensor(rows, device=device, dtype=dtype)


def fit_gabor_r2(kernels: torch.Tensor, k: int, steps: int = 400,
                 lr: float = 0.12) -> torch.Tensor:
    """Best-fit Gabor R2 per kernel. `kernels`: (M, k*k). Returns (M,).

    Every (kernel, start) pair is optimised in ONE batched Adam run, so the cost
    is a handful of seconds for the whole grid rather than M*32 separate fits.
    """
    if kernels.ndim != 2 or kernels.shape[1] != k * k:
        raise ValueError(f"kernels must be (M, {k*k}), got {tuple(kernels.shape)}")
    device, dtype = kernels.device, torch.float32
    w = kernels.to(dtype)
    w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    M = w.shape[0]

    x, y = _grid(k, device, dtype)
    p = _starts(k, device, dtype).repeat(M, 1).clone().requires_grad_(True)
    target = w.repeat_interleave(N_STARTS, dim=0)                  # (M*S, k*k)

    opt = torch.optim.Adam([p], lr=lr)
    for i in range(steps):
        if i == int(steps * 0.7):
            for gparam in opt.param_groups:
                gparam["lr"] = lr / 6
        s = _shape(p, x, y, k)
        num = (s * target).sum(1) ** 2
        den = (s * s).sum(1).clamp_min(1e-12)
        r2 = num / den                                              # ||target||=1
        opt.zero_grad(set_to_none=True)
        (-r2.sum()).backward()
        opt.step()

    with torch.no_grad():
        s = _shape(p, x, y, k)
        r2 = (s * target).sum(1) ** 2 / (s * s).sum(1).clamp_min(1e-12)
        return r2.view(M, N_STARTS).max(dim=1).values.clamp(0.0, 1.0)


def random_baseline_r2(k: int, n: int, steps: int = 400) -> torch.Tensor:
    """The null: the SAME metric on `n` random kernels of the same size."""
    g = torch.Generator().manual_seed(BASELINE_SEED + k)
    w = torch.randn(n, k * k, generator=g)
    return fit_gabor_r2(w, k, steps=steps)
