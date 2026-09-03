"""The eight metrics of the brief, all on VALIDATION, at the end of a run.

  1. reconstruction R2 = 1 - mse/var
  2. fraction of positive activations in z
  3. dead kernels: active in < 0.1 % of positions
  4. Gabor fit (main metric) -- and its random baseline. See `fv.probe.gabor`
  5. energy in the classic 6-D subspace, and ITS null 6/k^2
  6. effective dimension: PCA components for 95 % of the variance, over k^2
  7. redundancy: max cosine between distinct pairs
  8. encoder/decoder alignment: cosine between kernel i of each

THE ONE THAT IS READ WRONG IF ITS NULL IS NOT PRINTED NEXT TO IT
---------------------------------------------------------------
Metric 5. The fraction of energy in the classic 6-D subspace has null `6/k^2`:
0.667 in 3x3 but 0.074 in 9x9. Comparing raw fractions across different `k` is
comparing three scales. What is comparable is `energy_6d / (6/k^2)`, which is 1
when the kernel is indistinguishable from a random one. The 0.688 measured on
`fov16-mask-p20` is 1.03x its null -- that is what "no enrichment" means.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from fv.probe.gabor import fit_gabor_r2, random_baseline_r2
from fv.probe.model import L1Probe
from fv.probe.spectrum import (bootstrap_p95, random_spectral_baseline,
                               spectral_metrics)

# "active in < 0.1 % of positions", straight from section 5.3 of the brief.
DEAD_KERNEL_FRAC = 1e-3


def classic_basis(k: int) -> torch.Tensor:
    """The 6 classic filters at size k, orthonormalised: (6, k*k).

    DC, Sobel-x, Sobel-y, laplacian and the two diagonals. For k>3 they are
    built with the same separable recipe that defines them at 3x3 (smoothing
    [1,2,1] -> binomial of order k, derivative [-1,0,1] -> scaled central
    difference), which is the standard generalisation. They are orthonormalised
    with QR because raw they are not orthogonal, and "energy in the subspace"
    would then be an oblique projection -- not what the premise measures.
    """
    axis = torch.arange(k, dtype=torch.float64) - (k - 1) / 2
    smooth = torch.from_numpy(np.array([math.comb(k - 1, i) for i in range(k)],
                                       dtype=np.float64))
    smooth = smooth / smooth.sum()
    deriv = axis.clone()
    lap = axis ** 2 - (axis ** 2).mean()
    dc = torch.ones(k, dtype=torch.float64)

    def outer(a, b):
        return torch.outer(a, b).flatten()

    fil = torch.stack([
        outer(dc, dc),                          # DC
        outer(smooth, deriv),                   # Sobel-x
        outer(deriv, smooth),                   # Sobel-y
        outer(smooth, lap) + outer(lap, smooth),  # laplacian
        outer(deriv, deriv),                    # diagonal /
        outer(lap, lap),                        # diagonal \ (cross curvature)
    ])
    q, _ = torch.linalg.qr(fil.T)               # (k*k, 6) orthonormal
    return q.T.float()


def pca_dim_95(kernels: torch.Tensor) -> int:
    """Metric 6, as the brief words it: components for 95 % of the variance.

    Standard PCA over the K kernels seen as k^2-dimensional vectors, i.e.
    centred on the mean kernel. Capped by min(K-1, k^2), which is why
    `participation_ratio` is reported alongside: that one is not capped and is
    the comparable figure across different K.
    """
    W = kernels - kernels.mean(0, keepdim=True)
    ev = torch.linalg.svdvals(W) ** 2
    if float(ev.sum()) <= 0:
        return 0
    frac = torch.cumsum(ev, 0) / ev.sum()
    return int((frac < 0.95).sum().item()) + 1


def participation_ratio(kernels: torch.Tensor) -> float:
    """Effective dimension that is NOT capped by min(K, k^2), so it can be read
    across columns of the grid. `(sum l)^2 / sum l^2` over the PCA eigenvalues."""
    ev = torch.linalg.svdvals(kernels - kernels.mean(0, keepdim=True)) ** 2
    return float(ev.sum() ** 2 / ev.pow(2).sum().clamp_min(1e-12))


@torch.no_grad()
def final_metrics(m: L1Probe, val: torch.Tensor, var: float, lam: float,
                  batch: int = 512, gabor_steps: int = 400) -> dict:
    k, K = m.k, m.K
    W = m.encoder_kernels()                       # (K, k*k) -- the L1 kernels
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
    D = m.decoder_kernels()
    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)

    # -- 5. classic subspace, always next to its null
    B = classic_basis(k)
    energy = (Wn @ B.T).pow(2).sum(1)
    null = 6.0 / (k * k)

    # -- 6. effective dimension, both readings
    dim95 = pca_dim_95(W)
    pr = participation_ratio(W)

    # -- 7. redundancy
    cos = (Wn @ Wn.T).abs()
    cos.fill_diagonal_(0.0)

    # -- 8. encoder/decoder alignment. No flip: conv_transpose2d(w) is the exact
    #       adjoint of conv2d(w), so "tied weights" means the same orientation.
    align = (Wn * Dn).sum(1)

    # -- 4. the main metric, with its null. The baseline is estimated on
    #       max(4K, 64) random kernels rather than exactly K: it is a NULL
    #       DISTRIBUTION, and estimating it from 8 samples would add noise to
    #       the very number the criterion reads. The literal-K version is
    #       reported too, so nothing is lost.
    with torch.enable_grad():
        g_run = fit_gabor_r2(W, k, steps=gabor_steps)
        n_base = max(4 * K, 64)
        g_base = random_baseline_r2(k, n_base, steps=gabor_steps)
    gab, gab_base = float(g_run.median()), float(g_base.median())
    gab_base_K = float(g_base[:K].median())

    # -- 4b. The criterion the owner asked for on 2026-09-02, because an
    #        ABSOLUTE threshold is three different demands along the k axis: with
    #        the measured nulls, 0.25 is 52 % of the available margin at k=5,
    #        38 % at k=7 and 32 % at k=9.
    #
    #        `gabor_p95` is the p95 of the median of K RANDOM kernels (bootstrap),
    #        so "median > p95" is a one-sided 5 % test with no units.
    #        `gabor_delta_rel` = delta / (1 - null) puts the magnitude on the
    #        scale of what is REACHABLE at that k, which is the comparable one.
    gab_p95 = bootstrap_p95(g_base, K)
    margen = max(1.0 - gab_base, 1e-9)

    # -- 4c. Two template-free metrics (`fv.probe.spectrum`). They exist because
    #        metric 5 stops meaning the same thing once the input is contrast
    #        normalised -- measured, see that module.
    esp = spectral_metrics(W, k)
    esp_base = random_spectral_baseline(k, n_base)

    # -- 1, 2, 3. over the whole validation split
    err, err_int, act, n = 0.0, 0.0, None, 0
    edge = k // 2
    for i in range(0, val.shape[0], batch):
        x = val[i:i + batch]
        xh, z = m(x)
        b = x.shape[0]
        err += float(((xh - x) ** 2).mean()) * b
        if edge and x.shape[-1] > 2 * edge:
            c = slice(edge, -edge)
            err_int += float(((xh[..., c, c] - x[..., c, c]) ** 2).mean()) * b
        else:
            err_int += float(((xh - x) ** 2).mean()) * b
        a = (z > 0).float().mean((0, 2, 3))
        act = a * b if act is None else act + a * b
        n += b
    act = act / n

    return {
        # 1
        "err_rec": err / n / var,
        "err_rec_int": err_int / n / var,
        "r2_rec": 1.0 - err / n / var,
        "r2_rec_int": 1.0 - err_int / n / var,
        # 2
        "frac_activa": float(act.mean()),
        # 3
        "kernels_muertos": int((act < DEAD_KERNEL_FRAC).sum()),
        "umbral_muerto": DEAD_KERNEL_FRAC,
        # 4 -- the main one. `gabor_delta_rel` and `gabor_supera_p95` are what
        #      the criterion reads; `gabor_delta` is kept because it is what the
        #      brief names, and it is NOT comparable across k on its own.
        "gabor_r2": gab,
        "gabor_r2_base": gab_base,
        "gabor_r2_base_K": gab_base_K,
        "gabor_r2_base_n": n_base,
        "gabor_delta": gab - gab_base,
        "gabor_delta_rel": (gab - gab_base) / margen,
        "gabor_p95": gab_p95,
        "gabor_supera_p95": bool(gab > gab_p95),
        "gabor_r2_max": float(g_run.max()),
        "gabor_r2_min": float(g_run.min()),
        # 4c -- sin plantillas, y por eso sobreviven a la normalizacion
        **{f"{n}{suf}": v for n in ("conc_banda", "conc_orient")
           for suf, v in (
               ("", float(esp[n].median())),
               ("_base", float(esp_base[n].median())),
               ("_delta", float(esp[n].median() - esp_base[n].median())),
               ("_p95", bootstrap_p95(esp_base[n], K)),
               ("_supera_p95", bool(float(esp[n].median()) > bootstrap_p95(esp_base[n], K))))},
        "frec_central": float(esp["frec_central"].median()),
        "frec_central_base": float(esp_base["frec_central"].median()),
        # 5
        "energia_6d": float(energy.mean()),
        "energia_6d_sd": float(energy.std()),
        "nulo_6d": null,
        "enriquecimiento": float(energy.mean()) / null,
        # 6
        "dim_pca95": dim95,
        "dim_pca95_frac": dim95 / (k * k),
        "dim_efectiva": pr,
        "dim_max": float(min(K, k * k)),
        # 7
        "coseno_max": float(cos.max()),
        "n_pares_dup": int((cos > 0.9).sum() // 2),
        # 8
        "align_enc_dec": float(align.mean()),
        "align_enc_dec_min": float(align.min()),
        "lambda": lam,
    }
