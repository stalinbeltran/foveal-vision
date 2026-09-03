"""Two template-free metrics, from the 2-D FFT of each kernel.

WHY THEY EXIST (owner's review, 2026-09-02)
-------------------------------------------
The study was left without a single metric comparable along its main axis:

  · the **Gabor fit** breaks at the BOTTOM -- its null is 0.879 at k=3, so the
    ceiling on the difference there is 0.121 (`fv.probe.gabor`);
  · the **classic 6-D subspace** breaks at the TOP -- and that one is measured:
    `enriq` sits at 0.47-0.61 across the whole probe, i.e. BELOW its own null of
    1.0. The learned kernels have LESS energy in the classic subspace than
    random ones do.

The mechanism, measured on 2026-09-02: `classic_basis` builds the k>3 filters
with binomial smoothing, so they are low-frequency templates -- their radial
power at k=7 is 1.000 at DC, 0.448 at r=1, then ~0. The mandatory local contrast
normalisation of section 2 strips DC and the low frequencies from the INPUT (raw
views carry essentially all their power at DC; normalised ones peak at r=7), so
the learned kernels live in high frequency and come out nearly orthogonal to that
basis. Run the same cell without normalisation and `enriq` returns to 1.01.

These two numbers do not depend on any fixed template, which is what makes them
survive that.

  · `conc_banda`   -- is it BAND-PASS? The largest fraction of non-DC power
    inside a radial band 0.1 cycles/px wide.
  · `conc_orient`  -- is it ORIENTED? Circular concentration at DOUBLE angle,
    which is the right periodicity for an orientation (theta and theta+pi are
    the same orientation). 1.0 = perfectly oriented, 0.0 = isotropic.

⚠ AND THEY STILL CARRY A NULL, per the rule of this project. They do not explode
like the Gabor's, but they are NOT free of k: a 3x3 kernel has small spatial
support, so by the uncertainty principle its spectrum CANNOT be narrow. That is
not a defect of the metric -- it is the very premise of the experiment ("the
structure does not fit in 3x3") -- but it means the reading is the difference
against the null of that same k, never the raw value.
"""

from __future__ import annotations

import math

import torch

# Every kernel is zero-padded to this size before the FFT, so the frequency grid
# is IDENTICAL for k=3 and k=9. Zero-padding interpolates a spectrum, it does not
# change it; without it the radial bins would mean a different frequency at each
# k and nothing would be comparable.
FFT_N = 32

# Fixed PHYSICAL width, in cycles/px, of the band `conc_banda` looks for. Fixed
# and not "a fraction of the range" so it means the same thing at every k.
BANDA_ANCHO = 0.1

BASELINE_SEED = 20260902


def _power(kernels: torch.Tensor, k: int) -> torch.Tensor:
    """|FFT|^2 of each kernel, zero-padded to FFT_N. Returns (M, N, N)."""
    w = kernels.reshape(-1, k, k).to(torch.float32)
    w = w / w.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1)
    pad = FFT_N - k
    w = torch.nn.functional.pad(w, (0, pad, 0, pad))
    return torch.fft.fft2(w).abs() ** 2


def _freq_grid() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    f = torch.fft.fftfreq(FFT_N)                       # cycles/px, signed
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    rad = (fx ** 2 + fy ** 2).sqrt()
    ang = torch.atan2(fy, fx)
    return rad, ang, (rad > 1e-9)                      # mask: everything but DC


def spectral_metrics(kernels: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Per-kernel `conc_banda`, `conc_orient` and the spectral centroid."""
    P = _power(kernels, k)                             # (M, N, N)
    rad, ang, no_dc = _freq_grid()
    r, a = rad[no_dc], ang[no_dc]
    p = P[:, no_dc]                                    # (M, F)
    total = p.sum(1).clamp_min(1e-12)

    # -- banda: la mayor fraccion de potencia dentro de una banda radial de
    #    ancho fijo. Se prueban los bordes candidatos que hay de verdad (los
    #    radios presentes), no una rejilla arbitraria.
    bordes = torch.unique(r)
    dentro = (r[None, :] >= bordes[:, None]) & (r[None, :] < bordes[:, None] + BANDA_ANCHO)
    frac = (p[:, None, :] * dentro[None, :, :]).sum(-1) / total[:, None]   # (M, B)
    banda = frac.max(dim=1).values

    # -- orientacion: concentracion circular a DOBLE angulo. Es la periodicidad
    #    correcta: theta y theta+pi son la misma orientacion, y con angulo
    #    simple un filtro orientado daria 0 por simetria del espectro.
    orient = ((p * torch.cos(2 * a)).sum(1) ** 2
              + (p * torch.sin(2 * a)).sum(1) ** 2).sqrt() / total

    centroide = (p * r).sum(1) / total                 # cycles/px
    return {"conc_banda": banda, "conc_orient": orient, "frec_central": centroide}


def random_spectral_baseline(k: int, n: int) -> dict[str, torch.Tensor]:
    """The same metrics on `n` random kernels of the same size."""
    g = torch.Generator().manual_seed(BASELINE_SEED + k)
    return spectral_metrics(torch.randn(n, k * k, generator=g), k)


def bootstrap_p95(nulos: torch.Tensor, K: int, reps: int = 2000,
                  seed: int = BASELINE_SEED) -> float:
    """p95 de la MEDIANA de K valores nulos -- la prueba que pidio el dueno.

    El estadistico que se compara es la mediana sobre los K kernels de un run,
    asi que el nulo tiene que ser la distribucion de ESA mediana, no la de un
    kernel suelto. Comparar una mediana de 32 contra el p95 de valores sueltos
    daria una prueba mucho mas laxa de lo que aparenta.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, nulos.numel(), (reps, K), generator=g)
    return float(nulos[idx].median(dim=1).values.quantile(0.95))
