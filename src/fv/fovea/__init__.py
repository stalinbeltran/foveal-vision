"""G — the foveated geometry every other domain imports (contract (5)/(7)).

Pure arrays and arithmetic: this module must never import anything from fv.

Everything derives from the fundamental parameters (instructionsNewNN.md):
    N        side of the composite input the NN consumes
    c_frac   fraction of the input occupied by the centre (fovea)
    d        downsampling factor of the periphery
    pen_frac penetration fraction of the peripheral kernel into the centre
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FoveaError(ValueError):
    """A geometry problem, with a machine-readable code (api.md R4)."""

    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint}


def round_to_even(x: float) -> int:
    return 2 * int(round(x / 2.0))


@dataclass(frozen=True)
class FoveaDims:
    N: int
    c_frac: float
    d: int
    pen_frac: float
    center_out: int      # fovea side in the composite input == the labelled window (F1b)
    periph_out: int      # ring thickness in the composite input
    penetration: int     # rows/cols shared by both branches
    periph_band: int     # periph_out + penetration: useful band of the outer kernel
    periph_real: int     # periph_out * d: real pixels the ring condenses per side
    original_size: int   # center_out + 2*periph_real: original crop the view needs

    def as_dict(self) -> dict:
        return {
            "N": self.N, "c_frac": self.c_frac, "d": self.d, "pen_frac": self.pen_frac,
            "center_out": self.center_out, "periph_out": self.periph_out,
            "penetration": self.penetration, "periph_band": self.periph_band,
            "periph_real": self.periph_real, "original_size": self.original_size,
        }


def check_dims(N: int, c_frac: float, d: int, pen_frac: float) -> list[dict]:
    """All geometry problems of a parameter set, each with code/message/hint.

    Pure and cheap: called by every training gate (contract (2)) and by the
    sweep runner to discard invalid points before reserving anything.
    """
    problems: list[dict] = []

    def bad(code: str, message: str, hint: str) -> None:
        problems.append({"code": code, "message": message, "hint": hint})

    if N < 8 or N % 2 != 0:
        bad("n_must_be_even", f"N={N} debe ser par y >= 8",
            "elige un N par (la periferia reparte simetrico)")
        return problems
    if d < 1:
        bad("downsample_must_be_positive", f"d={d} debe ser >= 1", "usa d >= 1")
        return problems

    center_out = round_to_even(N * c_frac)
    periph_out = (N - center_out) // 2
    penetration = max(1, round(N * pen_frac))

    if center_out < 4:
        bad("center_too_small", f"center_out={center_out} con N={N}, c_frac={c_frac}",
            "sube c_frac o N: la fovea necesita al menos 4 px")
    if periph_out < 1:
        bad("no_periphery", f"c_frac={c_frac} deja periph_out=0 con N={N}",
            "baja c_frac: sin anillo periferico esta red es una CNN plana")
    if 2 * periph_out + center_out != N:
        bad("parity_broken", f"2*{periph_out}+{center_out} != {N}",
            "N y center_out deben tener la misma paridad (ambos pares)")
    if center_out >= 4 and penetration >= center_out // 2:
        bad("penetration_too_large",
            f"penetration={penetration} >= center_out//2={center_out // 2}",
            "baja pen_frac: el nucleo exclusivo del kernel central no puede desaparecer")
    return problems


def derive_dims(N: int, c_frac: float, d: int, pen_frac: float) -> FoveaDims:
    problems = check_dims(N, c_frac, d, pen_frac)
    if problems:
        p = problems[0]
        raise FoveaError(p["code"], p["message"], p["hint"])
    center_out = round_to_even(N * c_frac)
    periph_out = (N - center_out) // 2
    penetration = max(1, round(N * pen_frac))
    return FoveaDims(
        N=N, c_frac=c_frac, d=d, pen_frac=pen_frac,
        center_out=center_out, periph_out=periph_out, penetration=penetration,
        periph_band=periph_out + penetration,
        periph_real=periph_out * d,
        original_size=center_out + 2 * periph_out * d,
    )


# ---------------------------------------------------------------------------
# Search ranges as FUNCTIONS of the region (instructionsNewNN.md §3) — never
# constants. H consumes these; it does not define them.

def kernel_range(region_size: int) -> list[int]:
    """Odd kernels from 3 up to ~region/2, never exceeding the region."""
    k_max = region_size // 2
    if k_max % 2 == 0:
        k_max -= 1
    return [k for k in range(3, max(3, k_max) + 1, 2)]


def stride_range(region_size: int, n_layers: int = 2) -> list[int]:
    """Strides whose cumulative product does not collapse the region (<= region/4)."""
    max_cumulative = max(1, region_size // 4)
    s_max = max(1, int(round(max_cumulative ** (1.0 / n_layers))))
    return list(range(1, s_max + 1))


def downsample_range(periph_out: int, N: int, max_original: int | None = None) -> list[int]:
    """d such that the ring reduces to >=1px and the original crop stays bounded."""
    d_min, d_max = 1, 8
    if max_original:
        while (periph_out * d_max * 2 + (N - 2 * periph_out)) > max_original and d_max > 1:
            d_max -= 1
    return list(range(d_min, d_max + 1))


def build_search_space(N: int, c_frac: float = 0.8, pen_frac: float = 0.1,
                       n_layers: int = 2, max_original: int | None = None) -> dict:
    center_out = round_to_even(N * c_frac)
    periph_out = (N - center_out) // 2
    penetration = max(1, round(N * pen_frac))
    periph_band = periph_out + penetration
    return {
        "k_center": kernel_range(center_out),
        "k_periph": kernel_range(periph_band),
        "s_center": stride_range(center_out, n_layers),
        "s_periph": stride_range(periph_band, n_layers),
        "d": downsample_range(periph_out, N, max_original=max_original or 2 * N),
        "_center_out": center_out,
        "_periph_out": periph_out,
        "_penetration": penetration,
    }


# ---------------------------------------------------------------------------
# The composite view. EXCLUSIVE sampling: every composite pixel has exactly one
# origin (centre OR ring). The centre is copied untouched; ring cells average
# (or max) anisotropic blocks of the original crop:
#   - both coords in the ring  -> d x d block
#   - ring row, centre col     -> d x 1 block (co-registered with the fovea col)
#   - centre row, centre col   -> 1 x 1 (exact copy)
# This reproduces the coordinate table of instructionsNewNN.md §4 and keeps the
# fovea bit-identical to the direct crop (tested).

def _axis_edges(dims: FoveaDims) -> np.ndarray:
    """Start offset in the original crop for each of the N composite cells (+ end)."""
    m, c, d, N = dims.periph_real, dims.center_out, dims.d, dims.N
    po = dims.periph_out
    edges = []
    for k in range(N):
        if k < po:
            edges.append(k * d)
        elif k < po + c:
            edges.append(m + (k - po))
        else:
            edges.append(m + c + (k - po - c) * d)
    edges.append(dims.original_size)
    return np.asarray(edges, dtype=np.int64)


def _pool_axis(a: np.ndarray, edges: np.ndarray, axis: int, mode: str) -> np.ndarray:
    starts = edges[:-1]
    if mode == "avg":
        sums = np.add.reduceat(a, starts, axis=axis)
        counts = np.diff(edges).astype(a.dtype if a.dtype.kind == "f" else np.float32)
        shape = [1] * a.ndim
        shape[axis] = len(starts)
        return sums / counts.reshape(shape)
    if mode == "max":
        return np.maximum.reduceat(a, starts, axis=axis)
    raise FoveaError("unknown_pool_mode", f"pool_mode '{mode}' no existe",
                     "usa 'avg' o 'max'")


def build_foveated_input(crop: np.ndarray, dims: FoveaDims,
                         pool_mode: str = "avg") -> np.ndarray:
    """crop: float array (..., original_size, original_size) -> (..., N, N)."""
    if crop.shape[-1] != dims.original_size or crop.shape[-2] != dims.original_size:
        raise FoveaError(
            "crop_size_mismatch",
            f"el recorte es {crop.shape[-2]}x{crop.shape[-1]} y la vista necesita "
            f"{dims.original_size}x{dims.original_size}",
            "recorta original_size px alrededor de la ventana etiquetada")
    a = crop.astype(np.float32, copy=False)
    edges = _axis_edges(dims)
    a = _pool_axis(a, edges, axis=a.ndim - 2, mode=pool_mode)
    a = _pool_axis(a, edges, axis=a.ndim - 1, mode=pool_mode)
    # the centre must be the untouched crop (exclusive sampling, tested bit-exact)
    return a


def build_view(image: np.ndarray, wx0: int, wy0: int, dims: FoveaDims,
               pool_mode: str = "avg", pad_mode: str = "edge") -> tuple[np.ndarray, np.ndarray]:
    """Composite view + coverage mask for the labelled window at (wx0, wy0).

    image: full grayscale image (H, W) uint8/float. The labelled window is the
    fovea: center_out x center_out at (wx0, wy0). Returns (view (N,N) float32
    in [0,1], coverage (N,N) float32 fraction of real pixels per cell).

    Padding beyond the image border: 'edge' replicates the border row/col
    (decision C10: never plain zeros — zero means "no ink" and teaches a false
    rule); the coverage mask carries the real fraction per cell for debugging
    (F0 view), it is NOT fed to the net in v1.
    """
    H, W = image.shape
    m = dims.periph_real
    x0, y0 = wx0 - m, wy0 - m
    s = dims.original_size
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x0 + s - W)
    pad_b = max(0, y0 + s - H)
    sl = image[max(0, y0):min(H, y0 + s), max(0, x0):min(W, x0 + s)]
    if pad_l or pad_t or pad_r or pad_b:
        if pad_mode == "edge":
            crop = np.pad(sl, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
        elif pad_mode == "mean":
            crop = np.pad(sl, ((pad_t, pad_b), (pad_l, pad_r)),
                          mode="constant", constant_values=float(sl.mean()) if sl.size else 0.0)
        elif pad_mode == "zero":
            crop = np.pad(sl, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant")
        else:
            raise FoveaError("unknown_pad_mode", f"pad_mode '{pad_mode}' no existe",
                             "usa 'edge', 'mean' o 'zero'")
        inside = np.zeros((s, s), dtype=np.float32)
        inside[pad_t:s - pad_b or None, pad_l:s - pad_r or None] = 1.0
    else:
        crop = sl
        inside = np.ones((s, s), dtype=np.float32)
    view = build_foveated_input(crop.astype(np.float32) / 255.0
                                if crop.dtype == np.uint8 else crop.astype(np.float32),
                                dims, pool_mode=pool_mode)
    coverage = build_foveated_input(inside, dims, pool_mode="avg")
    return view.astype(np.float32), coverage.astype(np.float32)


# ---------------------------------------------------------------------------
# Branch masks. CONTRIBUTIVE overlap: in the penetration band both masks are 1
# and both branches contribute (they are applied to the INPUT, option A —
# masking after convolution was rejected, instructionsNewNN.md §7).

def build_masks(dims: FoveaDims) -> tuple[np.ndarray, np.ndarray]:
    N, po, pen = dims.N, dims.periph_out, dims.penetration
    center_mask = np.zeros((N, N), dtype=np.float32)
    periph_mask = np.zeros((N, N), dtype=np.float32)
    lo, hi = po, N - po
    center_mask[lo:hi, lo:hi] = 1.0
    inner_lo, inner_hi = po + pen, N - po - pen
    periph_mask[:, :] = 1.0
    periph_mask[inner_lo:inner_hi, inner_lo:inner_hi] = 0.0
    return center_mask, periph_mask
