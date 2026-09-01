"""G — the foveated geometry every other domain imports (contract (5)/(7)).

Pure arrays and arithmetic: this module must never import anything from fv.

Everything derives from five fundamental parameters, ALL OF THEM IN REAL PIXELS
of the source image (2026-08-25 reparameterisation). The fovea and the border
are stated independently, and HOW the border is reduced is a separate knob, so a
future reduction method can replace it without touching either definition:

    fovea_px           fovea side. It IS the labelled window of B (contract (1)a)
    border_px          blurry border thickness, per side, around the fovea
    border_reduce      real px of border condensed into ONE cell of the input
    overlap_fovea_px   px OF THE FOVEA the border branch also sees
    overlap_border_px  px OF THE BORDER the fovea branch also sees

`N` (the side of the composite input the NN consumes) is DERIVED from those, not
written by hand: N = fovea_px + 2*(border_px // border_reduce). The old spelling
(N, c_frac, d, pen_frac) is still READ from artefacts written before this change
and mapped to the canonical five — read old, write new, the same rule `channels`
follows for ch1/ch2. A config carrying both spellings is REFUSED, never
reconciled by precedence: that is "the same fact in two places", the failure mode
this project keeps paying for.
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


# How C divides the input into branches. Lives HERE and not in the builder
# because `fv.validation` must stay a pure, torch-free leaf (contract (7)) and
# still needs the vocabulary to refuse an unknown value at the gate.
REGIONS = ("split", "single")

# What C feeds STRAIGHT INTO THE HEAD about the image edge, bypassing the conv
# branches. Lives here, next to REGIONS and for the same reason: `fv.validation`
# is a pure torch-free leaf and still has to refuse an unknown value at the gate.
#
# ⚠ "edge" is the edge OF THE SOURCE IMAGE; "border" is the blurry ring of the
# foveated view. Spanish says `borde` for both and that is exactly why the code
# does not: `border_px` is a ring width, `edge_inputs` is about running out of
# image. Mixing them would make `border_px=0` read as "no image edge", which is
# false for every window of a flat control.
#
#   off   nothing: the head sees only what the branches produce (the net that
#         every artefact on disk was trained with)
#   pad   per side, the fraction of THIS VIEW's margin that is replicated
#         padding instead of real image. Saturates at the crop: it is 0 as soon
#         as the window sits `border_px` away from the edge
#   dist  per side, how close the LABELLED WINDOW is to the image edge, in
#         fovea units and capped at one fovea. Reaches further than `pad` and
#         works with `border_px=0`
#
# Both spellings are oriented the same way on purpose: 0 = no image edge on this
# side, 1 = the edge is right here. A reader (or a kernel weight) does not have
# to remember which mode inverts the sign.
EDGE_MODES = ("off", "pad", "dist")

# El segundo canal de ENTRADA: que parte de cada celda de la vista es relleno
# inventado en vez de imagen real. `off` = un solo canal, que es lo que habia.
MASK_MODES = ("off", "coverage")

# The order of the four numbers, declared ONCE — the same reason CORNER_NAMES is
# declared once in fv.metrics. Left, Top, Right, Bottom.
EDGE_SIDES = ("L", "T", "R", "B")

# The canonical geometry, in real pixels. `border_reduce` is the only one that
# is not a length: it is the current reduction METHOD's factor, kept apart from
# the two lengths on purpose (that separation is the point of this spelling).
GEOMETRY_FIELDS = ("fovea_px", "border_px", "border_reduce",
                   "overlap_fovea_px", "overlap_border_px")

# The pre-2026-08-25 spelling. Accepted on read, never written. `d` is listed
# here but handled apart: it was renamed to `border_reduce` because its MEANING
# changed — see `normalize_geometry`.
LEGACY_GEOMETRY_FIELDS = ("N", "c_frac", "pen_frac", "d")


def is_single_region(net: dict) -> bool:
    """Absent means 'split' — the behaviour of every artefact written before
    this field existed (plan-cnn-plana.md 2.1)."""
    return net.get("regions", "split") == "single"


# ---------------------------------------------------------------------------
# Normalisation: one spelling in, the canonical five out.

def _from_legacy(cfg: dict) -> dict:
    """(N, c_frac, d, pen_frac) -> the canonical five, EXACTLY.

    Bit-identical by construction: every derived number below is what the old
    derive_dims computed from the old four, so a migrated config builds the same
    network, loads the same checkpoint and produces the same view.
    """
    N = int(cfg["N"])
    c_frac = float(cfg["c_frac"])
    reduce_ = int(cfg.get("d", 2))
    pen_frac = float(cfg.get("pen_frac", 0.1))
    fovea_px = round_to_even(N * c_frac)
    border_cells = (N - fovea_px) // 2
    return {
        "fovea_px": fovea_px,
        "border_px": border_cells * reduce_,
        "border_reduce": reduce_,
        # the fovea is sampled 1:1, so its px and its cells are the same number
        "overlap_fovea_px": max(1, round(N * pen_frac)),
        # the old masks never grew the centre outwards: it saw zero border
        "overlap_border_px": 0,
    }


def normalize_geometry(cfg: dict) -> dict:
    """The canonical five, from whichever spelling `cfg` uses.

    Refuses a config that mixes both, and refuses a bare `d` without the rest of
    the old spelling: `d` used to mean "how much context", and it now means "how
    coarsely a border OF FIXED SIZE is condensed". Silently honouring it would
    build a different network from the one the caller has in mind, which is
    exactly the class of failure R4 exists to prevent.
    """
    new = [f for f in GEOMETRY_FIELDS if f in cfg and f != "border_reduce"]
    old = [f for f in LEGACY_GEOMETRY_FIELDS if f in cfg and f != "d"]
    if new and old:
        raise FoveaError(
            "geometry_double_spec",
            f"la config mezcla la geometria nueva {sorted(new)} con la vieja {sorted(old)}",
            "deja solo fovea_px/border_px/border_reduce/overlap_*_px: "
            "N, c_frac, pen_frac y d se derivan de esos")
    if old:
        if "N" not in cfg or "c_frac" not in cfg:
            raise FoveaError(
                "legacy_geometry_incomplete",
                f"geometria vieja incompleta: {sorted(old)} sin N y c_frac",
                "escribe la geometria nueva: fovea_px, border_px, border_reduce")
        return _from_legacy(cfg)
    if "d" in cfg and "border_reduce" not in cfg:
        raise FoveaError(
            "d_renamed",
            "'d' cambio de significado y ahora se llama 'border_reduce'",
            "antes 'd' agrandaba el contexto (borde = periph_out*d); hoy el borde "
            "es border_px y border_reduce solo dice cuantos px caben en una celda. "
            "Para barrer cuanto contexto ve la red, barre 'border_px'")
    out = {}
    for f in GEOMETRY_FIELDS:
        if f in cfg and cfg[f] is not None:
            out[f] = int(cfg[f])
    return out


def dims_of(net: dict) -> "FoveaDims":
    """Derive the geometry FROM A NETWORK CONFIG — the single place that knows
    `regions` affects what counts as a legal geometry, and the single place that
    knows how to read the old spelling.

    Six call sites derived dims straight from the four scalars, and each one that
    forgot the flag would raise `no_periphery` on a perfectly legal flat control,
    inside a training job. One definition, many readers (the failure mode this
    project keeps paying for: the same fact represented twice)."""
    return derive_dims(normalize_geometry(net), single_region=is_single_region(net))


@dataclass(frozen=True)
class FoveaDims:
    """The geometry, resolved. The canonical five are STORED; everything else is
    a derived view of them, computed once and here — never a second definition.
    """
    # --- canonical, real px of the source image
    fovea_px: int
    border_px: int
    border_reduce: int
    overlap_fovea_px: int
    overlap_border_px: int
    # --- derived, cells of the composite input
    border_cells: int          # border_px // border_reduce
    N: int                     # fovea_px + 2*border_cells: the input the NN sees
    overlap_border_cells: int  # overlap_border_px // border_reduce
    center_band: int           # cells the centre branch's mask covers, per side
    periph_band: int           # cells the border branch's mask covers, per side
    original_size: int         # fovea_px + 2*border_px: the real crop the view needs

    # --- the pre-reparameterisation names, kept because several domains read
    # them. Derived properties, so there is still ONE definition of each fact.
    @property
    def center_out(self) -> int:
        return self.fovea_px

    @property
    def periph_out(self) -> int:
        return self.border_cells

    @property
    def periph_real(self) -> int:
        return self.border_px

    @property
    def penetration(self) -> int:
        return self.overlap_fovea_px

    @property
    def d(self) -> int:
        return self.border_reduce

    @property
    def c_frac(self) -> float:
        return self.fovea_px / self.N

    def as_dict(self) -> dict:
        return {
            "fovea_px": self.fovea_px, "border_px": self.border_px,
            "border_reduce": self.border_reduce,
            "overlap_fovea_px": self.overlap_fovea_px,
            "overlap_border_px": self.overlap_border_px,
            "border_cells": self.border_cells, "N": self.N,
            "overlap_border_cells": self.overlap_border_cells,
            "center_band": self.center_band, "periph_band": self.periph_band,
            "original_size": self.original_size,
            # derived, for readers that still speak the old names
            "center_out": self.center_out, "periph_out": self.periph_out,
            "periph_real": self.periph_real, "penetration": self.penetration,
            "c_frac": self.c_frac,
        }


def check_dims(geom: dict, single_region: bool = False) -> list[dict]:
    """All geometry problems of a parameter set, each with code/message/hint.

    Pure and cheap: called by every training gate (contract (2)) and by the
    sweep runner to discard invalid points before reserving anything.

    `single_region` is C's `regions == "single"`: ONE unmasked branch over the
    whole N x N input (the flat-CNN control of protocolo.md 6, plan-cnn-plana.md).
    Three problems stop describing anything there and are therefore not raised:
    `no_border` (there is no border to be missing — that IS the control) and the
    two overlap problems (nothing overlaps: no mask is ever built). Every other
    problem still applies, and `build_masks` is untouched either way.
    """
    problems: list[dict] = []

    def bad(code: str, message: str, hint: str) -> None:
        problems.append({"code": code, "message": message, "hint": hint})

    fovea = int(geom.get("fovea_px", 0))
    border = int(geom.get("border_px", 0))
    reduce_ = int(geom.get("border_reduce", 1))
    ov_f = int(geom.get("overlap_fovea_px", 0))
    ov_b = int(geom.get("overlap_border_px", 0))

    if fovea < 4 or fovea % 2 != 0:
        bad("fovea_must_be_even", f"fovea_px={fovea} debe ser par y >= 4",
            "la fovea es la ventana etiquetada de B: reconstruye B con una ventana par")
        return problems
    if reduce_ < 1:
        bad("reduce_must_be_positive", f"border_reduce={reduce_} debe ser >= 1",
            "usa border_reduce >= 1: es cuantos px reales caben en una celda de borde")
        return problems
    if border < 0:
        bad("border_negative", f"border_px={border} no puede ser negativo",
            "usa border_px >= 0 (0 es la CNN plana: declara regions='single')")
        return problems
    if border % reduce_ != 0:
        lo = border - border % reduce_
        bad("border_not_divisible",
            f"border_px={border} no es multiplo de border_reduce={reduce_}",
            f"usa un borde multiplo de {reduce_} (p. ej. {lo} o {lo + reduce_}), "
            f"o cambia border_reduce")
        return problems

    border_cells = border // reduce_
    if border_cells < 1 and not single_region:
        bad("no_border", f"border_px={border} deja el anillo en 0 celdas",
            "sube border_px: sin borde esta red es una CNN plana "
            "(si eso es lo que quieres, declara regions='single')")

    if ov_f < 0 or ov_b < 0:
        bad("overlap_negative", f"solapes negativos: fovea={ov_f}, borde={ov_b}",
            "los solapes son px >= 0")
        return problems
    # the fovea is sampled 1:1, so its overlap needs no divisibility rule
    if ov_f >= fovea // 2 and not single_region:
        bad("penetration_too_large",
            f"overlap_fovea_px={ov_f} >= fovea_px//2={fovea // 2}",
            "baja overlap_fovea_px: el nucleo exclusivo del kernel central no "
            "puede desaparecer")
    if ov_b % reduce_ != 0:
        bad("overlap_border_not_divisible",
            f"overlap_border_px={ov_b} no es multiplo de border_reduce={reduce_}",
            f"el solape cae en celdas de borde: usa un multiplo de {reduce_}")
    elif border_cells >= 1 and ov_b >= border and not single_region:
        bad("overlap_border_too_large",
            f"overlap_border_px={ov_b} >= border_px={border}",
            "baja overlap_border_px: el borde exclusivo del kernel periferico no "
            "puede desaparecer (es el espejo de penetration_too_large)")
    return problems


def derive_dims(geom: dict, single_region: bool = False) -> FoveaDims:
    problems = check_dims(geom, single_region)
    if problems:
        p = problems[0]
        raise FoveaError(p["code"], p["message"], p["hint"])
    fovea = int(geom["fovea_px"])
    border = int(geom.get("border_px", 0))
    reduce_ = int(geom.get("border_reduce", 1))
    ov_f = int(geom.get("overlap_fovea_px", 0))
    ov_b = int(geom.get("overlap_border_px", 0))
    border_cells = border // reduce_
    ov_b_cells = ov_b // reduce_
    return FoveaDims(
        fovea_px=fovea, border_px=border, border_reduce=reduce_,
        overlap_fovea_px=ov_f, overlap_border_px=ov_b,
        border_cells=border_cells,
        N=fovea + 2 * border_cells,
        overlap_border_cells=ov_b_cells,
        center_band=fovea + 2 * ov_b_cells,
        periph_band=border_cells + ov_f,
        original_size=fovea + 2 * border,
    )


# ---------------------------------------------------------------------------
# Search ranges as FUNCTIONS of the region (instructionsNewNN.md 3) — never
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


def reduce_range(border_px: int) -> list[int]:
    """The reduction factors a border of this size admits: its divisors.

    The border is a LENGTH now, so `border_reduce` no longer changes how much
    context the net sees — only how coarsely it is condensed. That is why the
    old `max_original` cap is gone from here: the real crop is
    fovea_px + 2*border_px and does not move with the reduction.
    """
    if border_px <= 0:
        return [1]
    return [r for r in range(1, border_px + 1) if border_px % r == 0]


def border_range(fovea_px: int, border_reduce: int = 1,
                 max_original: int | None = None) -> list[int]:
    """Border widths (px) that keep the real crop bounded, on the cell grid.

    `max_original` is the largest real crop worth sampling; it defaults to
    3*fovea_px (a border as wide as the fovea, on each side). It is a SEARCH
    bound, not a law of the geometry: the hard limit is the image, and only B
    knows how big that is.
    """
    r = max(1, int(border_reduce))
    cap = int(max_original) if max_original else 3 * int(fovea_px)
    b_max = max(0, (cap - int(fovea_px)) // 2)
    return [b for b in range(r, b_max + 1, r)]


def overlap_fovea_range(fovea_px: int) -> list[int]:
    """How far the border branch may reach into the fovea: 0 .. fovea/2 - 1.

    0 is legal and NEW: it makes the two branches disjoint, which is the control
    for the contributive-overlap choice of instructionsNewNN.md 7. The old
    spelling could not express it (`penetration = max(1, ...)` had a floor of 1).
    """
    return list(range(0, max(1, int(fovea_px) // 2)))


def overlap_border_range(border_px: int, border_reduce: int = 1) -> list[int]:
    """How far the fovea branch may reach into the border: 0 .. border - reduce.

    Capped one cell short of the whole border by the mirror of
    penetration_too_large — the border branch keeps an exclusive part. With a
    narrow border there are few legal values; widening the range means widening
    `border_px`, which is the point of stating the two independently.
    """
    r = max(1, int(border_reduce))
    return [b for b in range(0, max(0, int(border_px) - r) + 1, r)]


def build_search_space(geom: dict, n_layers: int = 2,
                       max_original: int | None = None) -> dict:
    dims = derive_dims(normalize_geometry(geom))
    return {
        "k_center": kernel_range(dims.center_band),
        "k_periph": kernel_range(dims.periph_band),
        "s_center": stride_range(dims.center_band, n_layers),
        "s_periph": stride_range(dims.periph_band, n_layers),
        "border_px": border_range(dims.fovea_px, dims.border_reduce, max_original),
        "border_reduce": reduce_range(dims.border_px),
        "overlap_fovea_px": overlap_fovea_range(dims.fovea_px),
        "overlap_border_px": overlap_border_range(dims.border_px, dims.border_reduce),
        "_fovea_px": dims.fovea_px,
        "_border_cells": dims.border_cells,
        "_N": dims.N,
    }


# ---------------------------------------------------------------------------
# The composite view. EXCLUSIVE sampling: every composite pixel has exactly one
# origin (centre OR ring). The centre is copied untouched; ring cells average
# (or max) anisotropic blocks of the original crop:
#   - both coords in the ring  -> r x r block
#   - ring row, centre col     -> r x 1 block (co-registered with the fovea col)
#   - centre row, centre col   -> 1 x 1 (exact copy)
# This reproduces the coordinate table of instructionsNewNN.md 4 and keeps the
# fovea bit-identical to the direct crop (tested).

def _axis_edges(dims: FoveaDims) -> np.ndarray:
    """Start offset in the original crop for each of the N composite cells (+ end)."""
    m, c, r, N = dims.border_px, dims.fovea_px, dims.border_reduce, dims.N
    po = dims.border_cells
    edges = []
    for k in range(N):
        if k < po:
            edges.append(k * r)
        elif k < po + c:
            edges.append(m + (k - po))
        else:
            edges.append(m + c + (k - po - c) * r)
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


def pad_sides(image_shape: tuple, wx0: int, wy0: int,
              dims: FoveaDims) -> tuple[int, int, int, int]:
    """How many px of the crop fall OUTSIDE the image, per side (L, T, R, B).

    ONE definition, two readers: `build_view` pads by exactly this much, and
    `edge_features` turns it into what the head is told. Computing it twice --
    the same arithmetic in two places -- is the failure this project keeps
    paying for, and here it would be silent: the net would be told about an
    edge the view does not actually have.
    """
    H, W = image_shape
    m, s = dims.border_px, dims.original_size
    x0, y0 = int(wx0) - m, int(wy0) - m
    return (max(0, -x0), max(0, -y0), max(0, x0 + s - W), max(0, y0 + s - H))


def n_edge_features(mode: str) -> int:
    """How many extra head inputs `mode` produces. 0 for 'off' -- and 0 is not a
    special case anywhere downstream: an empty vector concatenates to nothing,
    so the net built with the default is the net that was already on disk."""
    if mode not in EDGE_MODES:
        raise FoveaError("unknown_edge_inputs", f"edge_inputs '{mode}' no existe",
                         f"usa uno de {sorted(EDGE_MODES)}")
    return 0 if mode == "off" else len(EDGE_SIDES)


def edge_features(image_shape: tuple, wx0: int, wy0: int, dims: FoveaDims,
                  mode: str = "off") -> np.ndarray:
    """Is there an image edge on each side of this window, and how close?

    The composite view CANNOT say this by itself. `pad_mode: edge` replicates
    the border row/col (C11: never plain zeros, because zero means "no ink" and
    teaches a false rule) -- and that replication is, by construction,
    indistinguishable from real image that happens to look like more of the
    same. So a paragraph flush against the top of the page and a paragraph cut
    in half by the top of the VIEW produce the same input, while their labels
    differ: in the first the TL/TR corners are really there, in the second the
    real corners are somewhere above and the window is seeing the middle.

    Returns float32 (n_edge_features(mode),) in [0, 1], in EDGE_SIDES order,
    with 0 = no image edge on this side in both modes (see EDGE_MODES).

    Pure arithmetic on the geometry: no image is read. That is what lets the
    dataloader and `predict_image` get the vector from the SAME function
    (contract (5)) without either of them owning the definition.
    """
    n = n_edge_features(mode)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    H, W = image_shape
    if mode == "pad":
        # the fraction of each side's margin that is padding rather than image.
        # `border_px == 0` leaves no margin to be padding, so this mode would be
        # a constant zero -- an input that says "no edge anywhere", always, on a
        # net asking to be told about edges. The gate refuses it (fv.validation)
        # instead of training on a dead feature.
        m = dims.border_px
        if m <= 0:
            raise FoveaError(
                "edge_pad_needs_border",
                "edge_inputs='pad' con border_px=0 no puede medir nada: sin "
                "margen no hay relleno que fraccionar",
                "usa edge_inputs='dist' (mide contra la fovea, no contra el "
                "margen) o dale un border_px > 0")
        return (np.asarray(pad_sides((H, W), wx0, wy0, dims), dtype=np.float32)
                / float(m)).clip(0.0, 1.0)
    # 'dist': how close the labelled window is to each edge, in fovea units.
    # Capped at one fovea because the effect is local -- past the next window
    # along, "how far from the edge am I" stops being about the edge and starts
    # being absolute position in the page, which is not what is being asked.
    f = float(dims.fovea_px)
    d = np.asarray((int(wx0), int(wy0),
                    W - (int(wx0) + dims.fovea_px),
                    H - (int(wy0) + dims.fovea_px)), dtype=np.float32)
    return (1.0 - np.clip(d, 0.0, f) / f).astype(np.float32)


def n_input_channels(mask_channel: str = "off") -> int:
    """Cuantos canales recibe la rama que VE el anillo. 1 con 'off'."""
    if mask_channel not in MASK_MODES:
        raise FoveaError("unknown_mask_channel",
                         f"mask_channel '{mask_channel}' no existe",
                         f"usa uno de {sorted(MASK_MODES)}")
    return 1 if mask_channel == "off" else 2


def input_stack(view: np.ndarray, coverage: np.ndarray,
                mask_channel: str = "off") -> np.ndarray:
    """(C, N, N) float32: la vista y, si se pide, cuanto de cada celda es RELLENO.

    UNA definicion de como se arma la entrada, para el dataloader, `predict_image`,
    la tabla de diagnostico y las sondas. Tres copias de esto divergen en si el
    canal es cobertura o su complemento, y ese fallo entrena una red que lee la
    senal al reves sin que nada falle.

    ⚠ El canal es `1 - coverage` (RELLENO), no la cobertura. Dos razones, y la
    segunda es la que obliga:
      - orientacion igual que `edge_features`: 0 = no hay borde aqui, 1 = todo
        esto es inventado. Dos entradas sobre lo mismo con signos opuestos es
        una trampa gratis.
      - la rama se MULTIPLICA por su mascara de region antes de convolucionar, y
        eso pone a 0 lo que cae fuera. Con la cobertura, ese 0 significaria
        "inventado" justo donde en realidad significa "no es mio"; con el
        relleno significa "nada que declarar", que es lo mismo que dicen los
        pixeles de imagen que ahi tambien van a 0.
    """
    if n_input_channels(mask_channel) == 1:
        return view[None].astype(np.float32)
    return np.stack([view, 1.0 - coverage]).astype(np.float32)


def build_view(image: np.ndarray, wx0: int, wy0: int, dims: FoveaDims,
               pool_mode: str = "avg", pad_mode: str = "edge") -> tuple[np.ndarray, np.ndarray]:
    """Composite view + coverage mask for the labelled window at (wx0, wy0).

    image: full grayscale image (H, W) uint8/float. The labelled window is the
    fovea: fovea_px x fovea_px at (wx0, wy0). Returns (view (N,N) float32
    in [0,1], coverage (N,N) float32 fraction of real pixels per cell).

    Padding beyond the image border: 'edge' replicates the border row/col
    (decision C10: never plain zeros — zero means "no ink" and teaches a false
    rule); the coverage mask carries the real fraction per cell for debugging
    (F0 view), it is NOT fed to the net in v1. It is also the honest way to see
    how much of a WIDE border is replicated padding rather than context.
    """
    H, W = image.shape
    m = dims.border_px
    x0, y0 = wx0 - m, wy0 - m
    s = dims.original_size
    pad_l, pad_t, pad_r, pad_b = pad_sides((H, W), wx0, wy0, dims)
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
# Branch masks. CONTRIBUTIVE overlap: where both masks are 1 both branches
# contribute (they are applied to the INPUT, option A — masking after
# convolution was rejected, instructionsNewNN.md 7).
#
# The two overlaps are INDEPENDENT and both are barrible:
#   overlap_fovea_px   grows the BORDER branch inwards, over the fovea
#   overlap_border_px  grows the FOVEA branch outwards, over the border
# Before the reparameterisation only the first existed (and never below 1 px),
# so "how much of each region is shared" could not be asked as a question.

def build_masks(dims: FoveaDims) -> tuple[np.ndarray, np.ndarray]:
    N, po = dims.N, dims.border_cells
    pen, ob = dims.overlap_fovea_px, dims.overlap_border_cells
    center_mask = np.zeros((N, N), dtype=np.float32)
    periph_mask = np.zeros((N, N), dtype=np.float32)
    lo, hi = po - ob, N - po + ob                  # the fovea branch, grown outwards
    center_mask[lo:hi, lo:hi] = 1.0
    inner_lo, inner_hi = po + pen, N - po - pen    # the fovea, shrunk inwards
    periph_mask[:, :] = 1.0
    periph_mask[inner_lo:inner_hi, inner_lo:inner_hi] = 0.0
    return center_mask, periph_mask
