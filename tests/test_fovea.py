"""Contract (2) and the sampling itself, with the spec's numbers as oracle."""

import numpy as np
import pytest

from fv.fovea import (FoveaError, border_range, build_foveated_input, build_masks,
                      build_search_space, build_view, check_dims, derive_dims,
                      dims_of, kernel_range, normalize_geometry,
                      overlap_border_range, overlap_fovea_range, reduce_range,
                      stride_range)

# The classic base, in the canonical spelling and in the pre-2026-08-25 one.
# They must be the same network — that is the whole claim of the migration.
CLASSIC = {"fovea_px": 16, "border_px": 4, "border_reduce": 2,
           "overlap_fovea_px": 2, "overlap_border_px": 0}
CLASSIC_LEGACY = {"N": 20, "c_frac": 0.8, "d": 2, "pen_frac": 0.1}


def classic():
    return derive_dims(CLASSIC)


def test_contract_02_classic_dims_match_the_spec():
    d = classic()
    assert d.fovea_px == 16
    assert d.border_px == 4
    assert d.border_cells == 2
    assert d.overlap_fovea_px == 2
    assert d.periph_band == 4
    assert d.center_band == 16
    assert d.N == 20
    assert d.original_size == 24
    # the pre-reparameterisation names still answer, as derived views
    assert (d.center_out, d.periph_out, d.periph_real, d.penetration, d.d) \
        == (16, 2, 4, 2, 2)
    assert d.c_frac == pytest.approx(0.8)


def test_contract_02_invalid_geometry_is_refused_with_reason():
    bad = dict(CLASSIC, overlap_fovea_px=8)     # 8 >= fovea//2
    assert any(p["code"] == "penetration_too_large" for p in check_dims(bad))
    with pytest.raises(FoveaError) as e:
        derive_dims(bad)
    assert e.value.code == "penetration_too_large"
    # control: a valid config passes (a check that always fails also "detects")
    assert check_dims(CLASSIC) == []


def test_contract_02_no_border_is_refused():
    problems = check_dims(dict(CLASSIC, border_px=0))
    assert any(p["code"] == "no_border" for p in problems)
    # ... unless the flat control declares itself
    assert check_dims(dict(CLASSIC, border_px=0), single_region=True) == []


# ---------------------------------------------------------------------------
# The reparameterisation itself (2026-08-25)

def test_legacy_geometry_reads_to_the_same_network_bit_for_bit():
    """The migration's whole claim: an artefact written before the change means
    exactly what it meant. Same dims, same masks, same view."""
    assert normalize_geometry(CLASSIC_LEGACY) == CLASSIC
    a, b = dims_of(CLASSIC_LEGACY), dims_of(CLASSIC)
    assert a == b
    for ma, mb in zip(build_masks(a), build_masks(b)):
        np.testing.assert_array_equal(ma, mb)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 80), dtype=np.uint8)
    np.testing.assert_array_equal(build_view(img, 24, 20, a)[0],
                                  build_view(img, 24, 20, b)[0])


def test_mixing_both_spellings_is_refused_not_reconciled():
    """The failure mode this project pays for is the same fact in two places
    resolved by precedence. Here it is detected instead."""
    with pytest.raises(FoveaError) as e:
        normalize_geometry(dict(CLASSIC, N=20, c_frac=0.8))
    assert e.value.code == "geometry_double_spec"


def test_bare_d_is_refused_because_its_meaning_changed():
    """`d` used to grow the context; it now only says how coarsely a border of
    FIXED size is condensed. Honouring it silently would build another net."""
    with pytest.raises(FoveaError) as e:
        normalize_geometry({"fovea_px": 16, "border_px": 4, "d": 2})
    assert e.value.code == "d_renamed"
    assert "border_px" in e.value.hint


def test_border_must_land_on_the_cell_grid():
    problems = check_dims(dict(CLASSIC, border_px=5))
    assert any(p["code"] == "border_not_divisible" for p in problems)


def test_border_and_reduce_are_independent():
    """The point of the spelling: the same 8 px of context, condensed two ways.
    Same real crop, different input side and different cost."""
    coarse = derive_dims({"fovea_px": 16, "border_px": 8, "border_reduce": 4,
                          "overlap_fovea_px": 2, "overlap_border_px": 0})
    fine = derive_dims({"fovea_px": 16, "border_px": 8, "border_reduce": 2,
                        "overlap_fovea_px": 2, "overlap_border_px": 0})
    assert coarse.original_size == fine.original_size == 32   # same real context
    assert (coarse.border_cells, coarse.N) == (2, 20)
    assert (fine.border_cells, fine.N) == (4, 24)


# ---------------------------------------------------------------------------
# The two overlaps, independent and both barrible

def test_overlap_border_grows_the_fovea_branch_outwards():
    d = derive_dims(dict(CLASSIC, overlap_border_px=2))
    assert d.overlap_border_cells == 1
    assert d.center_band == 18
    cm, pm = build_masks(d)
    # the outermost ring cell stays exclusive to the border branch ...
    assert cm[0, 10] == 0 and pm[0, 10] == 1
    # ... and the inner ring cell is now shared by both
    assert cm[1, 10] == 1 and pm[1, 10] == 1


def test_zero_overlap_makes_the_branches_disjoint():
    """Newly expressible: the old spelling had a floor of 1 px
    (penetration = max(1, ...)), so the control for the contributive-overlap
    choice of the spec could not even be written down."""
    d = derive_dims(dict(CLASSIC, overlap_fovea_px=0))
    cm, pm = build_masks(d)
    np.testing.assert_array_equal(cm * pm, np.zeros_like(cm))
    np.testing.assert_array_equal(cm + pm, np.ones_like(cm))


def test_overlap_border_cannot_eat_the_whole_border():
    """Mirror of penetration_too_large: the border branch keeps something of
    its own, or it is a subset of the fovea branch."""
    problems = check_dims(dict(CLASSIC, overlap_border_px=4))
    assert any(p["code"] == "overlap_border_too_large" for p in problems)
    assert check_dims(dict(CLASSIC, overlap_border_px=2)) == []


def test_overlap_border_must_land_on_the_cell_grid():
    problems = check_dims(dict(CLASSIC, overlap_border_px=1))
    assert any(p["code"] == "overlap_border_not_divisible" for p in problems)


# ---------------------------------------------------------------------------
# Ranges, computed and never written by hand

def test_contract_02b_ranges_reproduce_the_spec_examples():
    # instructionsNewNN.md §3: centro=16 -> [3,5,7]; centro=32 -> [3..15]
    assert kernel_range(16) == [3, 5, 7]
    assert kernel_range(32) == [3, 5, 7, 9, 11, 13, 15]
    # centro 16, 2 capas -> [1,2]; banda fina 4 -> [1]
    assert stride_range(16, 2) == [1, 2]
    assert stride_range(4, 2) == [1]
    ss = build_search_space(CLASSIC)
    assert ss["k_center"] == [3, 5, 7]
    assert ss["s_center"] == [1, 2]
    assert ss["s_periph"] == [1]
    assert ss["_fovea_px"] == 16
    assert ss["_N"] == 20


def test_reduce_range_is_the_divisors_of_the_border():
    assert reduce_range(4) == [1, 2, 4]
    assert reduce_range(12) == [1, 2, 3, 4, 6, 12]
    assert reduce_range(0) == [1]


def test_border_range_stays_on_the_grid_and_bounded():
    r = border_range(16, 2, max_original=48)
    assert r[0] == 2 and r[-1] == 16
    assert all(b % 2 == 0 for b in r)
    assert all(16 + 2 * b <= 48 for b in r)


def test_overlap_ranges_stop_one_short_of_eating_a_region():
    assert overlap_fovea_range(16) == list(range(0, 8))
    assert overlap_border_range(4, 2) == [0, 2]
    assert overlap_border_range(12, 2) == [0, 2, 4, 6, 8, 10]


# ---------------------------------------------------------------------------
# The sampling

def test_sampling_center_is_bit_exact_and_ring_matches_spec_table():
    """§4 table (N=20, original 24, reduce 2): ring px 0-3 pooled /2 -> px 0-1;
    centre px 4-19 copied -> px 2-17."""
    d = classic()
    rng = np.random.default_rng(0)
    crop = rng.random((24, 24)).astype(np.float32)
    view = build_foveated_input(crop, d)
    assert view.shape == (20, 20)
    # centre copied untouched (exclusive sampling)
    np.testing.assert_array_equal(view[2:18, 2:18], crop[4:20, 4:20])
    # top-left ring cell = mean of the 2x2 block
    assert view[0, 0] == pytest.approx(crop[0:2, 0:2].mean())
    # top band over a centre column: 2x1 block, co-registered with the fovea col
    assert view[0, 10] == pytest.approx(crop[0:2, 4 + 8:4 + 9].mean())
    # max pooling option
    vmax = build_foveated_input(crop, d, pool_mode="max")
    assert vmax[0, 0] == pytest.approx(crop[0:2, 0:2].max())
    np.testing.assert_array_equal(vmax[2:18, 2:18], crop[4:20, 4:20])


def test_masks_are_contributive_exactly_in_the_penetration_band():
    d = classic()
    cm, pm = build_masks(d)
    both = cm + pm
    # outer ring: only periph; core: only centre; penetration band: both
    assert both[0, 0] == 1 and pm[0, 0] == 1 and cm[0, 0] == 0
    assert both[10, 10] == 1 and cm[10, 10] == 1 and pm[10, 10] == 0
    assert both[2, 10] == 2  # penetration row: both contribute (summed)
    assert both[3, 10] == 2
    assert both[4, 10] == 1  # core starts


def test_view_padding_and_coverage():
    d = derive_dims({"fovea_px": 8, "border_px": 4, "border_reduce": 2,
                     "overlap_fovea_px": 1, "overlap_border_px": 0})
    assert (d.N, d.original_size) == (12, 16)
    img = np.full((36, 48), 200, dtype=np.uint8)
    view, cov = build_view(img, 0, 0, d)  # window at the corner: margin pads
    assert view.shape == (12, 12) and cov.shape == (12, 12)
    assert cov.min() < 1.0          # padded cells have partial coverage
    assert cov[6, 6] == 1.0         # the fovea is fully real
    view2, cov2 = build_view(img, 20, 14, d)  # interior window: no padding
    assert cov2.min() == 1.0


def test_coverage_measures_how_much_of_a_wide_border_is_padding():
    """A wide border on a small image is mostly replicated edge, not context.
    The coverage mask is what says so — the honest limit of `border_px`."""
    img = np.full((60, 80), 200, dtype=np.uint8)
    narrow = derive_dims(CLASSIC)                          # 4 px of border
    wide = derive_dims(dict(CLASSIC, border_px=16))        # 16 px of border
    _, cov_n = build_view(img, 0, 0, narrow)
    _, cov_w = build_view(img, 0, 0, wide)
    assert cov_w.mean() < cov_n.mean()
