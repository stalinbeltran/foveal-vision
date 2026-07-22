"""Contract (2) and the sampling itself, with the spec's numbers as oracle."""

import numpy as np
import pytest

from fv.fovea import (FoveaError, build_foveated_input, build_masks,
                      build_search_space, build_view, check_dims, derive_dims,
                      downsample_range, kernel_range, stride_range)


def classic():
    return derive_dims(20, 0.8, 2, 0.1)


def test_contract_02_classic_dims_match_the_spec():
    d = classic()
    assert d.center_out == 16
    assert d.periph_out == 2
    assert d.penetration == 2
    assert d.periph_band == 4
    assert d.periph_real == 4
    assert d.original_size == 24


def test_contract_02_invalid_geometry_is_refused_with_reason():
    problems = check_dims(20, 0.8, 2, 0.4)  # penetration 8 >= center//2
    assert any(p["code"] == "penetration_too_large" for p in problems)
    with pytest.raises(FoveaError) as e:
        derive_dims(20, 0.8, 2, 0.4)
    assert e.value.code == "penetration_too_large"
    # control: a valid config passes (a check that always fails also "detects")
    assert check_dims(20, 0.8, 2, 0.1) == []


def test_contract_02_no_periphery_is_refused():
    problems = check_dims(20, 1.0, 2, 0.05)
    assert any(p["code"] == "no_periphery" for p in problems)


def test_contract_02b_ranges_reproduce_the_spec_examples():
    # instructionsNewNN.md §3: centro=16 -> [3,5,7]; centro=32 -> [3..15]
    assert kernel_range(16) == [3, 5, 7]
    assert kernel_range(32) == [3, 5, 7, 9, 11, 13, 15]
    # centro 16, 2 capas -> [1,2]; banda fina 4 -> [1]
    assert stride_range(16, 2) == [1, 2]
    assert stride_range(4, 2) == [1]
    ss = build_search_space(20, 0.8, 0.1)
    assert ss["k_center"] == [3, 5, 7]
    assert ss["s_center"] == [1, 2]
    assert ss["s_periph"] == [1]
    assert ss["_center_out"] == 16


def test_downsample_range_bounded_by_original():
    ds = downsample_range(2, 20, max_original=40)
    assert ds[0] == 1
    assert all(2 * d * 2 + 16 <= 40 for d in ds)


def test_sampling_center_is_bit_exact_and_ring_matches_spec_table():
    """§4 table (N=20, original 24, d=2): ring px 0-3 pooled /2 -> px 0-1;
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
    d = derive_dims(12, 0.667, 2, 0.1)  # fovea 8, original 16
    img = np.full((36, 48), 200, dtype=np.uint8)
    view, cov = build_view(img, 0, 0, d)  # window at the corner: margin pads
    assert view.shape == (12, 12) and cov.shape == (12, 12)
    assert cov.min() < 1.0          # padded cells have partial coverage
    assert cov[6, 6] == 1.0         # the fovea is fully real
    view2, cov2 = build_view(img, 20, 14, d)  # interior window: no padding
    assert cov2.min() == 1.0
