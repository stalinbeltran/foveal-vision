"""Piece 2 — the base derivator (G/C, barrido-por-ejes.md §5)."""

import pytest

from fv.fovea import FoveaError
from fv.models.derive import base_label, derive_base, legacy_border_px


def test_derived_base_fovea_equals_window_size():
    """Contract ①a: for a given window_size, the derived config's fovea is W.

    Since the 2026-08-25 reparameterisation this is taken, not searched for —
    the fovea IS the window, so there is no N to find and nothing to loosen."""
    for W in (8, 12, 16, 20):
        out = derive_base(W)
        assert out["dims"].fovea_px == W
        assert out["config"]["fovea_px"] == W
        assert out["dims"].N == W + 2 * out["dims"].border_cells


def test_the_classic_base_is_reproduced_exactly():
    """The label every sweep, study and report on disk cites must not move."""
    out = derive_base(16)
    assert out["base_label"] == "ws16-p2-d2-L2"
    cfg = out["config"]
    assert (cfg["border_px"], cfg["border_reduce"]) == (4, 2)
    assert (cfg["overlap_fovea_px"], cfg["overlap_border_px"]) == (2, 0)
    assert out["dims"].N == 20 and out["dims"].original_size == 24


def test_odd_window_size_is_refused_with_reason():
    with pytest.raises(FoveaError) as e:
        derive_base(15)
    assert e.value.code == "window_size_must_be_even"


def test_border_px_is_the_knob_and_says_who_asked():
    out = derive_base(16, border_px=8)
    assert out["config"]["border_px"] == 8
    assert out["dims"].original_size == 32          # 8 px of real context per side
    assert out["derivation"]["field_origin"]["border_px"] == {"origin": "user"}
    assert out["base_label"] == "ws16-p4-d2-L2"


def test_flat_control_gets_no_border_and_says_why():
    """plan-cnn-plana.md: asking for the flat control and getting a net WITH a
    ring is a DIFFERENT experiment. The correction is recorded, never silent."""
    out = derive_base(16, overrides={"regions": "single", "border_reduce": 1,
                                     "n_layers": 4})
    assert out["config"]["border_px"] == 0
    assert out["base_label"] == "ws16-p0-d1-L4"
    assert any(c["field"] == "border_px" for c in out["corrections"])


def test_invalid_default_falls_to_valid_with_reason():
    """§5.2 step 4: a default invalid for this W is corrected, not fatal, and the
    correction carries its reason."""
    # a reduction that does not divide the border must fall onto a divisor
    out = derive_base(16, winners={"border_reduce": {"value": 99, "from": "test/step-0"}})
    assert out["config"]["border_reduce"] != 99
    assert any(c["field"] == "border_reduce" for c in out["corrections"])
    # and a border wider than the search bound falls back too
    wide = derive_base(16, winners={"border_px": {"value": 999, "from": "t/0"}})
    assert wide["config"]["border_px"] <= 16
    assert any(c["field"] == "border_px" for c in wide["corrections"])


def test_overlaps_are_corrected_against_their_own_ranges():
    out = derive_base(16, overrides={"overlap_border_px": 99})
    assert out["config"]["overlap_border_px"] < out["config"]["border_px"]
    assert any(c["field"] == "overlap_border_px" for c in out["corrections"])


def test_field_origin_marks_default_winner_user():
    out = derive_base(
        16,
        winners={"n_layers": {"value": 3, "from": "estudio-01/paso-1"}},
        overrides={"k_center": 3})
    fo = out["derivation"]["field_origin"]
    assert fo["n_layers"] == {"origin": "winner", "from": "estudio-01/paso-1"}
    assert fo["k_center"] == {"origin": "user"}
    assert fo["overlap_fovea_px"] == {"origin": "default"}
    # the fovea comes from the problem, and says so: it is neither a default a
    # user chose nor a winner some study carried (U1.6)
    assert fo["fovea_px"] == {"origin": "problem", "from": "window_size"}
    # a winner that expands depth also gets its default channel vector
    assert out["config"]["channels"] == [16, 16, 16]


def test_derivation_records_the_geometry_in_px():
    g = derive_base(16)["derivation"]["geometry"]
    assert g == {"fovea_px": 16, "border_px": 4, "border_reduce": 2,
                 "overlap_fovea_px": 2, "overlap_border_px": 0}


def test_base_label_shape():
    out = derive_base(16)
    assert base_label(out["dims"], 2) == "ws16-p2-d2-L2"


def test_legacy_c_frac_shim_reproduces_the_old_derivation():
    """Plans written before the change state a central FRACTION. The shim turns
    it into the length the old search would have produced — so those artefacts
    keep meaning exactly what they meant."""
    assert legacy_border_px(16, 0.8, 2) == 4          # ws16 -> N=20, 2 cells
    assert legacy_border_px(16, 1.0, 1, True) == 0    # the flat control
    assert legacy_border_px(16, 0.7, 2) == 6          # N=22 -> 3 cells
