"""Piece 2 — the base derivator (G/C, barrido-por-ejes.md §5)."""

import pytest

from fv.fovea import FoveaError
from fv.models.derive import base_label, derive_base, derive_geometry


def test_derived_base_fovea_equals_window_size():
    """Contract ①a: for a given window_size, the derived config's fovea is W."""
    for W in (8, 12, 16, 20):
        out = derive_base(W)
        assert out["dims"].center_out == W
        assert out["config"]["N"] == out["dims"].N


def test_smallest_n_wins_the_tie(monkeypatch):
    """D-G2: when several even N give the same fovea, pick the smallest."""
    N, cf, reason = derive_geometry(16, c_frac_target=0.8)
    assert N == 20 and reason is None          # ws16 -> N=20, periph_out=2
    assert derive_base(16)["base_label"] == "ws16-p2-d2-L2"


def test_odd_window_size_is_refused_with_reason():
    with pytest.raises(FoveaError) as e:
        derive_base(15)
    assert e.value.code == "window_size_must_be_even"


def test_loosening_c_frac_records_its_reason():
    """D-G3: if no even N hits W at the default c_frac, c_frac is loosened and
    the effective value travels with its reason (never blind)."""
    # force an impossible target so the exact path fails, the fallback fires
    N, cf, reason = derive_geometry(16, c_frac_target=0.999, c_frac_tol=0.2)
    assert reason is not None
    assert abs(cf - 16 / N) < 1e-9            # cf hits the fovea exactly
    from fv.fovea import round_to_even
    assert round_to_even(N * cf) == 16


def test_invalid_default_falls_to_valid_with_reason(monkeypatch):
    """§5.2 step 4: a default invalid for this W is corrected, not fatal, and the
    correction carries its reason."""
    import fv.models.derive as d
    # a winner with an absurd d must be corrected against downsample_range
    out = derive_base(16, winners={"d": {"value": 99, "from": "test/step-0"}})
    assert out["config"]["d"] != 99
    assert any(c["field"] == "d" for c in out["corrections"])


def test_field_origin_marks_default_winner_user():
    out = derive_base(
        16,
        winners={"n_layers": {"value": 3, "from": "estudio-01/paso-1"}},
        overrides={"k_center": 3})
    fo = out["derivation"]["field_origin"]
    assert fo["n_layers"] == {"origin": "winner", "from": "estudio-01/paso-1"}
    assert fo["k_center"] == {"origin": "user"}
    assert fo["pen_frac"] == {"origin": "default"}
    # a winner that expands depth also gets its default channel vector
    assert out["config"]["channels"] == [16, 16, 16]


def test_base_label_shape():
    out = derive_base(16)
    assert base_label(out["dims"], 2) == "ws16-p2-d2-L2"
