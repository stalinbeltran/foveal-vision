"""C `regions`: the flat-CNN control (F12, docs/plan-cnn-plana.md).

The first test is the one that matters: 'split' must be bit-identical to what
was there before this field existed, because a sweep of 20 runs is training
against it right now and the whole point of the control is comparability.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fv.fovea import build_masks, check_dims, dims_of, is_single_region
from fv.models.builder import build_model, full_config, network_trace
from fv.validation import check_network, check_run

# the L4 winner of plan-40h, which the p40-lr-L4 sweep is training right now
# (it WAS N=20, c_frac=0.8, d=2, pen_frac=0.1 before the 2026-08-25 spelling)
L4 = {"fovea_px": 16, "border_px": 4, "border_reduce": 2, "overlap_fovea_px": 2,
      "overlap_border_px": 0, "n_layers": 4, "channels": [16, 16, 16, 16]}
WS = 16  # window_size of dirty1000-80px-16px

MANIFEST = {"config": {"window_size": WS}, "has_images": True,
            "windows_per_split": {"val": 100}}


def _forward(cfg: dict, seed: int = 7) -> torch.Tensor:
    torch.manual_seed(seed)
    model = build_model(cfg)
    torch.manual_seed(seed + 1)
    n = dims_of(cfg).N
    x = torch.randn(3, 1, n, n)
    with torch.no_grad():
        return model(x)


# ---------------------------------------------------------------- no regression

def test_split_is_the_default_and_absent_means_split():
    assert full_config(L4)["regions"] == "split"
    assert not is_single_region(L4)
    assert not is_single_region({})


def test_split_is_bit_identical_with_and_without_the_new_field():
    """An artefact written before `regions` existed must build the same net."""
    a = _forward(dict(L4))
    b = _forward(dict(L4, regions="split"))
    assert torch.equal(a, b)


def test_the_running_sweeps_network_is_unchanged():
    """Golden numbers of the net p40-lr-L4 is training (measured 2026-08-09).
    If this moves, the flat-CNN work has silently changed the champion."""
    t = network_trace(L4)
    assert t["flat_features"] == 12800
    assert t["num_params"] == 168652
    assert t["dims"]["center_out"] == 16
    assert t["dims"]["periph_out"] == 2
    assert t["dims"]["original_size"] == 24
    assert sorted(t["branch_out"]) == ["center", "periph"]


def test_split_module_names_are_stable_so_checkpoints_still_load():
    keys = set(build_model(L4).state_dict())
    assert "center_convs.0.weight" in keys and "periph_convs.0.weight" in keys
    assert "center_mask" in keys and "periph_mask" in keys


def test_build_masks_is_untouched_by_this_feature():
    """The shared geometry function every domain imports: same numbers as ever."""
    cm, pm = build_masks(dims_of(L4))
    assert cm.shape == (20, 20) and pm.shape == (20, 20)
    assert int(cm.sum()) == 16 * 16          # centre square, periph_out=2
    assert cm[0, 0] == 0 and pm[0, 0] == 1   # outer ring: periphery only
    assert cm[10, 10] == 1 and pm[10, 10] == 0   # core: centre only
    assert cm[3, 3] == 1 and pm[3, 3] == 1       # penetration band: both


# ---------------------------------------------------------------- the control

def test_no_border_is_refused_for_split_and_allowed_for_single():
    flat = {"fovea_px": WS, "border_px": 0, "border_reduce": 1,
            "overlap_fovea_px": 2, "overlap_border_px": 0}
    geom = {"fovea_px": WS, "border_px": 0, "border_reduce": 1}
    codes = [p["code"] for p in check_dims(geom)]
    assert "no_border" in codes
    assert check_dims(geom, single_region=True) == []
    assert [p["code"] for p in check_network(flat)] == ["no_border"]
    assert check_network(dict(flat, regions="single")) == []


def test_single_has_one_unmasked_branch():
    cfg = full_config(dict(L4, border_px=0, border_reduce=1, regions="single"))
    model = build_model(cfg)
    assert model.single
    keys = set(model.state_dict())
    assert not any(k.startswith("periph_convs") for k in keys)
    assert "center_mask" not in keys and "periph_mask" not in keys
    # one branch over 16x16 with 16 channels, stride 1
    assert model.flat_features == WS * WS * 16
    assert list(model.kernels()) == ["single"]
    with torch.no_grad():
        out = model(torch.zeros(2, 1, WS, WS))
    assert out.shape == (2, 4, 3)


def test_an_unknown_regions_is_refused_at_the_gate():
    problems = check_network(dict(L4, regions="plana"))
    assert [p["code"] for p in problems] == ["unknown_regions"]
    assert "single" in problems[0]["hint"] and "split" in problems[0]["hint"]


def test_single_does_not_validate_the_branch_it_does_not_have():
    """k_periph/merge describe nothing in 'single' — they must not refuse it."""
    cfg = dict(L4, border_px=0, border_reduce=1, regions="single",
               k_periph=99, merge="sum", s_center=2, s_periph=1)
    assert check_network(cfg) == []
    # ...but for 'split' the same values are still refused
    assert [p["code"] for p in check_network(dict(L4, k_periph=99))] != []


# ------------------------------------------------------- the family of §3

FAMILY = {
    "base foveada L4":      dict(L4),
    "A mismo tensor":       dict(L4, border_px=2, border_reduce=1, regions="single"),
    "B misma area":         dict(L4, border_px=4, border_reduce=1, regions="single"),
    "C solo la ventana":    dict(L4, border_px=0, border_reduce=1, regions="single"),
    "E foveada sin comprimir": dict(L4, border_px=2, border_reduce=1),
}


@pytest.mark.parametrize("label", sorted(FAMILY))
def test_every_control_passes_the_same_gate_as_any_other_run(label):
    """All six controls train on the SAME B: contract (1)a holds for each."""
    cfg = full_config(FAMILY[label])
    assert check_run(MANIFEST, cfg) == [], label
    assert dims_of(cfg).fovea_px == WS, label


def test_the_controls_see_what_the_plan_says_they_see():
    areas = {k: dims_of(full_config(v)).original_size for k, v in FAMILY.items()}
    assert areas["base foveada L4"] == 24
    assert areas["A mismo tensor"] == 20
    assert areas["B misma area"] == 24      # same area as the base, uncompressed
    assert areas["C solo la ventana"] == WS
    assert areas["E foveada sin comprimir"] == 20


def test_the_derivation_gives_the_flat_base_that_was_asked_for():
    """Measured 2026-08-09: asking for the flat base produced ws16-p1-d1-L4 —
    a net WITH a ring. derive_geometry had a '>=1 periphery' floor, so it
    quietly loosened c_frac to 16/18 and returned a different control. It never
    lied (the reason was recorded), but nothing refused either.

    The 2026-08-25 spelling removes the search that caused it: the border is a
    length the caller states, so 0 is 0. The legacy shim still reproduces what
    the old c_frac meant, for plans written before the change."""
    from fv.models.derive import base_label, derive_base, legacy_border_px

    assert legacy_border_px(WS, 1.0, 1, single_region=True) == 0
    # split still could not go without a ring: c_frac=1 meant N=18, one cell
    assert legacy_border_px(WS, 1.0, 1) == 1

    out = derive_base(WS, overrides={"regions": "single", "border_reduce": 1,
                                     "n_layers": 4}, border_px=0)
    assert out["config"]["border_px"] == 0
    assert out["dims"].N == WS and out["dims"].border_cells == 0
    assert base_label(out["dims"], 4) == "ws16-p0-d1-L4"
    # an explicitly asked-for border is reported as the user's, not as a default
    assert out["derivation"]["field_origin"]["border_px"] == {"origin": "user"}


def test_a_study_can_declare_the_base_network_it_runs_on():
    """The paired comparison (§3.1): the SAME plan over the foveated base and
    over the flat control. Before this, a study derived C from the window_size
    alone and there was no way to say 'run this plan on the flat CNN'."""
    from fv.studies.driver import validate_plan

    plan = {"window_dataset": "b", "base_recipe": "corta", "objective": "f1",
            "seeds": 1, "axes": [{"axis": "lr", "range": [0.001, 0.002]}],
            "base_network": {"regions": "single", "border_reduce": 1,
                             "n_layers": 4},
            "border_px": 0}
    assert validate_plan(plan) == []
    # and the gate refuses, BEFORE creating anything, what would break later
    def codes(p):
        return [x["code"] for x in validate_plan({**plan, **p})]
    assert codes({"base_network": {"perifería": 0}}) == ["unknown_base_network_field"]
    assert codes({"base_network": {"fovea_px": 24}}) == ["base_network_breaks_window_size"]
    assert codes({"base_network": {"N": 24}}) == ["base_network_uses_old_geometry"]
    assert codes({"base_network": [1, 2]}) == ["base_network_must_be_a_map"]
    assert codes({"border_px": -1}) == ["border_px_out_of_range"]
    assert codes({"border_px": 1.5}) == ["border_px_out_of_range"]
    # a plan that states BOTH spellings is refused, never reconciled
    assert codes({"c_frac": 1.0}) == ["plan_double_geometry"]


def test_the_flat_control_really_is_cheaper_in_the_head():
    """The head holds ~97% of the parameters, so dropping a branch halves flat.
    Recorded because control D exists to compensate exactly this."""
    base = network_trace(full_config(FAMILY["base foveada L4"]))
    b = network_trace(full_config(FAMILY["B misma area"]))
    assert b["flat_features"] == 24 * 24 * 16      # one branch
    assert base["flat_features"] == 2 * 20 * 20 * 16   # two branches
    assert b["num_params"] < base["num_params"]
