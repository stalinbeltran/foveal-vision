r"""Verify EVERY sweep axis end to end: derive the base, expand the axis, and
actually run a real sweep (the same generate_sweep + run_sweep path a study
walks). An axis "works" only if its points build, train and get measured — not
if it merely validates. Proves the fix to the N-axis trap did not leave another
axis silently broken.

Runs in a throwaway store (temp dirs), on a small ws16 dataset, 1 epoch, a few
points per axis. f1 values are smoke (1 epoch); what we assert is that each axis
trains at least one point and that the fovea (and the old spelling)
are REFUSED at both gates.

    .\.venv\Scripts\python scripts\verify_axes.py [--dataset synth-b16]

ASCII output (cp1252 console).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from fv.sweeps.generate import generate_sweep
from fv.sweeps.runner import run_sweep, sweep_trials
from fv.sweeps.spec import (NETWORK_PARAMS, RECIPE_PARAMS, SweepError,
                            check_sweep)
from fv.sweeps.store import SweepStore
from fv.studies.driver import validate_plan
from fv.training.registry import RunStore

# (axis, range, cap) — cap 0 = run all points; categorical axes list every
# valid value so BOTH branches are exercised. Geometry axes use "auto": the
# range is calculated by fv.fovea, never hand-written (the project principle).
NETWORK_AXES = [
    ("border_px", "auto", 2),
    ("border_reduce", "auto", 2),
    ("overlap_fovea_px", "auto", 2),
    ("overlap_border_px", "auto", 2),
    ("n_layers", [1, 2, 3], 0),
    ("k_center", "auto", 2),
    ("k_periph", "auto", 2),
    ("s_center", "auto", 2),
    ("s_periph", "auto", 2),
    ("channels", [[16, 16], [8, 8]], 0),
    ("merge", ["concat", "sum"], 0),
    ("pool_mode", ["avg", "max"], 0),
    ("pad_mode", ["edge", "mean", "zero"], 0),
]
RECIPE_AXES = [
    ("lr", [0.001, 0.003], 0),
    ("optimizer", ["adam", "adamw", "sgd"], 0),
    ("momentum", [0.0, 0.9], 0),
    ("weight_decay", [0.0, 1e-4], 0),
    ("batch_size", [32, 64], 0),
    ("epochs", [1, 2], 0),
    ("scheduler", ["none", "cosine"], 0),
    ("patience", [0, 1], 0),
    ("lambda_pos", [1.0, 2.0], 0),
    ("pos_weight", [1.0, 2.0], 0),
    ("smooth_l1_beta", [0.05, 0.1], 0),
    ("monitor", ["val_loss", "val_f1"], 0),
    ("seed", [1, 2], 0),
]


def _run_axis(axis, rng, cap, dataset, ss, rs):
    name = f"axischk-{axis}".replace("[", "_").replace("]", "_")
    budget = {"epochs": 1}
    if cap:
        budget["points"] = cap
    try:
        enriched = generate_sweep(name, dataset, axis, rng, base_recipe="corta",
                                  objective="f1", budget=budget, sstore=ss)
    except SweepError as e:
        return {"axis": axis, "ok": False, "detail": f"generate [{e.code}] {e.message}"}
    st = run_sweep(name, ss, rs)
    trials = sweep_trials(name, ss, rs)
    trained = [t for t in trials["trials"] if t["status"] == "done"]
    return {
        "axis": axis,
        "ok": st["status"] == "done" and len(trained) >= 1,
        "detail": f"{st['done']}/{st['total']} entrenados, "
                  f"{len(enriched['discarded'])} descartados por geometria",
        "values": [t["point"].get(axis) for t in trained][:4],
    }


def _assert_refusals(dataset):
    """The fovea must be refused at BOTH gates (sweep and study), not run — and
    so must the pre-2026-08-25 spelling, by name and with its replacement."""
    out = []
    checks = [("fovea_px", "axis_breaks_window_size")]
    checks += [(a, "axis_renamed") for a in ("N", "c_frac", "pen_frac", "d")]
    for axis, want in checks:
        spec = {"space": {axis: [8, 10, 12]}, "strategy": "grid", "objective": "f1"}
        codes = {p["code"] for p in check_sweep(spec)}
        plan = {"window_dataset": dataset, "base_recipe": "corta", "objective": "f1",
                "seeds": 1, "budget": {"epochs": 1},
                "axes": [{"axis": axis, "range": [8, 10, 12]}]}
        pcodes = {p["code"] for p in validate_plan(plan)}
        ok = want in codes and want in pcodes
        out.append({"axis": axis, "ok": ok,
                    "detail": f"check_sweep={sorted(codes)} validate_plan={sorted(pcodes)}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify every sweep axis end to end")
    ap.add_argument("--dataset", default="synth-b16")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="axischk-")
    ss = SweepStore(root=os.path.join(tmp, "sweeps"))
    rs = RunStore(root=os.path.join(tmp, "runs"))

    print(f"dataset: {args.dataset}  (store temporal: {tmp})\n")
    failures = 0

    print("== ejes REHUSADOS (la fovea, y la ortografia anterior a 2026-08-25) ==")
    for r in _assert_refusals(args.dataset):
        mark = "OK " if r["ok"] else "XX "
        failures += 0 if r["ok"] else 1
        print(f"  {mark} {r['axis']:<16} {r['detail']}")

    for title, axes in (("RED (C)", NETWORK_AXES), ("RECETA (D)", RECIPE_AXES)):
        print(f"\n== ejes {title} ==")
        for axis, rng, cap in axes:
            assert axis in (NETWORK_PARAMS | RECIPE_PARAMS), axis
            r = _run_axis(axis, rng, cap, args.dataset, ss, rs)
            mark = "OK " if r["ok"] else "XX "
            failures += 0 if r["ok"] else 1
            vals = f"  valores={r['values']}" if r.get("values") else ""
            print(f"  {mark} {axis:<16} {r['detail']}{vals}")

    total = len(NETWORK_AXES) + len(RECIPE_AXES) + 2
    print(f"\n{total - failures}/{total} ejes verificados; fallos: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
