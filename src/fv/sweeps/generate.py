"""H (P1) — the generator: write the sweep recipe the human would have written.

The manual flow was: hand-write a full C config (~14 fields), name it, sweep one
axis, read the ranking, repeat. This automates the human, it does not invent a
path: derive the base from the problem (fv.models.derive), fix winners carried
from previous steps, declare ONE axis with its range, and hand the result to the
EXISTING sweep machinery (prepare_sweep). The only manual input left is
dataset + which axis + its range (barrido-por-ejes.md §0, §8).

The generator is NOT a laxer gate (§10): it validates the base with the same
check_run every training door uses, BEFORE reserving anything, and each expanded
point goes through check_network via prepare_sweep.
"""

from __future__ import annotations

from fv.models.derive import derive_base
from fv.sweeps.runner import prepare_sweep
from fv.sweeps.spec import SweepError
from fv.sweeps.store import SweepStore
from fv.training.recipe import RecipeStore
from fv.validation import check_run
from fv.windows.store import WindowDatasetStore


def build_generated_spec(window_dataset: str, axis: str, axis_range,
                         *, base_recipe: str = "corta", base_recipe_value: dict | None = None,
                         objective: str = "f1", budget: dict | None = None,
                         strategy: str = "grid", device: str = "cpu", seed: int = 1,
                         winners: dict | None = None, overrides: dict | None = None,
                         c_frac: float | None = None, study: str | None = None,
                         wstore: WindowDatasetStore | None = None,
                         rstore: RecipeStore | None = None) -> tuple[dict, dict]:
    """Derive the inline base from B's window_size, validate it with check_run,
    and build the sweep spec (D-H2: base_network=null + base_label + derivation +
    a single-axis space). Returns (spec, derived). Raises SweepError on a base
    that would not train — the reason travels, nothing is written."""
    wstore = wstore or WindowDatasetStore()
    manifest = wstore.manifest(window_dataset)
    window_size = int(manifest["config"]["window_size"])

    derived = derive_base(window_size, winners=winners, overrides=overrides, c_frac=c_frac)
    base = derived["config"]

    problems = check_run(manifest, base)   # the SAME gate, before reserving (§10.1)
    if problems:
        p = problems[0]
        raise SweepError(p["code"], p["message"], p["hint"])

    if base_recipe_value is None:
        base_recipe_value = (rstore or RecipeStore()).get(base_recipe).as_dict()

    spec = {
        "window_dataset": window_dataset,
        "base_network": None,                       # inline (U4/D-H2): no name
        "base_label": derived["base_label"],
        "base_network_value": base,
        "base_recipe": base_recipe,
        "base_recipe_value": base_recipe_value,
        "derivation": derived["derivation"],        # §5, §7.2 — how the base was reached
        "corrections": derived["corrections"],      # invalid defaults fixed, with reason
        "space": {axis: axis_range},                # the ONE axis (U5)
        "strategy": strategy,
        "objective": objective,
        "budget": budget or {},
        "device": device,
        "seed": seed,
    }
    if study:
        spec["study"] = study
    return spec, derived


def generate_sweep(name: str, window_dataset: str, axis: str, axis_range,
                   *, sstore: SweepStore | None = None, **kwargs) -> dict:
    """Build + validate + persist the generated sweep. Returns the enriched spec
    (points declared, discards with reasons) — ready for run_sweep, exactly like
    a named-base sweep."""
    spec, _derived = build_generated_spec(window_dataset, axis, axis_range, **kwargs)
    return prepare_sweep(name, spec, spec["base_network_value"], sstore)
