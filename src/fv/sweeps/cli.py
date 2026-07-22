"""fv-sweep: launch or resume a sweep WITHOUT the API — on the GPU server
there may be no browser (plan.md fase 7). ASCII output (cp1252 console).

The spec is a YAML/JSON file ("receta de recorrido"), e.g.:

    window_dataset: synth-b16
    base_network: fov-16
    base_recipe: corta
    space:
      d: auto            # calculated range from fv.fovea
      k_center: [3, 5]
      lr: [0.001, 0.003]
    strategy: grid
    objective: f1
    budget: {points: 0, epochs: 2}   # points 0 = all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from fv.models.store import NetworkStore
from fv.sweeps.runner import prepare_sweep, run_sweep, sweep_trials
from fv.sweeps.spec import SweepError
from fv.sweeps.store import SweepStore
from fv.training.recipe import RecipeStore


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a configuration sweep (receta de recorrido)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", help="YAML/JSON spec file (omit to resume an existing sweep)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    store = SweepStore()
    try:
        if args.spec:
            raw = Path(args.spec).read_text(encoding="utf-8")
            spec = yaml.safe_load(raw)
            net = NetworkStore().get(spec["base_network"])
            recipe = RecipeStore().get(spec["base_recipe"])
            spec["base_network_value"] = net
            spec["base_recipe_value"] = recipe.as_dict()
            spec["device"] = args.device
            enriched = prepare_sweep(args.name, spec, net, store)
            print(f"{len(enriched['points'])} puntos validos, "
                  f"{len(enriched['discarded'])} descartados por geometria")
        else:
            store.clear_stop(args.name)
            print(f"reanudando '{args.name}'")
        state = run_sweep(args.name, store, progress=_progress)
    except SweepError as e:
        print(f"\n  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    except Exception as e:
        code = getattr(e, "code", "error")
        hint = getattr(e, "hint", "")
        print(f"\n  [{code}] {e}\n    -> {hint}", file=sys.stderr)
        return 2
    print(f"\nestado final: {state.get('status')} "
          f"({state.get('done')}/{state.get('total')})")
    trials = sweep_trials(args.name, store)
    print(f"ranking por {trials['objective']} ({trials['direction']}):")
    for t in trials["trials"][:10]:
        print(f"  {t['run']}: {t['value']}  {json.dumps(t['point'])}")
    return 0


def _progress(done: int, total: int, run_name: str) -> None:
    print(f"  punto {done}/{total} terminado ({run_name})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
