"""fv-study: create and run an OAT study end to end WITHOUT the API — the GPU
server may have no browser (plan.md fase 7). ASCII output (cp1252 console).

The study normally GUIDES (the user confirms each winner). For the short,
unattended CPU validation runs the project wants, --auto auto-confirms the
suggested winner (cost/quality rule, D-W1) and walks the whole chain.

    fv-study --name est-01 --plan studies/est-01.yaml --auto --delta 0.02

Plan file (YAML/JSON), see formatos.md §4.7:

    window_dataset: synth-b16
    base_recipe: corta
    objective: f1
    seeds: 3
    budget: {epochs: 2}
    axes:
      - {axis: n_layers, range: [1, 2, 3]}
      - {axis: "channels[i]", range: [8, 16, 32], depends_on: n_layers}
      - {axis: k_center, range: auto}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from fv.studies.driver import (StudyError, advance, confirm, create_study,
                               status)
from fv.studies.store import StudyStore
from fv.sweeps.runner import run_sweep, sweep_trials
from fv.sweeps.store import SweepStore
from fv.sweeps.winner import suggest_winner
from fv.training.registry import RunStore


def main() -> int:
    ap = argparse.ArgumentParser(description="Create and run an OAT study")
    ap.add_argument("--name", required=True)
    ap.add_argument("--plan", help="YAML/JSON plan file (omit to continue an existing study)")
    ap.add_argument("--auto", action="store_true",
                    help="auto-confirm the suggested winner and walk the whole chain")
    # default None = δ measured from the seed dispersion (1-SE). A hard 0.0 here
    # would crown the top row in the UNATTENDED path — the one that carries a
    # winner through every later step of the study.
    ap.add_argument("--delta", type=float, default=None,
                    help="margen de calidad; por defecto se mide de las semillas (1-SE)")
    ap.add_argument("--cost-metric", default="seconds_per_epoch")
    # OFF by default: a nightly chain must not pay full-image inference per step
    # unless it was asked for (metrica-de-tarea.md §3.6)
    ap.add_argument("--task-score", action="store_true",
                    help="en cada paso, mide la metrica de tarea (parrafo por "
                         "imagen) del ganador sugerido")
    args = ap.parse_args()

    store, sstore, rstore = StudyStore(), SweepStore(), RunStore()
    try:
        if args.plan:
            plan = yaml.safe_load(Path(args.plan).read_text(encoding="utf-8"))
            create_study(args.name, plan, store)
            print(f"estudio '{args.name}' creado: "
                  f"{len(plan.get('axes', []))} ejes en el plan")
        if not args.auto:
            st = status(args.name, store)
            print(f"siguiente eje: {st['next_axis']}  (done={st['done']})")
            print("usa --auto para recorrer la cadena confirmando el ganador sugerido")
            return 0
        _run_chain(args, store, sstore, rstore)
    except StudyError as e:
        print(f"\n  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "code", "error")
        hint = getattr(e, "hint", "")
        print(f"\n  [{code}] {e}\n    -> {hint}", file=sys.stderr)
        return 2
    return 0


def _run_chain(args, store, sstore, rstore) -> None:
    while True:
        st = status(args.name, store)
        if st["done"]:
            print("\nestudio completo.")
            break
        out = advance(args.name, store, sstore)
        step = out["step"]
        print(f"\npaso {step['step']}: eje {step['axis']} sobre base {step['base_label']} "
              f"({step['points']} puntos, {step['discarded']} descartados)")
        run_sweep(step["sweep"], sstore, rstore, progress=_progress)
        trials = sweep_trials(step["sweep"], sstore, rstore)
        sug = suggest_winner(step["sweep"], delta=args.delta,
                             cost_metric=args.cost_metric, store=sstore, run_store=rstore)
        point = sug["suggested"]["point"]
        band = (f" banda {sug['best']['value_min']:.4f}-{sug['best']['value_max']:.4f}"
                f" n={sug['best']['n_seeds']}" if sug["best"]["n_seeds"] > 1 else "")
        print(f"  mejor: {sug['best']['point']} ({sug['best']['value']:.4f}{band})")
        print(f"  delta={sug['delta']:.4f} ({sug['delta_source']})")
        print(f"  {sug['tie_reason']}")
        print(f"  ganador sugerido ({args.cost_metric}): {point} -> se confirma auto")
        if args.task_score:
            from fv.task.report import print_task_score
            print_task_score("  metrica de tarea del sugerido (parrafo por imagen, val):",
                             sug["suggested"].get("runs", []), store=rstore)
        confirm(args.name, point, store)


def _progress(done: int, total: int, run_name: str) -> None:
    print(f"    punto {done}/{total} terminado ({run_name})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
