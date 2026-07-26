"""fv-oat: generate + run a single-axis sweep with a base DERIVED from the
problem (barrido-por-ejes.md P1). The only manual input is the dataset, the
axis and its range — the ~14 base fields are derived, not typed.

ASCII output (cp1252 console). Examples:

    fv-oat --name est-01-nlayers --window-dataset synth-b16 \
           --axis n_layers --range "[1,2,3]" --recipe corta --epochs 2

    fv-oat --name est-01-kcenter --window-dataset synth-b16 \
           --axis k_center --range auto
"""

from __future__ import annotations

import argparse
import json
import sys

from fv.sweeps.generate import generate_sweep
from fv.sweeps.runner import run_sweep, sweep_trials
from fv.sweeps.spec import SweepError
from fv.sweeps.store import SweepStore
from fv.sweeps.winner import suggest_winner


def _parse_range(text: str):
    if text.strip() == "auto":
        return "auto"
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate + run a single-axis OAT sweep (derived base)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--window-dataset", required=True)
    ap.add_argument("--axis", required=True, help="the ONE field of C/D to sweep")
    ap.add_argument("--range", required=True,
                    help="'auto' (geometry) or a JSON list, e.g. [3,5,7]")
    ap.add_argument("--recipe", default="corta")
    ap.add_argument("--objective", default="f1")
    ap.add_argument("--strategy", default="grid")
    ap.add_argument("--points", type=int, default=0, help="0 = all")
    ap.add_argument("--seeds", type=int, default=1,
                    help="replica seeds per axis value (>1 adds a seed axis; §11.1)")
    ap.add_argument("--epochs", type=int, default=0, help="0 = recipe default")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--c-frac", type=float, default=None)
    ap.add_argument("--winners", default=None,
                    help="JSON {field: {value, from}} carried from previous steps")
    ap.add_argument("--overrides", default=None, help="JSON {field: value} user tunables")
    ap.add_argument("--study", default=None)
    args = ap.parse_args()

    store = SweepStore()
    try:
        enriched = generate_sweep(
            args.name, args.window_dataset, args.axis, _parse_range(args.range),
            base_recipe=args.recipe, objective=args.objective, strategy=args.strategy,
            budget={"points": args.points, "epochs": args.epochs},
            device=args.device, seeds=args.seeds, c_frac=args.c_frac,
            winners=json.loads(args.winners) if args.winners else None,
            overrides=json.loads(args.overrides) if args.overrides else None,
            study=args.study, sstore=store)
        print(f"base {enriched['base_label']} (inline): "
              f"{len(enriched['points'])} puntos validos, "
              f"{len(enriched['discarded'])} descartados por geometria")
        for c in enriched.get("corrections", []):
            print(f"  correccion: {c['field']} {c['from']} -> {c['to']}  ({c['reason']})")
        state = run_sweep(args.name, store, progress=_progress)
    except SweepError as e:
        print(f"\n  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — surface code/hint if the domain set them
        code = getattr(e, "code", "error")
        hint = getattr(e, "hint", "")
        print(f"\n  [{code}] {e}\n    -> {hint}", file=sys.stderr)
        return 2
    print(f"\nestado final: {state.get('status')} "
          f"({state.get('done')}/{state.get('total')})")
    trials = sweep_trials(args.name, store)
    print(f"ranking por {trials['objective']} ({trials['direction']}) "
          f"medido en el checkpoint (best.pt, {'/'.join(trials['monitors'])}):")
    for t in trials["trials"][:10]:
        ep = f"ep {t['epoch']}/{t['epochs']}" if t["epoch"] else "sin checkpoint"
        print(f"  {t['run']}: {t['value']}  [{ep}]  {json.dumps(t['point'])}")
    if not trials["monitor_matches_objective"]:
        print(f"  aviso: el monitor ({'/'.join(trials['monitors'])}) elige la epoca "
              f"guardada y el ranking la mide con {trials['objective']}")
    try:
        sug = suggest_winner(args.name, store=store)   # delta auto (1-SE)
        print(f"\nveredicto: {'EMPATE' if sug['tie'] else 'ganador distinguible'} "
              f"(delta={sug['delta']:.4f}, {sug['delta_source']})")
        print(f"  {sug['tie_reason']}")
        print(f"  sugerido: {json.dumps(sug['suggested']['point'])} "
              f"({sug['suggested']['value']:.4f})")
    except SweepError as e:
        print(f"\nsin veredicto: {e.message}")
    return 0


def _progress(done: int, total: int, run_name: str) -> None:
    print(f"  punto {done}/{total} terminado ({run_name})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
