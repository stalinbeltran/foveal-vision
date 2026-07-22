"""fv-train: names only (B, C, D) — values live in stores; that rigidity is
what makes provenance hold by itself (api.md R7). device is a flag, not a
recipe field (contract (10)). ASCII output: the console is cp1252."""

from __future__ import annotations

import argparse
import sys

from fv.models.store import NetworkStore, NetworkStoreError
from fv.training.loop import train
from fv.training.recipe import RecipeStore, RecipeStoreError
from fv.training.registry import RunError


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a run from named B + C + D")
    ap.add_argument("--name", required=True, help="a NEW run name (never overwritten)")
    ap.add_argument("--window-dataset", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--device", default="cpu", help="execution (X), not part of the recipe")
    args = ap.parse_args()

    try:
        net = NetworkStore().get(args.network)
        recipe = RecipeStore().get(args.recipe)
    except (NetworkStoreError, RecipeStoreError) as e:
        print(f"\n  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    try:
        summary = train(args.name, args.window_dataset, args.network, net,
                        args.recipe, recipe, device=args.device,
                        progress=_progress)
    except RunError as e:
        print(f"\nNo se puede entrenar esto, y se ve antes del primer batch:\n\n"
              f"  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    except Exception as e:
        code = getattr(e, "code", "error")
        hint = getattr(e, "hint", "")
        print(f"\n  [{code}] {e}\n    -> {hint}", file=sys.stderr)
        return 2
    print(f"\nOK: {summary['epochs_run']} epocas, monitor {summary['monitor']} "
          f"best={summary['best']} (epoca {summary['best_epoch']})")
    return 0


def _progress(epoch: int, total: int, rec: dict) -> None:
    val = rec["val"]
    err = val["pos_err_px"]
    print(f"  epoca {epoch}/{total}  train_loss={rec['train_loss']:.4f}  "
          f"val_loss={val['loss']:.4f}  f1={val['f1']:.3f}  "
          f"pos_err_px={err if err is None else round(err, 2)}  "
          f"({rec['seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
