"""fv-train: names only (B, C, D) — values live in stores; that rigidity is
what makes provenance hold by itself (api.md R7). device is a flag, not a
recipe field (contract (10)). ASCII output: the console is cp1252."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

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
    ap.add_argument("--epochs", type=int, default=None,
                    help="cuantas epocas (pisa la de la receta). `epochs` es "
                         "guarda, no ajuste: no cambia el resultado, lo acota")
    args = ap.parse_args()

    try:
        net = NetworkStore().get(args.network)
        recipe = RecipeStore().get(args.recipe)
        if args.epochs is not None:
            if args.epochs < 1:
                print("  [bad_epochs] --epochs tiene que ser >= 1\n"
                      "    -> para continuar un run que ya existe: fv-continue",
                      file=sys.stderr)
                return 2
            # se pisa el VALOR, y el config.json del run guarda el que se uso de
            # verdad: la procedencia sigue diciendo lo que paso, no lo que decia
            # el fichero de la receta
            recipe = replace(recipe, epochs=args.epochs)
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
    _resumen(summary)
    return 0


def _resumen(summary: dict) -> None:
    print(f"\nOK: {summary['epochs_run']} epocas en total, monitor "
          f"{summary['monitor']} best={summary['best']} "
          f"(epoca {summary['best_epoch']})")
    if summary.get("stopped_early"):
        print("  paro por patience (dejo de mejorar). Para seguir igualmente:")
        print(f"    fv-continue --name {summary['run']} --more N --patience 0")
    print("  pesos:  best.pt -> evaluar   ·   last.pt -> continuar")
    print(f"    fv-continue --name {summary['run']} --more N")


def main_continue() -> int:
    """`fv-continue`: sigue un run que ya existe.

    Comando propio y no una bandera de `fv-train`, a proposito: al continuar NO
    se eligen red, dataset ni receta --salen del run-- y una bandera compartida
    invitaria a pasarlas para que se ignoren en silencio, que es la peor forma de
    no hacer caso a alguien.
    """
    from fv.training.loop import reanudar

    ap = argparse.ArgumentParser(
        description="Sigue entrenando un run que ya existe, desde su last.pt")
    ap.add_argument("--name", required=True, help="el run que ya existe")
    ap.add_argument("--more", type=int, required=True,
                    help="cuantas epocas MAS (no el total)")
    ap.add_argument("--patience", type=int, default=None,
                    help="cambia la paciencia solo para esta continuacion; 0 = "
                         "sin early-stop. Es lo UNICO ajustable al continuar")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--optimizador-limpio", action="store_true",
                    help="continua aunque el checkpoint no traiga optimizador "
                         "(runs de antes de que se guardara): la curva sufre")
    args = ap.parse_args()

    if args.more < 1:
        print("  [bad_more] --more tiene que ser >= 1", file=sys.stderr)
        return 2
    try:
        summary = reanudar(args.name, mas=args.more, patience=args.patience,
                           device=args.device, progress=_progress,
                           optimizador_limpio=args.optimizador_limpio)
    except RunError as e:
        print(f"\nNo se puede continuar, y se ve antes del primer batch:\n\n"
              f"  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\n  [{getattr(e, 'code', 'error')}] {e}\n"
              f"    -> {getattr(e, 'hint', '')}", file=sys.stderr)
        return 2
    print(f"\n  continuado desde la epoca {summary.get('continued_from')}")
    _resumen(summary)
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
