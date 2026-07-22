"""fv-extract: build a window dataset without the API (api.md §0).

ASCII-only output: the Windows console is cp1252 and a unicode arrow in a
--help string crashes with UnicodeEncodeError (measured in the sibling).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fv import settings
from fv.windows.extract import ExtractConfig, ExtractError, extract_windows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract labelled windows from a source into data/window-datasets/<name>")
    ap.add_argument("--source", required=True, help="source id (see the Fuentes screen)")
    ap.add_argument("--name", required=True, help="dataset name (a NEW subdir)")
    ap.add_argument("--window-size", type=int, default=16,
                    help="labelled window side = the fovea (default 16)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1, help="split seed (per image)")
    args = ap.parse_args()

    cfg = ExtractConfig(source=args.source, window_size=args.window_size,
                        stride=args.stride, val_frac=args.val_frac,
                        test_frac=args.test_frac, seed=args.seed)
    out = settings.window_datasets_root() / args.name
    try:
        manifest = extract_windows(cfg, out, progress=_progress)
    except ExtractError as e:
        print(f"\nNo se puede extraer, y se ve antes de escribir nada:\n\n"
              f"  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    except Exception as e:  # source errors carry the same shape
        code = getattr(e, "code", "error")
        hint = getattr(e, "hint", "")
        print(f"\n  [{code}] {e}\n    -> {hint}", file=sys.stderr)
        return 2
    print(f"\nOK: {manifest['num_windows']} ventanas de {manifest['num_samples']} imagenes "
          f"en {out}")
    print(f"  splits: {manifest['windows_per_split']}")
    print(f"  positivos por esquina: {manifest['positives_per_corner']}")
    if manifest["windows_per_split"]["val"] == 0:
        print("  AVISO: val vacio - este dataset no sirve para medir (entrenar se negara)")
    return 0


def _progress(done: int, total: int) -> None:
    if done % 20 == 0 or done == total:
        print(f"  {done}/{total} imagenes", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
