"""fv-resize: derive a reduced source without the API (api.md §0).

ASCII-only output: the Windows console is cp1252 and a unicode arrow in a
--help string crashes with UnicodeEncodeError (measured in the sibling).
"""

from __future__ import annotations

import argparse
import sys

from fv import settings
from fv.datasets.resize import RESAMPLE, ResizeConfig, ResizeError, resize_source


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total}", end="", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reduce a source into a derived one in data/sources/<name>")
    ap.add_argument("--source", required=True, help="source id (see the Fuentes screen)")
    ap.add_argument("--name", required=True, help="derived source name (a NEW subdir)")
    ap.add_argument("--width", type=int, help="target width; the height follows")
    ap.add_argument("--height", type=int, help="target height; the width follows")
    ap.add_argument("--resample", default="lanczos", choices=sorted(RESAMPLE),
                    help="for the IMAGES (default lanczos); masks are always nearest")
    args = ap.parse_args()

    cfg = ResizeConfig(source=args.source, width=args.width, height=args.height,
                       resample=args.resample)
    out = settings.local_sources_root() / args.name
    try:
        meta = resize_source(cfg, out, progress=_progress)
    except ResizeError as e:
        print(f"\nNo se puede reducir, y se ve antes de escribir nada:\n\n"
              f"  [{e.code}] {e.message}\n    -> {e.hint}", file=sys.stderr)
        return 2
    except Exception as e:  # source errors carry the same shape
        code = getattr(e, "code", "error")
        hint = getattr(e, "hint", "")
        print(f"\n  [{code}] {e}\n    -> {hint}", file=sys.stderr)
        return 2

    d = meta["derived"]
    print(f"\nOK: {meta['count']} imagenes a {d['size'][0]}x{d['size'][1]} "
          f"(escala real {d['scale'][0]:.6g} x {d['scale'][1]:.6g}) en {out}")
    print(f"    usala como: local/{args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
