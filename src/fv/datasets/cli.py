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


def main_publish() -> int:
    """`fv-publish-source`: deja una fuente en el repo de datos, para que viaje.

    Existe como comando propio y no como bandera de `fv-resize` porque son dos
    decisiones distintas: reducir es producir un dato, publicar es decidir que
    ese dato SOBREVIVE a rehacer la maquina.
    """
    import argparse

    from fv import settings
    from fv.datasets.publish import PublishError, publish_source, verify_against_windows

    ap = argparse.ArgumentParser(
        description="Publica una fuente en el repo de datos (viaja por git)")
    ap.add_argument("--source", required=True, help="p.ej. local/dirty-1000-80px")
    ap.add_argument("--solo-comprobar", action="store_true",
                    help="di si cuadra con los windows.npz y no copies nada")
    ap.add_argument("--force", action="store_true",
                    help="publica aunque NO cuadre, o reemplaza el destino")
    a = ap.parse_args()

    try:
        if a.solo_comprobar:
            v = verify_against_windows(a.source)
            print(f"  cuadran      : {v['comprobados'] or '(ninguno comprobable)'}")
            print(f"  sin windows.npz: {v['sin_npz'] or '(ninguno)'}")
            print(f"  DISCREPAN    : {v['discrepan'] or '(ninguno)'}")
            print(f"  concluyente  : {v['concluyente']}")
            return 0 if v["concluyente"] else 2
        r = publish_source(a.source, force=a.force)
    except PublishError as e:
        print(f"[{e.code}] {e.message}")
        if e.hint:
            print(f"  -> {e.hint}")
        return 1
    print(f"publicada '{r['source']}' -> {r['to']}")
    print(f"  {r['images']} imagenes, {r['bytes']/1e6:.2f} MB")
    print(f"  comprobada contra windows.npz: {r['verified']}")
    print(f"  OJO: falta el commit, en {settings.data_root()}")
    return 0
