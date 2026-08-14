"""A' — a derived source: the same A, at another resolution (organizacion.md A').

Moves pixels with PIL and coordinates with a recursive rescale, and rewrites
only what it must: we are a SECOND producer of a format whose first producer is
`image-text-sample-generator` (formatos.md §4.6), so every field we do not
consume is copied through untouched.

Three rules are load-bearing, and each one is a bug that was already paid for:

1. **The scale is measured from the output, not from the factor asked for.**
   Rounding the output size to whole pixels moves the real ratio, and there are
   TWO ratios (x and y) because the two roundings are independent. Scaling the
   quads by the requested factor drifts the geometry off the ink.
2. **Masks resample with NEAREST**, never interpolating. Interpolating a label
   mask invents classes that do not exist -- the continuous form of
   *absent != zero* (formatos.md §1).
3. **Coordinates rescale all, recursively, or none.** `box`, `quad`, and the
   nested `lines[]`/`words[]` all carry pixels. Rescaling the top level and
   forgetting the nested lists yields a file that looks right in the viewer and
   is wrong in the loader -- the trap measured in the sibling project.

Only reduction is allowed (`upscale_not_allowed`), and it is checked against
EVERY sample before a single byte is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from fv.datasets.loader import resolve_source, source_meta

# Images may be resampled any of these ways; masks may not (rule 2).
RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "box": Image.Resampling.BOX,
    "nearest": Image.Resampling.NEAREST,
}

DERIVED_FORMAT_OP = "resize"


class ResizeError(ValueError):
    """Same shape as SourceError: a code, a message, and what to do about it."""

    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


@dataclass(frozen=True)
class ResizeConfig:
    source: str
    width: int | None = None
    height: int | None = None
    resample: str = "lanczos"


def _scale_box(box: list, sx: float, sy: float) -> list:
    x, y, w, h = box
    return [round(x * sx, 2), round(y * sy, 2), round(w * sx, 2), round(h * sy, 2)]


def _scale_quad(quad: list, sx: float, sy: float) -> list:
    return [[round(px * sx, 2), round(py * sy, 2)] for px, py in quad]


def _rescale(node: Any, sx: float, sy: float) -> Any:
    """Every `box` and every `quad` at any depth. Rule 3: all or nothing.

    Keyed on the field name rather than on a list of known containers, so a
    level the first producer adds later (a `chars[]`, say) is rescaled the day
    it appears instead of silently keeping the old resolution.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "box" and isinstance(value, list):
                out[key] = _scale_box(value, sx, sy)
            elif key == "quad" and isinstance(value, list):
                out[key] = _scale_quad(value, sx, sy)
            else:
                out[key] = _rescale(value, sx, sy)
        return out
    if isinstance(node, list):
        return [_rescale(v, sx, sy) for v in node]
    return node


def _out_size(in_w: int, in_h: int, cfg: ResizeConfig) -> tuple[int, int]:
    """Proportional, driven by whichever dimension was asked for."""
    if cfg.width is not None:
        return cfg.width, max(1, round(in_h * cfg.width / in_w))
    return max(1, round(in_w * cfg.height / in_h)), cfg.height


def _records(root: Path) -> list[dict]:
    out = []
    with (root / "labels.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _check(cfg: ResizeConfig, records: list[dict]) -> tuple[int, int]:
    """Everything that can say no, said before anything is written."""
    if (cfg.width is None) == (cfg.height is None):
        raise ResizeError(
            "resize_needs_one_dimension",
            "hay que pedir exactamente una dimension: width O height",
            "la otra sale de la proporcion; pedir las dos deformaria la imagen")
    if cfg.resample not in RESAMPLE:
        raise ResizeError(
            "unknown_resample",
            f"remuestreo desconocido: '{cfg.resample}'",
            f"los validos son: {', '.join(sorted(RESAMPLE))}")
    if not records:
        raise ResizeError("empty_source", "la fuente no tiene muestras",
                          "revisa que su labels.jsonl no este vacio")

    sizes = {(int(r["labels"]["width"]), int(r["labels"]["height"])) for r in records}
    if len(sizes) > 1:
        raise ResizeError(
            "mixed_source_sizes",
            f"la fuente mezcla {len(sizes)} tamanos de imagen distintos",
            "el bloque 'derived' describe UN size y UNA escala, y con tamanos "
            "mezclados cualquiera de los dos mentiria; separa la fuente por tamano")

    in_w, in_h = sizes.pop()
    out_w, out_h = _out_size(in_w, in_h, cfg)
    if out_w > in_w or out_h > in_h:
        raise ResizeError(
            "upscale_not_allowed",
            f"pedido {out_w}x{out_h} desde {in_w}x{in_h}: solo se puede reducir",
            "ampliar inventa pixeles que la fuente no tiene; genera de nuevo "
            "en el proyecto hermano si necesitas mas resolucion")
    return out_w, out_h


def resize_source(
    cfg: ResizeConfig,
    out: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Write a reduced copy of `cfg.source` into `out`. Returns its dataset.json."""
    out = Path(out)
    if out.exists():
        raise ResizeError("destination_exists", f"ya existe {out}",
                          "elige otro nombre; una fuente no se sobrescribe")

    root = resolve_source(cfg.source)
    records = _records(root)
    out_w, out_h = _check(cfg, records)

    in_w = int(records[0]["labels"]["width"])
    in_h = int(records[0]["labels"]["height"])
    # Rule 1: measured from the output, and two of them.
    sx, sy = out_w / in_w, out_h / in_h

    (out / "images").mkdir(parents=True)
    image_filter = RESAMPLE[cfg.resample]
    lines = []
    for n, rec in enumerate(records):
        new = dict(rec)
        new["image"] = _resize_image(root, out, rec["image"], out_w, out_h, image_filter)
        if rec.get("mask"):
            # Rule 2: a mask is labels, not pixels.
            new["mask"] = _resize_image(root, out, rec["mask"], out_w, out_h,
                                        Image.Resampling.NEAREST)
        labels = _rescale(rec["labels"], sx, sy)
        labels["width"], labels["height"] = out_w, out_h
        new["labels"] = labels
        lines.append(json.dumps(new, ensure_ascii=False))
        if progress:
            progress(n + 1, len(records))

    (out / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The parent's provenance travels with the child: recipe, seed, spec_version
    # and -- deliberately -- the `holdout` flag, which must not be lost by the
    # act of reducing (is_holdout_source reads it, metrica-de-tarea.md §6.4).
    meta = dict(source_meta(root))
    meta.update({
        "id": out.name,
        "name": out.name,
        "count": len(records),
        "derived": {
            "from": cfg.source,
            "from_declared_id": source_meta(root).get("id"),
            "op": DERIVED_FORMAT_OP,
            "request": {"width": cfg.width} if cfg.width is not None
                       else {"height": cfg.height},
            "size": [out_w, out_h],
            "scale": [sx, sy],
            "resample": cfg.resample,
            "created": datetime.now(timezone.utc).isoformat(),
        },
    })
    (out / "dataset.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return meta


def _resize_image(root: Path, out: Path, rel: str, w: int, h: int, filt) -> str:
    src = root / rel
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.resize((w, h), filt).save(dst)
    return rel


__all__ = ["RESAMPLE", "ResizeConfig", "ResizeError", "resize_source"]
