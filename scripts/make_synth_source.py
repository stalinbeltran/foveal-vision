"""Generate a small synthetic source in data/sources/<name>/ for tests and E2E.

Same schema as image-text-sample-generator (labels.jsonl + dataset.json +
images/): light background, dark paragraph blocks made of text-like line
strokes, quad recorded per block. Small images so the whole source fits the
1 GB images budget with huge margin and epochs stay in seconds on CPU.

Usage: .venv\\Scripts\\python scripts\\make_synth_source.py --name synth-01 --count 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def make_image(rng: np.random.Generator, W: int, H: int):
    img = np.full((H, W), 235, dtype=np.uint8)
    img = (img.astype(np.int16) + rng.integers(-8, 8, size=(H, W))).clip(0, 255).astype(np.uint8)
    blocks = []
    n_blocks = int(rng.integers(1, 4))
    placed: list[tuple[int, int, int, int]] = []
    for b in range(n_blocks):
        for _attempt in range(20):
            pw = int(rng.integers(max(16, W // 5), max(20, W // 2)))
            ph = int(rng.integers(max(10, H // 6), max(14, H // 3)))
            x0 = int(rng.integers(1, max(2, W - pw - 1)))
            y0 = int(rng.integers(1, max(2, H - ph - 1)))
            box = (x0, y0, x0 + pw, y0 + ph)
            if all(box[2] <= p[0] or box[0] >= p[2] or box[3] <= p[1] or box[1] >= p[3]
                   for p in placed):
                placed.append(box)
                break
        else:
            continue
        x0, y0, x1, y1 = placed[-1]
        line_h = 3
        gap = 2
        yy = y0
        while yy + line_h <= y1:
            line_w = int((x1 - x0) * rng.uniform(0.7, 1.0))
            ink = int(rng.integers(20, 70))
            img[yy:yy + line_h, x0:x0 + line_w] = ink
            # word gaps
            n_gaps = max(1, line_w // 8)
            for _ in range(n_gaps):
                gx = x0 + int(rng.integers(0, max(1, line_w - 2)))
                img[yy:yy + line_h, gx:gx + 2] = 235
            yy += line_h + gap
        quad = [[float(x0), float(y0)], [float(x1), float(y0)],
                [float(x1), float(y1)], [float(x0), float(y1)]]
        blocks.append({"block_id": f"b{b}", "kind": "paragraph", "angle": 0.0,
                       "box": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                       "quad": quad})
    return img, blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--root", default=None, help="defaults to <repo>/data/sources")
    args = ap.parse_args()

    root = Path(args.root) if args.root else \
        Path(__file__).resolve().parents[1] / "data" / "sources"
    out = root / args.name
    if out.exists():
        print(f"ya existe {out} - no se sobrescribe", file=sys.stderr)
        return 2
    (out / "images").mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    lines = []
    for i in range(args.count):
        img, blocks = make_image(rng, args.width, args.height)
        rel = f"images/{i:06d}.png"
        Image.fromarray(img).save(out / rel)
        lines.append(json.dumps({
            "index": i, "image": rel,
            "labels": {"image_id": f"{args.name}/{i:06d}",
                       "width": args.width, "height": args.height,
                       "blocks": blocks}}))
    (out / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "dataset.json").write_text(json.dumps({
        "id": args.name, "name": args.name, "count": args.count,
        "seed": args.seed, "synthetic": True}, indent=2), encoding="utf-8")
    print(f"OK: {args.count} imagenes {args.width}x{args.height} en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
