"""A — the source: images + paragraph geometry, produced by another project.

We consume from labels.jsonl: index, image path, labels.{width,height} and
blocks[].{block_id,kind,angle,quad} with quad (4,2) clockwise from TL
(SAMPLE_FORMAT.md of image-text-sample-generator). Two roots: the external one
(FV_DATASETS_ROOT, read-only) and the local data/sources/ for derived and
synthetic sources; ids from the local root carry the prefix 'local/'.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from fv import settings


@dataclass(frozen=True)
class Block:
    block_id: str
    kind: str
    angle: float
    quad: np.ndarray  # (4, 2) clockwise from TL

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = self.quad[:, 0], self.quad[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@dataclass(frozen=True)
class Sample:
    index: int
    width: int
    height: int
    image_path: Path
    blocks: list[Block] = field(default_factory=list)

    def load_image(self) -> np.ndarray:
        img = Image.open(self.image_path).convert("L")
        return np.asarray(img, dtype=np.uint8)


class SourceError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def _roots() -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    # ext = settings.external_datasets_root()
    # if ext and ext.exists():
    #     roots.append(("", ext))
    local = settings.local_sources_root()
    if local.exists():
        roots.append(("local/", local))
    return roots


def discover_sources() -> list[dict]:
    out = []
    for prefix, root in _roots():
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not (d / "labels.jsonl").exists():
                continue
            meta = {}
            dj = d / "dataset.json"
            if dj.exists():
                try:
                    meta = json.loads(dj.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    meta = {}
            out.append({
                "id": prefix + d.name,
                "path": str(d),
                "declared_id": meta.get("id"),
                "count": meta.get("count"),
                "derived": meta.get("derived"),
            })
    return out


def resolve_source(source_id: str) -> Path:
    for prefix, root in _roots():
        if prefix and source_id.startswith(prefix):
            p = root / source_id[len(prefix):]
            if (p / "labels.jsonl").exists():
                return p
        elif not prefix:
            p = root / source_id
            if (p / "labels.jsonl").exists():
                return p
    known = ", ".join(s["id"] for s in discover_sources()) or "(ninguna)"
    raise SourceError(
        "source_not_found",
        f"no existe la fuente '{source_id}'",
        f"las fuentes disponibles son: {known}")


class SourceDataset:
    """Reader over one source. samples() parses the whole labels.jsonl — it is
    for the extractor, the only consumer that needs every block. To look at ONE
    image use sample_at(index) (offsets are cached lazily)."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.root = resolve_source(source_id)
        self.labels_path = self.root / "labels.jsonl"
        self._offsets: list[int] | None = None

    def _parse_line(self, line: str) -> Sample:
        rec = json.loads(line)
        labels = rec.get("labels", {})
        blocks = [
            Block(block_id=b.get("block_id", ""), kind=b.get("kind", ""),
                  angle=float(b.get("angle", 0.0)),
                  quad=np.asarray(b["quad"], dtype=np.float32))
            for b in labels.get("blocks", []) if "quad" in b
        ]
        return Sample(index=int(rec["index"]),
                      width=int(labels.get("width", 0)),
                      height=int(labels.get("height", 0)),
                      image_path=self.root / rec["image"],
                      blocks=blocks)

    def samples(self) -> list[Sample]:
        out = []
        with self.labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(self._parse_line(line))
        return out

    def _ensure_offsets(self) -> list[int]:
        if self._offsets is None:
            offsets, pos = [], 0
            with self.labels_path.open("rb") as f:
                for raw in f:
                    if raw.strip():
                        offsets.append(pos)
                    pos += len(raw)
            self._offsets = offsets
        return self._offsets

    def __len__(self) -> int:
        return len(self._ensure_offsets())

    def sample_at(self, index: int) -> Sample:
        offsets = self._ensure_offsets()
        if index < 0 or index >= len(offsets):
            raise SourceError("sample_not_found",
                              f"la fuente '{self.source_id}' no tiene la imagen {index}",
                              f"indices validos: 0..{len(offsets) - 1}")
        with self.labels_path.open("rb") as f:
            f.seek(offsets[index])
            return self._parse_line(f.readline().decode("utf-8"))
