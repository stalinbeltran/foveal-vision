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
    """Donde se buscan las fuentes, EN ORDEN: primero la de esta maquina, luego
    la publicada en el repo de datos.

    Las dos llevan el prefijo `local/` porque son el mismo espacio de nombres: el
    `source_id` de un manifest de ventanas dice `local/dirty-1000-80px` y tiene
    que resolver contra cualquiera de las dos. Que gane la de la maquina es
    deliberado -- quien tiene la fuente de verdad (con su copia grande al lado)
    trabaja sobre ella; el repo de datos es lo que hace que una maquina RECIEN
    HECHA tenga alguna. Es la misma escalera que `artefactos.resolver`.

    Sin repo de datos clonado las dos rutas coinciden, y entonces es una sola:
    duplicarla haria que `discover_sources` listase cada fuente dos veces.
    """
    roots: list[tuple[str, Path]] = []
    # ext = settings.external_datasets_root()
    # if ext and ext.exists():
    #     roots.append(("", ext))
    vistas: set[Path] = set()
    for d in (settings.local_sources_root(), settings.published_sources_root()):
        r = Path(d).resolve()
        if d.exists() and r not in vistas:
            vistas.add(r)
            roots.append(("local/", d))
    return roots


def source_meta(root: Path) -> dict:
    """The source's own `dataset.json`, or {} — ONE reader, several callers.

    Missing or unparseable is {}, never an error: this file is written by the
    sibling generator and a source is perfectly usable without it (the labels
    are in labels.jsonl). What a caller must NOT do is invent a default for a
    field it needs — an absent key is absent (formatos.md §2), and `{}.get(k)`
    says so.
    """
    dj = Path(root) / "dataset.json"
    if not dj.exists():
        return {}
    try:
        return json.loads(dj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def discover_sources() -> list[dict]:
    """Las fuentes visibles, sin repetir.

    Una misma fuente puede estar EN LAS DOS raices (la maquina la tiene y ademas
    esta publicada en el repo de datos): es un id, no dos. Gana la primera, que
    es el mismo orden que `resolve_source` -- si listase dos y resolviese una,
    el selector de la UI ofreceria una fila que abre la otra.
    """
    out = []
    vistos: set[str] = set()
    for prefix, root in _roots():
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not (d / "labels.jsonl").exists():
                continue
            sid = prefix + d.name
            if sid in vistos:
                continue
            vistos.add(sid)
            meta = source_meta(d)
            out.append({
                "id": sid,
                "path": str(d),
                "declared_id": meta.get("id"),
                "count": meta.get("count"),
                "derived": meta.get("derived"),
                # de cual de las dos raices salio: sin esto, "esta publicada?"
                # no se puede contestar desde la UI ni desde un preflight
                "published": _es_publicada(d),
            })
    return out


def _es_publicada(d: Path) -> bool:
    try:
        return Path(d).resolve().is_relative_to(
            settings.published_sources_root().resolve())
    except (OSError, ValueError):
        return False


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

    @property
    def meta(self) -> dict:
        """This source's `dataset.json` (same reader as discover_sources)."""
        return source_meta(self.root)

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
