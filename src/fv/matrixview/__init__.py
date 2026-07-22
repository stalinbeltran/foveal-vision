"""Matrix of numbers -> payload (numbers + min/max/mean + colour work).

Library-shaped: imports nothing from fv. The decisions it encodes (learned in
the sibling projects, librerias.md there):
- numbers travel, not images: the client decides the colour;
- normalisation is PER MAP (global makes low-activation maps all-black);
- truncation with notice (max_maps) so 128 filters don't kill the browser;
- the painter does not choose the colour work: signed data -> 'diverging'
  centred on 0, magnitude -> 'sequential'. The producer declares it.
"""

from __future__ import annotations

import numpy as np

MAX_MAPS = 64


def map_payload(matrix: np.ndarray, color: str, label: str | None = None) -> dict:
    m = np.asarray(matrix, dtype=np.float64)
    return {
        "label": label,
        "shape": list(m.shape),
        "min": float(m.min()) if m.size else 0.0,
        "max": float(m.max()) if m.size else 0.0,
        "mean": float(m.mean()) if m.size else 0.0,
        "color": color,  # 'sequential' | 'diverging'
        "matrix": [[float(v) for v in row] for row in m],
    }


def maps_payload(stack: np.ndarray, color: str, labels: list[str] | None = None,
                 max_maps: int = MAX_MAPS) -> dict:
    n = stack.shape[0]
    truncated = n > max_maps
    shown = min(n, max_maps)
    return {
        "count": n,
        "truncated": truncated,
        "color": color,
        "maps": [map_payload(stack[i], color,
                             labels[i] if labels else str(i)) for i in range(shown)],
    }
