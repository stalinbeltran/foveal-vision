"""The two figures of section 6, drawn with Pillow.

Pillow and not matplotlib because Pillow is already a hard dependency of this
repo (`pyproject.toml`) and matplotlib is not installed anywhere in the fleet;
`scripts/demo_contrafactico.py` set the precedent. A figure that needs a `pip
install` on a machine that gets rebuilt without warning is a figure nobody sees.

  · `contact_sheet`: the K encoder kernels as DIVERGENT maps on a COMMON colour
    scale, with the norm in each title. Common scale is the whole point -- per
    kernel autoscaling makes a dead kernel look as structured as a live one.
  · `code_maps`: one input next to its K `z` maps. This is the figure that
    answers, visually, "is the resulting image more generic?".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:                       # a machine without DejaVu still draws
        return ImageFont.load_default()


def _divergent(a: np.ndarray, vmax: float) -> Image.Image:
    """blue (negative) - white (zero) - red (positive), symmetric around 0."""
    t = np.clip(a / max(vmax, 1e-12), -1.0, 1.0)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
    r = 255 - (neg * 200)
    g = 255 - (np.maximum(pos, neg) * 200)
    b = 255 - (pos * 200)
    rgb = np.stack([r, g, b], -1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _sequential(a: np.ndarray, vmax: float) -> Image.Image:
    """white (0) - dark (vmax). For `z`, which is non-negative after the ReLU."""
    t = np.clip(a / max(vmax, 1e-12), 0.0, 1.0)
    v = (255 - t * 235).astype(np.uint8)
    return Image.fromarray(np.stack([v, v, np.clip(v.astype(int) + 12, 0, 255).astype(np.uint8)], -1), "RGB")


def _grid_layout(n: int) -> tuple[int, int]:
    cols = min(8, max(1, int(math.ceil(math.sqrt(n)))))
    return cols, int(math.ceil(n / cols))


def _canvas(min_w: int, texts, font, pad: int) -> int:
    """Ancho que de verdad hace falta.

    Se mide el texto en vez de suponerlo: con K=32 y k=9 los rotulos son mas
    anchos que las celdas, y una figura recortada es una figura que hay que
    volver a generar -- que es justo lo que cuesta caro cuando el run ya no
    esta.
    """
    probe = Image.new("RGB", (1, 1))
    d = ImageDraw.Draw(probe)
    ancho = max((int(d.textlength(t, font=font)) for t in texts), default=0)
    return max(min_w, ancho + 2 * pad)


def contact_sheet(kernels: np.ndarray, out_path: Path, title: str,
                  subtitle: str = "", cell: int = 74) -> Path:
    """kernels: (K, k, k). Common colour scale, norm in each title."""
    K = kernels.shape[0]
    vmax = float(np.abs(kernels).max())
    cols, rows = _grid_layout(K)
    f_tit, f_sub, f_lab = _font(19), _font(13), _font(12)
    pad, lab_h, head = 10, 17, 30 + (18 if subtitle else 0)
    foot = (f"escala de color COMUN a los {K} kernels: +-{vmax:.3f}  "
            f"(azul negativo · blanco cero · rojo positivo)")
    W = _canvas(pad + cols * (cell + pad),
                [title], f_tit, pad)
    W = max(W, _canvas(0, [subtitle, foot] if subtitle else [foot], f_sub, pad))
    H = head + rows * (cell + lab_h + pad) + 26
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((pad, 7), title, fill="black", font=f_tit)
    if subtitle:
        d.text((pad, 29), subtitle, fill=(95, 95, 95), font=f_sub)
    for i in range(K):
        c, r = i % cols, i // cols
        x0 = pad + c * (cell + pad)
        y0 = head + r * (cell + lab_h + pad)
        tile = _divergent(kernels[i], vmax).resize((cell, cell), Image.NEAREST)
        im.paste(tile, (x0, y0))
        d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(170, 170, 170))
        norm = float(np.linalg.norm(kernels[i]))
        d.text((x0, y0 + cell + 2), f"k{i} · {norm:.2f}", fill=(60, 60, 60), font=f_lab)
    d.text((pad, H - 20), foot, fill=(95, 95, 95), font=f_sub)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return out_path


def code_maps(view: np.ndarray, z: np.ndarray, out_path: Path, title: str,
              subtitle: str = "", cell: int = 66) -> Path:
    """view: (N, N) the input. z: (K, N, N) its code. Common scale across z."""
    K = z.shape[0]
    cols, rows = _grid_layout(K)
    f_tit, f_sub, f_lab = _font(19), _font(13), _font(12)
    pad, lab_h, head = 10, 17, 30 + (18 if subtitle else 0)
    inw = cell * 2 + pad
    vm = float(np.abs(view).max())
    zmax = float(z.max())
    foot = f"los {K} mapas z comparten escala 0..{zmax:.2f}: lo claro es cero"
    W = _canvas(pad + inw + pad + cols * (cell + pad), [title], f_tit, pad)
    W = max(W, _canvas(0, [subtitle, foot] if subtitle else [foot], f_sub, pad))
    H = head + max(rows * (cell + lab_h + pad), inw + lab_h) + 26
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((pad, 7), title, fill="black", font=f_tit)
    if subtitle:
        d.text((pad, 29), subtitle, fill=(95, 95, 95), font=f_sub)

    tile = _divergent(view, vm).resize((inw, inw), Image.NEAREST)
    im.paste(tile, (pad, head))
    d.rectangle([pad, head, pad + inw - 1, head + inw - 1], outline=(120, 120, 120))
    d.text((pad, head + inw + 2), f"entrada  +-{vm:.2f}", fill=(60, 60, 60), font=f_lab)

    x_off = pad + inw + pad
    for i in range(K):
        c, r = i % cols, i // cols
        x0 = x_off + c * (cell + pad)
        y0 = head + r * (cell + lab_h + pad)
        im.paste(_sequential(z[i], zmax).resize((cell, cell), Image.NEAREST), (x0, y0))
        d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=(170, 170, 170))
        act = float((z[i] > 0).mean()) * 100
        d.text((x0, y0 + cell + 2), f"z{i} · {act:.0f}%", fill=(60, 60, 60), font=f_lab)
    d.text((pad, H - 20), foot, fill=(95, 95, 95), font=f_sub)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)
    return out_path
