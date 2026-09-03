#!/usr/bin/env python3
"""Coge N ventanas al azar, las pasa por la sonda y pinta entrada contra salida.

    ../../.venv/bin/python nn/reconstruir.py                 # 10 ventanas
    ../../.venv/bin/python nn/reconstruir.py --n 6 --semilla 7

Es un autoencoder: su salida ES la reconstruccion de su entrada, asi que
"entrada contra salida" es literalmente lo que mide si ha aprendido algo.

⚠⚠ PERO "reconstruye bien" NO ES "ha aprendido". Esa es la trampa que este
experimento midio, y por eso la figura enfrenta los DOS brazos:

  · lambda=0        reconstruye PERFECTO (R2 = 1,000) porque aprende la
                    IDENTIDAD -- kernels delta, sigma 0,49 px contra 1,41 de uno
                    aleatorio. Copia la entrada. No hay filtros.
  · lambda calibrada  reconstruye PEOR (R2 ~ 0,88) y ESO es lo interesante: con
                    el codigo forzado a ser disperso (~5-7 % de activacion, 16
                    canales vivos, 0 muertos) tiene que quedarse con lo que
                    importa. Lo que pierda dice que NO capturo.

La red usa `modelo.py`, que no importa nada del repo. Los DATOS si vienen de
`fv` (es donde vive el dataset y el constructor de la vista foveada, y
reconstruirlo aqui mediria una entrada que la red nunca vio).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(REPO / "src"))

from modelo import cargar                                   # noqa: E402
from fv import settings                                     # noqa: E402
from fv.fovea import build_view, dims_of                     # noqa: E402
from fv.models.builder import NETWORK_DEFAULTS               # noqa: E402

# Las MISMAS constantes con las que se entreno (resultados/resumen.json).
SIGMA, EPS, VAR = 2.0, 0.0148309962823987, 0.3065567910671234
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _gauss1d(sigma, radio):
    x = torch.arange(-radio, radio + 1, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _borrosa(x, g):
    r = (g.numel() - 1) // 2
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, g.view(1, 1, -1, 1))


def normaliza(x, sigma=SIGMA, eps=EPS):
    g = _gauss1d(sigma, max(1, int(round(3 * sigma))))
    mu = _borrosa(x, g)
    var = _borrosa((x - mu) ** 2, g).clamp_min(0.0)
    return (x - mu) / (var.sqrt() + eps)


def ventanas_al_azar(n: int, semilla: int, split: int = 1):
    """n vistas 20x20 CRUDAS del split indicado (1 = validacion)."""
    dims = dims_of(dict(NETWORK_DEFAULTS))
    npz = settings.window_datasets_root() / "dirty1000-80px-16px-r20260827" / "windows.npz"
    z = np.load(npz)
    sel = np.flatnonzero(z["split"] == split)
    rng = np.random.default_rng(semilla)
    idx = rng.choice(sel, size=n, replace=False)
    lookup = {int(a): i for i, a in enumerate(z["images_sample_idx"])}
    out, quien = [], []
    for i in idx:
        fila = lookup[int(z["sample_idx"][i])]
        wx, wy = int(z["window_xy"][i, 0]), int(z["window_xy"][i, 1])
        v, _ = build_view(z["images"][fila], wx, wy, dims, pool_mode="avg", pad_mode="edge")
        out.append(v)
        quien.append({"indice": int(i), "imagen": fila, "ventana_xy": [wx, wy]})
    return torch.from_numpy(np.stack(out))[:, None], quien


# ------------------------------------------------------------------ pintar
def _font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except OSError:
        return ImageFont.load_default()


def _gris(a, vmin=0.0, vmax=1.0):
    """La vista tal cual: `build_view` entrega [0,1] con 1 = papel, 0 = tinta.

    ⚠ NO se invierte. La primera version lo hacia y pintaba el papel de negro:
    una columna casi toda negra parece un fallo de datos y no lo era.
    """
    t = np.clip((a - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    v = (t * 255).astype(np.uint8)
    return Image.fromarray(np.stack([v, v, v], -1), "RGB")


def _diverge(a, vmax):
    t = np.clip(a / max(vmax, 1e-9), -1, 1)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
    rgb = np.stack([255 - neg * 200, 255 - np.maximum(pos, neg) * 200, 255 - pos * 200], -1)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")


def _calor(a, vmax):
    t = np.clip(a / max(vmax, 1e-9), 0, 1)
    rgb = np.stack([255 - t * 40, 255 - t * 215, 255 - t * 235], -1)   # blanco -> rojo
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")


def _corto(nombre: str) -> str:
    """`k9-K16-lcal-s1` -> `k9 λ calibrada`. Los titulos tienen que caber."""
    partes = nombre.split("-")
    lam = "λ calibrada" if "lcal" in nombre else "λ = 0"
    return f"{partes[0]} {lam}"


def figura(vista, entrada, salidas, filas, out_path, titulo):
    n, gap = vista.shape[0], 10
    nombres = list(salidas)
    cols = ["la ventana|(vista 20x20)", "entrada a la red|(contraste normaliz.)"]
    cols += [f"salida|{_corto(nm)}" for nm in nombres]
    cols += [f"|error||{_corto(nombres[-1])}"]
    f_tit, f_col, f_lab = _font(20), _font(13), _font(12)
    # El ancho de celda lo decide el TEXTO, no un numero tecleado: con los
    # titulos largos la version anterior los solapaba y no se sabia que columna
    # era cual.
    sonda = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    ancho_txt = max(int(sonda.textlength(l, font=f_col))
                    for c in cols for l in c.split("|") if l)
    cel = max(104, ancho_txt + 4)
    izq, cab, pie = 46, 78, 30
    # El margen derecho tambien se MIDE: con los nombres de modelo dentro, un
    # numero tecleado recortaba justo los R2, que es lo que hay que leer.
    ancho_r2 = max(int(sonda.textlength(f"{_corto(nm)}:  R² +0.000", font=f_lab))
                   for nm in nombres)
    W = izq + len(cols) * (cel + gap) + ancho_r2 + 16
    H = cab + n * (cel + gap) + pie
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((10, 8), titulo, fill="black", font=f_tit)
    for j, c in enumerate(cols):
        x0 = izq + j * (cel + gap)
        for li, linea in enumerate([l for l in c.split("|") if l != ""] if not c.startswith("|error|")
                                   else ["|error|", _corto(nombres[-1])]):
            d.text((x0, cab - 36 + li * 16), linea, fill=(60, 60, 60), font=f_col)

    vmax_e = float(np.abs(entrada).max())
    err = np.abs(salidas[nombres[-1]] - entrada)
    vmax_err = float(err.max())
    for i in range(n):
        y0 = cab + i * (cel + gap)
        d.text((8, y0 + cel // 2 - 7), f"#{i+1}", fill=(120, 120, 120), font=f_lab)
        # La vista cruda se estira a su propio [min, max]: el texto ocupa poco
        # rango de gris y sin estirar la columna sale lavada y no se ve QUE
        # ventana es. Es una decision de PINTADO y no toca lo que ve la red.
        v = vista[i, 0]
        tiles = [_gris(v, float(v.min()), float(v.max())),
                 _diverge(entrada[i, 0], vmax_e)]
        tiles += [_diverge(salidas[nm][i, 0], vmax_e) for nm in nombres]
        tiles += [_calor(err[i, 0], vmax_err)]
        for j, t in enumerate(tiles):
            x0 = izq + j * (cel + gap)
            im.paste(t.resize((cel, cel), Image.NEAREST), (x0, y0))
            d.rectangle([x0, y0, x0 + cel - 1, y0 + cel - 1], outline=(175, 175, 175))
        x = izq + len(tiles) * (cel + gap) + 4
        for li, nm in enumerate(nombres):
            d.text((x, y0 + 8 + li * 17),
                   f"{_corto(nm)}:  R² {filas[i][nm]['r2']:+.3f}",
                   fill=(60, 60, 60), font=f_lab)
    d.text((10, H - 22),
           f"entrada y salidas comparten escala ±{vmax_e:.2f} (azul negativo · rojo positivo) · "
           f"el error va de 0 a {vmax_err:.2f} · la ventana se estira a su propio rango, "
           f"sólo para verla",
           fill=(95, 95, 95), font=f_lab)
    im.save(out_path)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="entrada contra salida de la sonda L1")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--semilla", type=int, default=2026)
    p.add_argument("--modelos", default="k9-K16-l0-s1,k9-K16-lcal-s1")
    p.add_argument("--salida", default=str(EXP / "inspeccion"))
    a = p.parse_args()

    salida = Path(a.salida); salida.mkdir(parents=True, exist_ok=True)
    vista, quien = ventanas_al_azar(a.n, a.semilla)
    entrada = normaliza(vista)

    modelos = {nm: cargar(AQUI / "pesos" / f"{nm}.pt") for nm in a.modelos.split(",")}
    salidas, filas = {}, [dict(q) for q in quien]
    for nm, m in modelos.items():
        with torch.no_grad():
            xh, z = m(entrada)
        salidas[nm] = xh.numpy()
        borde = m.k // 2
        c = slice(borde, -borde)
        for i in range(a.n):
            e, x = xh[i], entrada[i]
            filas[i][nm] = {
                "r2": 1 - float(((e - x) ** 2).mean()) / VAR,
                "r2_interior": 1 - float(((e[..., c, c] - x[..., c, c]) ** 2).mean()) / VAR,
                "activa": float((z[i] > 0).float().mean()),
                "canales_activos": int((z[i].amax((1, 2)) > 0).sum()),
            }

    (salida / "resultados.json").write_text(json.dumps(
        {"n": a.n, "semilla": a.semilla, "split": "validacion",
         "dataset": "dirty1000-80px-16px-r20260827",
         "sigma": SIGMA, "eps": EPS, "var_train": VAR,
         "modelos": list(modelos), "ventanas": filas}, indent=2, ensure_ascii=False))

    png = figura(vista.numpy(), entrada.numpy(), salidas, filas,
                 salida / "entrada-vs-salida.png",
                 f"Sonda L1 — {a.n} ventanas de validación al azar (semilla {a.semilla})")

    anchos = max(len(nm) for nm in modelos)
    print(f"\n{'ventana':>8} " + " ".join(f"{nm:>{anchos+12}}" for nm in modelos))
    for i, f in enumerate(filas):
        print(f"{'#'+str(i+1):>8} " + " ".join(
            f"R2 {f[nm]['r2']:+.3f} act {f[nm]['activa']*100:4.1f}%" for nm in modelos))
    print(f"{'MEDIA':>8} " + " ".join(
        f"R2 {np.mean([f[nm]['r2'] for f in filas]):+.3f} "
        f"act {np.mean([f[nm]['activa'] for f in filas])*100:4.1f}%" for nm in modelos))
    print(f"\nfigura: {png}\nresultados: {salida/'resultados.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
