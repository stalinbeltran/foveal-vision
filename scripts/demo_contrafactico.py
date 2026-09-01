"""Dos formas de hacer que la red ACIERTE en la MISMA ventana, y qué cuesta cada una.

La pregunta que ilustra: si defines "entrada mejorada" como *la entrada más
parecida a la original que hace que la red dé la salida correcta*, ¿qué sale?

Se resuelve el MISMO problema tres veces, cambiando sólo el ESPACIO en que vive
el cambio (delta), y se mide la norma del cambio en cada uno:

  (a) libre        -> delta = cualquier cosa en los 6.400 px del recorte
  (b) suave        -> delta = un campo de 5x5 interpolado (iluminación/fondo)
  (c) semántico    -> delta = alpha * (recorte_con_el_fondo_quitado - recorte)

(a) es la definición literal de "mínimo cambio". (c) es un cambio que un humano
llamaría "mejorar la entrada". Si (a) << (c) en norma, entonces optimizar "el
mínimo cambio que hace acertar a la red" NO produce (c): produce (a).

El delta se optimiza sobre los PÍXELES del recorte (80x80), no sobre la vista
(20x20) que ve la red: en la periferia una celda de la vista es la media de 256
px reales, así que un mínimo en el espacio de la vista no corresponde a ninguna
imagen. El muestreo foveado con pool_mode='avg' es un operador LINEAL separable
(fv.fovea._axis_edges + add.reduceat), así que la vista es P @ recorte @ P.T y
el gradiente hasta el píxel es exacto. El test de costura lo comprueba contra
la ruta de numpy que usan el dataloader y la inferencia.

Uso:
    python scripts/demo_contrafactico.py --ckpt <best.pt> --dataset <nombre> \
        --out data/demo-contrafactico [--n 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.fovea import (_axis_edges, build_view, dims_of, edge_features,  # noqa: E402
                      input_stack, n_edge_features, n_input_channels, pad_sides)
from fv.inference.checkpoint import load_model  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# --------------------------------------------------------------------------
# el muestreo foveado como matriz (exacto: avg-pool separable de bins desiguales)
# --------------------------------------------------------------------------
def pooling_matrix(dims) -> torch.Tensor:
    e = _axis_edges(dims)
    P = torch.zeros(dims.N, dims.original_size, dtype=torch.float64)
    for k in range(dims.N):
        P[k, e[k]:e[k + 1]] = 1.0 / float(e[k + 1] - e[k])
    return P


def view_of(crop01: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return P @ crop01 @ P.T


def tabla_geometria(dims) -> None:
    """Cuantos px REALES promedia cada celda de la vista. Decide en que espacio
    tiene sentido medir "el minimo cambio": en el centro vista<->pixel es 1:1,
    asi que un cambio ahi es una edicion de la imagen; en la periferia una celda
    es la media de varios px, y ahi puedes mover esos px conservando la media
    sin que la red vea NADA. Un minimo en el espacio de la vista no corresponde
    a ninguna imagen -- por eso el delta se optimiza sobre el recorte."""
    import collections
    w = np.diff(_axis_edges(dims))
    A = np.outer(w, w)
    print(f"geometria: fovea_px={dims.fovea_px} border_px={dims.border_px} "
          f"border_reduce={dims.border_reduce} -> N={dims.N}, recorte "
          f"{dims.original_size}x{dims.original_size}")
    for px, n in sorted(collections.Counter(A.ravel().tolist()).items()):
        print(f"  {n:4d} celdas promedian {px:4d} px reales -> {n * px:6d} px")
    centro = int((A == 1).sum())
    print(f"  total: {A.size} celdas / {int(A.sum())} px")
    print(f"  centro 1:1 = {centro} celdas ({100 * centro / A.size:.0f} % de las entradas)")
    print(f"  la periferia ve {int(A.sum()) - centro} px "
          f"({100 * (A.sum() - centro) / A.sum():.0f} % del recorte) "
          f"con el {100 * (A > 1).sum() / A.size:.0f} % de las entradas")


def net_input(crop01, P, mask_channel):
    v = view_of(crop01, P)
    if n_input_channels(mask_channel) == 1:
        return v[None, None]
    # ventana interior: coverage = 1 en todas las celdas -> el canal de relleno es 0
    return torch.stack([v, torch.zeros_like(v)])[None]


# --------------------------------------------------------------------------
# criterio de "acierta": las 4 decisiones de `exists`. Es la decisión binaria
# que luego mueve o no mueve un párrafo en la reconstrucción.
# --------------------------------------------------------------------------
def aciertos(logits, y):
    pred = (torch.sigmoid(logits[0, :, 0]) >= 0.5).float()
    return (pred == y[:, 0]).cpu().numpy()


def margen(logits, y):
    """>0 en cada esquina que está del lado correcto del umbral."""
    return (2.0 * y[:, 0] - 1.0) * logits[0, :, 0]


def resolver(crop0, y, model, P, mask_channel, edge, param, kappa=0.5,
             pasos=400, cs=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)):
    """min ||delta||^2 + c * sum(relu(kappa - margen)) sobre el espacio de `param`.

    param: 'libre' | 'suave' | (no se usa para el semántico, que es 1 escalar).
    Devuelve el delta de menor norma que consigue las 4 esquinas correctas.
    """
    mejor = None
    for c in cs:
        if param == "libre":
            z = torch.zeros_like(crop0, requires_grad=True)
            expand = lambda z: z
        else:
            z = torch.zeros(1, 1, 5, 5, dtype=crop0.dtype, requires_grad=True)
            expand = lambda z: TF.interpolate(z, size=crop0.shape, mode="bilinear",
                                              align_corners=False)[0, 0]
        opt = torch.optim.Adam([z], lr=0.02)
        for _ in range(pasos):
            opt.zero_grad()
            d = expand(z)
            x = (crop0 + d).clamp(0.0, 1.0)
            d_real = x - crop0                      # el delta que de verdad se aplica
            out = model(net_input(x, P, mask_channel).float(), edge)
            perdida = (d_real ** 2).sum() + c * torch.relu(kappa - margen(out, y)).sum()
            perdida.backward()
            opt.step()
        with torch.no_grad():
            x = (crop0 + expand(z)).clamp(0.0, 1.0)
            out = model(net_input(x, P, mask_channel).float(), edge)
            if aciertos(out, y).all():
                d = (x - crop0)
                n = float((d ** 2).sum().sqrt())
                if mejor is None or n < mejor[0]:
                    mejor = (n, x.detach().clone(), c)
    return mejor


def limpiar_fondo(crop0: torch.Tensor, k: int = 9) -> torch.Tensor:
    """Quita el fondo: dilatación en gris (max local) = estimación del fondo,
    y se resta. Los trazos finos y oscuros (la tinta, y también las rayas o la
    rejilla del fondo si son finas) sobreviven; el fondo se va a blanco."""
    bg = TF.max_pool2d(crop0[None, None].float(), kernel_size=k, stride=1,
                       padding=k // 2)[0, 0].to(crop0.dtype)
    return (crop0 - bg + 1.0).clamp(0.0, 1.0)


def resolver_semantico(crop0, y, model, P, mask_channel, edge, pasos=201):
    """Un solo grado de libertad: cuánto se aplica la limpieza de fondo."""
    limpio = limpiar_fondo(crop0)
    for i in range(pasos):
        a = i / (pasos - 1)
        x = ((1 - a) * crop0 + a * limpio).clamp(0.0, 1.0)
        with torch.no_grad():
            out = model(net_input(x, P, mask_channel).float(), edge)
        if aciertos(out, y).all():
            return float(((x - crop0) ** 2).sum().sqrt()), x, a
    x = limpio
    return float(((x - crop0) ** 2).sum().sqrt()), x, None


# --------------------------------------------------------------------------
# pintar
# --------------------------------------------------------------------------
def gris(a: torch.Tensor, escala: int) -> Image.Image:
    arr = (a.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr, "L").convert("RGB")
    return im.resize((im.width * escala, im.height * escala), Image.NEAREST)


def panel(crop0, columnas, P, out_path, titulo, pie):
    """columnas: lista de (nombre, subtitulo, x | None)."""
    n = crop0.shape[0]
    S = 320 // n                                   # 32 -> x10
    W = n * S
    f_tit = ImageFont.truetype(FONT, 19)
    f_col = ImageFont.truetype(FONT, 16)
    f_pie = ImageFont.truetype(FONT, 14)
    izq, gap, cab, sub, pie_h = 210, 22, 40, 46, 26 + 19 * len(pie)
    filas = ["la imagen\n(recorte 32x32)", "el cambio (delta)\namplificado",
             "lo que ve la red\n(vista 20x20)"]
    ancho = izq + len(columnas) * (W + gap) + 90
    alto = cab + sub + 3 * (W + 30) + pie_h
    im = Image.new("RGB", (ancho, alto), "white")
    d = ImageDraw.Draw(im)
    d.text((12, 10), titulo, fill="black", font=f_tit)
    for j, (nombre, subtitulo, x) in enumerate(columnas):
        x0 = izq + j * (W + gap)
        d.text((x0, cab), nombre, fill="black", font=f_col)
        d.text((x0, cab + 22), subtitulo, fill=(95, 95, 95), font=f_pie)
        delta = (x - crop0) if x is not None else None
        for i in range(3):
            y0 = cab + sub + i * (W + 30)
            if i == 0:
                sub_im = gris(x if x is not None else crop0, S)
            elif i == 1:
                if delta is None or float(delta.abs().max()) < 1e-9:
                    sub_im = Image.new("RGB", (W, W), (128, 128, 128))
                    d.text((x0 + 8, y0 + W // 2), "sin cambio", fill=(60, 60, 60), font=f_pie)
                else:
                    k = 0.45 / float(delta.abs().max())
                    sub_im = gris(0.5 + delta * k, S)
                    d.text((x0, y0 + W + 4),
                           f"x{k:.0f} · max|d| = {float(delta.abs().max()) * 255:.1f}/255",
                           fill=(95, 95, 95), font=f_pie)
            else:
                sub_im = gris(view_of(x if x is not None else crop0, P), W // 20)
            im.paste(sub_im, (x0, y0))
            d.rectangle([x0, y0, x0 + W - 1, y0 + W - 1], outline=(150, 150, 150))
    for i, t in enumerate(filas):
        for li, linea in enumerate(t.split("\n")):
            d.text((12, cab + sub + i * (W + 30) + W // 2 - 14 + li * 18),
                   linea, fill="black", font=f_pie)
    for i, linea in enumerate(pie):
        d.text((12, alto - pie_h + 10 + i * 19), linea, fill=(30, 30, 30), font=f_pie)
    im.save(out_path)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", help="ruta al windows.npz")
    ap.add_argument("--out", default="data/demo-contrafactico")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--max-scan", type=int, default=6000)
    ap.add_argument("--geometria", action="store_true",
                    help="solo la tabla de cuantos px reales promedia cada celda "
                         "de la vista, y sale")
    a = ap.parse_args()

    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    model = load_model(Path(a.ckpt))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"]["model"]
    dims = dims_of(cfg)
    mask_channel = cfg.get("mask_channel", "off")
    edge_mode = cfg.get("edge_inputs", "off")
    print(f"red: {Path(a.ckpt).parent.name} · N={dims.N} · recorte={dims.original_size} "
          f"· mask_channel={mask_channel} · edge_inputs={edge_mode}")

    if a.geometria:
        tabla_geometria(dims)
        return 0

    z = np.load(a.dataset)
    sel = z["split"] == a.split
    y_all, wxy, sidx = z["y"][sel], z["window_xy"][sel], z["sample_idx"][sel]
    imgs, isidx = z["images"], z["images_sample_idx"]
    fila = {int(s): i for i, s in enumerate(isidx)}
    print(f"dataset: {Path(a.dataset).parent.name} · {sel.sum()} ventanas en split {a.split}")

    P = pooling_matrix(dims)

    # --- costura: la vista de torch tiene que ser la de numpy, bit a bit
    k0 = next(k for k in range(len(y_all))
              if pad_sides(imgs.shape[1:], int(wxy[k, 0]), int(wxy[k, 1]), dims)
              == (0, 0, 0, 0))
    im0 = imgs[fila[int(sidx[k0])]]
    wx, wy = int(wxy[k0, 0]), int(wxy[k0, 1])
    v_np, _ = build_view(im0, wx, wy, dims)
    crop_np = im0[wy - dims.border_px:wy - dims.border_px + dims.original_size,
                  wx - dims.border_px:wx - dims.border_px + dims.original_size]
    v_t = view_of(torch.from_numpy(crop_np.astype(np.float64) / 255.0), P)
    print(f"costura vista torch vs numpy: max|dif| = "
          f"{np.abs(v_t.numpy() - v_np).max():.2e}")

    # --- buscar ventanas interiores donde la red FALLA alguna esquina.
    # Por LOTES: una a una son 28.000 forwards secuenciales (~2 min); asi son
    # segundos, y lo que se busca no cambia.
    idx_int = [i for i in range(len(y_all))
               if pad_sides(imgs.shape[1:], int(wxy[i, 0]), int(wxy[i, 1]), dims)
               == (0, 0, 0, 0)][:a.max_scan]
    fallos = []
    B = 512
    for lo in range(0, len(idx_int), B):
        trozo = idx_int[lo:lo + B]
        crops = torch.stack([torch.from_numpy(
            imgs[fila[int(sidx[i])]]
            [int(wxy[i, 1]) - dims.border_px:int(wxy[i, 1]) - dims.border_px + dims.original_size,
             int(wxy[i, 0]) - dims.border_px:int(wxy[i, 0]) - dims.border_px + dims.original_size]
            .astype(np.float64) / 255.0) for i in trozo])
        vs = torch.einsum("ij,bjk,lk->bil", P, crops, P)
        xin = (vs[:, None] if n_input_channels(mask_channel) == 1
               else torch.stack([vs, torch.zeros_like(vs)], dim=1))
        ee = torch.zeros(len(trozo), n_edge_features(edge_mode))
        with torch.no_grad():
            out = model(xin.float(), ee)
        yt = torch.from_numpy(y_all[trozo].astype(np.float64))
        pred = (torch.sigmoid(out[:, :, 0]) >= 0.5).double()
        ok = pred == yt[:, :, 0]
        marg = ((2.0 * yt[:, :, 0] - 1.0) * out[:, :, 0]).min(dim=1).values
        for k, i in enumerate(trozo):
            if not bool(ok[k].all()):
                # el margen mas NEGATIVO primero: fallos con conviccion, no del
                # filo. Ordenar por |margen| elegia las ventanas que cualquier
                # empujon da la vuelta, y ahi las tres rutas cuestan lo mismo.
                fallos.append((float(marg[k]), i, crops[k].clone(), yt[k].clone(),
                               ee[:1].clone(), int((~ok[k]).sum())))
    fallos.sort(key=lambda t: t[0])
    print(f"ventanas interiores miradas: {len(idx_int)} · con alguna esquina mal: "
          f"{len(fallos)} · margen mas negativo: {fallos[0][0]:.2f}")

    resumen = []
    for rank, (_, i, crop, yt, e, n_mal) in enumerate(fallos[:a.n]):
        base = float((crop ** 2).sum().sqrt())
        libre = resolver(crop, yt, model, P, mask_channel, e, "libre")
        suave = resolver(crop, yt, model, P, mask_channel, e, "suave")
        n_sem, x_sem, alpha = resolver_semantico(crop, yt, model, P, mask_channel, e)
        if libre is None:
            print(f"[{i}] el ataque libre no convergió; se salta"); continue
        n_lib, x_lib, _ = libre
        cols = [("(0) original", f"la red FALLA {n_mal} de 4 esquinas", None),
                ("(a) el MÍNIMO cambio, libre",
                 f"||d|| = {n_lib:.3f} = {100 * n_lib / base:.2f} % de ||x||  ->  acierta 4/4",
                 x_lib),
                ("(c) cambio SEMÁNTICO: quitar el fondo",
                 (f"||d|| = {n_sem:.3f} = {100 * n_sem / base:.2f} % de ||x||  ->  acierta 4/4"
                  if alpha is not None else
                  f"aplicado al 100 % ->  NO arregla la red (||d|| = {n_sem:.3f})"), x_sem)]
        d_lib = x_lib - crop
        px = int((d_lib.abs() > 0.5 / 255).sum())
        pie = [
            "El MISMO problema resuelto tres veces. Lo unico que cambia es EN QUE ESPACIO puede vivir el cambio.",
            f"(a) libre (1024 grados de libertad): ||d|| = {n_lib:.3f} · max|d| = "
            f"{float(d_lib.abs().max()) * 255:.1f}/255 · {px} de {dims.original_size ** 2} px "
            f"tocados ({100 * px / dims.original_size ** 2:.0f} %)",
            f"(b) suave, campo 5x5 (25 grados de libertad): "
            + (f"||d|| = {suave[0]:.3f} = {suave[0] / n_lib:.1f}x el libre" if suave
               else "no converge: no hay cambio suave que arregle esta ventana"),
            (f"(c) semantico, quitar el fondo (1 grado de libertad): ||d|| = {n_sem:.3f} = "
             f"{n_sem / n_lib:.1f}x el libre, con alpha = {alpha:.2f}" if alpha is not None else
             "(c) semantico, quitar el fondo (1 grado de libertad): aplicado AL 100 % la red "
             f"SIGUE FALLANDO. ||d|| = {n_sem:.3f}, y no es una solucion: no hay ratio que comparar"),
            "Cuantos menos grados de libertad, MAS caro sale el cambio -- o directamente no hay "
            "solucion. El minimo sin restricciones es siempre el menos semantico.",
            "AVISO: (a) y (b) son COTAS SUPERIORES (Adam sobre ||d||^2 + c*hinge, rejilla de c). "
            "El minimo real es menor o igual, nunca mayor.",
        ]
        p = outdir / f"contrafactico-{rank}-ventana{i}.png"
        panel(crop, cols, P, p, f"ventana {i} del split val · red {Path(a.ckpt).parent.name}", pie)
        print(f"  -> {p}")
        resumen.append(dict(ventana=i, n_mal=n_mal, libre=n_lib, suave=suave[0] if suave else None,
                            semantico=n_sem, alpha=alpha, px=px, base=base))
    print()
    print(f"{'ventana':>8} {'||d|| libre':>12} {'||d|| suave':>12} {'||d|| semant':>13} "
          f"{'px tocados':>11} {'razon sem/libre':>16}")
    for r in resumen:
        s = f"{r['suave']:.3f}" if r["suave"] else "no conv."
        print(f"{r['ventana']:>8} {r['libre']:>12.3f} {s:>12} {r['semantico']:>13.3f} "
              f"{r['px']:>11} {r['semantico'] / r['libre']:>15.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
