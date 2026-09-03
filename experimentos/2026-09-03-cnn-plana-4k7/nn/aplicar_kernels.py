#!/usr/bin/env python3
"""La evaluacion de ESTE experimento: aplicar los 4 kernels a 10 entradas fijas.

    python nn/aplicar_kernels.py --stop 00-sin-entrenar          # red sin entrenar
    python nn/aplicar_kernels.py --stop 01-2min --run plana-4k7-s1

⚠ NO se evalua la salida tipica de la red (las 12 cifras de las esquinas). El
encargo lo dice explicitamente: lo que se mira es la ENTRADA pasada por los
kernels, o sea `N_KERNELS` imagenes por cada imagen de entrada.

DOS DECISIONES, y las dos tienen motivo
---------------------------------------
1. **El set de visualizacion se congela una vez** en
   `evaluacion/set-visualizacion.json` y se reusa en TODOS los stops. Elegirlo de
   nuevo en cada stop haria que dos stops no fueran comparables, que es lo unico
   que esta figura existe para permitir.

2. **Los mapas se guardan SIN activar (con signo).** Con `n_layers: 1` esa capa
   es la ultima, y `_branch_forward` no pone ReLU en la ultima a proposito --
   "the last map stays pre-activation, introspection reads it signed"
   (builder.py:197). Aplicar los kernels ES la convolucion; poner un ReLU aqui
   tiraria la mitad de lo que hay que ver.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))

from fv import settings                                          # noqa: E402
from fv.fovea import build_view, dims_of, edge_features, input_stack  # noqa: E402
from fv.models.builder import build_model, full_config           # noqa: E402
from fv.training.registry import RunStore                        # noqa: E402

DATASET = "dirty1000-80px-16px-r20260827"
RED = REPO / "configs" / "networks" / "plana-20-4k7.yaml"
SEMILLA_RECETA = 1          # `seed` de configs/recipes/plan40.yaml
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _net() -> dict:
    return full_config(yaml.safe_load(RED.read_text()))


def set_visualizacion(n: int, semilla: int, split: int = 1) -> list[dict]:
    """Los n indices, elegidos UNA vez y guardados. Si ya existen, se releen."""
    fich = EXP / "evaluacion" / "set-visualizacion.json"
    if fich.exists():
        d = json.loads(fich.read_text())
        print(f"[set] releo el set congelado: {len(d['ventanas'])} ventanas "
              f"(semilla {d['semilla']})")
        return d["ventanas"]
    z = np.load(settings.window_datasets_root() / DATASET / "windows.npz")
    sel = np.flatnonzero(z["split"] == split)
    idx = np.random.default_rng(semilla).choice(sel, size=n, replace=False)
    lookup = {int(a): i for i, a in enumerate(z["images_sample_idx"])}
    vent = [{"indice": int(i), "fila_imagen": lookup[int(z["sample_idx"][i])],
             "ventana_xy": [int(z["window_xy"][i, 0]), int(z["window_xy"][i, 1])]}
            for i in idx]
    fich.parent.mkdir(parents=True, exist_ok=True)
    fich.write_text(json.dumps(
        {"dataset": DATASET, "split": "validacion", "semilla": semilla,
         "n": n, "ventanas": vent,
         "por_que": "congelado: dos stops solo son comparables sobre las MISMAS "
                    "entradas"}, indent=2, ensure_ascii=False))
    print(f"[set] congelado en {fich}")
    return vent


def entradas(vent: list[dict]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """(B, C, N, N) tal como la ve la red, los 4 escalares de borde, y la vista."""
    net = _net()
    dims = dims_of(net)
    z = np.load(settings.window_datasets_root() / DATASET / "windows.npz")
    xs, es, vistas = [], [], []
    for v in vent:
        img = z["images"][v["fila_imagen"]]
        wx, wy = v["ventana_xy"]
        vista, cob = build_view(img, wx, wy, dims,
                               pool_mode=net["pool_mode"], pad_mode=net["pad_mode"])
        xs.append(input_stack(vista, cob, net["mask_channel"]))
        es.append(edge_features(img.shape, wx, wy, dims, net["edge_inputs"]))
        vistas.append(vista)
    return (torch.from_numpy(np.stack(xs)).float(),
            torch.from_numpy(np.stack(es)).float(),
            np.stack(vistas))


def modelo(run: str | None):
    """El modelo del run, o la red SIN ENTRENAR con la semilla de la receta.

    La init se reproduce exacta: `torch.manual_seed(seed)` y `build_model` --
    comprobado el 2026-09-03 que construir los datasets en medio no consume el
    RNG global, asi que este `stop-00` es LA MISMA red que luego entrena.
    """
    net = _net()
    if run is None:
        torch.manual_seed(SEMILLA_RECETA)
        np.random.seed(SEMILLA_RECETA % 2 ** 32)
        m = build_model(net)
        etiqueta = "sin entrenar (semilla 1)"
    else:
        # `RunStore.path` resuelve las TRES ubicaciones posibles (plana, archivo
        # fechado, legado). Derivar la ruta a mano aqui es el fallo que este
        # repo ya tiene anotado: el run de hoy cayo en
        # `2026/09-septiembre/runs/`, no en `runs/`.
        ck = RunStore().path(run) / "last.pt"
        if not ck.exists():
            raise SystemExit(f"no esta {ck}")
        e = torch.load(ck, map_location="cpu", weights_only=False)
        m = build_model(net)
        m.load_state_dict(e["model"] if "model" in e else e["state_dict"])
        ep = e.get("epoch", "?")
        etiqueta = f"{run} · epoca {ep}"
    m.eval()
    return m, etiqueta


# ------------------------------------------------------------------ pintar
def _font(s):
    try:
        return ImageFont.truetype(FONT, s)
    except OSError:
        return ImageFont.load_default()


def _gris(a):
    t = np.clip((a - a.min()) / max(a.max() - a.min(), 1e-9), 0, 1)
    v = (t * 255).astype(np.uint8)
    return Image.fromarray(np.stack([v, v, v], -1), "RGB")


def _diverge(a, vmax):
    t = np.clip(a / max(vmax, 1e-9), -1, 1)
    pos, neg = np.clip(t, 0, 1), np.clip(-t, 0, 1)
    rgb = np.stack([255 - neg * 200, 255 - np.maximum(pos, neg) * 200, 255 - pos * 200], -1)
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")


def montaje(vistas, mapas, out, titulo, cel=92, gap=8, vmax=None, pie_extra=""):
    """Una fila por entrada: la ventana y sus N_KERNELS mapas."""
    n, K = mapas.shape[0], mapas.shape[1]
    # Escala COMUN a todos los mapas del stop: por-kernel escondería que un
    # kernel responde diez veces mas fuerte que otro, que es parte de lo que se
    # quiere ver.
    if vmax is None:
        vmax = float(np.abs(mapas).max())
    f_tit, f_col, f_lab = _font(19), _font(13), _font(12)
    cols = ["la ventana"] + [f"kernel {j}" for j in range(K)]
    izq, cab, pie = 40, 74, 28
    sonda = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    W = max(izq + len(cols) * (cel + gap) + 20,
            int(sonda.textlength(titulo, font=f_tit)) + 20)
    H = cab + n * (cel + gap) + pie
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((10, 8), titulo, fill="black", font=f_tit)
    for j, c in enumerate(cols):
        d.text((izq + j * (cel + gap), cab - 22), c, fill=(60, 60, 60), font=f_col)
    for i in range(n):
        y0 = cab + i * (cel + gap)
        d.text((6, y0 + cel // 2 - 7), f"#{i+1}", fill=(120, 120, 120), font=f_lab)
        tiles = [_gris(vistas[i])] + [_diverge(mapas[i, j], vmax) for j in range(K)]
        for j, t in enumerate(tiles):
            x0 = izq + j * (cel + gap)
            im.paste(t.resize((cel, cel), Image.NEAREST), (x0, y0))
            d.rectangle([x0, y0, x0 + cel - 1, y0 + cel - 1], outline=(175, 175, 175))
    d.text((10, H - 20),
           f"los {K*n} mapas comparten escala ±{vmax:.3f} (azul negativo · blanco cero · "
           f"rojo positivo) · sin activar, con signo · la ventana se estira para verla"
           + pie_extra,
           fill=(95, 95, 95), font=f_lab)
    im.save(out)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="aplica los kernels al set de visualizacion")
    p.add_argument("--stop", required=True, help="etiqueta, p.ej. 00-sin-entrenar")
    p.add_argument("--run", default=None, help="sin esto: la red SIN entrenar")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--semilla", type=int, default=2026)
    a = p.parse_args()

    vent = set_visualizacion(a.n, a.semilla)
    x, e, vistas = entradas(vent)
    m, etiqueta = modelo(a.run)
    conv = m.center_convs[0]
    with torch.no_grad():
        mapas = conv(x)                       # (B, K, N, N) SIN activar
        # ¿De QUE canal viene la respuesta? La entrada tiene dos (vista y
        # RELLENO), y un kernel que solo lea el relleno dibuja el marco de la
        # ventana y nada mas -- que se parece mucho a "ha aprendido algo" en la
        # figura. Se separa pasando cada canal con el otro a cero, sin sesgo:
        # asi la suma de las dos partes es exactamente el mapa menos el sesgo.
        porcanal = []
        for c in range(x.shape[1]):
            xc = torch.zeros_like(x); xc[:, c] = x[:, c]
            porcanal.append(torch.nn.functional.conv2d(
                xc, conv.weight, None, conv.stride, conv.padding))
    mapas = mapas.numpy()
    aporte = [np.abs(t.numpy()).mean(axis=(0, 2, 3)) for t in porcanal]   # [K] por canal

    dest = EXP / "evaluacion" / f"stop-{a.stop}"
    dest.mkdir(parents=True, exist_ok=True)
    vmax = float(np.abs(mapas).max())
    for i in range(mapas.shape[0]):
        for j in range(mapas.shape[1]):
            _diverge(mapas[i, j], vmax).resize((160, 160), Image.NEAREST).save(
                dest / f"entrada{i+1:02d}-kernel{j}.png")
    np.save(dest / "mapas.npy", mapas.astype(np.float32))
    W = m.center_convs[0].weight.detach()
    (dest / "resumen.json").write_text(json.dumps({
        "stop": a.stop, "modelo": etiqueta, "run": a.run,
        "n_kernels": int(mapas.shape[1]), "n_entradas": int(mapas.shape[0]),
        "vmax_comun": vmax,
        "kernels": {"forma": list(W.shape),
                    "norma_l2": [round(float(v), 5) for v in W.flatten(1).norm(dim=1)]},
        "respuesta_por_kernel": {
            f"kernel{j}": {
                "abs_media": round(float(np.abs(mapas[:, j]).mean()), 5),
                "abs_max": round(float(np.abs(mapas[:, j]).max()), 5),
                # cuanta de la respuesta viene de la VISTA y cuanta del RELLENO
                "aporte_vista": round(float(aporte[0][j]), 5),
                "aporte_relleno": round(float(aporte[1][j]), 5),
                "frac_vista": round(float(aporte[0][j] / max(aporte[0][j] + aporte[1][j], 1e-9)), 4),
            } for j in range(mapas.shape[1])},
    }, indent=2, ensure_ascii=False))
    png = montaje(vistas, mapas, dest / "montaje.png",
                  f"CNN plana 4×7×7 — los 4 kernels sobre el set de visualización  ·  {etiqueta}")
    # Y la MISMA cosa quitando a cada mapa su nivel (mediana). Hace falta porque
    # esta MEDIDO que el nivel constante es ~8x la estructura del texto (0,300
    # contra 0,037 en la epoca 3): con la escala comun, el rizo que dice si el
    # kernel ha aprendido algo es invisible. El nivel sale de la media del kernel
    # multiplicada por un papel casi uniforme, o sea que NO es informacion del
    # texto.
    # ⚠ Es una vista de PINTADO. Los 40 PNG y `mapas.npy` son la salida cruda.
    nivel = np.median(mapas, axis=(2, 3), keepdims=True)
    sinniv = mapas - nivel
    # ⚠ Y la escala sale del INTERIOR, no del mapa entero: el anillo de k//2 px
    # llega a valores ~5x los del interior (padding de ceros contra un papel de
    # valor ~1), asi que con la escala global el rizo del texto sigue invisible.
    # Se recorta el borde para elegir la escala, no para pintarlo.
    b = m.center_convs[0].kernel_size[0] // 2
    c = slice(b, -b)
    vmax_int = float(np.quantile(np.abs(sinniv[..., c, c]), 0.99))
    montaje(vistas, sinniv, dest / "montaje-sin-nivel.png",
            f"Lo mismo SIN el nivel de cada mapa (mediana restada)  ·  {etiqueta}",
            vmax=vmax_int,
            pie_extra=" · escala del p99 del INTERIOR: el anillo de "
                      f"{b} px queda saturado a proposito")

    print(f"\n[{a.stop}] {etiqueta}")
    print(f"  norma L2 de cada kernel: "
          + "  ".join(f"k{j} {float(v):.3f}" for j, v in enumerate(W.flatten(1).norm(dim=1))))
    print(f"  |respuesta| media:       "
          + "  ".join(f"k{j} {np.abs(mapas[:,j]).mean():.4f}" for j in range(mapas.shape[1])))
    print(f"  de la VISTA (resto: relleno): "
          + "  ".join(f"k{j} {aporte[0][j]/max(aporte[0][j]+aporte[1][j],1e-9)*100:4.1f}%"
                      for j in range(mapas.shape[1])))
    print(f"  {mapas.shape[0]*mapas.shape[1]} PNG + montaje.png + mapas.npy en {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
