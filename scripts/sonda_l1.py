#!/usr/bin/env python3
r"""Sonda L1: ¿pueden los kernels de la primera capa aprender filtros genéricos?

Qué es esto
-----------
Un autoencoder convolucional de UNA capa por lado sobre la MISMA vista 20x20 que
ve la red de producción. El modelo *son* los kernels: no hay nada detrás que
pueda arreglar un código malo, así que la presión cae entera sobre L1 -- que es
justo lo que no ocurre en `fov16-optimo-mask`, donde detrás hay una cabeza de
153.660 parámetros.

    .venv/bin/python scripts/sonda_l1.py --cronometrar        # UNA combinación
    .venv/bin/python scripts/sonda_l1.py --rejilla            # las 36
    .venv/bin/python scripts/sonda_l1.py --semillas 1,2,3 --solo k9-K32-l0.1

Es un experimento APARTE: no toca `src/fv/models/builder.py`, ni los configs de
`configs/networks/`, ni nada que cambie el significado de un checkpoint en disco.
Lo único que importa de `fv` es `fv.fovea.build_view` -- la MISMA función que usa
el dataloader y la inferencia (contrato (5)): si la vista se construyera aquí
otra vez, la sonda mediría un dato que la red no ve.

LO QUE HAY QUE RESPETAR SI SE TOCA ESTO
---------------------------------------
1. **El decodificador es lineal, de una capa y SIN sesgo, y sus átomos se
   renormalizan a L2=1 después de CADA paso.** Sin la renormalización el modelo
   tiene una salida degenerada gratis: multiplicar el codificador por 0,01 y el
   decodificador por 100 da la misma reconstrucción con la penalización cien
   veces menor, o sea que aprendería a hacer `z` PEQUEÑO en vez de DISPERSO.
   El codificador queda libre a propósito: es el que tiene que moverse.

2. **`var(x)` es una constante FIJA del train, calculada una vez** (se imprime y
   se guarda en el resumen), no la varianza del lote. Si cambiara por lote, `λ`
   significaría algo distinto en cada paso y el barrido en `λ` dejaría de ser
   comparable -- que es lo único que ese denominador existe para garantizar.

3. **Las vistas se PRECALCULAN una vez, no se construyen por ítem.**
   `build_view` cuesta 163 us/ventana *(medido 2026-09-02 en este droplet, 2
   vCPU)*; a 84.000 ventanas x 30 épocas x 36 runs serían ~4 h de puro Python
   armando la misma vista una y otra vez. Precalculadas: 14 s y 134 MB.

4. **La fracción de activación es DIAGNÓSTICO, no pérdida.** Dice si `λ` está en
   rango (5-15 % es el objetivo); no se optimiza contra ella.

5. **El enriquecimiento se lee contra SU PROPIO nulo, nunca en crudo.** La
   fracción de energía en el subespacio clásico 6D tiene nulo `6/k²`, o sea
   0,667 en 3x3 pero 0,074 en 9x9: comparar las fracciones crudas entre `k`
   distintos es comparar tres escalas. Lo que se compara es `energia_6d /
   (6/k²)`, que vale 1 cuando el kernel es indistinguible de uno aleatorio.

LO QUE ESTA SONDA NO PUEDE DECIR
--------------------------------
Que un kernel de 9x9 entrenado aquí sea genérico NO dice que el de 3x3 de
producción pudiera serlo: son dos preguntas (¿cabe la estructura? y ¿hay
presión?) y esta sonda mueve las dos a la vez. Por eso `--con-k3` existe y por
eso está encendido por defecto: `k=3` cuesta 9/164 del total (~5 %) y es el
único brazo que ancla contra el 0,688 ya medido en `fov16-mask-p20`.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch                                    # noqa: E402
import torch.nn as nn                           # noqa: E402
import torch.nn.functional as F                 # noqa: E402

from fv import settings                         # noqa: E402
from fv.fovea import build_view, dims_of        # noqa: E402
from fv.models.builder import NETWORK_DEFAULTS  # noqa: E402

DATASET = "dirty1000-80px-16px-r20260827"
SPLIT_TRAIN, SPLIT_VAL = 0, 1

# La rejilla del encargo. `k=3` NO estaba y se añade como ANCLA (ver arriba): sin
# él la sonda no se puede comparar con el 0,688 que la motiva.
KS_ENCARGO = [5, 7, 9]
KS_ANCLA = [3]
CANALES = [8, 16, 32]
LAMBDAS = [0.0, 0.03, 0.1, 0.3]


# ---------------------------------------------------------------- 1. los datos

def _gauss1d(sigma: float, radio: int) -> torch.Tensor:
    x = torch.arange(-radio, radio + 1, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _borrosa(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Gaussiana separable con relleno REPLICADO, igual que `pad_mode: edge`.

    Con relleno de ceros la media local del borde se hundiría y la sonda vería
    un anillo de contraste que la imagen no tiene.
    """
    r = (g.numel() - 1) // 2
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, g.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, g.view(1, 1, -1, 1))


def normaliza_contraste(x: torch.Tensor, sigma: float, eps: float) -> torch.Tensor:
    """(x - media local) / (sd local + eps), con sigma en píxeles de la vista.

    OBLIGATORIO, y no es cosmética: sin esto la componente de mayor varianza es
    el nivel medio de intensidad, y la pérdida gasta ahí sus primeros grados de
    libertad. Es literalmente lo que produjo el par duplicado k5/k7 de
    `fov16-mask-p20`, los dos DC negativo puro.
    """
    g = _gauss1d(sigma, max(1, int(round(3 * sigma))))
    mu = _borrosa(x, g)
    var = _borrosa((x - mu) ** 2, g).clamp_min(0.0)
    return (x - mu) / (var.sqrt() + eps)


def _vistas(arrays: dict, split: int, dims) -> np.ndarray:
    sel = arrays["split"] == split
    wxy = arrays["window_xy"][sel]
    sidx = arrays["sample_idx"][sel]
    imgs = arrays["images"]
    lookup = {int(a): i for i, a in enumerate(arrays["images_sample_idx"])}
    fila = np.asarray([lookup[int(s)] for s in sidx], dtype=np.int32)
    out = np.empty((len(fila), dims.N, dims.N), dtype=np.float32)
    for i in range(len(fila)):
        v, _ = build_view(imgs[fila[i]], int(wxy[i, 0]), int(wxy[i, 1]), dims,
                          pool_mode="avg", pad_mode="edge")
        out[i] = v
    return out


def preparar(dataset: str, sigma: float, eps: float | None, cache: Path,
             limite: int | None, verbose: bool = True) -> dict:
    """Vistas 20x20 de train y val, ya normalizadas. Se calculan UNA vez."""
    dims = dims_of(dict(NETWORK_DEFAULTS))
    fich = cache / f"{dataset}-s{sigma}-lim{limite or 0}.npz"
    if fich.exists():
        z = np.load(fich)
        d = {k: z[k] for k in z.files}
        d["eps"] = float(d["eps"]); d["var"] = float(d["var"])
        if verbose:
            print(f"[datos] caché {fich.name}: train {d['train'].shape} val {d['val'].shape}")
        return d

    npz = settings.window_datasets_root() / dataset / "windows.npz"
    if not npz.exists():
        raise SystemExit(f"no está {npz} -- ¿está clonado foveal-vision-data?")
    z = np.load(npz)
    arrays = {k: z[k] for k in z.files}
    t = time.time()
    tr = torch.from_numpy(_vistas(arrays, SPLIT_TRAIN, dims))[:, None]
    va = torch.from_numpy(_vistas(arrays, SPLIT_VAL, dims))[:, None]
    if verbose:
        print(f"[datos] vistas construidas en {time.time()-t:.1f} s: "
              f"train {tuple(tr.shape)} val {tuple(va.shape)}")

    # eps se MIDE del train y se guarda: es la mediana de la sd local, o sea el
    # contraste típico. Un eps tecleado a ojo decide en silencio cuánto ruido se
    # amplifica en las zonas en blanco, que son la mayoría de una vista de texto.
    if eps is None:
        g = _gauss1d(sigma, max(1, int(round(3 * sigma))))
        mu = _borrosa(tr[:4000], g)
        sd = _borrosa((tr[:4000] - mu) ** 2, g).clamp_min(0).sqrt()
        eps = float(sd.median())
        if verbose:
            print(f"[datos] eps medido = {eps:.4f} (mediana de la sd local del train)")

    tr = normaliza_contraste(tr, sigma, eps)
    va = normaliza_contraste(va, sigma, eps)
    if limite:
        g = torch.Generator().manual_seed(0)
        tr = tr[torch.randperm(tr.shape[0], generator=g)[:limite]]

    var = float(tr.var())          # constante FIJA del train (decisión 2)
    if verbose:
        print(f"[datos] var(x) del train = {var:.4f}  ->  denominador fijo de la pérdida")
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(fich, train=tr.numpy(), val=va.numpy(),
             eps=np.float32(eps), var=np.float32(var))
    return {"train": tr.numpy(), "val": va.numpy(), "eps": eps, "var": var}


# ---------------------------------------------------------------- 2. el modelo

class SondaL1(nn.Module):
    """Conv(1->K) + ReLU  ->  z  ->  ConvTranspose(K->1) sin sesgo.

    Stride 1 y padding k//2 en los dos lados: la resolución se conserva. No
    queremos una imagen más pequeña, queremos una más genérica al mismo tamaño.

    ⚠ El codificador replica el borde (`padding_mode='replicate'`, igual que
    `pad_mode: edge`). El decodificador NO puede: `nn.ConvTranspose2d` de PyTorch
    sólo admite `padding_mode='zeros'` -- lo comprueba su `__init__` y lanza. O
    sea que la reconstrucción del anillo exterior de k//2 píxeles ve ceros donde
    el codificador vio borde replicado. Con k=9 eso son 4 de cada 10 píxeles por
    lado, así que NO es un detalle: por eso `err_rec` se reporta también sobre el
    interior (`err_rec_int`), que es la cifra limpia.
    """

    def __init__(self, K: int, k: int):
        super().__init__()
        self.K, self.k = K, k
        self.enc = nn.Conv2d(1, K, k, stride=1, padding=k // 2,
                             padding_mode="replicate", bias=True)
        self.dec = nn.ConvTranspose2d(K, 1, k, stride=1, padding=k // 2, bias=False)

    def forward(self, x):
        z = F.relu(self.enc(x))
        return self.dec(z), z

    @torch.no_grad()
    def renormaliza(self):
        """Cada átomo del decodificador a norma L2 = 1. Después de CADA paso."""
        w = self.dec.weight                      # (K, 1, k, k)
        n = w.flatten(1).norm(dim=1).clamp_min(1e-8)
        w.div_(n.view(-1, 1, 1, 1))


# ---------------------------------------------------------------- 3. métricas

def _base_clasica(k: int) -> torch.Tensor:
    """Los 6 filtros clásicos a tamaño k, ortonormalizados: (6, k*k).

    DC, Sobel-x, Sobel-y, laplaciano y las dos diagonales. Para k>3 se
    construyen con la misma receta separable que los define en 3x3 (suavizado
    [1,2,1] -> binomial de orden k, derivada [-1,0,1] -> diferencia central
    escalada), que es la generalización estándar. Se ortonormaliza con QR porque
    en crudo no son ortogonales y la "energía en el subespacio" pediría una
    proyección oblicua, que no es lo que mide la premisa.
    """
    ejes = torch.arange(k, dtype=torch.float64) - (k - 1) / 2
    suav = torch.from_numpy(np.array([math.comb(k - 1, i) for i in range(k)],
                                     dtype=np.float64))
    suav = suav / suav.sum()
    der = ejes.clone()
    lap = ejes ** 2 - (ejes ** 2).mean()
    dc = torch.ones(k, dtype=torch.float64)

    def outer(a, b):
        return torch.outer(a, b).flatten()

    fil = torch.stack([
        outer(dc, dc),            # DC
        outer(suav, der),         # Sobel-x
        outer(der, suav),         # Sobel-y
        outer(suav, lap) + outer(lap, suav),   # laplaciano
        outer(der, der),          # diagonal /
        outer(lap, lap),          # diagonal \ (curvatura cruzada)
    ])
    q, _ = torch.linalg.qr(fil.T)             # (k*k, 6) ortonormal
    return q.T.float()


@torch.no_grad()
def metricas(m: SondaL1, val: torch.Tensor, var: float, lam: float,
             lote: int = 512) -> dict:
    k, K = m.k, m.K
    W = m.enc.weight.detach().flatten(1)          # (K, k*k) -- los kernels de L1
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)

    B = _base_clasica(k)                          # (6, k*k)
    energia = (Wn @ B.T).pow(2).sum(1)            # fracción por kernel
    nulo = 6.0 / (k * k)

    # dimensión efectiva: participation ratio de los autovalores. NO "componentes
    # al 95 %", que está topado por min(K, k*k) y no se puede comparar entre K.
    lamb = torch.linalg.svdvals(W - W.mean(0, keepdim=True)) ** 2
    pr = float(lamb.sum() ** 2 / lamb.pow(2).sum().clamp_min(1e-12))

    cos = (Wn @ Wn.T).abs()
    cos.fill_diagonal_(0.0)

    err, err_int, act, n = 0.0, 0.0, None, 0
    borde = k // 2
    for i in range(0, val.shape[0], lote):
        x = val[i:i + lote]
        xh, z = m(x)
        b = x.shape[0]
        err += float(((xh - x) ** 2).mean()) * b
        if borde and x.shape[-1] > 2 * borde:
            c = slice(borde, -borde)
            err_int += float(((xh[..., c, c] - x[..., c, c]) ** 2).mean()) * b
        else:
            err_int += float(((xh - x) ** 2).mean()) * b
        a = (z > 0).float().mean((0, 2, 3))
        act = a * b if act is None else act + a * b
        n += b
    act = act / n

    return {
        "energia_6d": float(energia.mean()),
        "energia_6d_sd": float(energia.std()),
        "nulo_6d": nulo,
        "enriquecimiento": float(energia.mean()) / nulo,
        "dim_efectiva": pr,
        "dim_max": float(min(K, k * k)),
        "coseno_max": float(cos.max()),
        "n_pares_dup": int((cos > 0.9).sum() // 2),
        "err_rec": err / n / var,
        "err_rec_int": err_int / n / var,
        "frac_activa": float(act.mean()),
        "kernels_muertos": int((act < 1e-4).sum()),
        "lambda": lam,
    }


# ---------------------------------------------------------------- 4. un run

def entrenar(datos: dict, K: int, k: int, lam: float, semilla: int,
             epocas: int, lote: int, lr: float, verbose: bool = True) -> dict:
    torch.manual_seed(semilla)
    tr = torch.from_numpy(datos["train"])
    va = torch.from_numpy(datos["val"])
    var = datos["var"]

    m = SondaL1(K, k)
    m.renormaliza()                     # también ANTES del primer paso
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(semilla)

    t0 = time.time()
    for ep in range(epocas):
        perm = torch.randperm(tr.shape[0], generator=g)
        tot, nb = 0.0, 0
        for i in range(0, tr.shape[0], lote):
            x = tr[perm[i:i + lote]]
            xh, z = m(x)
            rec = ((xh - x) ** 2).mean() / var
            loss = rec + lam * z.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m.renormaliza()             # decisión 1: después de CADA paso
            tot += float(loss); nb += 1
        if verbose and (ep + 1) % 10 == 0:
            print(f"    época {ep+1:3d}/{epocas}  loss {tot/nb:.4f}  "
                  f"({time.time()-t0:.0f} s)")

    r = metricas(m, va, var, lam)
    r.update(K=K, k=k, semilla=semilla, epocas=epocas,
             params=sum(p.numel() for p in m.parameters()),
             segundos=round(time.time() - t0, 1))
    return r


# ---------------------------------------------------------------- 5. CLI

def nombre(K: int, k: int, lam: float) -> str:
    return f"k{k}-K{K}-l{lam}"


def main() -> int:
    p = argparse.ArgumentParser(description="Sonda L1: ¿aprende L1 filtros genéricos?")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--lote", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--eps", type=float, default=None,
                   help="por defecto se MIDE: mediana de la sd local del train")
    p.add_argument("--limite", type=int, default=None,
                   help="submuestrea el train a N ventanas (de 84.000)")
    p.add_argument("--semillas", default="1")
    p.add_argument("--solo", default=None, help="una combinación: k9-K32-l0.1")
    p.add_argument("--con-k3", dest="k3", action="store_true", default=True,
                   help="incluye el ancla k=3 (por defecto sí)")
    p.add_argument("--sin-k3", dest="k3", action="store_false")
    p.add_argument("--cronometrar", action="store_true",
                   help="mide UNA combinación y estima la rejilla entera. No la lanza")
    p.add_argument("--rejilla", action="store_true", help="lanza la rejilla completa")
    p.add_argument("--salida", default=None)
    a = p.parse_args()

    cache = Path(os.environ.get("FV_SONDA_CACHE", "/tmp/sonda-l1-cache"))
    salida = Path(a.salida) if a.salida else settings.data_root() / "sondas" / "l1"

    ks = (KS_ANCLA if a.k3 else []) + KS_ENCARGO
    combos = [(K, k, l) for k in ks for K in CANALES for l in LAMBDAS]
    if a.solo:
        combos = [c for c in combos if nombre(c[0], c[1], c[2]) == a.solo]
        if not combos:
            raise SystemExit(f"'{a.solo}' no está en la rejilla")
    semillas = [int(s) for s in a.semillas.split(",")]

    datos = preparar(a.dataset, a.sigma, a.eps, cache, a.limite)

    if a.cronometrar:
        K, k, lam = 32, 9, 0.1          # la más cara de la rejilla
        print(f"\n[cronometrar] la combinación MÁS CARA: k={k} K={K} λ={lam}, "
              f"2 épocas para extrapolar")
        r = entrenar(datos, K, k, lam, 1, 2, a.lote, a.lr)
        seg_ep = r["segundos"] / 2
        coste = sum(KK * kk * kk for kk, KK, _ in
                    [(c[1], c[0], c[2]) for c in combos])
        unidad = seg_ep / (K * k * k)
        total = coste * unidad * a.epocas
        print(f"\n  {seg_ep:.1f} s/época  ->  {seg_ep*a.epocas/60:.1f} min este run")
        print(f"  rejilla de {len(combos)} runs x {a.epocas} épocas  ->  "
              f"~{total/3600:.1f} h en esta máquina (2 vCPU), extrapolado por K·k²")
        print(f"  (+ repetir las 3 mejores con 3 semillas: ~{total/len(combos)*6/3600:.1f} h más)")
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    if not a.rejilla and not a.solo:
        p.error("elige --cronometrar, --rejilla o --solo <combinación>")

    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for idx, (K, k, lam) in enumerate(combos, 1):
        for s in semillas:
            n = f"{nombre(K, k, lam)}-s{s}"
            print(f"\n[{idx}/{len(combos)}] {n}  ({K*k*k*2+K} parámetros)")
            r = entrenar(datos, K, k, lam, s, a.epocas, a.lote, a.lr)
            r["nombre"] = n
            filas.append(r)
            (salida / f"{n}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
            print(f"    enriquecimiento {r['enriquecimiento']:.2f}x  "
                  f"dim_ef {r['dim_efectiva']:.1f}/{r['dim_max']:.0f}  "
                  f"cos_max {r['coseno_max']:.2f}  "
                  f"activa {r['frac_activa']*100:.1f}%  "
                  f"err {r['err_rec_int']:.3f}")
    (salida / "resumen.json").write_text(json.dumps(
        {"dataset": a.dataset, "epocas": a.epocas, "eps": datos["eps"],
         "var": datos["var"], "runs": filas}, indent=2, ensure_ascii=False))
    print(f"\nresultados en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
