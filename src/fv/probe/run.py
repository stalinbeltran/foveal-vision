"""One run of the probe, and the artefacts it leaves behind.

THE GRID (section 4 of the brief)
---------------------------------
Full grid, not OAT: the interaction between `k` and lambda is the point.

    k       5, 7, 9     main axis: structure does not fit in 3x3
    K       8, 16, 32   with k=9 and K=32 the code is 32x overcomplete
    lambda  0, .03, .1, .3   lambda=0 is the mandatory control (PCA-like case)

`k=3` is added as an ANCHOR and is NOT in the brief's grid. Without it nothing
compares against the 0.688 measured on `fov16-mask-p20`, which is what motivates
the whole experiment. It costs 12/48 of the total.

THE ARTEFACTS, AND WHERE THEY LAND IN GIT
-----------------------------------------
Per run: `config.json`, `metrics.jsonl` (one line per epoch), `checkpoint.pt`,
`kernels_enc.npy`, `kernels_dec.npy`, `summary.json`.

⚠ `checkpoint.pt` does NOT enter git and that is deliberate, not an oversight:
`foveal-vision-data/.gitignore` drops every `.pt` except `runs/*/best.pt` and
`runs/*/last.pt`, because the owner fixed on 2026-08-31 that run weights are not
kept by default. THE KERNELS ARE THE DELIVERABLE HERE (section 1 of the brief:
"the model *is* the kernels"), and they travel as `.npy`, which does enter. So
the experiment stays reproducible from git while the rule stands. The checkpoint
is a local working artefact: 21 KB at most, worth having on the machine that ran
it, not worth a rule exception.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from fv.probe.metrics import final_metrics
from fv.probe.model import L1Probe

GRID_KS = [5, 7, 9]
GRID_KS_ANCHOR = [3]
GRID_CHANNELS = [8, 16, 32]
GRID_LAMBDAS = [0.0, 0.03, 0.1, 0.3]


def run_name(channels: int, k: int, lam: float) -> str:
    return f"k{k}-K{channels}-l{lam}"


@torch.no_grad()
def _val_pass(m: L1Probe, val: torch.Tensor, var: float, batch: int) -> tuple[float, float]:
    err, act, n = 0.0, 0.0, 0
    for i in range(0, val.shape[0], batch):
        x = val[i:i + batch]
        xh, z = m(x)
        b = x.shape[0]
        err += float(((xh - x) ** 2).mean()) * b
        act += float((z > 0).float().mean()) * b
        n += b
    return err / n / var, act / n


def fit(data: dict, channels: int, k: int, lam: float, seed: int,
        epochs: int, batch: int, lr: float, out_dir: Path | None = None,
        val_log: int = 4096, verbose: bool = True,
        name: str | None = None) -> tuple[L1Probe, list[dict], dict]:
    """The training loop alone: returns the model and its per-epoch curve.

    Split out from `train` so that `fv.probe.calibrate` can reuse it without a
    second copy of the loop. Two copies of a loop whose invariant is "renormalise
    after EVERY step" is exactly the shape of a silent divergence.

    `val_log` subsamples validation for the PER-EPOCH line of `metrics.jsonl`
    (a full 28,000-window pass every epoch is ~11 % overhead for a curve nobody
    reads at that precision). The FINAL metrics always use the whole split.
    """
    torch.manual_seed(seed)
    tr = torch.from_numpy(data["train"])
    va = torch.from_numpy(data["val"])
    var = data["var"]
    va_log = va[:val_log] if val_log and val_log < va.shape[0] else va

    m = L1Probe(channels, k)
    m.renormalize()                     # also BEFORE the first step
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    name = name or f"{run_name(channels, k, lam)}-s{seed}"
    lines: list[dict] = []
    jsonl = None
    if out_dir is not None:
        out_dir = Path(out_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({
            "nombre": name, "K": channels, "k": k, "lambda": lam, "semilla": seed,
            "epocas": epochs, "lote": batch, "lr": lr,
            "params": sum(p.numel() for p in m.parameters()),
            "var_train": var, "eps": data.get("eps"),
            "n_train": int(tr.shape[0]), "n_val": int(va.shape[0]),
            "val_log": int(va_log.shape[0]),
        }, indent=2, ensure_ascii=False))
        jsonl = (out_dir / "metrics.jsonl").open("w")

    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(tr.shape[0], generator=g)
        tot, rec_tot, pen_tot, nb = 0.0, 0.0, 0.0, 0
        for i in range(0, tr.shape[0], batch):
            x = tr[perm[i:i + batch]]
            xh, z = m(x)
            rec = ((xh - x) ** 2).mean() / var
            pen = z.abs().mean()
            loss = rec + lam * pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            m.renormalize()             # after EVERY step -- see model.py
            tot += float(loss.detach()); rec_tot += float(rec.detach())
            pen_tot += float(pen.detach()); nb += 1
        v_err, v_act = _val_pass(m, va_log, var, batch)
        fila = {"epoca": ep + 1, "loss": tot / nb, "rec": rec_tot / nb,
                "pena": pen_tot / nb, "val_err_rec": v_err, "val_frac_activa": v_act,
                "segundos": round(time.time() - t0, 1)}
        lines.append(fila)
        if jsonl is not None:
            jsonl.write(json.dumps(fila, ensure_ascii=False) + "\n")
            jsonl.flush()
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoca {ep+1:3d}/{epochs}  loss {tot/nb:.4f}  "
                  f"val_err {v_err:.3f}  activa {v_act*100:.1f}%  "
                  f"({time.time()-t0:.0f} s)")
    if jsonl is not None:
        jsonl.close()
    return m, lines, {"salida": out_dir, "segundos": time.time() - t0, "nombre": name}


def train(data: dict, channels: int, k: int, lam: float, seed: int,
          epochs: int, batch: int, lr: float, out_dir: Path | None = None,
          val_log: int = 4096, verbose: bool = True, gabor_steps: int = 400,
          name: str | None = None, extra: dict | None = None) -> dict:
    """One full run: train, measure the eight metrics, leave the artefacts."""
    m, lines, meta = fit(data, channels, k, lam, seed, epochs, batch, lr,
                         out_dir=out_dir, val_log=val_log, verbose=verbose, name=name)
    va = torch.from_numpy(data["val"])
    out_dir, name = meta["salida"], meta["nombre"]

    r = final_metrics(m, va, data["var"], lam, gabor_steps=gabor_steps)
    r.update(nombre=name, K=channels, k=k, semilla=seed, epocas=epochs,
             params=sum(p.numel() for p in m.parameters()),
             segundos=round(meta["segundos"], 1), **(extra or {}))

    if out_dir is not None:
        torch.save({"state_dict": m.state_dict(), "K": channels, "k": k},
                   out_dir / "checkpoint.pt")
        np.save(out_dir / "kernels_enc.npy",
                m.encoder_kernels().view(channels, k, k).numpy())
        np.save(out_dir / "kernels_dec.npy",
                m.decoder_kernels().view(channels, k, k).numpy())
        (out_dir / "summary.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    r["_curva"] = lines
    return r
