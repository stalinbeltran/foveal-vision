#!/usr/bin/env python3
"""Lee los tres brazos contra el criterio de `instrucciones/02-criterio.md`.

    python nn/leer_criterio.py

Evalua los `best.pt` de los cuatro (ancla + A + B + C) sobre EL MISMO split de
validacion y con LA MISMA funcion de metricas del repo (`fv.metrics`), y escribe
`resultados/resumen.json`.

⚠ La epoca que se compara es la que guarda `best.pt`, elegida por `val_loss` en
   los cuatro. Es la unica regla que no se elige a posteriori: coger el mejor f1
   de cada curva seria coger cuatro epocas distintas por cuatro criterios
   distintos, y siempre gana el que tuvo mas suerte.

⚠ ADEMAS del `pos_err_px` global, se parte el error por POSICION: el 24,1 % de
   las esquinas cae en el primer/ultimo pixel de la ventana, y el sesgo de
   contraccion del soft-argmax --si existe-- tiene que verse ahi y no en el
   interior. Sin este desglose, «A empeora» y «A empeora EN EL BORDE» se leen
   igual, y son dos causas distintas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(AQUI))

from fv.fovea import dims_of                                       # noqa: E402
from fv.metrics import (checkpoint_record, corner_scores,          # noqa: E402
                        detection_counts, pos_err_px)
from fv.models.builder import build_model, full_config             # noqa: E402
from fv.training.registry import RunStore                          # noqa: E402
from fv.windows.dataset import FoveatedWindowDataset               # noqa: E402
from fv.windows.store import WindowDatasetStore                    # noqa: E402
from entrenar_local import BRAZOS, DATASET, RED                    # noqa: E402
from red_local import CabezaSoftArgmax                             # noqa: E402

ANCLA = "plana-4k7-s1"
# ⚠ El `best.pt` de un run NO vive en el run: la regla del proyecto es que los
# pesos no se guardan por defecto, y cada experimento se copia los suyos a su
# `nn/pesos/` (regla 3 de `experimentos/README.md`). El ancla los tiene en SU
# carpeta; los tres brazos, en la de aqui (los copia `cadena.py`). Se mira
# primero el run --que es donde estan mientras entrena-- y luego el respaldo
# commiteado, que es el que sobrevive a rehacer la maquina.
RESPALDO = {
    ANCLA: REPO / "experimentos" / "2026-09-03-cnn-plana-4k7" / "nn" / "pesos",
    **{d["run"]: AQUI / "pesos" / b for b, d in BRAZOS.items()},
}
RED_YAML = REPO / "configs" / "networks" / f"{RED}.yaml"
# El umbral del criterio: ~2 sd de la dispersion epoca a epoca del ancla
# (ultimas 5 epocas: sd 0,068 px). Escrito ANTES de correr nada.
UMBRAL_POS = 0.15
UMBRAL_F1 = 0.01


def _donde(nombre: str) -> Path:
    """El directorio que tiene `best.pt` Y `metrics.jsonl`, o revienta diciendo
    los dos sitios que se miraron. Un `FileNotFoundError` a secas aqui se leeria
    como «todavia no ha entrenado», que es otra cosa."""
    for d in (RunStore().path(nombre), RESPALDO.get(nombre)):
        if d is not None and (d / "best.pt").exists() and (d / "metrics.jsonl").exists():
            return d
    raise FileNotFoundError(
        f"{nombre}: no hay best.pt+metrics.jsonl ni en {RunStore().path(nombre)} "
        f"ni en {RESPALDO.get(nombre)}")


def cargar(nombre: str, modo: str | None):
    """El modelo con sus pesos de `best.pt`. `modo=None` -> la red del repo."""
    d = _donde(nombre)
    cfg = full_config(yaml.safe_load(RED_YAML.read_text()))
    m = build_model(cfg) if modo is None else CabezaSoftArgmax(cfg, modo=modo)
    st = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(st["model"] if "model" in st else st["state_dict"])
    m.eval()
    recs = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    return m, checkpoint_record(recs, "val_loss"), len(recs)


def evaluar(m, loader, window_size: int) -> dict:
    logits, targets = [], []
    with torch.no_grad():
        for x, e, y in loader:
            logits.append(m(x, e).numpy())
            targets.append(y.numpy())
    lo = np.concatenate(logits)
    ta = np.concatenate(targets)
    ex = ta[:, :, 0]
    det = detection_counts(corner_scores(lo), ex)
    xy_p, xy_t = lo[:, :, 1:], ta[:, :, 1:]
    # borde = la esquina cae en el primer o ultimo pixel de la ventana etiquetada
    borde = ((xy_t < 1.0 / window_size) | (xy_t >= 1.0 - 1.0 / window_size)).any(axis=-1)
    return {
        "f1": det["f1"], "precision": det["precision"], "recall": det["recall"],
        "pos_err_px": pos_err_px(xy_p, xy_t, ex, window_size),
        "pos_err_px_borde": pos_err_px(xy_p, xy_t, ex * borde, window_size),
        "pos_err_px_interior": pos_err_px(xy_p, xy_t, ex * ~borde, window_size),
        "n_esquinas": int(ex.sum()),
        "frac_borde": float((ex * borde).sum() / max(1.0, ex.sum())),
    }


def main() -> int:
    cfg = full_config(yaml.safe_load(RED_YAML.read_text()))
    dims = dims_of(cfg)
    ws = WindowDatasetStore()
    arrays = ws.arrays(DATASET)
    manifest = ws.manifest(DATASET) if hasattr(ws, "manifest") else None
    window_size = int(json.loads(
        (ws.path(DATASET) / "manifest.json").read_text())["config"]["window_size"])
    val = FoveatedWindowDataset(arrays, dims, split=1, pool_mode=cfg["pool_mode"],
                                pad_mode=cfg["pad_mode"], edge_inputs=cfg["edge_inputs"],
                                mask_channel=cfg["mask_channel"])
    loader = DataLoader(val, batch_size=256, num_workers=0)

    quienes = [("ancla", ANCLA, None)]
    for b, d in BRAZOS.items():
        quienes.append((b, d["run"], d["modo"]))

    out = {"dataset": DATASET, "red": RED, "window_size": window_size,
           "umbral_pos_px": UMBRAL_POS, "umbral_f1": UMBRAL_F1, "brazos": {}}
    for etiqueta, run, modo in quienes:
        try:
            m, rec, n = cargar(run, modo)
        except FileNotFoundError as err:
            print(f"  ⚠ {etiqueta}: {err}")
            continue
        r = evaluar(m, loader, window_size)
        r["run"] = run
        r["epoca_best"] = rec["epoch"] if rec else None
        r["epocas_corridas"] = n
        r["val_loss_best"] = (rec["val"]["loss"] if rec else None)
        r["params_cabeza"] = (sum(p.numel() for p in m.head.parameters()) if modo is None
                              else m.presupuesto()["cabeza"])
        if modo == "softargmax":
            r["beta"] = float(torch.exp(m.log_beta))
        out["brazos"][etiqueta] = r
        print(f"  {etiqueta:6s} {run:14s} ep{r['epoca_best']:>3}  f1 {r['f1']:.4f}  "
              f"pos {r['pos_err_px']:.3f} px  (borde {r['pos_err_px_borde']:.3f} · "
              f"interior {r['pos_err_px_interior']:.3f})")

    a = out["brazos"].get("ancla")
    if a:
        print("\n  === contra el criterio (umbral pos ±%.2f px · f1 ±%.2f) ===" %
              (UMBRAL_POS, UMBRAL_F1))
        for b in ("A", "B", "C"):
            r = out["brazos"].get(b)
            if not r:
                continue
            dp = r["pos_err_px"] - a["pos_err_px"]
            df = r["f1"] - a["f1"]
            v = ("MEJORA" if dp < -UMBRAL_POS else
                 "EMPEORA" if dp > UMBRAL_POS else "no se mueve")
            ctrl = "ok" if abs(df) <= UMBRAL_F1 else "⚠ EL CONTROL FALLA"
            print(f"  {b}: pos {dp:+.3f} px -> {v}   |   f1 {df:+.4f} -> {ctrl}")
            r["delta_pos_px"] = dp
            r["delta_f1"] = df
            r["veredicto_pos"] = v
            r["control_f1"] = ctrl

    dest = EXP / "resultados"
    dest.mkdir(exist_ok=True)
    (dest / "resumen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  escrito {(dest / 'resumen.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
