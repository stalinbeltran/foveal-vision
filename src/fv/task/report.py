"""The task metric in words — ONE formatter, two CLIs (fv-oat, fv-study).

ASCII on purpose: these lines are printed on the Windows console (cp1252), and
an unattended study dies on the last line if a Greek letter reaches it.

Every line carries the band and the n of images, because a task F1 without them
is exactly the error this project just fixed in the ranking: with 20 images the
standard error is +-0.083 and the differences between neighbouring points are
0.01-0.05 (metrica-de-tarea.md §2).
"""

from __future__ import annotations

SMALL_SAMPLE = 100   # below this the number cannot separate nearby points


def _one(payload: dict) -> str:
    m = payload["macro"]
    sem = f" +- {m['sem']:.4f}" if m["sem"] is not None else ""
    iou = f"{payload['mean_iou']:.3f}" if payload["mean_iou"] is not None else "sin emparejar"
    return (f"    {payload['run']}: macro f1 {m['f1']:.4f}{sem} "
            f"(n={payload['images']} imagenes)  micro f1 "
            f"{payload['micro']['f1']:.4f}  IoU medio {iou}")


def task_lines(runs: list[str], split: str = "val", store=None) -> list[str]:
    """One line per run, plus the caveat when the split is too small.

    A run that cannot be scored says why and does NOT stop the caller: this is
    an informative report at the end of a sweep, never a gate.
    """
    from fv.task import task_score
    lines, payloads = [], []
    # by name, so the seeds of a point read in order: the caller hands them in
    # ranking order, which is not the order a human reads a list of replicas in
    for run in sorted(runs):
        try:
            payloads.append(task_score(run, split, store=store))
        except Exception as e:  # noqa: BLE001 — surface code/message, keep going
            lines.append(f"    {run}: sin metrica de tarea "
                         f"[{getattr(e, 'code', 'error')}] {e}")
    for p in payloads:
        lines.append(_one(p))
    if len(payloads) > 1:
        mean = sum(p["macro"]["f1"] for p in payloads) / len(payloads)
        lines.append(f"    media de las {len(payloads)} semillas: {mean:.4f}")
    if payloads and payloads[0]["images"] < SMALL_SAMPLE:
        sem = payloads[0]["macro"]["sem"]
        band = f"+-{sem:.3f}" if sem is not None else "desconocido"
        lines.append(f"    aviso: con {payloads[0]['images']} imagenes el error "
                     f"estandar es {band}: este numero NO distingue diferencias "
                     f"menores que eso (docs/metrica-de-tarea.md, seccion 4)")
    return lines


def print_task_score(title: str, runs: list[str], split: str = "val",
                     store=None) -> None:
    print(f"\n{title}")
    for line in task_lines(runs, split, store):
        print(line)
