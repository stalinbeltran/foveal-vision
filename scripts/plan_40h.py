r"""plan_40h.py — unattended screening + confirmation over ~40 h, resumable.

The decision rules are FROZEN in docs/plan-40h.md, committed before a single run
of this plan existed; this file only executes them. Re-running with no arguments
continues where it stopped (the machine loses power: that is the design premise,
not an edge case).

  .\.venv\Scripts\python.exe scripts\plan_40h.py

ASCII output only: the console is cp1252.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv.models.derive import full_config                      # noqa: E402
from fv.sweeps.generate import generate_sweep                 # noqa: E402
from fv.sweeps.runner import run_sweep                        # noqa: E402
from fv.sweeps.store import SweepStore                        # noqa: E402
from fv.sweeps.winner import suggest_winner                   # noqa: E402
from fv.training.loop import train                            # noqa: E402
from fv.training.recipe import Recipe                          # noqa: E402
from fv.training.registry import RunStore                     # noqa: E402

# ---------------------------------------------------------------- constants
# every one of these is justified in docs/plan-40h.md section 1
DATASET = "dirty1000-80px-16px"
LR = 0.0014            # winner of d1000-lr-1  (at the LEFT EDGE of its range)
BATCH = 85             # winner of d1000-batch_size-1
PATIENCE = 10          # measured floor: longest no-improve streak followed by
                       # an improvement is 6 epochs over the 70 d1000 runs
EPOCHS_SCREEN = 100    # cap; patience decides where each config actually stops
DELTA = 0.0067         # 1-SE of the 5 seeds of d1000-lr-1's winning point
SEEDS_CONFIRM = 5
BUDGET_HOURS = 36.0    # the confirmation sweep must fit under this. Was 34 when
                       # the user offered "unas 30 h"; they then offered 40+, so
                       # this is the stated budget, not a number tuned to a result
PREFIX = "p40-"
REPORT = ROOT / "plan-40h-report.json"

# the four screening configs (block 1). `lever` names what each one moves.
SCREEN = [
    ("base",   "referencia",       {}),
    ("depth",  "profundidad",      {"n_layers": 4, "channels": [16, 16, 16, 16]}),
    ("width",  "capacidad",        {"channels": [32, 32]}),
    # corners are labelled on the FOVEA only (contract (1)a): widen the centre
    # kernel, leave the periphery at 3
    ("kernel", "campo receptivo",  {"k_center": 5}),
]

# lever -> (axis, range) for block 2, fixed in advance (docs/plan-40h.md section 3.4)
NEXT_AXIS = {
    # [2..5], no [1..5]: el rango debe RODEAR al ganador del cribado y contener
    # L2, que es la red actual y la referencia de toda la afirmacion. L1 (campo
    # receptivo 3x3) esta dominado por el propio cribado. Ver docs/plan-40h.md S7
    #
    # El ORDEN no es el natural, y es a proposito: los puntos se entrenan en el
    # orden de la lista, asi que si el presupuesto se queda corto lo que falta
    # son los ultimos. Primero el ganador del cribado (4) y la referencia (2)
    # -- los dos que responden "la profundidad gana a la red actual" -- y al
    # final el 5, que solo afina donde esta el optimo. El ranking agrega por
    # valor, no por orden: esto no cambia ningun resultado, solo que se pierde
    # si algo se corta.
    "depth":  ("n_layers", [4, 2, 3, 5]),
    "width":  ("channels", [[16, 16], [24, 24], [32, 32], [48, 48], [64, 64]]),
    "kernel": ("k_center", "auto"),
    # no lever cleared delta -> lr was never bracketed from below
    "none":   ("lr", [0.0004, 0.0006, 0.0008, 0.0011, 0.0014]),
}


# what each screening run set on its axis — the value a trimmed range must keep
SCREEN_VALUE = {"depth": 4, "width": [32, 32], "kernel": 5, "none": None}

# fields whose value is only valid TOGETHER with the axis: carrying them from
# the screening winner while the axis returns to its default breaks the base
COUPLED_TO = {"n_layers": ("channels",), "channels": ("n_layers",)}


def trim_around(values: list, winner, keep: int = 3) -> list:
    """Keep `keep` values centred on the screening winner, never dropping it."""
    if winner is None or winner not in values:
        return values[:keep]
    i = values.index(winner)
    lo = max(0, min(i - keep // 2, len(values) - keep))
    return values[lo:lo + keep]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def recipe(**over) -> Recipe:
    fields = dict(lr=LR, batch_size=BATCH, epochs=EPOCHS_SCREEN, patience=PATIENCE,
                  optimizer="adam", monitor="val_loss", seed=1)
    fields.update(over)
    return Recipe(**fields)


# ------------------------------------------------------------------ block 1
def screen(rstore: RunStore) -> dict:
    """Train the four screening runs, skipping the ones already done."""
    results = {}
    for lever, what, over in SCREEN:
        name = f"{PREFIX}screen-{lever}"
        cfg = full_config(dict(over))
        if rstore.exists(name):
            st = (rstore.status(name) or {}).get("status")
            if st == "done":
                log(f"{name}: ya hecho, se salta")
                results[lever] = read_run(rstore, name, cfg)
                continue
            log(f"{name}: quedo en '{st}' -- se rehace")
            drop(rstore, name)
        log(f"{name} ({what}): {cfg['n_layers']} capas, canales {cfg['channels']}, "
            f"k_center={cfg['k_center']} -- hasta {EPOCHS_SCREEN} epocas, patience {PATIENCE}")
        train(name, DATASET, None, cfg, "plan40", recipe(), device="cpu", store=rstore)
        results[lever] = read_run(rstore, name, cfg)
        log(f"  -> val_loss={results[lever]['val_loss']:.4f} f1={results[lever]['f1']:.4f} "
            f"mejor epoca {results[lever]['best_epoch']}/{results[lever]['epochs_run']} "
            f"({results[lever]['seconds_per_epoch']:.0f} s/epoca)")
        save_report({"screen": results})
    return results


def drop(rstore: RunStore, name: str) -> None:
    """Delete an unfinished run of OURS. Nothing without the p40- prefix."""
    if not name.startswith(PREFIX):
        raise SystemExit(f"rehusado: {name} no lleva el prefijo {PREFIX}")
    p = rstore.path(name)
    for f in sorted(p.rglob("*"), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    p.rmdir()


def read_run(rstore: RunStore, name: str, cfg: dict) -> dict:
    """The value of a run is its CHECKPOINT's epoch, not the last one."""
    p = rstore.path(name)
    summary = json.loads((p / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (p / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    best = next(r for r in rows if r["epoch"] == summary["best_epoch"])
    return {"run": name, "config": cfg,
            "val_loss": best["val"]["loss"], "f1": best["val"]["f1"],
            "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
            "seconds_per_epoch": summary["seconds_per_epoch"],
            "curve": [r["val"]["loss"] for r in rows]}


# ---------------------------------------------------------- decision (S 3)
def decide(res: dict) -> tuple[str, str]:
    """The rule, verbatim from docs/plan-40h.md section 3. Returns (lever, why)."""
    base = res["base"]["val_loss"]
    gains = {k: base - v["val_loss"] for k, v in res.items() if k != "base"}
    cand = {k: g for k, g in gains.items() if g > DELTA}
    detail = ", ".join(f"{k} {g:+.4f}" for k, g in sorted(gains.items(), key=lambda kv: -kv[1]))
    if not cand:
        return "none", (f"ningun resorte supera delta={DELTA} sobre la base "
                        f"(mejoras: {detail}); se barre lr hacia abajo, que quedo "
                        f"sin acotar en el borde izquierdo de d1000-lr-1")
    top = max(cand.values())
    tied = [k for k, g in cand.items() if top - g <= DELTA]
    win = min(tied, key=lambda k: res[k]["seconds_per_epoch"]) if len(tied) > 1 else tied[0]
    why = f"mejoras sobre la base: {detail}; delta={DELTA}"
    if len(tied) > 1:
        why += (f"; empatan {sorted(tied)} dentro de delta -> gana el mas barato "
                f"({win}, {res[win]['seconds_per_epoch']:.0f} s/epoca)")
    return win, why


def confirm_epochs(res: dict) -> int:
    """Budget measured, not guessed: 1.25x the latest best epoch seen."""
    return int(min(EPOCHS_SCREEN, max(30, math.ceil(1.25 * max(
        v["best_epoch"] for v in res.values())))))


def epochs_for(axis: str, value, epochs_cap: int, res: dict) -> int:
    """How many epochs a point will REALLY run. `epochs_cap` is a cap, not a
    length: patience=10 stopped the screening runs at 71, 32 and 57 of 100, and
    the DEEPER config stopped soonest. Costing every point at the cap
    overestimates ~2x and would spend seeds we can actually afford, so use what
    the screening measured, interpolating between the depths we observed."""
    if axis != "n_layers":
        return epochs_cap
    obs = {2: res["base"]["epochs_run"], 4: res["depth"]["epochs_run"]}
    lo, hi = min(obs), max(obs)
    if value <= lo:
        return obs[lo]
    if value >= hi:
        return obs[hi]
    return round(obs[lo] + (obs[hi] - obs[lo]) * (value - lo) / (hi - lo))


def estimate_hours(axis: str, values, epochs: int, seeds: int, res: dict) -> float:
    """~35 s of dataloader per epoch (constant) + the model's own step cost."""
    from fv.models.builder import build_model
    import torch
    total = 0.0
    for v in values:
        over = {axis: v} if axis in ("n_layers", "channels", "k_center") else {}
        if axis == "n_layers":
            over = {"n_layers": v, "channels": [16] * v}
        cfg = full_config(over)
        m = build_model(cfg)
        x = torch.randn(BATCH, 1, cfg["N"], cfg["N"])
        y = torch.randn_like(m(x))
        opt = torch.optim.Adam(m.parameters(), LR)
        for _ in range(2):
            opt.zero_grad(); ((m(x) - y) ** 2).mean().backward(); opt.step()
        t = time.perf_counter()
        for _ in range(10):
            opt.zero_grad(); ((m(x) - y) ** 2).mean().backward(); opt.step()
        ms = (time.perf_counter() - t) / 10
        ep = epochs_for(axis, v, epochs, res)
        total += (35.0 + ms * 988) * ep * seeds         # 988 steps/epoch at batch 85
    return total / 3600.0


# ------------------------------------------------------------------ block 2
def confirm(res: dict, lever: str, why: str) -> dict:
    axis, rng = NEXT_AXIS[lever]
    epochs = confirm_epochs(res)
    seeds = SEEDS_CONFIRM
    values = [3, 5, 7] if rng == "auto" else list(rng)

    est = estimate_hours(axis, values, epochs, seeds, res)
    log(f"presupuesto: eje '{axis}' x {len(values)} valores x {seeds} semillas x "
        f"{epochs} epocas = {est:.1f} h estimadas")
    if est > BUDGET_HOURS:                      # guard, in the frozen order
        seeds = 3
        est = estimate_hours(axis, values, epochs, seeds, res)
        log(f"  guarda 1: semillas 5 -> 3 => {est:.1f} h")
    if est > BUDGET_HOURS and len(values) > 3:
        # recortar SIEMPRE alrededor del valor que gano el cribado. La regla
        # original se quedaba con los 3 mas baratos, que con eje n_layers son
        # [1,2,3] -- fuera justo el 4 que hay que confirmar. Un barrido de
        # confirmacion que no contiene al candidato no confirma nada.
        values = trim_around(values, SCREEN_VALUE.get(lever))
        rng = values
        est = estimate_hours(axis, values, epochs, seeds, res)
        log(f"  guarda 2: rango recortado alrededor del ganador -> {values} => {est:.1f} h")

    name = f"{PREFIX}confirm-{axis}".replace("[", "").replace("]", "")
    sstore = SweepStore()
    if not sstore.exists(name):
        # the winning screening config is the base, minus the axis being swept
        carried = {k: v for k, v in res[lever]["config"].items()} if lever != "none" else {}
        carried.pop(axis, None)
        carried.pop("N", None); carried.pop("c_frac", None)      # derived, never carried
        # a field COUPLED to the swept axis cannot be carried either: channels
        # must have length n_layers, so carrying [16,16,16,16] while n_layers
        # goes back to its default 2 makes the BASE invalid and check_run
        # rejects it before a single point is expanded. Each point resizes
        # channels to [16]*L on its own (barrido-por-ejes.md S6.1).
        for coupled in COUPLED_TO.get(axis, ()):
            carried.pop(coupled, None)
        generate_sweep(name, DATASET, axis, rng if rng != "auto" else "auto",
                       base_recipe="plan40",
                       base_recipe_value=dataclasses.asdict(
                           recipe(epochs=epochs)) | {"seed": 1},
                       objective="f1",     # best proxy for the task metric (S 9.7)
                       budget={"epochs": epochs}, seeds=seeds,
                       overrides=carried, device="cpu")
        log(f"recorrido '{name}' creado: {len(values)} valores x {seeds} semillas")
    else:
        log(f"recorrido '{name}' ya existe -- se reanuda")
    run_sweep(name, progress=lambda d, t, r: log(f"  punto {d}/{t}: {r}"))
    verdict = suggest_winner(name)
    return {"sweep": name, "axis": axis, "values": values, "epochs": epochs,
            "seeds": seeds, "lever": lever, "why": why,
            "winner": verdict.get("best", {}).get("point"),
            "value": verdict.get("best", {}).get("value"),
            "tie": verdict.get("tie"), "tie_reason": verdict.get("tie_reason")}


def save_report(patch: dict) -> None:
    cur = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    cur.update(patch)
    REPORT.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    t0 = time.time()
    log("BLOQUE 1 -- cribado (1 semilla; NO declara ganador, solo elige eje)")
    rstore = RunStore()
    res = screen(rstore)
    lever, why = decide(res)
    log(f"DECISION: resorte '{lever}'. {why}")
    save_report({"screen": res, "decision": {"lever": lever, "why": why,
                                             "delta": DELTA}})
    log(f"BLOQUE 2 -- confirmacion, {SEEDS_CONFIRM} semillas")
    out = confirm(res, lever, why)
    save_report({"confirm": out})
    log(f"LISTO en {(time.time()-t0)/3600:.1f} h. Veredicto: {out['winner']} "
        f"(tie={out['tie']})")
    log(f"informe: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
