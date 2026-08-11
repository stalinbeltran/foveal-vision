r"""Fase 3b (metrica-de-tarea.md §5): does the WINDOW proxy rank the same as the
TASK metric on an axis of C?

§2 validated the proxy on `lr` (D), which changes neither the architecture nor
the foveated view. An axis of C changes the RULE OF LOOKING, and protocolo.md §2
is explicit that no ranking metric may depend on the view. If the proxy degrades
when geometry moves, every OAT study is carrying forward the wrong winner.

This script computes NOTHING itself: every number comes from its single
definition — `sweep_trials` (the window value at the epoch best.pt kept),
`fv.task.task_score` (paragraph F1 per image, cached), `fv.metrics.spearman`,
and `suggest_winner` (the delta frontier). It refuses to fill in a run without a
checkpoint: a missing value is not a zero (formatos.md §2).

    .\.venv\Scripts\python scripts\proxy_vs_task.py --sweep proxy-c-d [--split val] [--json out.json]

ASCII output (cp1252 console).
"""

from __future__ import annotations

import argparse
import json
import time

from fv.metrics import permutation_test, spearman
from fv.sweeps.runner import sweep_trials
from fv.sweeps.spec import NETWORK_PARAMS, RECIPE_PARAMS
from fv.sweeps.winner import suggest_winner
from fv.task import task_score
from fv.training.registry import RunError

# The criterion of §5.4, WRITTEN BEFORE LOOKING (protocolo.md §1). Constants, so
# nobody nudges them after seeing the number.
SPEARMAN_PASS = 0.90
MIN_POINTS = 4


def _point_str(point: dict) -> str:
    return json.dumps(point, sort_keys=True, separators=(", ", ": "))


def _axes_of(rows: list[dict]) -> tuple[list[str], str]:
    """Which fields the sweep moves, and whether they are C (the view) or D.

    The whole question of Fase 3b is C-vs-D: a verdict that says «vale para C»
    after measuring an axis of `lr` would be a lie that reads like a result.
    """
    axes = sorted({k for r in rows for k in r["point"] if k != "seed"})
    dom = ("C (la red: cambia la vista foveada)" if all(a in NETWORK_PARAMS for a in axes)
           else "D (la receta: no cambia la vista)" if all(a in RECIPE_PARAMS for a in axes)
           else "C y D mezclados")
    return axes, dom


def _task_of(run: str, split: str, cache: dict) -> dict | None:
    """macro F1 of one run, or None WITH the reason printed — never a zero."""
    if run in cache:
        return cache[run]
    try:
        t = task_score(run, split)
    except RunError as e:
        print(f"  !! {run}: [{e.code}] {e.message}")
        cache[run] = None
        return None
    cache[run] = {"f1": t["macro"]["f1"], "sem": t["macro"]["sem"],
                  "sd": t["macro"]["sd"], "images": t["images"],
                  "micro_f1": t["micro"]["f1"], "cached": t["cached"]}
    return cache[run]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Correlacion proxy de ventana <-> metrica de tarea (Fase 3b)")
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--objective", default=None,
                    help="re-lee el mismo recorrido con OTRO proxy de ventana "
                         "(f1, pos_err_px, loss) sin tocar el spec — apartado 9.7")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="guarda el detalle por run y por punto")
    args = ap.parse_args()

    t0 = time.time()
    trials = sweep_trials(args.sweep, objective=args.objective)
    objective = trials["objective"]
    direction = trials["direction"]
    rows = trials["trials"]
    scored = [r for r in rows if r["value"] is not None]
    skipped = [r for r in rows if r["value"] is None]

    axes, domain = _axes_of(rows)
    print(f"recorrido: {args.sweep}   split: {args.split}")
    print(f"eje(s): {', '.join(axes)}   dominio: {domain}")
    print(f"objetivo de ventana: {objective} ({direction})   "
          f"valor de: {trials.get('value_from')}")
    if trials.get("objective_overridden"):
        print(f"  RE-LECTURA: el recorrido se entreno y ordeno con "
              f"'{trials['sweep_objective']}'; aqui se re-rankea con "
              f"'{objective}' sobre los MISMOS runs (apartado 9.7)")
    if not trials.get("monitor_matches_objective"):
        print(f"  aviso: el checkpoint lo elige {trials['monitors']}, "
              f"el ranking mide val_{objective} (legal, pero hay que saberlo)")
    print(f"runs: {len(rows)}; con valor de ventana: {len(scored)}; "
          f"sin valor (no rankeados, NO son ceros): {len(skipped)}")
    for r in skipped:
        why = (r.get("value_reason") or {}).get("message", f"status={r['status']}")
        print(f"  -- {r['run']}: {why}")
    if not scored:
        print("\nSIN DATOS: ningun run tiene valor de ventana todavia.")
        return 1

    # ---- por run -----------------------------------------------------------
    print(f"\nmidiendo la tarea de {len(scored)} runs ...")
    cache: dict = {}
    per_run = []
    for r in scored:
        t = _task_of(r["run"], args.split, cache)
        if t is None:
            continue
        per_run.append({"run": r["run"], "point": r["point"],
                        "window": r["value"], "task": t["f1"],
                        "task_sem": t["sem"], "images": t["images"],
                        "micro_f1": t["micro_f1"]})
    if len(per_run) < 2:
        print("\nSIN DATOS: menos de dos runs con las dos metricas.")
        return 1

    # orient the window series so that HIGHER IS BETTER, whatever the objective:
    # a `min` objective (pos_err_px) would otherwise anti-correlate by definition
    # and the sign would say the opposite of what it means.
    sign = 1.0 if direction == "max" else -1.0
    sp_run = spearman([sign * p["window"] for p in per_run],
                      [p["task"] for p in per_run])

    # ---- agregado por valor del eje ---------------------------------------
    w = suggest_winner(args.sweep, objective=args.objective)
    groups = []
    for g in w["trials"]:
        vals = [cache[r]["f1"] for r in g["runs"]
                if cache.get(r) is not None]
        if not vals:
            continue
        groups.append({"point": g["point"], "window": g["value"],
                       "task": sum(vals) / len(vals), "n_seeds": g["n_seeds"],
                       "n_task": len(vals), "window_sem": g.get("value_sem"),
                       "task_seeds": vals})
    sp_agg = spearman([sign * g["window"] for g in groups],
                      [g["task"] for g in groups]) if len(groups) >= 2 else None

    groups_by_window = sorted(groups, key=lambda g: g["window"],
                              reverse=(direction == "max"))
    print(f"\n{'punto':<24}{'ventana':>10}{'tarea':>10}{'n_seeds':>9}")
    for g in groups_by_window:
        print(f"{_point_str(g['point']):<24}{g['window']:>10.4f}"
              f"{g['task']:>10.4f}{g['n_seeds']:>9}")

    # ---- esta diferencia, ?es mas grande que reetiquetar las semillas? ------
    # The correlation says whether the two metrics AGREE; it says nothing about
    # whether the task differences are real. With 5 seeds that question gets
    # answered in every write-up, so it is answered HERE, once, by the tested
    # definition in fv.metrics — not by eyeballing a standard error.
    best_task = max(groups, key=lambda g: g["task"])
    pairs = []
    for g in groups_by_window:
        if _point_str(g["point"]) == _point_str(best_task["point"]):
            continue
        r = permutation_test(best_task["task_seeds"], g["task_seeds"])
        if r is None:
            continue
        pairs.append({"vs": g["point"], "diff": r["diff"], "p": r["p"],
                      "arrangements": r["arrangements"]})
    if pairs:
        print(f"\nel ganador por tarea contra cada punto "
              f"(permutacion exacta de las semillas, 2 colas):")
        for p in pairs:
            print(f"  {_point_str(best_task['point'])} vs {_point_str(p['vs']):<20}"
                  f"dif {p['diff']:+.4f}   p = {p['p']:.3f}   "
                  f"({p['arrangements']} arreglos)")

    # ---- el veredicto de §5.4 ---------------------------------------------
    frontier_pts = [_point_str(t["point"]) for t in w["frontier"]]
    task_winner_in_frontier = _point_str(best_task["point"]) in frontier_pts

    print(f"\nSpearman por run       (n={len(per_run)}): "
          f"{'None' if sp_run is None else f'{sp_run:+.3f}'}")
    print(f"Spearman agregado      (n={len(groups)}): "
          f"{'None' if sp_agg is None else f'{sp_agg:+.3f}'}")
    print(f"ganador por ventana : {_point_str(w['best']['point'])}  "
          f"(sugerido: {_point_str(w['suggested']['point'])})")
    print(f"ganador por tarea   : {_point_str(best_task['point'])}  "
          f"F1={best_task['task']:.4f}")
    print(f"frontera delta={w['delta']:.4f} ({w['delta_source']}): "
          f"{', '.join(frontier_pts)}")
    print(f"  el ganador por tarea {'CAE DENTRO' if task_winner_in_frontier else 'QUEDA FUERA'}"
          f" de la frontera")

    # inconclusive FIRST: a sweep that distinguishes nothing makes any
    # correlation meaningless, so it must not be read as a pass or a fail.
    if len(groups) < MIN_POINTS:
        verdict, code = (f"NO CONCLUYENTE: solo {len(groups)} puntos con valor "
                         f"(hacen falta >= {MIN_POINTS})"), "inconclusive"
    elif sp_agg is None:
        verdict, code = ("NO CONCLUYENTE: una de las dos series es constante, "
                         "la correlacion no esta definida"), "inconclusive"
    elif pairs and all(p["p"] > 0.05 for p in pairs):
        # The window frontier can be narrow — delta is 1-SE of ONE point's seeds
        # — while the TASK side separates nothing at all. Correlating a series
        # that distinguishes nothing produces a number that reads like a verdict
        # and is not one, so this has to be checked on the task side too, not
        # only on the window's frontier below.
        worst = min(p["p"] for p in pairs)
        verdict, code = (f"NO CONCLUYENTE: ninguna diferencia de TAREA se separa "
                         f"de reetiquetar las semillas (p mas bajo = {worst:.3f} "
                         f"> 0,05 en los {len(pairs)} pares). Este eje no "
                         f"distingue nada en la metrica que manda, asi que el "
                         f"Spearman ordena ruido: no dice si el proxy vale"
                         ), "inconclusive"
    elif len(w["frontier"]) == len(w["trials"]):
        verdict, code = (f"NO CONCLUYENTE: los {len(w['trials'])} puntos empatan "
                         f"dentro de delta={w['delta']:.4f}: este recorrido no "
                         f"distingue nada, asi que el Spearman no significa nada. "
                         f"Repite con mas epocas antes de concluir"), "inconclusive"
    elif sp_agg >= SPEARMAN_PASS and task_winner_in_frontier:
        verdict, code = (f"OK: el proxy de ventana ordena igual que la tarea sobre "
                         f"{', '.join(axes)} [{domain}] (agregado {sp_agg:+.3f} >= "
                         f"{SPEARMAN_PASS} y el ganador por tarea cae en la frontera). "
                         f"No se cambia nada; anotalo con su fecha y su dataset"), "pass"
    else:
        why = []
        if sp_agg < SPEARMAN_PASS:
            why.append(f"agregado {sp_agg:+.3f} < {SPEARMAN_PASS}")
        if not task_winner_in_frontier:
            why.append("el ganador por tarea queda fuera de la frontera delta")
        verdict, code = (f"NO: el proxy NO ordena como la tarea sobre "
                         f"{', '.join(axes)} [{domain}] ({'; '.join(why)}). "
                         f"Lee el apartado 5.5 de metrica-de-tarea.md: cambiar el "
                         f"objetivo NO es una linea en OBJECTIVES"), "fail"
    print(f"\nVEREDICTO -> {verdict}")
    print(f"({time.time() - t0:.1f} s)")

    if args.json_out:
        payload = {"sweep": args.sweep, "split": args.split,
                   "objective": objective, "direction": direction,
                   "axes": axes, "domain": domain,
                   "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "spearman_per_run": sp_run, "spearman_aggregated": sp_agg,
                   "criterion": {"spearman_pass": SPEARMAN_PASS,
                                 "min_points": MIN_POINTS},
                   "verdict": code, "verdict_text": verdict,
                   "delta": w["delta"], "delta_source": w["delta_source"],
                   "frontier": frontier_pts,
                   "window_winner": w["best"]["point"],
                   "suggested": w["suggested"]["point"],
                   "task_winner": best_task["point"],
                   "task_winner_in_frontier": task_winner_in_frontier,
                   "task_winner_vs_others": pairs,
                   "skipped_runs": [r["run"] for r in skipped],
                   "per_run": per_run, "per_point": groups_by_window}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"detalle -> {args.json_out}")
    return 0 if code == "pass" else 2 if code == "inconclusive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
