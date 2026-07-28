"""Validate that the UI specification in docs/ui/ is met (docs/ui/validador.md).

Not a test suite: it is fed by the rules themselves (the ```check blocks) and
reports every rule in one of four states. Exit code is non-zero only when a rule
is VIOLATED -- `no_verificable` is the map of what still needs a human, not a
failure.

Usage:
  .venv\\Scripts\\python scripts\\verify_spec.py                 (static)
  .venv\\Scripts\\python scripts\\verify_spec.py --live          (+ http/dom)
  .venv\\Scripts\\python scripts\\verify_spec.py --rule U3.1 -v
  .venv\\Scripts\\python scripts\\verify_spec.py --coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.speccheck import engine, extract, report  # noqa: E402
from tools.speccheck import verbs  # noqa: E402,F401  (registers the verbs)
from tools.speccheck.live import Live  # noqa: E402
from tools.speccheck.verbs.lint import preflight  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Valida la especificacion de docs/ui/")
    ap.add_argument("--live", action="store_true", help="incluye sustratos http/dom")
    ap.add_argument("--rule", action="append", help="una regla concreta (repetible)")
    ap.add_argument("--type", type=int, action="append", help="un tipo entero (repetible)")
    ap.add_argument("--coverage", action="store_true", help="solo el cuadro (sin el detalle)")
    ap.add_argument("--json", dest="json_out", help="vuelca el informe (recomputable: no se commitea)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    spec = extract.load(ROOT)
    print(f"\n  {len(spec.rules)} reglas en {len(spec.files)} ficheros de docs/ui/")

    problems = preflight(spec, ROOT)
    if problems:
        print(f"\n  LINT DE LA ESPECIFICACION: {len(problems)} problema(s).")
        print("  Un spec malformado hace que cualquier verde no signifique nada; no se sigue.\n")
        for p in problems:
            print(f"    - {report.ascii_(p)}")
        print()
        return 2

    only = list(args.rule or [])
    if args.type:
        only += [r.id for r in spec.rules.values() if r.type in args.type]

    mode = "live" if args.live else "static"
    live = Live()
    try:
        if mode == "live":
            live.ensure_backend(ROOT)
            for note in live.notes:
                print(f"  {report.ascii_(note)}")
        ctx = engine.Context(root=ROOT, mode=mode, spec=spec, base_url=live.base_url)
        results = engine.run(ctx, only or None)
    finally:
        # U7.13: lo que arranca esta herramienta, esta herramienta lo para.
        live.shutdown()

    print(f"  modo: {mode}" + ("  (los sustratos http/dom salen no_aplicable)" if mode == "static" else ""))
    print(report.render(results, verbose=args.verbose, summary_only=args.coverage))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            [{"rule": r.rule.id, "type": r.rule.type, "state": r.state,
              "strength": r.strength, "details": r.details} for r in results],
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  informe en {args.json_out} (recomputable: no se commitea)\n")

    return 1 if any(r.state == engine.VIOLATED for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
