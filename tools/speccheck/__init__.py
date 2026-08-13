"""Spec validator for docs/ui/ (see docs/ui/validador.md).

Not a test suite: a test says a behaviour is correct, this says a DECLARED RULE
holds across the system, and it is fed by the rule itself. The markdown owns the
spec (option A2); this package only reads it.

Deliberately imports nothing from `fv`: it reads files and (later) speaks HTTP,
like scripts/verify_ui.py. That keeps it able to run against a remote backend.
"""
