"""Where an artifact of E/H/I lives inside the data repository.

The data repo groups artifacts by month, `<year>/<NN>-<month>/{runs,sweeps,studies}/<name>`,
so that a directory listing stays readable. The month is **not** a timeline of
when each run executed: it is chosen once, when the *study* is created, and
everything belonging to that study inherits it. A sweep launched past midnight
stays with its study rather than starting a second folder for the same work.

This module is the single definition of that layout. The three stores
(RunStore, SweepStore, StudyStore) resolve through it, so a name maps to one
path no matter which month it was filed under.
"""

from __future__ import annotations

from pathlib import Path

from fv import settings

KINDS = ("runs", "sweeps", "studies")

# The file that makes a directory *be* an artifact of each kind, rather than an
# incidental folder that happens to sit there.
MARKER = {"runs": "config.json", "sweeps": "spec.json", "studies": "plan.json"}


def kind_root(kind: str) -> Path:
    """`<data-repo>` — the base every month directory hangs from."""
    if kind not in KINDS:
        raise ValueError(f"kind desconocido: {kind!r} (esperaba {KINDS})")
    return settings.data_root()


def month_dirs(kind: str) -> list[Path]:
    """Every `<year>/<month>/<kind>` directory that exists, newest month last.

    Sorted by name, which for `2026/08-agosto` sorts chronologically because the
    month carries its number.
    """
    root = kind_root(kind)
    if not root.exists():
        return []
    out = []
    for year in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for month in sorted(p for p in year.iterdir() if p.is_dir()):
            d = month / kind
            if d.is_dir():
                out.append(d)
    return out


def _all_dirs(kind: str) -> list[Path]:
    """Every candidate directory of `kind`, marker or not.

    A run that belongs to a sweep lives *inside* that sweep
    (`sweeps/<sweep>/runs/<run>`), so runs are collected from both places: the
    sweep-owned ones and the loose ones directly under `<month>/runs/`.
    """
    out: list[Path] = []
    for base in month_dirs(kind):
        out += [d for d in sorted(base.iterdir()) if d.is_dir()]
    if kind == "runs":
        for base in month_dirs("sweeps"):
            for sweep in sorted(base.iterdir()):
                runs = sweep / "runs"
                if runs.is_dir():
                    out += [d for d in sorted(runs.iterdir()) if d.is_dir()]
    return out


def iter_dirs(kind: str) -> list[Path]:
    """Every artifact directory of `kind` that carries its marker file."""
    marker = MARKER[kind]
    return [d for d in _all_dirs(kind) if (d / marker).exists()]


def find(kind: str, name: str) -> Path | None:
    """The directory of `name`, or None if no month holds it.

    Matches on the DIRECTORY, not on the marker file: a caller mid-delete (the
    sweep runner redoing a point removes status.json and config.json before
    removing the directory) must keep resolving to the same place it created,
    or it would delete one path and rebuild at another.
    """
    for d in _all_dirs(kind):
        if d.name == name:
            return d
    return None


def study_month(name: str) -> str | None:
    """The `<year>/<month>` a study was filed under, or None if it has none yet."""
    d = find("studies", name)
    if d is None:
        return None
    # .../<year>/<month>/studies/<name>
    return f"{d.parents[2].name}/{d.parents[1].name}"


def owning_study(kind: str, name: str, studies: list[str]) -> str | None:
    """Which study a sweep (or a run) belongs to, by the naming convention.

    A study's sweeps are named `<study>-s<i>-<axis>`, and their runs extend that
    prefix. Longest match wins, so `plana-confirm` is preferred over `plana`.
    """
    for s in sorted(studies, key=len, reverse=True):
        if name.startswith(s + "-"):
            return s
    return None


def new_path(kind: str, name: str, month: str | None = None) -> Path:
    """Where to CREATE `name`.

    `month` is the `<year>/<month>` to file it under; when omitted the current
    month is used. Callers that know the owning study pass that study's month,
    which is what keeps one study inside one folder.
    """
    m = month or settings.month_dir()
    return kind_root(kind) / m / kind / name


def resolve(kind: str, name: str, month: str | None = None) -> Path:
    """The path of `name`: where it already is, or where it would be created."""
    found = find(kind, name)
    return found if found is not None else new_path(kind, name, month)
