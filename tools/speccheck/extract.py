"""Pull the specification out of docs/ui/*.md (A2: the markdown rules).

A rule is a bolded lead of the form `**U3.1 - ...**`; its machine-checkable
projection is the fenced ```check U3.1 block that follows it. Nothing here
interprets a check -- that is the engine's job. A YAML block that does not parse
is a problem, never a silently skipped rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# The em dash is what the documents use; a plain hyphen is accepted so a rule
# written on a cp1252 keyboard still parses.
RULE_RE = re.compile(r"^\*\*(U(\d+)\.(\d+))\s+[—-]\s+(.+?)\*\*", re.M | re.S)
BLOCK_RE = re.compile(r"^```check[ \t]+(U\d+\.\d+)[ \t]*\r?\n(.*?)^```", re.M | re.S)
FILE_RE = re.compile(r"^(\d+)-.+\.md$")


@dataclass
class Rule:
    id: str
    type: int
    num: int
    title: str
    path: Path
    line: int


@dataclass
class Check:
    rule_id: str
    data: dict[str, Any]
    path: Path
    line: int

    @property
    def substrate(self) -> str:
        return str(self.data.get("substrate", ""))

    @property
    def kind(self) -> str | None:
        k = self.data.get("kind")
        return str(k) if k else None

    @property
    def strength(self) -> str:
        return str(self.data.get("strength", "strong"))

    @property
    def args(self) -> dict[str, Any]:
        return self.data.get("args") or {}

    @property
    def where(self) -> str:
        return f"{self.path.as_posix()}:{self.line}"


@dataclass
class Spec:
    rules: dict[str, Rule] = field(default_factory=dict)
    checks: dict[str, list[Check]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)

    def by_type(self, t: int) -> list[Rule]:
        return [r for r in self.rules.values() if r.type == t]


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_file(path: Path, root: Path, spec: Spec) -> None:
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(root)
    m = FILE_RE.match(path.name)
    file_type = int(m.group(1)) if m else 0

    for r in RULE_RE.finditer(text):
        rid, rtype, rnum = r.group(1), int(r.group(2)), int(r.group(3))
        if rid in spec.rules:
            spec.problems.append(
                f"{rid}: definida dos veces ({spec.rules[rid].path.as_posix()} y {rel.as_posix()})")
            continue
        if rtype != file_type:
            spec.problems.append(f"{rid}: vive en {rel.as_posix()}, que es del tipo {file_type}")
        spec.rules[rid] = Rule(rid, rtype, rnum, _norm(r.group(4)), rel, _line_of(text, r.start()))

    for b in BLOCK_RE.finditer(text):
        rid, body = b.group(1), b.group(2)
        line = _line_of(text, b.start())
        try:
            data = yaml.safe_load(body) or {}
        except yaml.YAMLError as exc:
            spec.problems.append(f"{rid}: bloque check ilegible en {rel.as_posix()}:{line} ({exc})")
            continue
        if not isinstance(data, dict):
            spec.problems.append(f"{rid}: el bloque check de {rel.as_posix()}:{line} no es un mapa")
            continue
        spec.checks.setdefault(rid, []).append(Check(rid, data, rel, line))


def load(root: Path, docs_dir: Path | None = None) -> Spec:
    docs = docs_dir or (root / "docs" / "ui")
    spec = Spec()
    for path in sorted(docs.glob("*.md"), key=lambda p: (FILE_RE.match(p.name) is None, p.name)):
        if not FILE_RE.match(path.name):
            continue  # validador.md and friends describe the tool, they are not the spec
        spec.files.append(path.relative_to(root))
        parse_file(path, root, spec)
    return spec
