"""Atomic JSON writes that survive Windows (formatos.md §4.2).

Windows will not replace a file another handle has open, and CPython opens for
reading without FILE_SHARE_DELETE. Measured in the sibling project: a reader
and a writer fighting for 4 s produced 5111 failed os.replace calls. Both
sides retry with a deadline; the POSIX pattern does not port.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_DEADLINE_S = 5.0


def write_json_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    deadline = time.monotonic() + _DEADLINE_S
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                try:
                    tmp.unlink(missing_ok=True)
                finally:
                    raise
            time.sleep(0.01)


def read_text_retrying(path: Path) -> str:
    path = Path(path)
    deadline = time.monotonic() + _DEADLINE_S
    while True:
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def read_json_retrying(path: Path) -> dict:
    deadline = time.monotonic() + _DEADLINE_S
    while True:
        text = read_text_retrying(path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # the writer may be mid-replace; a moment later the file is whole
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)
