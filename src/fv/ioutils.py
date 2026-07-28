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

# ---------------------------------------------------------------- yaml configs
# The envelope of a configs/*.yaml (C and D): what describes the FILE, not the
# object. 'name' is the filename, 'format_version' is the format — neither is a
# field of the network or of the recipe. ONE definition, read by every path of
# every store: the leak this closes was list() handing format_version to a form
# that POSTed it straight back, and save() calling it an unknown field.
CONFIG_FORMAT_VERSION = 1
CONFIG_ENVELOPE_FIELDS = ("format_version", "name")


def strip_envelope(cfg: dict) -> dict:
    """The object alone: what the store validates and what a run freezes."""
    return {k: v for k, v in cfg.items() if k not in CONFIG_ENVELOPE_FIELDS}


def with_envelope(cfg: dict) -> dict:
    """The object as it goes to disk: the envelope is written here, not passed in."""
    return {"format_version": CONFIG_FORMAT_VERSION, **strip_envelope(cfg)}


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
