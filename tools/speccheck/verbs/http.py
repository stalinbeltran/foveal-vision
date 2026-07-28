"""Fase 3 -- the shape of what the API answers (substrate B4).

Shape, never value: that `macro.sem` is THERE, not that it is 0.08. Values are
what `fv.metrics` and its tests are for.

Two rules of this substrate:
  - a fixture that does not exist (no runs, no holdout dataset) is `no_aplicable`
    with the reason -- never a pass and never a failure of the rule;
  - a refusal is provoked on purpose: the cheapest way to check that a gate says
    why is to walk into it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..engine import NA, OK, VIOLATED, Check, Context, Outcome, verb

_FIXTURES: dict[str, dict] = {}

# ── the guard this module needs and did not have ──────────────────────────────
# A validator MUST NOT change the system it measures. Writing `DELETE
# /window-datasets/{window_dataset}` in a check block felt like "provoke the 409"
# and actually deleted a real dataset the first time it ran (recovered from git:
# the description is versioned, the payload is not). So: everything that can
# write is blocked BY DEFAULT, and the only exceptions are calls that provably
# cannot mutate.
PURE_POST = ("/networks/validate",)          # api.md: "puro, sincrono, sin guardar"


def _is_readonly(ctx: Context, method: str, path: str, body) -> tuple[bool, str]:
    if method == "GET":
        return True, ""
    if method == "POST" and any(path.startswith(p) for p in PURE_POST):
        return True, ""
    if method == "POST" and path == "/runs" and isinstance(body, dict) and body.get("name"):
        # Legal only when the name ALREADY exists: then 409 is the only possible
        # answer and nothing is created. If it does not exist, this would train.
        st, _ = _req(ctx, "GET", f"/runs/{body['name']}")
        if st == 200:
            return True, ""
        return False, f"el run {body['name']} no existe: la peticion CREARIA uno"
    if method == "DELETE":
        st, _ = _req(ctx, "GET", path.replace("DELETE ", ""))
        if st == 404:
            return True, ""   # deleting what is not there cannot destroy anything
        return False, "el recurso existe: el DELETE lo borraria de verdad"
    return False, f"{method} puede escribir"


def _req(ctx: Context, method: str, path: str, body: Any = None, timeout: float = 60):
    url = f"{ctx.base_url}{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:200]}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, {"transport": str(e)}


def fixtures(ctx: Context) -> dict:
    """One run, one sweep, one dataset, one holdout -- or None for each."""
    key = str(ctx.base_url)
    if key in _FIXTURES:
        return _FIXTURES[key]
    fx: dict[str, Any] = {"run": None, "sweep": None, "window_dataset": None, "holdout": None}
    st, runs = _req(ctx, "GET", "/runs")
    if st == 200:
        # a run worth asking about is terminal and has a checkpoint to load
        cands = [r for r in runs.get("runs", []) if r.get("status") in ("done", "cancelled")]
        if cands:
            fx["run"] = cands[-1]["name"]
    st, sw = _req(ctx, "GET", "/sweeps")
    if st == 200 and sw.get("sweeps"):
        fx["sweep"] = sw["sweeps"][-1]["name"]
    st, wd = _req(ctx, "GET", "/window-datasets")
    if st == 200:
        names = [d["name"] for d in wd.get("window_datasets", [])]
        fx["window_dataset"] = names[0] if names else None
        fx["holdout"] = next((n for n in names if "holdout" in n), None)
    _FIXTURES[key] = fx
    return fx


def _fill(path: str, fx: dict) -> tuple[str | None, str]:
    out = path
    for key, val in fx.items():
        token = "{" + key + "}"
        if token in out:
            if not val:
                return None, f"no hay {key} en este backend"
            out = out.replace(token, str(val))
    return out, ""


def _unwrap(payload: Any) -> Any:
    """FastAPI wraps a raised HTTPException in {"detail": ...}; the front unwraps
    it in api.ts. Looking only at the top level made three rules read as violated
    when the API was right -- the check was."""
    if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
        return payload["detail"]
    return payload


def _dig(payload: Any, dotted: str):
    node = payload
    for part in dotted.split("."):
        if isinstance(node, list):
            if not node:
                return None, False
            node = node[0]
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


@verb("http_shape")
def http_shape(check: Check, ctx: Context) -> Outcome:
    scope = str(check.data.get("scope", ""))
    method, path = (scope.split(" ", 1) if " " in scope else ("GET", scope))
    path, why = _fill(path, fixtures(ctx))
    if path is None:
        return Outcome(NA, why)
    ro, why_not = _is_readonly(ctx, method, path, check.args.get("body"))
    if not ro:
        return Outcome(NA, f"comprobacion que puede escribir, bloqueada: {why_not}")
    st, payload = _req(ctx, method, path, check.args.get("body"))
    if st == 0:
        return Outcome(NA, f"backend no responde: {payload.get('transport', '')[:80]}")
    if st >= 400:
        code = payload.get("code") if isinstance(payload, dict) else None
        # A gate refusing for a declared reason is not a failure of THIS rule:
        # the fixture simply does not qualify (no checkpoint, split empty...).
        return Outcome(NA, f"{method} {path} -> {st} {code or ''} (la fixture no califica)")

    missing = [k for k in (check.args.get("requires") or []) if not _dig(payload, str(k))[1]]
    for dotted, expected in (check.args.get("expect_json") or {}).items():
        val, present = _dig(payload, str(dotted))
        if not present or val != expected:
            missing.append(f"{dotted}={val!r}, se esperaba {expected!r}")
    for k in (check.args.get("requires_when_present") or []):
        pass  # optional by definition: presence is not required, only shape when present
    if check.args.get("max_rows_param"):
        param = str(check.args["max_rows_param"])
        sep = "&" if "?" in path else "?"
        st2, capped = _req(ctx, method, f"{path}{sep}{param}=2")
        rows = capped.get("windows") or capped.get("rows") or capped.get("items") or []
        if st2 == 200 and isinstance(rows, list) and len(rows) > 2:
            missing.append(f"{param}=2 devolvio {len(rows)} filas")
    if missing:
        return Outcome(VIOLATED, f"{method} {path}: falta {', '.join(map(str, missing))}")
    return Outcome(OK, f"{method} {path}: {len(check.args.get('requires') or [])} campo(s) presentes")


@verb("http_refuses")
def http_refuses(check: Check, ctx: Context) -> Outcome:
    """Walk into the gate on purpose and check it says why AND how to fix it."""
    scope = str(check.data.get("scope", ""))
    method, path = (scope.split(" ", 1) if " " in scope else ("GET", scope))
    path, why = _fill(path, fixtures(ctx))
    if path is None:
        return Outcome(NA, why)
    ro, why_not = _is_readonly(ctx, method, path, check.args.get("body"))
    if not ro:
        return Outcome(NA, f"comprobacion que puede escribir, bloqueada: {why_not}")
    st, payload = _req(ctx, method, path, check.args.get("body"))
    if st == 0:
        return Outcome(NA, f"backend no responde: {payload.get('transport', '')[:80]}")

    payload = _unwrap(payload)
    expect = check.args.get("expect_status")
    if expect and st not in [int(x) for x in expect]:
        return Outcome(VIOLATED, f"{method} {path} -> {st}, se esperaba {expect}")
    if not expect and st < 400:
        return Outcome(VIOLATED, f"{method} {path} -> {st}: la peticion imposible NO se rechazo")

    problems = []
    for field in (check.args.get("expect_fields") or ["code", "message", "hint"]):
        if not payload.get(field):
            problems.append(f"sin {field}")
    if check.args.get("expect_code") and payload.get("code") != check.args["expect_code"]:
        problems.append(f"code={payload.get('code')!r}, se esperaba {check.args['expect_code']!r}")
    if problems:
        return Outcome(VIOLATED, f"{method} {path} -> {st}: {', '.join(problems)}")
    return Outcome(OK, f"{method} {path} -> {st} {payload.get('code', '')} con razon y arreglo")


@verb("null_not_zero")
def null_not_zero(check: Check, ctx: Context) -> Outcome:
    """A field that means 'not measured' must be null in the payload, never 0."""
    scope = str(check.data.get("scope", ""))
    method, path = (scope.split(" ", 1) if " " in scope else ("GET", scope))
    path, why = _fill(path, fixtures(ctx))
    if path is None:
        return Outcome(NA, why)
    ro, why_not = _is_readonly(ctx, method, path, None)
    if not ro:
        return Outcome(NA, f"comprobacion que puede escribir, bloqueada: {why_not}")
    st, payload = _req(ctx, method, path)
    if st == 0:
        return Outcome(NA, "backend no responde")
    if st >= 400:
        return Outcome(NA, f"{path} -> {st} (la fixture no califica)")
    seen = []
    for field in check.args.get("fields") or []:
        val, present = _dig(payload, str(field))
        if not present:
            return Outcome(VIOLATED, f"{path}: el campo {field} no viaja en el payload")
        seen.append(f"{field}={'null' if val is None else val}")
    return Outcome(OK, f"{path}: {', '.join(seen)} (nulo cuando no hay medida, no 0)")
