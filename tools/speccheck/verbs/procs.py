"""U7.13 -- the tool checks its own promise: what it started, it stopped.

`ports_free` runs last and looks at the real sockets. A validator that says
"I closed everything" without looking is the same kind of unverifiable promise
the holdout register (F14) exists to kill.
"""

from __future__ import annotations

from ..engine import NA, OK, VIOLATED, Check, Context, Outcome, verb
from ..live import port_busy


@verb("ports_free")
def ports_free(check: Check, ctx: Context) -> Outcome:
    if ctx.mode != "live":
        return Outcome(NA, "solo tiene sentido tras una corrida --live")
    ours = set(ctx.started_ports or [])
    busy = [p for p in (check.args.get("ports") or []) if port_busy(int(p))]
    mine = [p for p in busy if int(p) in ours]
    if mine:
        return Outcome(VIOLATED, f"la herramienta dejo escuchando: {mine}")
    if busy:
        return Outcome(OK, f"ocupados por otros (no los arranco esta herramienta): {busy}")
    return Outcome(OK, f"{check.args.get('ports')} libres")
