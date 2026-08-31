"""Is the process that owns a piece of durable state still alive?

The inherited trap (herencia.md §4): a job whose process died — a crash, an API
restart, this machine hibernating overnight — leaves its state file saying
"running" forever, because the cooperative-stop signal is only ever read by a
live runner. The cure is an explicit owner: whoever marks a sweep/run "running"
records its PID, and a reader can ask whether that PID is still there.

Cross-process by design: the reader (the API) and the owner (an API job thread
OR a separate `fv-sweep` CLI process) need not be the same process. This errs
SAFE — an owner that is genuinely alive is never reported dead, so a running
sweep is never wrongly reconciled; the only risk (a recycled PID reported alive)
leaves stale state untouched, exactly as today.
"""

from __future__ import annotations

import os
import signal
import sys


def morir_por_el_finally(aviso=None) -> None:
    """Hace que SIGTERM/SIGINT pasen por los `finally` de este proceso.

    POR QUE, y costo dinero el 2026-08-31
    -------------------------------------
    Todo lo que alquila una maquina la destruye en un `finally`
    (`entrenar_vast`, `estudio_flota`, `adoptar_vast`). Pero **el `finally` de
    Python cubre EXCEPCIONES, no senales**: SIGTERM termina el proceso sin
    desenrollar la pila, asi que el bloque que corta el gasto no llega a correr.

    Medido: se hizo `systemctl stop` del vigilante --que es justo lo que uno
    hace para terminar-- y la instancia de Vast siguio facturando. Hubo que
    destruirla a mano.

    `SystemExit` SI desenrolla, asi que convertir la senal en excepcion es lo
    que hace que el unico sitio donde se corta el gasto se ejecute siempre que
    el proceso acabe por su cuenta.

    ⚠ Contra SIGKILL no hay nada que hacer, y por eso NO sustituye a las otras
    dos redes: el comando de destruccion impreso al empezar, y el aviso de
    `cerrable.mjs` cuando hay una maquina viva sin vigilante.

    ⚠ Se llama lo PRIMERO de `main`, antes de alquilar o de tocar la API.
    Registrarlo tarde deja una ventana en la que una senal mata sin destruir --
    y esa ventana es justo el arranque, cuando la maquina ya existe pero el
    programa aun no ha llegado a su bucle.

    `aviso` es un `log` opcional, para que quede en el fichero que se mira.
    """
    def _handler(sig, _frame):
        if aviso:
            aviso(f"recibida senal {signal.Signals(sig).name}: "
                  f"salgo por el finally (se destruye lo alquilado)")
        raise SystemExit(143 if sig == signal.SIGTERM else 130)

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _handler)
        except ValueError:
            # no estamos en el hilo principal (p. ej. un job de la web app):
            # ahi no se pueden instalar handlers, y tampoco hace falta -- quien
            # recibe la senal es el proceso, no el hilo.
            pass


def pid_alive(pid: int | None) -> bool:
    """True if a process with this PID currently exists. None/≤0 → False."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False  # no such process (same-user query does not get denied)
        try:
            code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE  # an exited process is not alive
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True
