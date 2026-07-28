"""Bring up (and take down) what `--live` needs.

The rule this module exists to honour is U7.13: probar ejecutando, si; dejarlo
corriendo, no. Whatever it starts, it stops -- and `ports_free` then checks that
the promise was kept instead of assuming it.

It reuses a backend that is ALREADY listening (the user's, on :8010) instead of
fighting it for the port, and only starts its own when nobody answers.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def answers(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/runs", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def port_busy(port: int) -> bool:
    """Both families. Vite binds to ::1 only, so an IPv4-only probe reports the
    port FREE while the dev server is still listening -- a false green in exactly
    the rule about not leaving servers running (U7.13)."""
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family) as s:
                s.settimeout(0.5)
                if s.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            continue
    return False


def http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


@dataclass
class Live:
    base_url: str = ""
    front_url: str = ""
    started: list[subprocess.Popen] = field(default_factory=list)
    ports: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def ensure_backend(self, root: Path) -> str:
        # U7.10: se arranca uno PROPIO con el codigo de ahora. Reutilizar el que
        # ya escucha es como verificar contra un server stale -- da respuestas de
        # otra version y se tarda media tarde en verlo. Solo se reutiliza cuando
        # alguien lo pide a proposito con FV_API_BASE.
        wanted = os.environ.get("FV_API_BASE")
        if wanted and answers(wanted):
            self.base_url = wanted
            self.notes.append(f"backend ajeno reutilizado por FV_API_BASE: {wanted}")
            return self.base_url
        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, "-m", "fv.api", "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.started.append(proc)
        self.ports.append(port)
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            if answers(base, timeout=1.0):
                self.base_url = base
                self.notes.append(f"backend propio arrancado en {base} (se parara al terminar)")
                return base
            if proc.poll() is not None:
                self.notes.append("el backend propio murio al arrancar")
                return ""
            time.sleep(0.5)
        self.notes.append("el backend propio no llego a responder")
        return ""

    def ensure_front(self, root: Path, api_base: str) -> str:
        """Vite, pointed at OUR backend. Its port is fixed with strictPort and is
        the one in the CORS allowlist, so if it is busy we do not fight for it:
        the DOM substrate simply reports itself unavailable."""
        if port_busy(5173):
            self.notes.append("el puerto 5173 esta ocupado: el sustrato DOM no se puede levantar")
            return ""
        env = dict(os.environ, FV_API_URL=api_base)
        proc = subprocess.Popen(["npm.cmd" if os.name == "nt" else "npm", "run", "dev"],
                                cwd=root / "web", env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.started.append(proc)
        self.ports.append(5173)
        for _ in range(60):
            if http_ok("http://localhost:5173/"):
                self.front_url = "http://localhost:5173"
                self.notes.append(f"front propio arrancado en {self.front_url}")
                return self.front_url
            if proc.poll() is not None:
                self.notes.append("vite murio al arrancar")
                return ""
            time.sleep(0.5)
        self.notes.append("vite no llego a escuchar")
        return ""

    def shutdown(self) -> None:
        # children before parents (vite before its launcher), and on Windows the
        # tree needs taskkill: terminate() on npm leaves node listening.
        for proc in reversed(self.started):
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.started.clear()
