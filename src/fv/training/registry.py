"""E — the run store. Explicit state, no silent overwrite, provenance complete.

The three measured traps this makes impossible (herencia.md §4): state deduced
from which files exist (a crash stays 'running' forever) -> status.json;
silent overwrite (mkdir exist_ok + truncate) -> create() refuses; no
environment capture -> provenance carries python/torch/platform/device.
"""

from __future__ import annotations

import json
import subprocess
import sys
import socket
import time
from pathlib import Path

from fv import artefactos, settings
from fv.ioutils import read_json_retrying, read_text_retrying, write_json_atomic
from fv.proc import pid_alive


class RunError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
        return "unknown: not a git repository"
    except Exception as e:  # the reason, never a silent null (formatos.md §2)
        return f"unknown: {e}"


def environment(device: str) -> dict:
    try:
        import torch
        torch_v = torch.__version__
    except ImportError:
        torch_v = "not installed"
    return {"python": sys.version.split()[0], "torch": torch_v,
            "platform": sys.platform, "device": device}


class RunStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else settings.runs_root()

    def path(self, name: str) -> Path:
        """Donde ESTA el run: plano -> archivo fechado -> legado (fv.artefactos).
        Si no esta en ningun sitio devuelve la forma plana.

        La forma PLANA, no `destino()`: ver `SweepStore.path`. Y `destino()`
        tampoco serviria aqui, porque sin el `config` no sabe de que recorrido
        es el run."""
        return artefactos.resolver("runs", name, self.root / name)

    def destino(self, name: str, config: dict | None = None) -> Path:
        """Donde se ESCRIBE.

        Un run de un recorrido YA archivado se crea DENTRO de el
        (`<mes>/sweeps/<rec>/runs/<run>`): asi la relacion recorrido-runs es
        estructura de directorios y no un prefijo en el nombre, y el run hereda
        el mes de su recorrido -- y con el, el de su estudio.

        `path()` solo recibe el nombre y por eso no puede decidir esto; `create`
        si, porque el config trae `provenance.sweep`.

        Un run SUELTO (un benchmark, sin `provenance.sweep`) no se inventa un
        recorrido, pero si va bajo el mes de hoy. La forma plana queda solo para
        el run cuyo recorrido esta plano: ahi el mes lo separaria de su
        recorrido, y eso pesa mas que la fecha.
        """
        sweep = ((config or {}).get("provenance") or {}).get("sweep")
        d = artefactos.destino_agrupado("runs", name, recorrido=sweep)
        return d if d is not None else self.root / name

    def exists(self, name: str) -> bool:
        return (self.path(name) / "config.json").exists()

    def create(self, name: str, config: dict) -> Path:
        # se crea en el DESTINO, no en `path()`: si el run ya estuviera archivado
        # `path()` devolveria el archivo y esto escribiria dentro de el.
        d = self.destino(name, config)
        if d.exists():
            raise RunError("run_exists",
                           f"ya existe un run llamado '{name}'",
                           "elige otro nombre, o borra ese run primero: no se "
                           "sobrescribe nunca")
        d.mkdir(parents=True)
        write_json_atomic(d / "config.json", config)
        self.set_status(name, "queued")
        return d

    def set_status(self, name: str, status: str, **extra) -> None:
        payload = {"status": status, "updated_at": time.time()}
        payload.update(extra)
        # Un `pid` solo significa algo EN SU MAQUINA. Desde que un run se entrena
        # en otra y su `status.json` viaja hasta aqui (entrenar_vast.py), el
        # numero se lee contra la tabla de procesos equivocada -- ver `reconcile`.
        if payload.get("pid") is not None:
            payload.setdefault("host", socket.gethostname())
        write_json_atomic(self.path(name) / "status.json", payload)

    def status(self, name: str) -> dict:
        p = self.path(name) / "status.json"
        if not p.exists():
            return {"status": "unknown"}
        return read_json_retrying(p)

    def reconcile(self, name: str) -> dict:
        """Heal a stale 'running': if the training process that wrote it is gone
        (crash / API restart / hibernation), the run would read 'running'
        forever (the inherited trap this store exists to kill). Mark it
        'interrupted' — the sweep runner redoes any non-(done|cancelled) point
        on resume. Errs safe: a live or unknown owner is left be.

        ⚠ Y un pid de OTRA maquina es un dueño desconocido, no uno muerto. Desde
        que `entrenar_vast.py` entrena en Vast y se trae el `status.json`, el
        `pid` que llega es el de la maquina alquilada: aqui casi nunca existe, y
        sin esta comprobacion `reconcile` declaraba "interrupted" un run que
        estaba entrenando AHORA MISMO en otro sitio -- con lo que el guard de
        `reanudar` dejaba pasar un SEGUNDO entrenamiento sobre el mismo run.
        Comprobado el 2026-08-30 con `fov-optimo-p20`: `pid 822`, inexistente
        aqui, run vivo alli.

        Un `status.json` sin `host` es de antes de esto y se trata como siempre:
        no se puede saber de quien era, y romper el saneado de lo ya escrito
        seria peor que el caso raro que arregla."""
        st = self.status(name)
        if st.get("status") != "running":
            return st
        host = st.get("host")
        if host and host != socket.gethostname():
            return st                      # dueño en otra maquina: no se juzga
        pid = st.get("pid")
        if pid is None or pid_alive(pid):
            return st
        self.set_status(name, "interrupted", epoch=st.get("epoch", 0),
                        reason="el proceso que lo entrenaba ya no existe "
                               "(caida/reinicio/hibernacion)")
        return self.status(name)

    def config(self, name: str) -> dict:
        p = self.path(name) / "config.json"
        if not p.exists():
            raise RunError("run_not_found", f"no existe el run '{name}'",
                           "mira la lista en /runs")
        return read_json_retrying(p)

    def list(self) -> list[dict]:
        out = []
        # los tres sitios, sin repetir: lo nuevo (plano o agrupado), lo archivado
        # y lo legado. Mirar solo `self.root` dejaba fuera los 851 runs migrados.
        # newest first: the run you just trained is the one you want to look at,
        # and it keeps the default selection on a current (loadable) checkpoint
        # instead of the alphabetically-first, possibly-stale one.
        dirs = sorted((self.path(n) for n in artefactos.nombres("runs", self.root)),
                      key=lambda d: d.stat().st_mtime if d.exists() else 0,
                      reverse=True)
        for d in dirs:
            if not (d / "config.json").exists():
                continue
            cfg = read_json_retrying(d / "config.json")
            st = self.status(d.name)
            summary = {}
            sp = d / "summary.json"
            if sp.exists():
                summary = read_json_retrying(sp)
            prov = cfg.get("provenance", {})
            out.append({
                "name": d.name,
                "status": st.get("status"),
                "epoch": st.get("epoch"),
                "window_dataset": prov.get("window_dataset", {}).get("name"),
                "network": prov.get("network", {}).get("name"),
                "recipe": prov.get("recipe", {}).get("name"),
                "sweep": prov.get("sweep"),
                "best": summary.get("best"),
                "monitor": summary.get("monitor"),
                "epochs_run": summary.get("epochs_run"),
                "seconds_per_epoch": summary.get("seconds_per_epoch"),
                # ¿hay un best.pt AQUI? Es un hecho del disco, no una decision
                # de inferencia: si esta red esta aprobada lo dice `fv.inference`,
                # que este dominio no puede importar (contrato ⑦) y no debe.
                #
                # Se calcula sobre `d`, que ya esta resuelto, y por eso cuesta
                # 23 ms sobre los 862 runs de hoy; volviendo a resolver la ruta
                # (`self.path(name)`) costaria 799 ms -- 35x -- sobre una ruta
                # que la pantalla de Predecir sondea cada 3 s. Medido 2026-08-31.
                "has_checkpoint": (d / "best.pt").exists(),
            })
        return out

    def metrics_since(self, name: str, since: int = 0) -> dict:
        p = self.path(name) / "metrics.jsonl"
        records = []
        if p.exists():
            lines = read_text_retrying(p).splitlines()
            for line in lines[since:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a live run: the last line may be mid-write — normal
            return {"records": records, "next": since + len(records)}
        return {"records": [], "next": since}

    def request_stop(self, name: str, reason: str = "user") -> None:
        if not self.exists(name):
            raise RunError("run_not_found", f"no existe el run '{name}'", "")
        write_json_atomic(self.path(name) / "stop.json",
                          {"requested_at": time.time(), "reason": reason})

    def stop_requested(self, name: str) -> bool:
        return (self.path(name) / "stop.json").exists()

    def used_by_dataset(self, dataset_name: str) -> list[str]:
        out = []
        for r in self.list():
            if r.get("window_dataset") == dataset_name:
                out.append(r["name"])
        return out

    def used_by_sweep(self, sweep_name: str) -> list[str]:
        out = []
        for r in self.list():
            if r.get("sweep") == sweep_name:
                out.append(r["name"])
        return out

    def rename(self, name: str, new_name: str) -> None:
        if self.status(name).get("status") == "running":
            raise RunError("run_is_running", f"'{name}' esta corriendo",
                           "para el run antes de renombrarlo")
        if self.exists(new_name):
            raise RunError("run_exists", f"ya existe '{new_name}'", "elige otro nombre")
        if not self.exists(name):
            raise RunError("run_not_found", f"no existe el run '{name}'", "")
        self.path(name).rename(self.path(new_name))

    def delete(self, name: str) -> None:
        if not self.exists(name):
            raise RunError("run_not_found", f"no existe el run '{name}'", "nada que borrar")
        if self.status(name).get("status") == "running":
            raise RunError("run_is_running", f"'{name}' esta corriendo",
                           "para el run antes de borrarlo")
        d = self.path(name)
        for f in sorted(d.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        d.rmdir()
