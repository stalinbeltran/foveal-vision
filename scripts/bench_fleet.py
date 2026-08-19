#!/usr/bin/env python3
"""Mide s/epoca en droplets de distinta capacidad de vCPU, y los destruye.

La idea
-------
La maquina desde la que se corre esto NO se mide a si misma: crea una maquina
por tamano, le pone el dato y el benchmark, recoge el numero y la destruye. Ella
sigue viva; las medidas son desechables. Asi se compara hardware sin tener que
mudarse a cada maquina, y sin dejar nada encendido despues.

Por que droplets de CPU dedicada (gama c-) y no los s- baratos: en la gama
compartida el tiempo de CPU se reparte con los vecinos, asi que dos corridas
del mismo benchmark no son comparables entre si -- que es justo lo unico que
este benchmark tiene que garantizar. Se puede pedir cualquier plan con --sizes.

    python3 scripts/bench_fleet.py --vcpus 2,4,8
    python3 scripts/bench_fleet.py --sizes c-2,c-4,s-2vcpu-4gb --repeats 5
    python3 scripts/bench_fleet.py --reap        # destruye lo que quedara vivo

Tarda decenas de minutos, asi que NO se lanza dentro de un turno de chat: se
desacopla y avisa al terminar (CLAUDE.md del coordinador, "un mensaje es un
proceso que muere"):

    setsid sh -c 'cd ~/src/foveal-vision && python3 scripts/bench_fleet.py \
        --vcpus 2,4,8 > /tmp/fleet.log 2>&1; \
        node ~/src/telegram-coordinator/scripts/notify.mjs \
        "flota terminada: benchmarks/ y /tmp/fleet.log"' &

Lo que quede a medias se barre siempre con --reap: el tag es la red de
seguridad, porque un droplet olvidado se paga por segundo.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANZADOR = Path.home() / "src" / "digital-ocean-dropplet-auto-launching"
DO = LANZADOR / "scripts" / "do_droplet.py"

# Todos los droplets de medicion nacen con este tag y ninguna otra cosa lo usa:
# es lo que hace que `--reap` pueda barrerlos sin pensar y sin tocar nada mas.
TAG = "bench-efimero"

VOLUMEN = os.environ.get("BENCH_VOLUME", "/mnt/bench-data")
REPO = "https://github.com/stalinbeltran/foveal-vision.git"

# Torch desde el indice de CPU a proposito: el paquete por defecto de PyPI
# arrastra las bibliotecas CUDA (unos 2,5 GB) que en una maquina sin GPU no se
# usan para nada, y descargarlas es varios minutos por droplet.
TORCH = "torch --index-url https://download.pytorch.org/whl/cpu"


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def lanzador(*args: str, timeout: int = 1800, check: bool = True) -> str:
    """Llama al lanzador. Es quien habla con la API; aqui no se duplica eso."""
    cmd = [sys.executable, str(DO), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    salida = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        die(f"falló `do_droplet.py {' '.join(args)}` (código {proc.returncode}):\n{salida}")
    return salida


def ssh_base(ip: str, puerto: int) -> list[str]:
    return [
        "ssh", "-p", str(puerto),
        "-i", str(Path.home() / ".ssh" / "do_droplet"),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
        f"root@{ip}",
    ]


def remoto(ip: str, puerto: int, script: str, timeout: int = 2400) -> tuple[int, str]:
    """Ejecuta un script por stdin (nunca como argumento: eso sale en el `ps`)."""
    proc = subprocess.run(
        ssh_base(ip, puerto) + ["bash -s"],
        input=script.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout + proc.stderr).decode("utf-8", errors="replace")


def ip_de(nombre: str) -> str:
    return lanzador("ip", nombre).strip().splitlines()[-1].strip()


def esperar_ssh(ip: str, timeout: int = 600) -> int:
    limite = time.time() + timeout
    while time.time() < limite:
        for puerto in (22, 443):
            code, _ = remoto(ip, puerto, "true", timeout=30)
            if code == 0:
                return puerto
        time.sleep(10)
    die(f"{ip} no acepta SSH tras {timeout}s")


# --------------------------------------------------------------- una medicion


def preparar_script() -> str:
    """Deja el droplet listo para entrenar: dependencias, repo y venv.

    Se instala lo minimo y desde el indice de CPU. Nada de Claude Code ni gh:
    esta maquina existe para entrenar tres epocas y morirse.

    El `cloud-init status --wait` de la primera linea no es cortesia: el primer
    arranque esta haciendo package_upgrade e instalando medio entorno, y eso
    ocupa los nucleos. Medir encima de eso da un numero inflado con la misma
    pinta que uno bueno -- el propio bench_speed.py avisa si la carga esta alta,
    pero lo que hay que hacer es no empezar hasta que la maquina este quieta.
    """
    return f"""
set -eu
export DEBIAN_FRONTEND=noninteractive
cloud-init status --wait >/dev/null 2>&1 || true
if ! command -v git >/dev/null || ! python3 -m venv --help >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git python3-venv python3-pip build-essential >/dev/null
fi
[ -d /root/foveal-vision/.git ] || git clone -q {REPO} /root/foveal-vision
cd /root/foveal-vision
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q numpy pillow pyyaml
.venv/bin/python -m pip install -q {TORCH}
.venv/bin/python -m pip install -q -e . --no-deps
mkdir -p data
echo "PREPARADO $(.venv/bin/python -c 'import torch;print(torch.__version__)')"
"""


def medir_uno(size: str, repeats: int, sufijo: str, mantener: bool) -> dict:
    nombre = f"bench-{size}-{sufijo}"
    resultado = {"size": size, "droplet": nombre, "ok": False, "error": ""}
    creado = False
    try:
        log(f"{size}: creando droplet '{nombre}'…")
        lanzador(
            "launch", nombre, "--size", size, "--tag", TAG, "--no-provision", timeout=1800
        )
        creado = True
        ip = ip_de(nombre)
        puerto = esperar_ssh(ip)
        log(f"{size}: {ip}:{puerto} listo, instalando…")

        code, salida = remoto(ip, puerto, preparar_script(), timeout=2400)
        if code != 0:
            raise RuntimeError(f"la preparación falló:\n{salida[-2000:]}")

        # El dato se COPIA al disco local del droplet. No se monta el volumen ni
        # se lee por red: el benchmark tiene que medir la máquina, y leer el
        # dataset de un disco remoto mediría la red.
        log(f"{size}: copiando el dataset…")
        destino = "/root/foveal-vision/data"
        for rel in ("FINGERPRINT.json", "window-datasets"):
            origen = Path(VOLUMEN) / rel
            if not origen.exists():
                raise RuntimeError(f"no está {origen}: ¿el volumen no está montado aquí?")
            scp = [
                "scp", "-P", str(puerto), "-r", "-q",
                "-i", str(Path.home() / ".ssh" / "do_droplet"),
                "-o", "StrictHostKeyChecking=accept-new",
                str(origen), f"root@{ip}:{destino}/",
            ]
            if subprocess.run(scp, timeout=1800).returncode != 0:
                raise RuntimeError(f"no pude copiar {rel} a {nombre}")

        # Que lo copiado sea lo que dice la huella. Sin esto, un dataset a
        # medias daría un número más rápido y con la misma pinta.
        code, salida = remoto(
            ip, puerto,
            "cd /root/foveal-vision && .venv/bin/python scripts/bench_dataset.py "
            "verify --from data",
            timeout=300,
        )
        if code != 0:
            raise RuntimeError(f"el dataset copiado no coincide con su huella:\n{salida[-1500:]}")
        log(f"{size}: dataset verificado, midiendo ({repeats} repeticiones)…")

        code, salida = remoto(
            ip, puerto,
            "cd /root/foveal-vision && .venv/bin/python scripts/bench_speed.py "
            f"--repeats {repeats} --out /root/reporte.json",
            timeout=7200,
        )
        if code != 0:
            raise RuntimeError(f"el benchmark falló:\n{salida[-2000:]}")

        code, reporte = remoto(ip, puerto, "cat /root/reporte.json", timeout=120)
        if code != 0:
            raise RuntimeError("el benchmark terminó pero no pude leer el reporte")
        datos = json.loads(reporte)
        resultado.update(ok=True, reporte=datos,
                         s_por_epoca=datos["seconds_per_epoch"]["mean"])
        log(f"{size}: {resultado['s_por_epoca']:.2f} s/época")

        salida_local = ROOT / "benchmarks" / f"vcpu_{size}_{sufijo}.json"
        datos["fleet"] = {"size": size, "droplet": nombre, "lanzado_desde": os.uname().nodename}
        salida_local.write_text(json.dumps(datos, indent=2) + "\n", encoding="utf-8")
        resultado["fichero"] = str(salida_local)

    except Exception as exc:  # noqa: BLE001 -- un tamaño que falla no para la flota
        resultado["error"] = f"{type(exc).__name__}: {exc}"
        log(f"{size}: FALLÓ -- {resultado['error'].splitlines()[0]}")
    finally:
        # Pase lo que pase. Un droplet olvidado se paga por segundo, y el fallo
        # que deja máquinas vivas es el caro de verdad.
        if creado and not mantener:
            log(f"{size}: destruyendo '{nombre}'…")
            try:
                lanzador("destroy", nombre, "--yes", timeout=600, check=False)
            except Exception as exc:  # noqa: BLE001
                log(f"{size}: NO PUDE DESTRUIR '{nombre}' ({exc}). "
                    f"Hazlo a mano: do_droplet.py destroy {nombre} --yes")
        elif creado:
            log(f"{size}: '{nombre}' se queda vivo (--keep). Facturando.")
    return resultado


# ---------------------------------------------------------------------- flota


def comprobaciones() -> None:
    if not DO.exists():
        die(f"No está el lanzador en {DO}.\n"
            "  node ~/src/telegram-coordinator/scripts/bench-preflight.mjs --fix")
    if not (os.environ.get("DO_TOKEN") or os.environ.get("DIGITALOCEAN_TOKEN")):
        die("No hay DO_TOKEN en el entorno: esta máquina no puede crear droplets.\n"
            "  node ~/src/telegram-coordinator/scripts/bench-preflight.mjs")
    if not (Path.home() / ".ssh" / "do_droplet").exists():
        die("No hay ~/.ssh/do_droplet: podrías crear droplets y no entrar en ellos.\n"
            "  node ~/src/telegram-coordinator/scripts/bench-preflight.mjs --fix")
    if not (Path(VOLUMEN) / "FINGERPRINT.json").exists():
        die(f"No está el dataset en {VOLUMEN}.\n"
            "  python3 scripts/bench_dataset.py build && "
            f"python3 scripts/bench_dataset.py publish --to {VOLUMEN}")
    if not shutil.which("scp"):
        die("Falta scp, que es como viaja el dataset a los droplets.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vcpus", help="capacidades a medir con CPU dedicada: 2,4,8 -> c-2,c-4,c-8")
    ap.add_argument("--sizes", help="planes explícitos, separados por coma")
    ap.add_argument("--repeats", type=int, default=3, help="repeticiones por máquina")
    ap.add_argument("--parallel", type=int, default=4,
                    help="cuántas máquinas a la vez (son independientes: no se estorban)")
    ap.add_argument("--keep", action="store_true",
                    help="no destruir al terminar. Se quedan facturando: sólo para depurar")
    ap.add_argument("--reap", action="store_true",
                    help=f"destruir todo lo que lleve el tag {TAG} y salir")
    args = ap.parse_args()

    if args.reap:
        log(f"Barriendo todo lo que lleve el tag '{TAG}'…")
        print(lanzador("destroy", "--tag", TAG, "--yes", check=False))
        return 0

    if args.sizes:
        sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    elif args.vcpus:
        sizes = [f"c-{n.strip()}" for n in args.vcpus.split(",") if n.strip()]
    else:
        sizes = ["c-2", "c-4", "c-8"]

    comprobaciones()
    huella = json.loads((Path(VOLUMEN) / "FINGERPRINT.json").read_text(encoding="utf-8"))
    sufijo = time.strftime("%Y%m%d-%H%M%S")

    log(f"Flota: {len(sizes)} máquinas ({', '.join(sizes)}), {args.repeats} repeticiones cada una")
    log(f"Dataset: {huella['window_dataset']} · huella {huella['sha256_windows_npz'][:16]}…")
    log(f"Si esto se corta, barre con:  python3 {Path(__file__).name} --reap")

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futuros = {
            pool.submit(medir_uno, s, args.repeats, sufijo, args.keep): s for s in sizes
        }
        resultados = [f.result() for f in concurrent.futures.as_completed(futuros)]

    resultados.sort(key=lambda r: r["size"])
    log(f"\nTerminado en {(time.monotonic() - t0) / 60:.1f} min\n")

    print(f"{'plan':<16} {'vCPU':>5} {'s/época':>10} {'±':>7}  reporte")
    print("-" * 72)
    for r in resultados:
        if r["ok"]:
            hw = r["reporte"]["hardware"]
            sd = r["reporte"]["seconds_per_epoch"]["std"]
            print(f"{r['size']:<16} {hw['cpu_count_logical']:>5} "
                  f"{r['s_por_epoca']:>10.2f} {sd:>7.2f}  {Path(r['fichero']).name}")
        else:
            print(f"{r['size']:<16} {'-':>5} {'FALLÓ':>10} {'':>7}  {r['error'].splitlines()[0][:40]}")

    resumen = ROOT / "benchmarks" / f"vcpu-fleet_{sufijo}.json"
    resumen.write_text(
        json.dumps(
            {
                "sufijo": sufijo,
                "dataset": huella,
                "repeats": args.repeats,
                "resultados": [
                    {k: v for k, v in r.items() if k != "reporte"} for r in resultados
                ],
            },
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    log(f"\nResumen: {resumen}")
    log("Commitea los reportes: lo que no está empujado no existe.")

    vivos = lanzador("list", "--tag", TAG, check=False)
    if TAG in vivos and "No hay" not in vivos:
        log(f"\nAVISO: todavía hay máquinas con el tag {TAG}:\n{vivos}")
    return 0 if all(r["ok"] for r in resultados) else 1


if __name__ == "__main__":
    raise SystemExit(main())
