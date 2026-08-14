"""fv-bench: mide cuanto tarda ESTA maquina en entrenar, para comparar hardware
entre maquinas (o antes/despues de mover el proceso a otra maquina/droplet).

No es un estudio: no busca hiperparametros optimos, corre una red y una
receta CONGELADAS (configs/networks/bench-16.yaml, configs/recipes/bench.yaml
-- no tocar por razones de investigacion) sobre un dataset CONGELADO, y mide
s/epoca -- la misma metrica de costo que ya usan los estudios
(seconds_per_epoch, fv.training.loop).

El dataset es el dato REAL del proyecto: ventanas de local/dirty-1000-80px,
las 1000 imagenes del generador hermano reducidas a 80x60. Se midio antes
sobre una fuente sintetica de juguete (bench-synth-16) y se cambio el
2026-08-13 para que el benchmark mida a la maquina sobre el trabajo que de
verdad va a hacer: la fuente sintetica son barras, no texto renderizado, y
tenia 60 imagenes en vez de 1000.

⚠ Un reporte solo es comparable con otro que traiga el MISMO `window_dataset`.
El cambio de fuente movio el numero, asi que benchmarks/foveal_20260813-134338
(bench-synth-16) no se compara con los posteriores. Por eso el reporte guarda
el nombre del dataset: la comparacion se filtra, no se supone.

La extraccion (ventana 16, stride 8, seed 1) NO cambio al cambiar la fuente,
para que el dato sea la unica variable nueva.

Aviso medido (CLAUDE.md, nota 2026-08-08): el micro-benchmark de costo miente
bajo carga -- un entrenamiento ocupando los nucleos infla el numero. Por eso
este script mide la carga del sistema ANTES de arrancar y la deja en el
reporte en vez de fingir que no importa.

Uso:
    .venv/bin/python scripts/bench_speed.py
    .venv/bin/python scripts/bench_speed.py --repeats 5 --device cuda
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fv import settings
from fv.ioutils import write_json_atomic
from fv.models.store import NetworkStore
from fv.training.loop import train
from fv.training.recipe import RecipeStore
from fv.training.registry import RunStore, git_commit
from fv.windows.extract import ExtractConfig, extract_windows

BENCH_SOURCE_NAME = "dirty-1000-80px"
BENCH_SOURCE_ID = "local/dirty-1000-80px"
BENCH_DATASET = "bench-dirty1000-16"
BENCH_NETWORK = "bench-16"
BENCH_RECIPE = "bench"


def _ensure_dataset(root: Path) -> None:
    out = settings.window_datasets_root() / BENCH_DATASET
    if (out / "windows.npz").exists():
        return
    if out.exists():
        raise SystemExit(
            f"{out} existe pero sin windows.npz (dataset a medias) -- "
            "borralo y reintenta: no se reutiliza en ese estado")
    src = settings.local_sources_root() / BENCH_SOURCE_NAME
    if not (src / "labels.jsonl").exists():
        # No se fabrica: el dato del benchmark es el real, y viene del proyecto
        # hermano. Fabricar aqui una fuente de juguete daria un numero con el
        # mismo aspecto y otro significado, que es peor que no dar ninguno.
        raise SystemExit(
            f"falta la fuente {BENCH_SOURCE_ID} en {src}\n"
            "  El benchmark mide sobre el dato real, que se genera en\n"
            "  image-text-sample-generator (receta 'dirty', 1000 imagenes 640x480)\n"
            "  y se trae en dos pasos -- README, 'Traer una fuente del generador':\n"
            "    1) copiar el dataset a data/sources/dirty-1000\n"
            f"    2) .venv/bin/fv-resize --source local/dirty-1000 --name {BENCH_SOURCE_NAME} --width 80")
    print(f"extrayendo ventanas de benchmark en {out} ...")
    cfg = ExtractConfig(source=BENCH_SOURCE_ID, window_size=16, stride=8, seed=1)
    extract_windows(cfg, out)


def _cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "desconocido"


def _ram_total_gb() -> float | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 2)
        except OSError:
            pass
    return None


def _gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def _load_avg():
    try:
        return os.getloadavg()
    except (OSError, AttributeError):
        return None


def hardware_info(device: str) -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_total_gb": _ram_total_gb(),
        "gpu": _gpu_name(device),
    }


def _progress(epoch: int, total: int, rec: dict) -> None:
    print(f"    epoca {epoch}/{total}  ({rec['seconds']:.2f}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mide s/epoca de entrenamiento de esta maquina (red+receta congeladas)")
    ap.add_argument("--repeats", type=int, default=3, help="repeticiones para promediar ruido")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=None, help="ruta del reporte (default benchmarks/<host>_<ts>.json)")
    args = ap.parse_args()
    if args.repeats < 1:
        print("--repeats debe ser >= 1", file=sys.stderr)
        return 2

    root = settings.project_root()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("--device cuda pedido pero no hay CUDA disponible en esta maquina",
                  file=sys.stderr)
            return 2

    load_before = _load_avg()
    cpu_count = os.cpu_count() or 1
    if load_before and load_before[0] / cpu_count > 0.5:
        print(f"AVISO: carga del sistema alta antes de empezar (loadavg1={load_before[0]:.2f} "
              f"sobre {cpu_count} nucleos) -- el numero puede mentir bajo carga "
              "(ver CLAUDE.md, nota 2026-08-08 punto 7)", file=sys.stderr)

    _ensure_dataset(root)

    net = NetworkStore().get(BENCH_NETWORK)
    recipe = RecipeStore().get(BENCH_RECIPE)
    store = RunStore()

    ts = time.strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname()

    out = Path(args.out) if args.out else root / "benchmarks" / f"{host}_{ts}.json"
    if out.exists():
        print(f"{out} ya existe -- no se sobrescribe, elige otra salida", file=sys.stderr)
        return 2

    run_names, seconds_per_epoch = [], []
    for i in range(args.repeats):
        name = f"bench-{host}-{ts}-{i}"
        print(f"repeticion {i + 1}/{args.repeats}: {name}")
        summary = train(name, BENCH_DATASET, BENCH_NETWORK, net, BENCH_RECIPE, recipe,
                        device=args.device, store=store, progress=_progress)
        run_names.append(name)
        seconds_per_epoch.append(summary["seconds_per_epoch"])
        print(f"  -> {summary['seconds_per_epoch']:.3f} s/epoca")

    mean = float(np.mean(seconds_per_epoch))
    std = float(np.std(seconds_per_epoch)) if len(seconds_per_epoch) > 1 else 0.0

    report = {
        "format_version": 1,
        "hostname": host,
        "timestamp": ts,
        "git_commit": git_commit(root),
        "hardware": hardware_info(args.device),
        "device": args.device,
        "load_avg_before": list(load_before) if load_before else None,
        "window_dataset": BENCH_DATASET,
        "network": BENCH_NETWORK,
        "recipe": BENCH_RECIPE,
        "repeats": args.repeats,
        "runs": run_names,
        "seconds_per_epoch": {"values": seconds_per_epoch, "mean": round(mean, 3),
                              "std": round(std, 3)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, report)

    print(f"\nOK: {mean:.3f} s/epoca (+/- {std:.3f}, n={args.repeats}) en {host} [{args.device}]")
    print(f"reporte: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
