#!/usr/bin/env python3
"""El dato del benchmark: generarlo, publicarlo en el volumen, instalarlo.

El problema que resuelve
------------------------
`bench_speed.py` mide sobre un dataset CONGELADO (`bench-dirty1000-16`), y se
niega -bien- a fabricar una fuente de juguete si no lo encuentra. Pero en una
maquina recien hecha nunca esta, y reconstruirlo son tres pasos en dos repos
que hay que recordar en el orden correcto. Cuando no se recuerdan, pasa lo de
siempre: se da por imposible y se mide sobre otra cosa.

Aqui esta la cadena entera, con un comando por tramo:

    build      generador (mil renders) -> fuente 640x480 -> reducida a 80px
               -> ventanas 16/8/seed 1.  Es el tramo caro: decenas de minutos.
    publish    copia el resultado al volumen y le pone una huella SHA-256.
    install    copia del volumen (o de donde se le diga) al data/ local.
               Solo biblioteca estandar: corre en un droplet de medicion
               pelado, sin venv ni torch.
    verify     comprueba que lo que hay en disco es lo que dice la huella.

Que sea reproducible no es una promesa de este script: los specs del generador
estan congelados en git (`specs.jsonl`, seed 1) y la extraccion tiene su propia
semilla. Dos maquinas que corran `build` obtienen el mismo `windows.npz`; la
huella de `publish` es lo que lo convierte de promesa en comprobacion.

    python3 scripts/bench_dataset.py build
    python3 scripts/bench_dataset.py publish --to /mnt/bench-data
    python3 scripts/bench_dataset.py install --from /mnt/bench-data
    python3 scripts/bench_dataset.py verify --from /mnt/bench-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERADOR = ROOT.parent / "image-text-sample-generator"

# `fv.settings` es la UNICA que sabe donde vive el dato, y desde que el
# `windows.npz` se commitea eso ya no es `ROOT/data`. Se importa aunque este
# script sea stdlib a proposito (ver `cmd_install`): `settings` tambien lo es
# --solo `os` y `pathlib`--, asi que `install` sigue corriendo en un droplet
# recien hecho, sin venv ni dependencias.
sys.path.insert(0, str(ROOT / "src"))
from fv import settings  # noqa: E402

# Todo esto tiene que coincidir con scripts/bench_speed.py. Si alguna vez deja
# de coincidir, el benchmark medira sobre un dato que no es el suyo y no se
# notara: por eso build lo comprueba contra el propio bench_speed.py al final.
DATASET_GENERADOR = "dirty-1000-699b2e01"
FUENTE_GRANDE = "dirty-1000"
FUENTE = "dirty-1000-80px"
VENTANAS = "bench-dirty1000-16"
EXTRACCION = {"window_size": 16, "stride": 8, "seed": 1}

# Lo que se publica y se copia. La fuente reducida va porque permite volver a
# extraer con otros parametros sin repetir los mil renders; la grande de
# 640x480 no, que son 234 MB de cache regenerable (README, "la copia grande es
# desechable").
ARTEFACTOS = [f"window-datasets/{VENTANAS}", f"sources/{FUENTE}"]


def raiz_local(rel: str) -> Path:
    """Donde esta `rel` EN ESTA MAQUINA.

    El volumen y los droplets de medicion usan la forma plana `data/<rel>`, pero
    aqui los dos artefactos ya no viven bajo la misma raiz: las ventanas se
    fueron al repo de datos (se commitean) y las fuentes no (234 MB de renders
    regenerables). Por eso se resuelve uno a uno en vez de con un prefijo.
    """
    familia, _, nombre = rel.partition("/")
    if familia == "window-datasets":
        return settings.window_datasets_root() / nombre
    if familia == "sources":
        return settings.local_sources_root() / nombre
    return ROOT / "data" / rel
HUELLA = "FINGERPRINT.json"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(1)


def correr(cmd: list[str], cwd: Path, descripcion: str) -> None:
    log(f"\n$ {' '.join(str(c) for c in cmd)}")
    code = subprocess.run(cmd, cwd=str(cwd)).returncode
    if code != 0:
        die(f"{descripcion} fallo (codigo {code}).")


def venv_python(repo: Path) -> Path:
    """El interprete del venv del repo, que es el unico con sus dependencias."""
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        p = repo / rel
        if p.exists():
            return p
    die(
        f"No hay venv en {repo}.\n"
        f"  Crealo con:  cd {repo} && bash scripts/setup-linux.sh"
        if (repo / "scripts" / "setup-linux.sh").exists()
        else f"No hay venv en {repo}. Crealo con: python3 -m venv .venv && "
        ".venv/bin/pip install -e '.[train]'"
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True
        )
        return out.stdout.strip() or "?"
    except OSError:
        return "?"


def tamano(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def humano(n: int) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if n < 1024 or unidad == "GB":
            return f"{n:.0f} {unidad}" if unidad == "B" else f"{n:.1f} {unidad}"
        n /= 1024.0
    return f"{n:.1f} GB"


def npz_local() -> Path:
    return settings.window_datasets_root() / VENTANAS / "windows.npz"


# ------------------------------------------------------------------------ build


def cmd_build(args: argparse.Namespace) -> int:
    if npz_local().exists() and not args.force:
        log(f"Ya existe {npz_local()} ({humano(tamano(npz_local()))}).")
        log("  No se rehace. Para rehacerlo: --force (borra y vuelve a empezar).")
        return 0

    t0 = time.monotonic()

    # 1) Los mil renders, en el repo hermano.
    if not GENERADOR.exists():
        die(
            f"No esta el generador en {GENERADOR}.\n"
            "  clonalo:  git clone https://github.com/stalinbeltran/"
            "image-text-sample-generator.git"
        )
    imagenes = GENERADOR / "data" / "datasets" / DATASET_GENERADOR / "images"
    hechas = len(list(imagenes.glob("*.png"))) if imagenes.exists() else 0
    if hechas >= args.count:
        log(f"1/4 renders: ya hay {hechas} imagenes en el generador, no se repiten.")
    else:
        log(f"1/4 renders: faltan imagenes ({hechas} de {args.count}). Esto es lo lento.")
        correr(
            [str(venv_python(GENERADOR)), "scripts/build_dataset.py", DATASET_GENERADOR,
             "--end", str(args.count)],
            GENERADOR,
            "el renderizado del generador",
        )

    # 2) Traer la fuente: los tres ficheros que consume este proyecto.
    origen = GENERADOR / "data" / "datasets" / DATASET_GENERADOR
    destino = ROOT / "data" / "sources" / FUENTE_GRANDE
    if (destino / "labels.jsonl").exists():
        log(f"2/4 copia: {destino} ya esta.")
    else:
        log(f"2/4 copia: {origen} -> {destino}")
        destino.mkdir(parents=True, exist_ok=True)
        for sub in ("images", "masks"):
            if (origen / sub).exists():
                shutil.copytree(origen / sub, destino / sub, dirs_exist_ok=True)
        for fichero in ("labels.jsonl", "dataset.json"):
            shutil.copy2(origen / fichero, destino / fichero)
        log(f"    {humano(tamano(destino))} copiados")

    # 3) Reducir a 80 px. Es sobre la reducida sobre la que estan medidos los runs.
    py = venv_python(ROOT)
    reducida = ROOT / "data" / "sources" / FUENTE
    if (reducida / "labels.jsonl").exists():
        log(f"3/4 resize: {reducida} ya esta.")
    else:
        log("3/4 resize: 640x480 -> 80x60")
        correr(
            [str(py), "-m", "fv.datasets.cli", "--source", f"local/{FUENTE_GRANDE}",
             "--name", FUENTE, "--width", "80"],
            ROOT,
            "el resize",
        )

    # 4) Extraer las ventanas: esto es lo que come el benchmark.
    #
    # `extract_windows` se niega a escribir si el directorio ya existe ("no se
    # sobrescribe nunca"), y este repo trae `manifest.json` y `split.json`
    # COMMITEADOS justo ahi. O sea que en un clon limpio el directorio siempre
    # existe y extraer encima no termina nunca: el build moria aqui, en la
    # unica maquina donde hace falta. Se extrae aparte y se coloca el .npz al
    # lado de los ficheros de git.
    #
    # Eso ademas convierte al manifest commiteado de estorbo en COMPROBACION:
    # trae la huella de lo que se midio antes, asi que si la recien construida
    # no coincide, el dato no reproduce y medir sobre el daria un numero que no
    # es comparable con los anteriores. Mejor parar que publicar eso.
    log(f"4/4 ventanas: {EXTRACCION}")
    destino_v = npz_local().parent
    tmp = settings.window_datasets_root() / f".{VENTANAS}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    correr(
        [str(py), "-c",
         "import sys; sys.path.insert(0, 'src');"
         "from pathlib import Path;"
         "from fv.windows.extract import ExtractConfig, extract_windows;"
         f"cfg = ExtractConfig(source='local/{FUENTE}', **{EXTRACCION!r});"
         f"out = Path({str(tmp)!r});"
         "extract_windows(cfg, out);"
         "print('ventanas extraidas en', out)"],
        ROOT,
        "la extraccion de ventanas",
    )
    if not (tmp / "windows.npz").exists():
        die(f"La extraccion termino sin dejar {tmp / 'windows.npz'}.")

    referencia = destino_v / "manifest.json"
    if referencia.exists():
        antes = json.loads(referencia.read_text(encoding="utf-8")).get("fingerprint")
        ahora = json.loads((tmp / "manifest.json").read_text(encoding="utf-8")).get("fingerprint")
        if antes != ahora:
            shutil.rmtree(tmp, ignore_errors=True)
            die(f"El dato NO reproduce.\n"
                f"  el manifest de git dice: {antes}\n"
                f"  lo recien construido da: {ahora}\n"
                "  No es comparable con las medidas anteriores: revisa el "
                "generador antes de medir nada.")
        log(f"    huella {antes} igual a la de git: el dato reproduce.")

    destino_v.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tmp / "windows.npz", destino_v / "windows.npz")
    for f in ("manifest.json", "split.json"):
        if not (destino_v / f).exists():
            shutil.copy2(tmp / f, destino_v / f)
    shutil.rmtree(tmp, ignore_errors=True)

    # La comprobacion que evita el fallo silencioso: que lo construido sea lo
    # que bench_speed.py va a buscar, y no algo con el mismo aspecto.
    bench = (ROOT / "scripts" / "bench_speed.py").read_text(encoding="utf-8")
    for nombre, valor in (("BENCH_DATASET", VENTANAS), ("BENCH_SOURCE_NAME", FUENTE)):
        if f'{nombre} = "{valor}"' not in bench:
            die(
                f"bench_speed.py ya no usa {nombre} = \"{valor}\".\n"
                "  Lo que se acaba de construir NO es el dato de ese benchmark.\n"
                "  Actualiza las constantes de la cabecera de este script."
            )

    log(f"\nListo en {(time.monotonic() - t0) / 60:.1f} min.")
    log(f"  {npz_local()}  ({humano(tamano(npz_local()))})")
    if args.borrar_grande and destino.exists():
        shutil.rmtree(destino)
        log(f"  borrada la copia de 640x480 ({FUENTE_GRANDE}): se regenera cuando haga falta")
    log("\nSiguiente:  python3 scripts/bench_dataset.py publish --to /mnt/bench-data")
    return 0


# ---------------------------------------------------------------------- publish


def cmd_publish(args: argparse.Namespace) -> int:
    destino = Path(args.to)
    if not destino.is_dir():
        die(
            f"No existe {destino}.\n"
            "  Si es el volumen, montalo primero desde el lanzador:\n"
            "    python3 scripts/do_droplet.py volume attach bench-data"
        )
    if not npz_local().exists():
        die(f"No hay nada que publicar: falta {npz_local()}.\n  Corre antes: build")

    log(f"Publicando en {destino}…")
    for rel in ARTEFACTOS:
        origen = raiz_local(rel)
        if not origen.exists():
            log(f"  (falta {rel}, se omite)")
            continue
        dst = destino / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(origen, dst)
        log(f"  {rel}  {humano(tamano(dst))}")

    huella = {
        "format_version": 1,
        "creado": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sha256_windows_npz": sha256(npz_local()),
        "bytes_windows_npz": npz_local().stat().st_size,
        "window_dataset": VENTANAS,
        "fuente": FUENTE,
        "extraccion": EXTRACCION,
        "generador": {
            "dataset_id": DATASET_GENERADOR,
            "commit": git_commit(GENERADOR),
        },
        "foveal_commit": git_commit(ROOT),
        "artefactos": ARTEFACTOS,
    }
    (destino / HUELLA).write_text(
        json.dumps(huella, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"\nHuella: {huella['sha256_windows_npz'][:16]}…  ({destino / HUELLA})")
    log("Ese SHA es lo que hace comparables dos mediciones: si cambia, no lo son.")
    return 0


# ---------------------------------------------------------------------- install


def cmd_install(args: argparse.Namespace) -> int:
    """Copia el dato al disco LOCAL. Solo biblioteca estandar, a proposito.

    Corre en un droplet de medicion recien hecho, donde todavia no hay venv ni
    dependencias. Y copia en vez de montar porque el benchmark tiene que leer
    de disco local: medir con el dataset en un disco de red mide la red.
    """
    origen = Path(getattr(args, "from"))
    if not origen.is_dir():
        die(f"No existe {origen}.")
    huella_origen = origen / HUELLA
    if not huella_origen.is_file():
        die(f"{origen} no tiene {HUELLA}: no parece el dato del benchmark.")

    datos = ROOT / "data"
    for rel in json.loads(huella_origen.read_text(encoding="utf-8"))["artefactos"]:
        src = origen / rel
        if not src.exists():
            die(f"La huella declara {rel} pero no esta en {origen}.")
        dst = datos / rel
        if dst.exists() and not args.force:
            log(f"  {rel} ya esta en local, no se toca (--force para rehacer)")
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        log(f"  {rel}  {humano(tamano(dst))} -> {dst}")

    shutil.copy2(huella_origen, datos / HUELLA)
    return cmd_verify(argparse.Namespace(**{"from": str(datos)}))


# ----------------------------------------------------------------------- verify


def cmd_verify(args: argparse.Namespace) -> int:
    base = Path(getattr(args, "from"))
    huella_path = base / HUELLA
    if not huella_path.is_file():
        die(f"No hay {HUELLA} en {base}.")
    huella = json.loads(huella_path.read_text(encoding="utf-8"))

    npz = base / f"window-datasets/{huella['window_dataset']}/windows.npz"
    if not npz.is_file():
        die(f"La huella habla de {npz} y ese fichero no esta.")

    actual = sha256(npz)
    esperado = huella["sha256_windows_npz"]
    if actual != esperado:
        die(
            f"El dato NO es el de la huella.\n"
            f"  esperado: {esperado}\n"
            f"  en disco: {actual}\n"
            "  Una medicion sobre esto no se compara con las anteriores."
        )
    log(f"OK: {npz.name} coincide con la huella ({actual[:16]}…)")
    log(f"    {huella['window_dataset']} · extraccion {huella['extraccion']}")
    log(f"    del generador {huella['generador']['dataset_id']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="accion", required=True)

    p = sub.add_parser("build", help="genera el dato desde cero (lento: mil renders)")
    p.add_argument("--count", type=int, default=1000, help="cuantas imagenes (por defecto 1000)")
    p.add_argument("--force", action="store_true", help="rehacer aunque ya exista")
    p.add_argument(
        "--borrar-grande",
        action="store_true",
        help="borrar la copia de 640x480 al terminar (234 MB regenerables)",
    )
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("publish", help="copia el dato al volumen y le pone la huella")
    p.add_argument("--to", default="/mnt/bench-data")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("install", help="copia el dato del volumen al disco local")
    p.add_argument("--from", default="/mnt/bench-data", dest="from")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("verify", help="comprueba el dato contra su huella")
    p.add_argument("--from", default="/mnt/bench-data", dest="from")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
