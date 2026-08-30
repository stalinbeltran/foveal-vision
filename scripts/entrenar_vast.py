#!/usr/bin/env python3
"""Entrenar UN run en una maquina de Vast, trayendose los pesos SEGUN SE ESCRIBEN.

Por que existe, si ya esta `estudio_flota.py`
---------------------------------------------
La flota corre BARRIDOS: muchos puntos cortos, y su producto es la tabla. Por eso
su libro de a bordo se trae solo texto (`metrics.jsonl` y companeros) y **los
pesos se quedan en la maquina** hasta el tar final -- esta escrito en su propio
docstring, y para un barrido es la decision correcta: los `.pt` son ~700 KB por
run y lo que se lee es el numero.

Aqui el producto es **el modelo**. Un `best.pt` que solo baja al final es un
modelo que se pierde entero si la maquina se cae en la ultima epoca, y en Vast
eso no es hipotetico: son ordenadores de desconocidos alquilados por minutos.
Asi que esto se trae los pesos **en cada sonda**, no al final.

Y por eso mismo un run entrenado aqui **se puede continuar**: al bajar `last.pt`
con su estado (optimizador, contadores, generadores) baja lo que
`fv-continue` necesita. Ver `docs/entrenar.md`.

⚠ Lo que NO garantiza, y hay que decirlo
-----------------------------------------
La destruccion de la instancia va en un `finally` de ESTE proceso. Si el droplet
de control muere de golpe (SIGKILL, se destruye la maquina), el `finally` no
corre y **la instancia sigue facturando**. No hay interruptor dentro de la
maquina alquilada porque destruirse a si misma pediria el token de Vast, y ahi
no viaja ningun secreto a proposito.

Lo que si hay:
  - el `iid` y el comando exacto de destruccion se imprimen ANTES de nada mas;
  - `cerrable.mjs` cuenta las instancias vivas y pone el server en rojo;
  - `--horas-max` corta el entrenamiento (no la factura) por si algo se cuelga.

⚠ Y una diferencia con la flota que importa para comparar: un run continuado en
OTRA maquina no es bit a bit el mismo que si no se hubiera parado. `reanudar`
restaura los tres generadores, pero torch no promete el mismo flujo entre
maquinas distintas. Para entrenar un modelo da igual; para publicar una tabla
comparable, no.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANZADOR = ROOT.parent / "digital-ocean-dropplet-auto-launching"
sys.path.insert(0, str(ROOT / "src"))
if not (LANZADOR / "scripts" / "vast_instance.py").exists():
    raise SystemExit(
        f"Falta el lanzador en {LANZADOR}: sin el no se puede hablar con Vast.\n"
        "  git clone https://github.com/stalinbeltran/"
        "digital-ocean-dropplet-auto-launching")
sys.path.insert(0, str(LANZADOR / "scripts"))

import vast_instance as V                              # noqa: E402

from fv import settings                                # noqa: E402
from fv.training.registry import RunStore              # noqa: E402
from fv.windows.store import WindowDatasetStore        # noqa: E402

ENVIA = ["src", "scripts", "configs", "pyproject.toml"]
EXCLUYE = {"__pycache__", ".pytest_cache", ".venv", ".git", "node_modules"}
# Lo que se baja en CADA sonda. Los `.pt` estan aqui y esa es toda la diferencia
# con el libro de la flota: son 2,7 MB por sonda, y es lo que hace que una
# maquina que se cae no se lleve el modelo.
TRAER = ("metrics.jsonl", "status.json", "config.json", "summary.json",
         "best.pt", "last.pt")

INSTALL = """set -eu
cd /root/bench
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip >/dev/null
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy pillow pyyaml
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -e .
.venv/bin/python -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"
"""


def log(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> "NoReturn":                        # type: ignore[valid-type]
    log(f"\n{msg}")
    raise SystemExit(2)


# ---------------------------------------------------------------- preflight
def preflight(args) -> dict:
    """Todo lo que puede fallar, ANTES de alquilar. Descubrirlo a mitad es una
    maquina facturando para nada (R11)."""
    store = RunStore()
    wstore = WindowDatasetStore()

    npz = wstore.path(args.dataset) / "windows.npz"
    if not npz.exists():
        die(f"'{args.dataset}' no tiene windows.npz: no hay con que entrenar.\n"
            "  No se ha alquilado nada.")

    continuar = store.exists(args.name)
    if continuar and not args.continuar:
        die(f"ya existe el run '{args.name}'.\n"
            "  Para seguirlo: anade --continuar. Para uno nuevo: otro --name.\n"
            "  No se ha alquilado nada.")
    if args.continuar:
        if not continuar:
            die(f"--continuar pero no existe el run '{args.name}'.\n"
                "  No se ha alquilado nada.")
        if not (store.path(args.name) / "last.pt").exists():
            die(f"'{args.name}' no tiene last.pt: no hay desde donde seguir.\n"
                "  No se ha alquilado nada.")
        # ⚠ Un run en 'running' no se sube. Dos motivos, y el segundo es el sutil:
        # se estaria continuando algo que se esta entrenando AQUI (dos escrituras
        # sobre el mismo run), y ademas su `status.json` viaja con un `pid` de
        # ESTA maquina -- en la alquilada ese numero puede existir por
        # coincidencia y `reconcile` lo leeria como "sigue vivo", negandose a
        # continuar por un proceso que no tiene nada que ver.
        st = store.status(args.name).get("status")
        if st in ("running", "queued"):
            die(f"'{args.name}' esta en estado '{st}' en esta maquina: no se sube.\n"
                "  Espera a que acabe (o paralo) antes de continuarlo en Vast.\n"
                "  No se ha alquilado nada.")
    else:
        for f in (ROOT / "configs" / "networks" / f"{args.network}.yaml",
                  ROOT / "configs" / "recipes" / f"{args.recipe}.yaml"):
            if not f.exists():
                die(f"falta {f}.\n  No se ha alquilado nada.")

    V.load_env()
    try:
        V.token()
    except SystemExit:
        raise
    except Exception as e:                              # noqa: BLE001
        die(f"no hay token de Vast utilizable ({e}).\n  No se ha alquilado nada.")
    return {"store": store, "wstore": wstore, "continuar": bool(args.continuar)}


def construir_payload(args, ctx) -> Path:
    """Codigo + dataset + (si se continua) el run con su last.pt."""
    tmp = Path(tempfile.mkdtemp(prefix="tv-")) / "payload.tar.gz"

    def filtro(info: tarfile.TarInfo):
        return None if set(Path(info.name).parts) & EXCLUYE else info

    with tarfile.open(tmp, "w:gz") as tar:
        for nombre in ENVIA:
            origen = ROOT / nombre
            if not origen.exists():
                die(f"falta {origen}, que hace falta para entrenar")
            tar.add(str(origen), arcname=nombre, filter=filtro)
        # Origen y destino distintos A PROPOSITO: se lee del repo de datos y se
        # escribe en `data/window-datasets/`, que es donde cae el fallback de
        # `window_datasets_root()` en una maquina que no tiene ese repo (R6).
        tar.add(str(ctx["wstore"].path(args.dataset)),
                arcname=f"data/window-datasets/{args.dataset}", filter=filtro)
        if ctx["continuar"]:
            # el run entero, CON last.pt: es lo que `fv-continue` necesita
            tar.add(str(ctx["store"].path(args.name)),
                    arcname=f"data/runs/{args.name}", filter=filtro)
    return tmp


def _ssh(host: str, port: int, script: str, timeout: int = 90):
    """Como `V.ssh_capture` pero devolviendo TAMBIEN stderr.

    `V.ssh_capture` se queda solo con stdout, y el motivo por el que SSH falla
    --"Permission denied (publickey)", "Connection refused"-- viaja por stderr.
    Sin el, un rechazo de clave y una maquina que aun no levanta sshd se ven
    exactamente igual: `rc=255` y nada mas. No se toca aquella funcion porque la
    comparten otros; se envuelve aqui.
    """
    proc = subprocess.run(V.ssh_command(host, port) + ["bash -s"],
                          input=script.encode("utf-8"),
                          capture_output=True, timeout=timeout)
    return (proc.returncode,
            proc.stdout.decode("utf-8", errors="replace"),
            proc.stderr.decode("utf-8", errors="replace"))


def conectar(iid: int, minutos: float = 12.0,
             espera_s: float = 20.0) -> tuple[str, int]:
    """Devuelve (host, port) cuando SSH acepta la CLAVE de verdad.

    Junta las DOS trampas que este repo ya tenia medidas, y que no reutilizar me
    costo los dos primeros intentos (2026-08-30):

    1. **El banner no es el login.** `V.esperar_ssh` comprueba que sshd contesta,
       que llega ANTES de que la clave este en `authorized_keys`.
       `estudio_flota.sellar` reintenta 12 veces por esto, con la medida al lado
       (2026-08-24: sin reintentos, 3 de 5 maquinas fallaban en el primer comando
       autenticado). Primer intento: "Permission denied (publickey)" con la clave
       BIEN registrada.
    2. **El destino SSH que da la API puede no ser el definitivo.** La flota no
       usa `ssh_destino` a secas sino `resolver_destino`, con reintentos, porque
       MEDIDO el 2026-08-24 la API devolvio el mismo `host:puerto` para dos
       instancias distintas mientras arrancaban. Segundo intento: SSH no contesto
       nunca en el destino que se leyo una sola vez, al principio.

    Por eso aqui el destino se RE-PREGUNTA en cada vuelta en vez de leerse una
    vez: es barato y cubre las dos cosas.
    """
    fin = time.time() + minutos * 60
    intento, ultimo, deniegos = 0, "", 0
    while time.time() < fin:
        intento += 1
        try:
            info = V.instancia(iid)
            host, port = V.ssh_destino(info)
        except Exception as exc:                        # noqa: BLE001
            ultimo = f"no pude resolver el destino: {exc}"
            time.sleep(espera_s)
            continue
        code, salida, err = _ssh(host, port, "echo listo")
        if code == 0 and "listo" in salida:
            log(f"  SSH utilizable en {host}:{port} (intento {intento})")
            return host, port
        ultimo = f"{host}:{port} rc={code} {(err or salida).strip()[-120:]}"
        # ⚠ La asimetria de `estudio_flota.sellar`, aplicada aqui: un fallo de
        # TRANSPORTE mejora esperando (la maquina todavia no levanto sshd); un
        # "Permission denied" NO mejora nunca. Distinguirlos importa porque sin
        # esto son 12 minutos de reintentos ciegos y un diagnostico que apunta al
        # sitio equivocado -- me paso el 2026-08-30, y lo que lo escondia es que
        # `V.ssh_capture` se come stderr (devuelve solo stdout), asi que el
        # motivo real nunca llegaba al log.
        if "permission denied" in (err or "").lower():
            deniegos += 1
            if deniegos >= 3:
                raise RuntimeError(
                    f"{host}:{port} RECHAZA la clave ({V.clave_privada()}). Esto no "
                    f"mejora esperando.\n"
                    f"  Comprueba que esta registrada ANTES de alquilar:\n"
                    f"    python3 {LANZADOR}/scripts/vast_instance.py register-key\n"
                    f"  (la lista de claves de una instancia se fija al crearla: "
                    f"registrarla despues no sirve para esta)")
        if intento == 1 or intento % 3 == 0:
            log(f"  esperando a SSH ({ultimo})")
        time.sleep(espera_s)
    raise RuntimeError(f"SSH no llego a funcionar en {minutos:g} min: {ultimo}")


def dir_remoto(host: str, port: int, name: str) -> str:
    """Donde escribe el run EN LA MAQUINA, preguntandoselo a ella.

    ⚠ No se puede cablear `data/runs/<name>`: un run suelto NO va ahi, va bajo la
    carpeta del mes (`RunStore.destino` -> `artefactos.destino_agrupado`). Lo
    comprobe antes de alquilar nada; cablearlo habria dejado la maquina
    entrenando bien y a este lado bajando cero ficheros, que es el fallo que
    parece "no entreno" y cuesta el alquiler entero.

    Se pregunta en CADA sonda a proposito: al principio el run todavia no existe
    y `path()` devuelve la forma plana, asi que resolverlo una sola vez al
    empezar daria la ruta equivocada para siempre.
    """
    code, salida = V.ssh_capture(
        host, port,
        "cd /root/bench && FV_DATA_ROOT=/root/bench/data .venv/bin/python -c "
        "\"import sys;sys.path.insert(0,'src');"
        "from fv.training.registry import RunStore;"
        f"print(RunStore().path('{name}'))\" 2>/dev/null || true\n", 180)
    linea = (salida or "").strip().splitlines()
    return linea[-1].strip() if linea else ""


def traer(host: str, port: int, name: str, destino: Path, remoto_dir: str) -> list:
    """Baja los ficheros del run que existan. Los que no, se saltan sin ruido:
    `summary.json` no existe hasta que termina y `best.pt` no hasta la primera
    epoca -- pedirlos y fallar seria confundir 'todavia no' con 'error'."""
    if not remoto_dir:
        return []
    destino.mkdir(parents=True, exist_ok=True)
    traidos = []
    for f in TRAER:
        remoto = f"{remoto_dir}/{f}"
        local = destino / f
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            p = subprocess.run(
                V.ssh_command(host, port) + [f"cat {remoto} 2>/dev/null || true"],
                stdout=tmp, timeout=300)
        tmp_path = Path(tmp.name)
        try:
            if p.returncode == 0 and tmp_path.stat().st_size > 0:
                tmp_path.replace(local)
                traidos.append(f)
            else:
                tmp_path.unlink(missing_ok=True)
        except OSError:
            tmp_path.unlink(missing_ok=True)
    return traidos


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Entrena UN run en una maquina de Vast y baja los pesos segun se escriben")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="dirty1000-80px-16px-r20260827")
    ap.add_argument("--network", default="fov16-optimo")
    ap.add_argument("--recipe", default="plan40")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--continuar", action="store_true",
                    help="sigue un run que ya existe aqui (sube su last.pt)")
    ap.add_argument("--cpus", type=int, default=8, help="vCPU minimos")
    ap.add_argument("--max-cpus", type=int, default=32)
    ap.add_argument("--ram-gb", type=float, default=8.0)
    ap.add_argument("--max-precio", type=float, default=0.15, help="$/h")
    ap.add_argument("--disco-gb", type=float, default=12.0)
    ap.add_argument("--cada", type=int, default=60, help="segundos entre sondas")
    ap.add_argument("--horas-max", type=float, default=2.0)
    ap.add_argument("--prefijo", default="tv-")
    ap.add_argument("--yes", action="store_true", help="no preguntar")
    args = ap.parse_args()

    ctx = preflight(args)
    payload = construir_payload(args, ctx)
    mb = payload.stat().st_size / 1e6

    oferta = V.elegir_oferta(args.cpus, args.max_cpus, args.ram_gb, args.max_precio)
    precio = float(oferta.get("dph_total", 0.0))
    resumen = V.resumen_maquina(oferta)
    log(f"\nMaquina elegida: {resumen.get('cpu_name', '?')} · "
        f"{oferta.get('cpu_cores_effective', '?')} vCPU · "
        f"{oferta.get('cpu_ram', 0) / 1024:.0f} GB · {precio:.4f} $/h")
    log(f"Payload: {mb:.1f} MB · plazo maximo {args.horas_max:g} h "
        f"(techo de gasto {precio * args.horas_max:.3f} $)")
    if not args.yes and not V.confirmar("Alquilar?"):
        return 1

    etiqueta = f"{args.prefijo}{args.name}"[:60]
    iid = V.alquilar(oferta, etiqueta, V.cfg("VAST_IMAGE"), args.disco_gb)
    t0 = time.time()
    # Lo PRIMERO que se imprime, antes de nada que pueda fallar: si este proceso
    # muere aqui, esto es lo unico que queda para poder apagarla.
    log(f"\n  ALQUILADA: instancia {iid} ({etiqueta}) a {precio:.4f} $/h")
    log(f"  Si algo va mal, esto la apaga:")
    log(f"    python3 {LANZADOR}/scripts/vast_instance.py destroy {iid} --yes\n")

    destino = ctx["store"].destino(args.name) if not ctx["continuar"] \
        else ctx["store"].path(args.name)
    codigo = 1
    try:
        info = V.esperar_estado(iid, int(V.cfg("VAST_BOOT_TIMEOUT")))
        estado = (info.get("actual_status") or info.get("cur_state") or "?").lower()
        if estado != "running":
            # como la flota: una instancia que no arranca factura igual que una
            # buena, asi que se dice y se destruye, no se espera por si acaso
            raise RuntimeError(f"la instancia acabo en '{estado}', no arranco")
        host, port = conectar(iid)

        with payload.open("rb") as fh:
            p = subprocess.run(V.ssh_command(host, port) +
                               ["mkdir -p /root/bench && cat > /root/payload.tar.gz"],
                               stdin=fh, timeout=1800)
        if p.returncode != 0:
            raise RuntimeError("no pude subir el payload")
        if V.ssh_script(host, port,
                        "set -eu\ncd /root/bench\ntar -xzf /root/payload.tar.gz\n",
                        600) != 0:
            raise RuntimeError("no pude desempaquetar el payload")
        log("  payload subido y desempaquetado; instalando (tarda unos minutos)...")
        if V.ssh_script(host, port, INSTALL, 2400) != 0:
            raise RuntimeError("fallo la instalacion en la maquina")

        # FV_DATA_ROOT apunta dentro de /root/bench: sin el, `data_root()` cae al
        # repo de codigo y los runs se escribirian en otro sitio del que se leen.
        if ctx["continuar"]:
            cmd = (f".venv/bin/fv-continue --name {args.name} --more {args.epochs}")
        else:
            cmd = (f".venv/bin/fv-train --name {args.name} "
                   f"--window-dataset {args.dataset} --network {args.network} "
                   f"--recipe {args.recipe} --epochs {args.epochs}")
        lanzar = (f"set -eu\ncd /root/bench\nexport FV_DATA_ROOT=/root/bench/data\n"
                  f"nohup {cmd} > /root/train.log 2>&1 &\n"
                  f"echo lanzado\n")
        if V.ssh_script(host, port, lanzar, 120) != 0:
            raise RuntimeError("no pude lanzar el entrenamiento")
        log(f"  entrenando: {cmd}\n")

        plazo = t0 + args.horas_max * 3600
        ultimo = -1
        while True:
            time.sleep(args.cada)
            rdir = dir_remoto(host, port, args.name)
            traidos = traer(host, port, args.name, destino, rdir)
            n = 0
            met = destino / "metrics.jsonl"
            if met.exists():
                lineas = [l for l in met.read_text(encoding="utf-8").splitlines() if l.strip()]
                n = len(lineas)
                if n > ultimo and lineas:
                    m = json.loads(lineas[-1])
                    log(f"  epoca {m['epoch']}  train_loss={m['train_loss']:.4f}  "
                        f"val_loss={m['val']['loss']:.4f}  f1={m['val']['f1']:.3f}  "
                        f"({m['seconds']:.0f}s)  [bajados: {', '.join(traidos)}]")
                    ultimo = n
            code, estado = V.ssh_capture(
                host, port,
                f"cat {rdir}/status.json 2>/dev/null || true\n", 120)
            st = ""
            try:
                st = json.loads(estado).get("status", "")
            except (json.JSONDecodeError, AttributeError):
                pass
            if st in ("done", "error", "cancelled"):
                log(f"\n  el entrenamiento termino: {st}")
                break
            if time.time() > plazo:
                log("\n  AVISO: se agoto --horas-max; me traigo lo que haya")
                break
        traer(host, port, args.name, destino, dir_remoto(host, port, args.name))
        codigo = 0
    except Exception as exc:                            # noqa: BLE001
        log(f"\n  FALLO: {exc}")
        try:
            traer(host, port, args.name, destino,       # lo que haya, igualmente
                  dir_remoto(host, port, args.name))
            log("  (me traje lo que hubiera del run antes de destruir)")
        except Exception:                               # noqa: BLE001
            pass
    finally:
        vivida = time.time() - t0
        try:
            V.destruir(iid)
            log(f"\n  instancia {iid} DESTRUIDA. Vivio {vivida / 60:.1f} min, "
                f"{precio * vivida / 3600:.4f} $")
        except Exception as exc:                        # noqa: BLE001
            log(f"\n  AVISO GRAVE: no pude destruir {iid}: {exc}\n"
                f"  SIGUE FACTURANDO. Destruyela ya:\n"
                f"    python3 {LANZADOR}/scripts/vast_instance.py destroy {iid} --yes")
            return 3

    log(f"\n  el run esta en: {destino}")
    for f in TRAER:
        p = destino / f
        log(f"    {'ok ' if p.exists() else '-- '} {f}"
            + (f"  ({p.stat().st_size} B)" if p.exists() else ""))
    log(f"\n  para seguir entrenandolo:\n"
        f"    .venv/bin/fv-continue --name {args.name} --more N        (aqui)\n"
        f"    python3 scripts/entrenar_vast.py --name {args.name} --continuar --epochs N")
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
