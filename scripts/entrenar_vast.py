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
import statistics
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
        # ⚠ El temporal va AL LADO del destino, no en /tmp, y no es un detalle:
        # `Path.replace` es `os.replace`, que solo es atomico DENTRO del mismo
        # sistema de ficheros. Con el temporal en /tmp --que en muchas maquinas es
        # un tmpfs aparte-- daria EXDEV, el `except OSError` de abajo se lo
        # tragaria y la descarga fallaria EN SILENCIO para siempre. Aqui salio
        # bien por casualidad (mismo /dev/vda1, comprobado el 2026-08-30).
        #
        # Y la atomicidad importa de verdad: `best.pt` se lee MIENTRAS se
        # reemplaza -- la pantalla de revision usa el modelo con el
        # entrenamiento en marcha. Con un rename atomico, quien lee obtiene la
        # version vieja o la nueva, nunca media.
        with tempfile.NamedTemporaryFile(delete=False, dir=str(destino),
                                         prefix=f".{f}.") as tmp:
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
        except OSError as exc:
            # que no se pueda colocar un fichero NO es "todavia no esta": se dice
            tmp_path.unlink(missing_ok=True)
            log(f"  AVISO: no pude colocar {f}: {exc}")
    return traidos


# --------------------------------------------------- cuando cambiar de maquina
#
# Los DOS casos que se dan, y son distintos (R13: escritos antes de mirar):
#
#   se DEGRADO  la maquina iba bien y se puso lenta -- otro inquilino le comio
#               los nucleos. Se mide contra SI MISMA: mediana de las 3 ultimas
#               epocas contra la de las 3 primeras. El umbral 1.35 no es
#               inventado: es el que ya usa `estudio_flota --umbral-degradacion`.
#   nacio LENTA  el marketplace da maquinas muy distintas por el mismo precio.
#               Eso no se ve contra si misma --es lenta desde la primera epoca--
#               sino contra la MEJOR que hemos visto en esta corrida.
#
# Los dos piden un minimo de epocas antes de juzgar: una epoca suelta mide el
# arranque (cache fria, primer batch), no la maquina.
UMBRAL_DEGRADACION = 1.35
UMBRAL_LENTA = 1.6
MIN_EPOCAS_DEGRADACION = 6
MIN_EPOCAS_LENTA = 3


def _medianas(segs: list) -> "tuple[float, float] | None":
    if len(segs) < MIN_EPOCAS_DEGRADACION:
        return None
    return statistics.median(segs[:3]), statistics.median(segs[-3:])


def veredicto_maquina(segs: list, mejor: "float | None") -> "dict | None":
    """¿Hay que cambiar de maquina? Devuelve el motivo, o None para seguir."""
    par = _medianas(segs)
    if par:
        base, reciente = par
        if base and reciente / base > UMBRAL_DEGRADACION:
            return {"motivo": "degradada",
                    "detalle": f"se puso lenta: {reciente:.0f} s/epoca contra "
                               f"{base:.0f} al empezar ({reciente / base:.2f}x)"}
    if len(segs) >= MIN_EPOCAS_LENTA and mejor:
        mia = statistics.median(segs)
        if mia / mejor > UMBRAL_LENTA:
            return {"motivo": "lenta",
                    "detalle": f"nacio lenta: {mia:.0f} s/epoca contra {mejor:.0f} "
                               f"de la mejor de esta corrida ({mia / mejor:.2f}x)"}
    return None


def avisar(texto: str) -> None:
    """Un aviso a Telegram, si se puede. NUNCA rompe el entrenamiento: es una
    comodidad, y la fuente de verdad es el log y el run en disco (CLAUDE.md)."""
    coord = Path(os.environ.get("COORD_HOME") or (Path.home() / "src/telegram-coordinator"))
    notify = coord / "scripts" / "notify.mjs"
    if not notify.exists():
        return
    try:
        subprocess.run(["node", str(notify), texto], timeout=60,
                       capture_output=True)
    except Exception:                                   # noqa: BLE001
        pass


def una_maquina(args, ctx, destino, estado) -> dict:
    """Alquila UNA maquina, entrena en ella, y la destruye pase lo que pase.

    Devuelve por que se salio: `done` (termino el entrenamiento), `degradada`,
    `lenta`, `presupuesto`, `plazo`, o `fallo`. Quien llama decide si alquila
    otra -- aqui no, para que el camino de destruccion sea SIEMPRE el mismo y no
    dependa de la decision de seguir.
    """
    payload = construir_payload(args, ctx)
    oferta = V.elegir_oferta(args.cpus, args.max_cpus, args.ram_gb, args.max_precio)
    precio = float(oferta.get("dph_total", 0.0))
    vcpu = oferta.get("cpu_cores_effective", "?")
    log(f"\nMaquina: {vcpu} vCPU · {oferta.get('cpu_ram', 0) / 1024:.0f} GB · "
        f"{precio:.4f} $/h · payload {payload.stat().st_size / 1e6:.1f} MB")

    etiqueta = f"{args.prefijo}{args.name}"[:60]
    iid = V.alquilar(oferta, etiqueta, V.cfg("VAST_IMAGE"), args.disco_gb)
    t0 = time.time()
    # Lo PRIMERO, antes de nada que pueda fallar: si este proceso muere aqui,
    # esto es lo unico que queda para poder apagarla.
    log(f"  ALQUILADA: instancia {iid} ({etiqueta}) a {precio:.4f} $/h")
    log(f"    python3 {LANZADOR}/scripts/vast_instance.py destroy {iid} --yes")
    estado["instancias"].append(iid)

    salida = {"motivo": "fallo", "detalle": "", "iid": iid, "precio": precio,
              "vcpu": vcpu, "gasto": 0.0, "mediana": None, "epocas": 0}
    host = port = None
    epocas_aqui: list = []          # s/epoca MEDIDAS EN ESTA MAQUINA
    try:
        info = V.esperar_estado(iid, int(V.cfg("VAST_BOOT_TIMEOUT")))
        st = (info.get("actual_status") or info.get("cur_state") or "?").lower()
        if st != "running":
            raise RuntimeError(f"la instancia acabo en '{st}', no arranco")
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
        log("  payload subido; instalando (unos minutos)...")
        if V.ssh_script(host, port, INSTALL, 2400) != 0:
            raise RuntimeError("fallo la instalacion en la maquina")

        if ctx["continuar"]:
            cmd = (f".venv/bin/fv-continue --name {args.name} --more {args.epochs}"
                   f" --patience {args.patience}")
        else:
            cmd = (f".venv/bin/fv-train --name {args.name} "
                   f"--window-dataset {args.dataset} --network {args.network} "
                   f"--recipe {args.recipe} --epochs {args.epochs} "
                   f"--patience {args.patience}")
        lanzar = (f"set -eu\ncd /root/bench\nexport FV_DATA_ROOT=/root/bench/data\n"
                  f"nohup {cmd} > /root/train.log 2>&1 &\necho lanzado\n")
        if V.ssh_script(host, port, lanzar, 120) != 0:
            raise RuntimeError("no pude lanzar el entrenamiento")
        log(f"  entrenando: {cmd}")

        vistas = len(_epocas(destino))  # las que ya venian de antes
        while True:
            time.sleep(args.cada)
            rdir = dir_remoto(host, port, args.name)
            traer(host, port, args.name, destino, rdir)
            filas = _epocas(destino)
            for m in filas[vistas:]:
                epocas_aqui.append(m["seconds"])
                log(f"  epoca {m['epoch']}  train={m['train_loss']:.4f} "
                    f"val={m['val']['loss']:.4f} f1={m['val']['f1']:.3f} "
                    f"({m['seconds']:.0f}s)")
            vistas = len(filas)
            estado["gasto_vivo"] = estado["gasto"] + precio * (time.time() - t0) / 3600

            code, txt = V.ssh_capture(
                host, port, f"cat {rdir}/status.json 2>/dev/null || true\n", 120)
            try:
                sit = json.loads(txt).get("status", "")
            except (json.JSONDecodeError, AttributeError):
                sit = ""
            if sit in ("done", "error", "cancelled"):
                salida["motivo"] = "done" if sit == "done" else "fallo"
                salida["detalle"] = f"el entrenamiento termino: {sit}"
                break
            if estado["gasto_vivo"] >= args.presupuesto:
                salida["motivo"] = "presupuesto"
                salida["detalle"] = (f"techo de {args.presupuesto:.2f} $ alcanzado "
                                     f"({estado['gasto_vivo']:.3f} $)")
                break
            if time.time() > estado["t_fin"]:
                salida["motivo"] = "plazo"
                salida["detalle"] = "se agoto --horas-max"
                break
            if time.time() - estado["ultimo_aviso"] > args.aviso_cada * 3600:
                estado["ultimo_aviso"] = time.time()
                ult = filas[-1] if filas else None
                avisar(
                    f"'{args.name}' sigue: {len(filas)} epocas"
                    + (f", f1={ult['val']['f1']:.3f}" if ult else "")
                    + (f", {statistics.median(epocas_aqui):.0f} s/epoca"
                       if epocas_aqui else "")
                    + f", {estado['gasto_vivo']:.3f} $ de {args.presupuesto:.2f}")
            v = veredicto_maquina(epocas_aqui, estado["mejor"])
            if v and estado["cambios"] < args.max_cambios:
                salida.update(v)
                break
        traer(host, port, args.name, destino, dir_remoto(host, port, args.name))
    except Exception as exc:                            # noqa: BLE001
        salida["detalle"] = str(exc)
        log(f"  FALLO: {exc}")
        if host:
            try:
                traer(host, port, args.name, destino,
                      dir_remoto(host, port, args.name))
            except Exception:                           # noqa: BLE001
                pass
    finally:
        vivida = time.time() - t0
        salida["gasto"] = precio * vivida / 3600
        salida["epocas"] = len(epocas_aqui)
        try:
            V.destruir(iid)
            log(f"  instancia {iid} destruida ({salida['motivo']}). "
                f"{vivida / 60:.1f} min, {salida['gasto']:.4f} $")
        except Exception as exc:                        # noqa: BLE001
            log(f"  AVISO GRAVE: no pude destruir {iid}: {exc}\n"
                f"  SIGUE FACTURANDO:\n"
                f"    python3 {LANZADOR}/scripts/vast_instance.py destroy {iid} --yes")
            avisar(f"GRAVE: no pude destruir la instancia {iid} de Vast. SIGUE "
                   f"FACTURANDO. Destruyela: vast_instance.py destroy {iid} --yes")
    if epocas_aqui:
        salida["mediana"] = statistics.median(epocas_aqui)
        salida["epocas"] = len(epocas_aqui)
    return salida


def _epocas(destino: Path) -> list:
    f = destino / "metrics.jsonl"
    if not f.exists():
        return []
    out = []
    for l in f.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Entrena un run en Vast, cambiando de maquina si se vuelve lenta")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", default="dirty1000-80px-16px-r20260827")
    ap.add_argument("--network", default="fov16-optimo")
    ap.add_argument("--recipe", default="plan40")
    ap.add_argument("--epochs", type=int, default=300,
                    help="tope de epocas (guarda). Quien para de verdad es --patience")
    ap.add_argument("--patience", type=int, default=20,
                    help="epocas sin mejorar antes de parar")
    ap.add_argument("--continuar", action="store_true")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--max-cpus", type=int, default=32)
    ap.add_argument("--ram-gb", type=float, default=8.0)
    ap.add_argument("--max-precio", type=float, default=0.15, help="$/h")
    ap.add_argument("--disco-gb", type=float, default=12.0)
    ap.add_argument("--cada", type=int, default=60)
    ap.add_argument("--horas-max", type=float, default=8.0)
    ap.add_argument("--presupuesto", type=float, default=5.0,
                    help="techo DURO de gasto en $; por encima no se alquila mas")
    ap.add_argument("--max-cambios", type=int, default=6)
    ap.add_argument("--aviso-cada", type=float, default=1.0, help="horas entre avisos")
    ap.add_argument("--prefijo", default="tv-")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    ctx = preflight(args)
    destino = (ctx["store"].path(args.name) if ctx["continuar"]
               else ctx["store"].destino(args.name))
    if not args.yes and not V.confirmar(
            f"Entrenar '{args.name}' en Vast con techo de {args.presupuesto:.2f} $?"):
        return 1

    estado = {"gasto": 0.0, "gasto_vivo": 0.0, "mejor": None, "cambios": 0,
              "instancias": [], "t_fin": time.time() + args.horas_max * 3600,
              "ultimo_aviso": time.time()}
    historia: list = []

    while True:
        # ⚠ Se decide ANTES de alquilar, no despues: descubrir que no cabe con la
        # maquina ya encendida es justo el gasto que el techo existe para evitar.
        if estado["gasto"] >= args.presupuesto * 0.9:
            log(f"\nNo alquilo otra: llevo {estado['gasto']:.3f} $ y el techo es "
                f"{args.presupuesto:.2f} $ (margen del 10 % para no pasarme).")
            break
        r = una_maquina(args, ctx, destino, estado)
        estado["gasto"] += r["gasto"]
        historia.append(r)
        if r["mediana"]:
            estado["mejor"] = min(estado["mejor"] or r["mediana"], r["mediana"])
        # a partir de aqui SIEMPRE se continua: ya hay pesos que subir
        ctx["continuar"] = (destino / "last.pt").exists()

        if r["motivo"] in ("done", "presupuesto", "plazo"):
            log(f"\n{r['detalle']}")
            break
        if r["motivo"] == "fallo":
            estado["cambios"] += 1
            if estado["cambios"] > args.max_cambios:
                log(f"\n{estado['cambios']} maquinas seguidas con problemas; paro.")
                break
            log(f"  -> pruebo con otra maquina ({estado['cambios']}/{args.max_cambios})")
            continue
        estado["cambios"] += 1
        log(f"  -> cambio de maquina: {r['detalle']} "
            f"({estado['cambios']}/{args.max_cambios})")
        avisar(f"cambio de maquina en '{args.name}': {r['detalle']}. "
               f"Llevo {len(_epocas(destino))} epocas y {estado['gasto']:.3f} $")
        if estado["cambios"] > args.max_cambios:
            log("\nDemasiados cambios de maquina; paro.")
            break

    filas = _epocas(destino)
    ultima = filas[-1] if filas else None
    resumen = (f"'{args.name}': {len(filas)} epocas, "
               f"{len(historia)} maquina(s), {estado['gasto']:.3f} $")
    if ultima:
        resumen += (f", f1={ultima['val']['f1']:.3f} "
                    f"val_loss={ultima['val']['loss']:.4f}")
    log(f"\n{resumen}")
    for i, h in enumerate(historia, 1):
        log(f"  {i}. {h['vcpu']} vCPU · {h['epocas']} epocas · "
            f"mediana {h['mediana'] and round(h['mediana'])} s · "
            f"{h['gasto']:.4f} $ · salio por {h['motivo']}")
    log(f"\n  el run esta en: {destino}")
    for f in TRAER:
        pth = destino / f
        log(f"    {'ok ' if pth.exists() else '-- '} {f}"
            + (f"  ({pth.stat().st_size} B)" if pth.exists() else ""))
    avisar(f"entrenamiento terminado. {resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
