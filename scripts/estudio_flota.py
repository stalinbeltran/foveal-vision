#!/usr/bin/env python3
"""Un recorrido repartido entre varias maquinas alquiladas, y el coste medido.

Que problema resuelve
---------------------
Un estudio de este proyecto es "N valores de un eje x M semillas", y hasta hoy
se corria en secuencia en una sola maquina: el recorrido `p40-lr-L4` fueron
20 runs y **36,9 h de reloj**. Los runs son independientes entre si, asi que ese
tiempo era una decision, no una necesidad.

Dos formas de repartir, y NO son equivalentes
---------------------------------------------
`--reparto seed` (por defecto) -- una maquina por SEMILLA, cada una corre todos
los valores del eje para la suya. La semilla es el eje replica, asi que cada
maquina mide TODOS los valores: si una maquina es mas lenta, mas vieja o de otra
familia de CPU, esa rareza entra por igual en todo lo que se compara. Es un
diseño por BLOQUES y el efecto de la maquina se cancela en la comparacion.

`--reparto run` -- una maquina por PUNTO (valor x semilla). Es el maximo
paralelismo: el reloj pasa a ser el de UN run, no el de una cadena de runs. El
efecto de la maquina ya no se cancela, pero tampoco se alinea con el eje: queda
repartido al azar entre las observaciones, asi que **añade ruido sin sesgar** a
ningun valor. Con pocas semillas ese reparto al azar puede desequilibrarse por
suerte, y por eso no es el modo por defecto.

⚠ Lo que NO se ofrece, a proposito: una maquina por VALOR DEL EJE. Ahi la
maquina quedaria confundida con la respuesta -- un `lr` podria ganar por haberle
tocado el host bueno -- y eso no se arregla despues con estadistica.

Maquinas SIEMPRE distintas
--------------------------
`elegir_ofertas_distintas` (en el lanzador) coge una oferta por `machine_id`
aunque la siguiente cueste mas. Vast publica varias ofertas por maquina fisica
-una por GPU libre- asi que "las N mas baratas" son a menudo N replicas del
mismo host: comparten CPU, disco y suerte, y si ese host se cae se lleva varios
lotes a la vez.

Y el que falla queda apuntado
-----------------------------
Un host que falla vuelve a salir en el catalogo mañana, y mas barato que el
resto, asi que la eleccion por precio vuelve a caer en el. Por eso el fallo se
apunta en `vast-bloqueadas.json` del repo del lanzador, que **se commitea**: la
maquina de control es efimera y lo que no esta en el remoto no existe.

QUE SE APUNTA (y que no, que es igual de importante):

- SI: no llega a arrancar, sshd no contesta, la subida falla, la instalacion
  falla, el proceso muere sin dejar codigo de salida, se agota el plazo. Todo
  eso es la maquina.
- NO: el entrenamiento arranca y termina con puntos fallidos. Eso es codigo o
  dato, se repetiria en cualquier maquina, y bloquear hosts por ello vaciaria el
  mercado sin arreglar nada.

El coste, medido y no estimado
------------------------------
`flota.json` guarda por maquina el desglose que hace falta para decidir si un
reparto compensa: `arranque_s`, `subida_s`, `instalacion_s` (el PEAJE, que se
paga entero por maquina y por eso crece con el numero de maquinas) y
`entrenamiento_s` (el trabajo, que es el mismo se reparta como se reparta).
Comparar dos repartos es comparar esas dos columnas.

    python3 scripts/estudio_flota.py --sweep lr-alto-L4 --dry-run
    python3 scripts/estudio_flota.py --sweep lr-alto-L4 --reparto seed
    python3 scripts/estudio_flota.py --sweep lr-alto-L4-b --reparto run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANZADOR = ROOT.parent / "digital-ocean-dropplet-auto-launching"
sys.path.insert(0, str(ROOT / "src"))
if not (LANZADOR / "scripts" / "vast_instance.py").exists():
    raise SystemExit(
        f"\nERROR: no esta el lanzador en {LANZADOR}.\n"
        "  clonalo:  git clone https://github.com/stalinbeltran/"
        "digital-ocean-dropplet-auto-launching.git\n"
    )
sys.path.insert(0, str(LANZADOR / "scripts"))

import vast_instance as V                          # noqa: E402
from fv.sweeps.runner import point_run_name        # noqa: E402
from fv.sweeps.spec import expand_points           # noqa: E402
from fv.sweeps.store import SweepStore             # noqa: E402

# Lo que viaja a la maquina. Nada mas: son ordenadores de desconocidos alquilados
# por minutos, y ahi no va ningun secreto (CLAUDE.md del lanzador, "Vast.ai").
ENVIA = ["src", "scripts", "configs", "pyproject.toml"]
EXCLUYE = {"__pycache__", ".pytest_cache", ".venv", ".git", "node_modules"}

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

_impresion = threading.Lock()
_registro = threading.Lock()      # bloquear_maquina lee-modifica-escribe un fichero
_reparto = threading.Lock()


def log(msg: str = "") -> None:
    with _impresion:
        print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(2)


# ---------------------------------------------------------------- la particion


def particion(valid: list, modo: str, seeds: list, sweep: str) -> list:
    """Que puntos corre cada maquina. Es la unica decision de diseño del script.

    Devuelve lotes {etiqueta, puntos (indices GLOBALES), descripcion}. Los
    indices son globales a proposito: es lo que hace que los runs de todas las
    maquinas se llamen igual que si hubieran corrido aqui de corrido, y por eso
    se juntan luego en un solo recorrido.
    """
    if modo == "seed":
        lotes = []
        for s in seeds:
            idx = [i for i, p in enumerate(valid) if p["overrides"].get("seed") == s]
            if idx:
                lotes.append({"etiqueta": f"s{s}", "puntos": idx,
                              "descripcion": f"semilla {s} ({len(idx)} puntos)"})
        return lotes
    if modo == "run":
        return [{"etiqueta": f"p{i}", "puntos": [i],
                 "descripcion": point_run_name(sweep, i, p["overrides"])}
                for i, p in enumerate(valid)
                if p["overrides"].get("seed") in seeds or not seeds]
    die(f"reparto '{modo}' no existe: usa 'seed' o 'run'")


# ------------------------------------------------------------------- el payload


def construir_payload(sweep: str, dataset: str) -> Path:
    """El tar que se sube: codigo + el recorrido + el dataset YA EXTRAIDO.

    El dataset se extrae UNA VEZ aqui y se manda hecho, en vez de que cada
    maquina lo extraiga: asi las N maquinas entrenan sobre el MISMO fichero,
    byte a byte, y comparar entre maquinas significa algo. Extraerlo en cada una
    seria pedir que N extracciones coincidan: promesa mas fuerte, ganancia
    ninguna.
    """
    npz = ROOT / "data" / "window-datasets" / dataset / "windows.npz"
    if not npz.exists():
        die(f"falta {npz}.\n"
            f"  El dataset de ventanas no esta extraido, y sin el no hay que entrenar.")
    tmp = Path(tempfile.mkdtemp(prefix="flota-")) / "payload.tar.gz"

    def filtro(info: tarfile.TarInfo) -> "tarfile.TarInfo | None":
        return None if set(Path(info.name).parts) & EXCLUYE else info

    with tarfile.open(tmp, "w:gz") as tar:
        for nombre in ENVIA:
            origen = ROOT / nombre
            if not origen.exists():
                die(f"falta {origen}, que hace falta para entrenar")
            tar.add(str(origen), arcname=nombre, filter=filtro)
        tar.add(str(ROOT / "sweeps" / sweep), arcname=f"sweeps/{sweep}", filter=filtro)
        tar.add(str(npz.parent), arcname=f"data/window-datasets/{dataset}",
                filter=filtro)
    return tmp


# --------------------------------------------------------- reparto de maquinas


class Maquinas:
    """Da ofertas de maquinas DISTINTAS, y nunca repite una ya usada.

    Se pide un colchon de repuestos de una vez (una sola consulta al catalogo) y
    se van entregando bajo cerrojo: asi un reintento tras un fallo no puede
    volver a caer ni en la maquina que acaba de fallar ni en la de otro lote.
    """

    def __init__(self, cuantas: int, repuestos: int, cpus: int, max_cpus: int,
                 min_ram: float, max_price: float, cpu: str = ""):
        self.pool = V.elegir_ofertas_distintas(
            cuantas + repuestos, cpus=cpus, max_cpus=max_cpus,
            min_ram_gb=min_ram, max_price=max_price, cpu=cpu)
        self.entregadas: list = []

    def siguiente(self) -> "dict | None":
        with _reparto:
            if not self.pool:
                return None
            o = self.pool.pop(0)
            self.entregadas.append(o)
            return o


def apuntar_fallo(oferta: dict, motivo: str, etiqueta: str) -> None:
    mid = oferta.get("machine_id")
    if mid is None:
        return
    with _registro:
        r = V.bloquear_maquina(mid, motivo, etiqueta)
    log(f"    maquina {mid} APUNTADA en la lista negra ({r['fallos']} fallo(s)): {motivo}")


# ---------------------------------------------------------- un lote, una maquina


def correr_lote(lote: dict, oferta: dict, payload: Path, sweep: str, hilos: int,
                plazo_s: int, cada_s: int, disco_gb: float) -> dict:
    """Alquila, instala, corre los puntos del lote, se trae los runs y destruye.

    La destruccion va en `finally` y no es opcional: si algo revienta se pierde
    la medida, no el dinero.
    """
    tag = lote["etiqueta"]
    etiqueta = f"estudio-{sweep}-{tag}"
    precio = float(oferta.get("dph_total") or 0)
    maquina = V.resumen_maquina(oferta)
    t0 = time.time()
    resultado = {"lote": tag, "puntos": lote["puntos"], "que": lote["descripcion"],
                 "oferta": oferta.get("id"), "machine_id": oferta.get("machine_id"),
                 "maquina": maquina, "usd_hora": round(precio, 5), "ok": False,
                 "error": None, "epocas": None, "s_por_epoca": None}

    log(f"[{tag}] oferta {oferta.get('id')} maquina {oferta.get('machine_id')} "
        f"{maquina['vcpu']:g} vCPU {maquina['ram_gb']:g} GB {precio:.4f} $/h "
        f"{maquina['ubicacion']}")

    iid = V.alquilar(oferta, etiqueta, V.cfg("VAST_IMAGE"), disco_gb)
    resultado["instancia"] = iid
    log(f"[{tag}] instancia {iid} alquilada")
    try:
        info = V.esperar_estado(iid, int(V.cfg("VAST_BOOT_TIMEOUT")))
        estado = (info.get("actual_status") or info.get("cur_state") or "?").lower()
        if estado != "running":
            raise RuntimeError(f"la instancia acabo en '{estado}', no arranco")
        host, port = V.ssh_destino(info)
        if not V.esperar_ssh(host, port):
            raise RuntimeError(f"sshd no contesto en {host}:{port}")
        resultado["arranque_s"] = round(time.time() - t0, 1)
        log(f"[{tag}] SSH listo en {host}:{port} ({resultado['arranque_s'] / 60:.1f} min), "
            f"subiendo {payload.stat().st_size / 1e6:.1f} MB")

        t_sub = time.time()
        with payload.open("rb") as fh:
            p = subprocess.run(V.ssh_command(host, port) + ["cat > /root/payload.tar.gz"],
                               stdin=fh, timeout=900)
        if p.returncode != 0:
            raise RuntimeError("no pude subir el payload")
        if V.ssh_script(host, port,
                        "set -eu\nmkdir -p /root/bench\n"
                        "tar -xzf /root/payload.tar.gz -C /root/bench\n"
                        "rm -f /root/payload.tar.gz\n", 600) != 0:
            raise RuntimeError("el payload subio pero no se pudo desempaquetar")
        resultado["subida_s"] = round(time.time() - t_sub, 1)

        log(f"[{tag}] instalando (torch tarda minutos)...")
        t_inst = time.time()
        if V.ssh_script(host, port, INSTALL, 2400) != 0:
            raise RuntimeError("fallo la instalacion de dependencias")
        resultado["instalacion_s"] = round(time.time() - t_inst, 1)
        # El PEAJE: todo lo que hay que esperar antes de la primera epoca. Se paga
        # entero POR MAQUINA, asi que es la parte del coste que crece al repartir
        # mas fino -- y es exactamente lo que hay que mirar para decidir si
        # compensa.
        resultado["peaje_s"] = round(time.time() - t0, 1)
        log(f"[{tag}] listo en {resultado['peaje_s'] / 60:.1f} min "
            f"(arranque {resultado['arranque_s'] / 60:.1f} + subida "
            f"{resultado['subida_s']:.0f}s + instalacion "
            f"{resultado['instalacion_s']:.0f}s). Entrenando...")

        # Desacoplado (`setsid`) a proposito: si la sesion de SSH se corta -y en
        # una maquina alquilada se corta- el entrenamiento sigue, y el vigilante
        # de abajo lo vuelve a encontrar. El codigo de salida se deja en un
        # fichero porque es la unica forma de distinguir "sigue corriendo" de
        # "termino" sin mantener viva la sesion que puede caerse.
        puntos = ",".join(str(i) for i in lote["puntos"])
        arranque = (
            "set -eu\ncd /root/bench\nrm -f /root/estudio.rc /root/estudio.log\n"
            f"export OMP_NUM_THREADS={hilos} MKL_NUM_THREADS={hilos}\n"
            "setsid nohup .venv/bin/python scripts/estudio_lote.py "
            f"--sweep {sweep} --puntos {puntos} --que '{tag}' "
            "--rc /root/estudio.rc > /root/estudio.log 2>&1 < /dev/null &\n"
            "sleep 2\necho lanzado\n"
        )
        if V.ssh_script(host, port, arranque, 300) != 0:
            raise RuntimeError("no pude lanzar el entrenamiento")

        t_ent = time.time()
        rc, ultima = None, ""
        while True:
            time.sleep(cada_s)
            code, salida = V.ssh_capture(
                host, port,
                "set +e\n"
                "echo RC=$(cat /root/estudio.rc 2>/dev/null)\n"
                "echo EPOCAS=$(cat /root/bench/runs/*/metrics.jsonl 2>/dev/null | wc -l)\n"
                "echo HECHOS=$(ls -d /root/bench/runs/*/ 2>/dev/null | wc -l)\n"
                "tail -n 2 /root/estudio.log 2>/dev/null\n", timeout=180)
            if code != 0:
                # una sonda fallida no es la maquina caida: se reintenta hasta el plazo
                log(f"[{tag}] la sonda de SSH no contesto, se reintenta")
                continue
            campos = {}
            for linea in salida.splitlines():
                if "=" in linea and linea.split("=", 1)[0] in ("RC", "EPOCAS", "HECHOS"):
                    k, v = linea.split("=", 1)
                    campos[k] = v.strip()
            epocas = int(campos.get("EPOCAS") or 0)
            transcurrido = time.time() - t_ent
            spe = transcurrido / epocas if epocas else None
            resultado["epocas"] = epocas
            resultado["s_por_epoca"] = round(spe, 1) if spe else None
            # La ultima linea del log REMOTO viaja hasta aqui a proposito: es
            # donde `estudio_lote.py` dice "punto 2/3 terminado: <run>", y sin
            # traerla el vigilante solo sabe contar directorios.
            ultima = salida.strip().splitlines()[-1] if salida.strip() else ""
            eco = ultima.split("  ", 1)[-1].strip() if "punto" in ultima else ""
            log(f"[{tag}] {transcurrido / 60:5.1f} min · {epocas} epocas · "
                f"{campos.get('HECHOS', '?')} runs · "
                + (f"{spe:.1f} s/epoca" if spe else "aun sin epoca")
                + (f" · {eco}" if eco else ""))
            if campos.get("RC"):
                rc = int(campos["RC"])
                break
            if transcurrido > plazo_s:
                raise RuntimeError(
                    f"plazo agotado ({plazo_s / 3600:.1f} h) con {epocas} epocas hechas")

        resultado["entrenamiento_s"] = round(time.time() - t_ent, 1)
        log(f"[{tag}] entrenamiento terminado (rc={rc}) en "
            f"{resultado['entrenamiento_s'] / 60:.1f} min. Recogiendo los runs...")

        # Se recogen SIEMPRE, tambien con rc != 0: los puntos que si terminaron
        # son medidas buenas, y tirarlas obligaria a repetirlas.
        if V.ssh_script(host, port,
                        "set -eu\ncd /root/bench\ntar -czf /root/runs.tar.gz runs\n",
                        600) != 0:
            raise RuntimeError("no pude empaquetar los runs en la maquina")
        local = Path(tempfile.mkdtemp(prefix=f"runs-{tag}-")) / "runs.tar.gz"
        with local.open("wb") as fh:
            p = subprocess.run(V.ssh_command(host, port) + ["cat /root/runs.tar.gz"],
                               stdout=fh, timeout=900)
        if p.returncode != 0 or local.stat().st_size == 0:
            raise RuntimeError("no pude traerme los runs de la maquina")
        with tarfile.open(local, "r:gz") as tar:
            tar.extractall(ROOT)
            # Los nombres se leen del TAR, no del directorio local: `runs/`
            # acumula los de todos los estudios anteriores.
            traidos = sorted({Path(m.name).parts[1] for m in tar.getmembers()
                              if len(Path(m.name).parts) > 1})
        resultado["runs"] = traidos
        log(f"[{tag}] {len(traidos)} runs traidos: {', '.join(traidos)}")
        resultado["ok"] = rc == 0
        if rc != 0:
            # El entrenamiento CORRIO: no es culpa de la maquina, asi que no se
            # apunta. Se dice y se sigue.
            resultado["error"] = (f"el lote termino con rc={rc}: algun punto "
                                  f"fallo. Ultima linea: {ultima}")
        return resultado
    finally:
        vivida = time.time() - t0
        resultado["segundos_vivida"] = round(vivida, 1)
        resultado["usd"] = round(precio * vivida / 3600, 5)
        try:
            V.destruir(iid)
            log(f"[{tag}] instancia {iid} destruida. Vivio {vivida / 60:.1f} min, "
                f"{resultado['usd']:.4f} $")
        except Exception as exc:                       # noqa: BLE001
            log(f"[{tag}] AVISO GRAVE: no pude destruir {iid}: {exc}\n"
                f"    SIGUE FACTURANDO. Destruyela ya:\n"
                f"    python3 {LANZADOR}/scripts/vast_instance.py destroy {iid} --yes")


def lote_con_reintentos(lote: dict, maquinas: Maquinas, payload: Path, sweep: str,
                        hilos: int, plazo_s: int, cada_s: int, disco_gb: float,
                        intentos: int) -> dict:
    tag = lote["etiqueta"]
    ultimo = {"lote": tag, "puntos": lote["puntos"], "ok": False, "error": "sin intentos"}
    for intento in range(1, intentos + 1):
        oferta = maquinas.siguiente()
        if oferta is None:
            ultimo["error"] = ("no quedan maquinas distintas libres para reintentar; "
                               "sube --repuestos o afloja las condiciones")
            break
        try:
            r = correr_lote(lote, oferta, payload, sweep, hilos, plazo_s, cada_s,
                            disco_gb)
            r["intento"] = intento
            if r["ok"] or r.get("runs"):
                return r
            ultimo = r
        except Exception as exc:                        # noqa: BLE001
            motivo = f"{type(exc).__name__}: {exc}"[:200]
            log(f"[{tag}] FALLO en el intento {intento}/{intentos}: {motivo}")
            apuntar_fallo(oferta, motivo, f"estudio-{sweep}-{tag}")
            ultimo = {"lote": tag, "puntos": lote["puntos"], "ok": False,
                      "error": motivo, "oferta": oferta.get("id"),
                      "machine_id": oferta.get("machine_id"), "intento": intento,
                      "bloqueada": True}
            if intento < intentos:
                log(f"[{tag}] reintentando en OTRA maquina...")
    return ultimo


# ------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", required=True, help="recorrido ya creado (sweeps/<name>)")
    ap.add_argument("--reparto", choices=("seed", "run"), default="seed",
                    help="seed: una maquina por semilla (bloques, por defecto). "
                         "run: una maquina por punto (maximo paralelismo)")
    ap.add_argument("--seeds", default="", help="semillas a correr (por defecto, todas)")
    ap.add_argument("--cpus", type=int, default=8, help="vCPU efectivas minimas")
    ap.add_argument("--max-cpus", type=int, default=32)
    ap.add_argument("--min-ram", type=float, default=8.0, metavar="GB")
    ap.add_argument("--cpu", default="",
                    help="exige esta CPU (subcadena, p.ej. 'E5-26'). MEDIDO: dentro "
                         "de la familia Xeon E5-26xx v3/v4 el entrenamiento sale "
                         "IDENTICO bit a bit entre maquinas, y diverge al cruzar de "
                         "familia (hasta 0,0457 en f1). Fijarla convierte el ruido "
                         "de maquina en cero -- ver docs/plan-lr-alto.md §7.4")
    ap.add_argument("--max-price", type=float, default=None, metavar="USD_HORA")
    ap.add_argument("--disk", type=float, default=16.0, metavar="GB")
    ap.add_argument("--hilos", type=int, default=8,
                    help="hilos de torch, IGUALES en todas: una maquina con mas "
                         "nucleos no debe entrenar distinto que otra")
    ap.add_argument("--horas-max", type=float, default=6.0,
                    help="plazo por lote; al agotarse se destruye la maquina")
    ap.add_argument("--cada", type=int, default=120, help="segundos entre sondas")
    ap.add_argument("--repuestos", type=int, default=3,
                    help="maquinas distintas de reserva para los reintentos")
    ap.add_argument("--intentos", type=int, default=2, help="intentos por lote")
    ap.add_argument("--paralelo", type=int, default=0,
                    help="cuantas maquinas a la vez (0 = todas)")
    ap.add_argument("--dry-run", action="store_true",
                    help="ensena que maquinas se cogerian y no alquila nada")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    V.load_env()
    store = SweepStore()
    if not store.exists(args.sweep):
        die(f"no existe el recorrido '{args.sweep}'. Crealo primero con su script.")
    spec = store.spec(args.sweep)
    dataset = spec["window_dataset"]
    valid, _ = expand_points(spec, spec["base_network_value"])
    todas = sorted({p["overrides"].get("seed") for p in valid} - {None})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else todas
    lotes = particion(valid, args.reparto, seeds, args.sweep)
    if not lotes:
        die(f"el reparto '{args.reparto}' no produjo ningun lote")

    log(f"Recorrido '{args.sweep}': {len(valid)} puntos, dataset {dataset}")
    log(f"Eje: {json.dumps(spec['space'])}")
    log(f"Reparto '{args.reparto}': {len(lotes)} maquinas")
    for l in lotes:
        log(f"   {l['etiqueta']:>5} -> puntos {l['puntos']}  ({l['descripcion']})")

    tope = args.max_price or V.limite_precio()
    maquinas = Maquinas(len(lotes), args.repuestos, args.cpus, args.max_cpus,
                        args.min_ram, tope, args.cpu)
    log(f"\n{len(maquinas.pool)} maquinas DISTINTAS disponibles "
        f"({len(lotes)} a usar + {len(maquinas.pool) - len(lotes)} de repuesto):")
    log("  " + V.cabecera_ofertas())
    for o in maquinas.pool:
        log(f"  {V.oferta_fila(o)}   maquina {o.get('machine_id')}")
    coste = sum(float(o.get("dph_total") or 0) for o in maquinas.pool[:len(lotes)])
    log(f"\nCoste maximo si las {len(lotes)} vivieran {args.horas_max:g} h: "
        f"{coste * args.horas_max:.2f} $ ({coste:.4f} $/h entre todas).")

    if args.dry_run:
        log("--dry-run: no se alquila nada.")
        return 0
    if not args.yes and not V.confirmar("¿Lanzo la flota?"):
        log("Cancelado. No se ha alquilado nada.")
        return 1

    payload = construir_payload(args.sweep, dataset)
    log(f"Payload listo: {payload.stat().st_size / 1e6:.1f} MB "
        f"(codigo + recorrido + dataset ya extraido)")

    t0 = time.time()
    obreros = args.paralelo or len(lotes)
    with ThreadPoolExecutor(max_workers=obreros) as pool:
        futuros = [pool.submit(lote_con_reintentos, l, maquinas, payload, args.sweep,
                               args.hilos, int(args.horas_max * 3600), args.cada,
                               args.disk, args.intentos) for l in lotes]
        resultados = [f.result() for f in futuros]

    reloj = time.time() - t0
    gasto = sum(float(r.get("usd") or 0) for r in resultados)
    buenas = [r for r in resultados if r.get("ok")]
    peaje = sum(float(r.get("peaje_s") or 0) for r in resultados)
    trabajo = sum(float(r.get("entrenamiento_s") or 0) for r in resultados)
    vividas = sum(float(r.get("segundos_vivida") or 0) for r in resultados)
    reporte = {
        "recorrido": args.sweep, "dataset": dataset, "cuando": V.ahora_iso(),
        "reparto": args.reparto, "maquinas": len(lotes), "cpu": args.cpu or None,
        "reloj_min": round(reloj / 60, 1), "usd": round(gasto, 4),
        "hilos": args.hilos,
        # El desglose que decide si un reparto compensa: el peaje se paga entero
        # por maquina (crece al repartir mas fino), el trabajo no.
        "peaje_min": round(peaje / 60, 1), "trabajo_min": round(trabajo / 60, 1),
        "maquina_min": round(vividas / 60, 1),
        "peaje_pct": round(100 * peaje / vividas, 1) if vividas else None,
        "lotes": resultados,
    }
    destino = ROOT / "sweeps" / args.sweep / "flota.json"
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")

    log("\n" + "=" * 68)
    log(f"  {len(buenas)}/{len(lotes)} lotes completos en {reloj / 60:.1f} min de "
        f"RELOJ. Gastado: {gasto:.4f} $   (reparto '{args.reparto}')")
    log(f"  Maquina-minutos: {vividas / 60:.1f}  =  peaje {peaje / 60:.1f} "
        f"({reporte['peaje_pct']}%) + trabajo {trabajo / 60:.1f}")
    for r in resultados:
        marca = "ok " if r.get("ok") else "FALLO"
        log(f"  {marca} {r['lote']:>5}: maquina {r.get('machine_id')} · "
            f"{r.get('epocas') or 0} epocas · "
            + (f"{r['s_por_epoca']:.1f} s/epoca" if r.get("s_por_epoca") else "-")
            + (f" · {r['error']}" if r.get("error") else ""))
    log(f"  Reporte: {destino.relative_to(ROOT)}")
    log("  Comprueba que no queda nada vivo:")
    log(f"    python3 {LANZADOR}/scripts/vast_instance.py list")
    log("=" * 68)
    return 0 if len(buenas) == len(lotes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
