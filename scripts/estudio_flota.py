#!/usr/bin/env python3
"""Uno o varios recorridos repartidos entre maquinas alquiladas, y el coste medido.

Que problema resuelve
---------------------
Un estudio de este proyecto es "N valores de un eje x M semillas", y hasta el
2026-08-23 se corria en secuencia en una sola maquina: el recorrido `p40-lr-L4`
fueron 20 runs y **36,9 h de reloj**. Los runs son independientes entre si, asi
que ese tiempo era una decision, no una necesidad.

Dos formas de repartir, y NO son equivalentes
---------------------------------------------
`--reparto seed` (por defecto) -- una maquina por SEMILLA de cada recorrido, y
cada una corre todos los valores del eje para la suya. La semilla es el eje
replica, asi que cada maquina mide TODOS los valores: si una maquina es mas
lenta, mas vieja o de otra familia de CPU, esa rareza entra por igual en todo lo
que se compara. Es un diseño por BLOQUES y el efecto de la maquina se cancela en
la comparacion.

`--reparto run` -- una maquina por PUNTO (valor x semilla). Es el maximo
paralelismo: el reloj pasa a ser el de UN run, no el de una cadena de runs. El
efecto de la maquina ya no se cancela, pero tampoco se alinea con el eje: queda
repartido al azar entre las observaciones, asi que **añade ruido sin sesgar** a
ningun valor. Con pocas semillas ese reparto al azar puede desequilibrarse por
suerte, y por eso no es el modo por defecto.

⚠ Lo que NO se ofrece, a proposito: una maquina por VALOR DEL EJE. Ahi la
maquina quedaria confundida con la respuesta -- un `lr` podria ganar por haberle
tocado el host bueno -- y eso no se arregla despues con estadistica.

VARIOS recorridos en una sola flota
-----------------------------------
`--sweep` se repite. Los recorridos comparten UN pozo de maquinas, y esa es la
razon de que exista: dos procesos consultando el catalogo por su cuenta eligen
por precio, o sea que eligen **las mismas** ofertas, y el segundo en alquilar se
encuentra la maquina ocupada. Con un pozo unico bajo cerrojo eso no puede pasar.

Exigen el MISMO dataset de ventanas, y el mensaje lo dice si no: el payload lleva
un `windows.npz` y mezclarlos silenciosamente entrenaria la mitad del estudio
sobre otro dato.

La CRIBA de velocidad: alquilar de mas y quedarse con las rapidas
------------------------------------------------------------------
MEDIDO el 2026-08-23 (docs/plan-lr-alto.md §6.3): entre tres maquinas del mismo
catalogo el s/epoca fue 36,3 · 50,5 · 53,3 -- un factor **1,47**, y la de MAS
vCPU (16) fue la mas LENTA. El numero de nucleos y el precio, que es por lo que
se filtra al elegir oferta, no predicen la velocidad de este trabajo.

`--criba K` alquila `K` maquinas de mas, le pide a cada una unos segundos de
entrenamiento de verdad (`scripts/sonda_velocidad.py`), y se queda con las mas
rapidas. Las descartadas se destruyen ahi mismo: pagan su peaje (~3,5 min) y se
van. El criterio esta escrito ANTES y sale en el log con sus numeros:

1. se descarta toda maquina por debajo de `--umbral-velocidad` (0,75) x la
   MEDIANA de la cohorte -- "significativamente mas lenta" es esto y no una
   impresion;
2. de las que sobreviven se queda con las `N` mas rapidas;
3. si sobreviven menos de `N`, se completan con las mejores descartadas y **se
   dice en voz alta**: una maquina lenta es peor que ninguna solo hasta que la
   alternativa es un punto sin medir.

⚠ Por que la criba no contamina el resultado: se selecciona por VELOCIDAD, y con
`--cpu` fijando la familia de CPU el entrenamiento sale **identico bit a bit**
entre maquinas de esa familia (medido, plan-lr-alto §7.4). O sea que se elige
sobre una variable que no mueve la respuesta. **Sin `--cpu` esa garantia no
existe** y la criba pasa a seleccionar sobre algo que si puede moverla; por eso
`--criba` avisa cuando se usa sin fijar CPU.

La VIGILANCIA: la velocidad de una maquina alquilada cambia sola
-----------------------------------------------------------------
Una maquina rapida a las 10:00 puede tener otro inquilino a las 11:00. El
vigilante no vuelve a correr la sonda -- eso robaria nucleos justo a lo que se
esta midiendo -- sino que **lee los tiempos por epoca del propio entrenamiento**,
que ya estan en `metrics.jsonl` y no cuestan nada:

  base    = mediana de las 3 primeras epocas de este run
  reciente= mediana de las 3 ultimas
  degradada si reciente/base > --umbral-degradacion (1,35) en DOS sondas seguidas

Dos sondas seguidas y no una: una epoca lenta suelta es ruido normal.

`--degradado avisar` (por defecto) lo dice y sigue. `--degradado abandonar`
destruye la maquina y reparte lo que le quedaba a otra -- lo cual solo es barato
porque el libro de a bordo esta en git (abajo).

⚠ Una maquina degradada NO se apunta en la lista negra. Un inquilino ajeno se va;
un host roto no. Bloquear por lentitud vaciaria el catalogo sin arreglar nada, y
la lista negra tiene escrito que es para maquinas que fallan, no que van lentas.

El LIBRO DE A BORDO: cada epoca, en git
----------------------------------------
`--git` hace que el vigilante se traiga de cada maquina, en cada sonda, los
ficheros pequenos del run -- `metrics.jsonl`, `status.json`, `config.json`,
`summary.json` -- y que un hilo aparte los commitee y los empuje. Con `--cada 60`
y epocas de 40-60 s eso es, en la practica, **una entrada por epoca**.

Los pesos (`*.pt`) NO van a git, y no es un olvido: `.gitignore` lo dice desde
siempre (`/runs/*/*.pt`) porque son ~700 KB por run y por epoca, y el repo se
comeria gigabytes por estudio. Lo que va es el resultado, que es lo que se lee.

Que compra eso, exactamente:

- **Nada de lo terminado se pierde.** Antes, una maquina que se caia al cuarto
  de sus cinco runs se llevaba tambien los tres que ya habia hecho: los runs solo
  bajaban al final, en un tar. Ahora bajan segun se escriben.
- **Relanzar continua.** Al arrancar, la flota mira `runs/` y salta todo punto
  cuyo `status.json` diga `done`. Un lote que se queda sin puntos pendientes ni
  siquiera alquila maquina. Es lo que convierte una caida en un rearranque barato
  en vez de en repetirlo todo.
- **Sobrevive a esta maquina.** El droplet de control es efimero y lo que no esta
  empujado no existe (CLAUDE.md). El libro en el remoto es la unica copia que
  aguanta que se rehaga el servidor.

⚠ Hasta donde llega: se reanuda por PUNTO, no por epoca. Un run cortado a mitad
se repite entero. Reanudar a media epoca pediria llevarse los pesos y el estado
de Adam, y ademas **cambiaria el experimento**: el flujo de numeros aleatorios del
dataloader no se retoma igual en otra maquina, asi que el run reanudado ya no
seria bit a bit el que se pidio (plan-lr-alto §7.4 mide que eso importa). Se
prefiere repetir un run a publicar uno que nadie puede reproducir.

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
  falla, la sonda de velocidad no devuelve nada, el proceso muere sin dejar
  codigo de salida, se agota el plazo. Todo eso es la maquina.
- NO: el entrenamiento arranca y termina con puntos fallidos. Eso es codigo o
  dato, se repetiria en cualquier maquina, y bloquear hosts por ello vaciaria el
  mercado sin arreglar nada. Tampoco la lentitud sobrevenida (arriba).

El coste, medido y no estimado
------------------------------
`flota.json` guarda por maquina el desglose que hace falta para decidir si un
reparto compensa: `arranque_s`, `subida_s`, `instalacion_s` (el PEAJE, que se
paga entero por maquina y por eso crece con el numero de maquinas) y
`entrenamiento_s` (el trabajo, que es el mismo se reparta como se reparta).
Comparar dos repartos es comparar esas dos columnas.

Y ANTES de alquilar: `--estimar` (o `--dry-run`, que lo incluye) imprime cuanto
va a tardar y cuanto va a costar, con la procedencia de cada coeficiente
(`scripts/estudio_estimar.py`).

    python3 scripts/estudio_flota.py --sweep bs5-L4 --estimar
    python3 scripts/estudio_flota.py --sweep bs5-L4 --dry-run
    python3 scripts/estudio_flota.py --sweep bs5-L4 --sweep nl5-L4 --sweep d5-L4 \
        --reparto seed --cpu 'E5-26' --criba 4 --git
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANZADOR = ROOT.parent / "digital-ocean-dropplet-auto-launching"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if not (LANZADOR / "scripts" / "vast_instance.py").exists():
    raise SystemExit(
        f"\nERROR: no esta el lanzador en {LANZADOR}.\n"
        "  clonalo:  git clone https://github.com/stalinbeltran/"
        "digital-ocean-dropplet-auto-launching.git\n"
    )
sys.path.insert(0, str(LANZADOR / "scripts"))

import estudio_estimar as EST                       # noqa: E402
import vast_instance as V                           # noqa: E402
from fv.sweeps.runner import point_run_name         # noqa: E402
from fv.sweeps.spec import expand_points            # noqa: E402
from fv.sweeps.store import SweepStore              # noqa: E402

# Lo que viaja a la maquina. Nada mas: son ordenadores de desconocidos alquilados
# por minutos, y ahi no va ningun secreto (CLAUDE.md del lanzador, "Vast.ai").
ENVIA = ["src", "scripts", "configs", "pyproject.toml"]
EXCLUYE = {"__pycache__", ".pytest_cache", ".venv", ".git", "node_modules"}

# Los ficheros del libro de a bordo: pequenos, textuales y los unicos que git
# quiere. Los pesos se quedan en la maquina (ver el docstring).
LIBRO = ("metrics.jsonl", "status.json", "config.json", "summary.json")

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

# Lo que lleva gastado la flota, apuntado al DESTRUIR y no al terminar un lote.
# Una maquina que muere por excepcion -fallo al preparar, degradacion, plazo
# agotado- tambien factura, y antes no aparecia en el total: el reporte decia
# menos de lo gastado, que es la direccion peligrosa del error.
GASTO: list = []

# Que etiqueta tiene reclamado cada destino SSH. MEDIDO el 2026-08-24: la API de
# Vast devolvio el MISMO `ssh_host:puerto` (ssh4.vast.ai:21482) para dos
# instancias recien alquiladas -- 48581482 y 48581483 -- mientras las dos estaban
# arrancando. Las dos hebras subieron el payload a la MISMA maquina, la segunda
# instalacion borro el tar de la primera, y el fallo se leyo como "el payload
# subio pero no se pudo desempaquetar": se destruyeron las dos maquinas y se
# apunto en la lista negra a dos hosts que no habian hecho nada.
#
# Que se rompio de verdad no fue eso, que era ruidoso, sino lo que habria pasado
# si la carrera hubiera salido al reves: dos LOTES entrenando en la misma maquina,
# compartiendo `/root/bench/runs/`, peleandose por los nucleos -- y otra maquina
# alquilada sin hacer nada y facturando. Eso no habria dado un error, habria dado
# numeros.
DESTINOS: dict = {}

_impresion = threading.Lock()
_registro = threading.Lock()      # bloquear_maquina lee-modifica-escribe un fichero
_reparto = threading.Lock()
_disco = threading.Lock()         # escribir el libro de a bordo en runs/
_destinos = threading.Lock()
_git = threading.Lock()


def log(msg: str = "") -> None:
    with _impresion:
        print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    raise SystemExit(2)


class Suplantada(RuntimeError):
    """El destino SSH que la API dio para esta instancia lleva a OTRA.

    NO se apunta en la lista negra, y es la parte que importa: la maquina no ha
    hecho nada mal -- quien se equivoco fue el catalogo al publicar el mismo
    `ssh_host:puerto` para dos instancias a la vez. Bloquear al host por esto
    seria castigar al inocente y, peor, ir vaciando el catalogo por un fallo que
    no esta en el.
    """


SELLO = "/root/.duenno-estudio"


def resolver_destino(iid: int, etiqueta: str, info: dict, intentos: int = 6,
                     espera_s: float = 10.0) -> tuple:
    """(host, port) que NADIE MAS tenga reclamado, releyendo a la API si hace falta.

    Es la primera de las dos defensas y la barata: si el destino que la API da ya
    lo tiene otro, se vuelve a preguntar -- el dato suele corregirse solo en unos
    segundos, porque es informacion de una instancia que aun esta arrancando.
    """
    ultimo = None
    for intento in range(1, intentos + 1):
        host, port = V.ssh_destino(info)
        if not host or not port:
            ultimo = f"la API no da destino SSH todavia ({host!r}:{port})"
        else:
            clave = f"{host}:{port}"
            with _destinos:
                duenno = DESTINOS.get(clave)
                if duenno is None or duenno == etiqueta:
                    DESTINOS[clave] = etiqueta
                    return host, port
            ultimo = (f"{clave} ya lo tiene reclamado [{duenno}]; la API esta "
                      f"publicando el mismo destino para dos instancias")
        if intento < intentos:
            log(f"    [{etiqueta}] {ultimo}. Repregunto a la API "
                f"({intento}/{intentos - 1})...")
            time.sleep(espera_s)
            try:
                info = V.instancia(iid)
            except Exception as exc:                    # noqa: BLE001
                ultimo = f"no pude releer la instancia: {type(exc).__name__}: {exc}"
    raise Suplantada(f"sin destino SSH propio tras {intentos} intentos: {ultimo}")


def soltar_destino(host: str, port: int, etiqueta: str) -> None:
    with _destinos:
        if DESTINOS.get(f"{host}:{port}") == etiqueta:
            del DESTINOS[f"{host}:{port}"]


def sellar(host: str, port: int, nonce: str, etiqueta: str,
           intentos: int = 8, espera_s: float = 15.0) -> None:
    """Escribe un sello en la maquina y lo vuelve a leer. REINTENTA el transporte.

    Es la segunda defensa contra la suplantacion, y la que de verdad cierra el
    agujero: el registro de destinos solo sabe lo que ESTE proceso ha repartido,
    asi que no puede distinguir "soy el primero" de "soy el equivocado". El sello
    no depende de creerse a nadie -- si dos hebras acaban en la misma maquina, la
    segunda pisa el fichero y la primera lo nota al releerlo.

    ⚠ Y hace de PUERTA DE ENTRADA, que es lo que obliga a reintentar. MEDIDO el
    2026-08-24 17:16: sin reintentos, 3 de las 5 primeras maquinas fallaron aqui
    con rc=255 y acabaron en la lista negra sin haber hecho nada. La causa es la
    que plan-lr-alto §6.4 dejo apuntada como sospecha sin poder medirla: **el
    banner de sshd llega antes que la clave**. `V.esperar_ssh` comprueba el
    banner, que NO es lo mismo que "SSH funciona"; el sello es el primer comando
    que necesita autenticarse de verdad, asi que se come esa carrera entera.

    De ahi la asimetria, que es deliberada:

    - `rc != 0` es **transporte**: la maquina todavia no acepta la clave. Se
      reintenta hasta `intentos`, y solo entonces se declara fallo suyo.
    - un sello que se lee y **no coincide** es una suplantacion: eso no mejora
      esperando, asi que se lanza a la primera.
    """
    ultimo = ""
    for intento in range(1, intentos + 1):
        code, salida = V.ssh_capture(
            host, port, f"set -eu\nprintf '%s' '{nonce}' > {SELLO}\ncat {SELLO}\n",
            timeout=180)
        leido = (salida or "").strip().splitlines()[-1].strip() if salida.strip() else ""
        if code == 0:
            if leido != nonce:
                raise Suplantada(f"el sello de {host}:{port} dice '{leido[:40]}' y "
                                 f"no '{nonce}': hay otra hebra en esta misma maquina")
            if intento > 1:
                log(f"    [{etiqueta}] SSH utilizable al intento {intento} "
                    f"(el banner llego antes que la clave)")
            return
        ultimo = f"rc={code} {(salida or '').strip()[-120:]}"
        if intento < intentos:
            time.sleep(espera_s)
    raise RuntimeError(f"no pude poner el sello de propiedad tras {intentos} "
                       f"intentos en {intentos * espera_s / 60:.0f} min: {ultimo}")


def comprobar_sello(host: str, port: int, nonce: str) -> None:
    """Vuelve a mirar el sello. Se llama JUSTO antes de arrancar el
    entrenamiento, que es el momento en el que equivocarse deja de ser ruidoso y
    pasa a producir numeros."""
    code, salida = V.ssh_capture(host, port, f"cat {SELLO} 2>/dev/null || true\n",
                                 timeout=120)
    leido = (salida or "").strip().splitlines()[-1].strip() if salida.strip() else ""
    if code != 0 or leido != nonce:
        raise Suplantada(f"el sello ya no es mio antes de entrenar (leido "
                         f"'{leido[:40]}', esperaba '{nonce}')")


# ------------------------------------------------- el libro de a bordo (git)


class Libro:
    """Los ficheros pequenos de cada run, traidos de las maquinas y commiteados.

    Un hilo aparte commitea, y no cada hilo el suyo, por una razon practica: git
    serializa con `.git/index.lock` y N hilos peleandose por el se traducen en
    fallos intermitentes que parecen otra cosa. Aqui el trafico de red va en
    paralelo (cada maquina el suyo) y la escritura en git en serie.

    Un fallo de `git push` NO para la flota: se apunta, se sigue, y se reintenta
    en la vuelta siguiente. Que el estudio se cayera porque la red parpadeo seria
    cambiar un problema pequeno por uno grande. Lo que si se dice es cuantas
    vueltas lleva sin poder empujar.
    """

    def __init__(self, activo: bool, cada_s: int, empujar: bool = True):
        self.activo = activo
        self.cada_s = max(20, cada_s)
        self.empujar = empujar
        self.sucio = threading.Event()
        self.parar = threading.Event()
        self.commits = 0
        self.fallos_push = 0
        self.ultimo_error = None
        self._hilo = None

    # -- lado de las maquinas: dejar ficheros en runs/ ------------------------

    def guardar(self, tar_local: Path) -> list:
        """Extrae en el repo el tar que trajo una maquina. Devuelve los runs
        tocados. Los nombres se leen del TAR y no del directorio, porque `runs/`
        acumula los de todos los estudios anteriores."""
        with _disco:
            with tarfile.open(tar_local, "r:gz") as tar:
                miembros = [m for m in tar.getmembers()
                            if m.isfile() and Path(m.name).name in LIBRO]
                if not miembros:
                    return []
                tar.extractall(ROOT, members=miembros)
                tocados = sorted({Path(m.name).parts[1] for m in miembros
                                  if len(Path(m.name).parts) > 1})
        if tocados:
            self.sucio.set()
        return tocados

    # -- lado de git ---------------------------------------------------------

    def arrancar(self) -> None:
        if not self.activo:
            return
        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

    def _bucle(self) -> None:
        while not self.parar.is_set():
            self.parar.wait(self.cada_s)
            if self.sucio.is_set():
                self.sucio.clear()
                self.commit("estudio: libro de a bordo (epocas en curso)")

    def cerrar(self, mensaje: str) -> None:
        self.parar.set()
        if self._hilo:
            self._hilo.join(timeout=10)
        if self.activo:
            self.commit(mensaje)

    def commit(self, mensaje: str) -> bool:
        if not self.activo:
            return False
        with _git:
            try:
                subprocess.run(["git", "add", "--", "runs", "sweeps"], cwd=str(ROOT),
                               capture_output=True, timeout=120)
                hay = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                     cwd=str(ROOT), timeout=60)
                if hay.returncode == 0:
                    return False           # nada que commitear, y no es un fallo
                c = subprocess.run(["git", "commit", "-q", "-m", mensaje],
                                   cwd=str(ROOT), capture_output=True, text=True,
                                   timeout=120)
                if c.returncode != 0:
                    self.ultimo_error = (c.stderr or c.stdout or "?").strip()[:200]
                    log(f"    [git] no pude commitear: {self.ultimo_error}")
                    return False
                self.commits += 1
                if not self.empujar:
                    return True
                p = subprocess.run(["git", "push"], cwd=str(ROOT), capture_output=True,
                                   text=True, timeout=300)
                if p.returncode != 0:
                    self.fallos_push += 1
                    self.ultimo_error = (p.stderr or p.stdout or "?").strip()[:200]
                    log(f"    [git] commit {self.commits} hecho pero el push fallo "
                        f"({self.fallos_push} seguidos): {self.ultimo_error}")
                    return True
                self.fallos_push = 0
                return True
            except (subprocess.TimeoutExpired, OSError) as exc:
                self.ultimo_error = f"{type(exc).__name__}: {exc}"[:200]
                log(f"    [git] {self.ultimo_error}")
                return False


# ---------------------------------------------------------------- resumir


def puntos_pendientes(sweep: str, valid: list) -> tuple[list, list]:
    """(pendientes, hechos) leyendo `runs/` -- que es lo que git trae de vuelta.

    "Hecho" es exactamente lo que `run_sweep` considera hecho (`done` o
    `cancelled`), y no otra cosa: si aqui se saltara un punto que el runner
    fuera a rehacer, el recorrido quedaria con un hueco que solo se ve al final,
    contando. Un `error`/`running`/a medias cuenta como PENDIENTE y se repite.
    """
    pendientes, hechos = [], []
    for i, p in enumerate(valid):
        nombre = point_run_name(sweep, i, p["overrides"])
        st = ROOT / "runs" / nombre / "status.json"
        estado = None
        if st.exists():
            try:
                estado = json.loads(st.read_text(encoding="utf-8")).get("status")
            except (OSError, json.JSONDecodeError):
                estado = None
        (hechos if estado in ("done", "cancelled") else pendientes).append(i)
    return pendientes, hechos


# ---------------------------------------------------------------- la particion


def particion(sweeps: list, modo: str) -> list:
    """Que puntos corre cada maquina. Es la unica decision de diseño del script.

    Devuelve lotes {sweep, etiqueta, puntos (indices GLOBALES del recorrido),
    descripcion}. Los indices son globales a proposito: es lo que hace que los
    runs de todas las maquinas se llamen igual que si hubieran corrido aqui de
    corrido, y por eso se juntan luego en un solo recorrido.
    """
    lotes = []
    for s in sweeps:
        nombre, valid, pendientes = s["nombre"], s["valid"], s["pendientes"]
        if modo == "seed":
            semillas = sorted({valid[i]["overrides"].get("seed") for i in pendientes}
                              - {None})
            for sem in semillas:
                idx = [i for i in pendientes
                       if valid[i]["overrides"].get("seed") == sem]
                if idx:
                    lotes.append({"sweep": nombre, "etiqueta": f"{nombre}-s{sem}",
                                  "puntos": idx,
                                  "descripcion": f"{nombre} semilla {sem} "
                                                 f"({len(idx)} puntos)"})
            if not semillas and pendientes:       # recorrido sin eje `seed`
                lotes.append({"sweep": nombre, "etiqueta": f"{nombre}-todo",
                              "puntos": list(pendientes),
                              "descripcion": f"{nombre} ({len(pendientes)} puntos)"})
        elif modo == "run":
            for i in pendientes:
                lotes.append({"sweep": nombre, "etiqueta": f"{nombre}-p{i}",
                              "puntos": [i],
                              "descripcion": point_run_name(nombre, i,
                                                            valid[i]["overrides"])})
        else:
            die(f"reparto '{modo}' no existe: usa 'seed' o 'run'")
    return lotes


# ------------------------------------------------------------------- el payload


def construir_payload(sweeps: list, dataset: str) -> Path:
    """El tar que se sube: codigo + los recorridos + el dataset YA EXTRAIDO.

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
        for s in sweeps:
            tar.add(str(ROOT / "sweeps" / s["nombre"]),
                    arcname=f"sweeps/{s['nombre']}", filter=filtro)
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

    def quedan(self) -> int:
        with _reparto:
            return len(self.pool)


def apuntar_fallo(oferta: dict, motivo: str, etiqueta: str) -> None:
    mid = oferta.get("machine_id")
    if mid is None:
        return
    with _registro:
        r = V.bloquear_maquina(mid, motivo, etiqueta)
    log(f"    maquina {mid} APUNTADA en la lista negra ({r['fallos']} fallo(s)): {motivo}")


# ------------------------------------------- fase A: preparar (y medir) una maquina


class Maquina:
    """Una instancia alquilada, ya instalada y ya medida. Sabe destruirse."""

    def __init__(self, oferta: dict, iid: int, host: str, port: int, etiqueta: str,
                 nonce: str = ""):
        self.oferta, self.iid, self.host, self.port = oferta, iid, host, port
        self.etiqueta = etiqueta
        self.nonce = nonce
        self.precio = float(oferta.get("dph_total") or 0)
        self.resumen = V.resumen_maquina(oferta)
        self.t0 = time.time()
        self.tiempos: dict = {}
        self.sonda: dict = {}
        self.viva = True

    @property
    def mid(self):
        return self.oferta.get("machine_id")

    def usd(self) -> float:
        return round(self.precio * (time.time() - self.t0) / 3600, 5)

    def destruir(self, por_que: str) -> None:
        if not self.viva:
            return
        self.viva = False
        soltar_destino(self.host, self.port, self.etiqueta)
        vivida = time.time() - self.t0
        with _reparto:
            GASTO.append({"etiqueta": self.etiqueta, "machine_id": self.mid,
                          "por_que": por_que, "minutos": round(vivida / 60, 2),
                          "usd": round(self.precio * vivida / 3600, 5)})
        try:
            V.destruir(self.iid)
            log(f"    [{self.etiqueta}] instancia {self.iid} destruida ({por_que}). "
                f"Vivio {vivida / 60:.1f} min, {self.precio * vivida / 3600:.4f} $")
        except Exception as exc:                       # noqa: BLE001
            log(f"    [{self.etiqueta}] AVISO GRAVE: no pude destruir {self.iid}: {exc}\n"
                f"    SIGUE FACTURANDO. Destruyela ya:\n"
                f"    python3 {LANZADOR}/scripts/vast_instance.py destroy {self.iid} --yes")


def preparar(oferta: dict, payload: Path, etiqueta: str, hilos: int, disco_gb: float,
             sonda_sweep: str, sonda_punto: int, sonda_pasos: int) -> Maquina:
    """Alquila, sube, instala y CRONOMETRA. Lanza si algo de eso falla.

    La sonda va aqui y no despues por una razon de dinero: una maquina que no
    sirve se descubre antes de darle trabajo, y lo unico que ha costado es su
    peaje. Y va DESPUES de instalar porque mide entrenamiento de verdad, que
    necesita torch -- medir otra cosa (una multiplicacion de matrices sintetica)
    ordenaria las maquinas por un criterio que no es el del trabajo.
    """
    m = None
    maquina = V.resumen_maquina(oferta)
    log(f"[{etiqueta}] oferta {oferta.get('id')} maquina {oferta.get('machine_id')} "
        f"{maquina['vcpu']:g} vCPU {maquina['ram_gb']:g} GB "
        f"{float(oferta.get('dph_total') or 0):.4f} $/h {maquina['ubicacion']} "
        f"· {maquina.get('cpu') or '?'}")
    t0 = time.time()
    iid = V.alquilar(oferta, f"estudio-{etiqueta}", V.cfg("VAST_IMAGE"), disco_gb)
    log(f"[{etiqueta}] instancia {iid} alquilada")
    try:
        info = V.esperar_estado(iid, int(V.cfg("VAST_BOOT_TIMEOUT")))
        estado = (info.get("actual_status") or info.get("cur_state") or "?").lower()
        if estado != "running":
            raise RuntimeError(f"la instancia acabo en '{estado}', no arranco")
        host, port = resolver_destino(iid, etiqueta, info)
        if not V.esperar_ssh(host, port):
            soltar_destino(host, port, etiqueta)
            raise RuntimeError(f"sshd no contesto en {host}:{port}")
        # El sello va ANTES de subir nada: si esta maquina no es la nuestra, que
        # se sepa mientras lo unico perdido sea medio minuto de arranque
        nonce = f"{etiqueta}-{iid}-{os.getpid()}"
        sellar(host, port, nonce, etiqueta)
        m = Maquina(oferta, iid, host, port, etiqueta, nonce)
        m.t0 = t0
        m.tiempos["arranque_s"] = round(time.time() - t0, 1)
        log(f"[{etiqueta}] SSH listo en {host}:{port} "
            f"({m.tiempos['arranque_s'] / 60:.1f} min), "
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
        m.tiempos["subida_s"] = round(time.time() - t_sub, 1)

        log(f"[{etiqueta}] instalando (torch tarda minutos)...")
        t_inst = time.time()
        if V.ssh_script(host, port, INSTALL, 2400) != 0:
            raise RuntimeError("fallo la instalacion de dependencias")
        m.tiempos["instalacion_s"] = round(time.time() - t_inst, 1)
        # El PEAJE: todo lo que hay que esperar antes de la primera epoca. Se paga
        # entero POR MAQUINA, asi que es la parte del coste que crece al repartir
        # mas fino -- y es exactamente lo que hay que mirar para decidir si
        # compensa.
        m.tiempos["peaje_s"] = round(time.time() - t0, 1)

        t_son = time.time()
        code, salida = V.ssh_capture(
            host, port,
            f"set -eu\ncd /root/bench\n"
            f"export OMP_NUM_THREADS={hilos} MKL_NUM_THREADS={hilos}\n"
            f".venv/bin/python scripts/sonda_velocidad.py --sweep {sonda_sweep} "
            f"--punto {sonda_punto} --pasos {sonda_pasos}\n", timeout=600)
        linea = ""
        for l in reversed(salida.strip().splitlines()):
            if l.strip().startswith("{"):
                linea = l.strip()
                break
        if code != 0 or not linea:
            raise RuntimeError(f"la sonda de velocidad no devolvio nada (rc={code}): "
                               f"{salida.strip()[-200:]}")
        m.sonda = json.loads(linea)
        if not m.sonda.get("ok"):
            raise RuntimeError(f"la sonda fallo: {m.sonda.get('error')}")
        m.tiempos["sonda_s"] = round(time.time() - t_son, 1)
        log(f"[{etiqueta}] listo en {m.tiempos['peaje_s'] / 60:.1f} min "
            f"(arranque {m.tiempos['arranque_s'] / 60:.1f} + subida "
            f"{m.tiempos['subida_s']:.0f}s + instalacion "
            f"{m.tiempos['instalacion_s']:.0f}s) · SONDA "
            f"{m.sonda['s_paso'] * 1000:.0f} ms/paso "
            f"(~{m.sonda['s_epoca_estimada']:.0f} s/epoca) · "
            f"{m.sonda.get('cpu', '?')}")
        return m
    except Exception:
        # se destruye AQUI y no en el que llama: quien alquila, paga y limpia
        if m is not None:
            m.destruir("fallo al preparar")
        else:
            with _reparto:
                GASTO.append({"etiqueta": etiqueta, "machine_id": oferta.get("machine_id"),
                              "por_que": "fallo antes de estar lista",
                              "minutos": round((time.time() - t0) / 60, 2),
                              "usd": round(float(oferta.get("dph_total") or 0)
                                           * (time.time() - t0) / 3600, 5)})
            try:
                V.destruir(iid)
                log(f"    [{etiqueta}] instancia {iid} destruida (fallo al preparar)")
            except Exception as exc:                   # noqa: BLE001
                log(f"    [{etiqueta}] AVISO GRAVE: no pude destruir {iid}: {exc}\n"
                    f"    SIGUE FACTURANDO: python3 {LANZADOR}/scripts/"
                    f"vast_instance.py destroy {iid} --yes")
        raise


def preparar_con_reintentos(pool: Maquinas, payload: Path, etiqueta: str, hilos: int,
                            disco_gb: float, sonda: tuple, intentos: int) -> "Maquina | None":
    for intento in range(1, intentos + 1):
        oferta = pool.siguiente()
        if oferta is None:
            log(f"[{etiqueta}] no quedan maquinas distintas libres")
            return None
        try:
            return preparar(oferta, payload, etiqueta, hilos, disco_gb, *sonda)
        except Suplantada as exc:
            # NO se apunta: el host no ha hecho nada, se equivoco el catalogo al
            # publicar el mismo destino SSH para dos instancias a la vez
            log(f"[{etiqueta}] destino SSH suplantado (intento {intento}/{intentos}): "
                f"{exc}. Se coge OTRA maquina y el host NO va a la lista negra.")
        except Exception as exc:                        # noqa: BLE001
            motivo = f"{type(exc).__name__}: {exc}"[:200]
            log(f"[{etiqueta}] FALLO preparando (intento {intento}/{intentos}): {motivo}")
            apuntar_fallo(oferta, motivo, f"estudio-{etiqueta}")
    return None


# ----------------------------------------------------- fase B: la criba de velocidad


def cribar(maquinas: list, cuantas: int, umbral: float) -> tuple:
    """Se queda con las `cuantas` mas rapidas y destruye el resto. Criterio en §
    del docstring del modulo, escrito antes de ver ningun numero.

    Devuelve (elegidas, descartadas, informe). Ordena por `s_paso`, que es lo que
    la sonda mide directamente; `s_epoca_estimada` solo se imprime.
    """
    vivas = [m for m in maquinas if m is not None and m.viva]
    informe = {"umbral": umbral, "cohorte": len(vivas), "necesarias": cuantas}
    if not vivas:
        return [], [], informe
    orden = sorted(vivas, key=lambda m: m.sonda["s_paso"])
    mediana = statistics.median(m.sonda["s_paso"] for m in orden)
    # "significativamente mas lenta" = tarda mas de mediana/umbral por paso
    tope = mediana / umbral
    informe["mediana_s_paso"] = round(mediana, 5)
    informe["tope_s_paso"] = round(tope, 5)
    rapidas = [m for m in orden if m.sonda["s_paso"] <= tope]
    lentas = [m for m in orden if m.sonda["s_paso"] > tope]

    log("")
    log(f"CRIBA DE VELOCIDAD: {len(vivas)} maquinas medidas, hacen falta {cuantas}")
    log(f"  mediana {mediana * 1000:.0f} ms/paso · se descarta por encima de "
        f"{tope * 1000:.0f} ms/paso (mediana / {umbral:g})")
    for m in orden:
        rel = m.sonda["s_paso"] / mediana
        marca = "LENTA " if m in lentas else "      "
        log(f"  {marca}{m.etiqueta:>18} {m.sonda['s_paso'] * 1000:7.0f} ms/paso "
            f"({rel:4.2f}x) ~{m.sonda['s_epoca_estimada']:5.0f} s/epoca · "
            f"{m.resumen['vcpu']:g} vCPU · {m.precio:.4f} $/h · "
            f"{(m.sonda.get('cpu') or '?')[:34]}")

    elegidas = rapidas[:cuantas]
    completadas = []
    if len(elegidas) < cuantas:
        # una maquina lenta es peor que ninguna SOLO hasta que la alternativa es
        # un punto sin medir. Se dice en voz alta, que es la parte que importa.
        faltan = cuantas - len(elegidas)
        completadas = lentas[:faltan]
        elegidas = elegidas + completadas
        log(f"  ⚠ solo {len(rapidas)} pasaron el umbral y hacen falta {cuantas}: "
            f"se completan con {len(completadas)} de las LENTAS "
            f"({', '.join(m.etiqueta for m in completadas)}). El estudio corre, "
            f"pero su reloj lo marcan estas.")
    descartadas = [m for m in vivas if m not in elegidas]
    informe.update({
        "elegidas": [{"etiqueta": m.etiqueta, "machine_id": m.mid,
                      "s_paso": m.sonda["s_paso"], "cpu": m.sonda.get("cpu"),
                      "usd_hora": m.precio} for m in elegidas],
        "descartadas": [{"etiqueta": m.etiqueta, "machine_id": m.mid,
                         "s_paso": m.sonda["s_paso"], "cpu": m.sonda.get("cpu"),
                         "razon": "mas lenta que el umbral" if m in lentas
                                  else "sobraba (mas lenta que las elegidas)"}
                        for m in descartadas],
        "completadas_con_lentas": [m.etiqueta for m in completadas],
    })
    if descartadas:
        log(f"  se destruyen {len(descartadas)}: "
            f"{', '.join(m.etiqueta for m in descartadas)}")
        for m in descartadas:
            m.destruir("descartada por la criba")
        informe["usd_descartadas"] = round(sum(m.usd() for m in descartadas), 5)
        informe["min_descartadas"] = round(
            sum(time.time() - m.t0 for m in descartadas) / 60, 1)
    ganancia = (max(m.sonda["s_paso"] for m in vivas)
                / max(m.sonda["s_paso"] for m in elegidas)) if elegidas else 1.0
    informe["ganancia_vs_peor"] = round(ganancia, 2)
    log(f"  quedan {len(elegidas)}: de {min(m.sonda['s_paso'] for m in elegidas) * 1000:.0f} "
        f"a {max(m.sonda['s_paso'] for m in elegidas) * 1000:.0f} ms/paso. "
        f"La peor elegida va {ganancia:.2f}x mas rapida que la peor de la cohorte.")
    log("")
    return elegidas, descartadas, informe


# ------------------------------------------ fase C: entrenar en una maquina ya lista


PULL = ("cd /root/bench 2>/dev/null && find runs -type f "
        r"\( -name metrics.jsonl -o -name status.json -o -name config.json "
        r"-o -name summary.json \) -print0 2>/dev/null "
        "| tar --null -czf - -T - 2>/dev/null || true")


def traer_libro(m: "Maquina", libro: Libro) -> list:
    """Se trae los ficheros pequenos de los runs de esta maquina. Nunca lanza:
    una sonda perdida no puede tumbar un entrenamiento que va bien."""
    try:
        destino = Path(tempfile.mkdtemp(prefix=f"libro-{m.etiqueta}-")) / "libro.tar.gz"
        with destino.open("wb") as fh:
            p = subprocess.run(V.ssh_command(m.host, m.port) + [PULL],
                               stdout=fh, stderr=subprocess.DEVNULL, timeout=240)
        if p.returncode != 0 or destino.stat().st_size < 30:
            return []
        return libro.guardar(destino)
    except Exception as exc:                            # noqa: BLE001
        log(f"    [{m.etiqueta}] no pude traerme el libro esta vuelta: "
            f"{type(exc).__name__}: {exc}")
        return []


def segundos_por_epoca(run: str) -> list:
    """Los tiempos por epoca que el propio entrenamiento ya escribio."""
    p = ROOT / "runs" / run / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    try:
        for linea in p.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            s = json.loads(linea).get("seconds")
            if s:
                out.append(float(s))
    except (OSError, json.JSONDecodeError):
        return []
    return out


def mirar_degradacion(runs: list, umbral: float) -> "dict | None":
    """¿Se ha vuelto lenta esta maquina? Se mira con los tiempos por epoca del
    propio entrenamiento -- volver a correr la sonda robaria nucleos justo a lo
    que se esta midiendo, y falsearia la medida que se quiere proteger.

    Hacen falta al menos 6 epocas del MISMO run: 3 para la base y 3 para el
    tramo reciente. Comparar entre runs distintos no valdria, porque cambiar de
    punto cambia legitimamente el coste por epoca (otro batch, otra profundidad).
    """
    peor = None
    for run in runs:
        segs = segundos_por_epoca(run)
        if len(segs) < 6:
            continue
        base = statistics.median(segs[:3])
        reciente = statistics.median(segs[-3:])
        if not base:
            continue
        razon = reciente / base
        if razon > umbral and (peor is None or razon > peor["razon"]):
            peor = {"run": run, "base_s": round(base, 1),
                    "reciente_s": round(reciente, 1), "razon": round(razon, 2),
                    "epocas": len(segs)}
    return peor


class Degradada(RuntimeError):
    """La maquina sigue viva pero se ha vuelto lenta: se abandona y se reparte
    lo que le quedaba. No es un fallo del host, asi que NO se apunta en la lista
    negra (un inquilino ajeno se va; un host roto no)."""


def entrenar_lote(m: "Maquina", lote: dict, hilos: int, plazo_s: int, cada_s: int,
                  libro: Libro, umbral_deg: float, degradado: str) -> dict:
    """Corre los puntos del lote en una maquina YA preparada y se trae los runs.

    La destruccion va en `finally` y no es opcional: si algo revienta se pierde
    la medida, no el dinero.
    """
    tag, sweep = lote["etiqueta"], lote["sweep"]
    resultado = {"lote": tag, "sweep": sweep, "puntos": lote["puntos"],
                 "que": lote["descripcion"], "oferta": m.oferta.get("id"),
                 "machine_id": m.mid, "maquina": m.resumen,
                 "usd_hora": round(m.precio, 5), "ok": False, "error": None,
                 "epocas": None, "s_por_epoca": None, "instancia": m.iid,
                 "sonda": m.sonda, **m.tiempos}
    nombres = [point_run_name(sweep, i, lote["valid"][i]["overrides"])
               for i in lote["puntos"]]
    try:
        # el momento en que equivocarse deja de ser ruidoso y pasa a dar numeros
        if m.nonce:
            comprobar_sello(m.host, m.port, m.nonce)
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
        if V.ssh_script(m.host, m.port, arranque, 300) != 0:
            raise RuntimeError("no pude lanzar el entrenamiento")

        t_ent = time.time()
        rc, ultima, sospechas = None, "", 0
        while True:
            time.sleep(cada_s)
            code, salida = V.ssh_capture(
                m.host, m.port,
                "set +e\n"
                "echo RC=$(cat /root/estudio.rc 2>/dev/null)\n"
                "echo EPOCAS=$(cat /root/bench/runs/*/metrics.jsonl 2>/dev/null | wc -l)\n"
                "echo HECHOS=$(ls -d /root/bench/runs/*/ 2>/dev/null | wc -l)\n"
                "tail -n 2 /root/estudio.log 2>/dev/null\n", timeout=180)
            if code != 0:
                # una sonda fallida no es la maquina caida: se reintenta hasta el plazo
                log(f"[{tag}] la sonda de SSH no contesto, se reintenta")
                if time.time() - t_ent > plazo_s:
                    raise RuntimeError(f"plazo agotado ({plazo_s / 3600:.1f} h) "
                                       f"y la maquina no contesta")
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

            # el libro de a bordo: en cada vuelta, y por eso `--cada` marca la
            # granularidad con la que se puede reanudar
            traidos = traer_libro(m, libro)

            # La ultima linea del log REMOTO viaja hasta aqui a proposito: es
            # donde `estudio_lote.py` dice "punto 2/3 terminado: <run>", y sin
            # traerla el vigilante solo sabe contar directorios.
            ultima = salida.strip().splitlines()[-1] if salida.strip() else ""
            eco = ultima.split("  ", 1)[-1].strip() if "punto" in ultima else ""
            log(f"[{tag}] {transcurrido / 60:5.1f} min · {epocas} epocas · "
                f"{campos.get('HECHOS', '?')} runs · "
                + (f"{spe:.1f} s/epoca" if spe else "aun sin epoca")
                + (f" · {eco}" if eco else "")
                + (f" · libro: {len(traidos)}" if traidos else ""))

            deg = mirar_degradacion(nombres, umbral_deg)
            if deg:
                sospechas += 1
                log(f"[{tag}] ⚠ VELOCIDAD DEGRADADA ({sospechas}/2): {deg['run']} "
                    f"empezo a {deg['base_s']:.0f} s/epoca y va por "
                    f"{deg['reciente_s']:.0f} ({deg['razon']:.2f}x) tras "
                    f"{deg['epocas']} epocas")
                resultado["degradacion"] = deg
                if sospechas >= 2 and degradado == "abandonar":
                    raise Degradada(
                        f"se volvio {deg['razon']:.2f}x mas lenta en {deg['run']} "
                        f"({deg['base_s']:.0f} -> {deg['reciente_s']:.0f} s/epoca)")
            else:
                sospechas = 0

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
        traidos = recoger_runs(m)
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
        # antes de soltarla, un ultimo tiron del libro: lo que haya escrito en la
        # ultima vuelta no puede quedarse en una maquina que va a desaparecer
        traer_libro(m, libro)
        resultado["segundos_vivida"] = round(time.time() - m.t0, 1)
        resultado["usd"] = m.usd()
        m.destruir("lote terminado")


def recoger_runs(m: "Maquina") -> list:
    """El tar entero de `runs/`, al final. Redundante con el libro a proposito:
    el libro trae texto cada vuelta y esto trae lo que falte (y comprueba que lo
    que hay en disco es lo que la maquina tenia)."""
    if V.ssh_script(m.host, m.port,
                    "set -eu\ncd /root/bench\ntar -czf /root/runs.tar.gz runs\n",
                    600) != 0:
        raise RuntimeError("no pude empaquetar los runs en la maquina")
    local = Path(tempfile.mkdtemp(prefix=f"runs-{m.etiqueta}-")) / "runs.tar.gz"
    with local.open("wb") as fh:
        p = subprocess.run(V.ssh_command(m.host, m.port) + ["cat /root/runs.tar.gz"],
                           stdout=fh, timeout=900)
    if p.returncode != 0 or local.stat().st_size == 0:
        raise RuntimeError("no pude traerme los runs de la maquina")
    with _disco:
        with tarfile.open(local, "r:gz") as tar:
            tar.extractall(ROOT)
            # Los nombres se leen del TAR, no del directorio local: `runs/`
            # acumula los de todos los estudios anteriores.
            return sorted({Path(x.name).parts[1] for x in tar.getmembers()
                           if len(Path(x.name).parts) > 1})


def lote_con_reintentos(lote: dict, pool: Maquinas, payload: Path,
                        hilos: int, plazo_s: int, cada_s: int, disco_gb: float,
                        intentos: int, libro: Libro, umbral_deg: float,
                        degradado: str, sonda: tuple) -> dict:
    """Corre un lote, y si la maquina falla lo reintenta en OTRA.

    Al reintentar recalcula los puntos que faltan leyendo `runs/`: gracias al
    libro de a bordo, los que ya terminaron no se repiten. Es la diferencia
    entre perder un run y perder el lote entero.
    """
    tag = lote["etiqueta"]
    ultimo = {"lote": tag, "sweep": lote["sweep"], "puntos": lote["puntos"],
              "ok": False, "error": "sin intentos"}
    for intento in range(1, intentos + 1):
        pendientes, hechos = puntos_pendientes(lote["sweep"], lote["valid"])
        mios = [i for i in lote["puntos"] if i in set(pendientes)]
        if not mios:
            log(f"[{tag}] ya no queda nada pendiente (el libro dice que estan "
                f"los {len(lote['puntos'])} puntos). No se alquila nada.")
            return {"lote": tag, "sweep": lote["sweep"], "puntos": lote["puntos"],
                    "ok": True, "error": None, "sin_maquina": True,
                    "runs": [point_run_name(lote["sweep"], i,
                                            lote["valid"][i]["overrides"])
                             for i in lote["puntos"]]}
        if len(mios) < len(lote["puntos"]):
            log(f"[{tag}] intento {intento}: quedan {len(mios)} de "
                f"{len(lote['puntos'])} puntos ({len(lote['puntos']) - len(mios)} "
                f"ya estaban en el libro)")
        activo = dict(lote, puntos=mios)

        m = lote.pop("maquina", None)
        if m is None:
            m = preparar_con_reintentos(pool, payload, tag, hilos, disco_gb,
                                        sonda, intentos)
        if m is None:
            ultimo["error"] = ("no quedan maquinas distintas libres; sube "
                               "--repuestos o afloja las condiciones")
            break
        try:
            r = entrenar_lote(m, activo, hilos, plazo_s, cada_s, libro,
                              umbral_deg, degradado)
            r["intento"] = intento
            if r["ok"] or r.get("runs"):
                return r
            ultimo = r
        except Suplantada as exc:
            log(f"[{tag}] la maquina reservada resulto no ser la mia: {exc}. "
                f"NO va a la lista negra; se coge otra.")
            ultimo = {"lote": tag, "sweep": lote["sweep"], "puntos": activo["puntos"],
                      "ok": False, "error": f"suplantada: {exc}",
                      "machine_id": m.mid, "intento": intento, "suplantada": True}
        except Degradada as exc:
            # NO se apunta en la lista negra: la maquina funciona, tiene compania
            log(f"[{tag}] ABANDONADA por lentitud (intento {intento}/{intentos}): {exc}")
            ultimo = {"lote": tag, "sweep": lote["sweep"], "puntos": activo["puntos"],
                      "ok": False, "error": f"degradada: {exc}",
                      "machine_id": m.mid, "intento": intento, "degradada": True}
            if intento < intentos:
                log(f"[{tag}] se reparte lo que quedaba en OTRA maquina...")
        except Exception as exc:                        # noqa: BLE001
            motivo = f"{type(exc).__name__}: {exc}"[:200]
            log(f"[{tag}] FALLO en el intento {intento}/{intentos}: {motivo}")
            apuntar_fallo(m.oferta, motivo, f"estudio-{tag}")
            ultimo = {"lote": tag, "sweep": lote["sweep"], "puntos": activo["puntos"],
                      "ok": False, "error": motivo, "oferta": m.oferta.get("id"),
                      "machine_id": m.mid, "intento": intento, "bloqueada": True}
            if intento < intentos:
                log(f"[{tag}] reintentando en OTRA maquina...")
    return ultimo


# ------------------------------------------------------------------------ main


def cargar_sweeps(nombres: list, store: SweepStore) -> tuple:
    sweeps, datasets = [], set()
    for nombre in nombres:
        if not store.exists(nombre):
            die(f"no existe el recorrido '{nombre}'. Crealo primero con su script.")
        spec = store.spec(nombre)
        valid, _ = expand_points(spec, spec["base_network_value"])
        pendientes, hechos = puntos_pendientes(nombre, valid)
        sweeps.append({"nombre": nombre, "spec": spec, "valid": valid,
                       "pendientes": pendientes, "hechos": hechos})
        datasets.add(spec["window_dataset"])
    if len(datasets) > 1:
        die("los recorridos de una misma flota tienen que compartir dataset de "
            f"ventanas, y hay {len(datasets)}: {sorted(datasets)}.\n"
            "  El payload lleva UN windows.npz; mezclarlos entrenaria media flota\n"
            "  sobre otro dato y la comparacion no significaria nada.\n"
            "  Lanza una flota por dataset.")
    return sweeps, datasets.pop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="append", required=True,
                    help="recorrido ya creado (sweeps/<name>); repetible")
    ap.add_argument("--reparto", choices=("seed", "run"), default="seed",
                    help="seed: una maquina por recorrido x semilla (bloques, por "
                         "defecto). run: una maquina por punto (maximo paralelismo)")
    ap.add_argument("--cpus", type=int, default=8, help="vCPU efectivas minimas")
    ap.add_argument("--max-cpus", type=int, default=32)
    ap.add_argument("--min-ram", type=float, default=8.0, metavar="GB")
    ap.add_argument("--cpu", default="",
                    help="exige esta CPU (subcadena, p.ej. 'E5-26'). MEDIDO: dentro "
                         "de la familia Xeon E5-26xx v3/v4 el entrenamiento sale "
                         "IDENTICO bit a bit entre maquinas, y diverge al cruzar de "
                         "familia (hasta 0,0457 en f1). Fijarla convierte el ruido "
                         "de maquina en cero -- ver docs/plan-lr-alto.md §7.4. "
                         "Ademas es lo que hace SEGURA la criba de velocidad")
    ap.add_argument("--max-price", type=float, default=None, metavar="USD_HORA")
    ap.add_argument("--disk", type=float, default=16.0, metavar="GB")
    ap.add_argument("--hilos", type=int, default=8,
                    help="hilos de torch, IGUALES en todas: una maquina con mas "
                         "nucleos no debe entrenar distinto que otra")
    ap.add_argument("--horas-max", type=float, default=6.0,
                    help="plazo por lote; al agotarse se destruye la maquina")
    ap.add_argument("--cada", type=int, default=60,
                    help="segundos entre sondas. Marca la granularidad del libro "
                         "de a bordo: con epocas de 40-60 s, 60 es ~una por epoca")
    ap.add_argument("--repuestos", type=int, default=3,
                    help="maquinas distintas de reserva para los reintentos")
    ap.add_argument("--intentos", type=int, default=2, help="intentos por lote")
    ap.add_argument("--criba", type=int, default=0, metavar="K",
                    help="alquila K maquinas de mas, mide unos segundos de "
                         "entrenamiento en todas y se queda con las mas rapidas")
    ap.add_argument("--umbral-velocidad", type=float, default=0.75,
                    help="se descarta la maquina que no llegue a este x la mediana "
                         "de pasos/s de la cohorte")
    ap.add_argument("--sonda-pasos", type=int, default=40,
                    help="pasos cronometrados por la sonda de velocidad")
    ap.add_argument("--umbral-degradacion", type=float, default=1.35,
                    help="se avisa si el s/epoca reciente supera este x el inicial")
    ap.add_argument("--degradado", choices=("avisar", "abandonar"), default="avisar",
                    help="que hacer con una maquina que se vuelve lenta")
    ap.add_argument("--git", action="store_true",
                    help="commitea y empuja el libro de a bordo en cada sonda")
    ap.add_argument("--sin-push", action="store_true",
                    help="con --git, commitea pero no empuja (para probar)")
    ap.add_argument("--paralelo", type=int, default=0,
                    help="cuantas maquinas a la vez (0 = todas)")
    ap.add_argument("--estimar", action="store_true",
                    help="imprime tiempo y coste estimados y no alquila nada")
    ap.add_argument("--dry-run", action="store_true",
                    help="ensena que maquinas se cogerian y no alquila nada")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    V.load_env()
    store = SweepStore()
    sweeps, dataset = cargar_sweeps(args.sweep, store)
    lotes = particion(sweeps, args.reparto)
    for l in lotes:
        l["valid"] = next(s["valid"] for s in sweeps if s["nombre"] == l["sweep"])

    total_puntos = sum(len(s["valid"]) for s in sweeps)
    total_hechos = sum(len(s["hechos"]) for s in sweeps)
    log(f"{len(sweeps)} recorrido(s) sobre {dataset}: {total_puntos} puntos, "
        f"{total_hechos} ya en el libro, {total_puntos - total_hechos} pendientes")
    for s in sweeps:
        log(f"   {s['nombre']}: eje {json.dumps(s['spec']['space'])} · "
            f"{len(s['hechos'])}/{len(s['valid'])} hechos")
    if not lotes:
        log("No queda ningun punto pendiente. Nada que alquilar.")
        return 0
    log(f"Reparto '{args.reparto}': {len(lotes)} maquinas")
    for l in lotes:
        log(f"   {l['etiqueta']:>18} -> puntos {l['puntos']}  ({l['descripcion']})")

    if args.estimar or args.dry_run:
        pend = {s["nombre"]: set(s["pendientes"]) for s in sweeps}
        try:
            todos = [EST.estimar_sweep(s["nombre"], args.reparto, pend[s["nombre"]])
                     for s in sweeps]
            _imprimir_estimacion(todos, len(lotes), args.reparto)
        except Exception as exc:                        # noqa: BLE001
            log(f"(no pude estimar: {type(exc).__name__}: {exc})")
    if args.estimar:
        return 0

    cuantas = len(lotes) + max(0, args.criba)
    if args.criba and not args.cpu:
        log("⚠ --criba SIN --cpu: se seleccionara por velocidad sobre maquinas de "
            "familias de CPU distintas, que MIDEN DISTINTO (plan-lr-alto §7.4). "
            "La eleccion deja de ser inocua para el resultado. Usa --cpu 'E5-26'.")
    tope = args.max_price or V.limite_precio()
    pool = Maquinas(cuantas, args.repuestos, args.cpus, args.max_cpus,
                    args.min_ram, tope, args.cpu)
    log(f"\n{len(pool.pool)} maquinas DISTINTAS disponibles "
        f"({len(lotes)} a usar + {args.criba} de criba + "
        f"{len(pool.pool) - cuantas} de repuesto):")
    log("  " + V.cabecera_ofertas())
    for o in pool.pool:
        log(f"  {V.oferta_fila(o)}   maquina {o.get('machine_id')}")
    coste = sum(float(o.get("dph_total") or 0) for o in pool.pool[:cuantas])
    log(f"\nCoste maximo si las {cuantas} vivieran {args.horas_max:g} h: "
        f"{coste * args.horas_max:.2f} $ ({coste:.4f} $/h entre todas).")

    if args.dry_run:
        log("--dry-run: no se alquila nada.")
        return 0
    if not args.yes and not V.confirmar("¿Lanzo la flota?"):
        log("Cancelado. No se ha alquilado nada.")
        return 1

    payload = construir_payload(sweeps, dataset)
    log(f"Payload listo: {payload.stat().st_size / 1e6:.1f} MB "
        f"(codigo + {len(sweeps)} recorridos + dataset ya extraido)")

    libro = Libro(args.git, args.cada, empujar=not args.sin_push)
    libro.arrancar()
    if args.git:
        log("Libro de a bordo ACTIVO: cada sonda baja metrics/status de cada "
            "maquina y un hilo aparte los commitea"
            + ("" if libro.empujar else " (sin push, --sin-push)"))

    # La sonda mide con la config del primer punto pendiente del primer recorrido:
    # se trata de ORDENAR maquinas, y para eso lo que importa es que todas midan
    # LO MISMO, no que midan justo su propio punto.
    con_pendientes = next(s for s in sweeps if s["pendientes"])
    sonda = (con_pendientes["nombre"], con_pendientes["pendientes"][0],
             args.sonda_pasos)

    t0 = time.time()
    reservas, criba_informe = [], None
    if args.criba:
        log(f"\nFASE A: preparando {cuantas} maquinas ({len(lotes)} necesarias + "
            f"{args.criba} para cribar)...")
        with ThreadPoolExecutor(max_workers=cuantas) as pe:
            futuros = [pe.submit(preparar_con_reintentos, pool, payload, f"c{k}",
                                 args.hilos, args.disk, sonda, args.intentos)
                       for k in range(cuantas)]
            preparadas = [f.result() for f in futuros]
        reservas, _desc, criba_informe = cribar(
            [m for m in preparadas if m], len(lotes), args.umbral_velocidad)
        # la mas rapida al lote mas largo: el reloj lo marca la cadena mas larga
        # la mas rapida al lote mas largo, ATADA al lote: si se dejara en una
        # cola compartida, el orden en que arrancan las hebras decidiria el
        # emparejamiento y este calculo no serviria de nada
        reservas.sort(key=lambda m: m.sonda["s_paso"])
        lotes.sort(key=lambda l: -len(l["puntos"]))
        for m, l in zip(reservas, lotes):
            l["maquina"] = m
            log(f"  {l['etiqueta']:>18} ({len(l['puntos'])} puntos) -> maquina "
                f"{m.mid} ({m.sonda['s_paso'] * 1000:.0f} ms/paso)")
        sin = [l["etiqueta"] for l in lotes if "maquina" not in l]
        if sin:
            log(f"  ⚠ {len(sin)} lotes sin maquina de la criba ({', '.join(sin)}): "
                f"alquilaran la suya cuando les toque")
        log("")

    obreros = args.paralelo or len(lotes)
    with ThreadPoolExecutor(max_workers=obreros) as pe:
        futuros = [pe.submit(lote_con_reintentos, l, pool, payload,
                             args.hilos, int(args.horas_max * 3600), args.cada,
                             args.disk, args.intentos, libro,
                             args.umbral_degradacion, args.degradado, sonda)
                   for l in lotes]
        resultados = [f.result() for f in futuros]

    reloj = time.time() - t0
    # TODAS las maquinas que se alquilaron, incluidas las que murieron por el
    # camino y las que descarto la criba. Sumar solo los lotes que terminaron
    # daria un numero mas bonito y mas bajo que la factura.
    gasto = sum(float(g["usd"]) for g in GASTO)
    buenas = [r for r in resultados if r.get("ok")]
    peaje = sum(float(r.get("peaje_s") or 0) for r in resultados)
    trabajo = sum(float(r.get("entrenamiento_s") or 0) for r in resultados)
    vividas = sum(float(r.get("segundos_vivida") or 0) for r in resultados)
    reporte = {
        "recorridos": [s["nombre"] for s in sweeps], "dataset": dataset,
        "cuando": V.ahora_iso(), "reparto": args.reparto, "maquinas": len(lotes),
        "cpu": args.cpu or None, "reloj_min": round(reloj / 60, 1),
        "usd": round(gasto, 4), "hilos": args.hilos,
        "criba": criba_informe,
        "gasto_por_maquina": GASTO,
        "maquinas_alquiladas": len(GASTO),
        "git": {"activo": args.git, "commits": libro.commits,
                "fallos_push": libro.fallos_push, "ultimo_error": libro.ultimo_error},
        # El desglose que decide si un reparto compensa: el peaje se paga entero
        # por maquina (crece al repartir mas fino), el trabajo no.
        "peaje_min": round(peaje / 60, 1), "trabajo_min": round(trabajo / 60, 1),
        "maquina_min": round(vividas / 60, 1),
        "peaje_pct": round(100 * peaje / vividas, 1) if vividas else None,
        "lotes": resultados,
    }
    for s in sweeps:
        destino = ROOT / "sweeps" / s["nombre"] / "flota.json"
        destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    log("\n" + "=" * 68)
    log(f"  {len(buenas)}/{len(lotes)} lotes completos en {reloj / 60:.1f} min de "
        f"RELOJ. Gastado: {gasto:.4f} $ en {len(GASTO)} maquinas alquiladas "
        f"(reparto '{args.reparto}')")
    otras = [g for g in GASTO if g["por_que"] != "lote terminado"]
    if otras:
        log(f"  De eso, {sum(g['usd'] for g in otras):.4f} $ en {len(otras)} maquinas "
            f"que NO entrenaron hasta el final: "
            + ", ".join(f"{g['etiqueta']} ({g['por_que']})" for g in otras[:6])
            + (" …" if len(otras) > 6 else ""))
    log(f"  Maquina-minutos: {vividas / 60:.1f}  =  peaje {peaje / 60:.1f} "
        f"({reporte['peaje_pct']}%) + trabajo {trabajo / 60:.1f}")
    for r in resultados:
        marca = "ok " if r.get("ok") else "FALLO"
        log(f"  {marca} {r['lote']:>18}: maquina {r.get('machine_id')} · "
            f"{r.get('epocas') or 0} epocas · "
            + (f"{r['s_por_epoca']:.1f} s/epoca" if r.get("s_por_epoca") else "-")
            + (f" · {r['error']}" if r.get("error") else ""))
    if args.git:
        log(f"  Libro de a bordo: {libro.commits} commits"
            + (f", {libro.fallos_push} push fallidos (ULTIMO: {libro.ultimo_error})"
               if libro.fallos_push else ", empujados"))
    log("  Reporte: sweeps/<recorrido>/flota.json")
    log("  Comprueba que no queda nada vivo:")
    log(f"    python3 {LANZADOR}/scripts/vast_instance.py list")
    log("=" * 68)
    libro.cerrar(f"estudio: flota terminada ({len(buenas)}/{len(lotes)} lotes, "
                 f"{reloj / 60:.0f} min, {gasto:.4f} $)")
    return 0 if len(buenas) == len(lotes) else 1


def _imprimir_estimacion(todos: list, maquinas: int, reparto: str) -> None:
    recargo = 1.0 + EST.RECARGO_POR_MAQUINA * max(0, maquinas - 6)
    peaje_total = maquinas * EST.PEAJE_MIN
    log("")
    log("ESTIMACION (procedencia de cada coeficiente en scripts/estudio_estimar.py)")
    for e in todos:
        largos = [sum(p["min_med"] for p in lote) for lote in e["lotes"]]
        if not largos:
            continue
        log(f"  {e['sweep']:>10}: {len(e['puntos'])} runs en {len(e['lotes'])} "
            f"maquinas · la mas cargada ~{max(largos):.0f} min")
    for etiqueta, k in (("optimista", "min"), ("central  ", "med"),
                        ("pesimista", "max")):
        reloj = max((max(sum(p[f"min_{k}"] for p in lote) for lote in e["lotes"])
                     for e in todos if e["lotes"]), default=0) + EST.PEAJE_MIN
        maq = sum(sum(p[f"min_{k}"] for p in lote) for e in todos
                  for lote in e["lotes"]) + peaje_total
        log(f"  {etiqueta}: RELOJ {reloj / 60:5.1f} h · maquina-horas "
            f"{maq / 60:6.1f} · {maq / 60 * EST.USD_HORA * recargo:6.2f} $")
    log("")


if __name__ == "__main__":
    raise SystemExit(main())
