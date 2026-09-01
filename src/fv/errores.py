"""El log de errores: lo que se rompio cuando nadie estaba mirando.

Por que existe
--------------
Pedido por el dueno el 2026-09-01: *"si algo ocurre hoy y no salta a la vista
nunca me entero"*. Y tiene la evidencia de ese mismo dia: la app llevaba **8
horas** sin poder cargar una red y el fallo espero a que el la eligiera en el
movil. Un error que solo existe mientras alguien mira la pantalla no existe.

Donde vive, y por que ahi (R7 · R8 · R9)
-----------------------------------------
    <repo de datos>/errores/<anio>/<mes>/<anio>-<mes>.jsonl

- **En el repo de DATOS** (R7): lo produce el sistema al correr, no quien lo
  escribio. Y es la misma carpeta por meses que `conversaciones/` y los runs.
- **Solo se anade, un fichero por mes** (R8): es historial. El *estado* --que
  esta roto AHORA-- lo contesta el autochequeo de arranque, que es otra pregunta
  y vive en otro sitio.
- **No se puede re-derivar** (R9): si se pierde con la maquina, la unica forma de
  volver a tenerlo es que vuelva a fallar.

Las cuatro decisiones que hay que respetar si se toca
-----------------------------------------------------
1. ⚠⚠ **REGISTRAR UN ERROR NO PUEDE ROMPER NADA.** Esto se llama desde el manejador
   de excepciones: si `registrar` lanza, la peticion que ya iba mal muere de otra
   forma y encima se pierde el motivo original. Todo va dentro de un `try` que se
   traga lo suyo y escribe en stderr. Es la regla R2 aplicada al unico sitio
   donde no hay segunda oportunidad.

2. ⚠⚠ **SE AGRUPAN LAS REPETICIONES, o el log se come el disco.** El front sondea
   `/runs` cada 3 s: un fallo permanente ahi son **28.800 lineas al dia** en un
   repo de git. Un error identico dentro de la ventana no escribe otra linea; al
   cerrarse la ventana se anade UNA que dice cuantas veces mas paso.
   ⚠ La primera SI se escribe en el acto: agrupar no puede significar perder la
   unica noticia de algo que acaba de tirar el proceso.

3. **Se distingue lo que ROMPE de lo que se RECHAZA.** Un 400 con su razon es la
   puerta funcionando, y hay 109 codigos de esos: mezclarlos con los fallos
   inesperados haria un log que nadie lee (el patron del aviso que sale siempre).
   Van los dos, con `nivel` distinto, y la pantalla filtra por defecto a `error`.

4. **Se redacta por patron antes de escribir.** El repo es privado desde el
   2026-09-01, pero privado es un permiso, no un borrado: git no olvida y un
   secreto que se cuela hay que ROTARLO. Un mensaje o una traza llevan rutas,
   valores y a veces entornos enteros.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback as _tb
from datetime import datetime, timezone
from pathlib import Path

from fv import settings

MESES = ("01-enero", "02-febrero", "03-marzo", "04-abril", "05-mayo", "06-junio",
         "07-julio", "08-agosto", "09-septiembre", "10-octubre", "11-noviembre",
         "12-diciembre")

NIVELES = ("error", "rechazo")

# Formas de secreto conocidas. Mismo criterio que el archivador de conversaciones
# del coordinador: se redacta por PATRON, que es lo que cubre lo que no sale de
# un fichero de secretos (una traza con un entorno dentro, una URL con `?t=`).
PATRONES = (
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "[CLAVE-ANTHROPIC]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[TOKEN-GITHUB]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[TOKEN-GITHUB]"),
    (re.compile(r"dop_v1_[a-f0-9]{64}"), "[TOKEN-DIGITALOCEAN]"),
    (re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"), "[TOKEN-TELEGRAM]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "[CLAVE-PRIVADA]"),
    # `?t=` es como viaja el token de la web app a un navegador de movil
    (re.compile(r"([?&]t=)[A-Za-z0-9_-]{8,}"), r"\1[TOKEN]"),
)

# LA CADENCIA. Una ventana fija no sirve para las dos cosas que hay que cubrir:
#
#   · una racha corta (un dedo nervioso, un sondeo que falla tres veces) quiere
#     resolucion fina -- saber que empezo y cuando paro;
#   · algo roto de forma permanente NO quiere resolucion ninguna: con 60 s fijos,
#     un fallo que se repite cada 3 s durante una semana escribe 1.440 lineas al
#     dia, o sea 10.000 en la semana. Un log asi no se lee, y ademas es un repo
#     de git.
#
# Asi que la ventana CRECE mientras el problema siga: 1 min -> 5 -> 15 -> 1 h.
# La primera vez se escribe SIEMPRE en el acto; el resumen llega pronto (1 min)
# para que se vea que es recurrente, y luego se espacia. Peor caso de algo roto
# todo el dia: ~27 lineas en vez de 1.440.
#
# ⚠ Y si una ventana se cierra SIN repeticiones, la clave vuelve al primer
# escalon: el problema paro, y si reaparece es una noticia nueva y no la cola de
# una vieja. Sin esa vuelta atras, un fallo de la manana silenciaria una hora del
# mismo fallo por la tarde.
TOPE_TRAZA = 16_000
VENTANAS_S = (60.0, 300.0, 900.0, 3600.0)
VENTANA_S = VENTANAS_S[0]          # el primer escalon, que es el que se cita fuera
_repeticiones: dict[tuple, dict] = {}
_version: str | None = None


def _redactar(txt: str) -> str:
    for patron, con in PATRONES:
        txt = patron.sub(con, txt)
    return txt


def mes_path(cuando: datetime | None = None) -> Path:
    """El fichero del mes. La MISMA forma de carpetas que `conversaciones/`."""
    c = cuando or datetime.now(timezone.utc)
    return (settings.errores_root() / f"{c:%Y}" / MESES[c.month - 1]
            / f"{c:%Y-%m}.jsonl")


def version_codigo() -> str:
    """El sha del codigo que produjo el error, cacheado.

    ⚠ No es adorno: la clase de averia mas cara de diagnosticar es la del PROCESO
    mas viejo que el artefacto (2026-09-01, 8 horas). Sin saber que codigo lo
    escribio, un error de hace tres dias no se puede situar.
    """
    global _version
    if _version is None:
        try:
            _version = subprocess.run(
                ["git", "-C", str(settings.project_root()), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5).stdout.strip() or "?"
        except Exception:                                # noqa: BLE001
            _version = "?"
    return _version


def registrar(code: str, message: str, *, hint: str = "", nivel: str = "error",
              origen: str = "api", donde: str = "", traza: str | object = "",
              extra: dict | None = None) -> dict | None:
    """Anade una linea. Devuelve la linea escrita, o None si se agrupo/no se pudo.

    ⚠ NUNCA lanza. Ver la decision 1 del modulo.
    """
    try:
        ahora = datetime.now(timezone.utc)
        if isinstance(traza, BaseException):
            traza = "".join(_tb.format_exception(type(traza), traza,
                                                 traza.__traceback__))
        linea = {
            "cuando": ahora.isoformat(timespec="seconds"),
            "nivel": nivel if nivel in NIVELES else "error",
            "code": str(code)[:120],
            "message": _redactar(str(message))[:2000],
            "hint": _redactar(str(hint))[:1000],
            "origen": str(origen)[:60],
            "donde": _redactar(str(donde))[:300],
            "maquina": socket.gethostname()[:60],
            "version": version_codigo(),
            "pid": os.getpid(),
        }
        if traza:
            # ⚠ Se conserva el FINAL, no el principio: la causa esta abajo
            # (`RuntimeError: ...`) y la cabecera del stack es lo prescindible.
            #
            # El tope es 16 KB y no los 4.000 que tenia: aquello lo elegi por un
            # limite de atomicidad que NO existe --medido el 2026-09-01, 8
            # procesos escribiendo lineas de 100 KB a la vez, 0 corrompidas: los
            # 4.096 de PIPE_BUF son garantia para pipes, no para ficheros
            # regulares--. Una traza real de torch son 2.037 bytes, pero una
            # excepcion encadenada o una recursion pasan de eso facil.
            # ⚠ En NFS esto SI se rompe. Si el repo de datos acaba en un montaje
            # de red, hay que volver a mirarlo.
            t = _redactar(str(traza))
            if len(t) > TOPE_TRAZA:
                # una traza cortada que no dice que esta cortada te hace buscar
                # un marco que nunca estuvo
                t = (f"[... traza recortada: se guardan los ultimos "
                     f"{TOPE_TRAZA} de {len(t)} bytes ...]\n" + t[-TOPE_TRAZA:])
            linea["traza"] = t
        if extra:
            linea["extra"] = json.loads(_redactar(json.dumps(extra, default=str))[:2000])

        clave = (linea["nivel"], linea["code"], linea["origen"], linea["donde"])
        t = time.time()
        prev = _repeticiones.get(clave)
        escalon = prev["escalon"] if prev else 0
        if prev and t - prev["desde"] < VENTANAS_S[escalon]:
            prev["n"] += 1
            return None                       # se agrupa: no escribe (decision 2)
        if prev and prev["n"]:
            # La ventana se cerro con repeticiones: UNA linea que dice cuantas.
            # Lleva `repeticiones` para que la cuenta de sucesos sea cierta: una
            # linea NO es un suceso, y quien agregue tiene que sumar 1 + esto.
            _escribir({**prev["linea"],      # arrastra traza, hint y lo demas
                       "cuando": ahora.isoformat(timespec="seconds"),
                       "message": f"y {prev['n']} vez/veces mas en "
                                  f"{VENTANAS_S[escalon]:.0f} s: "
                                  f"{prev['linea']['message']}",
                       "repeticiones": prev["n"]})
            escalon = min(escalon + 1, len(VENTANAS_S) - 1)   # sigue roto: espacia
        else:
            escalon = 0                       # paro: la proxima vez es noticia
        _repeticiones[clave] = {"desde": t, "n": 0, "linea": linea,
                                "escalon": escalon}
        _escribir(linea)
        return linea
    except Exception as e:                               # noqa: BLE001
        # la ultima red: registrar no puede tumbar a quien lo llama
        print(f"[errores] no pude registrar '{code}': {e}", file=sys.stderr, flush=True)
        return None


def cerrar_ventanas() -> int:
    """Vuelca las repeticiones que quedan en memoria. Devuelve cuantas lineas.

    ⚠ EL LIMITE QUE ESTO TAPA A MEDIAS, y hay que decirlo entero: mientras una
    ventana esta abierta, sus repeticiones viven en memoria y **el contador del
    log no las incluye todavia**. O sea que el numero de sucesos va con hasta una
    ventana de retraso.

    Lo que NUNCA se pierde es la NOTICIA: la primera vez se escribe en el acto,
    asi que "esto paso" siempre esta. Lo que se puede perder es la MULTIPLICIDAD
    --"paso 25 veces"-- si el proceso muere a mitad de ventana.

    Por eso esto se engancha a la salida del proceso: un `systemctl restart` es
    un SIGTERM y ahi si corre, que es el caso comun. Contra SIGKILL no hay nada
    que hacer, como siempre.
    """
    n = 0
    ahora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for clave, prev in list(_repeticiones.items()):
        if prev.get("n"):
            try:
                _escribir({**prev["linea"], "cuando": ahora,
                           "message": f"y {prev['n']} vez/veces mas (ventana "
                                      f"cerrada al salir el proceso): "
                                      f"{prev['linea']['message']}",
                           "repeticiones": prev["n"]})
                n += 1
            except Exception:                            # noqa: BLE001
                pass
        prev["n"] = 0
    return n


atexit.register(cerrar_ventanas)


def _escribir(linea: dict) -> None:
    p = mes_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(linea, ensure_ascii=False) + "\n")


def _ficheros() -> list[Path]:
    raiz = settings.errores_root()
    return sorted(raiz.rglob("*.jsonl"), reverse=True) if raiz.exists() else []


def consultar(*, nivel: str | None = None, code: str | None = None,
              origen: str | None = None, q: str | None = None,
              desde: str | None = None, hasta: str | None = None,
              limit: int = 100, offset: int = 0,
              sin_traza: bool = False) -> dict:
    """Los errores que casan, lo mas reciente primero, PAGINADOS.

    ⚠ El filtrado y el conteo van aqui y no en el navegador (U4.3): el dueno pidio
    esto contando con que habra muchos, y mandar el fichero entero para filtrarlo
    en el front es exactamente lo que deja de funcionar cuando llega ese momento.

    Devuelve tambien las FACETAS --cuantos hay por nivel, code y origen-- porque
    con un log grande la pregunta no es "enseñamelos" sino "¿de que hay?": sin
    ellas el filtro es adivinar un valor a ciegas.
    """
    filas: list[dict] = []
    for f in _ficheros():
        mes = f.stem                                     # "2026-09"
        if desde and mes < desde[:7]:
            continue
        if hasta and mes > hasta[:7]:
            continue
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                filas.append(json.loads(ln))
            except json.JSONDecodeError:
                continue                                 # una linea rota no tumba el log
    filas.sort(key=lambda r: r.get("cuando", ""), reverse=True)

    def casa(r: dict) -> bool:
        if nivel and r.get("nivel") != nivel:
            return False
        if code and r.get("code") != code:
            return False
        if origen and r.get("origen") != origen:
            return False
        if desde and r.get("cuando", "") < desde:
            return False
        if hasta and r.get("cuando", "") > hasta:
            return False
        if q:
            aguja = q.lower()
            if aguja not in " ".join(str(r.get(k, "")) for k in
                                     ("code", "message", "hint", "donde",
                                      "origen", "version")).lower():
                return False
        return True

    casan = [r for r in filas if casa(r)]

    # ⚠ UNA LINEA NO ES UN SUCESO. Al agrupar las repeticiones, una linea con
    # `repeticiones: 340` son 341 veces que paso eso. Contar lineas diria "3"
    # donde pasaron 900 -- y un contador que miente es peor que no tenerlo,
    # porque se usa para decidir que se mira primero.
    def veces(r: dict) -> int:
        return 1 + int(r.get("repeticiones", 0) or 0)

    facetas = {"nivel": {}, "code": {}, "origen": {}, "version": {}}
    for r in casan:
        n = veces(r)
        for campo in facetas:
            v = r.get(campo)
            if v:
                facetas[campo][v] = facetas[campo].get(v, 0) + n
    for campo, cuenta in facetas.items():
        facetas[campo] = dict(sorted(cuenta.items(), key=lambda kv: -kv[1]))

    pagina = casan[offset:offset + limit]
    if sin_traza:
        # La lista no manda las trazas: la pantalla solo las enseña al ABRIR una
        # fila, y con el sondeo cada 5 s serian kilobytes por vuelta que nadie
        # mira. Se dice que existe, y se piden aparte.
        pagina = [{**r, "traza": None, "tiene_traza": bool(r.get("traza"))}
                  for r in pagina]
    return {
        "errores": pagina,
        "total": len(casan),                 # LINEAS que casan
        "sucesos": sum(veces(r) for r in casan),          # VECES que pasaron
        "total_sin_filtro": len(filas),
        "sucesos_sin_filtro": sum(veces(r) for r in filas),
        "facetas": facetas,                  # contadas en SUCESOS, no en lineas
        "meses": [f.stem for f in _ficheros()],
        "donde": str(settings.errores_root()),
    }
