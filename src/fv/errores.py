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

# Ventana de agrupacion (decision 2). 60 s es el mismo numero que usa
# `review._repetida`, y por el mismo motivo: es lo que dura un dedo nervioso o un
# sondeo del front.
VENTANA_S = 60.0
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
            linea["traza"] = _redactar(str(traza))[-4000:]
        if extra:
            linea["extra"] = json.loads(_redactar(json.dumps(extra, default=str))[:2000])

        clave = (linea["nivel"], linea["code"], linea["origen"], linea["donde"])
        t = time.time()
        prev = _repeticiones.get(clave)
        if prev and t - prev["desde"] < VENTANA_S:
            prev["n"] += 1
            return None                       # se agrupa: no escribe (decision 2)
        if prev and prev["n"]:
            # la ventana se cerro con repeticiones: se dice cuantas, en UNA linea
            _escribir({**prev["linea"],
                       "cuando": ahora.isoformat(timespec="seconds"),
                       "code": prev["linea"]["code"],
                       "message": f"y {prev['n']} vez/veces mas en "
                                  f"{VENTANA_S:.0f} s: {prev['linea']['message']}",
                       "repeticiones": prev["n"]})
        _repeticiones[clave] = {"desde": t, "n": 0, "linea": linea}
        _escribir(linea)
        return linea
    except Exception as e:                               # noqa: BLE001
        # la ultima red: registrar no puede tumbar a quien lo llama
        print(f"[errores] no pude registrar '{code}': {e}", file=sys.stderr, flush=True)
        return None


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
              limit: int = 100, offset: int = 0) -> dict:
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
    facetas = {"nivel": {}, "code": {}, "origen": {}, "version": {}}
    for r in casan:
        for campo in facetas:
            v = r.get(campo)
            if v:
                facetas[campo][v] = facetas[campo].get(v, 0) + 1
    for campo, cuenta in facetas.items():
        facetas[campo] = dict(sorted(cuenta.items(), key=lambda kv: -kv[1]))
    return {
        "errores": casan[offset:offset + limit],
        "total": len(casan),
        "total_sin_filtro": len(filas),
        "facetas": facetas,
        "meses": [f.stem for f in _ficheros()],
        "donde": str(settings.errores_root()),
    }
