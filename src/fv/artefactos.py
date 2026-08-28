"""Resolucion de rutas de artefactos de estudio: plano -> archivo fechado -> legado.

Por que hacen falta TRES sitios y no uno
----------------------------------------
Los artefactos de estudio se escriben ahora en `foveal-vision-data` (settings.
`data_root`), pero lo que ya existe esta repartido en dos formas distintas:

  1. **fechado**, `<data>/<anio>/<mes>/sweeps/<recorrido>/runs/<run>/` -- lo que
     dejo la migracion Y lo que se escribe DE AHORA EN ADELANTE. Un run vive
     dentro de su recorrido y de su mes, y esa relacion es estructura de
     directorios a proposito. `index.json` es el mapa de lo migrado: sin el no se
     puede encontrar un run viejo sin saber su mes.
  2. **plano**, `<data>/runs/<run>/` -- la forma que `RunStore.path()` uso
     siempre, y en la que todavia hay cosas escritas. Ya NO se escribe aqui: ver
     `destino_agrupado`.
  3. **legado**, `<foveal-vision>/runs/<run>/` -- lo que este repo todavia tiene
     y la migracion copio, mas lo que se escribio DESPUES de aquella copia.

Se busca **plano -> fechado -> legado** y se ESCRIBE siempre en (1). Que se lea
lo plano primero es lo que hace que lo ya escrito ahi siga apareciendo mientras
no se migre; que se escriba fechado es lo que hace que no crezca.

⚠ Esto es una escalera para poder migrar sin parar el mundo, no un diseño
permanente. Cuando (3) se vacie, `legado()` se borra y queda una escalera de dos.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fv import settings


@lru_cache(maxsize=8)
def _indice_de(raiz: str) -> dict:
    try:
        return json.loads((Path(raiz) / "index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _indice() -> dict:
    """El mapa del archivo fechado. Cacheado: se lee por cada `path()` y son 851
    entradas. Si falta o esta roto se devuelve vacio -- el archivo pasa a no
    existir para el resolutor, que es degradar, no romper.

    ⚠ La cache va por RAIZ, no global. Cacheada sin argumentos, el primer repo
    que se mirase se quedaba pegado: un test que apunta FV_DATA_ROOT a un
    temporal seguia viendo el indice del repo REAL, y `resolver` devolvia rutas
    de medidas de verdad. Lo mismo valdria para cualquier proceso que cambie de
    raiz en caliente.
    """
    raiz = settings.data_archive_root()
    return _indice_de(str(raiz)) if raiz is not None else {}


def _archivado(clase: str, nombre: str) -> Path | None:
    entrada = (_indice().get(clase) or {}).get(nombre)
    if not entrada:
        return None
    raiz = settings.data_archive_root()
    return (raiz / entrada["path"]) if raiz else None


def resolver(clase: str, nombre: str, plano: Path) -> Path:
    """La ruta donde ESTA `nombre`, o `plano` si no esta en ningun sitio.

    `plano` es tambien la respuesta cuando no existe todavia: crear siempre
    ocurre en la forma nueva.
    """
    if plano.exists():
        return plano
    arch = _archivado(clase, nombre)
    if arch is not None and arch.exists():
        return arch
    # ... y lo agrupado HOY, que aun no esta en `index.json`. Sin este escalon,
    # `path()` de un recorrido recien creado bajo el mes de su estudio devolvia
    # la ruta PLANA (que no existe), y el `set_state` que sigue a `create`
    # escribia en un directorio inexistente. `path()` no recibe el estudio, asi
    # que no puede recalcular el destino agrupado: tiene que encontrarlo.
    reciente = _dir_por_mes(clase, nombre)
    if reciente is not None:
        return reciente
    legado = settings.project_root() / clase / nombre
    if legado.exists():
        return legado
    return plano


def nombres(clase: str, plano: Path) -> list[str]:
    """Todos los nombres visibles de esa clase, sin repetir, en el orden de
    prioridad de `resolver`."""
    vistos: dict[str, None] = {}
    for d in (plano, settings.project_root() / clase):
        if d.exists():
            for x in sorted(d.iterdir()):
                if x.is_dir():
                    vistos.setdefault(x.name, None)
    for n in (_indice().get(clase) or {}):
        vistos.setdefault(n, None)
    # ... y lo escrito HOY agrupado por mes, que todavia no esta en `index.json`
    # (lo escribe el migrador, despues). Sin esto, un estudio recien creado no
    # aparece en su propia lista: se escribe en <mes>/studies/ y se buscaba solo
    # en las tres fuentes de arriba.
    for d in _dirs_por_mes(clase):
        vistos.setdefault(d.name, None)
    return list(vistos)


def _dirs_por_mes(clase: str) -> list[Path]:
    """Los directorios de `clase` bajo las carpetas de mes del repo de datos.

    Incluye los runs que viven DENTRO de un recorrido
    (`<mes>/sweeps/<rec>/runs/<run>`), que es donde caen los de un recorrido
    archivado.
    """
    raiz = settings.data_archive_root() or settings.data_root()
    if not raiz.exists():
        return []
    out: list[Path] = []
    for anio in sorted(p for p in raiz.iterdir()
                       if p.is_dir() and p.name.isdigit() and len(p.name) == 4):
        for mes in sorted(p for p in anio.iterdir() if p.is_dir()):
            base = mes / clase
            if base.is_dir():
                out += [d for d in sorted(base.iterdir()) if d.is_dir()]
            if clase == "runs" and (mes / "sweeps").is_dir():
                for rec in sorted((mes / "sweeps").iterdir()):
                    if (rec / "runs").is_dir():
                        out += [d for d in sorted((rec / "runs").iterdir())
                                if d.is_dir()]
    return out


# --- Agrupar por estudio al ESCRIBIR ------------------------------------------
#
# Lo de arriba resuelve donde ESTA algo. Esto decide donde se CREA, y es una
# decision distinta: el usuario pidio no ver un mismo estudio disperso en varias
# carpetas de mes solo porque unos recorridos corrieron al dia siguiente. Asi que
# el mes lo elige EL ESTUDIO al crearse, y lo hereda todo lo suyo.
#
# El mes AGRUPA para poder leer el directorio; no es una linea temporal de cada
# run. Un recorrido lanzado el dia 1 del mes siguiente se queda con su estudio, y
# un run vive DENTRO de su recorrido -- la misma forma que dejo la migracion, asi
# que lo nuevo y lo archivado tienen una sola estructura y no dos.

MESES_ES = {
    1: "01-enero", 2: "02-febrero", 3: "03-marzo", 4: "04-abril",
    5: "05-mayo", 6: "06-junio", 7: "07-julio", 8: "08-agosto",
    9: "09-septiembre", 10: "10-octubre", 11: "11-noviembre", 12: "12-diciembre",
}


def mes_actual() -> str:
    """`2026/08-agosto` para hoy, en UTC."""
    import datetime as dt
    d = dt.datetime.now(dt.UTC)
    return f"{d.year}/{MESES_ES[d.month]}"


def _mes_de(ruta: Path) -> str | None:
    """El `<anio>/<mes>` de una ruta ya archivada, o None si es plana.

    Se reconoce por la forma: `.../<anio>/<mes>/<clase>/<nombre>`, con el anio
    en digitos. Una ruta plana (`<data>/runs/<run>`) no la cumple.
    """
    p = ruta.parts
    for i in range(len(p) - 3):
        if p[i].isdigit() and len(p[i]) == 4 and p[i + 2] in ("runs", "sweeps", "studies"):
            return f"{p[i]}/{p[i + 1]}"
    return None


def _dir_por_mes(clase: str, nombre: str) -> Path | None:
    """El directorio de `nombre` bajo una carpeta de mes, si esta ahi.

    Mira el disco y no solo `index.json`, porque lo creado HOY todavia no esta
    en el indice: sin esto, un estudio recien creado no agruparia a sus propios
    recorridos -- justo el caso que el agrupamiento existe para cubrir.
    """
    for d in _dirs_por_mes(clase):
        if d.name == nombre:
            return d
    return None


def _mes_por_sus_recorridos(estudio: str) -> str | None:
    """El mes donde ya vive algun recorrido de `estudio`, o None.

    Un estudio de aqui no siempre es un directorio `studies/<nombre>/`: solo el
    motor OAT del API lo crea. Los `scripts/estudio_*.py` -- que es como se ha
    lanzado TODO lo que se ha medido en este proyecto -- nombran su estudio en el
    `spec.json` de cada recorrido y no crean el artefacto nunca.

    Buscar el mes SOLO por el directorio del estudio dejaba entonces sin mes a
    todo lo lanzado por script. Medido el 2026-08-28: el tanteo `do-t` de
    `dropout-2026-08-28` se escribio en `<data>/sweeps/do-t/` y sus runs en
    `<data>/runs/`, o sea en la raiz plana y sin un solo aviso.

    Se recorren en orden de mes (`_dirs_por_mes` va ordenado), asi que gana el
    mas antiguo: es el criterio 3 del README del repo de datos -- un estudio se
    fecha por su recorrido mas viejo, no por el ultimo que se lanzo.
    """
    for d in _dirs_por_mes("sweeps"):
        try:
            spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if spec.get("study") == estudio:
            mes = _mes_de(d)
            if mes:
                return mes
    return None


def mes_del_estudio(estudio: str) -> str | None:
    """El mes bajo el que vive un estudio, o None si no hay ni rastro de el.

    Por su directorio si lo tiene; si no, por el de sus recorridos, que es la
    unica huella que deja un estudio lanzado por script.
    """
    d = _archivado("studies", estudio) or _dir_por_mes("studies", estudio)
    if d is not None:
        return _mes_de(d)
    return _mes_por_sus_recorridos(estudio)


def destino_agrupado(clase: str, nombre: str, *, estudio: str | None = None,
                     recorrido: str | None = None) -> Path | None:
    """Donde CREAR `nombre`, bajo su carpeta de mes.

    ⚠ Antes esto devolvia None en cuanto no sabia a que estudio pertenecia algo,
    con este motivo: "es la respuesta honesta, en vez de inventarse una carpeta
    de mes que separaria lo que deberia ir junto". El razonamiento tenia un
    agujero, y costo un estudio entero escrito donde no debia (§
    `_mes_por_sus_recorridos`): **la alternativa a la carpeta de mes no era "no
    separar", era la RAIZ PLANA** -- un tercer sitio, sin fecha, en el que el
    estudio queda igual de separado de los demas y ademas sin decirlo.

    Lo que hay que conservar es *un estudio en UN mes*, y eso se conserva
    **heredando** el mes (`mes_del_estudio`), no renunciando a tener uno. Asi que
    ahora solo se devuelve None en el unico caso en que el mes de verdad
    separaria: un run cuyo recorrido esta plano. Ese run se queda con su
    recorrido -- peor sitio, pero no OTRO sitio.
    """
    raiz = settings.data_archive_root() or settings.data_root()

    if clase == "runs":
        if recorrido:
            # un run vive DENTRO de su recorrido: hereda su mes sin calcular nada
            d = _archivado("sweeps", recorrido) or _dir_por_mes("sweeps", recorrido)
            return (d / "runs" / nombre) if d is not None else None
        # un run suelto (un benchmark) no se inventa un recorrido, pero si tiene
        # mes: sin recorrido ni estudio con los que agruparse, el suyo es el de
        # hoy. Es lo que el README del repo de datos ya dice de los sueltos.
        return raiz / mes_actual() / "runs" / nombre

    if clase == "sweeps":
        # el mes de su estudio si ese estudio ya tiene uno; si no, este recorrido
        # lo ESTRENA -- y los siguientes del mismo estudio lo heredaran de el,
        # aunque se lancen el mes que viene
        mes = (mes_del_estudio(estudio) if estudio else None) or mes_actual()
        return raiz / mes / "sweeps" / nombre

    if clase == "studies":
        # un estudio ESTRENA mes... salvo que sus propios recorridos ya hayan
        # elegido uno: por script el recorrido se crea ANTES que el estudio (o el
        # estudio no se crea nunca), y el orden no puede partir en dos lo mismo
        return raiz / (mes_del_estudio(nombre) or mes_actual()) / "studies" / nombre

    return None
