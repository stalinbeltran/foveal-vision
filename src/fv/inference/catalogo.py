"""QUE redes se guardan para inferir, y de donde salen sus pesos.

La regla, y la puso el dueno el 2026-08-31
------------------------------------------
**Los pesos de un run NO se guardan por defecto.** Solo se conservan --y solo se
pueden usar para inferir en la web app-- los de las redes que el dueno APRUEBA
una a una. Hoy es exactamente una: `demo-fov16-optimo`.

Por que la regla es esa y no "guardalos todos": hay **862 runs** en el repo de
datos (medido el 2026-08-31) y cada uno son **2,7 MB** de pesos (680 KB `best.pt`
+ 2,0 MB `last.pt`). Guardarlos todos son ~2,3 GB en un repo que hoy pesa 49 MB,
y git guarda **todas** las versiones que se commitean, no la ultima. La mayoria
de esos runs son puntos de un barrido: lo que se lee de ellos es el numero de su
tabla, no el modelo.

Y por que la lista es EXPLICITA y no una heuristica ("el mejor f1", "los de este
mes"): porque una heuristica decide sola y en silencio, y lo que esta en juego es
que git se llene de una y que la web app infiera con una red que nadie eligio.
Una lista se lee, se discute y se revierte.

Las dos preguntas que este modulo contesta, y son distintas
-----------------------------------------------------------
1. **¿esta red esta aprobada?** -> `esta_aprobada`, contra `inferencia.json` del
   repo de DATOS. Es la decision del dueno, versionada junto a los pesos que
   gobierna.
2. **¿hay pesos utilizables AHORA?** -> `checkpoint_de`, que mira dos sitios:

       antesala (`data/inferencia/<run>/`)  -> lo que llega mientras se entrena
       definitivo (`<repo de datos>/.../<run>/`) -> lo aprobado y commiteado

   **Gana la antesala**, a proposito: durante un entrenamiento la version buena
   es la que acaba de bajar, y mirar el modelo en marcha es justo para lo que
   `entrenar_vast.py` se trae los pesos en cada sonda. Cuando el run termina, la
   promocion mueve la antesala al definitivo y el orden deja de importar.

⚠ La antesala NO esta aprobada por estar ahi. Un run en antesala se puede usar
para MIRAR como va, y eso es distinto de "esta red se conserva": lo primero es
provisional y muere con la maquina, lo segundo es un commit. `promover` es la
frontera entre las dos cosas, y es explicita.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fv import settings
from fv.ioutils import read_json_retrying, write_json_atomic

# Los dos ficheros de un run, y SOLO estos dos. La misma pareja que la tercera
# excepcion del .gitignore del repo de datos, y por los mismos motivos:
#   best.pt  la MEJOR epoca segun el monitor. Es la que infiere.
#   last.pt  la MAS actualizada, con el estado entero (optimizador, contadores,
#            los tres generadores). Es la que permite CONTINUAR.
# Cualquier otro nombre se rechaza en la puerta: un endpoint que acepta un nombre
# de fichero libre acepta `../../algo`, y ademas dejaria en la antesala cosas que
# la promocion no sabria que hacer con ellas.
PESOS = ("best.pt", "last.pt")

# El que infiere. `last.pt` lleva ademas el estado del optimizador y NO es el
# modelo que se prueba: la seleccion por el monitor es lo que hace a `best.pt`
# comparable con los numeros publicados.
CHECKPOINT_INFERENCIA = "best.pt"


class CatalogoError(ValueError):
    def __init__(self, code: str, message: str, hint: str):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def catalogo_path() -> Path:
    """`inferencia.json`, en la RAIZ del repo de datos.

    Ahi y no en el repo de codigo porque gobierna unos ficheros que viven ahi:
    la lista y los pesos que nombra tienen que viajar y revertirse juntos. Un
    catalogo en el repo de codigo podria nombrar pesos que ese clon no tiene.
    """
    return settings.data_root() / "inferencia.json"


def leer() -> dict:
    """El catalogo, o uno vacio si no existe todavia.

    Vacio NO es un error: una maquina sin el fichero es una que aun no ha
    aprobado ninguna red, y eso es un estado legitimo (y el que tenia el
    proyecto hasta el 2026-08-30). Lo que no puede pasar es que un catalogo
    ilegible se lea como vacio en silencio -- eso desaprobaria todo sin decirlo,
    asi que se propaga con su razon.
    """
    p = catalogo_path()
    if not p.exists():
        return {"version": 1, "runs": {}}
    try:
        d = read_json_retrying(p)
    except Exception as e:
        raise CatalogoError(
            "catalogo_ilegible",
            f"{p} no se puede leer: {e}",
            "arreglalo o borralo: un catalogo roto NO se lee como vacio, porque "
            "eso desaprobaria todas las redes sin decirlo") from e
    d.setdefault("runs", {})
    return d


def aprobadas() -> list[str]:
    return sorted(leer()["runs"])


def esta_aprobada(run: str) -> bool:
    return run in leer()["runs"]


def entrada(run: str) -> dict | None:
    return leer()["runs"].get(run)


# --------------------------------------------------------------- la antesala

def staging_dir(run: str) -> Path:
    return settings.inference_staging_root() / run


def en_antesala(run: str) -> list[str]:
    """Que pesos hay en la antesala de este run, en orden de `PESOS`."""
    d = staging_dir(run)
    return [f for f in PESOS if (d / f).exists()]


def antesala_completa() -> dict[str, list[str]]:
    """Todo lo que hay en la antesala, por run. Vacio si no hay ni directorio."""
    raiz = settings.inference_staging_root()
    if not raiz.exists():
        return {}
    return {d.name: en_antesala(d.name)
            for d in sorted(raiz.iterdir()) if d.is_dir() and en_antesala(d.name)}


def guardar_en_antesala(run: str, fichero: str, datos: bytes) -> Path:
    """Deja `fichero` en la antesala de `run`, de forma ATOMICA.

    Atomica y con el temporal AL LADO del destino, no en /tmp. Las dos cosas
    tienen su motivo medido:

      - `best.pt` se lee MIENTRAS se reemplaza (la pantalla de revision usa el
        modelo con el entrenamiento en marcha). Con un rename atomico quien lee
        obtiene la version vieja o la nueva, nunca media. Es lo mismo que ya hace
        `entrenar_vast.traer`.
      - `os.replace` solo es atomico DENTRO del mismo sistema de ficheros. Con el
        temporal en /tmp --que en muchas maquinas es un tmpfs aparte-- daria
        EXDEV y el reemplazo fallaria, en el mejor caso ruidosamente.
    """
    if fichero not in PESOS:
        raise CatalogoError(
            "peso_desconocido",
            f"'{fichero}' no es un fichero de pesos de un run",
            f"solo se aceptan {list(PESOS)}: un nombre libre aceptaria rutas "
            f"como '../../algo' y dejaria en la antesala cosas que la promocion "
            f"no sabria mover")
    _comprobar_nombre(run)
    d = staging_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    destino = d / fichero
    tmp = d / f".{fichero}.parcial"
    tmp.write_bytes(datos)
    tmp.replace(destino)
    return destino


def _comprobar_nombre(run: str) -> None:
    """Un nombre de run es un nombre de directorio, no una ruta.

    Se comprueba aqui y no solo en el endpoint porque este modulo construye
    rutas con el: `staging_dir('../../etc')` saldria del arbol. La puerta del
    API ya filtra, pero una funcion que compone una ruta con un dato de fuera
    no puede confiar en que su unico llamador de hoy siga siendo el unico.
    """
    if not run or run != Path(run).name or run in (".", ".."):
        raise CatalogoError(
            "nombre_de_run_invalido",
            f"'{run}' no es un nombre de run valido",
            "un nombre de run es un nombre de directorio: sin '/', sin '..'")


# ------------------------------------------------------- de donde sale el peso

def checkpoint_de(run: str, runs_store, fichero: str = CHECKPOINT_INFERENCIA
                  ) -> tuple[Path | None, str | None]:
    """(ruta, origen) del peso utilizable de `run`, o (None, None).

    origen: 'antesala' (entrenando ahora) | 'catalogo' (aprobado y guardado).

    ⚠ Solo devuelve el definitivo si el run esta APROBADO, aunque el fichero
    este en disco. Un `.pt` sin aprobar en el repo de datos es algo que se
    colo --una copia a mano, un tar desempaquetado-- y servirlo haria que la app
    infiriera con una red que nadie eligio, que es justo lo que la lista existe
    para impedir.
    """
    p = staging_dir(run) / fichero
    if p.exists():
        return p, "antesala"
    if esta_aprobada(run):
        p = Path(runs_store.path(run)) / fichero
        if p.exists():
            return p, "catalogo"
    return None, None


# ------------------------------------------------------------- la promocion

def autochequeo(runs_store) -> list[dict]:
    """¿Puede la app cargar HOY todo lo que dice poder inferir?

    Una fila por red servible --las aprobadas y las que estan en la antesala--
    con `ok` y, si no, el `code`/`message`/`hint` de la negativa.

    ⚠ POR QUE AL ARRANCAR Y NO SOLO AL PULSAR
    ------------------------------------------
    Hay una clase de averia que NINGUN test puede encontrar: la que aparece
    porque el PROCESO lleva vivo desde antes que el artefacto. Un test corre
    siempre una version de todo; un servicio de larga vida, no.

    Paso el 2026-09-01: el servicio llevaba corriendo desde las 23:42 y
    `mask_channel` se commiteo a las 02:08. La red nueva no cargaba, y el fallo
    espero **8 horas** a que el dueno la eligiera en el movil -- que es el peor
    sitio para enterarse y el que hace que un sistema parezca de mala calidad.
    Cargarlas al arrancar convierte eso en una linea del log a los 3 segundos.

    ⚠ NO se niega a arrancar si algo falla, y es deliberado (R2: o degrada con un
    defecto DECLARADO, o falla antes de empezar). La app sirve datasets, runs,
    recorridos y estudios; que un `.pt` no cargue no puede tumbar todo eso. Se
    degrada --esa red no se puede usar-- y se DICE, en el log y en el payload de
    `GET /inference`, para que la pantalla lo marque en vez de fallar al pulsar.

    ⚠ Y no toca `MODEL_CACHE`: comprueba con `load_model`, que es el mismo codigo
    de carga sin el efecto de dejar N modelos residentes. Un chequeo que cambia
    el consumo de memoria del proceso es otra cosa distinta de un chequeo.
    """
    from fv.inference.checkpoint import CheckpointError, load_model  # noqa: PLC0415

    filas = []
    for run in sorted(set(aprobadas()) | set(antesala_completa())):
        ck, origen = checkpoint_de(run, runs_store)
        if ck is None:
            filas.append({"run": run, "origen": None, "ok": False,
                          "code": "sin_pesos",
                          "message": f"'{run}' esta en el catalogo y no tiene "
                                     f"pesos en ninguna parte",
                          "hint": "reentrenalo, o retiralo del catalogo con "
                                  "DELETE /inference/approved/<run>"})
            continue
        try:
            load_model(ck)
            filas.append({"run": run, "origen": origen, "ok": True})
        except CheckpointError as e:
            filas.append({"run": run, "origen": origen, "ok": False,
                          "code": e.code, "message": e.message, "hint": e.hint})
        except Exception as e:                          # noqa: BLE001
            # cualquier otra cosa tambien se declara: un autochequeo que se traga
            # una excepcion inesperada es un autochequeo que miente
            filas.append({"run": run, "origen": origen, "ok": False,
                          "code": "carga_fallida", "message": f"{type(e).__name__}: {e}",
                          "hint": "si el run es RECIENTE, mira primero si este "
                                  "proceso lleva vivo desde antes que el: "
                                  "reiniciar es gratis. Si no, el log del "
                                  "servicio tiene la traza"})
    return filas


def promover(run: str, runs_store, motivo: str = "", origen: str = "") -> dict:
    """Antesala -> repo de datos, y ADEMAS aprueba: es la misma decision.

    Promover es la orden de "esta red se guarda": copia `best.pt` y `last.pt` al
    directorio del run en el repo de datos y anota la entrada en el catalogo. No
    son dos pasos porque no son dos decisiones -- unos pesos en el repo de datos
    que nadie aprobo son 2,7 MB que git no suelta nunca y que la app no usaria.

    NO commitea. Igual que `fv-train`, imprime que hay que hacerlo: entrenar (o
    promover) no deberia escribir en el historial de nadie sin que se lo pidan.
    El comando exacto va en la respuesta.
    """
    _comprobar_nombre(run)
    hay = en_antesala(run)
    if not hay:
        raise CatalogoError(
            "antesala_vacia",
            f"no hay pesos en la antesala de '{run}'",
            f"sube {list(PESOS)} con PUT /inference/staging/{run}/<fichero>, o "
            f"entrena con la antesala como destino")
    destino = Path(runs_store.path(run))
    if not destino.exists():
        raise CatalogoError(
            "run_sin_directorio",
            f"'{run}' no tiene directorio en el repo de datos ({destino})",
            "los pesos acompanan a la descripcion del run (config, metrics, "
            "summary): promuevelos cuando el run exista, o trae antes su "
            "descripcion")
    copiados = []
    for f in hay:
        # mismo rename atomico que en la antesala: aqui tambien se puede estar
        # leyendo el fichero mientras se reemplaza
        tmp = destino / f".{f}.parcial"
        shutil.copyfile(staging_dir(run) / f, tmp)
        tmp.replace(destino / f)
        copiados.append(f)

    cat = leer()
    import datetime as dt
    cat["runs"][run] = {
        "cuando": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ficheros": copiados,
        "motivo": motivo or "aprobada para inferencia",
        "origen": origen or "promocion",
    }
    write_json_atomic(catalogo_path(), cat)
    return {
        "run": run,
        "copiados": copiados,
        "destino": str(destino),
        "catalogo": str(catalogo_path()),
        # lo que falta para que sobreviva a rehacer la maquina, textual: "lo que
        # no esta empujado, no existe"
        "commit": f"cd {settings.data_root()} && git add -A && "
                  f"git commit -m 'pesos de {run} para inferencia' && git push",
    }


def retirar(run: str) -> dict:
    """Saca a `run` del catalogo. NO borra sus pesos del disco.

    Separado a proposito: retirar es una decision reversible (se vuelve a
    aprobar) y borrar no lo es. Y los pesos que ya estan commiteados no se van
    del historial de git por borrarlos del arbol, asi que borrarlos daria una
    sensacion de limpieza que no es cierta.
    """
    cat = leer()
    if run not in cat["runs"]:
        raise CatalogoError("no_aprobada", f"'{run}' no esta en el catalogo",
                            f"las aprobadas son {aprobadas() or '(ninguna)'}")
    cat["runs"].pop(run)
    write_json_atomic(catalogo_path(), cat)
    return {"run": run, "aprobadas": sorted(cat["runs"])}


def limpiar_antesala(run: str) -> list[str]:
    """Borra la antesala de un run. Lo definitivo no se toca."""
    _comprobar_nombre(run)
    d = staging_dir(run)
    borrados = []
    if d.exists():
        for f in sorted(d.iterdir()):
            if f.is_file():
                f.unlink()
                borrados.append(f.name)
        d.rmdir()
    return borrados
