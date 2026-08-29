"""Revision A OJO de lo que la red detecta: que rangos ya se miraron, y que
imagenes quedaron marcadas para volver.

Por que existe
--------------
La metrica de tarea (`fv.task`) contesta *cuanto* acierta el modelo sobre un
split. No contesta *que* esta fallando: para eso hay que mirar las cajas encima
de la imagen. Esto es el registro de esa mirada, y su valor entero es
acumulativo -- lo util no es la sesion de hoy, es *«estas 40 ya las vi, ensename
otras»* dentro de tres semanas.

Los DOS ficheros, que son dos cosas con reglas de edicion opuestas (R8)
----------------------------------------------------------------------
  1. **historial**, `reviews/<anio>-<mes>.jsonl` -- una linea por rango mirado.
     SOLO SE AÑADE. Es lo que contesta "que he revisado ya" y "que mire ayer".
     Reescribir una linea es perder justo lo que hace util el fichero.
  2. **estado**, `reviews/marks.json` -- las imagenes marcadas para volver. SE
     REESCRIBE: marcar y desmarcar es un interruptor, y la pregunta que contesta
     es *«que hay pendiente HOY»*, no *«que marque alguna vez»*.

Meterlos en el mismo sitio obligaria a leer todo el historial y ordenar por
fecha para saber que hay marcado ahora -- que es exactamente el fallo que R8
describe.

Una mirada queda registrada aunque no se marque nada
----------------------------------------------------
El historial lo escribe el ENDPOINT que infiere, no un boton de "guardar". Es la
misma decision que `fv.task.record_holdout_touch` ("una mirada cacheada sigue
siendo una mirada"): lo que se registra es que alguien MIRO, y una mirada que
depende de que el usuario pulse algo despues es justo la que no se registra.

Por eso cada linea guarda tambien la `source`: si algun dia se revisa desde aqui
un holdout, queda rastro, en vez de haber abierto una puerta a los datos
reservados que no pasa por ningun libro.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fv import settings
from fv.ioutils import read_json_retrying, read_text_retrying, write_json_atomic

MARKS = "marks.json"


def _root() -> Path:
    d = settings.reviews_root()
    d.mkdir(parents=True, exist_ok=True)
    return d


def donde_se_guarda() -> dict:
    """Donde caen estos ficheros y si ESO se commitea en algun sitio.

    Va en el payload a proposito: `data_root()` cae al repo de codigo cuando el
    repo de datos no esta clonado (R2, degradar), y ahi la revision se escribe
    igual pero no la guarda nadie. Sin decirlo, el usuario revisa 200 imagenes y
    las pierde al rehacer la maquina, sin un solo error por el camino -- el fallo
    silencioso que este proyecto rechaza en todas partes.
    """
    root = settings.reviews_root()
    return {"path": str(root),
            "in_data_repo": settings.data_root() != settings.project_root()}


def _mes(cuando: datetime) -> Path:
    return _root() / f"{cuando:%Y-%m}.jsonl"


def _repetida(line: dict, cuando: datetime, ventana_s: int = 60) -> bool:
    """¿Es esta linea la MISMA mirada que la ultima, hace nada?"""
    path = _mes(cuando)
    if not path.exists():
        return False
    lineas = [ln for ln in read_text_retrying(path).splitlines() if ln.strip()]
    if not lineas:
        return False
    try:
        ult = json.loads(lineas[-1])
        antes = datetime.fromisoformat(ult["when"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return False
    mismo = all(ult.get(k) == line.get(k) for k in
                ("window_dataset", "split", "run", "offset", "count"))
    return mismo and (cuando - antes).total_seconds() < ventana_s


def record_review(*, window_dataset: str, split: str, source: str, run: str,
                  indices: list[int], offset: int, knobs: dict | None = None,
                  when: datetime | None = None) -> dict:
    """Añade UNA linea: este rango, de este split, mirado con este run."""
    cuando = when or datetime.now(timezone.utc)
    line = {
        "when": cuando.isoformat(timespec="seconds"),
        "window_dataset": window_dataset, "split": split, "source": source,
        "run": run, "offset": offset, "count": len(indices),
        "indices": [int(i) for i in indices],
        "knobs": knobs or {},
    }
    # Mirar el MISMO rango dos veces seguidas en menos de un minuto no es una
    # revision nueva: es un remontaje del componente, un doble toque, o mover un
    # slider. Se descarta porque este fichero se COMMITEA -- el ruido no solo
    # ensucia la lista que se lee en el movil, engorda el repo de datos para
    # siempre. Un repaso de verdad (volver a ese rango mañana) si deja linea.
    if _repetida(line, cuando):
        return line
    path = _mes(cuando)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


def _todas() -> list[dict]:
    out: list[dict] = []
    root = settings.reviews_root()
    if not root.exists():
        return out
    for f in sorted(root.glob("*.jsonl")):
        for ln in read_text_retrying(f).splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    # una linea rota no puede tumbar el historial entero: se
                    # salta. Es la misma regla que el registry del coordinador.
                    continue
    return out


def reviews(*, window_dataset: str | None = None, split: str | None = None,
            since_days: int | None = None) -> list[dict]:
    """El historial, lo mas reciente primero."""
    corte = None
    if since_days is not None:
        corte = datetime.now(timezone.utc) - timedelta(days=since_days)
    out = []
    for r in _todas():
        if window_dataset and r.get("window_dataset") != window_dataset:
            continue
        if split and r.get("split") != split:
            continue
        if corte is not None:
            try:
                if datetime.fromisoformat(r["when"]) < corte:
                    continue
            except (KeyError, ValueError):
                continue
        out.append(r)
    return sorted(out, key=lambda r: r.get("when", ""), reverse=True)


def reviewed_indices(window_dataset: str, split: str) -> list[int]:
    """Que indices de este split ya se han mirado ALGUNA vez."""
    vistos: set[int] = set()
    for r in reviews(window_dataset=window_dataset, split=split):
        vistos.update(int(i) for i in r.get("indices", []))
    return sorted(vistos)


def next_unreviewed_offset(indices: list[int], vistos: list[int],
                           count: int) -> int:
    """El primer offset cuyo rango tiene ALGO sin mirar.

    Devuelve 0 cuando no queda nada por ver -- y el que llama lo distingue
    mirando `pending`, no adivinando por el 0. Un "no queda nada" que se lee como
    "empieza por el principio" es un bucle infinito para el que revisa.
    """
    pend = set(vistos)
    for off in range(0, len(indices), max(1, count)):
        if any(i not in pend for i in indices[off:off + count]):
            return off
    return 0


# --- marcas: ESTADO, se reescribe -------------------------------------------

def _clave(window_dataset: str, split: str, index: int) -> str:
    return f"{window_dataset}|{split}|{int(index)}"


def marks() -> dict:
    p = settings.reviews_root() / MARKS
    return read_json_retrying(p) if p.exists() else {}


def mark_list() -> list[dict]:
    """Las marcas vivas, lo mas reciente primero."""
    return sorted(marks().values(), key=lambda m: m.get("when", ""), reverse=True)


def set_mark(*, window_dataset: str, split: str, index: int, marked: bool,
             note: str = "", source: str = "", run: str = "") -> dict:
    """Marca o desmarca. Devuelve el estado resultante de ESA imagen."""
    todas = marks()
    k = _clave(window_dataset, split, index)
    if marked:
        todas[k] = {"window_dataset": window_dataset, "split": split,
                    "index": int(index), "note": note, "source": source,
                    "run": run,
                    "when": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    else:
        todas.pop(k, None)
    # `_root()` ANTES de escribir: crea el directorio. `write_json_atomic` no
    # crea padres, asi que marcar sin haber revisado nunca reventaba con un
    # FileNotFoundError sobre el .tmp -- el unico orden en que este endpoint se
    # puede llamar primero, y justo el que no cubria el camino feliz.
    destino = _root() / MARKS
    write_json_atomic(destino, todas)
    return {"key": k, "marked": bool(marked), "marks": len(todas)}


def marked_in(window_dataset: str, split: str) -> list[int]:
    return sorted(m["index"] for m in marks().values()
                  if m.get("window_dataset") == window_dataset
                  and m.get("split") == split)
