"""Los casos en que una red FALLA, como dataset de ventanas propio.

Que hace, en una linea: pasa una red (E) por todas las imagenes de un dataset de
ventanas (B) usando inferencia de imagen completa (F), puntua cada imagen a nivel
de PARRAFO, y escribe un B NUEVO con las peores. El resultado es un dataset de
ventanas de pleno derecho -- mismo `windows.npz`, mismo `manifest.json`, mismo
`split.json` -- asi que `fv-train` puede entrenar sobre el y la web app puede
mirarlo, sin que nada sepa que salio de aqui.

Por que un dataset y no un informe
----------------------------------
Un informe de "estas 87 imagenes salieron mal" se lee una vez y se pierde. Un
dataset se entrena, se vuelve a inferir, se compara consigo mismo dentro de seis
meses y se commitea. La peticion era esa: que los fallos se puedan USAR.

QUE ES UN ERROR, y esta escrito antes de mirar ningun numero
------------------------------------------------------------
Un **error** es un parrafo que la red no encontro (`fn`) o uno que se invento
(`fp`), emparejando prediccion y verdad por IoU >= `iou_threshold`. Es
exactamente `fv.metrics.paragraph_f1`, que es la metrica que IMPORTA en este
proyecto (protocolo.md §2) -- no se define aqui un segundo numero.

**Peor = mas errores.** Los empates se rompen, en este orden: menor f1 de
parrafo, menor IoU medio de los emparejados (sin emparejar nada = lo peor de
todo), e indice de imagen. Los dos ultimos escalones existen solo para que el
orden sea DETERMINISTA: un criterio que cambia de resultado al repetirlo no
decide nada.

⚠ EL CHOQUE CON EL CONTRATO ⑬, declarado y no roto en silencio
---------------------------------------------------------------
`docs/organizacion.md` ⑬ dice que la verdad de parrafos sale de la FUENTE (A) y
que si la fuente no esta se falla, «nunca se puntua contra las etiquetas de
ventana». Este modulo lo respeta por defecto (`verdad="fuente"`).

Pero tiene un segundo camino, `verdad="ventanas"`, que hay que PEDIR: reconstruye
las cajas verdaderas desde `y` del propio npz. No es lo que ⑬ prohibe -- ⑬
prohibe puntuar contra las etiquetas de ventana *como si fueran* parrafos; esto
recompone los parrafos y luego puntua igual que siempre. Se puede hacer porque el
extractor guarda la esquina VERBATIM: `y[ci] = (1, (cx-wx0)/n, (cy-wy0)/n)`, y
`wx0 + y*n` la devuelve.

Y aun asi es un camino DEGRADADO, con su defecto medido (R2: o degrada con un
defecto declarado, o falla antes de empezar):

- **Un parrafo cortado por el borde de la imagen no se recupera.** Su esquina cae
  fuera de la rejilla, ninguna ventana la ve, y no hay nada de donde sacarla.
  Medido el 2026-09-02 sobre `dirty1000-80px-16px-r20260827`: **989 de 1000**
  imagenes se reconstruyen exactas; las 11 restantes tienen algun parrafo que se
  sale por arriba o por abajo. Esas quedan marcadas `gt_completa: false` y
  **fuera de la seleccion** por defecto, porque su cuenta de errores no es
  creible: lo que la red detecto ahi se contaria como invento.
- Por que existe el camino: la fuente de este dataset se perdio al rehacer la
  maquina (sus PNG nunca entraron en git) y el `windows.npz` si esta commiteado.
  Sin este camino, los fallos de las tres redes aprobadas no se pueden mirar.

Lo que se reconstruye queda escrito en el manifest del dataset resultante
(`fallidos.verdad`), para que nadie lea dentro de un ano estos numeros como si
se hubieran medido contra A.

La precision de la reconstruccion, medida
-----------------------------------------
`TOL_PX` separa "la misma esquina vista por dos ventanas" de "dos esquinas
distintas". Los dos lados estan medidos el 2026-09-02 sobre ese dataset:

- dispersion de una misma esquina vista desde ventanas distintas: **0,0 px
  exactos** (con ventana 16, potencia de dos, la ida y vuelta en float32 no
  pierde nada; con otro tamano perderia del orden de 1e-5 px).
- separacion minima entre dos esquinas DISTINTAS del mismo tipo en una misma
  imagen: **9,52 px** (percentil 1: 13,14).

O sea que cualquier tolerancia entre ~1e-4 y ~4 px da el mismo resultado. 0,25 px
esta en mitad del hueco por escala logaritmica, y no es una eleccion fina.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fv import settings
from fv.datasets.loader import SourceDataset, SourceError
from fv.inference.catalogo import CHECKPOINT_INFERENCIA, checkpoint_de, esta_aprobada
from fv.inference.checkpoint import load_model
from fv.inference.predict import predict_image
from fv.ioutils import write_json_atomic
from fv.metrics import CORNER_NAMES, paragraph_f1
from fv.training.registry import RunStore
from fv.windows.store import WindowDatasetStore

# El hueco entre "la misma esquina" y "dos esquinas": ver el docstring. Las dos
# fronteras estan medidas, y esta constante vive lejos de las dos.
TOL_PX = 0.25

SPLIT_NAMES = ("train", "val", "test")
POLITICAS_SPLIT = ("conservar", "rehacer", "train")

# Como se acorta el nombre de un run para bautizar su dataset de fallos. Es
# COSMETICA: la procedencia completa va en el manifest, asi que acortar de mas no
# pierde informacion, solo legibilidad. Se quitan por delante y por detras.
PREFIJOS_RUIDO = ("demo-", "fov16-", "fov-")
SUFIJOS_RUIDO = ("-p20", "-p40", "-optimo-mask")
SUFIJO_FALLIDOS = "-fallidos"


class FallidosError(ValueError):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


@dataclass
class Criterio:
    """Todo lo que decide QUE entra en el dataset. Viaja al manifest entero.

    Los cinco primeros son los knobs de F (inferencia): no son propiedades de la
    red ni del dataset, son de la lectura que se hace de la salida, y el numero
    no significa nada sin ellos -- por eso se guardan con el resultado, igual que
    hace `predict_image`.
    """
    threshold: float = 0.5
    stride: int | None = None
    nms_radius: float | None = None
    min_size: float | None = None
    iou_threshold: float = 0.5
    # ...y los tres que deciden la seleccion
    min_errores: int = 1
    max_imagenes: int = 0            # 0 = sin tope
    incluir_gt_parcial: bool = False


# --------------------------------------------------------------- el nombre

def nombre_corto(run: str) -> str:
    """'fov16-edge-p20' -> 'edge'. Cosmetico y por eso reversible: el manifest
    guarda el nombre entero del run."""
    corto, cambiando = run, True
    while cambiando:
        cambiando = False
        for p in PREFIJOS_RUIDO:
            if corto.startswith(p) and len(corto) > len(p):
                corto, cambiando = corto[len(p):], True
                break
    for s in SUFIJOS_RUIDO:
        if corto.endswith(s) and len(corto) > len(s):
            corto = corto[:-len(s)]
            break
    return corto or run


def nombre_dataset(run: str) -> str:
    return nombre_corto(run) + SUFIJO_FALLIDOS


# ------------------------------------------------------- la verdad, dos vias

def _agrupa(puntos: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Une los puntos que son la MISMA esquina vista desde ventanas distintas.

    Enlace simple con `TOL_PX`. Encadenar seria un riesgo si las esquinas
    verdaderas estuviesen a menos de dos tolerancias la una de la otra; estan a
    9,5 px (medido) y la tolerancia es 0,25.
    """
    grupos: list[list[tuple[float, float]]] = []
    for px, py in sorted(puntos):
        for g in grupos:
            if abs(g[0][0] - px) <= TOL_PX and abs(g[0][1] - py) <= TOL_PX:
                g.append((px, py))
                break
        else:
            grupos.append([(px, py)])
    return [(float(np.mean([p[0] for p in g])), float(np.mean([p[1] for p in g])))
            for g in grupos]


def _cerca(pts: list[tuple[float, float]], x: float, y: float) -> bool:
    return any(abs(a - x) <= TOL_PX and abs(b - y) <= TOL_PX for a, b in pts)


def verdad_de_ventanas(arrays: dict, window_size: int) -> dict[int, dict]:
    """Recompone los parrafos verdaderos desde las etiquetas de esquina de B.

    Devuelve {indice de A: {"cajas": [(x0,y0,x1,y1)...], "completa": bool}}.

    `completa` es False cuando el numero de cajas recompuestas no coincide con el
    numero de esquinas de cada tipo: falta alguna esquina, o sea que hay un
    parrafo cortado por el borde de la imagen y su caja no se puede saber. NO se
    inventa una caja parcial -- una caja a medias puntuaria como error de la red
    un fallo del dato (formatos.md §2: ausente no es cero).
    """
    y = arrays["y"]
    sample_idx = arrays["sample_idx"]
    window_xy = arrays["window_xy"]
    n = float(window_size)

    ax = window_xy[:, None, 0] + y[:, :, 1] * n
    ay = window_xy[:, None, 1] + y[:, :, 2] * n
    existe = y[:, :, 0] >= 0.5

    por_imagen: dict[int, list[list[tuple[float, float]]]] = {}
    for fila in np.nonzero(existe.any(axis=1))[0]:
        s = int(sample_idx[fila])
        acc = por_imagen.setdefault(s, [[], [], [], []])
        for ci in range(4):
            if existe[fila, ci]:
                acc[ci].append((float(ax[fila, ci]), float(ay[fila, ci])))

    out: dict[int, dict] = {}
    for s in np.unique(sample_idx):
        s = int(s)
        acc = por_imagen.get(s, [[], [], [], []])
        C = [_agrupa(v) for v in acc]                 # TL TR BR BL
        cajas = []
        for x0, y0 in C[0]:
            for x1, y1 in C[2]:
                if x1 <= x0 or y1 <= y0:
                    continue
                if _cerca(C[1], x1, y0) and _cerca(C[3], x0, y1):
                    cajas.append((x0, y0, x1, y1))
        esperadas = {len(c) for c in C}
        completa = len(esperadas) == 1 and len(cajas) == next(iter(esperadas))
        out[s] = {"cajas": cajas, "completa": bool(completa),
                  "esquinas": {c: C[i] for i, c in enumerate(CORNER_NAMES)}}
    return out


def verdad_de_fuente(source_id: str, indices: list[int],
                     kinds: set[str]) -> dict[int, dict]:
    """La verdad como la lee `fv.task`: los bloques de A, filtrados por kind.

    El filtro por kind NO es opcional, por la misma razon que alli: un dataset
    extraido de parrafos no se puntua contra lineas.
    """
    # el orden de esquinas es SEMANTICA, no adorno (formatos.md §4.1): se importa
    # del extractor en vez de reescribirlo, que es como nacen dos convenios que
    # se creen el mismo
    from fv.windows.extract import _corners_of

    source = SourceDataset(source_id)
    out = {}
    for i in indices:
        s = source.sample_at(int(i))
        cajas = [tuple(b.bbox) for b in s.blocks if b.kind in kinds]
        esquinas: dict[str, list] = {c: [] for c in CORNER_NAMES}
        for caja in cajas:
            for ci, pt in enumerate(_corners_of(caja)):
                esquinas[CORNER_NAMES[ci]].append(pt)
        out[int(i)] = {"cajas": cajas, "completa": True, "esquinas": esquinas}
    return out


# ------------------------------------------------------------- la evaluacion

def esquinas_acertadas(predichas: list[dict], verdaderas: dict[str, list],
                       tol_px: float) -> dict:
    """¿Vio la red las esquinas, aunque luego las emparejara mal?

    ⚠ POR QUE ESTO VA EN EL DATASET Y NO ES UN EXTRA
    -------------------------------------------------
    "La red falla en esta imagen" son DOS averias distintas con arreglos
    opuestos, y sin este numero no se distinguen:

      - **no vio la esquina** -> es la red (C/D): mas capacidad, mas datos, otro
        punto del barrido;
      - **la vio y el emparejado la junto mal** -> es F, y concretamente el
        `_reconstruct` de `fv.inference.predict`, que es un heredado voraz TL->BR
        y lleva escrito en su propio docstring que es «the place to touch if
        paragraphs come out wrong while corners come out right».

    Medido el 2026-09-02 sobre las tres redes aprobadas: el f1 de VENTANA sale
    0,947-0,954 (coincide con el publicado) y aun asi 35-49 % de las imagenes
    tienen algun error de parrafo. Mirando cuatro a ojo, las cajas predichas unen
    el TL de un parrafo con el BR de OTRO. O sea que la segunda averia domina, y
    un dataset que no lo dijera mandaria a reentrenar redes que estan bien.

    Emparejamiento voraz por cercania dentro de `tol_px`, por tipo de esquina.
    """
    out = {"tp": 0, "fp": 0, "fn": 0}
    for cname in CORNER_NAMES:
        pred = sorted((d for d in predichas if d["corner"] == cname),
                      key=lambda d: -d["score"])
        libres = list(verdaderas.get(cname, []))
        for d in pred:
            mejor, mejor_j = None, -1
            for j, (vx, vy) in enumerate(libres):
                dist = ((d["x"] - vx) ** 2 + (d["y"] - vy) ** 2) ** 0.5
                if dist <= tol_px and (mejor is None or dist < mejor):
                    mejor, mejor_j = dist, j
            if mejor_j >= 0:
                libres.pop(mejor_j)
                out["tp"] += 1
            else:
                out["fp"] += 1
        out["fn"] += len(libres)
    return out


def _split_por_imagen(arrays: dict) -> dict[int, int]:
    """{indice de A: 0|1|2}. El split es POR IMAGEN por contrato (formatos.md
    §4.1: «jamas por ventana»), asi que la primera ventana de cada imagen lo
    dice entero."""
    out: dict[int, int] = {}
    for i, s in enumerate(arrays["sample_idx"]):
        s = int(s)
        if s not in out:
            out[s] = int(arrays["split"][i])
    return out


def _checkpoint(run: str, store: RunStore) -> tuple[Path, str]:
    """(ruta, origen) de los pesos con los que se va a inferir.

    ⚠ Aqui NO manda el catalogo, y es deliberado. `checkpoint_de` exige que la
    red este APROBADA porque su pregunta es «¿que puede SERVIR la web app?», y
    servir una red que nadie eligio es justo lo que esa lista impide. La pregunta
    de aqui es otra: «¿donde falla ESTA red?», sobre un run que alguien acaba de
    nombrar a mano. Exigir aprobacion dejaria fuera los 44 runs con pesos que no
    estan en la lista (medido el 2026-09-02: 47 con `best.pt`, 3 aprobados), o
    sea casi todo lo que hay que analizar.

    Lo que no puede pasar es que se confunda una cosa con la otra: el origen y el
    `aprobada` viajan en el payload y en el manifest del dataset resultante.
    """
    ck, origen = checkpoint_de(run, store)
    if ck is None:
        p = Path(store.path(run)) / CHECKPOINT_INFERENCIA
        if p.exists():
            return p, "run"
        # Un run sin pesos NO es un run malo: es uno cuyos pesos no se guardaron,
        # que es el defecto del proyecto desde el 2026-08-31. Se dice cual es la
        # diferencia, porque los arreglos son opuestos (aprobar / reentrenar).
        raise FallidosError(
            "sin_pesos",
            f"'{run}' no tiene {CHECKPOINT_INFERENCIA} en ninguna parte",
            "esta en el catalogo pero sus pesos no estan en disco: reentrenalo"
            if esta_aprobada(run) else
            "ni en su directorio, ni en la antesala: los pesos de un run NO se "
            "guardan por defecto (docs/inferencia.md)")
    return ck, origen


def evaluar(run: str, dataset: str, criterio: Criterio, *,
            verdad: str = "fuente", split: str = "todo",
            store: RunStore | None = None,
            wstore: WindowDatasetStore | None = None,
            progreso=None) -> dict:
    """Pasa `run` por las imagenes de `dataset` y puntua cada una.

    No escribe nada. Devuelve el payload entero, incluida la lista por imagen en
    el orden en que estan en el dataset (el ORDEN DE PEOR va aparte, en
    `ordenar`): mezclar medir y ordenar hace que no se pueda mirar lo medido sin
    aceptar el criterio.
    """
    store = store or RunStore()
    wstore = wstore or WindowDatasetStore()
    manifest = wstore.manifest(dataset)
    arrays = wstore.arrays(dataset)
    ck, origen_ck = _checkpoint(run, store)

    if split not in ("todo", *SPLIT_NAMES):
        raise FallidosError("split_desconocido", f"split '{split}' no existe",
                            f"usa todo, {', '.join(SPLIT_NAMES)}")
    indices = [int(i) for i in arrays["images_sample_idx"]]
    if split != "todo":
        del_split = set(wstore.split_map(dataset).get(split) or [])
        if not del_split:
            raise FallidosError(
                "split_vacio", f"el split '{split}' de '{dataset}' no tiene imagenes",
                "usa --split todo, o reconstruye el dataset con ese split > 0")
        indices = [i for i in indices if i in del_split]

    window_size = int(manifest["config"]["window_size"])
    kinds = set(manifest["config"]["target_kinds"])
    source_id = manifest["source_id"]

    if verdad == "fuente":
        try:
            verdades = verdad_de_fuente(source_id, indices, kinds)
        except SourceError as e:
            # el contrato ⑬ literal: se falla con razon, no se cae al otro camino
            raise FallidosError(
                "verdad_necesita_fuente",
                f"la verdad de parrafos sale de la fuente '{source_id}', y esa "
                f"fuente no esta ({getattr(e, 'message', e)})",
                "recuperala, o pide explicitamente el camino degradado con "
                "--verdad ventanas (recompone las cajas desde el propio npz; "
                "pierde los parrafos cortados por el borde)") from e
    elif verdad == "ventanas":
        verdades = verdad_de_ventanas(arrays, window_size)
    else:
        raise FallidosError("verdad_desconocida", f"verdad '{verdad}' no existe",
                            "usa fuente (por defecto) o ventanas")

    fila_de = {int(a): i for i, a in enumerate(arrays["images_sample_idx"])}
    split_de = _split_por_imagen(arrays)

    model = load_model(ck)
    # Cuando dos esquinas son "la misma": el mismo radio con el que la propia
    # inferencia decide que dos detecciones son una (el NMS). Definirlo aparte
    # seria una segunda escala para el mismo hecho.
    tol_esquina = float(criterio.nms_radius if criterio.nms_radius is not None
                        else model.dims.center_out / 2)
    imagenes = arrays["images"]
    por_imagen, knobs = [], None
    for k, idx in enumerate(indices):
        img = imagenes[fila_de[idx]]
        out = predict_image(model, img, threshold=criterio.threshold,
                            stride=criterio.stride, nms_radius=criterio.nms_radius,
                            min_size=criterio.min_size)
        knobs = out["knobs"]
        pred = [(p["x0"], p["y0"], p["x1"], p["y1"]) for p in out["paragraphs"]]
        v = verdades.get(idx, {"cajas": [], "completa": False, "esquinas": {}})
        r = paragraph_f1(pred, v["cajas"], criterio.iou_threshold)
        esq = esquinas_acertadas(out["corners"], v["esquinas"] or {}, tol_esquina)
        por_imagen.append({
            "index": idx,
            "split": SPLIT_NAMES[split_de[idx]],
            "errores": r["fp"] + r["fn"],
            "f1": r["f1"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "mean_iou": r["mean_iou"],
            "gt_completa": bool(v["completa"]),
            "n_verdad": len(v["cajas"]), "n_prediccion": len(pred),
            # el diagnostico que separa "no vio la esquina" de "la junto mal"
            "esquinas": esq,
            "solo_emparejado": bool(esq["fp"] == 0 and esq["fn"] == 0
                                    and r["fp"] + r["fn"] > 0),
            "prediccion": [[round(c, 2) for c in b] for b in pred],
            "verdad": [[round(float(c), 2) for c in b] for b in v["cajas"]],
        })
        if progreso:
            progreso(k + 1, len(indices))

    return {
        "run": run, "dataset": dataset, "split": split,
        "source": source_id, "window_size": window_size,
        "checkpoint": CHECKPOINT_INFERENCIA, "checkpoint_origen": origen_ck,
        "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
        "aprobada": esta_aprobada(run),
        "fingerprint_base": manifest["fingerprint"],
        "verdad": {
            "origen": "fuente" if verdad == "fuente" else "ventanas-reconstruidas",
            "imagenes": len(por_imagen),
            "gt_incompleta": sum(1 for r in por_imagen if not r["gt_completa"]),
            "aviso": None if verdad == "fuente" else
                     "las cajas verdaderas se recompusieron desde las etiquetas "
                     "de ventana del propio npz porque la fuente no esta "
                     "(contrato 13 degradado a proposito): un parrafo cortado "
                     "por el borde de la imagen NO se recupera",
        },
        "knobs": {**(knobs or {}), "iou_threshold": criterio.iou_threshold},
        "criterio": asdict(criterio),
        # el reparto de la culpa, agregado: cuantas de las imagenes con error
        # tienen TODAS las esquinas bien (o sea, la culpa es del emparejado de F,
        # no de la red)
        "diagnostico": {
            "tol_esquina_px": tol_esquina,
            "imagenes_con_error": sum(1 for r in por_imagen if r["errores"]),
            "solo_emparejado": sum(1 for r in por_imagen if r["solo_emparejado"]),
            "esquinas": {k: sum(r["esquinas"][k] for r in por_imagen)
                         for k in ("tp", "fp", "fn")},
        },
        "per_image": por_imagen,
    }


def ordenar(por_imagen: list[dict]) -> list[dict]:
    """De peor a mejor, con el criterio del docstring del modulo.

    `mean_iou` None -> -1.0: no haber emparejado NADA es peor que haberlo hecho
    mal, y sin esto ordenaria como si fuera cero, que aqui significa otra cosa.
    """
    return sorted(por_imagen,
                  key=lambda r: (-r["errores"], r["f1"],
                                 -1.0 if r["mean_iou"] is None else r["mean_iou"],
                                 r["index"]))


def seleccionar(por_imagen: list[dict], criterio: Criterio) -> list[dict]:
    """Las que entran en el dataset, ya ordenadas de peor a mejor."""
    elegibles = [r for r in ordenar(por_imagen)
                 if r["errores"] >= criterio.min_errores
                 and (criterio.incluir_gt_parcial or r["gt_completa"])]
    if criterio.max_imagenes > 0:
        elegibles = elegibles[:criterio.max_imagenes]
    return elegibles


# ------------------------------------------------------------- la escritura

def _subconjunto(arrays: dict, indices: list[int]) -> dict:
    """Los seis arrays de B, recortados a esas imagenes de A.

    `sample_idx` conserva los indices ORIGINALES de A -- no se renumera. Es lo
    que permite cruzar una fila de este dataset con la del dataset base y con la
    fuente; y es correcto por contrato, porque `sample_idx` nunca indexo
    `images` (para eso esta `images_sample_idx`, formatos.md §4.1).
    """
    quiero = np.asarray(sorted(indices), dtype=np.int32)
    mw = np.isin(arrays["sample_idx"], quiero)
    mi = np.isin(arrays["images_sample_idx"], quiero)
    return {"y": arrays["y"][mw], "sample_idx": arrays["sample_idx"][mw],
            "window_xy": arrays["window_xy"][mw], "split": arrays["split"][mw],
            "images": arrays["images"][mi],
            "images_sample_idx": arrays["images_sample_idx"][mi]}


def _comprobar_politica(politica: str) -> None:
    """En la PUERTA, no al escribir: inferir 1000 imagenes cuesta ~40 s por red,
    y enterarse despues de eso de que un argumento estaba mal escrito convierte
    un error de tecleo en una espera. La misma razon por la que `evaluar`
    comprueba `split` y `verdad` antes de cargar el modelo."""
    if politica not in POLITICAS_SPLIT:
        raise FallidosError("split_salida_desconocida",
                            f"politica de split '{politica}' no existe",
                            f"usa {', '.join(POLITICAS_SPLIT)} "
                            f"(conservar es el defecto)")


def _resplit(sub: dict, politica: str, val_frac: float, test_frac: float,
             seed: int) -> dict:
    """El split del dataset nuevo. Tres politicas, y la de por defecto CONSERVA.

    - `conservar`: cada imagen se queda con el split que tenia en el dataset
      base. Es el defecto porque es el unico que no miente sobre lo que la red
      evaluada vio: una imagen que estaba en train sigue estando en train.
    - `rehacer`: reparto nuevo por imagen (el mismo `_assign_splits` del
      extractor, misma semantica de semilla). Para cuando el dataset se va a usar
      para entrenar una red NUEVA, a la que el split viejo no le dice nada.
    - `train`: todo a train. Para usarlo entero como material de entrenamiento y
      medir en otro sitio.
    """
    if politica == "conservar":
        return sub
    from fv.windows.extract import _assign_splits

    ids = sub["images_sample_idx"]
    if politica == "rehacer":
        por_imagen = _assign_splits(len(ids), val_frac, test_frac, seed)
    else:                                    # 'train': ya comprobado en la puerta
        por_imagen = np.zeros(len(ids), dtype=np.int8)
    de = {int(a): int(por_imagen[i]) for i, a in enumerate(ids)}
    sub = dict(sub)
    sub["split"] = np.asarray([de[int(s)] for s in sub["sample_idx"]], dtype=np.int8)
    return sub


def escribir(destino: Path, base_manifest: dict, arrays: dict,
             elegidas: list[dict], evaluacion: dict, *,
             split_salida: str = "conservar", val_frac: float = 0.2,
             test_frac: float = 0.2, seed: int = 1, png: bool = False) -> dict:
    """Deja el dataset en disco. NO commitea (como `fv-train` y `promover`)."""
    _comprobar_politica(split_salida)
    destino = Path(destino)
    if destino.exists():
        # misma regla que `extract_windows`: no se sobrescribe NUNCA. Un dataset
        # que cambia bajo el mismo nombre invalida en silencio todo lo medido
        # contra el (por eso existe el `fingerprint`).
        raise FallidosError(
            "dataset_ya_existe", f"ya existe un dataset en {destino}",
            "elige otro nombre con --nombre, o borralo a mano: no se sobrescribe")
    indices = [r["index"] for r in elegidas]
    if not indices:
        raise FallidosError(
            "sin_fallos",
            "ninguna imagen cumple el criterio: no hay dataset que escribir",
            "baja --min-errores, quita el filtro de --split, o alegrate")

    sub = _resplit(_subconjunto(arrays, indices), split_salida,
                   val_frac, test_frac, seed)
    # Se construye AL LADO y se renombra al final. Escribir directamente en el
    # destino deja, si el proceso muere a mitad --y este comando se lanza desde
    # Telegram, donde el bot se reinicia--, un directorio con manifest y sin npz
    # que ademas BLOQUEA el reintento con `dataset_ya_existe`. Es el mismo patron
    # que `guardar_en_antesala`, y por lo mismo el temporal va en el mismo
    # sistema de ficheros: `os.replace` solo es atomico dentro de uno.
    parcial = destino.with_name(destino.name + ".parcial")
    if parcial.exists():
        for f in sorted(parcial.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        parcial.rmdir()
    destino, final = parcial, destino
    destino.mkdir(parents=True)
    npz = destino / "windows.npz"
    np.savez_compressed(npz, **sub)

    S, H, W = sub["images"].shape
    manifest = dict(base_manifest)
    manifest.pop("name", None)                 # lo pone el store al leer, no es del fichero
    cfg = dict(base_manifest["config"])
    if split_salida == "rehacer":
        cfg.update({"val_frac": val_frac, "test_frac": test_frac, "seed": seed})
    elif split_salida == "train":
        cfg.update({"val_frac": 0.0, "test_frac": 0.0})
    manifest.update({
        "fingerprint": "sha256:" + hashlib.sha256(npz.read_bytes()).hexdigest(),
        "config": cfg,
        "images": {"shape": [S, H, W], "bytes": int(S * H * W),
                   "budget_bytes": base_manifest.get("images", {}).get("budget_bytes")},
        "num_samples": S,
        "num_windows": int(sub["y"].shape[0]),
        "windows_per_split": {n: int((sub["split"] == i).sum())
                              for i, n in enumerate(SPLIT_NAMES)},
        "positives_per_corner": {c: int((sub["y"][:, i, 0] >= 0.5).sum())
                                 for i, c in enumerate(CORNER_NAMES)},
        # todo lo que hace falta para saber que es esto y volver a producirlo
        "fallidos": {
            "que": f"las {S} imagenes en que '{evaluacion['run']}' falla peor, "
                   f"de las {evaluacion['verdad']['imagenes']} de "
                   f"'{evaluacion['dataset']}'",
            "creado": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "producido_por": "scripts/dataset_fallidos.py",
            "base": {"dataset": evaluacion["dataset"],
                     "fingerprint": evaluacion["fingerprint_base"],
                     "split_evaluado": evaluacion["split"]},
            "red": {"run": evaluacion["run"],
                    "checkpoint": evaluacion["checkpoint"],
                    "origen": evaluacion["checkpoint_origen"],
                    "sha256": evaluacion["checkpoint_sha256"],
                    "aprobada": evaluacion["aprobada"]},
            "verdad": evaluacion["verdad"],
            "knobs": evaluacion["knobs"],
            "criterio": evaluacion["criterio"],
            "diagnostico": evaluacion["diagnostico"],
            "split_salida": split_salida,
            "errores": {"total": sum(r["errores"] for r in elegidas),
                        "fp": sum(r["fp"] for r in elegidas),
                        "fn": sum(r["fn"] for r in elegidas),
                        "maximo_en_una_imagen": max(r["errores"] for r in elegidas)},
        },
    })
    write_json_atomic(destino / "manifest.json", manifest)
    de_split = _split_por_imagen(sub)
    write_json_atomic(destino / "split.json",
                      {n: sorted(a for a, s in de_split.items() if s == i)
                       for i, n in enumerate(SPLIT_NAMES)})
    # El diagnostico por imagen: que se predijo, que era verdad y cuantos errores
    # salieron. NO es parte del contrato de B (nada lo lee para entrenar); esta
    # para que "esta imagen esta aqui por esto" no haya que reconstruirlo.
    write_json_atomic(destino / "fallos.json", {
        **{k: v for k, v in evaluacion.items() if k != "per_image"},
        "elegidas": elegidas,
        "descartadas": len(evaluacion["per_image"]) - len(elegidas),
    })

    escritos = []
    if png:
        from PIL import Image
        d = destino / "imagenes"
        d.mkdir()
        for fila, a in enumerate(sub["images_sample_idx"]):
            p = d / f"{int(a):06d}.png"
            Image.fromarray(sub["images"][fila], mode="L").save(p)
            escritos.append(p.name)

    destino.replace(final)                  # el dataset aparece entero o no aparece
    destino = final
    return {"destino": str(destino), "imagenes": S,
            "ventanas": int(sub["y"].shape[0]),
            "windows_per_split": manifest["windows_per_split"],
            "png": len(escritos),
            "fingerprint": manifest["fingerprint"],
            "commit": f"cd {settings.data_root()} && git add -A && "
                      f"git commit -m 'dataset de fallos de {evaluacion['run']}' "
                      f"&& git push"}


def crear(run: str, *, dataset: str | None = None, nombre: str | None = None,
          criterio: Criterio | None = None, verdad: str = "fuente",
          split: str = "todo", split_salida: str = "conservar",
          val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 1,
          png: bool = False, seco: bool = False,
          store: RunStore | None = None, wstore: WindowDatasetStore | None = None,
          progreso=None) -> dict:
    """Evalua y escribe, de una. `seco=True` mide y no escribe nada."""
    store = store or RunStore()
    wstore = wstore or WindowDatasetStore()
    criterio = criterio or Criterio()
    if not seco:
        _comprobar_politica(split_salida)     # antes de los ~40 s de inferencia
    dataset = dataset or dataset_de(run, store)
    ev = evaluar(run, dataset, criterio, verdad=verdad, split=split,
                 store=store, wstore=wstore, progreso=progreso)
    elegidas = seleccionar(ev["per_image"], criterio)
    nombre = nombre or nombre_dataset(run)
    resultado = {"nombre": nombre, "evaluacion": ev, "elegidas": elegidas}
    if seco:
        resultado["escrito"] = None
        return resultado
    resultado["escrito"] = escribir(
        wstore.path(nombre), wstore.manifest(dataset), wstore.arrays(dataset),
        elegidas, ev, split_salida=split_salida, val_frac=val_frac,
        test_frac=test_frac, seed=seed, png=png)
    return resultado


def dataset_de(run: str, store: RunStore | None = None) -> str:
    """El dataset con el que se entreno `run`, que es el defecto sensato.

    Sin procedencia no se inventa uno: un dataset elegido por el script seria una
    comparacion entre cosas que nadie decidio comparar.
    """
    store = store or RunStore()
    prov = (store.config(run).get("provenance") or {}).get("window_dataset") or {}
    if not prov.get("name"):
        raise FallidosError(
            "run_sin_procedencia",
            f"'{run}' no dice de que dataset salio",
            "pasale el dataset a mano con --dataset")
    return prov["name"]
