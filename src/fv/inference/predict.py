"""F — apply a run to a full image: sliding fovea window -> corner detections
-> NMS -> greedy TL->BR reconstruction. Returns ALL stages (without the raw
one, "the paragraph came out wrong" is not diagnosable).

The foveated view comes from THE SAME fv.fovea.build_view the dataloader uses
(contract (5)). Knobs (threshold, stride, nms_radius, min_size) are F, not D:
post-hoc, in units of the labelled window, echoed back in the payload.
"""

from __future__ import annotations

import numpy as np
import torch

from fv.fovea import build_view, edge_features, input_stack
from fv.metrics import CORNER_NAMES

# Como se arman los parrafos a partir de las esquinas. Es un knob de F --post-hoc,
# no toca la red-- y por eso viaja en `knobs` y entra en la clave de cache de la
# metrica de tarea, como los otros cuatro.
#
#   "tlbr"  el heredado: empareja TL con BR por confianza. TIRA TR y BL.
#   "quad"  usa las cuatro esquinas para decidir que dos son del mismo parrafo.
#
# ⚠ El defecto sigue siendo el heredado A PROPOSITO: cambiarlo movería todos los
# numeros de metrica de tarea ya publicados, y eso es una decision del dueno, no
# de este fichero. Lo medido con cada uno, en docs/dataset-fallidos.md.
RECONSTRUCTS = ("tlbr", "quad")
RECONSTRUCT_DEFAULT = "tlbr"


def _positions(length: int, n: int, stride: int) -> list[int]:
    if length < n:
        return []
    xs = list(range(0, length - n + 1, stride))
    if xs[-1] != length - n:
        xs.append(length - n)
    return xs


def predict_image(model, image: np.ndarray, threshold: float = 0.5,
                  stride: int | None = None, nms_radius: float | None = None,
                  min_size: float | None = None,
                  reconstruct: str = RECONSTRUCT_DEFAULT,
                  corner_tol: float | None = None) -> dict:
    dims = model.dims
    n = dims.center_out                      # the labelled window = the fovea
    stride = stride if stride else max(1, n // 2)
    nms_radius = nms_radius if nms_radius is not None else n / 2
    min_size = min_size if min_size is not None else 4.0
    if reconstruct not in RECONSTRUCTS:
        raise ValueError(f"reconstruct '{reconstruct}' no existe: "
                         f"usa {list(RECONSTRUCTS)}")
    # Cuando una esquina detectada "esta donde deberia": el MISMO radio con el
    # que el NMS decide que dos detecciones son la misma. Una segunda escala para
    # el mismo hecho serian dos numeros que se pueden desincronizar.
    corner_tol = corner_tol if corner_tol is not None else nms_radius
    H, W = image.shape

    xs = _positions(W, n, stride)
    ys = _positions(H, n, stride)
    views, edges, origins = [], [], []
    for wy0 in ys:
        for wx0 in xs:
            v, cov = build_view(image, wx0, wy0, dims,
                                pool_mode=model.cfg["pool_mode"],
                                pad_mode=model.cfg["pad_mode"])
            views.append(input_stack(v, cov, model.cfg.get("mask_channel", "off")))
            # the SAME fv.fovea function the dataloader calls (contract (5)).
            # Here it matters more than for the view: these windows come from a
            # sliding grid over a WHOLE image, so the edge ones are a fixed
            # fraction of every prediction, and getting the signal wrong at
            # inference would show up as a border artefact, not as an error.
            edges.append(edge_features(image.shape, wx0, wy0, dims,
                                       model.cfg["edge_inputs"]))
            origins.append((wx0, wy0))
    raw = []
    if views:
        # ya vienen con su eje de canal desde `input_stack`
        batch = torch.from_numpy(np.stack(views))
        edge = torch.from_numpy(np.stack(edges))
        with torch.no_grad():
            out = model(batch, edge).numpy()
        scores = 1.0 / (1.0 + np.exp(-out[:, :, 0]))
        for i, (wx0, wy0) in enumerate(origins):
            for ci in range(4):
                s = float(scores[i, ci])
                if s >= threshold:
                    cx = wx0 + float(out[i, ci, 1]) * n
                    cy = wy0 + float(out[i, ci, 2]) * n
                    raw.append({"corner": CORNER_NAMES[ci], "score": s,
                                "x": round(cx, 2), "y": round(cy, 2),
                                "window": [wx0, wy0]})

    corners = _nms(raw, nms_radius)
    paragraphs = (_reconstruct(corners, min_size) if reconstruct == "tlbr"
                  else _reconstruct_quad(corners, min_size, corner_tol))
    return {"raw": raw, "corners": corners, "paragraphs": paragraphs,
            # the corner vocabulary travels with the answer (fv.metrics is its
            # ONE definition): a reader that keeps its own copy drifts silently
            "corner_order": list(CORNER_NAMES),
            "image_size": [W, H],
            "knobs": {"threshold": threshold, "stride": stride,
                      "nms_radius": nms_radius, "min_size": min_size,
                      "window_size": n, "reconstruct": reconstruct,
                      "corner_tol": corner_tol}}


def _nms(dets: list[dict], radius: float) -> list[dict]:
    out = []
    for cname in CORNER_NAMES:
        group = sorted((d for d in dets if d["corner"] == cname),
                       key=lambda d: -d["score"])
        kept: list[dict] = []
        for d in group:
            if all((d["x"] - k["x"]) ** 2 + (d["y"] - k["y"]) ** 2 > radius ** 2
                   for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


def _reconstruct(corners: list[dict], min_size: float) -> list[dict]:
    """Greedy TL->BR pairing (inherited heuristic — the place to touch if
    paragraphs come out wrong while corners come out right).

    ⚠ MEDIDO EL 2026-09-02: ES AQUI DONDE SE PIERDE LA MAYORIA DE LOS PARRAFOS.
    Sobre las 1000 imagenes de `dirty1000-80px-16px-r20260827`, las tres redes
    aprobadas detectan esquinas al 95-99 % y aun asi el 35-49 % de las imagenes
    salen con algun error de parrafo; en el 43-75 % de ESAS, las cuatro esquinas
    estan bien y lo unico que falla es esto. Mirado a ojo: las cajas unen el TL
    de un parrafo con el BR de OTRO.

    Y la causa cabe en una frase: **la red predice CUATRO tipos de esquina y esta
    funcion usa DOS.** `TR` y `BL` se calculan, pasan el NMS, viajan en `corners`
    ...y se tiran. La unica prueba de que un TL y un BR son del mismo parrafo es
    que el BR este abajo a la derecha y que los dos tengan score alto -- o sea
    CONFIANZA, que no dice nada sobre pertenecer al mismo bloque.

    Se conserva como `reconstruct="tlbr"` --el defecto-- porque cambiarlo movería
    TODA la metrica de tarea publicada. La alternativa es `"quad"`, abajo.
    """
    tls = sorted((c for c in corners if c["corner"] == "TL"), key=lambda c: -c["score"])
    brs = [c for c in corners if c["corner"] == "BR"]
    used: set[int] = set()
    boxes = []
    for tl in tls:
        best, best_j = None, -1
        for j, br in enumerate(brs):
            if j in used:
                continue
            if br["x"] - tl["x"] >= min_size and br["y"] - tl["y"] >= min_size:
                score = tl["score"] * br["score"]
                if best is None or score > best:
                    best, best_j = score, j
        if best_j >= 0:
            used.add(best_j)
            br = brs[best_j]
            boxes.append({"x0": tl["x"], "y0": tl["y"], "x1": br["x"], "y1": br["y"],
                          "score": round(best, 4), "corners": 2})
    return boxes


def _nearest(dets: list[dict], x: float, y: float, tol: float) -> int:
    """Indice de la deteccion mas cercana a (x, y) dentro de `tol`, o -1."""
    best, best_i, tol2 = None, -1, tol * tol
    for i, d in enumerate(dets):
        d2 = (d["x"] - x) ** 2 + (d["y"] - y) ** 2
        if d2 <= tol2 and (best is None or d2 < best):
            best, best_i = d2, i
    return best_i


def _reconstruct_quad(corners: list[dict], min_size: float,
                      corner_tol: float) -> list[dict]:
    """Empareja usando las CUATRO esquinas: un TL y un BR son del mismo parrafo
    si la red vio tambien el TR en (x1, y0) y el BL en (x0, y1).

    Es la misma prueba geometrica con la que `fv.fallidos` recompone la VERDAD
    desde las etiquetas de ventana --alli sale exacta en 989 de 1000 imagenes--,
    aplicada a las detecciones y por tanto con tolerancia.

    Tres decisiones, y las tres importan:

    1. **Ordena por APOYO antes que por score.** Una caja respaldada por sus
       cuatro esquinas se queda con ellas antes de que una de dos se las lleve.
       Es exactamente el fallo que arregla: hoy gana la pareja mas CONFIADA, y la
       confianza no dice nada sobre pertenecer al mismo bloque.
    2. **Cada esquina se consume una vez, las cuatro.** Un TR ya usado por un
       parrafo no puede respaldar otro: un rectangulo tiene sus esquinas y no las
       comparte.
    3. **Degrada, no exige.** Una caja con solo TL+BR (`corners: 2`) sigue
       valiendo, la ultima. Sin eso, un parrafo al que la red no vio una esquina
       dejaria de detectarse -- se cambiaria un fallo de precision por uno de
       recall, que no es arreglar.

    ⚠ La caja se sigue construyendo con TL y BR, aunque haya cuatro esquinas.
    Promediar las dos estimaciones de cada lado (`x0` de TL y de BL, etc.) es
    plausible que suba el IoU, pero es OTRO cambio: mezclarlo aqui haria que la
    mejora medida no se pudiera atribuir a ninguno de los dos.
    """
    por = {c: [d for d in corners if d["corner"] == c] for c in CORNER_NAMES}
    TL, TR, BR, BL = (por[c] for c in CORNER_NAMES)
    cand = []
    for i, a in enumerate(TL):
        for k, d in enumerate(BR):
            if d["x"] - a["x"] < min_size or d["y"] - a["y"] < min_size:
                continue
            usa = {"TL": i, "BR": k}
            scores = [a["score"], d["score"]]
            j = _nearest(TR, d["x"], a["y"], corner_tol)
            if j >= 0:
                usa["TR"] = j
                scores.append(TR[j]["score"])
            m = _nearest(BL, a["x"], d["y"], corner_tol)
            if m >= 0:
                usa["BL"] = m
                scores.append(BL[m]["score"])
            cand.append({"apoyo": len(scores), "medio": sum(scores) / len(scores),
                         "area": (d["x"] - a["x"]) * (d["y"] - a["y"]),
                         "score": a["score"] * d["score"], "usa": usa,
                         "caja": (a["x"], a["y"], d["x"], d["y"])})
    # Apoyo primero; a igual apoyo, LA MAS PEQUENA; y solo entonces el score.
    #
    # ⚠ El area no es un detalle de desempate: es la evidencia que resuelve el
    # caso que el apoyo NO resuelve. En una pagina con parrafos en REJILLA --o
    # simplemente alineados-- el TL de uno y el BR del de al lado forman un
    # rectangulo cuyo TR y BL TAMBIEN existen, porque son de los dos parrafos
    # verdaderos. Esa caja falsa tiene apoyo 4 y scores de 1,00, igual que las
    # buenas: contar esquinas no la distingue.
    #
    # Lo que si la distingue es que una caja que se traga dos parrafos es
    # ESTRICTAMENTE MAS GRANDE que cualquiera de los dos, y que quedarsela deja
    # huerfanas las esquinas de ambos. Tomar primero la pequena deja sitio a que
    # las demas se formen; al reves, no. Los parrafos no se anidan, asi que el
    # riesgo simetrico --una caja pequena espuria DENTRO de un parrafo grande de
    # verdad-- necesitaria esquinas de otro bloque solapando con este.
    #
    # Medido el 2026-09-02 sobre las 987 imagenes con verdad completa de
    # dirty1000-80px-16px-r20260827, macro f1 medio de las tres redes aprobadas:
    #     apoyo, score                 -> no se probo (el apoyo solo ya empata)
    #     apoyo, residuo, score        -> 0,9525
    #     apoyo, AREA, score           -> 0,9578   <- gana en las TRES redes
    # (el residuo --cuanto se desvia cada esquina de donde el rectangulo dice--
    # se probo y pierde: separa 1,5 de 1,6 px donde el area separa 490 de 936.)
    #
    # El tercer criterio es el score y el cuarto el orden de generacion, que es
    # determinista: sin el, dos cajas empatadas podrian salir en distinto orden
    # entre ejecuciones.
    cand.sort(key=lambda c: (-c["apoyo"], c["area"], -c["medio"]))
    tomadas: dict[str, set] = {c: set() for c in CORNER_NAMES}
    boxes = []
    for c in cand:
        if any(idx in tomadas[k] for k, idx in c["usa"].items()):
            continue
        for k, idx in c["usa"].items():
            tomadas[k].add(idx)
        x0, y0, x1, y1 = c["caja"]
        boxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                      "score": round(c["score"], 4), "corners": c["apoyo"]})
    return boxes
