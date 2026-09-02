"""Como se arman los parrafos con las esquinas (knob de F: `reconstruct`).

Por que estos tests y no otros: el heredado `tlbr` es donde se pierde la mayoria
de los parrafos --las redes detectan esquinas al 95-99 % y aun asi el 35-49 % de
las imagenes salen con error-- y la causa cabe en una frase: la red predice
CUATRO tipos de esquina y esa funcion usa DOS. Lo que se comprueba aqui es que
`quad` usa las cuatro, que las usa BIEN (el caso de la rejilla, que es el unico
donde tener las cuatro no basta), y que el defecto NO ha cambiado.
"""

from __future__ import annotations

import pytest


def _c(corner, x, y, score=1.0):
    return {"corner": corner, "x": float(x), "y": float(y), "score": score,
            "window": [0, 0]}


def _caja(x0, y0, x1, y1, score=1.0):
    """Las cuatro esquinas de un rectangulo, en el orden de CORNER_NAMES."""
    return [_c("TL", x0, y0, score), _c("TR", x1, y0, score),
            _c("BR", x1, y1, score), _c("BL", x0, y1, score)]


def _cajas(boxes):
    return sorted((round(b["x0"], 2), round(b["y0"], 2),
                   round(b["x1"], 2), round(b["y1"], 2)) for b in boxes)


# ------------------------------------------------- el fallo que esto arregla

# Dos parrafos apilados, con la confianza repartida como se reparte de verdad:
# no por parrafo, sino POR ESQUINA (cada una depende de como se vea ella, no de
# con quien va). Aqui el TL de arriba es el mas confiado y su PROPIO BR el menos,
# asi que el heredado --que ordena los TL por confianza y luego se queda con el
# BR mas confiado que le encaje-- prefiere el BR del parrafo de ABAJO.
APILADOS = [
    _c("TL", 10, 10, 1.00), _c("TR", 40, 10, 1.00),      # parrafo de arriba
    _c("BR", 40, 20, 0.90), _c("BL", 10, 20, 1.00),
    _c("TL", 10, 30, 0.95), _c("TR", 40, 30, 1.00),      # parrafo de abajo
    _c("BR", 40, 45, 1.00), _c("BL", 10, 45, 1.00),
]


def test_tlbr_une_el_TL_de_un_parrafo_con_el_BR_DE_OTRO():
    """El fallo, reproducido: una sola caja que engloba los dos parrafos.

    Es lo que se vio a ojo el 2026-09-02 sobre dirty1000-80px. Y notese QUE hace
    falta para reproducirlo: que el BR equivocado sea el mas CONFIADO. Con los
    scores empatados el heredado acierta de casualidad, por el orden en que
    genera los candidatos -- que es otra forma de decir que no esta decidiendo
    nada.
    """
    from fv.inference.predict import _reconstruct

    salida = _cajas(_reconstruct(APILADOS, 4.0))
    assert (10.0, 10.0, 40.0, 45.0) in salida, (
        "si esto falla, el heredado ya no tiene el fallo y este test sobra")
    assert salida != [(10.0, 10.0, 40.0, 20.0), (10.0, 30.0, 40.0, 45.0)]


def test_quad_lo_arregla_usando_las_cuatro_esquinas():
    """Las MISMAS esquinas y los mismos scores: lo unico que cambia es que ahora
    se usan las cuatro."""
    from fv.inference.predict import _reconstruct_quad

    assert _cajas(_reconstruct_quad(APILADOS, 4.0, 8.0)) == [
        (10.0, 10.0, 40.0, 20.0), (10.0, 30.0, 40.0, 45.0)]


def test_el_caso_de_la_REJILLA_es_el_que_pide_el_residuo():
    """Cuatro parrafos en 2x2: el TL de uno y el BR del de al lado forman un
    rectangulo cuyo TR y BL TAMBIEN existen (son de los dos parrafos buenos).

    O sea que la caja falsa tiene apoyo 4 y scores de 1,00 igual que las buenas:
    contar esquinas no la distingue. Lo unico que la separa es que sus esquinas
    encajan PEOR -- de ahi que el residuo vaya antes que el score al ordenar.
    Medido en la img 151 de dirty1000-80px-16px-r20260827.
    """
    from fv.inference.predict import _reconstruct_quad

    # las esquinas de la derecha, 2 px descuadradas respecto de las de la
    # izquierda: es lo que pasa en una pagina real, y es toda la evidencia que hay
    corners = (_caja(2, 10, 37, 24) + _caja(42, 12, 74, 22)
               + _caja(4, 34, 36, 56) + _caja(46, 33, 73, 50))
    salida = _cajas(_reconstruct_quad(corners, 4.0, 8.0))
    assert salida == [(2.0, 10.0, 37.0, 24.0), (4.0, 34.0, 36.0, 56.0),
                      (42.0, 12.0, 74.0, 22.0), (46.0, 33.0, 73.0, 50.0)]
    # y la falsa que uniria los dos de arriba NO sale
    assert (2.0, 10.0, 74.0, 22.0) not in salida


# ------------------------------------------------------------ como degrada

def test_una_esquina_que_falta_no_hace_desaparecer_el_parrafo():
    """Degrada, no exige: con solo TL+BR la caja sigue valiendo, la ultima.

    Sin esto se cambiaria un fallo de precision por uno de recall, que no es
    arreglar: un parrafo al que la red no vio el TR dejaria de detectarse.
    """
    from fv.inference.predict import _reconstruct_quad

    corners = [c for c in _caja(10, 10, 40, 20) if c["corner"] != "TR"]
    salida = _reconstruct_quad(corners, 4.0, 8.0)
    assert _cajas(salida) == [(10.0, 10.0, 40.0, 20.0)]
    assert salida[0]["corners"] == 3

    solo = [c for c in _caja(10, 10, 40, 20) if c["corner"] in ("TL", "BR")]
    assert _reconstruct_quad(solo, 4.0, 8.0)[0]["corners"] == 2


def test_una_esquina_no_respalda_dos_parrafos():
    """Un rectangulo tiene sus esquinas y no las comparte."""
    from fv.inference.predict import _reconstruct_quad

    corners = _caja(10, 10, 40, 20) + [_c("BR", 60, 20)]
    salida = _reconstruct_quad(corners, 4.0, 8.0)
    usados = [b["corners"] for b in salida]
    assert usados.count(4) == 1                     # solo una puede llevarse el TR
    assert len(salida) <= 2


def test_min_size_sigue_mandando():
    from fv.inference.predict import _reconstruct, _reconstruct_quad

    corners = _caja(10, 10, 12, 12)
    assert _reconstruct_quad(corners, 4.0, 8.0) == []
    assert _reconstruct(corners, 4.0) == []


def test_el_orden_es_determinista():
    """Dos cajas empatadas en todo tienen que salir siempre igual: un criterio
    que cambia al repetirlo no decide nada."""
    from fv.inference.predict import _reconstruct_quad

    corners = _caja(10, 10, 30, 20) + _caja(40, 10, 60, 20)
    primero = _cajas(_reconstruct_quad(corners, 4.0, 8.0))
    for _ in range(5):
        assert _cajas(_reconstruct_quad(list(corners), 4.0, 8.0)) == primero


# --------------------------------------------- el knob, y que el defecto no cambia

def test_el_defecto_sigue_siendo_el_heredado(world):
    """⚠ Cambiar el defecto movería TODA la metrica de tarea publicada, asi que
    es una decision del dueno y no un descuido. Este test la congela."""
    from fv.inference.predict import RECONSTRUCT_DEFAULT, RECONSTRUCTS

    assert RECONSTRUCT_DEFAULT == "tlbr"
    assert set(RECONSTRUCTS) == {"tlbr", "quad"}


def test_predict_image_echa_los_knobs_nuevos_y_rechaza_lo_que_no_conoce(world):
    from fv.inference.checkpoint import load_model
    from fv.inference.predict import predict_image
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    from fv.windows.store import WindowDatasetStore
    from tests.conftest import TINY_NET

    store = RunStore()
    train("r-knobs", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)
    model = load_model(store.path("r-knobs") / "best.pt")
    img = WindowDatasetStore().arrays(world["dataset"])["images"][0]

    out = predict_image(model, img)
    assert out["knobs"]["reconstruct"] == "tlbr"
    # el defecto de la tolerancia es el radio del NMS: una sola escala para
    # "dos detecciones son la misma esquina"
    assert out["knobs"]["corner_tol"] == out["knobs"]["nms_radius"]

    out = predict_image(model, img, reconstruct="quad", corner_tol=3.0)
    assert out["knobs"]["reconstruct"] == "quad"
    assert out["knobs"]["corner_tol"] == 3.0

    with pytest.raises(ValueError, match="reconstruct"):
        predict_image(model, img, reconstruct="a-ojo")


def test_la_cache_de_la_metrica_de_tarea_distingue_las_dos_reconstrucciones(world):
    """Si el knob no entrara en la clave, cambiar de reconstruccion --o cambiar
    su DEFECTO-- serviria numeros calculados con la otra, bajo el mismo nombre.
    Es el fallo silencioso que esta cache tiene prohibido."""
    from fv import settings
    from fv.task import task_score
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    from tests.conftest import TINY_NET

    store = RunStore()
    train("r-cache", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)

    def ficheros():
        d = settings.cache_root() / "task"
        return sorted(p.name for p in d.glob("*.json")) if d.exists() else []

    a = task_score("r-cache", "val", store=store)
    assert a["cached"] is False and len(ficheros()) == 1
    assert task_score("r-cache", "val", store=store)["cached"] is True

    b = task_score("r-cache", "val", reconstruct="quad", store=store)
    assert b["cached"] is False, "la reconstruccion tiene que cambiar la clave"
    assert len(ficheros()) == 2
    assert b["knobs"]["reconstruct"] == "quad"
    assert a["knobs"]["reconstruct"] == "tlbr"


def test_a_igual_apoyo_gana_la_caja_MAS_PEQUENA():
    """Es la evidencia que resuelve lo que el apoyo NO resuelve.

    Dos parrafos alineados: la caja que se los traga los dos tiene sus cuatro
    esquinas detectadas --son las de ellos-- y con scores identicos. Contar
    esquinas no la distingue; su TAMANO si. Y no es simetrico: un span es
    siempre mayor que sus partes, y los parrafos no se anidan.
    """
    from fv.inference.predict import _reconstruct_quad

    # perfectamente alineados en x: el span (10,10)-(40,45) tiene TR en (40,10)
    # y BL en (10,45), que EXISTEN y estan a distancia CERO
    corners = _caja(10, 10, 40, 20) + _caja(10, 30, 40, 45)
    salida = _cajas(_reconstruct_quad(corners, 4.0, 8.0))
    assert salida == [(10.0, 10.0, 40.0, 20.0), (10.0, 30.0, 40.0, 45.0)]
    assert (10.0, 10.0, 40.0, 45.0) not in salida, (
        "el span tiene apoyo 4 y residuo 0: si sale, el area dejo de mandar")
