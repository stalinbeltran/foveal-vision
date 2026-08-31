"""`edge_inputs` — las entradas que le dicen a la cabeza si la IMAGEN se acaba.

Lo que se prueba está repartido por la consecuencia del fallo (R10), y aquí las
tres caras son muy desiguales:

  1. que `off` NO sea la red de siempre  -> catastrófico y silencioso: todos los
     checkpoints y todas las configs en disco cambiarían de significado.
  2. que la señal esté mal orientada (L por R, o el signo al revés) -> la red
     aprende lo contrario y entrena igual de bien. No falla nada nunca.
  3. que la señal llegue a las convoluciones -> es justo lo que este diseño
     descarta; sin comprobarlo, "va a la cabeza" es una intención, no un hecho.

El 2 se prueba con posiciones calculadas a mano sobre una imagen de tamaño
conocido, no comparando la función consigo misma.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fv.fovea import (EDGE_MODES, EDGE_SIDES, FoveaError, dims_of,
                      edge_features, n_edge_features, pad_sides)
from fv.models.builder import (NETWORK_DEFAULTS, build_model, full_config,
                               network_trace)
from fv.validation import check_network

# fóvea 16, borde 4 px en 2 celdas -> N=20, recorte real 24x24
BASE = {"fovea_px": 16, "border_px": 4, "border_reduce": 2,
        "overlap_fovea_px": 2, "overlap_border_px": 0, "n_layers": 2}
DIMS = dims_of(BASE)
IMG = (60, 60)          # (H, W)


def _modelo(modo: str, seed: int = 11):
    torch.manual_seed(seed)
    return build_model(dict(BASE, edge_inputs=modo))


# --------------------------------------------------------------- 1. no mover nada

def test_off_es_el_default_y_no_anade_ninguna_entrada():
    """Un artefacto escrito antes de que este campo existiera tiene que seguir
    significando exactamente la misma red."""
    assert NETWORK_DEFAULTS["edge_inputs"] == "off"
    assert full_config(dict(BASE))["edge_inputs"] == "off"
    assert n_edge_features("off") == 0
    assert edge_features(IMG, 0, 0, DIMS, "off").shape == (0,)


def test_off_construye_la_red_que_ya_estaba_en_disco():
    """Mismas claves, mismas formas y la MISMA salida bit a bit: la cabeza es
    `Linear(flat + 0)`, que es `Linear(flat)`. Sin esto, añadir el campo
    invalidaría todos los checkpoints del proyecto."""
    viejo = _modelo("off")
    # una config que ni menciona el campo, como las guardadas antes de existir
    torch.manual_seed(11)
    sin_campo = build_model({k: v for k, v in BASE.items()})
    assert set(viejo.state_dict()) == set(sin_campo.state_dict())
    for k, t in viejo.state_dict().items():
        assert t.shape == sin_campo.state_dict()[k].shape
    # y carga estricta en los dos sentidos
    sin_campo.load_state_dict(viejo.state_dict(), strict=True)
    x = torch.randn(3, 1, DIMS.N, DIMS.N)
    viejo.eval(), sin_campo.eval()
    with torch.no_grad():
        assert torch.equal(viejo(x), sin_campo(x))


def test_encendido_solo_ensancha_la_cabeza():
    """+4 entradas -> +48 pesos (4 x 12 salidas) y ni uno más. El coste de esto
    tiene que ser despreciable o deja de ser la opción barata que se eligió."""
    off, pad = network_trace(dict(BASE)), network_trace(dict(BASE, edge_inputs="pad"))
    assert pad["edge_features"] == len(EDGE_SIDES) == 4
    assert pad["flat_features"] == off["flat_features"]      # las ramas no cambian
    assert pad["head_inputs"] == off["head_inputs"] + 4
    assert pad["num_params"] == off["num_params"] + 4 * 12


def test_un_checkpoint_sin_borde_no_entra_en_una_red_con_borde():
    """Y al revés. Son redes DISTINTAS, así que el fallo tiene que ser ruidoso:
    cargar por la fuerza dejaría 4 columnas de la cabeza sin entrenar."""
    with pytest.raises(RuntimeError):
        _modelo("pad").load_state_dict(_modelo("off").state_dict(), strict=True)


# --------------------------------------- 2. la señal dice lo que dice que dice

def test_pad_mide_que_fraccion_del_margen_es_relleno():
    """Con borde de 4 px: la ventana en (0,0) tiene el margen izquierdo y el
    superior ENTEROS fuera de la imagen -> 1,0; en (2,0) le entran 2 de 4 -> 0,5."""
    assert EDGE_SIDES == ("L", "T", "R", "B")
    np.testing.assert_allclose(edge_features(IMG, 0, 0, DIMS, "pad"), [1, 1, 0, 0])
    np.testing.assert_allclose(edge_features(IMG, 2, 0, DIMS, "pad"), [0.5, 1, 0, 0])
    np.testing.assert_allclose(edge_features(IMG, 44, 44, DIMS, "pad"), [0, 0, 1, 1])
    # ...y en cuanto la ventana se aleja `border_px` del borde, se apaga
    np.testing.assert_allclose(edge_features(IMG, 22, 22, DIMS, "pad"), [0, 0, 0, 0])


def test_pad_es_exactamente_el_relleno_que_construye_la_vista():
    """La vista y la señal salen de la MISMA aritmética (`pad_sides`). Si se
    calcularan por separado, a la red se le podría hablar de un borde que su
    entrada no tiene, y nada lo delataría."""
    for wx0, wy0 in [(0, 0), (1, 3), (22, 22), (44, 44)]:
        lados = pad_sides(IMG, wx0, wy0, DIMS)
        np.testing.assert_allclose(edge_features(IMG, wx0, wy0, DIMS, "pad"),
                                   np.asarray(lados) / DIMS.border_px)


def test_dist_llega_mas_lejos_que_pad_y_ambos_apuntan_igual():
    """`dist` es la razón de que haya dos modos: con `border_px`=4 y fóvea 16,
    `pad` ya está apagado a 4 px del borde, y el problema que esto ataca
    (¿es el principio del párrafo o su mitad?) alcanza a toda la ventana."""
    lejos = 4                                 # fuera del alcance de `pad`
    assert list(edge_features(IMG, lejos, 0, DIMS, "pad"))[0] == 0.0
    assert edge_features(IMG, lejos, 0, DIMS, "dist")[0] > 0.0
    # 0 = no hay borde por este lado, 1 = el borde está aquí — en LOS DOS modos
    np.testing.assert_allclose(edge_features(IMG, 0, 0, DIMS, "dist"), [1, 1, 0, 0])
    np.testing.assert_allclose(edge_features(IMG, 44, 44, DIMS, "dist"), [0, 0, 1, 1])
    # y satura a una fóvea: el efecto es local, no es "dónde estoy en la página".
    # (20,20) sobre 60x60 deja los cuatro lados a >= 16 px = una fóvea entera.
    np.testing.assert_allclose(edge_features(IMG, 20, 20, DIMS, "dist"), [0, 0, 0, 0])
    np.testing.assert_allclose(edge_features(IMG, 8, 0, DIMS, "dist"), [0.5, 1, 0, 0])


def test_el_caso_del_usuario_distingue_pegado_al_borde_de_cortado_a_la_mitad():
    """Las dos ventanas que la vista NO puede distinguir: una pegada al borde
    superior de la imagen (arriba no hay más página) y otra idéntica en el
    medio (arriba sí la hay, sólo que fuera de la vista). La señal las separa
    y es la única entrada que lo hace."""
    pegada = edge_features(IMG, 20, 0, DIMS, "dist")
    en_medio = edge_features(IMG, 20, 20, DIMS, "dist")
    assert pegada[EDGE_SIDES.index("T")] == 1.0
    assert en_medio[EDGE_SIDES.index("T")] == 0.0
    assert not np.array_equal(pegada, en_medio)


# ------------------------------------- 3. entra por la cabeza, no por las convs

def test_la_senal_entra_SOLO_por_la_cabeza():
    """Lo que se pidió, comprobado por su consecuencia exacta: si el vector sólo
    se conecta a la Linear final, cambiarlo tiene que mover la salida en
    EXACTAMENTE `(b - a) @ W[:, -4:].T` — la proyección lineal por las cuatro
    columnas nuevas, sin ningún término más. Cualquier camino alternativo (un
    canal, un sesgo dependiente, una capa intermedia) rompe esta igualdad.

    Es más fuerte que "los mapas no cambian": `feature_maps` ni siquiera acepta
    el vector, así que compararlo consigo mismo no probaría nada."""
    m = _modelo("dist").eval()
    x = torch.randn(2, 1, DIMS.N, DIMS.N)
    a, b = torch.zeros(2, 4), torch.rand(2, 4)
    with torch.no_grad():
        salto = (m(x, b) - m(x, a)).view(2, 12)
        esperado = (b - a) @ m.head.weight[:, -4:].T
    assert salto.abs().sum() > 0                     # control: la señal hace algo
    torch.testing.assert_close(salto, esperado, rtol=1e-5, atol=1e-6)


def test_ninguna_conv_ensancha_su_entrada():
    """La otra mitad del mismo hecho, estructural: si la señal fuera un canal,
    la primera conv de cada rama pasaría de in_channels=1 a 2."""
    off, dist = _modelo("off"), _modelo("dist")
    for m in (off, dist):
        assert m.center_convs[0].in_channels == 1
        assert m.periph_convs[0].in_channels == 1
    # la ÚNICA capa que cambia de forma es la cabeza
    distintas = [k for k in off.state_dict()
                 if off.state_dict()[k].shape != dist.state_dict()[k].shape]
    assert distintas == ["head.weight"]


def test_el_gradiente_del_borde_solo_llega_a_la_cabeza():
    m = _modelo("pad")
    x = torch.zeros(2, 1, DIMS.N, DIMS.N)      # entrada muerta: sin señal de imagen
    e = torch.ones(2, 4)
    m(x, e).sum().backward()
    g = m.head.weight.grad
    assert g is not None and g[:, -4:].abs().sum() > 0        # las columnas de borde sí
    assert m.center_convs[0].weight.grad.abs().sum() == 0     # los kernels no


# ------------------------------------------------- la puerta: nada en silencio

def test_un_modo_desconocido_se_rechaza_en_vez_de_caer_a_off():
    """Caer a `off` entrenaría una red SIN la señal mientras su config —y todo
    el barrido construido sobre ella— dice que la tiene."""
    problemas = check_network(dict(BASE, edge_inputs="bordes"))
    assert [p["code"] for p in problemas] == ["unknown_edge_inputs"]
    with pytest.raises(FoveaError) as ex:
        n_edge_features("bordes")
    assert ex.value.code == "unknown_edge_inputs"


def test_pad_sin_borde_se_rechaza_porque_mediria_siempre_cero():
    """`border_px=0` (la CNN plana) no deja margen del que dar una fracción: la
    señal sería una constante 0 en una red que pide que le hablen del borde."""
    plano = dict(BASE, regions="single", border_px=0, edge_inputs="pad")
    assert [p["code"] for p in check_network(plano)] == ["edge_pad_needs_border"]
    with pytest.raises(FoveaError) as ex:
        edge_features(IMG, 0, 0, dims_of(plano), "pad")
    assert ex.value.code == "edge_pad_needs_border"
    # `dist` sí funciona ahí: mide contra la fóvea, que siempre existe
    assert check_network(dict(plano, edge_inputs="dist")) == []
    np.testing.assert_allclose(
        edge_features(IMG, 0, 0, dims_of(plano), "dist"), [1, 1, 0, 0])


def test_el_forward_se_niega_en_vez_de_suponer_ceros():
    """Un vector ausente NO puede valer 0: eso significa «no hay borde por
    ningún lado», que es falso justo en las ventanas para las que existe."""
    m = _modelo("pad")
    x = torch.zeros(1, 1, DIMS.N, DIMS.N)
    with pytest.raises(FoveaError) as ex:
        m(x)
    assert ex.value.code == "edge_inputs_missing"
    with pytest.raises(FoveaError) as ex:
        m(x, torch.zeros(1, 2))
    assert ex.value.code == "edge_inputs_shape"


# ------------------------------------------------------- es un eje de verdad (C)

def test_edge_inputs_es_un_eje_barrible_de_c():
    """La trampa que costó `dropout`: `full_config` filtra por NETWORK_DEFAULTS,
    así que un campo que no esté ahí se descarta y el barrido entrena N veces la
    MISMA red sin avisar."""
    from fv.sweeps.spec import NETWORK_PARAMS
    assert "edge_inputs" in NETWORK_PARAMS
    from fv.models.derive import STATIC_FIELDS
    assert "edge_inputs" in STATIC_FIELDS
    # y cada punto del eje construye una red distinta de verdad
    params = {m: network_trace(dict(BASE, edge_inputs=m))["num_params"]
              for m in EDGE_MODES}
    assert params["pad"] == params["dist"] > params["off"]


def test_el_dataloader_entrega_el_vector_con_cada_ventana(world):
    """Y lo entrega SIEMPRE, también apagado (vector vacío): un loader que a
    veces da 2-tuplas y a veces 3 rompe en la rama que nadie corrió."""
    from fv.windows.dataset import FoveatedWindowDataset
    from fv.windows.store import WindowDatasetStore
    from tests.conftest import TINY_NET

    arrays = WindowDatasetStore().arrays(world["dataset"])
    dims = dims_of(TINY_NET)
    for modo, ancho in (("off", 0), ("pad", 4), ("dist", 4)):
        ds = FoveatedWindowDataset(arrays, dims, split=0, edge_inputs=modo)
        x, e, y = ds[0]
        assert x.shape == (1, dims.N, dims.N) and y.shape == (4, 3)
        assert e.shape == (ancho,) and e.dtype == torch.float32
        lote = torch.utils.data.default_collate([ds[i] for i in range(4)])
        assert lote[1].shape == (4, ancho)
        # y el modelo lo consume tal cual sale del loader
        build_model(dict(TINY_NET, edge_inputs=modo))(lote[0], lote[1])


def test_un_run_entero_entrena_con_la_senal_encendida(world, tmp_path):
    """End to end: la señal atraviesa la puerta, el loader, el bucle y la
    evaluación. Es lo que separa «el campo existe» de «se puede entrenar»."""
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from tests.conftest import TINY_NET

    import json

    from fv.training.registry import RunStore

    net = dict(TINY_NET, edge_inputs="dist")
    store = RunStore()
    out = train("r-edge", world["dataset"], "n-edge", net, "corta",
                Recipe(epochs=1, batch_size=32, seed=1), store=store)
    assert out["epochs_run"] == 1 and out["best"] is not None
    cfg = json.loads((store.path("r-edge") / "config.json").read_text(encoding="utf-8"))
    assert cfg["network"]["edge_inputs"] == "dist"
    # y el checkpoint que deja se vuelve a cargar con la cabeza ensanchada
    from fv.inference.checkpoint import load_model
    m = load_model(store.path("r-edge") / "best.pt")
    assert m.n_edge == 4 and m.head.in_features == m.flat_features + 4


def test_un_recorrido_recorre_el_eje_de_verdad(world):
    """El mismo camino que anda un estudio (`generate_sweep` + `run_sweep`), que
    es lo que `scripts/verify_axes.py` comprueba para cada eje — aquí como test,
    porque una comprobación que hay que acordarse de lanzar no existe (R17).

    Lo que atrapa: que los tres puntos entrenen **redes distintas**. La trampa
    que costó `dropout` no da error, da tres runs iguales con nombres distintos."""
    from fv.sweeps.generate import generate_sweep
    from fv.sweeps.runner import run_sweep, sweep_trials
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore

    from tests.test_generate import RECIPE

    ss, rs = SweepStore(), RunStore()
    generate_sweep("ei-chk", world["dataset"], "edge_inputs", list(EDGE_MODES),
                   base_recipe="corta", base_recipe_value=RECIPE, objective="f1",
                   budget={"epochs": 1}, sstore=ss)
    estado = run_sweep("ei-chk", ss, rs)
    trials = sweep_trials("ei-chk", ss, rs)["trials"]
    hechos = [t for t in trials if t["status"] == "done"]
    assert estado["status"] == "done" and len(hechos) == len(EDGE_MODES)
    assert sorted(t["point"]["edge_inputs"] for t in hechos) == sorted(EDGE_MODES)
    # y cada punto entrenó una red distinta: los dos encendidos tienen 48 pesos
    # más que el apagado. Sin esto, un eje "verde" puede ser tres veces lo mismo.
    import json
    tam = {}
    for t in hechos:
        cfg = json.loads((rs.path(t["run"]) / "config.json").read_text(encoding="utf-8"))
        tam[t["point"]["edge_inputs"]] = network_trace(cfg["network"])["num_params"]
    assert tam["pad"] == tam["dist"] == tam["off"] + 4 * 12
