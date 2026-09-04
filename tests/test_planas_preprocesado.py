"""Las estructuras de `2026-09-04-planas-sobre-preprocesado`, fijadas.

Cada test corresponde a una decision del README de ese experimento. Lo que se fija
aqui es sobre todo la premisa del encargo --«las 3 cnn seran identicas, salvo por
los parametros afectados por los datasets de entrada»--, que es una invariante y no
una intencion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

RAIZ = Path(__file__).resolve().parents[1]
EXP = RAIZ / "experimentos" / "2026-09-04-planas-sobre-preprocesado"


def _modulo(nombre: str, ruta: Path):
    """Carga un fichero como modulo con nombre PROPIO.

    ⚠ No vale `sys.path.insert` + `import red_local`: hay DOS experimentos con un
    `nn/red_local.py`, y el primero que importe se queda con el nombre en
    `sys.modules` -- el segundo recibe el modulo del otro y sus tests fallan por
    algo que no tiene nada que ver. Medido el 2026-09-04: 9 fallos en el
    experimento detenido en cuanto se anadio el test del nuevo.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def red():
    sys.path.insert(0, str(RAIZ / "src"))
    return _modulo("red_local_planas_preproc", EXP / "nn" / "red_local.py")


# ⚠ EXCEPCION AL FIXTURE GLOBAL, con motivo -- la misma que `test_preproceso.py`:
# `conftest.py` apunta `FV_DATA_ROOT` a un temporal VACIO para que ningun test toque
# el repo de datos real, y eso es una barrera de seguridad, no comodidad. Pero estos
# dos tests existen justamente para contrastar contra el dataset CONSTRUIDO, y con el
# fixture puesto se saltaban siempre: una comprobacion que nunca corre no existe (R17).
# Se apunta al repo hermano y se LEE; no se escribe nada. Si no esta, se salta.
PREPROCESADO = RAIZ.parent / "foveal-vision-data" / "preprocesado"


def _npz(brazo: str):
    carp = sorted(PREPROCESADO.glob(f"{brazo}-*"))
    if not carp:
        pytest.skip(f"no esta {PREPROCESADO}/{brazo}-*: dataset sin construir")
    return carp[0] / "preprocesado.npz"


# ------------------------------------------------- los optimos, con procedencia
def test_los_optimos_son_los_de_ESTADO_md(red):
    """Si alguien los cambia sin querer, las tres redes dejan de ser «la foveada
    optima» y el experimento mide otra cosa. Los valores y su evidencia estan en
    `estudios-redes-neuronales/ESTADO.md`, seccion «Red foveada»."""
    assert red.OPTIMOS_FOVEADA == {
        "n_layers": 4,      # cerrado: 4 -> 0,9341 contra 3 -> 0,9246 y 5 -> 0,9136
        "k": 3,             # `k_center` cerrado: 5 y 7 peores Y mas caros
        "canales": 16,      # `channels` cerrado 20/20: 16 es el suelo util
        "stride": 1,        # `s_center`: un solo valor legal
        "dropout": 0.0,     # tanteo; 0,1 es el PEOR de los cuatro
    }


def test_el_relleno_es_k_medios_como_la_foveada_NO_cero(red):
    """`builder.py:145` calcula `pad = k_center // 2`. Los siete gemelos usan 0,
    pero eso era su EJE, no un optimo medido: ESTADO.md no tiene ninguna fila que
    diga que 0 gane. Cambiarlo mueve la cabeza 3,2x (5.184 -> 1.600 en el 1k3)."""
    m = red.construir("1k3")
    assert m.pad == red.OPTIMOS_FOVEADA["k"] // 2 == 1
    for c in m.convs:
        assert c.padding == (1, 1)
    # y con `same` la resolucion NO cae por las capas
    assert m.forma()[-3][1] == (16, 18, 18)


# ------------------------------ LA premisa del encargo: identicas salvo la entrada
@pytest.mark.parametrize("brazo", ["1k5", "1k7"])
def test_las_convoluciones_son_IDENTICAS_entre_brazos(red, brazo):
    """«Las 3 cnn seran identicas, salvo por los parametros afectados por los
    datasets de entrada». Las convs no dependen del tamano de entrada, asi que
    tienen que salir iguales hasta en el numero de parametros."""
    a, b = red.construir("1k3"), red.construir(brazo)
    assert len(a.convs) == len(b.convs) == 4
    for ca, cb in zip(a.convs, b.convs):
        assert (ca.kernel_size, ca.stride, ca.padding, ca.out_channels, ca.in_channels) \
            == (cb.kernel_size, cb.stride, cb.padding, cb.out_channels, cb.in_channels)
    n = lambda m: sum(p.numel() for c in m.convs for p in c.parameters())
    assert n(a) == n(b) == 7120
    assert a.drop.p == b.drop.p == 0.0


def test_lo_UNICO_que_cambia_es_la_cabeza(red):
    """El complemento del anterior: si las cabezas salieran iguales, el test de
    arriba pasaria sin probar nada (los tres brazos serian la misma red)."""
    anchos = {b: red.construir(b).flat_features for b in red.ENTRADAS}
    assert anchos == {"1k3": 5184, "1k5": 4096, "1k7": 3136}
    assert len(set(anchos.values())) == 3


# --------------------------------------------- la forma casa con el dato de verdad
def test_la_entrada_declarada_casa_con_el_npz_construido(red):
    """R4: la forma se DECLARA en `ENTRADAS`, pero tiene que casar con el dataset.
    Si no casan, el entrenamiento revienta al primer lote -- o peor, no revienta."""
    np = pytest.importorskip("numpy")
    for brazo, esperada in red.ENTRADAS.items():
        real = tuple(np.load(_npz(brazo))["x"].shape[1:])
        assert real == esperada, brazo


@pytest.mark.parametrize("brazo", ["1k3", "1k5", "1k7"])
def test_un_forward_da_cuatro_esquinas_por_tres(red, brazo):
    c, h, w = red.ENTRADAS[brazo]
    assert tuple(red.construir(brazo)(torch.zeros(5, c, h, w)).shape) == (5, 4, 3)


def test_la_relu_va_ENTRE_capas_y_no_tras_la_ultima(red):
    """Igual que `builder._branch_forward`. Se comprueba por el comportamiento: el
    mapa final tiene que poder ser NEGATIVO (queda pre-activacion)."""
    m = red.construir("1k3").eval()
    with torch.no_grad():
        mapa = m._ramas(torch.randn(8, 1, 18, 18))
    assert (mapa < 0).any(), "el mapa final no puede estar activado"


def test_no_toca_produccion(red):
    """Instruccion del dueno (2026-09-03) que rige toda la serie."""
    assert "padding=pad" in (RAIZ / "src" / "fv" / "models" / "builder.py").read_text()


# ------------------------------------------------------------- el hueco declarado
def test_el_dataset_NO_trae_los_escalares_de_borde_y_por_eso_n_edge_es_0(red):
    """⚠ Hallazgo, no capricho: el `.npz` guarda `x`, `y` y `split` -- NO
    `window_xy`, asi que `edge_features` no se puede calcular al entrenar. La red se
    construye con `n_edge=0` y esto lo deja fijado: si algun dia se anaden los 4
    escalares al dataset, este test falla y obliga a mirar la cabeza."""
    np = pytest.importorskip("numpy")
    claves = set(np.load(_npz("1k3")).files)
    assert claves == {"x", "y", "split"}
    m = red.construir("1k3")
    assert m.n_edge == 0
    assert m.head.in_features == m.flat_features
