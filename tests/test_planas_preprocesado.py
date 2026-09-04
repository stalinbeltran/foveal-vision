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


# ----------------------------------------- la definicion, con su procedencia
def test_los_parametros_son_los_que_pidio_el_dueno(red):
    """Redefinicion del 2026-09-04: «Las capas de la nn seran 2... Cada capa va a
    tener solo 2 canales. El padding va a ser siempre sin padding... el stride va a
    ser la mitad del ancho del kernel (redondeado)». Si alguien los mueve sin
    querer, las tres redes dejan de ser lo pedido y el preliminar mide otra cosa."""
    assert red.PARAMS == {"n_layers": 2, "canales": 2, "k": 3, "pad": 0,
                          "dropout": 0.0}
    assert red.STRIDE == 2                       # (3+1)//2: la mitad, REDONDEADA


def test_sin_relleno_y_el_stride_va_en_TODAS_las_capas(red):
    """«El padding del kernel va a ser siempre sin padding» y «cada capa va a
    reducir el tamano de los features». `builder.py` pone el stride solo en la
    primera capa (D-S1); aqui va en las dos, y es deliberado."""
    m = red.construir("1k3")
    assert m.pad == 0
    for c in m.convs:
        assert c.padding == (0, 0)
        assert c.stride == (2, 2)
    lados = [f[1][-1] for f in m.forma()[:3]]
    assert lados == [18, 8, 3]
    assert lados[0] > lados[1] > lados[2], "cada capa tiene que reducir"


def test_k3_es_lo_UNICO_que_cabe_con_estas_condiciones(red):
    """El encargo no dijo el tamano del kernel. Con 2 capas, sin relleno y stride =
    mitad de k, solo k=3 sobrevive: k=5 y k=7 dejan el mapa en 0 o negativo. Esto lo
    fija para que nadie lo suba «porque el optimo foveado lo permite»."""
    def lado(n, k, s, capas=2):
        for _ in range(capas):
            n = (n - k) // s + 1
        return n
    for n0 in (18, 16, 14):
        assert lado(n0, 3, 2) >= 1, "k=3 tiene que caber en los tres brazos"
    # y los otros dos NO caben
    assert min(lado(n0, 5, 3) for n0 in (18, 16, 14)) <= 0
    assert min(lado(n0, 7, 4) for n0 in (18, 16, 14)) <= 0


def test_el_coste_es_MINIMO_y_esta_medido(red):
    """«Queremos resultados preliminares, vamos a reducir el coste al minimo»."""
    tot = {b: sum(p.numel() for p in red.construir(b).parameters())
           for b in red.ENTRADAS}
    assert tot == {"1k3": 286, "1k5": 286, "1k7": 166}
    assert max(tot.values()) < 69_340 / 100, "242x menos que la version de 4 capas"


# ------------------------------ LA premisa del encargo: identicas salvo la entrada
@pytest.mark.parametrize("brazo", ["1k5", "1k7"])
def test_las_convoluciones_son_IDENTICAS_entre_brazos(red, brazo):
    """«Las 3 cnn seran identicas, salvo por los parametros afectados por los
    datasets de entrada». Las convs no dependen del tamano de entrada, asi que
    tienen que salir iguales hasta en el numero de parametros."""
    a, b = red.construir("1k3"), red.construir(brazo)
    assert len(a.convs) == len(b.convs) == 2
    for ca, cb in zip(a.convs, b.convs):
        assert (ca.kernel_size, ca.stride, ca.padding, ca.out_channels, ca.in_channels) \
            == (cb.kernel_size, cb.stride, cb.padding, cb.out_channels, cb.in_channels)
    n = lambda m: sum(p.numel() for c in m.convs for p in c.parameters())
    assert n(a) == n(b) == 58
    assert a.drop.p == b.drop.p == 0.0


def test_lo_unico_que_cambia_es_la_cabeza_y_1k3_y_1k5_EMPATAN(red):
    """⚠ Con k=3 y stride 2 el `1k3` y el `1k5` caen los DOS en 18 features, o sea
    que son ISO-FEATURES por construccion -- justo el confound que en la version de
    4 capas habia que corregir con anclas externas. No se buscaba; se fija aqui
    porque es lo que hace comparables esos dos brazos."""
    anchos = {b: red.construir(b).flat_features for b in red.ENTRADAS}
    assert anchos == {"1k3": 18, "1k5": 18, "1k7": 8}
    assert anchos["1k3"] == anchos["1k5"], "iso-features por construccion"


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
