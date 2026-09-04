"""Las decisiones de `2026-09-04-preproceso-kernel-congelado`, fijadas.

Cada test corresponde a una decisión escrita en el README de ese experimento o en
su criterio. Si alguna se rompe, se rompe AQUI y no en un entrenamiento de tres
horas que sale con un numero creible.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

RAIZ = Path(__file__).resolve().parents[1]
EXP = RAIZ / "experimentos" / "2026-09-04-preproceso-kernel-congelado"
COMUN = RAIZ / "experimentos" / "comun"


@pytest.fixture(scope="module")
def red_local():
    sys.path.insert(0, str(EXP / "nn"))
    sys.path.insert(0, str(COMUN))
    sys.path.insert(0, str(RAIZ / "src"))
    import red_local
    return red_local


# --------------------------------------------------------------- la geometria
@pytest.mark.parametrize("kf,features", [(3, 256), (5, 196), (7, 144)])
def test_la_cabeza_se_dimensiona_sola_y_da_las_features_declaradas(
        red_local, kf, features):
    """324/256/196/144 es EL confound del experimento: si cambia, el criterio y
    las anclas iso-features dejan de valer."""
    assert red_local.construir(kf).flat_features == features


def test_las_features_coinciden_con_las_anclas_ya_corridas(red_local):
    """256 y 196 son exactamente las de `1k5` y `1k7` crudos. Es lo que hace que
    el control iso-features salga a coste cero; si deja de coincidir, la columna
    «Δ misma época» de `comparativa.py` compara cosas distintas."""
    assert red_local.construir(3).flat_features == 16 * 16   # = 1k5 crudo
    assert red_local.construir(5).flat_features == 14 * 14   # = 1k7 crudo


# ------------------------------------------------------------- la congelacion
def test_la_capa_congelada_no_se_mueve_con_el_optimizador(red_local):
    """Si L1 entrenara, esto seria OTRO experimento --y sus numeros seguirian
    saliendo igual de creibles--. Por eso se comprueba, no se da por hecho."""
    m = red_local.construir(5)
    antes = m.center_convs[0].weight.detach().clone()
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    m(torch.randn(2, 2, 20, 20), torch.zeros(2, m.n_edge)).sum().backward()
    opt.step()
    assert torch.equal(antes, m.center_convs[0].weight.detach())


def test_la_capa_congelada_es_el_kernel_del_experimento_del_que_sale(red_local):
    """La identidad del brazo la da el TENSOR, no el nombre del run (R16)."""
    sys.path.insert(0, str(COMUN))
    from preproceso import cargar_kernel
    for kf in (3, 5, 7):
        m = red_local.construir(kf)
        esperado = cargar_kernel(f"1k{kf}", pesos=red_local.PESOS)
        assert torch.equal(m.center_convs[0].weight.detach(), esperado.peso)
        assert not m.center_convs[0].weight.requires_grad


# ------------------------------------------------- por que hay una ReLU en medio
def test_sin_relu_las_dos_convoluciones_colapsan_en_una(red_local):
    """El motivo de que la ReLU este puesta: sin ella el brazo seria un
    subconjunto estricto de un gemelo YA CORRIDO, y el estudio degenerado."""
    import torch.nn.functional as F
    m = red_local.construir(5)
    w1, b1 = m.center_convs[0].weight.detach(), m.center_convs[0].bias.detach()
    w2, b2 = m.center_convs[1].weight.detach(), m.center_convs[1].bias.detach()
    x = torch.randn(3, 2, 20, 20)
    compuesto = F.conv2d(F.conv2d(x, w1, b1), w2, b2)
    k1, k2 = w1.shape[-1], w2.shape[-1]
    w_eff = torch.zeros(1, w1.shape[1], k1 + k2 - 1, k1 + k2 - 1)
    for c in range(w1.shape[1]):
        w_eff[0, c] = F.conv2d(w1[0, c][None, None], torch.flip(w2, [2, 3]),
                               padding=k2 - 1)[0, 0]
    una = F.conv2d(x, w_eff, b2 + b1 * w2.sum())
    assert torch.allclose(compuesto, una, atol=1e-5)


def test_la_relu_esta_puesta_o_sea_que_la_red_NO_colapsa(red_local):
    """El complemento del anterior: la red de verdad (con su ReLU) NO es lineal,
    asi que no equivale a ninguna convolucion suelta."""
    m = red_local.construir(5).eval()
    x = torch.randn(4, 2, 20, 20)
    e = torch.zeros(4, m.n_edge)
    with torch.no_grad():
        # linealidad: f(2x) == 2·f(x) solo si no hay ReLU de por medio
        assert not torch.allclose(m(2 * x, e), 2 * m(x, e), atol=1e-4)


# --------------------------------------------------- el lector unico de metricas
def test_serie_leer_sin_sub_sigue_leyendo_los_gemelos_igual():
    """`sub` se anadio para los brazos; los siete gemelos NO pueden cambiar."""
    sys.path.insert(0, str(COMUN))
    import serie
    d = serie.leer("2026-09-04-cnn-plana-1k3-sinpadding")
    assert d is not None
    por_ep, best, _L = d
    assert por_ep[37]["val"]["f1"] == pytest.approx(0.680, abs=5e-4)
    assert serie.leer("2026-09-04-cnn-plana-1k3-sinpadding", "") == d


def test_serie_leer_con_sub_encuentra_un_brazo():
    sys.path.insert(0, str(COMUN))
    import serie
    if not (EXP / "nn" / "pesos" / "1k5" / "metrics.jsonl").exists():
        pytest.skip("el brazo 1k5 no esta corrido en esta copia")
    assert serie.leer(EXP.name, "1k5") is not None


# ------------------------------------------- la invariante de la comparacion
def test_comparativa_no_declara_antes_de_la_epoca_11():
    """Medido en esta serie: el orden a la ep. 3 sale INVERTIDO respecto al
    final. Un informe que declare ahi concluye lo contrario de lo que resulta."""
    r = subprocess.run([sys.executable, str(EXP / "nn" / "comparativa.py")],
                       capture_output=True, text=True, cwd=RAIZ)
    if "sin correr" in r.stdout or not r.stdout.strip():
        pytest.skip("no hay brazos corridos en esta copia")
    epocas = [l for l in r.stdout.splitlines() if "NO SE DECLARA NADA" in l]
    declara = [l for l in r.stdout.splitlines() if "el preproceso" in l]
    assert bool(epocas) != bool(declara), (
        "o se declara, o se dice que no se declara -- nunca las dos ni ninguna")


def test_el_experimento_no_toca_produccion(red_local):
    """La instruccion del dueno (2026-09-03) que rige toda la serie: el parche es
    SOLO en memoria.

    ⚠ Se llama al guard de verdad (`_comprobar_intacto`) en vez de buscar la
    cadena `builder.py` por los ficheros: la primera version de este test hacia
    eso y saltaba por la RUTA CITADA EN UN DOCSTRING -- un falso positivo que
    habria que silenciar cada vez que alguien documenta de que fichero hereda.
    """
    sys.path.insert(0, str(EXP / "nn"))
    import entrenar_local
    from fv.models import builder as _builder
    entrenar_local._comprobar_intacto()          # revienta si produccion cambio
    assert _builder.build_model is entrenar_local._ORIGINAL
    # y construir un brazo no puede dejar el simbolo global pisado
    red_local.construir(3)
    assert _builder.build_model is entrenar_local._ORIGINAL
