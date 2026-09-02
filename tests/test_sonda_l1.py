"""La sonda L1: los invariantes que, si se rompen, la hacen medir OTRA cosa.

No se testea "que entrene". Se testean las tres decisiones cuya rotura es
SILENCIOSA -- el experimento seguiria corriendo y dando numeros creibles:

1. la renormalizacion del decodificador (sin ella, `lambda` premia hacer `z`
   pequeno en vez de disperso, y el barrido en lambda no mide nada);
2. el nulo de la metrica de enriquecimiento (si la base clasica no fuera
   ortonormal, "energia en el subespacio" seria una proyeccion oblicua y el
   6/k^2 dejaria de ser el valor de un kernel aleatorio);
3. que la resolucion se conserva (el encargo dice explicitamente que no quiere
   una imagen mas pequena).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("sonda_l1", ROOT / "scripts" / "sonda_l1.py")
sonda = importlib.util.module_from_spec(_spec)
sys.modules["sonda_l1"] = sonda
_spec.loader.exec_module(sonda)


# ------------------------------------------------- 1. la salida degenerada

@pytest.mark.parametrize("k", [3, 5, 9])
def test_los_atomos_del_decodificador_quedan_a_norma_uno(k):
    m = sonda.SondaL1(K=6, k=k)
    m.renormaliza()
    n = m.dec.weight.detach().flatten(1).norm(dim=1)
    assert torch.allclose(n, torch.ones(6), atol=1e-5)


def test_sin_renormalizar_el_modelo_puede_encoger_z_en_vez_de_dispersarlo():
    """La salida degenerada existe de verdad: es lo que la decision 1 bloquea.

    Escalar el codificador por a y el decodificador por 1/a deja la
    reconstruccion IGUAL y divide la penalizacion por a. Sin el freno, bajar
    `mean(|z|)` sale gratis.
    """
    m = sonda.SondaL1(K=4, k=3)
    x = torch.randn(2, 1, 20, 20)
    with torch.no_grad():
        xh0, z0 = m(x)
        a = 0.01
        m.enc.weight.mul_(a); m.enc.bias.mul_(a)
        m.dec.weight.div_(a)
        xh1, z1 = m(x)
    assert torch.allclose(xh0, xh1, atol=1e-4)          # misma reconstruccion
    assert float(z1.abs().mean()) < float(z0.abs().mean()) * 0.02   # 50x menos pena
    m.renormaliza()                                      # ...y el freno lo deshace
    assert torch.allclose(m.dec.weight.detach().flatten(1).norm(dim=1),
                          torch.ones(4), atol=1e-5)


def test_la_renormalizacion_sobrevive_a_un_paso_del_optimizador():
    m = sonda.SondaL1(K=5, k=5)
    m.renormaliza()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.randn(4, 1, 20, 20)
    for _ in range(3):
        xh, z = m(x)
        loss = ((xh - x) ** 2).mean() + 0.1 * z.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        m.renormaliza()
    n = m.dec.weight.detach().flatten(1).norm(dim=1)
    assert torch.allclose(n, torch.ones(5), atol=1e-5)


# ------------------------------------------------- 2. el nulo de la metrica

@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_la_base_clasica_es_ortonormal_y_tiene_6_filtros(k):
    B = sonda._base_clasica(k)
    assert B.shape == (6, k * k)
    assert torch.allclose(B @ B.T, torch.eye(6), atol=1e-4)


@pytest.mark.parametrize("k", [3, 5, 9])
def test_un_kernel_aleatorio_da_enriquecimiento_1(k):
    """El nulo 6/k^2 tiene que ser el valor MEDIDO de un kernel aleatorio.

    Es lo que hace que "0,688 en 3x3" se lea como "indistinguible de aleatorio":
    sin este anclaje, la fraccion cruda no significa nada.
    """
    B = sonda._base_clasica(k)
    g = torch.Generator().manual_seed(0)
    W = torch.randn(4000, k * k, generator=g)
    W = W / W.norm(dim=1, keepdim=True)
    frac = (W @ B.T).pow(2).sum(1).mean()
    assert float(frac) == pytest.approx(6.0 / (k * k), rel=0.05)


@pytest.mark.parametrize("k", [3, 5])
def test_los_propios_filtros_clasicos_dan_el_enriquecimiento_maximo(k):
    """El otro extremo: si los kernels SON la base, la energia es 1."""
    B = sonda._base_clasica(k)
    frac = (B @ B.T).pow(2).sum(1)
    assert torch.allclose(frac, torch.ones(6), atol=1e-4)


# ------------------------------------------------- 3. la resolucion se conserva

@pytest.mark.parametrize("k", [3, 5, 7, 9])
@pytest.mark.parametrize("K", [8, 32])
def test_la_vista_entra_y_sale_a_20x20(k, K):
    m = sonda.SondaL1(K=K, k=k)
    x = torch.zeros(2, 1, 20, 20)
    xh, z = m(x)
    assert tuple(z.shape) == (2, K, 20, 20)
    assert tuple(xh.shape) == (2, 1, 20, 20)


def test_el_codificador_replica_el_borde_como_pad_mode_edge():
    assert sonda.SondaL1(K=4, k=5).enc.padding_mode == "replicate"


def test_convtranspose_no_admite_replicate_y_por_eso_hay_err_rec_int():
    """Fija la razon por la que la metrica interior existe.

    Si una version de torch pasara a admitirlo, este test falla y hay que
    revisar la decision -- que es exactamente lo que se quiere que pase.
    """
    with pytest.raises(ValueError):
        torch.nn.ConvTranspose2d(4, 1, 5, padding=2, bias=False,
                                 padding_mode="replicate")


# ------------------------------------------------- 4. el preprocesado

def test_la_normalizacion_deja_media_local_cero_y_no_explota_en_lo_plano():
    x = torch.zeros(1, 1, 20, 20)
    x[:, :, 8:12, 8:12] = 1.0                     # una mancha, el resto plano
    y = sonda.normaliza_contraste(x, sigma=2.0, eps=0.0148)
    assert torch.isfinite(y).all()
    # La comprobacion es una RAZON, no un umbral tecleado: en la esquina queda
    # una respuesta real (la cola de la gaussiana llega, sigma=2 y la mancha esta
    # a 4 px), y lo que importa es que sea despreciable frente a la mancha. Un
    # umbral absoluto habria que reajustarlo al tocar sigma; esto no.
    esquina = float(y[0, 0, :4, :4].abs().max())
    mancha = float(y.abs().max())
    assert mancha > 1.0                           # la mancha SI sobrevive
    assert esquina < mancha / 100                 # lo plano NO se amplifica


def test_una_constante_se_normaliza_a_cero():
    y = sonda.normaliza_contraste(torch.full((1, 1, 20, 20), 0.7), 2.0, 0.0148)
    assert float(y.abs().max()) < 1e-3
