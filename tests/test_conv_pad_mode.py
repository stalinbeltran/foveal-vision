"""`conv_pad_mode`: con que rellena la CONVOLUCION fuera del borde de la vista.

Por que existe el campo, y por que su defecto es `zeros`: el 2026-09-03 se midio
que rellenar con ceros mete un salto sistematico en el anillo de k//2 px --9,4x el
interior en la plana de 7x7, y el 64 % de las celdas de la periferia de la foveada
ENTRENADA cambian si se cambia el relleno--. La vista se recorta de una imagen mas
grande, asi que esos pixeles EXISTEN y ponerles cero es la regla falsa que
`pad_mode: edge` evita (decision C10).

⚠ Lo que estos tests protegen NO es que `replicate` sea mejor --eso no esta
medido-- sino que **anadir el campo no cambio nada de lo que ya habia**. Un
cambio de defecto aqui no rompe ningun checkpoint (no toca ni un parametro), asi
que fallaria en SILENCIO sobre todas las tablas publicadas.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

torch = pytest.importorskip("torch")

from fv.fovea import CONV_PAD_MODES, FoveaError          # noqa: E402
from fv.models.builder import NETWORK_DEFAULTS, build_model, full_config  # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parents[1]
REDES = sorted((RAIZ / "configs" / "networks").glob("*.yaml"))


def _cfg(nombre="plana-20-4k7", **extra):
    base = yaml.safe_load((RAIZ / "configs" / "networks" / f"{nombre}.yaml").read_text())
    return full_config({**base, **extra})


def test_el_defecto_es_zeros_y_eso_es_lo_que_habia_antes():
    """Si alguien cambia este defecto, TODAS las redes guardadas pasan a
    significar otra cosa y ningun checkpoint deja de cargar."""
    assert NETWORK_DEFAULTS["conv_pad_mode"] == "zeros"


@pytest.mark.parametrize("red", [f.stem for f in REDES])
def test_ninguna_config_del_repo_cambia_al_anadir_el_campo(red):
    """El campo es nuevo: ninguna config lo declara, asi que todas tienen que
    seguir construyendose con `zeros`."""
    m = build_model(_cfg(red))
    assert m.center_convs[0].padding_mode == "zeros"


def test_declararlo_como_zeros_da_EXACTAMENTE_la_misma_red():
    torch.manual_seed(1)
    a = build_model(_cfg())
    torch.manual_seed(1)
    b = build_model(_cfg(conv_pad_mode="zeros"))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_replicate_llega_a_las_convoluciones_y_cambia_la_salida():
    torch.manual_seed(1)
    a = build_model(_cfg())
    torch.manual_seed(1)
    b = build_model(_cfg(conv_pad_mode="replicate"))
    assert b.center_convs[0].padding_mode == "replicate"
    # los PESOS son identicos: la unica variable es el relleno, que es lo que
    # hace que el experimento de control compare una sola cosa
    assert torch.equal(a.center_convs[0].weight, b.center_convs[0].weight)
    x = torch.randn(2, 2, 20, 20)
    assert not torch.allclose(a.center_convs[0](x), b.center_convs[0](x))


def test_el_cambio_se_queda_en_el_ANILLO_y_no_toca_el_interior():
    """Fija el alcance: `conv_pad_mode` sólo puede mover las celdas a menos de
    k//2 del borde. Si algún día moviera el interior, sería otro bug."""
    torch.manual_seed(1)
    a = build_model(_cfg())
    torch.manual_seed(1)
    b = build_model(_cfg(conv_pad_mode="replicate"))
    k = a.center_convs[0].kernel_size[0]
    c = slice(k // 2, -(k // 2))
    x = torch.randn(2, 2, 20, 20)
    ya, yb = a.center_convs[0](x), b.center_convs[0](x)
    assert torch.allclose(ya[..., c, c], yb[..., c, c], atol=1e-6)


def test_tambien_llega_a_la_rama_periferica_de_una_red_split():
    m = build_model(_cfg("fov16-optimo-mask", conv_pad_mode="replicate"))
    assert m.center_convs[0].padding_mode == "replicate"
    assert m.periph_convs[0].padding_mode == "replicate"


def test_un_valor_que_no_existe_se_RECHAZA_al_construir():
    """Ruidoso, no silencioso: torch aceptaria cualquier cadena rara sólo al
    llegar al forward, y para entonces ya estarías entrenando."""
    with pytest.raises(FoveaError) as e:
        build_model(_cfg(conv_pad_mode="patata"))
    # el CODIGO, no solo el tipo: es lo que un cliente puede casar (api.md R4)
    assert e.value.code == "unknown_conv_pad_mode"
    assert "patata" in e.value.message


def test_los_modos_declarados_son_los_que_torch_admite():
    for modo in CONV_PAD_MODES:
        torch.nn.Conv2d(1, 1, 3, padding=1, padding_mode=modo)
