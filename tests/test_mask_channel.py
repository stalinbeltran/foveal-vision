"""El canal de relleno: que la red sepa que pixeles son inventados.

El fallo que ataca esta MEDIDO (docs/plan-mask-channel-2026-09-01.md): en la
esquina pegada al borde de la imagen el recall cae de 0,97 a 0,61, porque
`pad_mode: edge` replica la fila del borde y eso es indistinguible de imagen
real que sigue. Aqui se fija el mecanismo, no que mejore.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fv.fovea import (MASK_MODES, build_masks, build_view, derive_dims,
                      input_stack, n_input_channels, FoveaError)
from fv.models.builder import build_model, full_config, network_trace
from fv.validation import check_network

BASE = {"fovea_px": 16, "border_px": 8, "border_reduce": 4, "overlap_fovea_px": 7,
        "overlap_border_px": 0, "n_layers": 4, "channels": [16, 16, 16, 16]}
CHICA = {"fovea_px": 16, "border_px": 4, "border_reduce": 2, "overlap_fovea_px": 2,
         "overlap_border_px": 0, "n_layers": 2, "channels": [16, 16]}


def _imagen(H=60, W=80):
    """Papel claro con un parrafo PEGADO al borde superior izquierdo: el caso."""
    im = np.full((H, W), 244, dtype=np.uint8)
    im[0:20, 0:30] = 90
    return im


def test_off_es_el_defecto_y_no_mueve_nada():
    """La razon por la que el defecto es 'off': una config guardada tiene que
    seguir significando lo mismo y los checkpoints en disco seguir cargando."""
    cfg = full_config(BASE)
    assert cfg["mask_channel"] == "off"
    off = build_model(cfg)
    assert off.center_convs[0].weight.shape[1] == 1
    assert off.periph_convs[0].weight.shape[1] == 1
    # y el state_dict tiene exactamente las mismas claves y formas
    con = build_model(full_config({**BASE, "mask_channel": "coverage"}))
    iguales = {k: v.shape for k, v in off.state_dict().items()}
    otras = {k: v.shape for k, v in con.state_dict().items()}
    assert set(iguales) == set(otras)
    distintas = [k for k in iguales if iguales[k] != otras[k]]
    assert distintas == ["periph_convs.0.weight"], distintas


def test_el_canal_va_solo_a_la_rama_que_ve_el_anillo():
    """MEDIDO: bajo la mascara del centro la cobertura es 1,000 SIEMPRE, tambien
    en la ventana (0,0) -- la fovea esta dentro de la imagen por construccion.
    Darle el canal al centro serian 144 pesos equivalentes a un sesgo."""
    for geo in (BASE, CHICA):
        d = derive_dims(geo)
        cm, pm = build_masks(d)
        _v, cov = build_view(_imagen(), 0, 0, d, pad_mode="edge")
        assert (cov[cm > 0] == 1.0).all(), geo         # el centro no ve relleno
        assert (cov[pm > 0] < 1.0).any(), geo          # la periferia si
    m = build_model(full_config({**BASE, "mask_channel": "coverage"}))
    assert m.center_convs[0].weight.shape[1] == 1
    assert m.periph_convs[0].weight.shape[1] == 2


def test_el_canal_es_el_RELLENO_no_la_cobertura():
    """La orientacion, que es la trampa: 0 = imagen real, 1 = todo inventado.
    Igual que `edge_features` (0 = no hay borde). Invertirlo entrena una red que
    lee la senal al reves sin que nada falle."""
    d = derive_dims(BASE)
    v, cov = build_view(_imagen(), 0, 0, d, pad_mode="edge")
    x = input_stack(v, cov, "coverage")
    assert x.shape == (2, d.N, d.N) and x.dtype == np.float32
    np.testing.assert_allclose(x[0], v)
    np.testing.assert_allclose(x[1], 1.0 - cov)
    assert x[1].max() == pytest.approx(1.0)            # esquina: hay relleno
    # y en una ventana interior no hay nada que declarar
    v2, cov2 = build_view(_imagen(), 32, 24, d, pad_mode="edge")
    assert input_stack(v2, cov2, "coverage")[1].max() == 0.0


def test_con_off_input_stack_da_un_canal():
    d = derive_dims(BASE)
    v, cov = build_view(_imagen(), 0, 0, d, pad_mode="edge")
    assert n_input_channels("off") == 1
    assert input_stack(v, cov, "off").shape == (1, d.N, d.N)


def test_el_coste_es_de_una_rama():
    """+144 pesos (una conv de 1 a 2 canales de entrada), no +288."""
    off = sum(p.numel() for p in build_model(full_config(BASE)).parameters())
    con = sum(p.numel() for p in
              build_model(full_config({**BASE, "mask_channel": "coverage"})).parameters())
    assert con - off == 144
    assert (con - off) / off < 0.001


def test_la_plana_si_recibe_el_canal_entera():
    """El control sin regiones convoluciona TODO, asi que ahi el relleno si le
    llega a la unica rama que hay."""
    plana = {**CHICA, "regions": "single"}
    m = build_model(full_config({**plana, "mask_channel": "coverage"}))
    assert m.single
    assert m.center_convs[0].weight.shape[1] == 2
    d = m.dims
    v, cov = build_view(_imagen(), 0, 0, d, pad_mode="edge")
    x = torch.from_numpy(input_stack(v, cov, "coverage"))[None]
    assert m(x).shape == (1, 4, 3)


def test_forward_con_los_dos_modos():
    for modo, canales in (("off", 1), ("coverage", 2)):
        m = build_model(full_config({**BASE, "mask_channel": modo}))
        x = torch.zeros(3, canales, m.dims.N, m.dims.N)
        assert m(x).shape == (3, 4, 3)


def test_la_puerta_se_niega_antes_de_entrenar():
    """R2: o degrada con un defecto declarado, o falla ANTES de empezar."""
    p = check_network({**BASE, "mask_channel": "nope"})
    assert [x["code"] for x in p] == ["unknown_mask_channel"]
    assert p[0]["hint"]
    # sin margen no hay relleno del que hablar: seria una entrada muerta
    sin_borde = {"fovea_px": 16, "border_px": 0, "border_reduce": 1,
                 "overlap_fovea_px": 0, "overlap_border_px": 0, "n_layers": 2,
                 "regions": "single", "mask_channel": "coverage"}
    p = check_network(sin_borde)
    assert [x["code"] for x in p] == ["mask_needs_border"]
    assert p[0]["hint"]
    assert check_network({**BASE, "mask_channel": "coverage"}) == []


def test_es_un_eje_barrible_de_C():
    """La trampa que costo `dropout`: un campo que no esta en NETWORK_DEFAULTS
    hace que el barrido entrene N veces la MISMA red sin avisar."""
    from fv.models.builder import NETWORK_DEFAULTS
    assert "mask_channel" in NETWORK_DEFAULTS
    assert network_trace(full_config({**BASE, "mask_channel": "coverage"})
                         )["mask_channel"] == "coverage"


def test_el_modo_desconocido_se_nombra_en_fovea_tambien():
    with pytest.raises(FoveaError) as e:
        n_input_channels("mascara")
    assert e.value.code == "unknown_mask_channel"
    assert set(MASK_MODES) == {"off", "coverage"}


def test_la_sonda_se_niega_sin_cobertura():
    """Una sonda que rellenara la cobertura con ceros diria 'no hay relleno en
    ninguna celda' justo en las ventanas del borde, que son las que se miran."""
    from fv.inference.introspect import feature_maps_payload
    m = build_model(full_config({**BASE, "mask_channel": "coverage"}))
    d = m.dims
    v, cov = build_view(_imagen(), 0, 0, d, pad_mode="edge")
    with pytest.raises(FoveaError) as e:
        feature_maps_payload(m, v)
    assert e.value.code == "mask_channel_missing"
    assert feature_maps_payload(m, v, cov)["branches"]


def test_un_proceso_viejo_lo_dice_y_NO_manda_a_reentrenar(tmp_path):
    """Las dos averias tienen el mismo sintoma y arreglos opuestos.

    Paso el 2026-09-01: la web app llevaba viva desde antes de que existiera
    `mask_channel`, y el mensaje generico mandaba a REENTRENAR --gastar en Vast
    para arreglar un modelo que estaba perfecto-- cuando lo que hacia falta era
    reiniciar el servicio.
    """
    import torch
    from fv.inference.checkpoint import CheckpointError, load_model

    cfg = full_config(BASE)
    m = build_model(cfg)
    # un checkpoint escrito por un codigo MAS NUEVO: trae un campo que este
    # proceso no conoce
    p = tmp_path / "best.pt"
    torch.save({"config": {"model": {**cfg, "campo_del_futuro": "x"}},
                "model": m.state_dict()}, p)
    with pytest.raises(CheckpointError) as e:
        load_model(p)
    assert e.value.code == "checkpoint_de_codigo_mas_nuevo"
    assert "campo_del_futuro" in e.value.message
    assert "reinicia" in e.value.hint.lower()
    assert "NO reentrenes" in e.value.hint

    # y el checkpoint normal sigue cargando
    torch.save({"config": {"model": cfg}, "model": m.state_dict()}, p)
    assert load_model(p).cfg["mask_channel"] == "off"
