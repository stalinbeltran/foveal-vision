"""Publicar una fuente en el repo de datos, para que viaje por git.

Lo que se fija aqui es sobre todo el GUARD: esta medido que re-renderizar no da
el mismo dato, asi que publicar una fuente que no cuadra con el `windows.npz`
commiteado haria revisar imagenes que el modelo nunca vio -- y sin un solo error
por el camino.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture()
def publicable(world):
    """`world` ya deja el repo de datos SEPARADO del de codigo -- `FV_ROOT` va a
    tmp_path y el autouse de conftest manda `FV_DATA_ROOT` a `tmp_path/_repo-datos`
    --, que es exactamente la condicion para que publicar signifique algo. Montar
    otra raiz aqui solo servia para dejar el dataset donde no estaba."""
    from fv import settings
    assert settings.published_sources_root() != settings.local_sources_root()
    return world


def test_una_maquina_sin_fuentes_LAS_VE_si_estan_publicadas(publicable):
    """El motivo entero de esto: `/data/sources/` esta en el .gitignore, asi que
    una maquina recien hecha tiene CERO fuentes (medido el 2026-08-29)."""
    from fv import settings
    from fv.datasets.loader import discover_sources
    from fv.datasets.publish import publish_source
    import shutil

    r = publish_source(publicable["source"])
    assert r["images"] == 10
    # ...y ahora se borra la fuente de la MAQUINA, como si fuera un clon limpio
    shutil.rmtree(settings.local_sources_root())
    ids = [s["id"] for s in discover_sources()]
    assert publicable["source"] in ids
    assert [s for s in discover_sources() if s["id"] == publicable["source"]][0]["published"]


def test_la_fuente_publicada_SIRVE_para_inferir_y_para_la_verdad(publicable):
    """Que aparezca en la lista no basta: tiene que abrir imagen y bloques."""
    from fv import settings
    from fv.datasets.loader import SourceDataset
    from fv.datasets.publish import publish_source
    import shutil

    antes = np.asarray(SourceDataset(publicable["source"]).sample_at(0).load_image())
    publish_source(publicable["source"])
    shutil.rmtree(settings.local_sources_root())

    ds = SourceDataset(publicable["source"])
    s = ds.sample_at(0)
    assert np.array_equal(np.asarray(s.load_image()), antes)
    assert s.blocks and s.blocks[0].kind == "paragraph"


def test_ABORTA_si_la_fuente_no_es_la_que_guarda_el_npz(publicable):
    """El fallo caro, y el unico que este modulo existe para impedir.

    Se simula un re-render: se cambia un pixel de una imagen de la fuente. El
    `windows.npz` commiteado sigue teniendo la vieja, asi que publicar esa fuente
    dejaria la revision mirando imagenes que el modelo nunca vio."""
    from fv.datasets.loader import resolve_source
    from fv.datasets.publish import PublishError, publish_source
    from PIL import Image

    root = resolve_source(publicable["source"])
    rec = json.loads((root / "labels.jsonl").read_text(encoding="utf-8").splitlines()[0])
    p = root / rec["image"]
    a = np.asarray(Image.open(p).convert("L")).copy()
    a[0, 0] = 255 - a[0, 0]
    Image.fromarray(a).save(p)

    with pytest.raises(PublishError) as e:
        publish_source(publicable["source"])
    assert e.value.code == "source_does_not_match_windows"
    assert e.value.hint


def test_force_publica_igual_pero_hay_que_pedirlo(publicable):
    """Negarse siempre seria un callejon sin salida cuando de verdad quieres la
    fuente nueva. Lo que no puede es pasar sin que nadie lo pida."""
    from fv.datasets.loader import resolve_source
    from fv.datasets.publish import publish_source
    from PIL import Image

    root = resolve_source(publicable["source"])
    rec = json.loads((root / "labels.jsonl").read_text(encoding="utf-8").splitlines()[0])
    a = np.asarray(Image.open(root / rec["image"]).convert("L")).copy()
    a[0, 0] = 255 - a[0, 0]
    Image.fromarray(a).save(root / rec["image"])
    r = publish_source(publicable["source"], force=True)
    assert r["images"] == 10


def test_no_se_publica_lo_GRANDE_porque_de_git_no_se_quita(publicable, monkeypatch):
    """234 MB de renders regenerables en el historial de git son para siempre."""
    from fv.datasets.publish import MAX_PUBLICABLE, PublishError, publish_source
    monkeypatch.setattr("fv.datasets.publish.MAX_PUBLICABLE", 1000)
    with pytest.raises(PublishError) as e:
        publish_source(publicable["source"])
    assert e.value.code == "source_too_big"
    assert "0 MB" not in str(e.value), "un tope que dice 0 MB no informa de nada"
    assert "REDUCIDA" in e.value.hint


def test_sin_repo_de_datos_se_NIEGA_en_vez_de_copiarse_encima(world):
    """`data_root()` cae al repo de codigo cuando no hay hermano (R2). Ahi las
    dos raices son la misma y publicar seria copiar una fuente sobre si misma."""
    from fv.datasets.publish import PublishError, publish_source
    import os
    os.environ.pop("FV_DATA_ROOT", None)
    from fv import settings
    if settings.published_sources_root().resolve() != settings.local_sources_root().resolve():
        pytest.skip("hay repo de datos: este caso no aplica")
    with pytest.raises(PublishError) as e:
        publish_source(world["source"])
    assert e.value.code == "no_data_repo"


def test_la_verificacion_distingue_NO_PUDE_MIRAR_de_CUADRA(publicable):
    """Un dataset sin `windows.npz` no comprueba nada, y decir que cuadra seria
    el falso verde. Mismo criterio que el `NO SE` de cerrable.mjs."""
    from fv import settings
    from fv.datasets.publish import verify_against_windows

    v = verify_against_windows(publicable["source"])
    assert v["concluyente"] and v["comprobados"]

    (settings.window_datasets_root() / publicable["dataset"] / "windows.npz").unlink()
    v2 = verify_against_windows(publicable["source"])
    assert not v2["concluyente"]
    assert v2["sin_npz"] == [publicable["dataset"]]
    assert not v2["discrepan"]


def test_la_fuente_de_la_maquina_gana_a_la_publicada(publicable):
    """Prioridad declarada: quien tiene la fuente de verdad trabaja sobre ella;
    el repo de datos es para la maquina que no la tiene."""
    from fv.datasets.loader import _roots, resolve_source
    from fv.datasets.publish import publish_source
    from fv import settings

    publish_source(publicable["source"])
    assert len(_roots()) == 2
    assert resolve_source(publicable["source"]).resolve() == (
        settings.local_sources_root() / "mini").resolve()


def test_no_se_lista_dos_veces_estando_en_las_dos_raices(publicable):
    """Un id, no dos: si listase dos y resolviese una, el selector ofreceria una
    fila que abre la otra."""
    from fv.datasets.loader import discover_sources
    from fv.datasets.publish import publish_source
    publish_source(publicable["source"])
    ids = [s["id"] for s in discover_sources()]
    assert ids.count(publicable["source"]) == 1
