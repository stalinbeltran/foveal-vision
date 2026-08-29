"""La revision a ojo (F x B): que rangos se miraron, y que quedo marcado.

Cada test construye su mundo (tests.md). Lo que se fija aqui no son numeros --
son las COSTURAS: que mirar deje rastro sin pulsar nada, que el historial solo
crezca, que las marcas se reescriban, y que las cajas que se pintan sean las
mismas que puntua la metrica de tarea.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TINY_NET


@pytest.fixture()
def trained(world):
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    train("rev-run", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)
    return {"run": "rev-run", "store": store, **world}


@pytest.fixture()
def client(trained):
    from fv.api.app import create_app
    return TestClient(create_app(), raise_server_exceptions=False), trained


def test_mirar_deja_rastro_sin_pulsar_nada(client):
    """La costura central: el registro lo escribe el que INFIERE.

    Si hiciera falta un boton de guardar, la mirada que no se guarda es
    indistinguible de la que no ocurrio -- y entonces "que he revisado ya" miente
    justo cuando importa."""
    c, w = client
    from fv import review

    assert review.reviews() == []
    r = c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 2})
    assert r.status_code == 200, r.text
    hist = review.reviews()
    assert len(hist) == 1
    assert hist[0]["split"] == "val"
    assert hist[0]["count"] == len(r.json()["images"])
    # la fuente viaja en la linea: mirar un holdout desde aqui deja rastro
    assert hist[0]["source"] == w["source"]


def test_el_historial_solo_se_añade_y_las_marcas_se_reescriben(client):
    """R8, comprobada en disco: dos ficheros con reglas de edicion opuestas.

    Juntarlos obligaria a leer todo el historial y ordenar por fecha para saber
    que hay marcado hoy."""
    c, w = client
    from fv import review, settings

    c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 1})
    c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 1,
                                  "offset": 1})
    assert len(review.reviews()) == 2          # historial: solo crece

    ds = review.reviews()[0]["window_dataset"]
    c.post("/review/marks", json={"window_dataset": ds, "split": "val",
                                  "index": 0, "marked": True})
    c.post("/review/marks", json={"window_dataset": ds, "split": "val",
                                  "index": 0, "marked": False})
    assert review.mark_list() == []            # estado: se reescribe
    assert len(review.reviews()) == 2          # ...y no toca el historial

    jsonl = sorted((settings.reviews_root()).glob("*.jsonl"))
    assert jsonl, "el historial va en JSONL, uno por mes"
    assert len(jsonl[0].read_text(encoding="utf-8").strip().splitlines()) == 2


def test_lo_ya_revisado_se_acumula_entre_sesiones(client):
    """El valor entero de esto es acumulativo: 'estas ya las vi, ensename otras'."""
    c, w = client
    from fv import review

    a = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                      "count": 1, "offset": 0}).json()
    b = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                      "count": 1, "offset": 1}).json()
    vistos = review.reviewed_indices(a["window_dataset"], "val")
    assert vistos == sorted([a["images"][0]["index"], b["images"][0]["index"]])
    assert b["reviewed"] == 2
    assert b["pending"] == b["total"] - 2


def test_saltar_a_lo_no_revisado_avanza_y_no_da_vueltas(client):
    """`next_offset` tiene que llevar a algo NUEVO mientras quede algo nuevo."""
    c, w = client
    from fv import review

    ctx = c.get(f"/review/context?run={w['run']}&split=val&count=1").json()
    total = ctx["total"]
    assert total >= 2, "el mundo mini necesita al menos 2 imagenes en val"

    off = 0
    for _ in range(total):
        r = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                          "count": 1, "offset": off}).json()
        off = r["next_offset"]
    assert r["pending"] == 0
    # agotado, `next_offset` cae a 0 -- y lo que distingue "no queda nada" de
    # "vuelve a empezar" es `pending`, no el 0
    assert review.next_unreviewed_offset(list(range(total)),
                                         list(range(total)), 1) == 0


def test_los_splits_no_se_contaminan(client):
    """Revisar val no puede marcar como vistas las de train."""
    c, w = client
    from fv import review

    r = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                      "count": 2}).json()
    assert review.reviewed_indices(r["window_dataset"], "train") == []
    t = c.get(f"/review/context?run={w['run']}&split=train&count=2").json()
    assert t["reviewed"] == 0


def test_las_cajas_son_las_MISMAS_que_puntua_la_metrica_de_tarea(client):
    """La costura que hace util la pantalla: si dibujara otras cajas, mirar no
    diagnosticaria el numero que se publica."""
    c, w = client
    from fv.inference.checkpoint import load_model
    from fv.inference.predict import predict_image
    from fv.datasets.loader import SourceDataset

    # ⚠ El umbral es 0.1 A PROPOSITO. Con el 0.5 por defecto esta red de juguete
    # (1 epoca) no detecta NADA, y entonces esta prueba comparaba [] con [] y
    # habria pasado igual con el endpoint devolviendo cajas inventadas. Un test
    # que no puede fallar no es un test. La asercion de abajo lo fija.
    r = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                      "count": 2, "threshold": 0.1}).json()
    assert sum(len(i["paragraphs"]) for i in r["images"]) > 0, \
        "sin ninguna caja esta comparacion no comprueba nada"
    model = load_model(w["store"].path(w["run"]) / "best.pt")
    source = SourceDataset(w["source"])
    for img in r["images"]:
        out = predict_image(model, source.sample_at(img["index"]).load_image(),
                            threshold=0.1)
        assert img["paragraphs"] == out["paragraphs"]


def test_la_verdad_se_filtra_por_kind_como_en_task_score(client):
    """Un dataset extraido de parrafos no se compara contra lineas. El filtro es
    el mismo que usa `task_score`, y por eso la verdad dibujada es la que cuenta."""
    c, w = client
    from fv.datasets.loader import SourceDataset
    from fv.windows.store import WindowDatasetStore

    r = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                      "count": 2}).json()
    kinds = set(WindowDatasetStore().manifest(r["window_dataset"])["config"]["target_kinds"])
    source = SourceDataset(w["source"])
    assert sum(len(i["truth"]) for i in r["images"]) > 0, \
        "sin ninguna verdad esta comparacion no comprueba nada"
    for img in r["images"]:
        esperados = [b for b in source.sample_at(img["index"]).blocks
                     if b.kind in kinds]
        assert len(img["truth"]) == len(esperados)


def test_indices_explicitos_dan_UNA_imagen_y_rechazan_lo_de_otro_split(client):
    """Es como pide la pagina de detalle. Un indice que no esta en el split es un
    400 con su motivo, no una imagen de otro sitio."""
    c, w = client
    r0 = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                       "count": 1}).json()
    idx = r0["images"][0]["index"]

    uno = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                        "indices": [idx]})
    assert uno.status_code == 200
    assert [i["index"] for i in uno.json()["images"]] == [idx]

    from fv.windows.store import WindowDatasetStore
    train_idx = WindowDatasetStore().split_map(r0["window_dataset"])["train"][0]
    malo = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                         "indices": [int(train_idx)]})
    assert malo.status_code == 400
    assert malo.json()["detail"]["code"] == "index_not_in_split"


def test_el_lote_esta_acotado_por_la_ruta(client):
    """Sin tope, un N grande desde el movil cuelga la peticion y parece que la
    pagina esta rota. El tope es de la ruta, no una convencion del cliente."""
    c, w = client
    r = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                      "count": 9999}).json()
    assert len(r["images"]) <= 60
    assert len(r["images"]) <= r["total"]


def test_dice_si_lo_que_guarda_acaba_en_el_repo_de_datos(client):
    """`data_root()` cae al repo de codigo cuando no hay repo de datos (R2). Ahi
    la revision se escribe igual y no la commitea nadie: revisar 200 imagenes y
    perderlas sin un solo error es justo el fallo silencioso que este proyecto
    rechaza. Por eso viaja en el payload."""
    c, w = client
    r = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                      "count": 1}).json()
    assert "storage" in r
    assert set(r["storage"]) == {"path", "in_data_repo"}
    assert isinstance(r["storage"]["in_data_repo"], bool)


def test_una_linea_rota_no_tumba_el_historial(client):
    """Regla de siempre aqui: se salta con aviso, no se cae. El JSONL lo escriben
    procesos que pueden morir a medias."""
    c, w = client
    from fv import review, settings

    c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 1})
    f = sorted(settings.reviews_root().glob("*.jsonl"))[0]
    with f.open("a", encoding="utf-8") as fh:
        fh.write("{esto no es json\n")
    assert len(review.reviews()) == 1


def test_marcar_sobrevive_y_se_ve_en_el_lote_siguiente(client):
    """La marca es del par (dataset, split, indice), no de la sesion: por eso se
    vuelve a ver al pasar por ese rango otra vez."""
    c, w = client
    r0 = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                       "count": 2}).json()
    idx = r0["images"][0]["index"]
    assert c.post("/review/marks", json={
        "window_dataset": r0["window_dataset"], "split": "val",
        "index": idx, "marked": True, "note": "caja partida"}).status_code == 200

    r1 = c.post("/review/batch", json={"run": w["run"], "split": "val",
                                       "count": 2}).json()
    marcadas = {i["index"]: i["marked"] for i in r1["images"]}
    assert marcadas[idx] is True
    assert all(v is False for k, v in marcadas.items() if k != idx)

    lista = c.get("/review/marks").json()["marks"]
    assert len(lista) == 1 and lista[0]["note"] == "caja partida"


def test_el_historial_se_filtra_por_dias(client):
    """Es lo que pinta 'ayer' y 'la semana pasada' sin traerse el historial entero."""
    c, w = client
    from datetime import datetime, timedelta, timezone
    from fv import review

    c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 1})
    review.record_review(window_dataset="mini-b8", split="val", source="x",
                         run=w["run"], indices=[0], offset=0,
                         when=datetime.now(timezone.utc) - timedelta(days=30))
    assert len(review.reviews()) == 2
    assert len(review.reviews(since_days=7)) == 1


def test_un_split_vacio_se_niega_con_su_motivo(client):
    """R2: o degrada con un defecto declarado, o falla ANTES de empezar.

    El codigo es el MISMO que levanta `task_score` ante el mismo problema, y por
    tanto el mismo 409 de `CONFLICT_CODES`: dos endpoints que contestan distinto
    a la misma causa son dos vocabularios que el cliente tiene que aprender."""
    c, w = client
    r = c.post("/review/batch", json={"run": w["run"], "split": "noexiste",
                                      "count": 2})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "split_empty"
    assert r.json()["detail"]["hint"]


def test_marcar_ANTES_de_haber_revisado_nunca(client):
    """El unico orden en que /review/marks se llama primero -- y el que no cubria
    el camino feliz. `write_json_atomic` no crea padres, asi que el mkdir tiene
    que ir antes de escribir: reventaba con FileNotFoundError sobre el .tmp."""
    c, w = client
    from fv import settings
    assert not settings.reviews_root().exists()
    r = c.post("/review/marks", json={"window_dataset": "demo", "split": "val",
                                      "index": 3, "marked": True})
    assert r.status_code == 200, r.text
    assert r.json()["marks"] == 1
    assert c.get("/review/marks").json()["marks"][0]["index"] == 3


def test_mirar_dos_veces_el_mismo_rango_no_duplica_la_linea(client):
    """Este fichero se COMMITEA: un remontaje del componente o un doble toque no
    puede dejar una linea cada vez. Un repaso de verdad, en cambio, si."""
    c, w = client
    from datetime import datetime, timedelta, timezone
    from fv import review

    peticion = {"run": w["run"], "split": "val", "count": 1, "offset": 0}
    for _ in range(4):
        c.post("/review/batch", json=peticion)
    assert len(review.reviews()) == 1

    ds = review.reviews()[0]["window_dataset"]
    # ...y volver mañana al mismo rango SI es una revision nueva
    review.record_review(window_dataset=ds, split="val", source=w["source"],
                         run=w["run"], indices=[0], offset=0,
                         when=datetime.now(timezone.utc) + timedelta(hours=20))
    assert len(review.reviews()) == 2


def test_cambiar_de_rango_siempre_deja_linea(client):
    """El dedupe mira (dataset, split, run, offset, count): mover el rango es
    otra mirada, y perderla seria perder justo lo que hace util el fichero."""
    c, w = client
    from fv import review
    c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 1, "offset": 0})
    c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 1, "offset": 1})
    c.post("/review/batch", json={"run": w["run"], "split": "val", "count": 1, "offset": 0})
    assert len(review.reviews()) == 3
