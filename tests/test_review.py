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

from fv.metrics import CORNER_NAMES

from tests.conftest import TINY_NET


@pytest.fixture()
def trained(world):
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    train("rev-run", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)
    # Desde el 2026-08-31 tener `best.pt` en disco NO basta para inferir: la red
    # tiene que estar APROBADA (fv.inference.catalogo). Aqui se aprueba a mano y
    # no por la antesala porque lo que este fichero prueba es la REVISION; la
    # politica de aprobacion tiene sus propios tests en test_inferencia.py.
    from fv.inference import catalogo
    from fv.ioutils import write_json_atomic
    write_json_atomic(catalogo.catalogo_path(),
                      {"version": 1, "runs": {"rev-run": {"motivo": "fixture"}}})
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
    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 2})
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

    c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 1})
    c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 1,
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

    a = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                      "count": 1, "offset": 0}).json()
    b = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                      "count": 1, "offset": 1}).json()
    vistos = review.reviewed_indices(a["window_dataset"], "val")
    assert vistos == sorted([a["images"][0]["index"], b["images"][0]["index"]])
    assert b["reviewed"] == 2
    assert b["pending"] == b["total"] - 2


def test_saltar_a_lo_no_revisado_avanza_y_no_da_vueltas(client):
    """`next_offset` tiene que llevar a algo NUEVO mientras quede algo nuevo."""
    c, w = client
    from fv import review

    ctx = c.get(f"/review/context?window_dataset={w['dataset']}&split=val&count=1").json()
    total = ctx["total"]
    assert total >= 2, "el mundo mini necesita al menos 2 imagenes en val"

    off = 0
    for _ in range(total):
        r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
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

    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                      "count": 2}).json()
    assert review.reviewed_indices(r["window_dataset"], "train") == []
    t = c.get(f"/review/context?window_dataset={w['dataset']}&split=train&count=2").json()
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
    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
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

    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
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
    r0 = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                       "count": 1}).json()
    idx = r0["images"][0]["index"]

    uno = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                        "indices": [idx]})
    assert uno.status_code == 200
    assert [i["index"] for i in uno.json()["images"]] == [idx]

    from fv.windows.store import WindowDatasetStore
    train_idx = WindowDatasetStore().split_map(r0["window_dataset"])["train"][0]
    malo = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                         "indices": [int(train_idx)]})
    assert malo.status_code == 400
    assert malo.json()["detail"]["code"] == "index_not_in_split"


def test_el_lote_esta_acotado_por_la_ruta(client):
    """Sin tope, un N grande desde el movil cuelga la peticion y parece que la
    pagina esta rota. El tope es de la ruta, no una convencion del cliente."""
    c, w = client
    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                      "count": 9999}).json()
    assert len(r["images"]) <= 60
    assert len(r["images"]) <= r["total"]


def test_dice_si_lo_que_guarda_acaba_en_el_repo_de_datos(client):
    """`data_root()` cae al repo de codigo cuando no hay repo de datos (R2). Ahi
    la revision se escribe igual y no la commitea nadie: revisar 200 imagenes y
    perderlas sin un solo error es justo el fallo silencioso que este proyecto
    rechaza. Por eso viaja en el payload."""
    c, w = client
    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                      "count": 1}).json()
    assert "storage" in r
    assert set(r["storage"]) == {"path", "in_data_repo"}
    assert isinstance(r["storage"]["in_data_repo"], bool)


def test_una_linea_rota_no_tumba_el_historial(client):
    """Regla de siempre aqui: se salta con aviso, no se cae. El JSONL lo escriben
    procesos que pueden morir a medias."""
    c, w = client
    from fv import review, settings

    c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 1})
    f = sorted(settings.reviews_root().glob("*.jsonl"))[0]
    with f.open("a", encoding="utf-8") as fh:
        fh.write("{esto no es json\n")
    assert len(review.reviews()) == 1


def test_marcar_sobrevive_y_se_ve_en_el_lote_siguiente(client):
    """La marca es del par (dataset, split, indice), no de la sesion: por eso se
    vuelve a ver al pasar por ese rango otra vez."""
    c, w = client
    r0 = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
                                       "count": 2}).json()
    idx = r0["images"][0]["index"]
    assert c.post("/review/marks", json={
        "window_dataset": r0["window_dataset"], "split": "val",
        "index": idx, "marked": True, "note": "caja partida"}).status_code == 200

    r1 = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val",
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

    c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 1})
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
    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "noexiste",
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

    peticion = {"window_dataset": w["dataset"], "run": w["run"],
                "split": "val", "count": 1, "offset": 0}
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
    c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 1, "offset": 0})
    c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 1, "offset": 1})
    c.post("/review/batch", json={"window_dataset": w["dataset"], "run": w["run"], "split": "val", "count": 1, "offset": 0})
    assert len(review.reviews()) == 3


# --- lo que manda es el DATASET, no el run ----------------------------------

def test_el_contexto_NO_devuelve_los_runs_de_la_maquina_sino_los_del_dataset(client):
    """El fallo que se vio en el server real: el select principal era el run y
    ahi son 859, con lo que la pantalla quedaba inservible. Ahora el servidor los
    filtra, y el front ni pide /runs."""
    c, w = client
    ctx = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    assert [r["name"] for r in ctx["runs"]] == [w["run"]]
    assert ctx["runs"][0]["has_checkpoint"] is True

    # un run de OTRO dataset no aparece aqui
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.windows.extract import ExtractConfig, extract_windows
    from fv import settings
    from tests.conftest import TINY_NET
    extract_windows(ExtractConfig(source=w["source"], window_size=8, stride=7,
                                  val_frac=0.2, test_frac=0.2, seed=2),
                    settings.window_datasets_root() / "otro-b8")
    train("otro-run", "otro-b8", "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=w["store"])
    ctx2 = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    assert "otro-run" not in [r["name"] for r in ctx2["runs"]]


def test_solo_se_ofrecen_datasets_CON_npz(client):
    """Un manifest sin `windows.npz` describe un dataset cuyo dato se perdio (16
    de 18 en el server real). Ofrecerlo es prometer imagenes que no existen."""
    c, w = client
    from fv import settings
    import json as _json
    vacio = settings.window_datasets_root() / "sin-dato"
    vacio.mkdir(parents=True)
    (vacio / "manifest.json").write_text(
        _json.dumps({"source_id": w["source"], "num_samples": 3,
                     "config": {"target_kinds": ["paragraph"]},
                     "windows_per_split": {"val": 1}}), encoding="utf-8")

    nombres = [d["name"] for d in c.get("/review/datasets").json()["datasets"]]
    assert w["dataset"] in nombres
    assert "sin-dato" not in nombres


def test_SIN_run_se_ven_las_imagenes_igual(client):
    """Lo que hace utilizable la pantalla en una maquina que solo tiene el repo
    de datos: los `*.pt` no viajan por git, asi que exigir un run seria no poder
    mirar nunca el dataset que SI viajo."""
    c, w = client
    r = c.post("/review/batch", json={"window_dataset": w["dataset"],
                                      "split": "val", "count": 2})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["inferred"] is False
    assert d["run"] is None
    assert len(d["images"]) == 2
    assert all(i["paragraphs"] == [] for i in d["images"])
    # ...y mirar sin modelo SIGUE siendo mirar
    from fv import review
    assert len(review.reviews()) == 1


def test_un_run_SIN_checkpoint_se_niega_diciendo_por_que(client):
    """No basta con no ofrecerlo: si llega igual, el error tiene que explicar POR
    QUE no hay pesos y que hacer, no un 500 opaco.

    ⚠ El mensaje cambio el 2026-08-31 y el motivo importa: antes decia que los
    pesos "no viajan por git", que era cierto hasta que se abrio la tercera
    excepcion del .gitignore. Ahora la razon verdadera es otra -- los pesos solo
    se guardan si la red se APRUEBA para inferencia-- y el hint dice que hay que
    reentrenar, que es lo unico que se puede hacer con un run que no se aprobo."""
    c, w = client
    (w["store"].path(w["run"]) / "best.pt").unlink()
    r = c.post("/review/batch", json={"window_dataset": w["dataset"],
                                      "split": "val", "count": 1,
                                      "run": w["run"]})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "run_has_no_checkpoint"
    assert "reentrenarlo" in r.json()["detail"]["hint"]
    ctx = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    assert ctx["runs"][0]["has_checkpoint"] is False


def test_sin_fuente_las_imagenes_salen_del_npz_y_se_DICE_que_no_hay_verdad(client):
    """El caso del server real: 0 fuentes. Las imagenes estan en el npz (que si
    esta commiteado) y la verdad no, porque vive en labels.jsonl de A. Dibujar un
    overlay vacio se leeria como "la red no se dejo nada"."""
    c, w = client
    from fv import settings
    import shutil
    shutil.rmtree(settings.local_sources_root())

    d = c.post("/review/batch", json={"window_dataset": w["dataset"],
                                      "split": "val", "count": 2}).json()
    assert d["truth_available"] is False
    assert all(i["truth"] == [] for i in d["images"])
    assert d["image_base"] == f"/api/window-datasets/{w['dataset']}/samples"
    # y la imagen se sirve de verdad, no es una promesa del payload
    img = c.get(f"{d['image_base'][4:]}/{d['images'][0]['index']}/image")
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"


def test_los_pixeles_del_npz_son_los_MISMOS_que_los_de_la_fuente(client):
    """Si no lo fueran, mirar sin fuente ensenaria otra cosa que mirar con ella,
    y nadie sabria cual de las dos es la buena."""
    c, w = client
    import io

    import numpy as np
    from PIL import Image

    from fv.datasets.loader import SourceDataset
    d = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    i = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    idx = 0
    con_fuente = np.asarray(SourceDataset(w["source"]).sample_at(idx).load_image())
    r = c.get(f"/window-datasets/{w['dataset']}/samples/{idx}/image")
    del_npz = np.asarray(Image.open(io.BytesIO(r.content)).convert("L"))
    assert np.array_equal(con_fuente, del_npz)
    assert d["image_base"].endswith("/samples") and i["image_size"]


def test_el_servidor_decide_de_donde_sale_el_PNG(client):
    """`image_base` viaja en el payload para que el front no tenga su propia
    copia de la regla "hay fuente o no": dos copias divergen."""
    c, w = client
    from fv import settings
    import shutil
    con = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    assert con["image_base"].startswith("/api/sources/")
    shutil.rmtree(settings.local_sources_root())
    sin = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    assert sin["image_base"].startswith("/api/window-datasets/")


def test_un_dataset_que_no_existe_cae_al_primero_valido(client):
    """El nombre recordado en el navegador puede haber desaparecido. Caer al
    primero es mejor que un 404 sobre una pantalla que si tiene algo que ensenar
    -- pero el que cae lo decide el SERVIDOR, y lo dice."""
    c, w = client
    ctx = c.get("/review/context?window_dataset=no-existe&split=val").json()
    assert ctx["window_dataset"] == w["dataset"]


def _extrae(w, nombre, window_size=8, stride=7, seed=3):
    from fv import settings
    from fv.windows.extract import ExtractConfig, extract_windows
    extract_windows(ExtractConfig(source=w["source"], window_size=window_size,
                                  stride=stride, val_frac=0.2, test_frac=0.2,
                                  seed=seed),
                    settings.window_datasets_root() / nombre)
    return nombre


def test_el_dataset_por_defecto_es_uno_QUE_PUEDA_inferir(client):
    """Sin `window_dataset` mandaba el primero de la lista, y eso abria la
    pantalla en el que no tiene con que inferir.

    Medido el 2026-08-30 en un dev recien hecho: el primero era
    `bench-dirty1000-16-r20260827`, con CERO runs, mientras el de al lado tenia
    un modelo con pesos. La pantalla abria diciendo "este dataset no tiene ningun
    run en esta maquina" -- que se lee como "aqui no hay nada que hacer".

    ⚠ El caso cambio el 2026-09-02: "no tener runs propios" ya NO significa "no
    se puede inferir", porque las redes se ofrecen por COMPATIBILIDAD y no por
    procedencia. Asi que el que hay que saltarse es el INCOMPATIBLE -- aqui uno
    con ventana de 16 px frente a una fovea de 8.
    """
    c, w = client
    _extrae(w, "aaa-b16", window_size=16)      # ordena ANTES, y NO encaja

    ctx = c.get("/review/context?split=val").json()
    assert ctx["window_dataset"] == w["dataset"]
    # ...y pedirlo explicitamente sigue mandando: esto elige un DEFECTO, no impone
    ctx2 = c.get("/review/context?window_dataset=aaa-b16&split=val").json()
    assert ctx2["window_dataset"] == "aaa-b16"
    assert ctx2["run_sugerido"] is None        # ninguna encaja: no se sugiere


def test_un_dataset_SIN_runs_propios_ofrece_igual_las_redes_que_pueden_inferir(client):
    """El fallo que reporto el dueno el 2026-09-02 probando los datasets de
    fallos: la pantalla decia «este dataset no tiene ningun run en esta maquina».

    Y no los tiene -- no los necesita. Una red infiere sobre cualquier dataset
    cuya geometria encaje, y de que dataset SALIO no dice nada sobre eso. Filtrar
    por procedencia dejaba sin modelo a todo dataset nuevo: un holdout, uno de
    fallos, cualquiera recien extraido.
    """
    c, w = client
    nuevo = _extrae(w, "sin-runs-b8")

    ctx = c.get(f"/review/context?window_dataset={nuevo}&split=val").json()
    fila = next(r for r in ctx["runs"] if r["name"] == w["run"])
    assert fila["has_checkpoint"] is True
    assert fila["compatible"] is True
    assert fila["propio"] is False                     # se ve que es de otro
    assert fila["window_dataset"] == w["dataset"]
    assert ctx["run_sugerido"] == w["run"]

    # ...y de verdad infiere: la costura entera, no solo la lista
    r = c.post("/review/batch", json={"window_dataset": nuevo, "run": w["run"],
                                      "split": "val", "count": 1})
    assert r.status_code == 200, r.text
    assert "paragraphs" in r.json()["images"][0]


def test_una_red_INCOMPATIBLE_se_marca_con_su_motivo_y_no_se_esconde(client):
    """Se MARCA, no se esconde -- la misma regla que un run sin pesos. Una red
    escondida no se distingue de una que no existe, y entonces no sabes si el
    problema es la geometria o que falta traer algo.

    Y el motivo sale de `check_compatible`, que es LA definicion del contrato ①:
    escribir aqui una segunda seria tener dos que se pueden desincronizar.
    """
    c, w = client
    otro = _extrae(w, "otra-ventana-b16", window_size=16)

    ctx = c.get(f"/review/context?window_dataset={otro}&split=val").json()
    fila = next(r for r in ctx["runs"] if r["name"] == w["run"])
    assert fila["has_checkpoint"] is True              # pesos, si
    assert fila["compatible"] is False                 # encaje, no
    assert fila["problems"][0]["code"] == "window_size_mismatch"
    assert "16" in fila["problems"][0]["message"]
    assert fila["problems"][0]["hint"]


def test_forzar_una_pareja_incompatible_DA_EL_ERROR_en_vez_de_cero_cajas(client):
    """Sin esto la respuesta seria una rejilla de imagenes sin una sola caja --
    indistinguible de «la red no detecto nada», que es el fallo silencioso.

    Se falla ANTES de inferir (R2), y con el codigo del validador.
    """
    c, w = client
    otro = _extrae(w, "forzar-b16", window_size=16)

    r = c.post("/review/batch", json={"window_dataset": otro, "run": w["run"],
                                      "split": "val", "count": 1})
    assert r.status_code >= 400, r.text
    cuerpo = r.json()["detail"] if "detail" in r.json() else r.json()
    assert cuerpo["code"] == "window_size_mismatch"
    assert w["run"] in cuerpo["message"] and otro in cuerpo["message"]

    # ...y sin red se sigue pudiendo MIRAR: la incompatibilidad es de la pareja
    ok = c.post("/review/batch", json={"window_dataset": otro, "split": "val",
                                       "count": 1})
    assert ok.status_code == 200, ok.text


def test_sin_ningun_checkpoint_el_defecto_es_el_de_siempre(client):
    """El otro lado de la regla: si NINGUNO puede inferir, no hay nada que
    preferir y se cae al primero. Una pantalla que se quedara en blanco por no
    encontrar su favorito seria peor que la de antes."""
    c, w = client
    from fv.windows.extract import ExtractConfig, extract_windows
    from fv import settings
    extract_windows(ExtractConfig(source=w["source"], window_size=8, stride=7,
                                  val_frac=0.2, test_frac=0.2, seed=3),
                    settings.window_datasets_root() / "aaa-b8")
    (w["store"].path(w["run"]) / "best.pt").unlink()

    ctx = c.get("/review/context?split=val").json()
    assert ctx["window_dataset"] == "aaa-b8"
    assert ctx["run_sugerido"] is None


def test_el_servidor_sugiere_QUE_run_abrir(client):
    """Cual abrir lo decide el servidor, que es quien ya sabe cual tiene pesos.

    Si lo decidiera el front habria dos copias de la regla; y sin sugerencia, una
    maquina recien lanzada abre pidiendo que adivines cual de sus 10 runs trajo
    checkpoint -- que es justo lo que no se puede saber mirando la lista."""
    c, w = client
    ctx = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    assert ctx["run_sugerido"] == w["run"]

    # y desaparece cuando el run deja de poder inferir: sugerir uno que da 409
    # seria peor que no sugerir ninguno
    (w["store"].path(w["run"]) / "best.pt").unlink()
    ctx2 = c.get(f"/review/context?window_dataset={w['dataset']}&split=val").json()
    assert ctx2["run_sugerido"] is None


def test_los_mandos_de_inferencia_LLEGAN_y_vuelven_dichos(client):
    """Los knobs (F) son lo unico ajustable sin reentrenar, y la pantalla los
    ofrece: si no llegasen, mover el deslizador no haria nada y las cajas
    seguirian siendo las mismas -- que se lee como "el modelo no reacciona".

    Y vuelven ECHADOS en la respuesta porque tres de los cuatro tienen un `auto`
    derivado del tamano de ventana de la red (`predict.py`): sin verlos, no hay
    forma de saber en que numero se convirtio cada uno, y la UI tendria que
    guardar su propia copia de esa regla.
    """
    c, w = client
    pedido = {"window_dataset": w["dataset"], "run": w["run"], "split": "val",
              "count": 1, "threshold": 0.42, "stride": 3,
              "nms_radius": 2.5, "min_size": 7.0}
    k = c.post("/review/batch", json=pedido).json()["knobs"]
    assert k["threshold"] == 0.42
    assert k["stride"] == 3
    assert k["nms_radius"] == 2.5
    assert k["min_size"] == 7.0

    # ...y lo que NO se manda lo deriva el servidor de la ventana de la RED, que
    # es justo por lo que la UI manda "auto" callando en vez de mandar un 0
    solo = c.post("/review/batch", json={"window_dataset": w["dataset"],
                                         "run": w["run"], "split": "val",
                                         "count": 1}).json()["knobs"]
    n = solo["window_size"]
    assert solo["stride"] == max(1, n // 2)
    assert solo["nms_radius"] == n / 2
    assert solo["min_size"] == 4.0
    assert solo["threshold"] == 0.5


def test_subir_el_umbral_no_puede_anadir_cajas(client):
    """La direccion del mando, que es lo que un cableado al reves invierte sin
    fallar: mas exigente nunca detecta MAS. Se fija como desigualdad y no como
    numero porque el modelo del test entrena una epoca."""
    c, w = client

    def cajas(th):
        r = c.post("/review/batch", json={"window_dataset": w["dataset"],
                                          "run": w["run"], "split": "val",
                                          "count": 1, "threshold": th})
        return len(r.json()["images"][0]["paragraphs"])

    assert cajas(0.99) <= cajas(0.05)


def test_las_detecciones_van_a_peticion_y_no_por_defecto(client):
    """El detalle pide los puntos; la rejilla NO los recibe.

    Es una decision de tamano: la rejilla trae hasta REVIEW_MAX imagenes y la
    etapa cruda son decenas de puntos por imagen. Que se declare en la peticion
    (y no se deduzca de `len(indices) == 1`) es lo que impide que la rejilla
    cambie de payload el dia que pida una sola imagen."""
    c, w = client
    cuerpo = {"window_dataset": w["dataset"], "run": w["run"], "split": "val",
              "count": 1, "threshold": 0.05}

    callado = c.post("/review/batch", json=cuerpo).json()
    assert "corners" not in callado["images"][0]
    assert "raw" not in callado["images"][0]
    # ...pero las cajas siguen estando: lo que cambia es el detalle, no la caja
    assert "paragraphs" in callado["images"][0]

    # `with_detections` son las ESQUINAS, que son baratas (~10 por imagen) y las
    # pide tambien la rejilla para las miniaturas
    pedido = c.post("/review/batch", json={**cuerpo, "with_detections": True}).json()
    img = pedido["images"][0]
    assert "corners" in img
    assert "raw" not in img, "la nube cruda es ~3x las esquinas: va aparte"

    # ...y la nube PRE-NMS es el segundo nivel, solo para el detalle
    con_crudas = c.post("/review/batch",
                        json={**cuerpo, "with_detections": True,
                              "with_raw": True}).json()["images"][0]
    assert "corners" in con_crudas and "raw" in con_crudas
    # el vocabulario viaja CON la respuesta indexada por el (U4.2): sin esto el
    # front tendria que guardar su copia del orden de esquinas
    assert pedido["corner_order"] == list(CORNER_NAMES)
    assert callado["corner_order"] == list(CORNER_NAMES)
    # cada punto trae lo que se dibuja: su ranura, su score y donde cae
    for d in con_crudas["corners"] + con_crudas["raw"]:
        assert d["corner"] in pedido["corner_order"]
        assert 0.0 <= d["score"] <= 1.0
        assert (0 <= d["x"] <= con_crudas["width"]
                and 0 <= d["y"] <= con_crudas["height"])
    # el NMS es lo unico que separa las dos etapas: nunca puede haber MAS
    # esquinas que ventanas crudas de las que salieron
    assert len(con_crudas["corners"]) <= len(con_crudas["raw"])


def test_sin_modelo_no_hay_ranuras_de_que_hablar(client):
    """`corner_order` a None y no lista vacia: ausente no es cero (formatos 1).

    La pantalla lo usa para decidir si ensena el filtro de esquinas, y una lista
    vacia diria 'este dataset no tiene esquinas' en vez de 'no se infirio'."""
    c, w = client
    r = c.post("/review/batch", json={"window_dataset": w["dataset"], "split": "val",
                                      "count": 1, "with_detections": True}).json()
    assert r["inferred"] is False
    assert r["corner_order"] is None
    assert r["images"][0]["paragraphs"] == []
    assert "corners" not in r["images"][0]


def test_una_red_que_no_guarda_su_valor_se_dice_en_vez_de_darse_por_buena(client):
    """La compatibilidad se comprueba con la red POR VALOR (`config.json`), no
    cargando el `.pt`. Un run viejo que no la guarda no se puede juzgar -- y
    entonces se dice, en vez de colarlo como compatible.

    Es la regla de siempre aqui: entre un fallo ruidoso y uno silencioso, el
    ruidoso. Darlo por bueno acabaria en una rejilla sin cajas y sin motivo.
    """
    c, w = client
    from fv.ioutils import write_json_atomic

    otro = _extrae(w, "sin-valor-b8")
    cfg = w["store"].config(w["run"])
    cfg["provenance"]["network"].pop("value", None)
    cfg.pop("network", None)
    write_json_atomic(w["store"].path(w["run"]) / "config.json", cfg)

    ctx = c.get(f"/review/context?window_dataset={otro}&split=val").json()
    fila = next(r for r in ctx["runs"] if r["name"] == w["run"])
    assert fila["compatible"] is None                  # no se sabe, no "si"
    assert fila["problems"][0]["code"] == "network_value_missing"
    assert ctx["run_sugerido"] is None                 # no se sugiere lo dudoso

    # y forzarlo tampoco infiere a ciegas
    r = c.post("/review/batch", json={"window_dataset": otro, "run": w["run"],
                                      "split": "val", "count": 1})
    assert r.status_code >= 400
    assert r.json()["detail"]["code"] == "network_value_missing"
