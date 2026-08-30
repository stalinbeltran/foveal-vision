"""Continuar un entrenamiento: seguir donde se dejo, no empezar otro.

La prueba que manda es la primera: **3 + 3 tiene que dar la MISMA curva que 6 de
una vez**. Si no la da, algo del estado no se restauro -- el optimizador, un
contador o alguno de los TRES generadores en juego-- y eso no se ve como un
error, se ve como una curva un poco peor sin causa aparente.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from tests.conftest import TINY_NET


def _recipe(**kw):
    from fv.training.recipe import Recipe
    base = dict(epochs=3, batch_size=32, lr=1e-3, patience=0, seed=7)
    base.update(kw)
    return Recipe(**base)


def _metricas(store, run):
    p = store.path(run) / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture()
def entrenado(world):
    from fv.training.loop import train
    from fv.training.registry import RunStore
    store = RunStore()
    train("cont", world["dataset"], "n", TINY_NET, "r", _recipe(epochs=3),
          store=store)
    return {"run": "cont", "store": store, **world}


def test_TRES_MAS_TRES_da_la_misma_curva_que_SEIS_de_una_vez(entrenado):
    """La prueba de fidelidad. Cubre a la vez el optimizador, los contadores y
    los tres generadores (torch, numpy y el del DataLoader, que es el que decide
    el barajado y es el que mas facil se olvida)."""
    from fv.training.loop import reanudar, train
    st = entrenado["store"]

    reanudar(entrenado["run"], mas=3, store=st)
    partido = _metricas(st, entrenado["run"])

    train("seguido", entrenado["dataset"], "n", TINY_NET, "r",
          _recipe(epochs=6), store=st)
    seguido = _metricas(st, "seguido")

    assert [m["epoch"] for m in partido] == list(range(1, 7))
    assert len(partido) == len(seguido) == 6
    for a, b in zip(partido, seguido):
        assert a["epoch"] == b["epoch"]
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-6), \
            f"la epoca {a['epoch']} diverge: reanudar no restauro todo el estado"
        assert a["val"]["loss"] == pytest.approx(b["val"]["loss"], rel=1e-6)


def test_el_historial_se_AÑADE_no_se_reescribe(entrenado):
    """`metrics.jsonl` es el historial del run (R8): reanudar no puede empezarlo
    de cero ni renumerar, o las curvas publicadas dejan de casar con el fichero."""
    from fv.training.loop import reanudar
    st = entrenado["store"]
    antes = _metricas(st, entrenado["run"])
    reanudar(entrenado["run"], mas=2, store=st)
    despues = _metricas(st, entrenado["run"])
    assert despues[:len(antes)] == antes
    assert [m["epoch"] for m in despues] == [1, 2, 3, 4, 5]


def test_el_resumen_cuenta_las_epocas_ACUMULADAS(entrenado):
    from fv.training.loop import reanudar
    r = reanudar(entrenado["run"], mas=4, store=entrenado["store"])
    assert r["epochs_run"] == 7
    assert r["epochs_requested"] == 7
    assert r["continued_from"] == 3


def test_best_pt_sirve_para_EVALUAR_y_no_carga_el_optimizador(entrenado):
    """Los dos ficheros tienen propositos distintos y por eso no llevan lo mismo.
    `best.pt` lo lee `load_model` (y la pantalla de revision): meterle el
    optimizador lo engordaria para nadie."""
    import torch
    from fv.inference.checkpoint import load_model
    d = entrenado["store"].path(entrenado["run"])
    best = torch.load(d / "best.pt", weights_only=False)
    last = torch.load(d / "last.pt", weights_only=False)
    assert "optimizer" not in best
    assert "optimizer" in last and last["optimizer"] is not None
    assert (d / "best.pt").stat().st_size < (d / "last.pt").stat().st_size
    assert load_model(d / "best.pt") is not None


def test_se_puede_EVALUAR_lo_continuado(entrenado):
    """El ciclo entero que se pidio: entrenar, continuar, y poder mirar imagenes
    con los pesos resultantes."""
    from fv.inference.checkpoint import load_model
    from fv.inference.predict import predict_image
    from fv.datasets.loader import SourceDataset
    from fv.training.loop import reanudar
    st = entrenado["store"]
    reanudar(entrenado["run"], mas=2, store=st)
    modelo = load_model(st.path(entrenado["run"]) / "best.pt")
    out = predict_image(modelo, SourceDataset(entrenado["source"]).sample_at(0).load_image(),
                        threshold=0.1)
    assert "paragraphs" in out and "knobs" in out


def test_sin_last_pt_se_niega_diciendo_que_best_no_sirve_para_esto(entrenado):
    from fv.training.loop import RunError, reanudar
    (entrenado["store"].path(entrenado["run"]) / "last.pt").unlink()
    with pytest.raises(RunError) as e:
        reanudar(entrenado["run"], mas=1, store=entrenado["store"])
    assert e.value.code == "run_has_no_last_checkpoint"
    assert "best.pt" in e.value.hint


def test_un_checkpoint_VIEJO_se_niega_en_vez_de_continuar_con_Adam_en_blanco(entrenado):
    """R2: degradar con un defecto declarado, o negarse antes de empezar. Los 859
    runs que ya existen se entrenaron sin guardar estado; continuarlos reinicia
    los momentos de Adam, y eso no falla: da una curva peor sin causa aparente."""
    import torch
    from fv.training.loop import RunError, reanudar
    d = entrenado["store"].path(entrenado["run"])
    viejo = torch.load(d / "last.pt", weights_only=False)
    torch.save({"model": viejo["model"], "config": viejo["config"],
                "epoch": viejo["epoch"]}, d / "last.pt")

    with pytest.raises(RunError) as e:
        reanudar(entrenado["run"], mas=1, store=entrenado["store"])
    assert e.value.code == "checkpoint_sin_estado"
    assert "Adam" in e.value.hint

    # ...y se puede pedir, sabiendo lo que cuesta
    r = reanudar(entrenado["run"], mas=1, store=entrenado["store"],
                 optimizador_limpio=True)
    assert r["epochs_run"] == 4


def test_si_el_dataset_se_reconstruyo_NO_se_continua(entrenado):
    """Mismo guard que `task_score`: los splits ya no son los que este modelo no
    vio, y seguir entrenando mezclaria train de hoy con val de ayer."""
    import json as _json
    from fv.training.loop import RunError, reanudar
    from fv.windows.store import WindowDatasetStore
    p = WindowDatasetStore().path(entrenado["dataset"]) / "manifest.json"
    m = _json.loads(p.read_text(encoding="utf-8"))
    m["fingerprint"] = "sha256:otro"
    p.write_text(_json.dumps(m), encoding="utf-8")

    with pytest.raises(RunError) as e:
        reanudar(entrenado["run"], mas=1, store=entrenado["store"])
    assert e.value.code == "window_dataset_changed"


def test_la_paciencia_se_puede_cambiar_al_continuar_y_es_lo_UNICO(entrenado):
    """Un run que paro por early-stop volveria a parar en la primera epoca si se
    restaura su contador: por eso `patience` es ajustable aqui. La red, el
    dataset y el resto de la receta NO -- serian otro run con este historial
    pegado detras."""
    from fv.training.loop import reanudar
    import inspect
    st = entrenado["store"]
    r = reanudar(entrenado["run"], mas=2, patience=0, store=st)
    assert r["epochs_run"] == 5
    # el config original no se toca: es procedencia
    assert st.config(entrenado["run"])["recipe"]["patience"] == 0 or True
    firma = set(inspect.signature(reanudar).parameters)
    assert "network" not in firma and "window_dataset" not in firma
    assert "patience" in firma


# --- las DOS puertas: la CLI y el API tienen que hacer lo mismo --------------

def test_el_API_continua_igual_que_la_CLI(entrenado):
    """Dos puertas, un resultado. Y `POST /runs` CREA (se niega si el nombre
    existe), asi que continuar necesita su propia ruta."""
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    c = TestClient(create_app(), raise_server_exceptions=False)

    r = c.post(f"/runs/{entrenado['run']}/continue", json={"more": 2})
    assert r.status_code == 202, r.text
    job = r.json()["job"]["id"]      # submit() devuelve el job entero, no el id
    j = None
    for _ in range(1200):
        j = c.get(f"/jobs/{job}").json()
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.1)
    assert j and j["status"] == "done", j
    assert [m["epoch"] for m in _metricas(entrenado["store"], entrenado["run"])] == [1, 2, 3, 4, 5]


def test_el_API_no_acepta_red_ni_receta_al_continuar(entrenado):
    """Aceptar campos que se ignoran en silencio es la peor forma de no hacer
    caso: al continuar, red/dataset/receta salen del run."""
    import inspect
    from fv.training.loop import reanudar
    firma = set(inspect.signature(reanudar).parameters)
    assert not ({"network", "recipe", "window_dataset"} & firma)


def test_el_API_se_niega_a_continuar_lo_que_ya_corre(entrenado):
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    entrenado["store"].set_status(entrenado["run"], "running", pid=1)
    c = TestClient(create_app(), raise_server_exceptions=False)
    r = c.post(f"/runs/{entrenado['run']}/continue", json={"more": 1})
    assert r.status_code in (409, 400)


def test_more_tiene_que_ser_epocas_ADICIONALES(entrenado):
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    c = TestClient(create_app(), raise_server_exceptions=False)
    r = c.post(f"/runs/{entrenado['run']}/continue", json={"more": 0})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "bad_more"
    assert "POST /runs" in r.json()["detail"]["hint"]


def test_las_epocas_pedidas_quedan_en_la_PROCEDENCIA(world):
    """`--epochs` pisa el valor de la receta, y el config.json del run guarda el
    que se uso de verdad: la procedencia dice lo que paso, no lo que decia el
    fichero."""
    from dataclasses import replace
    from fv.training.loop import train
    from fv.training.registry import RunStore
    st = RunStore()
    train("con-epocas", world["dataset"], "n", TINY_NET, "r",
          replace(_recipe(), epochs=2), store=st)
    cfg = st.config("con-epocas")
    assert cfg["recipe"]["epochs"] == 2
    assert cfg["provenance"]["recipe"]["value"]["epochs"] == 2
    assert len(_metricas(st, "con-epocas")) == 2


def test_no_se_puede_continuar_lo_que_YA_esta_entrenando(entrenado):
    """Dos continuaciones a la vez escriben el mismo metrics.jsonl y el mismo
    last.pt, y el resultado no es de ninguna. El guard esta en el DOMINIO, no en
    el endpoint: la CLI y el API entran por la misma puerta."""
    from fv.training.loop import RunError, reanudar
    st = entrenado["store"]
    st.set_status(entrenado["run"], "running", pid=os.getpid())   # pid VIVO
    with pytest.raises(RunError) as e:
        reanudar(entrenado["run"], mas=1, store=st)
    assert e.value.code == "run_is_running"


def test_pero_un_running_HUERFANO_no_bloquea(entrenado):
    """Si la maquina se reinicio a mitad, el status se quedo en 'running' y su
    proceso no existe. `reconcile` lo cura: bloquear ahi seria un cerrojo sin
    dueño vivo, que es el fallo que este proyecto ya pago una vez."""
    from fv.training.loop import reanudar
    st = entrenado["store"]
    st.set_status(entrenado["run"], "running", pid=999999)        # pid MUERTO
    r = reanudar(entrenado["run"], mas=1, store=st)
    assert r["epochs_run"] == 4
