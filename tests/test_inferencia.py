"""Qué redes se guardan para inferir, y por dónde llegan sus pesos.

La regla la puso el dueño el 2026-08-31: los pesos de un run **no se guardan por
defecto**; sólo los de las redes que él aprueba una a una, y **sólo ésas** puede
usar la web app para inferir. Aquí se fija esa regla y el camino que la
implementa (antesala -> promoción -> catálogo).

Repartido por la consecuencia del fallo (R10), que aquí es muy desigual:

  1. **la puerta del endpoint** -> escribe ficheros en el disco del server a
     partir de datos de fuera. Un nombre de fichero libre acepta `../../algo`, y
     un `.pt` es un pickle: código. Es lo único de este fichero que puede
     comprometer la máquina, así que se prueba lo que se RECHAZA, no lo que pasa.
  2. **que una red sin aprobar acabe infiriendo** -> silencioso: la app enseña
     cajas de una red que nadie eligió y no falla nada. Es el caso que la lista
     existe para impedir.
  3. **que la promoción pierda o corrompa un peso** -> se pierden horas de
     máquina, y ya pasó una vez (`fov-optimo-p20`, ~1 h 40 min reentrenando).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fv import settings
from fv.inference import catalogo
from fv.inference.catalogo import CatalogoError


@pytest.fixture()
def app_limpia(tmp_path, monkeypatch):
    """API sobre un repo de datos y una antesala vacíos y temporales."""
    monkeypatch.setenv("FV_ROOT", str(tmp_path / "codigo"))
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    (tmp_path / "datos").mkdir(parents=True)
    (tmp_path / "codigo").mkdir(parents=True)
    from fv.api.app import create_app
    # `raise_server_exceptions=False` para que el handler de errores de dominio
    # RESPONDA en vez de que el cliente re-lance: parte de lo que se prueba aquí
    # es justamente que un `.pt` ilegible da un 400 con su razón y no un 500.
    return TestClient(create_app(), raise_server_exceptions=False)


def _run_falso(nombre: str, con_pesos: bool = False) -> None:
    """Un run mínimo en el repo de datos: lo que la promoción necesita ver."""
    from fv.training.registry import RunStore
    d = RunStore().destino(nombre, {})
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"provenance": {}}), encoding="utf-8")
    if con_pesos:
        (d / "best.pt").write_bytes(b"\x00pesos")


# ------------------------------------------------- 1. la puerta del endpoint

@pytest.mark.parametrize("fichero", ["otro.pt", "../../escape.pt", "best.PT", ""])
def test_solo_se_aceptan_best_y_last(app_limpia, fichero):
    """Un nombre de fichero libre compone una ruta con datos de fuera. La lista
    es de DOS y se comprueba por igualdad, no por sufijo ni por sanitización."""
    r = app_limpia.put(f"/inference/staging/run1/{fichero}", content=b"x")
    assert r.status_code in (400, 404, 405)
    if r.status_code == 400:
        assert r.json()["detail"]["code"] == "peso_desconocido"
    assert not list(settings.inference_staging_root().rglob("*escape*"))


@pytest.mark.parametrize("run", ["../fuera", "a/b", ".", ".."])
def test_el_nombre_del_run_no_puede_ser_una_ruta(run):
    """Se comprueba en el MÓDULO, no sólo en el endpoint: quien compone una ruta
    con un dato de fuera no puede confiar en que su único llamador de hoy siga
    siendo el único mañana."""
    with pytest.raises(CatalogoError) as ex:
        catalogo.guardar_en_antesala(run, "best.pt", b"x")
    assert ex.value.code in ("nombre_de_run_invalido", "peso_desconocido")


def test_un_cuerpo_vacio_se_rechaza_en_vez_de_dejar_un_peso_de_0_bytes(app_limpia):
    """Un `best.pt` de 0 bytes es peor que ninguno: existe, así que la app lo
    ofrece, y falla al cargarlo — «la red detecta mal» otra vez."""
    r = app_limpia.put("/inference/staging/run1/best.pt", content=b"")
    assert r.status_code == 400 and r.json()["detail"]["code"] == "checkpoint_vacio"
    assert not (catalogo.staging_dir("run1") / "best.pt").exists()


def test_hay_un_techo_de_tamano(app_limpia, monkeypatch):
    """Sin techo, una subida es un disco lleno — y un disco lleno tumba el
    entrenamiento que estaba corriendo."""
    import fv.api.app as mod
    monkeypatch.setattr(mod, "MAX_CHECKPOINT_BYTES", 10)
    r = app_limpia.put("/inference/staging/run1/best.pt", content=b"x" * 11)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "checkpoint_demasiado_grande"


def test_la_subida_no_carga_el_pickle(app_limpia):
    """Se guardan BYTES. `torch.load` es lo que ejecuta el pickle de un `.pt`, y
    en la subida no hay ninguna razón para hacerlo: unos bytes que no son un
    checkpoint tienen que aterrizar sin ejecutarse y fallar más tarde, al usarlos."""
    r = app_limpia.put("/inference/staging/run1/best.pt", content=b"esto no es un .pt")
    assert r.status_code == 201
    assert (catalogo.staging_dir("run1") / "best.pt").read_bytes() == b"esto no es un .pt"


# ------------------------- 2. una red sin aprobar NO infiere, aunque tenga pesos

def test_un_peso_en_disco_sin_aprobar_no_se_usa(app_limpia):
    """El fallo silencioso que la lista existe para impedir: un `.pt` que se coló
    (una copia a mano, un tar desempaquetado) haría inferir con una red que nadie
    eligió, y la app no distinguiría eso de una aprobada."""
    _run_falso("colado", con_pesos=True)
    from fv.training.registry import RunStore
    ckpt, origen = catalogo.checkpoint_de("colado", RunStore())
    assert ckpt is None and origen is None
    r = app_limpia.get("/runs").json()["runs"]
    fila = next(x for x in r if x["name"] == "colado")
    assert fila["has_checkpoint"] is True      # el fichero SÍ está...
    assert fila["approved"] is False           # ...pero no está aprobada...
    assert fila["inference"] is None           # ...así que no infiere


def test_el_error_distingue_los_tres_casos(app_limpia):
    """El mensaje viejo decía siempre «espera a que termine una época», que sobre
    un run terminado hace días con 107 épocas es falso — y manda a esperar algo
    que no va a pasar."""
    _run_falso("con-pesos-sin-aprobar", con_pesos=True)
    _run_falso("sin-nada")
    # se pregunta por una ruta de INFERENCIA (predict), no de introspección:
    # `/kernels` no exige aprobación a propósito (ver `_model_para_mirar`)
    r = app_limpia.post("/runs/con-pesos-sin-aprobar/predict",
                        json={"source": "local/x", "index": 0})
    assert r.json()["detail"]["code"] == "run_not_approved_for_inference"
    r = app_limpia.post("/runs/sin-nada/predict", json={"source": "local/x", "index": 0})
    d = r.json()["detail"]
    assert d["code"] == "run_has_no_checkpoint"
    assert "reentrenarlo" in d["hint"]         # dice qué hacer, no «espera»


def test_introspeccionar_NO_pide_aprobacion(app_limpia):
    """Mirar los kernels de un run que pediste por su nombre no es inferir con
    una red que nadie eligió. Y exigirlo rompería el flujo local: `fv-train`
    deja `best.pt` en el directorio del run, no en la antesala, así que un run
    recién entrenado aquí no se podría ni mirar sin commitear 2,7 MB antes.

    Se comprueba que la puerta NO es la del catálogo: llega hasta cargar el
    checkpoint (y ahí falla porque los bytes son falsos), en vez de pararse en
    `run_not_approved_for_inference`."""
    _run_falso("solo-para-mirar", con_pesos=True)
    assert not catalogo.esta_aprobada("solo-para-mirar")
    r = app_limpia.get("/runs/solo-para-mirar/kernels")
    codigo = (r.json().get("detail") or {}).get("code")
    # llega hasta CARGAR el checkpoint --y ahi se queja de que estos bytes no lo
    # son-- en vez de pararse en la puerta del catalogo
    assert codigo == "checkpoint_ilegible", codigo


def test_unos_bytes_que_no_son_un_checkpoint_dan_un_error_con_razon(app_limpia):
    """El endpoint acepta bytes sin cargarlos (no se ejecuta el pickle en la
    subida), así que unos bytes equivocados se descubren al USARLOS. Ese momento
    tiene que traer su razón y su arreglo, no un 500 opaco en la pantalla."""
    _run_falso("basura")
    app_limpia.put("/inference/staging/basura/best.pt", content=b"no soy un .pt")
    r = app_limpia.get("/runs/basura/kernels")
    d = r.json()["detail"]
    assert d["code"] == "checkpoint_ilegible"
    assert "vuelve a subirlo" in d["hint"]


def test_la_antesala_gana_al_definitivo(app_limpia):
    """Durante un entrenamiento la versión buena es la que acaba de bajar: la
    del catálogo es de la sonda anterior. Cuando el run termina, la promoción
    iguala las dos y el orden deja de importar."""
    from fv.training.registry import RunStore
    rs = RunStore()
    _run_falso("r")
    app_limpia.put("/inference/staging/r/best.pt", content=b"\x00v1")
    app_limpia.post("/inference/staging/r/promote")
    app_limpia.delete("/inference/staging/r")          # promovido y recogido
    _, origen = catalogo.checkpoint_de("r", rs)
    assert origen == "catalogo"
    # ...y llega la sonda siguiente
    catalogo.guardar_en_antesala("r", "best.pt", b"\x00mas-nuevo")
    ckpt, origen = catalogo.checkpoint_de("r", rs)
    assert origen == "antesala" and ckpt.read_bytes() == b"\x00mas-nuevo"


def test_un_catalogo_roto_no_se_lee_como_vacio():
    """Leerlo como vacío desaprobaría TODAS las redes sin decirlo — el fallo
    silencioso, otra vez, y en la dirección que rompe la app entera."""
    p = catalogo.catalogo_path()
    original = p.read_text(encoding="utf-8") if p.exists() else None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{roto", encoding="utf-8")
        with pytest.raises(CatalogoError) as ex:
            catalogo.leer()
        assert ex.value.code == "catalogo_ilegible"
    finally:
        if original is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(original, encoding="utf-8")


def test_sin_catalogo_no_hay_aprobadas_pero_tampoco_error(app_limpia):
    """Una máquina sin el fichero es una que aún no aprobó ninguna red: es un
    estado legítimo (el que tenía el proyecto hasta el 2026-08-30)."""
    assert not catalogo.catalogo_path().exists()
    assert catalogo.aprobadas() == []
    assert app_limpia.get("/inference").json()["aprobadas"] == {}


# ------------------------------------------------- 3. la promoción no pierde

def test_promover_copia_los_dos_pesos_y_aprueba(app_limpia):
    """Es UNA decisión: unos pesos en el repo de datos que nadie aprobó son
    2,7 MB que git no suelta nunca y que la app no usaría."""
    _run_falso("nuevo")
    app_limpia.put("/inference/staging/nuevo/best.pt", content=b"\x00best")
    app_limpia.put("/inference/staging/nuevo/last.pt", content=b"\x00last")
    r = app_limpia.post("/inference/staging/nuevo/promote",
                        json={"motivo": "la de la prueba"})
    assert r.status_code == 200
    d = r.json()
    assert sorted(d["copiados"]) == ["best.pt", "last.pt"]
    from fv.training.registry import RunStore
    destino = RunStore().path("nuevo")
    assert (destino / "best.pt").read_bytes() == b"\x00best"
    assert (destino / "last.pt").read_bytes() == b"\x00last"
    assert catalogo.esta_aprobada("nuevo")
    assert catalogo.entrada("nuevo")["motivo"] == "la de la prueba"
    # y dice cómo hacer que sobreviva a rehacer la máquina
    assert "git push" in d["commit"]


def test_promover_sin_antesala_se_niega(app_limpia):
    """Nunca aprueba una red cuyos pesos no ha visto: aprobar sin fichero deja
    una entrada que promete algo que no está."""
    _run_falso("vacio")
    r = app_limpia.post("/inference/staging/vacio/promote")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "antesala_vacia"
    assert not catalogo.esta_aprobada("vacio")


def test_promover_un_run_que_no_existe_se_niega(app_limpia):
    """Los pesos acompañan a la descripción del run (config, metrics, summary).
    Sueltos, no se puede saber ni con qué red ni con qué dato se entrenaron."""
    app_limpia.put("/inference/staging/fantasma/best.pt", content=b"\x00x")
    r = app_limpia.post("/inference/staging/fantasma/promote")
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "run_sin_directorio"


def test_retirar_no_borra_los_pesos(app_limpia):
    """Retirar es reversible y borrar no. Y un peso ya commiteado no se va del
    historial de git por borrarlo del árbol: borrarlo daría una sensación de
    limpieza que no es cierta."""
    _run_falso("r2")
    app_limpia.put("/inference/staging/r2/best.pt", content=b"\x00b")
    app_limpia.post("/inference/staging/r2/promote")
    from fv.training.registry import RunStore
    destino = RunStore().path("r2")
    assert app_limpia.delete("/inference/approved/r2").status_code == 200
    assert not catalogo.esta_aprobada("r2")
    assert (destino / "best.pt").exists()      # el fichero sigue ahí


def test_limpiar_la_antesala_no_toca_lo_definitivo(app_limpia):
    _run_falso("r3")
    app_limpia.put("/inference/staging/r3/best.pt", content=b"\x00b")
    app_limpia.post("/inference/staging/r3/promote")
    app_limpia.delete("/inference/staging/r3")
    from fv.training.registry import RunStore
    assert (RunStore().path("r3") / "best.pt").exists()
    assert catalogo.en_antesala("r3") == []
    # y como ya está aprobada, sigue infiriendo desde el catálogo
    _, origen = catalogo.checkpoint_de("r3", RunStore())
    assert origen == "catalogo"


def test_la_escritura_es_atomica_y_no_deja_parciales(app_limpia):
    """`best.pt` se lee MIENTRAS se reemplaza (la revisión usa el modelo con el
    entrenamiento en marcha): quien lee obtiene la versión vieja o la nueva,
    nunca media."""
    app_limpia.put("/inference/staging/r4/best.pt", content=b"\x00v1")
    app_limpia.put("/inference/staging/r4/best.pt", content=b"\x00v2-mas-largo")
    d = catalogo.staging_dir("r4")
    assert (d / "best.pt").read_bytes() == b"\x00v2-mas-largo"
    assert [f.name for f in d.iterdir()] == ["best.pt"]   # ni un .parcial


# --------------------------------------------------------- lo que ve la app

def test_la_antesala_esta_fuera_del_repo_de_datos():
    """git guarda TODAS las versiones que se commitean: una sonda cada pocas
    épocas serían decenas de `last.pt` de 2 MB por run. Lo de en medio no puede
    tocar el repo de datos."""
    antesala = settings.inference_staging_root().resolve()
    datos = settings.data_root().resolve()
    assert not antesala.is_relative_to(datos)
    # ...y tampoco es la caché, que se puede borrar sin perder nada
    assert antesala != settings.cache_root().resolve()


def test_la_antesala_esta_ignorada_por_git():
    """Comprobado con git, no leyendo el .gitignore: lo que decide es git."""
    import subprocess
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]
    p = subprocess.run(["git", "check-ignore", "-q", "data/inferencia/x/best.pt"],
                       cwd=raiz)
    assert p.returncode == 0, "data/inferencia/ NO está ignorado en el repo de código"
