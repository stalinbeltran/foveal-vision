"""El dataset de casos fallidos (`fv.fallidos`).

Cada test construye su mundo (tests.md): una fuente de 10 imagenes, un dataset de
ventanas sobre ella y un run de una epoca. Los numeros no tienen que ser buenos
-- lo que se comprueba son las COSTURAS:

1. la verdad recompuesta desde B es la MISMA que la de A (es toda la licencia que
   se toma este modulo sobre el contrato 13, y sin este test es una creencia);
2. lo que sale es un dataset de ventanas de verdad, y se puede entrenar sobre el;
3. el criterio de "peor" es el declarado y es determinista;
4. sin fuente y sin pedir el camino degradado, se falla con razon.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import TINY_NET


@pytest.fixture()
def trained(world):
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    train("fallos-run", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)
    return {"run": "fallos-run", "store": store, **world}


def _aprobar(run: str) -> None:
    """El catalogo vive en el repo de datos, que en tests es un tmp_path."""
    from fv.inference.catalogo import catalogo_path
    from fv.ioutils import write_json_atomic
    write_json_atomic(catalogo_path(),
                      {"version": 1, "runs": {run: {"ficheros": ["best.pt"]}}})


# ------------------------------------------------------------- (1) la verdad

def test_la_verdad_recompuesta_desde_b_es_la_de_la_fuente(world):
    """LA costura. Si esto falla, `--verdad ventanas` esta midiendo otra cosa."""
    from fv.datasets.loader import SourceDataset
    from fv.fallidos import verdad_de_ventanas
    from fv.windows.store import WindowDatasetStore

    wstore = WindowDatasetStore()
    arrays = wstore.arrays(world["dataset"])
    manifest = wstore.manifest(world["dataset"])
    recompuesta = verdad_de_ventanas(arrays, manifest["config"]["window_size"])

    source = SourceDataset(world["source"])
    assert len(recompuesta) == manifest["num_samples"]
    comparadas = 0
    for idx, v in recompuesta.items():
        verdad = sorted(tuple(round(c, 4) for c in b.bbox)
                        for b in source.sample_at(idx).blocks
                        if b.kind == "paragraph")
        if not v["completa"]:
            continue                       # el borde: su propio test, abajo
        assert sorted(tuple(round(c, 4) for c in caja)
                      for caja in v["cajas"]) == verdad
        comparadas += 1
    assert comparadas >= 8, "casi todas las imagenes deberian reconstruirse"


def test_un_parrafo_cortado_por_el_borde_se_marca_y_no_se_inventa(tmp_path,
                                                                  monkeypatch):
    """La esquina que ninguna ventana ve no se puede recuperar, y eso se DICE.

    Se dice en vez de recomponer una caja a medias: una caja parcial puntuaria
    como error de la red lo que es una perdida del dato.
    """
    import json

    from PIL import Image

    monkeypatch.setenv("FV_ROOT", str(tmp_path))
    monkeypatch.setenv("FV_DATASETS_ROOT", str(tmp_path / "no-external"))
    from fv import settings
    from fv.fallidos import verdad_de_ventanas
    from fv.windows.extract import ExtractConfig, extract_windows
    from fv.windows.store import WindowDatasetStore

    W, H = 32, 32
    out = settings.local_sources_root() / "borde"
    (out / "images").mkdir(parents=True)
    lineas = []
    for i, (x0, y0, x1, y1) in enumerate([(4.0, 4.0, 20.0, 20.0),      # entera
                                          (4.0, 20.0, 20.0, 40.0)]):   # se sale
        img = np.full((H, W), 220, dtype=np.uint8)
        img[int(y0):min(H, int(y1)), int(x0):int(x1)] = 40
        Image.fromarray(img).save(out / f"images/{i:06d}.png")
        quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        lineas.append(json.dumps({"index": i, "image": f"images/{i:06d}.png",
                                  "labels": {"width": W, "height": H, "blocks": [
                                      {"block_id": "b0", "kind": "paragraph",
                                       "angle": 0.0, "quad": quad}]}}))
    (out / "labels.jsonl").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    (out / "dataset.json").write_text(json.dumps({"id": "borde", "count": 2}),
                                      encoding="utf-8")

    extract_windows(ExtractConfig(source="local/borde", window_size=8, stride=4,
                                  val_frac=0.0, test_frac=0.0, seed=1),
                    settings.window_datasets_root() / "borde-b8")
    v = verdad_de_ventanas(WindowDatasetStore().arrays("borde-b8"), 8)

    assert v[0]["completa"] is True
    assert v[0]["cajas"] == [(4.0, 4.0, 20.0, 20.0)]
    # la de abajo pierde BR y BL: se marca y NO se inventa una caja
    assert v[1]["completa"] is False
    assert v[1]["cajas"] == []


def test_sin_fuente_se_falla_con_razon_y_no_se_cae_al_camino_degradado(trained):
    """Contrato 13: «nunca se puntua contra las etiquetas de ventana».

    El camino degradado existe, pero hay que PEDIRLO. Que un defecto se caiga
    solo al camino de al lado es como se mide una cosa creyendo medir otra: los
    numeros saldrian, nadie veria un error, y significarian otra cosa.

    La fuente se BORRA de verdad en vez de simularlo, que es exactamente el
    estado en que quedo esta maquina (los PNG de `dirty-1000-80px` nunca entraron
    en git y se fueron con el server anterior). Asi el `source_not_found` que se
    envuelve es el que produce el loader, no uno inventado por el test.
    """
    import shutil

    from fv import settings
    from fv.datasets.loader import SourceError, resolve_source
    from fv.fallidos import Criterio, FallidosError, evaluar

    shutil.rmtree(settings.local_sources_root() / "mini")
    with pytest.raises(SourceError) as se:
        resolve_source(trained["source"])
    assert se.value.code == "source_not_found"

    with pytest.raises(FallidosError) as e:
        evaluar(trained["run"], trained["dataset"], Criterio(),
                store=trained["store"])
    assert e.value.code == "verdad_necesita_fuente"
    assert "--verdad ventanas" in e.value.hint

    # ...y el camino degradado si funciona sin ella: es para lo que existe
    ev = evaluar(trained["run"], trained["dataset"], Criterio(),
                 verdad="ventanas", store=trained["store"])
    assert ev["verdad"]["origen"] == "ventanas-reconstruidas"
    assert ev["verdad"]["imagenes"] == 10


def test_las_puertas_de_evaluar_rechazan_lo_que_no_entienden(trained):
    """Un argumento malo se dice por su nombre y antes de inferir nada."""
    from fv.fallidos import Criterio, FallidosError, evaluar

    with pytest.raises(FallidosError) as e:
        evaluar(trained["run"], trained["dataset"], Criterio(), split="holdout",
                store=trained["store"])
    assert e.value.code == "split_desconocido"

    with pytest.raises(FallidosError) as e:
        evaluar(trained["run"], trained["dataset"], Criterio(),
                verdad="adivinala", store=trained["store"])
    assert e.value.code == "verdad_desconocida"


def test_un_split_vacio_se_dice_en_vez_de_medir_cero_imagenes(trained):
    """Medir sobre cero imagenes daria un dataset vacio con pinta de resultado."""
    from fv import settings
    from fv.fallidos import Criterio, FallidosError, evaluar
    from fv.windows.extract import ExtractConfig, extract_windows

    extract_windows(ExtractConfig(source=trained["source"], window_size=8,
                                  stride=6, val_frac=0.0, test_frac=0.0, seed=1),
                    settings.window_datasets_root() / "sin-val")
    with pytest.raises(FallidosError) as e:
        evaluar(trained["run"], "sin-val", Criterio(), split="val",
                store=trained["store"])
    assert e.value.code == "split_vacio"


def test_una_politica_de_split_desconocida_se_rechaza(trained):
    from fv.fallidos import Criterio, FallidosError, crear

    with pytest.raises(FallidosError) as e:
        crear(trained["run"], criterio=Criterio(min_errores=0),
              nombre="mala-politica", split_salida="a-ojo",
              store=trained["store"])
    assert e.value.code == "split_salida_desconocida"


def test_un_run_sin_procedencia_no_elige_dataset_por_su_cuenta(trained):
    """Elegir uno seria comparar dos cosas que nadie decidio comparar."""
    from fv.fallidos import FallidosError, dataset_de
    from fv.ioutils import write_json_atomic

    cfg = trained["store"].config(trained["run"])
    cfg.pop("provenance", None)
    write_json_atomic(trained["store"].path(trained["run"]) / "config.json", cfg)
    with pytest.raises(FallidosError) as e:
        dataset_de(trained["run"], trained["store"])
    assert e.value.code == "run_sin_procedencia"
    assert "--dataset" in e.value.hint


# ------------------------------------------------- (2) lo que sale es un B

def test_lo_que_sale_es_un_dataset_de_ventanas_entrenable(trained):
    """Un B de pleno derecho: el store lo lee y `fv-train` entrena sobre el.

    Es el punto de todo esto -- "para probar inferencias o entrenar nuevas nn".
    Si solo fuera un informe, este test no existiria.
    """
    from fv.fallidos import Criterio, crear
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.windows.store import WindowDatasetStore

    res = crear(trained["run"], criterio=Criterio(min_errores=0),
                store=trained["store"])
    nombre = res["nombre"]

    wstore = WindowDatasetStore()
    m = wstore.manifest(nombre)
    arrays = wstore.arrays(nombre)
    assert m["num_samples"] == len(res["elegidas"])
    assert m["num_windows"] == int(arrays["y"].shape[0])
    assert set(arrays) == {"y", "sample_idx", "window_xy", "split", "images",
                           "images_sample_idx"}
    assert m["source_id"] == wstore.manifest(trained["dataset"])["source_id"]
    # el fingerprint describe ESTE npz (contrato 8), no el que venia heredado
    import hashlib
    assert m["fingerprint"] == "sha256:" + hashlib.sha256(
        (wstore.path(nombre) / "windows.npz").read_bytes()).hexdigest()

    train("sobre-fallos", nombre, "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=16), store=trained["store"])
    prov = trained["store"].config("sobre-fallos")["provenance"]
    assert prov["window_dataset"]["name"] == nombre


def test_un_subconjunto_estricto_tiene_huella_propia(trained):
    """Menos imagenes -> otro npz -> otra huella.

    ⚠ Y el reciproco tambien vale, que es lo que hace util comprobarlo: si el
    criterio deja pasar TODAS las imagenes, el npz sale byte a byte igual que el
    del dataset base y la huella coincide. Eso no es un fallo del recorte, es la
    prueba de que el recorte no toca nada que no tenga que tocar.
    """
    from fv.fallidos import Criterio, crear
    from fv.windows.store import WindowDatasetStore

    wstore = WindowDatasetStore()
    base = wstore.manifest(trained["dataset"])

    todo = crear(trained["run"], criterio=Criterio(min_errores=0),
                 nombre="todas", store=trained["store"])
    assert len(todo["elegidas"]) == base["num_samples"]
    assert wstore.manifest("todas")["fingerprint"] == base["fingerprint"]

    pocas = crear(trained["run"], criterio=Criterio(min_errores=0, max_imagenes=3),
                  nombre="tres", store=trained["store"])
    assert len(pocas["elegidas"]) == 3
    m = wstore.manifest("tres")
    assert m["num_samples"] == 3
    assert m["fingerprint"] != base["fingerprint"]
    assert m["num_windows"] < base["num_windows"]


def test_conserva_los_indices_originales_de_a(trained):
    """`sample_idx` no se renumera: es lo que permite cruzar con el base y con A."""
    from fv.fallidos import Criterio, crear
    from fv.windows.store import WindowDatasetStore

    res = crear(trained["run"], criterio=Criterio(min_errores=0),
                store=trained["store"])
    arrays = WindowDatasetStore().arrays(res["nombre"])
    elegidas = {r["index"] for r in res["elegidas"]}
    assert set(int(a) for a in arrays["images_sample_idx"]) == elegidas
    assert set(int(s) for s in arrays["sample_idx"]) == elegidas
    # y las ventanas que viajan son TODAS las de esas imagenes, ni una mas
    base = WindowDatasetStore().arrays(trained["dataset"])
    assert int(arrays["y"].shape[0]) == int(np.isin(base["sample_idx"],
                                                    sorted(elegidas)).sum())


def test_la_procedencia_queda_escrita_en_el_manifest(trained):
    """Dentro de un ano, este dataset tiene que poder decir que es.

    Incluido --y sobre todo-- de donde salio la verdad: leer estos numeros como
    si se hubieran medido contra la fuente seria creerse mas de lo que hay.
    """
    from fv.fallidos import Criterio, crear
    from fv.windows.store import WindowDatasetStore

    res = crear(trained["run"], criterio=Criterio(min_errores=0),
                verdad="ventanas", store=trained["store"])
    f = WindowDatasetStore().manifest(res["nombre"])["fallidos"]
    assert f["red"]["run"] == trained["run"]
    assert f["red"]["checkpoint"] == "best.pt"
    assert len(f["red"]["sha256"]) == 64
    assert f["base"]["dataset"] == trained["dataset"]
    assert f["verdad"]["origen"] == "ventanas-reconstruidas"
    assert "borde" in f["verdad"]["aviso"]
    assert f["criterio"]["iou_threshold"] == 0.5
    assert f["knobs"]["threshold"] == 0.5


def test_no_sobrescribe_nunca(trained):
    """Misma regla que `extract_windows`: un dataset que cambia bajo el mismo
    nombre invalida en silencio todo lo medido contra el."""
    from fv.fallidos import Criterio, FallidosError, crear

    crear(trained["run"], criterio=Criterio(min_errores=0), store=trained["store"])
    with pytest.raises(FallidosError) as e:
        crear(trained["run"], criterio=Criterio(min_errores=0),
              store=trained["store"])
    assert e.value.code == "dataset_ya_existe"
    assert "--nombre" in e.value.hint


# --------------------------------------------------------- (3) el criterio

def test_peor_es_mas_errores_y_el_orden_es_determinista():
    """El criterio declarado, sobre datos a mano: no depende de que entrene nada."""
    from fv.fallidos import ordenar

    filas = [
        {"index": 1, "errores": 1, "f1": 0.8, "mean_iou": 0.9},
        {"index": 2, "errores": 3, "f1": 0.1, "mean_iou": 0.5},
        {"index": 3, "errores": 1, "f1": 0.8, "mean_iou": 0.6},
        {"index": 4, "errores": 0, "f1": 1.0, "mean_iou": 0.95},
        {"index": 5, "errores": 1, "f1": 0.4, "mean_iou": None},
    ]
    assert [r["index"] for r in ordenar(filas)] == [2, 5, 3, 1, 4]
    # sin emparejar (None) es PEOR que emparejar mal, no mejor ni igual a cero
    assert ordenar(filas)[1]["index"] == 5
    assert [r["index"] for r in ordenar(list(reversed(filas)))] == [2, 5, 3, 1, 4]


def test_el_tope_se_queda_con_las_peores_no_con_las_primeras():
    from fv.fallidos import Criterio, seleccionar

    filas = [{"index": i, "errores": i % 4, "f1": 0.5, "mean_iou": 0.5,
              "gt_completa": True} for i in range(12)]
    elegidas = seleccionar(filas, Criterio(min_errores=1, max_imagenes=3))
    assert [r["errores"] for r in elegidas] == [3, 3, 3]


def test_la_verdad_incompleta_se_excluye_salvo_que_se_pida():
    """Su cuenta de errores no es creible: lo que la red detecto donde falta la
    verdad se contaria como invento suyo."""
    from fv.fallidos import Criterio, seleccionar

    filas = [{"index": 0, "errores": 5, "f1": 0.0, "mean_iou": None,
              "gt_completa": False},
             {"index": 1, "errores": 1, "f1": 0.5, "mean_iou": 0.6,
              "gt_completa": True}]
    assert [r["index"] for r in seleccionar(filas, Criterio())] == [1]
    assert [r["index"] for r in
            seleccionar(filas, Criterio(incluir_gt_parcial=True))] == [0, 1]


def test_sin_fallos_no_escribe_un_dataset_vacio(trained):
    from fv.fallidos import Criterio, FallidosError, crear

    with pytest.raises(FallidosError) as e:
        crear(trained["run"], criterio=Criterio(min_errores=10_000),
              store=trained["store"])
    assert e.value.code == "sin_fallos"


# ------------------------------------------------------- (4) el split y el nombre

def test_el_split_se_conserva_por_defecto_y_se_puede_rehacer(trained):
    """Conservar es el defecto porque es el unico que no miente sobre lo que la
    red evaluada vio: una imagen que estaba en train sigue en train."""
    from fv.fallidos import Criterio, crear
    from fv.windows.store import WindowDatasetStore

    wstore = WindowDatasetStore()
    base = {int(a): s for a, s in
            zip(wstore.arrays(trained["dataset"])["images_sample_idx"],
                [int(wstore.arrays(trained["dataset"])["split"][
                    wstore.arrays(trained["dataset"])["sample_idx"] == int(a)][0])
                 for a in wstore.arrays(trained["dataset"])["images_sample_idx"]])}

    res = crear(trained["run"], criterio=Criterio(min_errores=0),
                nombre="conserva", store=trained["store"])
    a = wstore.arrays("conserva")
    for i, s in zip(a["sample_idx"], a["split"]):
        assert int(s) == base[int(i)]

    res = crear(trained["run"], criterio=Criterio(min_errores=0),
                nombre="a-train", split_salida="train", store=trained["store"])
    assert set(int(s) for s in wstore.arrays("a-train")["split"]) == {0}
    assert wstore.manifest("a-train")["windows_per_split"]["val"] == 0
    assert wstore.split_map("a-train")["val"] == []
    assert len(wstore.split_map("a-train")["train"]) == len(res["elegidas"])


def test_el_nombre_corto_sale_de_la_red():
    from fv.fallidos import nombre_dataset

    assert nombre_dataset("demo-fov16-optimo") == "optimo-fallidos"
    assert nombre_dataset("fov16-edge-p20") == "edge-fallidos"
    assert nombre_dataset("fov16-mask-p20") == "mask-fallidos"
    assert nombre_dataset("do-v-0000-dropout0p0_seed1") == \
        "do-v-0000-dropout0p0_seed1-fallidos"


# ----------------------------------- (5) de quien es la culpa de cada fallo

def test_distingue_no_ver_la_esquina_de_emparejarla_mal():
    """Las dos averias tienen el mismo sintoma ("falla en esta imagen") y
    arreglos opuestos: reentrenar la red, o tocar `_reconstruct` de F."""
    from fv.fallidos import esquinas_acertadas

    verdad = {"TL": [(10.0, 10.0)], "TR": [(30.0, 10.0)],
              "BR": [(30.0, 20.0)], "BL": [(10.0, 20.0)]}

    # las cuatro vistas, y cerca: la red hizo su trabajo
    todas = [{"corner": c, "x": p[0][0], "y": p[0][1], "score": 0.9}
             for c, p in verdad.items()]
    assert esquinas_acertadas(todas, verdad, 8.0) == {"tp": 4, "fp": 0, "fn": 0}

    # una que no vio: eso SI es de la red
    assert esquinas_acertadas(todas[:3], verdad, 8.0) == {"tp": 3, "fp": 0, "fn": 1}

    # una vista lejos cuenta como inventada Y como no vista, que es lo que es
    lejos = [dict(todas[0], x=70.0, y=50.0), *todas[1:]]
    assert esquinas_acertadas(lejos, verdad, 8.0) == {"tp": 3, "fp": 1, "fn": 1}

    # y la tolerancia manda: la misma deteccion, con otro radio, ya empareja
    assert esquinas_acertadas(lejos, verdad, 100.0)["tp"] == 4


def test_solo_emparejado_marca_las_imagenes_con_las_esquinas_bien(trained):
    """La bandera existe en cada imagen del dataset, no solo agregada."""
    from fv.fallidos import Criterio, evaluar

    ev = evaluar(trained["run"], trained["dataset"], Criterio(),
                 verdad="ventanas", store=trained["store"])
    for r in ev["per_image"]:
        esperado = (r["esquinas"]["fp"] == 0 and r["esquinas"]["fn"] == 0
                    and r["errores"] > 0)
        assert r["solo_emparejado"] is esperado
    d = ev["diagnostico"]
    assert d["solo_emparejado"] == sum(1 for r in ev["per_image"]
                                       if r["solo_emparejado"])
    assert d["imagenes_con_error"] == sum(1 for r in ev["per_image"] if r["errores"])
    assert d["tol_esquina_px"] > 0


def test_un_dataset_a_medias_no_bloquea_el_reintento(trained, monkeypatch):
    """Se construye al lado y se renombra: aparece entero o no aparece.

    Este comando se lanza desde Telegram, donde el bot se reinicia y se lleva por
    delante lo que este corriendo. Un directorio con manifest y sin npz seria
    ademas lo PEOR: `dataset_ya_existe` bloquearia el reintento, o sea que un
    corte dejaria el nombre quemado hasta que alguien lo borrara a mano.

    ⚠ Y no uses `monkeypatch.undo()` para volver atras: revierte TODO lo que ese
    monkeypatch haya puesto, incluido el `FV_DATA_ROOT` del fixture autouse de
    conftest -- o sea que a partir de ahi el test escribe en el repo de datos de
    VERDAD. Por eso el parche se apaga con una bandera y no se deshace.
    """
    import fv.fallidos as F
    from fv.fallidos import Criterio, crear
    from fv.windows.store import WindowDatasetStore

    wstore = WindowDatasetStore()
    real, estado = F.write_json_atomic, {"romper": True}

    def a_veces(*a, **k):
        if estado["romper"]:
            raise RuntimeError("el bot se reinicio a mitad")
        return real(*a, **k)

    monkeypatch.setattr("fv.fallidos.write_json_atomic", a_veces)
    with pytest.raises(RuntimeError):
        crear(trained["run"], criterio=Criterio(min_errores=0), nombre="cortado",
              store=trained["store"])
    assert not wstore.path("cortado").exists()

    estado["romper"] = False
    res = crear(trained["run"], criterio=Criterio(min_errores=0),
                nombre="cortado", store=trained["store"])
    assert wstore.manifest("cortado")["num_samples"] == len(res["elegidas"])
    assert not wstore.path("cortado.parcial").exists()
