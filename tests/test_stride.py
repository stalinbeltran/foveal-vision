"""El barrido del stride de EXTRACCION (docs/barrido-stride.md §7).

Metodo de tests.md §2: se testea la costura, no la funcion. Aqui la mitad que
puede romperse no es la rejilla nueva -- es la de siempre, que sostiene 700+ runs
ya pagados y todas las tablas publicadas. Por eso los primeros tests preguntan si
lo viejo sigue saliendo IGUAL, no si lo nuevo funciona.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from fv.windows.extract import ExtractConfig, ExtractError, _positions, extract_windows


# ---------------------------------------------------------------- no-regresion

def test_eval_stride_default_is_bit_identical(world):
    """`eval_stride=None` => el MISMO windows.npz, huella incluida.

    Es la promesa que protege todo lo ya medido: si esto se rompe, cambian los
    datasets de los que salieron todos los f1 publicados, y no lo diria nadie.
    """
    from fv import settings
    raiz = settings.window_datasets_root()
    viejo = json.loads((raiz / "mini-b8" / "manifest.json").read_text(encoding="utf-8"))

    cfg = ExtractConfig(source="local/mini", window_size=8, stride=6,
                        val_frac=0.2, test_frac=0.2, seed=1)   # sin eval_stride
    nuevo = extract_windows(cfg, raiz / "control")

    assert nuevo["fingerprint"] == viejo["fingerprint"]
    assert nuevo["num_windows"] == viejo["num_windows"]
    assert nuevo["windows_per_split"] == viejo["windows_per_split"]
    assert (hashlib.sha256((raiz / "control" / "windows.npz").read_bytes()).hexdigest()
            == hashlib.sha256((raiz / "mini-b8" / "windows.npz").read_bytes()).hexdigest())


def test_eval_stride_none_equals_explicit_same_value(world):
    """Declararlo igual que el de train no puede cambiar nada (control)."""
    from fv import settings
    raiz = settings.window_datasets_root()
    cfg = ExtractConfig(source="local/mini", window_size=8, stride=6,
                        val_frac=0.2, test_frac=0.2, seed=1, eval_stride=6)
    m = extract_windows(cfg, raiz / "explicito")
    viejo = json.loads((raiz / "mini-b8" / "manifest.json").read_text(encoding="utf-8"))
    assert m["fingerprint"] == viejo["fingerprint"]


# ------------------------------------------------------------- el mecanismo B

def test_eval_stride_splits_use_their_own_grid(world):
    """train sigue a `stride`; val y test siguen a `eval_stride`.

    Se comprueba contra el conteo calculado con `_positions`, que es la misma
    funcion que corta: la costura es que el manifest diga lo que de verdad hay.
    """
    from fv import settings
    raiz = settings.window_datasets_root()
    cfg = ExtractConfig(source="local/mini", window_size=8, stride=2,
                        val_frac=0.2, test_frac=0.2, seed=1, eval_stride=6)
    m = extract_windows(cfg, raiz / "st2-ev6")

    H, W, n = 36, 48, 8
    por_img_train = len(_positions(H, n, 2)) * len(_positions(W, n, 2))
    por_img_eval = len(_positions(H, n, 6)) * len(_positions(W, n, 6))
    reparto = json.loads((raiz / "st2-ev6" / "split.json").read_text(encoding="utf-8"))

    assert m["windows_per_split"]["train"] == por_img_train * len(reparto["train"])
    assert m["windows_per_split"]["val"] == por_img_eval * len(reparto["val"])
    assert m["windows_per_split"]["test"] == por_img_eval * len(reparto["test"])
    assert por_img_train > por_img_eval          # control: la rejilla de train es mas densa


def test_eval_stride_same_images_across_strides(world):
    """Dos brazos con distinto `stride` y el mismo `seed` de B reparten LAS
    MISMAS imagenes. Si esto se rompe, cada brazo se evalua sobre otras imagenes
    y la comparacion no significa nada -- sin que falle nada: saldrian numeros."""
    from fv import settings
    raiz = settings.window_datasets_root()
    repartos = []
    for s in (2, 6):
        cfg = ExtractConfig(source="local/mini", window_size=8, stride=s,
                            val_frac=0.2, test_frac=0.2, seed=1, eval_stride=3)
        extract_windows(cfg, raiz / f"brazo-{s}")
        repartos.append(json.loads(
            (raiz / f"brazo-{s}" / "split.json").read_text(encoding="utf-8")))
    assert repartos[0] == repartos[1]


def test_eval_stride_shared_grid_gives_same_eval_count(world):
    """El punto del estudio: con rejilla de eval FIJA, los brazos se examinan
    del MISMO numero de ventanas aunque su train sea de tamanos muy distintos."""
    from fv import settings
    raiz = settings.window_datasets_root()
    ms = []
    for s in (2, 6):
        cfg = ExtractConfig(source="local/mini", window_size=8, stride=s,
                            val_frac=0.2, test_frac=0.2, seed=1, eval_stride=3)
        ms.append(extract_windows(cfg, raiz / f"fija-{s}"))
    assert ms[0]["windows_per_split"]["val"] == ms[1]["windows_per_split"]["val"]
    assert ms[0]["windows_per_split"]["test"] == ms[1]["windows_per_split"]["test"]
    assert ms[0]["windows_per_split"]["train"] != ms[1]["windows_per_split"]["train"]


def test_eval_stride_invalid_is_refused_with_reason(world):
    from fv import settings
    cfg = ExtractConfig(source="local/mini", window_size=8, stride=6, eval_stride=0)
    with pytest.raises(ExtractError) as e:
        extract_windows(cfg, settings.window_datasets_root() / "malo")
    assert e.value.code == "eval_stride_invalid"
    assert e.value.hint


def test_eval_stride_travels_to_the_manifest(world):
    """Un objeto ensena la definicion con la que se hizo: sin esto, dos brazos
    solo se distinguirian por el nombre del directorio."""
    from fv import settings
    cfg = ExtractConfig(source="local/mini", window_size=8, stride=4,
                        val_frac=0.2, test_frac=0.2, seed=1, eval_stride=3)
    m = extract_windows(cfg, settings.window_datasets_root() / "conmanifest")
    assert m["config"]["stride"] == 4
    assert m["config"]["eval_stride"] == 3


# ------------------------------------------------------------- el mecanismo D

def test_windows_per_epoch_exact_count():
    from fv.training.sampling import VentanasPorEpoca
    for pool, por_epoca in ((100, 10), (100, 100), (10, 25), (7, 7)):
        s = VentanasPorEpoca(pool, por_epoca, seed=1)
        assert len(s) == por_epoca
        assert len(s.indices_de(1)) == por_epoca
        assert len(list(iter(s))) == por_epoca


def test_windows_per_epoch_no_replacement_within_pass():
    """Con W <= pool ninguna ventana sale dos veces en la misma epoca."""
    from fv.training.sampling import VentanasPorEpoca
    idx = VentanasPorEpoca(100, 40, seed=1).indices_de(1)
    assert len(set(idx.tolist())) == 40


def test_windows_per_epoch_repeats_by_whole_permutations():
    """Con W > pool el reparto es por permutaciones completas, no con reemplazo:
    ninguna ventana puede salir dos veces mas que otra."""
    from fv.training.sampling import VentanasPorEpoca
    idx = VentanasPorEpoca(10, 25, seed=1).indices_de(1)
    cuentas = np.bincount(idx, minlength=10)
    assert cuentas.min() >= 2 and cuentas.max() <= 3
    assert int(cuentas.sum()) == 25


def test_windows_per_epoch_is_reproducible():
    from fv.training.sampling import VentanasPorEpoca
    a = VentanasPorEpoca(1000, 50, seed=7).indices_de(3)
    b = VentanasPorEpoca(1000, 50, seed=7).indices_de(3)
    assert (a == b).all()
    otra_semilla = VentanasPorEpoca(1000, 50, seed=8).indices_de(3)
    otra_epoca = VentanasPorEpoca(1000, 50, seed=7).indices_de(4)
    assert not (a == otra_semilla).all()      # control
    assert not (a == otra_epoca).all()        # control


def test_windows_per_epoch_advances_each_pass():
    """El DataLoader llama a __iter__ una vez por epoca: dos pasadas seguidas
    tienen que dar subconjuntos distintos, o cada epoca veria lo mismo."""
    from fv.training.sampling import VentanasPorEpoca
    s = VentanasPorEpoca(1000, 50, seed=1)
    assert list(iter(s)) != list(iter(s))


def _train_loader(world, por_epoca: int):
    """El loader de train que armaria el loop, sin entrenar.

    Testea la costura de verdad -- cuantos lotes consume una epoca -- en vez de
    leer el codigo fuente. Re-entrenar seria lento y testearia torch (tests.md §5).
    """
    from torch.utils.data import DataLoader
    from fv.fovea import dims_of
    from fv.models.builder import full_config
    from fv.training.sampling import VentanasPorEpoca
    from fv.windows.dataset import FoveatedWindowDataset
    from fv.windows.store import WindowDatasetStore
    from tests.conftest import TINY_NET

    net = full_config(TINY_NET)
    arrays = WindowDatasetStore().arrays(world["dataset"])
    ds = FoveatedWindowDataset(arrays, dims_of(net), split=0,
                               pool_mode=net["pool_mode"], pad_mode=net["pad_mode"])
    if por_epoca > 0:
        return ds, DataLoader(ds, batch_size=10, num_workers=0,
                              sampler=VentanasPorEpoca(len(ds), por_epoca, 1))
    return ds, DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)


def test_windows_per_epoch_caps_the_epoch(world):
    """Una epoca consume `windows_per_epoch` ventanas, no el pool entero."""
    ds, loader = _train_loader(world, 50)
    assert len(ds) > 50                                  # control: el pool es mayor
    vistas = sum(x.shape[0] for x, _ in loader)
    assert vistas == 50


def test_windows_per_epoch_above_pool_repeats(world):
    """Con W mayor que el pool, la epoca sigue entregando exactamente W."""
    ds, loader = _train_loader(world, len(_train_loader(world, 0)[0]) * 3)
    vistas = sum(x.shape[0] for x, _ in loader)
    assert vistas == len(ds) * 3


def test_windows_per_epoch_zero_keeps_the_old_loader(world):
    """Con 0 no se construye sampler: se recorre el pool entero, como siempre."""
    from fv.training.recipe import Recipe
    assert Recipe().windows_per_epoch == 0
    ds, loader = _train_loader(world, 0)
    vistas = sum(x.shape[0] for x, _ in loader)
    assert vistas == len(ds)


def test_recipe_accepts_old_specs_without_the_field():
    """`Recipe(**spec['base_recipe_value'])` sobre un spec ya escrito -- que no
    lleva el campo -- tiene que seguir construyendo. Ausente = 0, no error."""
    from fv.training.recipe import Recipe
    viejo = {"lr": 0.0014, "batch_size": 85, "epochs": 100, "seed": 1}
    assert Recipe(**viejo).windows_per_epoch == 0


# ---------------------------------------------------------------- la flota

def _flota():
    import importlib.util
    ruta = Path(__file__).resolve().parents[1] / "scripts" / "estudio_flota.py"
    spec = importlib.util.spec_from_file_location("_flota_test", ruta)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pytest.skip("estudio_flota necesita el lanzador clonado al lado")
    return mod


def test_multi_dataset_payload_carries_all(tmp_path, monkeypatch):
    """El tar de N datasets los lleva los N. Con uno, exactamente lo de siempre."""
    F = _flota()
    raiz = tmp_path / "repo"
    for sub in ("src", "scripts", "configs"):
        (raiz / sub).mkdir(parents=True)
        (raiz / sub / "x.py").write_text("x = 1", encoding="utf-8")
    (raiz / "pyproject.toml").write_text("[project]", encoding="utf-8")
    for d in ("ds-a", "ds-b"):
        p = raiz / "data" / "window-datasets" / d
        p.mkdir(parents=True)
        (p / "windows.npz").write_bytes(b"npz")
        (p / "manifest.json").write_text("{}", encoding="utf-8")
    (raiz / "sweeps" / "s1").mkdir(parents=True)
    (raiz / "sweeps" / "s1" / "spec.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(F, "ROOT", raiz)
    # los recorridos ya no cuelgan de ROOT: viven en el repo de datos
    # (fv.settings.data_root), asi que el aislamiento va por el almacen.
    monkeypatch.setattr(F.SWEEPS, "root", raiz / "sweeps")

    import tarfile
    tar = F.construir_payload([{"nombre": "s1"}], ["ds-a", "ds-b"])
    with tarfile.open(tar) as t:
        nombres = t.getnames()
    assert any(n.endswith("data/window-datasets/ds-a/windows.npz") for n in nombres)
    assert any(n.endswith("data/window-datasets/ds-b/windows.npz") for n in nombres)

    uno = F.construir_payload([{"nombre": "s1"}], ["ds-a"])
    with tarfile.open(uno) as t:
        solo = t.getnames()
    assert not any("ds-b" in n for n in solo)


def test_multi_dataset_missing_npz_dies_before_renting(tmp_path, monkeypatch):
    """Falta un npz => muere con su razon ANTES de tocar Vast. Descubrirlo a
    mitad son maquinas ya alquiladas y facturando para nada."""
    F = _flota()
    raiz = tmp_path / "repo"
    (raiz / "data" / "window-datasets" / "ds-a").mkdir(parents=True)
    (raiz / "data" / "window-datasets" / "ds-a" / "windows.npz").write_bytes(b"n")
    monkeypatch.setattr(F, "ROOT", raiz)
    # los recorridos ya no cuelgan de ROOT: viven en el repo de datos
    # (fv.settings.data_root), asi que el aislamiento va por el almacen.
    monkeypatch.setattr(F.SWEEPS, "root", raiz / "sweeps")
    with pytest.raises(SystemExit):
        F.construir_payload([{"nombre": "s1"}], ["ds-a", "ds-que-no-esta"])


def test_payload_accepts_a_bare_string(tmp_path, monkeypatch):
    """Compatibilidad: la llamada de un solo dataset como cadena sigue valiendo."""
    F = _flota()
    raiz = tmp_path / "repo"
    for sub in ("src", "scripts", "configs"):
        (raiz / sub).mkdir(parents=True)
        (raiz / sub / "x.py").write_text("x = 1", encoding="utf-8")
    (raiz / "pyproject.toml").write_text("[project]", encoding="utf-8")
    p = raiz / "data" / "window-datasets" / "ds-a"
    p.mkdir(parents=True)
    (p / "windows.npz").write_bytes(b"npz")
    (raiz / "sweeps" / "s1").mkdir(parents=True)
    (raiz / "sweeps" / "s1" / "spec.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(F, "ROOT", raiz)
    # los recorridos ya no cuelgan de ROOT: viven en el repo de datos
    # (fv.settings.data_root), asi que el aislamiento va por el almacen.
    monkeypatch.setattr(F.SWEEPS, "root", raiz / "sweeps")
    assert F.construir_payload([{"nombre": "s1"}], "ds-a").exists()


# ------------------------------------------------------------- el vigilante

def test_vigilante_sobrantes_respects_ajena():
    """La rama de sobrantes NO puede destruir lo que `juzgar` declaro ajeno.

    Es el fallo encontrado el 2026-08-27: al terminar sus recorridos, un
    vigilante destruia TODA instancia con el prefijo -- incluidas las de otro
    estudio vivo, que momentos antes habia declarado ajenas.
    """
    fuente = (Path(__file__).resolve().parents[1] / "scripts" /
              "vigilante_avance.py").read_text(encoding="utf-8")
    assert "ajenas_por_nombre" in fuente
    assert "i not in danadas and i not in ajenas_por_nombre" in fuente


def test_vigilante_prefix_is_a_parameter():
    """Dos estudios a la vez no pueden compartir espacio de nombres."""
    fuente = (Path(__file__).resolve().parents[1] / "scripts" /
              "vigilante_avance.py").read_text(encoding="utf-8")
    assert 'PREFIJO: list = [PREFIJO_DEF]' in fuente
    assert '"--prefijo"' in fuente
    assert 'PREFIJO[0] = args.prefijo' in fuente
    # y la flota lo hereda al relanzar, o la flota nueva naceria con etiquetas
    # que este vigilante ya no reconoceria como suyas
    assert '"--prefijo", args.prefijo' in fuente


# --------------------------------------------------------------- el comparador

def _modulo(nombre: str):
    import importlib.util
    ruta = Path(__file__).resolve().parents[1] / "scripts" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(f"_{nombre}_test", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _StoreFalso:
    """Un SweepStore con specs en memoria: el comparador solo les pide `spec`."""

    def __init__(self, specs: dict):
        self._specs = specs

    def exists(self, n): return n in self._specs
    def spec(self, n): return self._specs[n]
    def used_by_study(self, estudio):
        return sorted(n for n, s in self._specs.items() if s.get("study") == estudio)


def _brazo(stride: int, eval_stride: int = 5, por_epoca: int = 84_000,
           estudio: str = "E") -> dict:
    return {"study": estudio, "objective": "f1", "seed": 1,
            "window_dataset": f"ds-st{stride:02d}",
            "base_recipe_value": {"windows_per_epoch": por_epoca},
            "eje_dataset": {"campo": "stride", "valor": stride,
                            "eval_stride": eval_stride, "estudio": estudio}}


def test_stride_informe_groups_by_stride():
    """La costura de la que depende todo el comparador: `aggregate_seeds` agrupa
    por el punto SIN `seed`, asi que dandole {"stride": s, "seed": k} agrupa por
    stride. Si esto dejara de ser cierto, la tabla mezclaria brazos."""
    from fv.sweeps.winner import aggregate_seeds
    scored = [
        {"run": "a1", "point": {"stride": 1, "seed": 1}, "value": 0.90,
         "seconds_per_epoch": 40},
        {"run": "a2", "point": {"stride": 1, "seed": 2}, "value": 0.80,
         "seconds_per_epoch": 42},
        {"run": "b1", "point": {"stride": 16, "seed": 1}, "value": 0.60,
         "seconds_per_epoch": 41},
        {"run": "b2", "point": {"stride": 16, "seed": 2}, "value": 0.50,
         "seconds_per_epoch": 39},
    ]
    grupos = aggregate_seeds(scored, "max", "seconds_per_epoch")
    por_stride = {g["point"]["stride"]: g for g in grupos}
    assert set(por_stride) == {1, 16}
    assert por_stride[1]["value"] == pytest.approx(0.85)     # calculado a mano
    assert por_stride[16]["value"] == pytest.approx(0.55)
    assert por_stride[1]["n_seeds"] == 2
    assert "seed" not in por_stride[1]["point"]              # el eje es el stride


def test_stride_informe_refuses_mixed_eval_grid():
    """Brazos con rejillas de evaluacion distintas => se NIEGA.

    Seria la trampa de barrido-stride.md 2.1 disfrazada de tabla: cada brazo
    examinado de otra cosa y los f1 comparados como si fueran el mismo numero.
    """
    I = _modulo("estudio_stride_informe")
    store = _StoreFalso({"a": _brazo(1, eval_stride=5),
                         "b": _brazo(16, eval_stride=3)})
    with pytest.raises(SystemExit):
        I.brazos_del_estudio(store, "E", [])


def test_stride_informe_refuses_mixed_budget():
    """Brazos con distinto `windows_per_epoch` => se NIEGA: la tabla mediria el
    presupuesto y no la densidad."""
    I = _modulo("estudio_stride_informe")
    store = _StoreFalso({"a": _brazo(1, por_epoca=84_000),
                         "b": _brazo(16, por_epoca=0)})
    with pytest.raises(SystemExit):
        I.brazos_del_estudio(store, "E", [])


def test_stride_informe_refuses_a_sweep_without_the_label():
    """Un recorrido sin `eje_dataset` no dice que valor representa; meterlo en la
    tabla seria inventarselo."""
    I = _modulo("estudio_stride_informe")
    otro = _brazo(2)
    del otro["eje_dataset"]
    otro["study"] = "E"
    store = _StoreFalso({"a": _brazo(1), "b": otro})
    with pytest.raises(SystemExit):
        I.brazos_del_estudio(store, "E", [])


def test_stride_informe_orders_arms_by_stride():
    I = _modulo("estudio_stride_informe")
    store = _StoreFalso({"z": _brazo(16), "a": _brazo(1), "m": _brazo(4)})
    brazos = I.brazos_del_estudio(store, "E", [])
    assert [b[2] for b in brazos] == [1, 4, 16]


def test_stride_informe_refuses_when_the_study_has_no_arms():
    I = _modulo("estudio_stride_informe")
    with pytest.raises(SystemExit):
        I.brazos_del_estudio(_StoreFalso({}), "no-existe", [])


# ------------------------------------------------------------------- el humo

def test_humo_sweeps_are_a_separate_study():
    """Los recorridos de validacion NO pueden llamarse como los del estudio: si
    se llamaran igual, la flota los daria por hechos y el estudio quedaria
    'medido' con las 3 epocas de la prueba."""
    E = _modulo("estudio_stride")
    assert E.nombre_sweep(1) == "stride-01"
    assert E.nombre_sweep(1, humo=True) == "stride-h01"
    assert E.nombre_sweep(1) != E.nombre_sweep(1, humo=True)


def test_eval_stride_cannot_be_one_of_the_arms():
    """Si la rejilla de evaluacion fuera uno de los strides barridos, ese brazo
    habria entrenado sobre las posiciones exactas del examen."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/estudio_stride.py",
         "--strides", "1,4,16", "--eval-stride", "4", "--solo-recorridos"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "esta ENTRE los strides barridos" in (r.stderr + r.stdout)


def test_progreso_names_an_axis_that_lives_in_the_dataset():
    """El monitor tiene que saber decir QUE esta mirando.

    Con el eje en el dataset, `space` solo trae la replica y `estudio_progreso`
    imprimia «eje ?» y «media PARCIAL por ?». Un monitor que no nombra lo que
    muestra es medio monitor.
    """
    fuente = (Path(__file__).resolve().parents[1] / "scripts" /
              "estudio_progreso.py").read_text(encoding="utf-8")
    assert 'eje_ds = spec.get("eje_dataset") or {}' in fuente
    assert 'eje, valor_del_dataset = eje_ds["campo"], eje_ds.get("valor")' in fuente
    # y el fallback no puede pisar a un eje de verdad: solo actua si no hay
    assert 'if eje is None and eje_ds.get("campo"):' in fuente


def test_vigilante_only_sees_fleets_of_its_own_study(tmp_path):
    """`flota_viva` exige LAS TRES: ser el script, del workspace, y de mis puntos.

    Se prueba con un proceso de VERDAD, no simulando `pgrep`: la version anterior
    de este test devolvia PIDs inventados, y eso dejo de valer en cuanto la
    comprobacion paso a mirar `/proc` -- que es justo lo que la hace fiable.
    """
    import subprocess
    import time
    V = _modulo("vigilante_avance")
    raiz = Path(__file__).resolve().parents[1]
    falso = tmp_path / "estudio_flota.py"
    falso.write_text("import time\ntime.sleep(8)\n", encoding="utf-8")

    proc = subprocess.Popen([sys.executable, str(falso), "--sweep", "stride-01"],
                            cwd=str(raiz))
    try:
        time.sleep(0.7)
        # (1) es un python ejecutando el script, (2) su cwd es el workspace y
        # (3) menciona mi recorrido -> ES mi flota
        assert V.flota_viva(["stride-01", "stride-02"]) == proc.pid
        # (3) falla: mismo proceso, otro estudio -> no es cosa mia
        assert V.flota_viva(["otro-recorrido"]) != proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_vigilante_ignores_a_fleet_from_another_workspace(tmp_path):
    """(2) el CWD: la misma flota lanzada desde OTRO workspace no es mia.

    La flota se lanza con ruta relativa, asi que su linea de comando no contiene
    el workspace por ningun lado (CLAUDE.md del coordinador). Sin mirar el cwd,
    dos copias del repo se pisarian la regla 4.
    """
    import subprocess
    import time
    V = _modulo("vigilante_avance")
    falso = tmp_path / "estudio_flota.py"
    falso.write_text("import time\ntime.sleep(8)\n", encoding="utf-8")
    otro = tmp_path / "otro-workspace"
    otro.mkdir()

    proc = subprocess.Popen([sys.executable, str(falso), "--sweep", "stride-01"],
                            cwd=str(otro))
    try:
        time.sleep(0.7)
        assert V.flota_viva(["stride-01"]) != proc.pid
        assert V.flota_viva() != proc.pid
    finally:
        proc.kill()
        proc.wait()


def test_stride_informe_says_when_there_is_no_noise_band():
    """Sin replicas, delta = 0 y CUALQUIER diferencia «supera» delta: R1 corona
    un ganador y R3 marca rupturas por construccion. Un «delta = 0,0000» parece
    precision y es ausencia, asi que se dice con todas las letras."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/estudio_stride_informe.py",
         "--estudio", "stride-2026-08-27-humo", "--json", "/tmp/informe-banda.json"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        pytest.skip("el estudio de humo no tiene medidas en este arbol")
    if "| 1 |" in r.stdout and "| 16 |" in r.stdout:      # los dos brazos, 1 semilla
        assert "no hay banda de ruido medida" in r.stdout
        assert "Nada de lo de abajo declara nada" in r.stdout
        assert "R3 · Monotonía.** ⚠ **No evaluable sin banda" in r.stdout


def test_flota_viva_ignores_processes_that_only_mention_the_script():
    """Un proceso que MENCIONA `estudio_flota.py` no es una flota.

    MEDIDO el 2026-08-27, y es el fallo mas caro de esta tanda: un avisador
    armado con `while pgrep -f "estudio_flota.py --sweep stride-01"` llevaba esa
    cadena en su propia linea de comando, asi que `pgrep -f` lo devolvia y el
    vigilante lo conto como flota. Resultado: **19 vueltas (~3 h) diciendo "hay
    una flota viva" sin relanzar los 14 puntos que faltaban** -- un estudio que
    parece vigilado y no avanza, que es justo el sintoma que la regla 4 existe
    para evitar.

    (El mismo `pgrep` casaba tambien con el propio avisador, asi que ademas nunca
    salio el aviso: se esperaba a si mismo.)
    """
    import subprocess
    import time
    V = _modulo("vigilante_avance")
    raiz = Path(__file__).resolve().parents[1]
    señuelo = subprocess.Popen(
        ["sh", "-c", "sleep 8 # estudio_flota.py --sweep stride-01"], cwd=str(raiz))
    try:
        time.sleep(0.5)
        # el señuelo SI sale en pgrep -- si no, el test no estaria probando nada
        salida = subprocess.run(["pgrep", "-f", "estudio_flota.py"],
                                capture_output=True, text=True).stdout.split()
        assert str(señuelo.pid) in salida, "el señuelo no aparece en pgrep"
        # ...y aun asi no cuenta como flota
        assert V.flota_viva(["stride-01"]) != señuelo.pid
        assert V.flota_viva() != señuelo.pid
    finally:
        señuelo.kill()
        señuelo.wait()


def test_estudio_stride_refuses_a_missing_source_with_the_fix():
    """La fuente esta en .gitignore: un clon limpio no la trae. Se comprueba
    ANTES de nada y con el comando que la reconstruye al lado -- lo que se
    descubre a mitad se resuelve improvisando."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/estudio_stride.py",
         "--fuente", "local/no-existe-seguro", "--strides", "16", "--solo-datasets"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=180)
    assert r.returncode != 0
    salida = r.stderr + r.stdout
    assert "no encuentro la fuente" in salida
    assert "bench_dataset.py build" in salida          # el arreglo, no solo el fallo


def test_sonda_projection_respects_the_budget():
    """La sonda proyecta s/epoca a partir de los pasos de UNA epoca, y con
    `windows_per_epoch` una epoca ya no es el pool entero.

    MEDIDO el 2026-08-27 en la validacion en Vast: anuncio «~392 s/epoca» para un
    brazo cuya epoca real son 988 pasos. La criba no se equivocaba -- ordena por
    ms/paso -- pero el numero del log si, por 20,9x.
    """
    fuente = (Path(__file__).resolve().parents[1] / "scripts" /
              "sonda_velocidad.py").read_text(encoding="utf-8")
    assert 'por_epoca = int(getattr(receta, "windows_per_epoch", 0) or 0)' in fuente
    assert "ventanas_epoca = por_epoca if por_epoca > 0 else len(ds)" in fuente
    assert '"ventanas_pool": len(ds),' in fuente      # la procedencia viaja


def test_stride_informe_never_contradicts_itself_with_one_arm():
    """Un informe que dice dos cosas opuestas es peor que uno que calla.

    Con un solo brazo medido, min(strides) == max(strides), y R1 imprimia a la
    vez «la densidad no compra nada en este rango» y «el eje no queda cerrado por
    arriba». Sale al leer un estudio A MEDIAS, que es el caso normal mientras la
    flota corre.
    """
    import subprocess
    raiz = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "scripts/estudio_stride_informe.py",
         "--estudio", "stride-2026-08-27-humo",
         "--json", "/tmp/informe-test.json"],
        cwd=str(raiz), capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        pytest.skip("el estudio de humo no tiene medidas en este arbol")
    salida = r.stdout
    no_compra = "la densidad no compra nada" in salida
    no_cerrado = "no queda cerrado por arriba" in salida
    assert not (no_compra and no_cerrado), (
        "R1 afirma las dos cosas a la vez:\n" + salida)
    # y con un brazo lo que toca es decir que no se puede leer
    if salida.count("| stride |") and salida.count("`dirty1000") == 1:
        assert "R1 · Saturación.** ⚠ **No evaluable" in salida
        assert "R3 · Monotonía.** ⚠ **No evaluable" in salida


def test_r4_counts_steps_not_seconds():
    """R4 tiene que medir el PRESUPUESTO, no el reloj.

    La primera version comparaba `s/epoca` entre brazos con un tope del 15 %, y
    dio FALLO en el estudio completo por ruido de maquina: MEDIDO el 2026-08-27,
    dentro de un mismo brazo -misma config, cinco maquinas alquiladas- el
    cociente max/min de s/epoca llego a 2,50, mas que el 1,53 que separaba las
    medias de los brazos. Un control que da falsa alarma se acaba ignorando, y
    entonces no hay control. Ahora lee los pasos del `config.json` que devolvio
    cada maquina, que es exacto y no tiene ruido.
    """
    fuente = (Path(__file__).resolve().parents[1] / "scripts" /
              "estudio_stride_informe.py").read_text(encoding="utf-8")
    assert "pasos_por_run" in fuente
    assert 'w = int(cfg.get("windows_per_epoch", 0) or 0)' in fuente
    assert "pasos de gradiente por época" in fuente
    # y s/epoca sigue saliendo, pero declarado como lo que es
    assert "*Información, no control*" in fuente
    # el veredicto ya no puede depender de un umbral de segundos
    assert "DESVIO_MAX_R4" not in fuente.split("def main")[1]


def test_r2_reports_the_real_group_sizes():
    """`permutation_test` devuelve `n`/`m`, no `n_a`/`n_b`: el informe imprimia
    «(None vs None semillas)», que es un hueco con pinta de dato."""
    fuente = (Path(__file__).resolve().parents[1] / "scripts" /
              "estudio_stride_informe.py").read_text(encoding="utf-8")
    assert "contraste.get('n_a')" not in fuente
    assert "contraste.get('n')" in fuente and "contraste.get('m')" in fuente
