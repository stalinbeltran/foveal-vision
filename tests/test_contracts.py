"""One test per contract of organizacion.md §2, named by number (tests.md §3)."""

import numpy as np
import pytest

from tests.conftest import TINY_NET


def test_contract_01_window_size_mismatch_is_refused_before_reserving(world):
    from fv.validation import check_run
    from fv.windows.store import WindowDatasetStore
    manifest = WindowDatasetStore().manifest(world["dataset"])
    bad = dict(TINY_NET, N=20, c_frac=0.8)  # fovea 16 vs window 8
    problems = check_run(manifest, bad)
    assert any(p["code"] == "window_size_mismatch" for p in problems)
    # control: the matching net passes compatibility
    assert not [p for p in check_run(manifest, TINY_NET)
                if p["code"] == "window_size_mismatch"]


def test_monitor_and_objective_are_one_table_and_the_gates_agree(tmp_path):
    """A monitor names a val metric ('val_f1'); an objective names the metric
    ('f1'). Two vocabularies over ONE table — they were written twice and the
    direction half only knew 'val_f1', so a recipe saying 'f1' found the value
    and then kept the WORST epoch. Every gate refuses it now, with one code."""
    from fv.metrics import MONITORS, VAL_METRICS, checkpoint_record, monitor_key
    from fv.sweeps.spec import OBJECTIVES, check_sweep
    from fv.studies.driver import validate_plan
    from fv.training.recipe import RecipeStore, RecipeStoreError

    assert OBJECTIVES == VAL_METRICS                      # one table, two names
    assert set(MONITORS) == {f"val_{k}" for k in OBJECTIVES}
    assert all(monitor_key(m) in OBJECTIVES for m in MONITORS)

    # the direction is real, not a set membership: f1 keeps the HIGHEST epoch
    recs = [{"epoch": 1, "val": {"f1": 0.9, "loss": 0.5}},
            {"epoch": 2, "val": {"f1": 0.3, "loss": 0.1}}]
    assert checkpoint_record(recs, "val_f1")["epoch"] == 1
    assert checkpoint_record(recs, "val_loss")["epoch"] == 2

    # gate D: the recipe store, saving and reading
    store = RecipeStore(tmp_path)
    with pytest.raises(RecipeStoreError) as e:
        store.save("bad", {"monitor": "f1"})
    assert e.value.code == "unknown_monitor"
    assert not (tmp_path / "bad.yaml").exists()           # nothing reserved
    (tmp_path / "hand.yaml").write_text("monitor: f1\n", encoding="utf-8")
    with pytest.raises(RecipeStoreError) as e:
        store.get("hand")                                 # hand-edited file too
    assert e.value.code == "unknown_monitor"
    store.save("good", {"monitor": "val_f1"})             # control

    # gate H: a sweep axis over monitor
    problems = check_sweep({"space": {"monitor": ["f1"]}, "objective": "f1"})
    assert any(p["code"] == "unknown_monitor" for p in problems)
    assert not [p for p in check_sweep({"space": {"monitor": list(MONITORS)},
                                        "objective": "f1"})
                if p["code"] == "unknown_monitor"]        # control

    # gate I: the same axis in a study plan
    plan = {"window_dataset": "b", "base_recipe": "r", "objective": "f1",
            "axes": [{"axis": "monitor", "range": ["f1"]}]}
    assert any(p["code"] == "unknown_monitor" for p in validate_plan(plan))
    plan["axes"] = [{"axis": "monitor", "range": ["val_f1"]}]
    assert not [p for p in validate_plan(plan) if p["code"] == "unknown_monitor"]


def test_contract_01b_view_needs_images(world):
    from fv.validation import check_compatible
    from fv.windows.store import WindowDatasetStore
    manifest = dict(WindowDatasetStore().manifest(world["dataset"]))
    manifest["has_images"] = False   # a B that cannot feed the view
    problems = check_compatible(manifest, TINY_NET)
    assert any(p["code"] == "view_needs_images" for p in problems)


def test_contract_02_merge_sum_needs_equal_strides():
    from fv.validation import check_network
    bad = dict(TINY_NET, merge="sum", s_center=2, s_periph=1)
    assert any(p["code"] == "merge_sum_needs_equal_strides"
               for p in check_network(bad))
    ok = dict(TINY_NET, merge="sum", s_center=1, s_periph=1)
    assert not check_network(ok)


def test_contract_02_even_kernel_is_refused():
    from fv.validation import check_network
    assert any(p["code"] == "kernel_must_be_odd"
               for p in check_network(dict(TINY_NET, k_center=4)))


def test_contract_03_provenance_carries_name_value_and_fingerprint(world):
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    recipe = Recipe(epochs=1, batch_size=32)
    train("prov-run", world["dataset"], "tiny-net", TINY_NET, "tiny-recipe",
          recipe, store=store)
    cfg = store.config("prov-run")
    prov = cfg["provenance"]
    assert prov["network"]["name"] == "tiny-net"
    assert prov["network"]["value"]["N"] == TINY_NET["N"]
    assert prov["recipe"]["name"] == "tiny-recipe"
    assert prov["window_dataset"]["fingerprint"].startswith("sha256:")
    assert prov["environment"]["device"] == "cpu"
    assert "git_commit" in prov
    # execution (X) lives OUTSIDE the recipe (contract 10)
    assert "device" not in cfg["recipe"]


def test_contract_03_run_never_overwritten(world):
    from fv.training.registry import RunError, RunStore
    store = RunStore()
    store.create("dup", {"a": 1})
    with pytest.raises(RunError) as e:
        store.create("dup", {"a": 2})
    assert e.value.code == "run_exists"


def test_parametric_builder_no_regression_for_two_layers():
    """D-C2/D-C3/§12: with channels=[16,32] EXPLICIT (not the new default) the
    parametric builder reproduces the shape and param count of the old fixed
    two-layer net (captured before the change: 317612 params, flat 25600)."""
    from fv.models.builder import network_trace
    cfg = {"N": 20, "c_frac": 0.8, "d": 2, "pen_frac": 0.1, "n_layers": 2,
           "k_center": 3, "k_periph": 3, "s_center": 1, "s_periph": 1,
           "channels": [16, 32], "merge": "concat", "pool_mode": "avg",
           "pad_mode": "edge"}
    t = network_trace(cfg)
    assert t["num_params"] == 317612
    assert t["flat_features"] == 25600
    # the legacy ch1/ch2 form maps to the SAME model (read old, write channels)
    legacy = dict(cfg); del legacy["channels"]; legacy["ch1"] = 16; legacy["ch2"] = 32
    assert network_trace(legacy)["num_params"] == 317612


def test_parametric_builder_default_channels_are_constant_sixteen():
    """D-C2: a derived net (no channels, no ch1/ch2) defaults to [16]*n_layers."""
    from fv.models.builder import full_config
    assert full_config({"n_layers": 3})["channels"] == [16, 16, 16]
    assert full_config({"n_layers": 2})["channels"] == [16, 16]


def test_parametric_builder_three_layers_builds_and_forwards():
    """§12: n_layers=3 constructs and forwards with the corner-head shape."""
    import torch
    from fv.models.builder import build_model
    cfg = dict(TINY_NET); cfg.pop("ch1"); cfg.pop("ch2")
    cfg.update(n_layers=3, channels=[4, 8, 8])
    model = build_model(cfg)
    assert len(model.center_convs) == 3 and len(model.periph_convs) == 3
    out = model(torch.zeros(1, 1, cfg["N"], cfg["N"]))
    assert out.shape == (1, 4, 3)


def test_parametric_builder_stride_only_on_first_layer():
    """D-S1: the branch stride subsamples once (first layer); depth does not
    change the total subsampling, so branch_out is independent of n_layers."""
    from fv.models.builder import network_trace
    base = dict(TINY_NET); base.pop("ch1"); base.pop("ch2")
    base.update(s_center=2, s_periph=2)
    two = network_trace(dict(base, n_layers=2, channels=[4, 8]))
    three = network_trace(dict(base, n_layers=3, channels=[4, 8, 8]))
    assert two["branch_out"] == three["branch_out"]


def test_channels_length_must_match_n_layers():
    from fv.validation import check_network
    bad = dict(TINY_NET); bad.pop("ch1"); bad.pop("ch2")
    bad.update(n_layers=3, channels=[4, 8])  # only 2 for 3 layers
    assert any(p["code"] == "channels_length_mismatch" for p in check_network(bad))


def test_contract_04_checkpoint_rebuilds_the_net_without_yaml(world):
    import torch
    from fv.inference.checkpoint import load_model
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    train("ckpt-run", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)
    model = load_model(store.path("ckpt-run") / "best.pt")
    assert model.dims.center_out == 8       # geometry included
    out = model(torch.zeros(1, 1, 12, 12))
    assert out.shape == (1, 4, 3)


def test_contract_05_dataloader_and_inference_build_the_same_view(world):
    """The seam, not the function: both sides call THE SAME fv.fovea and the
    views are bit-identical for the same window."""
    import fv.inference.predict as predict_mod
    import fv.windows.dataset as dataset_mod
    from fv.fovea import build_view, derive_dims
    assert dataset_mod.build_view is predict_mod.build_view is build_view

    from fv.windows.store import WindowDatasetStore
    from fv.windows.dataset import FoveatedWindowDataset
    arrays = WindowDatasetStore().arrays(world["dataset"])
    dims = derive_dims(**{k: TINY_NET[k] for k in ("N", "c_frac", "d", "pen_frac")})
    ds = FoveatedWindowDataset(arrays, dims, split=0)
    x, _y = ds[0]
    img = arrays["images"][ds.image_row[0]]
    wx0, wy0 = (int(v) for v in ds.window_xy[0])
    view, _cov = build_view(img, wx0, wy0, dims)
    np.testing.assert_array_equal(x.numpy()[0], view)


def test_contract_07_import_directions():
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "fv"
    # settings e ioutils son HOJAS (raices del proyecto y escritura/formato de
    # fichero), no dominios: los puede usar cualquiera sin crear una direccion.
    # No se concede, se comprueba — si una hoja importase un dominio, dejaria de
    # serlo y esta linea lo diria antes que la regla de abajo.
    leaves = {"settings", "ioutils"}
    for leaf in leaves:
        tree = ast.parse((src / f"{leaf}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.startswith("fv."):
                raise AssertionError(f"{leaf}.py importa {node.module}: ya no es hoja")
    rules = {
        "fovea": set(), "metrics": set(), "matrixview": set(),
        "validation": {"fovea"},
        "models": {"fovea"},
        "windows": {"datasets", "fovea", "metrics"},
        "inference": {"models", "fovea", "matrixview", "metrics"},
        # ⑬ fv.task cruza E×A vía F: consume los dominios de abajo y NADA de
        # arriba (ni api, ni sweeps, ni studies) — es una métrica, no una puerta
        "task": {"datasets", "diagnostics", "inference", "metrics",
                 "training", "windows"},
    }
    for mod, allowed in rules.items():
        p = src / mod
        files = list(p.rglob("*.py")) if p.is_dir() else [src / f"{mod}.py"]
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and \
                        node.module.startswith("fv."):
                    dep = node.module.split(".")[1]
                    if dep == mod:
                        continue
                    assert dep in allowed | leaves, \
                        f"{f.name}: fv.{mod} importa fv.{dep} (no permitido)"


def test_contract_08_fingerprint_tracks_content_and_split_is_per_image(world):
    from fv import settings
    from fv.windows.extract import ExtractConfig, extract_windows
    from fv.windows.store import WindowDatasetStore
    store = WindowDatasetStore()
    m1 = store.manifest(world["dataset"])
    # same config, same content -> same fingerprint
    cfg = ExtractConfig(source=world["source"], window_size=8, stride=6,
                        val_frac=0.2, test_frac=0.2, seed=1)
    m2 = extract_windows(cfg, settings.window_datasets_root() / "twin")
    assert m1["fingerprint"] == m2["fingerprint"]
    # different content (other split seed) -> other fingerprint
    cfg3 = ExtractConfig(source=world["source"], window_size=8, stride=6,
                         val_frac=0.2, test_frac=0.2, seed=2)
    m3 = extract_windows(cfg3, settings.window_datasets_root() / "other")
    assert m3["fingerprint"] != m1["fingerprint"]
    # split is per image: no sample_idx appears in two splits
    arrays = store.arrays(world["dataset"])
    for s in np.unique(arrays["sample_idx"]):
        assert len(np.unique(arrays["split"][arrays["sample_idx"] == s])) == 1


def test_contract_09_objective_cannot_be_loss_if_lambda_in_space():
    from fv.sweeps.spec import check_sweep
    bad = {"space": {"lambda_pos": [0.1, 1.0]}, "objective": "loss"}
    assert any(p["code"] == "objective_varies_with_space" for p in check_sweep(bad))
    ok = {"space": {"lambda_pos": [0.1, 1.0]}, "objective": "f1"}
    assert not check_sweep(ok)   # control


def test_contract_10_device_is_not_recipe_identity():
    from fv.training.recipe import Recipe, RecipeStoreError, RecipeStore
    assert "device" not in Recipe().as_dict()
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        store = RecipeStore(Path(td))
        with pytest.raises(RecipeStoreError) as e:
            store.save("bad", {"lr": 0.001, "device": "cuda"})
        assert e.value.code == "execution_inside_recipe"


def test_contract_11_same_seed_same_weights_with_control(world):
    import torch
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    r = Recipe(epochs=2, batch_size=32, seed=5)
    train("rep-a", world["dataset"], "n", TINY_NET, "r", r, store=store)
    train("rep-b", world["dataset"], "n", TINY_NET, "r", r, store=store)
    wa = torch.load(store.path("rep-a") / "last.pt", weights_only=False)["model"]
    wb = torch.load(store.path("rep-b") / "last.pt", weights_only=False)["model"]
    for k in wa:
        assert torch.equal(wa[k], wb[k]), f"{k} difiere con la misma semilla"
    # control: another seed must differ, or "they repeat" is also satisfied by
    # a loop that ignores the seed entirely
    r2 = Recipe(epochs=2, batch_size=32, seed=6)
    train("rep-c", world["dataset"], "n", TINY_NET, "r", r2, store=store)
    wc = torch.load(store.path("rep-c") / "last.pt", weights_only=False)["model"]
    assert any(not torch.equal(wa[k], wc[k]) for k in wa)


def test_ranking_seam_the_value_describes_the_checkpoint_that_survives(world):
    """E↔H: the number that ranks a point must come from the epoch `best.pt`
    kept, not from the last epoch — best.pt is what diagnostics load and what a
    study carries forward. Asserts the SEAM (the loop's own best_epoch vs what
    the ranking picks), not the function: those two drifting apart is exactly
    how a ranking starts describing weights nobody has."""
    from fv.metrics import checkpoint_record
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    r = Recipe(epochs=3, batch_size=32, seed=2, monitor="val_loss")
    summary = train("seam", world["dataset"], "n", TINY_NET, "r", r, store=store)
    records = store.metrics_since("seam", 0)["records"]
    rec = checkpoint_record(records, "val_loss")
    assert rec is not None
    assert rec["epoch"] == summary["best_epoch"]        # the seam
    assert rec["val"]["loss"] == summary["best"]


def test_ranking_uses_the_checkpoint_epoch_not_the_last(world):
    """The regression the user hit: with a monitor that peaks BEFORE the end, the
    ranking must report the objective of the kept checkpoint. The last epoch's
    value stays visible as `value_last`, but it does not rank."""
    import json
    from fv.sweeps.runner import point_run_name, prepare_sweep, sweep_trials
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    store, rstore = SweepStore(), RunStore()
    spec = {"window_dataset": world["dataset"], "base_network": "tiny",
            "base_network_value": TINY_NET, "base_recipe": "quick",
            "base_recipe_value": {"epochs": 3, "batch_size": 32, "monitor": "val_loss"},
            "space": {"lr": [0.001]}, "strategy": "grid", "objective": "f1"}
    enriched = prepare_sweep("rank1", spec, TINY_NET, store)
    run = point_run_name("rank1", 0, enriched["points"][0])
    # a run whose val_loss bottoms at epoch 2 while f1 keeps wobbling
    d = rstore.create(run, {"recipe": {"monitor": "val_loss"}, "network": TINY_NET})
    rows = [{"epoch": 1, "val": {"loss": 0.9, "f1": 0.10}},
            {"epoch": 2, "val": {"loss": 0.3, "f1": 0.80}},   # best.pt lives here
            {"epoch": 3, "val": {"loss": 0.7, "f1": 0.55}}]   # the last epoch
    (d / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rstore.set_status(run, "done", epoch=3)
    t = sweep_trials("rank1", store, rstore)
    row = t["trials"][0]
    assert row["value"] == 0.80 and row["epoch"] == 2      # the checkpoint's f1
    assert row["value_last"] == 0.55                       # the old rule, kept in view
    assert row["epochs"] == 3
    assert t["value_from"] == "checkpoint"
    # monitor (val_loss) != objective (f1): the reader must be told
    assert t["monitors"] == ["val_loss"]
    assert t["monitor_matches_objective"] is False


def test_ranking_refuses_to_invent_a_value_without_a_checkpoint(world):
    """A monitor that never measured means the loop never wrote best.pt: there is
    no checkpoint to describe, so the point has NO value and carries the reason —
    it does not silently fall back to the last epoch (formatos §2)."""
    from fv.sweeps.runner import point_run_name, prepare_sweep, sweep_trials
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    import json
    store, rstore = SweepStore(), RunStore()
    spec = {"window_dataset": world["dataset"], "base_network": "tiny",
            "base_network_value": TINY_NET, "base_recipe": "quick",
            "base_recipe_value": {"epochs": 2, "batch_size": 32,
                                  "monitor": "val_pos_err_px"},
            "space": {"lr": [0.001]}, "strategy": "grid", "objective": "f1"}
    enriched = prepare_sweep("rank2", spec, TINY_NET, store)
    run = point_run_name("rank2", 0, enriched["points"][0])
    d = rstore.create(run, {"recipe": {"monitor": "val_pos_err_px"}, "network": TINY_NET})
    rows = [{"epoch": 1, "val": {"loss": 0.9, "f1": 0.10, "pos_err_px": None}},
            {"epoch": 2, "val": {"loss": 0.3, "f1": 0.80, "pos_err_px": None}}]
    (d / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rstore.set_status(run, "done", epoch=2)
    row = sweep_trials("rank2", store, rstore)["trials"][0]
    assert row["value"] is None                      # never 0, never the last epoch
    assert row["value_reason"]["code"] == "no_checkpoint"
    assert row["value_last"] == 0.80                 # visible, but it does not rank


def test_the_api_serves_every_vocabulary_the_ui_would_otherwise_copy(world):
    """Costura front↔backend: cada lista que una pantalla necesita se SIRVE.

    El patrón que produjo la mayoría de los fallos de este repo es «el mismo dato
    en dos sitios y solo uno se actualiza». Las cuatro copias que vivían en el
    front (defaults de C, ejes de geometría, objetivos, orden de esquinas) ya
    habían divergido en dos casos. Este test falla si alguna vuelve a existir
    sólo en Python: si el API no lo sirve, la pantalla tendrá que inventárselo.
    """
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    from fv.models.builder import full_config
    from fv.sweeps.spec import GEOMETRY_AUTO, LOSS_WEIGHT_PARAMS, OBJECTIVES
    client = TestClient(create_app())

    # C: los defaults salen resueltos por full_config, la MISMA regla del builder
    nets = client.get("/networks").json()
    assert nets["defaults"] == full_config({})
    assert nets["defaults"]["channels"] == [16] * nets["defaults"]["n_layers"]

    # H/I: el vocabulario de ejes y objetivos, de las mismas constantes
    axes = client.get("/sweeps/axes").json()
    assert axes["objectives"] == sorted(OBJECTIVES)
    assert axes["geometry_auto"] == sorted(GEOMETRY_AUTO)
    assert axes["loss_weight_params"] == sorted(LOSS_WEIGHT_PARAMS)


def test_corner_order_travels_with_every_payload_indexed_by_it(world):
    """B/E/F: quien dibuja `y`/`scores` por POSICIÓN necesita saber qué significa
    cada fila. El orden es del dataset (manifest), no una constante del lector."""
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    from fv.metrics import CORNER_NAMES
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    client = TestClient(create_app())
    ds = world["dataset"]
    w = client.get(f"/window-datasets/{ds}/windows/0").json()
    assert w["corner_order"] == list(CORNER_NAMES)
    assert len(w["y"]) == len(w["corner_order"])     # una fila por esquina

    store = RunStore()
    train("co", ds, "n", TINY_NET, "r", Recipe(epochs=1, batch_size=32), store=store)
    s = client.get("/runs/co/diagnostics/summary").json()
    assert s["corner_order"] == list(CORNER_NAMES)


def test_contract_09_the_study_gate_is_not_laxer_than_the_sweep_gate(world):
    """⑨ en la puerta de I, no solo en la de H.

    `validate_plan` aceptaba objetivo `loss` con un eje de peso de la pérdida;
    `check_sweep` lo rechaza. El estudio moría dentro de `advance`, media cadena
    después — la trampa que R4 prohíbe. La pantalla lo tapaba no ofreciendo
    'loss' en su select, que es peor: escondía el hueco en vez de cerrarlo."""
    from fv.studies.driver import validate_plan
    from fv.sweeps.spec import check_sweep
    plan = {"window_dataset": world["dataset"], "base_recipe": "corta",
            "objective": "loss", "seeds": 1,
            "axes": [{"axis": "lambda_pos", "range": [0.5, 1.0]}]}
    codes = [p["code"] for p in validate_plan(plan)]
    assert "objective_varies_with_space" in codes
    # la misma combinación, en la puerta de H: el mismo código
    sweep_codes = [p["code"] for p in check_sweep(
        {"space": {"lambda_pos": [0.5, 1.0]}, "objective": "loss"})]
    assert "objective_varies_with_space" in sweep_codes
    # control: con un objetivo de tarea, el mismo plan pasa
    ok = dict(plan, objective="f1")
    assert "objective_varies_with_space" not in [p["code"] for p in validate_plan(ok)]


def test_contract_13_task_metric_needs_the_source(world):
    """⑬ E×A vía F — la métrica de tarea se puntúa contra los párrafos de la
    FUENTE, no contra las etiquetas de ventana.

    B guarda las imágenes pero no los párrafos verdaderos: son cosas distintas
    (una esquina dentro de una ventana ≠ un párrafo de la imagen). Así que la
    costura es `manifest["source_id"]` — y si la fuente no está, se falla con la
    razón, nunca se puntúa contra lo que sí hay a mano."""
    from fv import settings
    from fv.task import task_score
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunError, RunStore
    from fv.windows.store import WindowDatasetStore
    store = RunStore()
    train("t13", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)

    manifest = WindowDatasetStore().manifest(world["dataset"])
    out = task_score("t13", "val", store=store)
    assert out["source"] == manifest["source_id"]        # la verdad viene de A
    # una imagen del split = una unidad de muestra; el n viaja con el número
    assert out["images"] == len(WindowDatasetStore().split_map(
        world["dataset"])["val"])
    assert out["macro"]["sem"] is not None

    # sin la fuente no hay métrica de tarea (y el dataset de ventanas sigue ahí)
    (settings.local_sources_root() / "mini" / "labels.jsonl").unlink()
    with pytest.raises(RunError) as e:
        task_score("t13", "test", store=store)
    assert e.value.code == "task_needs_source"


def test_no_validation_split_refuses_to_train(world):
    from fv import settings
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunError, RunStore
    from fv.windows.extract import ExtractConfig, extract_windows
    cfg = ExtractConfig(source=world["source"], window_size=8, stride=6,
                        val_frac=0.0, test_frac=0.0, seed=1)
    extract_windows(cfg, settings.window_datasets_root() / "no-val")
    store = RunStore()
    with pytest.raises(RunError) as e:
        train("x", "no-val", "n", TINY_NET, "r", Recipe(epochs=1), store=store)
    assert e.value.code == "no_validation_split"
    assert not store.exists("x")   # the name was NOT reserved
