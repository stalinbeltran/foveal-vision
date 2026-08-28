"""One test per contract of organizacion.md §2, named by number (tests.md §3)."""

import inspect
import sys

import numpy as np
import pytest

from tests.conftest import TINY_NET


def test_contract_01_window_size_mismatch_is_refused_before_reserving(world):
    from fv.validation import check_run
    from fv.windows.store import WindowDatasetStore
    manifest = WindowDatasetStore().manifest(world["dataset"])
    bad = dict(TINY_NET, fovea_px=16)  # fovea 16 vs window 8
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
    assert prov["network"]["value"]["fovea_px"] == TINY_NET["fovea_px"]
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
    from fv.fovea import dims_of
    from fv.models.builder import network_trace
    cfg = {"fovea_px": 16, "border_px": 4, "border_reduce": 2,
           "overlap_fovea_px": 2, "overlap_border_px": 0, "n_layers": 2,
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
    from fv.fovea import dims_of
    from fv.models.builder import build_model
    cfg = dict(TINY_NET); cfg.pop("ch1"); cfg.pop("ch2")
    cfg.update(n_layers=3, channels=[4, 8, 8])
    model = build_model(cfg)
    assert len(model.center_convs) == 3 and len(model.periph_convs) == 3
    n = dims_of(cfg).N
    out = model(torch.zeros(1, 1, n, n))
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
    assert model.dims.fovea_px == 8         # geometry included
    out = model(torch.zeros(1, 1, 12, 12))
    assert out.shape == (1, 4, 3)


def test_contract_05_dataloader_and_inference_build_the_same_view(world):
    """The seam, not the function: both sides call THE SAME fv.fovea and the
    views are bit-identical for the same window."""
    import fv.inference.predict as predict_mod
    import fv.windows.dataset as dataset_mod
    from fv.fovea import build_view, dims_of
    assert dataset_mod.build_view is predict_mod.build_view is build_view

    from fv.windows.store import WindowDatasetStore
    from fv.windows.dataset import FoveatedWindowDataset
    arrays = WindowDatasetStore().arrays(world["dataset"])
    dims = dims_of(TINY_NET)
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
    # ...and split.json (indexes of A) says the SAME as the per-window array
    smap = store.split_map(world["dataset"])
    for i, name in enumerate(("train", "val", "test")):
        from_npz = {int(x) for x in
                    np.unique(arrays["sample_idx"][arrays["split"] == i])}
        assert set(smap[name]) == from_npz, f"split.json vs .npz disagree on {name}"
    # the SPLIT seed is B's and only B's: D's seed (the replica axis) is not an
    # input to the extractor at all, so N replicas share one val by construction.
    from dataclasses import fields
    assert "seed" in {f.name for f in fields(ExtractConfig)}
    from fv.training.recipe import Recipe
    recipe_only = {f.name for f in fields(Recipe)} - {f.name for f in fields(ExtractConfig)}
    assert "lr" in recipe_only          # control: the sets really are different
    src = inspect.getsource(extract_windows) + inspect.getsource(
        sys.modules["fv.windows.extract"]._assign_splits)
    assert "Recipe" not in src, "the extractor must not know about D's recipe"


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

def test_the_reparameterisation_does_not_move_a_single_weight():
    """2026-08-25: stating the geometry in real px must be a RENAME, not a
    change. Module names are untouched, so a checkpoint trained under the old
    spelling loads strict into a net built from the new one, and both forward
    bit-identically. The param count is the documented golden number of the L4
    base that was training when this landed (168652)."""
    import io
    import torch
    from fv.models.builder import build_model, full_config
    legacy = {"N": 20, "c_frac": 0.8, "d": 2, "pen_frac": 0.1, "n_layers": 4,
              "channels": [16, 16, 16, 16]}
    canon = {"fovea_px": 16, "border_px": 4, "border_reduce": 2,
             "overlap_fovea_px": 2, "overlap_border_px": 0, "n_layers": 4,
             "channels": [16, 16, 16, 16]}
    torch.manual_seed(0); a = build_model(full_config(legacy))
    torch.manual_seed(0); b = build_model(full_config(canon))
    assert {k: tuple(v.shape) for k, v in a.state_dict().items()}         == {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sum(v.numel() for v in a.state_dict().values()) == 168652
    buf = io.BytesIO(); torch.save(a.state_dict(), buf); buf.seek(0)
    b.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    x = torch.randn(2, 1, 20, 20)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_dropout_off_is_the_net_that_was_already_on_disk():
    """`dropout` (2026-08-27) is a NEW field of C, and every config, checkpoint
    and run already on disk was written without it. Off (the default) it must
    therefore be a no-op in every sense that can be observed: the state_dict is
    the same set of tensors with the same shapes and the same count, a checkpoint
    saved before the field existed loads `strict`, and the forward is bit
    identical in BOTH modes. nn.Dropout holds no parameters, so this holds by
    construction — it is asserted because 'by construction' is what silently
    stops being true.

    The golden number is the same 168652 of the L4 base above: adding the field
    must not add a weight."""
    import io
    import torch
    from fv.models.builder import build_model, full_config
    base = {"fovea_px": 16, "border_px": 4, "border_reduce": 2,
            "overlap_fovea_px": 2, "overlap_border_px": 0, "n_layers": 4,
            "channels": [16, 16, 16, 16]}
    torch.manual_seed(0); a = build_model(full_config(base))               # no field
    torch.manual_seed(0); b = build_model(full_config(base | {"dropout": 0.0}))
    assert full_config(base)["dropout"] == 0.0        # absent == off, never invented
    assert {k: tuple(v.shape) for k, v in a.state_dict().items()} ==            {k: tuple(v.shape) for k, v in b.state_dict().items()}
    assert sum(v.numel() for v in b.state_dict().values()) == 168652
    buf = io.BytesIO(); torch.save(a.state_dict(), buf); buf.seek(0)
    b.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    x = torch.randn(4, 1, 20, 20)
    for mode in (True, False):
        a.train(mode); b.train(mode)
        torch.manual_seed(7); ya = a(x)
        torch.manual_seed(7); yb = b(x)
        assert torch.equal(ya, yb), f"dropout=0.0 no es identidad en train={mode}"


def test_dropout_on_acts_in_train_and_never_in_eval():
    """The other half: a non-zero dropout has to actually do something, and has
    to do it ONLY while training. A dropout still active at eval would make every
    val number — and therefore best.pt and every sweep ranking — depend on a coin
    flip, which is the kind of failure that produces a perfectly credible table.

    The training loop already flips model.train()/model.eval() around each epoch
    (loop.py), so asserting the module honours the mode is asserting the seam."""
    import torch
    from fv.models.builder import build_model, full_config
    cfg = full_config({"fovea_px": 16, "border_px": 4, "border_reduce": 2,
                       "overlap_fovea_px": 2, "overlap_border_px": 0,
                       "n_layers": 2, "dropout": 0.5})
    torch.manual_seed(0)
    model = build_model(cfg)
    x = torch.randn(8, 1, 20, 20)
    model.train()
    torch.manual_seed(1); y1 = model(x)
    torch.manual_seed(2); y2 = model(x)
    assert not torch.equal(y1, y2), "dropout=0.5 no cambia nada en train"
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x)), "dropout sigue activo en eval"


def test_dropout_out_of_range_is_refused_at_the_gate():
    """R4: a probability outside [0, 1) is refused with reason and hint BEFORE a
    run name is reserved, not by a torch exception inside the job. 1.0 is the one
    that matters: nn.Dropout accepts it and zeroes EVERYTHING, so the net would
    train on nothing and still write a perfectly normal-looking run."""
    from fv.validation import check_network
    ok = {"fovea_px": 16, "border_px": 4, "border_reduce": 2,
          "overlap_fovea_px": 2, "overlap_border_px": 0, "n_layers": 2}
    assert check_network(ok) == []
    assert check_network(ok | {"dropout": 0.25}) == []
    for bad in (1.0, -0.1, 1.5, "mucho"):
        problems = check_network(ok | {"dropout": bad})
        assert any(p["code"] == "dropout_out_of_range" for p in problems), bad
        assert all(p.get("hint") for p in problems)


def test_dropout_is_a_sweepable_axis_of_c():
    """The reason this was implemented at all: it had to become an axis. Before
    2026-08-27 `dropout` was in three documents and in no dict, so `full_config`
    dropped it and the N points of a sweep would have trained the SAME net while
    the table said otherwise. Assert the axis both passes the gate and produces
    points that actually differ."""
    from fv.sweeps.spec import check_sweep, expand_points
    from fv.models.builder import full_config
    base = full_config({"fovea_px": 16, "border_px": 4, "border_reduce": 2,
                        "overlap_fovea_px": 2, "overlap_border_px": 0,
                        "n_layers": 2})
    spec = {"space": {"dropout": [0.0, 0.1, 0.25]}, "objective": "f1"}
    assert check_sweep(spec) == []
    valid, discarded = expand_points(spec, base)
    assert discarded == []
    assert [p["network"]["dropout"] for p in valid] == [0.0, 0.1, 0.25]
    # and it is C, not D: it must not leak into the recipe overrides
    assert all(p["recipe_overrides"] == {} for p in valid)


def test_every_run_on_disk_still_resolves_its_geometry():
    """The migration's other half: 478 runs were written with the old spelling
    and their provenance is the only record of the net that made them. A reader
    that cannot read them turns history into noise."""
    import json
    import pathlib as _pl
    from fv.fovea import dims_of
    runs = _pl.Path("runs")
    if not runs.exists():
        pytest.skip("sin runs en disco (artefactos ignorados por git)")
    seen = 0
    for r in sorted(runs.iterdir()):
        cfg_path = r / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        net = (cfg.get("provenance", {}).get("network", {}).get("value")
               or cfg.get("network"))
        if not net:
            continue
        dims_of(net)          # raises if the old spelling stopped being readable
        seen += 1
    assert seen > 0


# ------------------------------------- el agrupamiento por estudio al escribir
#
# Los artefactos de estudio viven en `foveal-vision-data` agrupados por mes. El
# mes lo elige EL ESTUDIO al crearse y lo hereda todo lo suyo: el mes AGRUPA
# para poder leer el directorio, no fecha cada run. Sin test esto es un
# comentario, y es justo lo que se pidio.

def test_a_study_keeps_its_sweeps_and_runs_in_one_month(monkeypatch, tmp_path):
    """Un recorrido creado el mes SIGUIENTE se queda con su estudio.

    Es la razon de ser del agrupamiento: un mismo estudio repartido en dos
    carpetas por el mero paso de la medianoche seria lo que estas carpetas
    existen para evitar.
    """
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    from fv import artefactos
    from fv.ioutils import write_json_atomic
    from fv.studies.store import StudyStore
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore

    # un estudio archivado en JULIO (como si lo hubiera dejado la migracion)
    d = tmp_path / "datos" / "2026" / "07-julio" / "studies" / "viejo"
    d.mkdir(parents=True)
    write_json_atomic(d / "plan.json", {"axes": []})
    assert artefactos.mes_del_estudio("viejo") == "2026/07-julio"

    # ... cuyo recorrido se crea HOY (agosto): hereda julio, no el mes actual
    SweepStore().create("viejo-s1-lr", {"study": "viejo", "window_dataset": "x"})
    sw = SweepStore().path("viejo-s1-lr")
    assert "07-julio" in str(sw), sw

    # ... y su run vive DENTRO del recorrido, luego tambien en julio
    RunStore().create("viejo-s1-lr-0000-a", {"provenance": {"sweep": "viejo-s1-lr"}})
    run = RunStore().path("viejo-s1-lr-0000-a")
    assert run.parent.parent == sw, run
    assert "07-julio" in str(run), run

    # y todo sigue siendo visible por su nombre, en los tres listados
    assert "viejo" in [s["name"] for s in StudyStore().list()]
    assert "viejo-s1-lr" in [s["name"] for s in SweepStore().list()]


def test_a_run_of_a_sweep_is_stored_inside_it(monkeypatch, tmp_path):
    """La relacion recorrido-runs es estructura, no un prefijo en el nombre."""
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    from fv.studies.store import StudyStore
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore
    StudyStore().create("est", {"axes": []}, {"steps": []})
    SweepStore().create("est-s0-lr", {"study": "est", "window_dataset": "x"})
    d = RunStore().create("est-s0-lr-0000-a", {"provenance": {"sweep": "est-s0-lr"}})
    assert d.parent.name == "runs" and d.parent.parent.name == "est-s0-lr"
    # un run suelto (un benchmark) NO se inventa un recorrido ni un mes
    loose = RunStore().create("bench-1", {"provenance": {}})
    assert loose.parent.name == "runs" and loose.parent.parent.name != "est-s0-lr"


def test_deleting_a_study_still_finds_its_grouped_sweeps(monkeypatch, tmp_path):
    """La cascada de borrado ve los recorridos agrupados.

    `used_by_study` miraba solo la raiz plana: con el recorrido agrupado bajo el
    mes de su estudio, borrar el estudio los dejaba HUERFANOS — el bug que la
    cascada existe para evitar.
    """
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    from fv.studies.store import StudyStore
    from fv.sweeps.store import SweepStore
    StudyStore().create("est", {"axes": []}, {"steps": []})
    SweepStore().create("est-s0-lr", {"study": "est", "window_dataset": "x"})
    assert SweepStore().used_by_study("est") == ["est-s0-lr"]


def test_nothing_new_is_ever_written_to_the_flat_root(monkeypatch, tmp_path):
    """El agujero que dejo `do-t` en la raiz plana el 2026-08-28.

    Los `scripts/estudio_*.py` nombran su estudio en el `spec.json` y NO crean
    el `studies/<nombre>/` -- solo el motor OAT del API lo crea. Como el mes se
    buscaba unicamente por ese directorio, un estudio lanzado por script (o sea,
    todos los que se han medido aqui) no tenia mes, y el recorrido y sus runs
    caian en `<data>/sweeps/` y `<data>/runs/`, sin fecha y sin un solo aviso.
    """
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore

    # exactamente lo que hace estudio_dropout.py: un estudio que nadie ha creado
    sw = SweepStore().create("do-t", {"study": "dropout-2026-08-28",
                                      "window_dataset": "x"})
    run = RunStore().create("do-t-0000-a", {"provenance": {"sweep": "do-t"}})

    datos = tmp_path / "datos"
    assert not (datos / "sweeps").exists(), "el recorrido cayo en la raiz plana"
    assert not (datos / "runs").exists(), "el run cayo en la raiz plana"
    assert sw.parent.parent.name.endswith("agosto") or sw.parent.parent.parent.name
    assert sw.relative_to(datos).parts[0].isdigit(), sw   # <anio>/<mes>/sweeps/...
    assert run.parent.parent == sw, run                   # y dentro de SU recorrido
    # y se sigue encontrando por su nombre, que es para lo que sirve la cascada
    assert SweepStore().path("do-t") == sw
    assert RunStore().path("do-t-0000-a") == run


def test_a_study_without_an_artifact_still_keeps_one_month(monkeypatch, tmp_path):
    """El segundo recorrido de un estudio hereda el mes del primero.

    Es el invariante que el agrupamiento existe para dar ("un estudio no se
    reparte entre carpetas de mes"), y hay que conservarlo tambien cuando el
    estudio no es un directorio: la fase 2 de un estudio se lanza dias despues
    de la fase 1 y puede caer al otro lado de la medianoche del dia 1.
    """
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    from fv import artefactos
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore

    monkeypatch.setattr(artefactos, "mes_actual", lambda: "2026/08-agosto")
    tanteo = SweepStore().create("do-t", {"study": "do-2026", "window_dataset": "x"})
    assert "2026/08-agosto" in str(tanteo)

    # ...y la fase 2 se lanza en SEPTIEMBRE: se queda con su estudio
    monkeypatch.setattr(artefactos, "mes_actual", lambda: "2026/09-septiembre")
    completo = SweepStore().create("do-v", {"study": "do-2026", "window_dataset": "x"})
    assert "2026/08-agosto" in str(completo), completo
    run = RunStore().create("do-v-0000-a", {"provenance": {"sweep": "do-v"}})
    assert "2026/08-agosto" in str(run), run
    assert not (tmp_path / "datos" / "2026" / "09-septiembre").exists()

    # y el artefacto del estudio, si alguien lo crea DESPUES, va con los suyos
    from fv.studies.store import StudyStore
    assert "2026/08-agosto" in str(StudyStore().create("do-2026", {"axes": []}, {}))


def test_a_loose_run_gets_a_month_but_never_an_invented_sweep(monkeypatch, tmp_path):
    """Un benchmark no pertenece a ningun recorrido, pero si a un mes.

    "Un huerfano no se inventa un padre" es sobre el RECORRIDO, no sobre la
    fecha: el README del repo de datos ya coloca los sueltos en `<mes>/runs/`.
    """
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    from fv.training.registry import RunStore
    d = RunStore().create("bench-foveal-1", {"provenance": {}})
    assert d.parent.name == "runs"
    assert d.parent.parent.parent.name.isdigit(), d      # <anio>/<mes>/runs/<run>
    assert not (tmp_path / "datos" / "runs").exists()


def test_what_is_already_flat_stays_visible(monkeypatch, tmp_path):
    """Lo escrito en la raiz plana se sigue leyendo mientras no se migre.

    `path()` resuelve desde la forma PLANA, no desde `destino()`. Pasarle el
    destino -- que desde el arreglo siempre esta fechado -- haria invisible de
    golpe todo lo que ya hay escrito ahi: el tanteo `do-t` que estaba corriendo
    el 2026-08-28 mientras se escribia esto, entre otras cosas.
    """
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path / "datos"))
    from fv.ioutils import write_json_atomic
    from fv.sweeps.store import SweepStore
    from fv.training.registry import RunStore

    plano_sw = tmp_path / "datos" / "sweeps" / "viejo"
    plano_sw.mkdir(parents=True)
    write_json_atomic(plano_sw / "spec.json", {"study": "x", "window_dataset": "y"})
    plano_run = tmp_path / "datos" / "runs" / "viejo-0000-a"
    plano_run.mkdir(parents=True)
    write_json_atomic(plano_run / "config.json", {"provenance": {"sweep": "viejo"}})

    assert SweepStore().path("viejo") == plano_sw
    assert SweepStore().exists("viejo")
    assert "viejo" in [s["name"] for s in SweepStore().list()]
    assert RunStore().path("viejo-0000-a") == plano_run
    # y un run nuevo de ESE recorrido se queda con el: el mes lo separaria
    assert RunStore().destino("viejo-0001-b",
                              {"provenance": {"sweep": "viejo"}}).parent == plano_run.parent


def test_the_archive_index_is_cached_per_root(monkeypatch, tmp_path):
    """La cache del indice va por RAIZ, no global.

    Cacheada sin argumentos, el primer repo mirado se quedaba pegado y un test
    apuntando a un temporal seguia resolviendo contra el repo de datos REAL.
    """
    import json
    from fv import artefactos
    a, b = tmp_path / "a", tmp_path / "b"
    for raiz, nombre in ((a, "run-de-a"), (b, "run-de-b")):
        raiz.mkdir()
        (raiz / "index.json").write_text(
            json.dumps({"runs": {nombre: {"path": f"2026/08-agosto/runs/{nombre}"}}}),
            encoding="utf-8")
    monkeypatch.setenv("FV_DATA_ROOT", str(a))
    assert "run-de-a" in artefactos.nombres("runs", a / "runs")
    monkeypatch.setenv("FV_DATA_ROOT", str(b))
    nombres_b = artefactos.nombres("runs", b / "runs")
    assert "run-de-b" in nombres_b and "run-de-a" not in nombres_b


# ------------------------------- recoger lo que quedo plano (recoger_planos.py)
#
# El arreglo de arriba deja de ESCRIBIR plano; no mueve lo ya escrito. Eso lo
# hace `scripts/recoger_planos.py`, y estas son las dos cosas que si se rompen
# cuestan datos ya pagados: mover bajo los pies de una flota viva, y fechar por
# el mtime en vez de por el JSON del propio artefacto.

from pathlib import Path  # noqa: E402  (para los helpers de abajo)


def _recoger():
    import importlib.util
    ruta = Path(__file__).resolve().parents[1] / "scripts" / "recoger_planos.py"
    spec = importlib.util.spec_from_file_location("_recoger_test", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _datos_planos(raiz: Path, estado_ultimo: str = "done") -> Path:
    """Una replica de lo que `do-t` dejo plano el 2026-08-28, con sus fechas."""
    import json
    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / "index.json").write_text(
        json.dumps({"sweeps": {}, "runs": {}, "studies": {}}), encoding="utf-8")
    sw = raiz / "sweeps" / "do-t"
    sw.mkdir(parents=True)
    (sw / "spec.json").write_text(
        json.dumps({"study": "dropout-2026-08-28"}), encoding="utf-8")
    (sw / "state.json").write_text(
        json.dumps({"status": "queued", "updated_at": 1787880431.2}), encoding="utf-8")
    for i, (ts, estado) in enumerate([(1787882986.2, "done"),
                                      (1787882671.4, estado_ultimo)]):
        r = raiz / "runs" / f"do-t-000{i}-a"
        r.mkdir(parents=True)
        (r / "config.json").write_text(
            json.dumps({"provenance": {"sweep": "do-t"}}), encoding="utf-8")
        (r / "status.json").write_text(
            json.dumps({"status": estado, "updated_at": ts}), encoding="utf-8")
    # un benchmark suelto de JULIO: su mes es el suyo, no el de hoy
    b = raiz / "runs" / "bench-foveal-9"
    b.mkdir(parents=True)
    (b / "config.json").write_text(json.dumps({"provenance": {}}), encoding="utf-8")
    (b / "status.json").write_text(
        json.dumps({"status": "done", "updated_at": 1783000000.0}), encoding="utf-8")
    return raiz


def test_recoger_dates_by_the_json_not_by_today(monkeypatch, tmp_path):
    """Lo recogido va al mes en que se GENERO, leido de su propio JSON.

    Es la regla del README del repo de datos, y la razon es que el mtime en un
    clon limpio es la fecha del checkout: fechar por ahi movería en bloque todo
    el archivo al mes en que alguien clonó.
    """
    datos = _datos_planos(tmp_path / "datos")
    monkeypatch.setenv("FV_DATA_ROOT", str(datos))
    mod = _recoger()
    monkeypatch.setattr(mod.artefactos, "mes_actual", lambda: "2031/01-enero")

    movimientos, _ = mod.planear(datos)
    destinos = {o.name: d.relative_to(datos).as_posix() for o, d in movimientos}
    assert destinos["do-t"] == "2026/08-agosto/sweeps/do-t"
    # el run vive DENTRO de su recorrido, y hereda su mes
    assert destinos["do-t-0000-a"] == "2026/08-agosto/sweeps/do-t/runs/do-t-0000-a"
    # y el suelto se queda en SU julio, no en el mes de hoy ni en el del sweep
    assert destinos["bench-foveal-9"] == "2026/07-julio/runs/bench-foveal-9"
    assert not any("2031" in v for v in destinos.values())


def test_recoger_refuses_while_something_is_still_alive(monkeypatch, tmp_path):
    """No se mueve un recorrido vivo.

    Mover el directorio bajo los pies de quien escribe deja los runs a medias en
    el sitio viejo y al escritor apuntando a un sitio que ya no lee nadie: datos
    ya pagados, perdidos sin un solo error.
    """
    datos = _datos_planos(tmp_path / "datos", estado_ultimo="running")
    monkeypatch.setenv("FV_DATA_ROOT", str(datos))
    motivos = _recoger().motivos_para_no_tocar(datos)
    assert any("running" in m for m in motivos), motivos
    # ...y con todo terminado, adelante
    limpio = _datos_planos(tmp_path / "limpio")
    assert _recoger().motivos_para_no_tocar(limpio) == []


def test_recoger_does_not_mistake_its_own_shell_for_a_fleet(monkeypatch, tmp_path):
    """`pgrep -f` casa con la linea de comando entera, incluida la del shell que
    lanzo esto. Un falso "hay una flota viva" bloquea la recogida para siempre,
    que es peor que no comprobar: yo no soy una flota, y mi padre tampoco."""
    import os
    datos = _datos_planos(tmp_path / "datos")
    monkeypatch.setenv("FV_DATA_ROOT", str(datos))
    mod = _recoger()
    mios = mod._yo_y_mis_padres()
    assert str(os.getpid()) in mios
    assert str(os.getppid()) in mios
    # el pgrep se lo cree todo: lo que filtra es la exclusion de arriba
    monkeypatch.setattr(mod.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": str(os.getpid())})())
    assert mod.motivos_para_no_tocar(datos) == []
