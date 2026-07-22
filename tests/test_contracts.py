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
    rules = {
        "fovea": set(), "metrics": set(), "matrixview": set(),
        "validation": {"fovea"},
        "models": {"fovea"},
        "windows": {"datasets", "fovea", "metrics", "ioutils"},
        "inference": {"models", "fovea", "matrixview", "metrics"},
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
                    assert dep in allowed | {"settings"}, \
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
