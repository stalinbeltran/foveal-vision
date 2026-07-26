"""The task metric (contract ⑬): per IMAGE, truth from A, cached by knobs.

Every test builds its own tiny world (tests.md): a 10-image source, a window
dataset over it, and a 1-epoch run — the numbers do not have to be good, the
SEAMS have to hold.
"""

from __future__ import annotations

import pytest

from tests.conftest import TINY_NET


@pytest.fixture()
def trained(world):
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    store = RunStore()
    train("task-run", world["dataset"], "n", TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=store)
    return {"run": "task-run", "store": store, **world}


def _cache_files():
    from fv import settings
    d = settings.cache_root() / "task"
    return sorted(d.glob("*.json")) if d.exists() else []


def test_task_score_matches_a_hand_computed_case(trained):
    """The seam, not the function: macro and micro are what paragraph_f1 says
    about the SAME predictions, image by image."""
    from fv.datasets.loader import SourceDataset
    from fv.inference.checkpoint import load_model
    from fv.inference.predict import predict_image
    from fv.metrics import paragraph_f1
    from fv.task import task_score
    from fv.windows.store import WindowDatasetStore

    out = task_score(trained["run"], "val", store=trained["store"])

    model = load_model(trained["store"].path(trained["run"]) / "best.pt")
    source = SourceDataset(trained["source"])
    indices = WindowDatasetStore().split_map(trained["dataset"])["val"]
    assert out["images"] == len(indices)
    assert [r["index"] for r in out["per_image"]] == list(indices)

    by_hand = []
    for i in indices:
        s = source.sample_at(int(i))
        p = predict_image(model, s.load_image())
        pred = [(b["x0"], b["y0"], b["x1"], b["y1"]) for b in p["paragraphs"]]
        true = [b.bbox for b in s.blocks if b.kind == "paragraph"]
        by_hand.append(paragraph_f1(pred, true, 0.5))

    assert out["macro"]["f1"] == pytest.approx(
        sum(r["f1"] for r in by_hand) / len(by_hand))
    assert out["micro"]["tp"] == sum(r["tp"] for r in by_hand)
    assert out["micro"]["fp"] == sum(r["fp"] for r in by_hand)
    assert out["micro"]["fn"] == sum(r["fn"] for r in by_hand)
    # the band travels with the number: sd over IMAGES -> sem (metrica §3.3)
    assert out["macro"]["sem"] == pytest.approx(
        out["macro"]["sd"] / len(by_hand) ** 0.5)
    assert out["knobs"]["iou_threshold"] == 0.5
    assert out["checkpoint"] == "best.pt"


def test_task_score_refuses_a_changed_dataset(trained):
    """Fingerprint moved -> the split is another one, so nothing is scored."""
    from fv import settings
    from fv.ioutils import read_json_retrying, write_json_atomic
    from fv.task import task_score
    from fv.training.registry import RunError

    p = settings.window_datasets_root() / trained["dataset"] / "manifest.json"
    m = read_json_retrying(p)
    m["fingerprint"] = "sha256:" + "de" * 32
    write_json_atomic(p, m)

    with pytest.raises(RunError) as e:
        task_score(trained["run"], "val", store=trained["store"])
    assert e.value.code == "window_dataset_changed"
    assert not _cache_files()          # refused BEFORE inferring anything


def test_task_score_refuses_without_source(trained):
    """The truth lives in A. With A gone it fails with the reason — it does NOT
    fall back to the window labels, which measure another thing."""
    from fv import settings
    from fv.task import task_score
    from fv.training.registry import RunError

    (settings.local_sources_root() / "mini" / "labels.jsonl").unlink()
    with pytest.raises(RunError) as e:
        task_score(trained["run"], "val", store=trained["store"])
    assert e.value.code == "task_needs_source"
    assert "mini" in e.value.message and e.value.hint


def test_task_score_is_cached_by_knobs(trained):
    """Knobs are part of the key: changing one re-infers (unlike the diagnostics
    threshold, which only re-reads stored scores)."""
    from fv.task import task_score
    a = task_score(trained["run"], "val", store=trained["store"])
    assert a["cached"] is False
    b = task_score(trained["run"], "val", store=trained["store"])
    assert b["cached"] is True and b["macro"]["f1"] == a["macro"]["f1"]
    c = task_score(trained["run"], "val", threshold=0.9, store=trained["store"])
    assert c["cached"] is False
    assert len(_cache_files()) == 2
    d = task_score(trained["run"], "val", iou_threshold=0.75,
                   store=trained["store"])
    assert d["cached"] is False and len(_cache_files()) == 3


def test_task_score_scores_best_not_last(trained):
    """best.pt is the file that survives and the one Diagnostico/Predecir load;
    last.pt is another model. The number describes best.pt."""
    import torch
    from fv.task import task_score
    run_dir = trained["store"].path(trained["run"])
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    expected = task_score(trained["run"], "val", store=trained["store"])

    # a LOUD last.pt: exists-bias at +10 makes every window fire, so its
    # reconstruction cannot coincide with the quiet one by accident
    with torch.no_grad():
        for i in (0, 3, 6, 9):
            ckpt["model"]["head.bias"][i] += 10.0
    torch.save(ckpt, run_dir / "last.pt")
    torch.save(ckpt, run_dir / "loud.pt")

    again = task_score(trained["run"], "val", store=trained["store"])
    assert again["macro"]["f1"] == expected["macro"]["f1"]

    from fv.inference.checkpoint import load_model
    from fv.inference.predict import predict_image
    from fv.datasets.loader import SourceDataset
    from fv.windows.store import WindowDatasetStore
    loud = load_model(run_dir / "loud.pt")
    idx = WindowDatasetStore().split_map(trained["dataset"])["val"][0]
    s = SourceDataset(trained["source"]).sample_at(int(idx))
    assert len(predict_image(loud, s.load_image())["paragraphs"]) > 0  # control


def test_task_score_mean_iou_is_null_without_matches(trained):
    """No match -> null, never 0 (formatos.md §2). min_size huge = no box can be
    reconstructed, so there is nothing to match."""
    from fv.task import task_score
    out = task_score(trained["run"], "val", min_size=10_000.0,
                     store=trained["store"])
    assert out["mean_iou"] is None
    assert all(r["mean_iou"] is None for r in out["per_image"])
    assert out["micro"]["tp"] == 0 and out["macro"]["f1"] == 0.0


def test_task_score_does_not_serve_a_stale_number_after_retraining(trained):
    """§9.10 — the cache must not lie.

    That the key CONTAINS `ckpt.st_mtime_ns` and that the NUMBER changes when the
    weights change are two different claims, and only the second one is what the
    person looking at the screen cares about. Retrain in place (a new best.pt) and
    the answer must be recomputed, not served from the previous run's file.
    """
    import torch
    from fv.task import task_score
    before = task_score(trained["run"], "val", store=trained["store"])
    assert before["cached"] is False
    assert task_score(trained["run"], "val", store=trained["store"])["cached"] is True

    # the same run, trained again: same name, same dataset, other weights
    ckpt_path = trained["store"].path(trained["run"]) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for i in (0, 3, 6, 9):       # exists-bias up: every window fires now
            ckpt["model"]["head.bias"][i] += 10.0
    torch.save(ckpt, ckpt_path)

    after = task_score(trained["run"], "val", store=trained["store"])
    assert after["cached"] is False, "sirvio el numero viejo de un best.pt que ya no existe"
    assert len(_cache_files()) == 2   # the old entry survives, it just is not used
    # the NUMBER moved, not only the key. A 1-epoch model on a tiny world scores
    # F1 0.0 whatever it predicts, so the assertion rides on the count of
    # predictions, which is what the new weights actually changed: firing on
    # every window cannot leave the false positives where they were.
    assert after["micro"]["fp"] > before["micro"]["fp"]
    assert sum(r["fp"] for r in after["per_image"]) == after["micro"]["fp"]


def test_task_score_refuses_a_holdout_that_shares_the_source(trained):
    """§6: another B extracted from the SAME source is training data wearing
    another name — the guard is the whole point of a holdout."""
    from fv import settings
    from fv.task import task_score
    from fv.training.registry import RunError
    from fv.windows.extract import ExtractConfig, extract_windows

    cfg = ExtractConfig(source=trained["source"], window_size=8, stride=6,
                        val_frac=0.2, test_frac=0.2, seed=7)
    extract_windows(cfg, settings.window_datasets_root() / "fake-holdout")
    with pytest.raises(RunError) as e:
        task_score(trained["run"], "test", window_dataset="fake-holdout",
                   store=trained["store"])
    assert e.value.code == "holdout_shares_source"


# ---------------------------------------------------------------------------
# Fase 4 (§6): the holdout path. The DATA is not here yet — generating a proper
# holdout source is a decision (F11/F13) — but the code path is, and these fix
# it so the day the source exists nothing has to be re-derived.

@pytest.fixture()
def holdout(world):
    """A second source the run never saw, and a 100%-test B over it (§6.2)."""
    from fv import settings
    from fv.windows.extract import ExtractConfig, extract_windows
    from tests.conftest import make_source
    make_source(settings.local_sources_root(), "mini-holdout", count=6, seed=99)
    cfg = ExtractConfig(source="local/mini-holdout", window_size=8, stride=6,
                        val_frac=0.0, test_frac=1.0, seed=1)
    manifest = extract_windows(cfg,
                               settings.window_datasets_root() / "mini-holdout-b8")
    return {"source": "local/mini-holdout", "dataset": "mini-holdout-b8",
            "manifest": manifest}


def test_task_score_scores_another_dataset(trained, holdout):
    """§6.3: with `window_dataset=<other B>` the payload names THAT dataset and
    THAT source — the number must say what it measured, or it is unreadable."""
    from fv.task import task_score
    out = task_score(trained["run"], "test", window_dataset=holdout["dataset"],
                     store=trained["store"])
    assert out["window_dataset"] == holdout["dataset"]
    assert out["source"] == holdout["source"]
    assert out["images"] == 6              # the whole holdout, not a slice
    assert len(out["per_image"]) == 6
    # and it is a different measurement from the run's own val, not the same
    # cache entry wearing another label
    own = task_score(trained["run"], "val", store=trained["store"])
    assert own["window_dataset"] == trained["dataset"]
    assert own["source"] == trained["source"]


def test_holdout_dataset_can_be_100_percent_test(holdout):
    """§6.2: val_frac 0 + test_frac 1 puts EVERY image in test, and that dataset
    then refuses to train — here a virtue, not an obstacle: the fuga is made
    physically impossible instead of trusted to a flag."""
    from fv.validation import check_run
    from fv.windows.store import WindowDatasetStore
    m = holdout["manifest"]
    assert m["windows_per_split"]["train"] == 0
    assert m["windows_per_split"]["val"] == 0
    assert m["windows_per_split"]["test"] > 0
    smap = WindowDatasetStore().split_map(holdout["dataset"])
    assert len(smap["test"]) == 6 and not smap["train"] and not smap["val"]

    problems = check_run(WindowDatasetStore().manifest(holdout["dataset"]), TINY_NET)
    assert "no_validation_split" in {p["code"] for p in problems}


def test_task_score_on_holdout_ignores_the_run_fingerprint(trained, holdout):
    """That fingerprint protects the RUN's split (the images its best.pt never
    saw). It says nothing about another B, so rebuilding the run's own dataset
    must not block the holdout number."""
    from fv import settings
    from fv.ioutils import read_json_retrying, write_json_atomic
    from fv.task import task_score
    from fv.training.registry import RunError

    p = settings.window_datasets_root() / trained["dataset"] / "manifest.json"
    m = read_json_retrying(p)
    m["fingerprint"] = "sha256:" + "ab" * 32
    write_json_atomic(p, m)

    # its own val is refused ...
    with pytest.raises(RunError) as e:
        task_score(trained["run"], "val", store=trained["store"])
    assert e.value.code == "window_dataset_changed"
    # ... and the holdout still answers
    out = task_score(trained["run"], "test", window_dataset=holdout["dataset"],
                     store=trained["store"])
    assert out["images"] == 6 and out["source"] == holdout["source"]
