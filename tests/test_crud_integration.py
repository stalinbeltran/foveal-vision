"""CRUD across domains is INTERCONNECTED: deleting an object on one screen must
never silently break another (api.md R4 — refuse before, with reason+fix; never
a stack trace inside a job later). This pins the reference graph:

  - snapshot refs (run/sweep carry C/D VALUES inline): deleting C/D is allowed
    and does NOT break the run's detail/diagnostics.
  - by-name refs re-resolved later (a run/sweep/study RETRAINS on B by name; a
    study re-resolves base_recipe D by name at advance): deleting the target is
    REFUSED with the list of referrers.
"""

import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TINY_NET


@pytest.fixture()
def client(world):
    from fv.api.app import create_app
    return TestClient(create_app(), raise_server_exceptions=False)


def _wait_job(client, job_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.1)
    raise TimeoutError(job_id)


def _make_named(client):
    assert client.post("/networks", json=dict(TINY_NET, name="tiny")).status_code == 200
    assert client.post("/recipes", json={"name": "quick", "epochs": 1,
                                         "batch_size": 32}).status_code == 200


def _train(client, world, name="r", net="tiny", recipe="quick"):
    r = client.post("/runs", json={"name": name, "window_dataset": world["dataset"],
                                   "network": net, "recipe": recipe})
    assert r.status_code == 202, r.text
    job = _wait_job(client, r.json()["job"]["id"])
    assert job["status"] == "done", job.get("error")


# --------- snapshot refs: deleting C/D must NOT break the run that used them ---
def test_delete_network_and_recipe_keep_run_pages_working(world, client):
    _make_named(client)
    _train(client, world, name="r")
    # a run snapshots the network/recipe VALUES in its own config, so deleting
    # them is allowed and the run's detail + diagnostics keep working.
    assert client.delete("/networks/tiny").status_code == 200
    assert client.get("/runs/r").status_code == 200
    assert client.get("/runs/r/diagnostics/summary?split=val").status_code == 200
    assert client.get("/runs/r/kernels").status_code == 200
    # no study/sweep uses this recipe -> deleting it is allowed too, run still ok
    assert client.delete("/recipes/quick").status_code == 200
    assert client.get("/runs/r/diagnostics/summary?split=val").status_code == 200


# --------- by-name refs: deleting B/D a STUDY fixes must be REFUSED ------------
def test_dataset_delete_refused_while_a_study_fixes_it(world, client):
    _make_named(client)
    plan = {"name": "est", "window_dataset": world["dataset"], "base_recipe": "quick",
            "objective": "f1", "seeds": 1, "budget": {"epochs": 1},
            "axes": [{"axis": "n_layers", "range": [1, 2]}]}
    assert client.post("/studies", json=plan).status_code == 201
    # the study will RETRAIN on this dataset (by name) at advance: deleting it now
    # would break the study later -> refuse at the gate, naming the referrer.
    r = client.delete(f"/window-datasets/{world['dataset']}")
    assert r.status_code == 409, r.text
    assert "est" in r.json()["detail"]["message"]


def test_recipe_delete_refused_while_a_study_uses_it(world, client):
    _make_named(client)
    plan = {"name": "est", "window_dataset": world["dataset"], "base_recipe": "quick",
            "objective": "f1", "seeds": 1, "budget": {"epochs": 1},
            "axes": [{"axis": "n_layers", "range": [1, 2]}]}
    assert client.post("/studies", json=plan).status_code == 201
    # advance re-resolves base_recipe by name (generate.py) -> deleting it breaks
    # the study -> refuse, naming the study.
    r = client.delete("/recipes/quick")
    assert r.status_code == 409, r.text
    assert "est" in r.json()["detail"]["message"]
    # once the study is gone, the recipe frees up
    assert client.delete("/studies/est").status_code == 200
    assert client.delete("/recipes/quick").status_code == 200


# --------- a runless SWEEP also pins its dataset (guard must see sweeps) -------
def test_dataset_delete_refused_while_a_runless_sweep_fixes_it(world, client):
    # a sweep fixes B in its spec and retrains on it by name; even with no
    # surviving child runs, deleting B must be refused (store-level check).
    from fv.sweeps.store import SweepStore
    ss = SweepStore()
    ss.create("sw-runless", {"window_dataset": world["dataset"],
                             "base_network_value": {}, "base_recipe": "quick",
                             "base_recipe_value": {}, "points": []})
    r = client.delete(f"/window-datasets/{world['dataset']}")
    assert r.status_code == 409, r.text
    assert "sw-runless" in r.json()["detail"]["message"]


# --------- deleting a study's generated sweep must not break the study page ---
def test_delete_generated_sweep_leaves_study_readable(world, client):
    _make_named(client)
    plan = {"name": "est", "window_dataset": world["dataset"], "base_recipe": "quick",
            "objective": "f1", "seeds": 1, "budget": {"epochs": 1},
            "axes": [{"axis": "n_layers", "range": [1, 2]}]}
    assert client.post("/studies", json=plan).status_code == 201
    a = client.post("/studies/est/advance", json={})
    assert a.status_code == 202
    sweep = a.json()["step"]["sweep"]
    _wait_job(client, a.json()["job"]["id"])
    # delete the generated sweep (cascade) — the study keeps its recorded step,
    # and its screen must still render (reads only plan/progress, not the sweep).
    assert client.delete(f"/sweeps/{sweep}").status_code == 200
    st = client.get("/studies/est")
    assert st.status_code == 200
    assert st.json()["steps"][0]["sweep"] == sweep  # dangling name, but readable
