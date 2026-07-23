"""The HTTP gates: 400 before job AND before reserving the name; 409 semantics."""

import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import TINY_NET


@pytest.fixture()
def client(world):
    from fv.api.app import create_app
    return TestClient(create_app(), raise_server_exceptions=False)


def _wait_job(client, job_id, timeout=120):
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


def test_contract_01_http_400_before_job_and_no_run_created(world, client):
    _make_named(client)
    bad = dict(TINY_NET, N=20, c_frac=0.8, name="big")
    client.post("/networks", json=bad)
    r = client.post("/runs", json={"name": "never", "window_dataset": world["dataset"],
                                   "network": "big", "recipe": "quick"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "window_size_mismatch"
    runs = client.get("/runs").json()["runs"]
    assert all(x["name"] != "never" for x in runs)   # neither job nor run reserved


def test_full_flow_train_diagnose_predict(world, client):
    _make_named(client)
    r = client.post("/runs", json={"name": "api-run", "window_dataset": world["dataset"],
                                   "network": "tiny", "recipe": "quick"})
    assert r.status_code == 202
    job = _wait_job(client, r.json()["job"]["id"])
    assert job["status"] == "done", job.get("error")

    # detail with provenance; metrics incremental
    d = client.get("/runs/api-run").json()
    assert d["config"]["provenance"]["network"]["name"] == "tiny"
    m = client.get("/runs/api-run/metrics?since=0").json()
    assert m["next"] == 1 and m["records"][0]["epoch"] == 1
    m2 = client.get(f"/runs/api-run/metrics?since={m['next']}").json()
    assert m2["records"] == []   # never resent

    # duplicated name -> 409, untouched
    r2 = client.post("/runs", json={"name": "api-run", "window_dataset": world["dataset"],
                                    "network": "tiny", "recipe": "quick"})
    assert r2.status_code == 409

    # diagnostics: summary, gallery, evidence
    s = client.get("/runs/api-run/diagnostics/summary?split=val").json()
    assert s["windows"] > 0 and "detection" in s
    g = client.get("/runs/api-run/diagnostics/windows?split=val&limit=4").json()
    assert len(g["items"]) <= 4 and g["total"] > 0
    e = client.get("/runs/api-run/diagnostics/evidence?split=val").json()
    assert len(e["bands"]) == 4

    # introspection per branch + input view with coverage
    k = client.get("/runs/api-run/kernels").json()
    assert set(k["branches"]) == {"center", "periph"}
    assert k["branches"]["center"]["color"] == "diverging"
    iv = client.post("/runs/api-run/input-view",
                     json={"window_dataset": world["dataset"], "index": 0}).json()
    assert len(iv["channels"]) == 4
    fm = client.post("/runs/api-run/feature-maps",
                     json={"window_dataset": world["dataset"], "index": 0}).json()
    assert set(fm["branches"]) == {"center", "periph"}

    # predict: the three stages + knobs echoed in window units
    p = client.post("/runs/api-run/predict",
                    json={"source": world["source"], "index": 0}).json()
    assert set(p) >= {"raw", "corners", "paragraphs", "knobs", "truth"}
    assert p["knobs"]["window_size"] == 8

    # dataset in use -> 409 with the list
    del409 = client.delete(f"/window-datasets/{world['dataset']}")
    assert del409.status_code == 409
    assert "api-run" in del409.json()["detail"]["message"]


def test_contract_09_http_sweep_refused_before_reserving(world, client):
    _make_named(client)
    r = client.post("/sweeps", json={
        "name": "bad-sweep", "window_dataset": world["dataset"],
        "base_network": "tiny", "base_recipe": "quick",
        "space": {"lambda_pos": [0.1, 1.0]}, "objective": "loss"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "objective_varies_with_space"
    assert client.get("/sweeps/bad-sweep").status_code == 404  # nothing reserved


def test_sweep_over_http(world, client):
    _make_named(client)
    r = client.post("/sweeps", json={
        "name": "s1", "window_dataset": world["dataset"],
        "base_network": "tiny", "base_recipe": "quick",
        "space": {"lr": [0.001, 0.003]}, "objective": "f1",
        "budget": {"points": 2, "epochs": 1}})
    assert r.status_code == 202
    job = _wait_job(client, r.json()["job"]["id"], timeout=180)
    assert job["status"] == "done", job.get("error")
    trials = client.get("/sweeps/s1/trials").json()
    assert len(trials["trials"]) == 2
    assert trials["trials"][0]["value"] is not None
    # a sweep child cannot be deleted alone (would orphan it from its siblings)
    assert client.delete("/runs/s1-0000").status_code == 409
    # deleting the sweep cascades to its runs — the supported path, no orphan left
    d = client.delete("/sweeps/s1")
    assert d.status_code == 200
    assert set(d.json()["runs_deleted"]) == {"s1-0000", "s1-0001"}
    assert client.get("/sweeps/s1").status_code == 404
    assert client.get("/runs/s1-0000").status_code == 404


def test_network_validate_returns_dims_and_ranges(world, client):
    r = client.post("/networks/validate", json=TINY_NET).json()
    assert r["valid"] and r["trace"]["dims"]["center_out"] == 8
    assert r["ranges"]["k_center"]
    bad = client.post("/networks/validate",
                      json=dict(TINY_NET, pen_frac=0.45)).json()
    assert not bad["valid"]
    assert bad["problems"][0]["code"] == "penetration_too_large"
