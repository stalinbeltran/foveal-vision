"""The web app served as ONE process: the door, the `/api` mount and the SPA.

Why these and not others (R10, effort by consequence of failure):

  · The door is the only thing between a public IP and an API that DELETES
    datasets, runs, sweeps and studies. A regression here is not a bug, it is a
    published capability -- so it is tested from the outside, per credential.
  · The `/api` mount is the collision that breaks silently: the front's routes
    and the API's resources share names (`/runs`, `/sweeps`...), so getting it
    wrong answers JSON where a screen should be, and only a human notices.
  · Refusing to start (no build / exposed with no token) is R2(b): if it ever
    degrades into starting anyway, the failure moves from boot to whenever
    somebody scans the port.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fv.api.web import COOKIE, HEADER, WebError, create_web_app, is_loopback

TOKEN = "secreto-de-prueba"


@pytest.fixture()
def dist(tmp_path):
    """A minimal `web/dist`: what `npm run build` leaves, in miniature."""
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (d / "assets" / "index-abc.js").write_text("console.log(1)", encoding="utf-8")
    return d


@pytest.fixture()
def abierta(dist):
    """Served with no token: what a laptop gets on 127.0.0.1."""
    return TestClient(create_web_app(dist=dist), raise_server_exceptions=False)


@pytest.fixture()
def cerrada(dist):
    """Served with a token, and the client seen as REMOTE (TestClient's default
    peer is `testclient`, which is not loopback)."""
    return TestClient(create_web_app(dist=dist, token=TOKEN),
                      raise_server_exceptions=False)


# ----------------------------------------------------------------- the door


def test_sin_token_no_se_entra(cerrada):
    for path in ("/", "/runs", "/api/runs"):
        assert cerrada.get(path).status_code == 401, path


def test_el_401_del_api_trae_code_message_hint(cerrada):
    # R4 of api.md: every error carries the three fields, and a client that
    # cannot read the reason cannot show the fix.
    detail = cerrada.get("/api/runs").json()["detail"]
    assert detail["code"] == "web_token_required"
    assert detail["message"] and detail["hint"]


def test_el_401_del_navegador_es_una_pagina(cerrada):
    r = cerrada.get("/", headers={"accept": "text/html"})
    assert r.status_code == 401
    assert "text/html" in r.headers["content-type"]


def test_la_cabecera_abre(cerrada):
    assert cerrada.get("/api/runs", headers={HEADER: TOKEN}).status_code == 200


def test_un_token_equivocado_no_abre(cerrada):
    assert cerrada.get("/api/runs", headers={HEADER: TOKEN + "x"}).status_code == 401
    assert cerrada.get("/api/runs", headers={HEADER: ""}).status_code == 401


def test_la_query_pone_la_cookie_y_se_quita_de_la_url(dist):
    # El movil solo puede traer el token en la URL. En cuanto entra, se guarda en
    # cookie y se rebota a la misma ruta sin el, para que deje de viajar en
    # marcadores, enlaces y Referer.
    c = TestClient(create_web_app(dist=dist, token=TOKEN), follow_redirects=False)
    r = c.get(f"/?t={TOKEN}")
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    assert r.cookies[COOKIE] == TOKEN
    # y ya con la cookie, sin token en la URL
    assert c.get("/api/runs").status_code == 200


def test_la_query_conserva_el_resto_de_parametros(dist):
    c = TestClient(create_web_app(dist=dist, token=TOKEN), follow_redirects=False)
    r = c.get(f"/runs?t={TOKEN}&filtro=abc")
    assert r.headers["location"] == "/runs?filtro=abc"


def test_loopback_entra_sin_token(dist):
    # No es una comodidad: es como `cerrable.mjs` pregunta desde la propia
    # maquina si hay un entrenamiento vivo antes de decir si se puede apagar.
    # ⚠ Solo vale mientras nada haga de proxy delante (ver fv/api/web.py).
    c = TestClient(create_web_app(dist=dist, token=TOKEN), client=("127.0.0.1", 9999))
    assert c.get("/api/runs").status_code == 200


def test_sin_token_no_hay_puerta(abierta):
    # El caso de desarrollo: nada cambia respecto a antes de esto.
    assert abierta.get("/api/runs").status_code == 200


@pytest.mark.parametrize("host,local", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True),
    ("0.0.0.0", False), ("159.203.188.195", False), ("", False)])
def test_que_cuenta_como_local(host, local):
    assert is_loopback(host) is local


# -------------------------------------------------- el reparto de las rutas


def test_el_api_vive_bajo_api_y_la_raiz_es_del_front(abierta):
    # La colision: `/runs` es una PANTALLA y un RECURSO. Servido en la raiz, el
    # API se comia la pantalla y solo se notaba abriendola.
    assert abierta.get("/api/runs").json() == {"runs": []}
    r = abierta.get("/runs")
    assert r.status_code == 200
    assert "<div id=root>" in r.text


def test_una_ruta_profunda_la_resuelve_el_front(abierta):
    # BrowserRouter: `/runs/mi-run` no es un fichero, lo resuelve el navegador.
    assert "<div id=root>" in abierta.get("/runs/mi-run").text


def test_un_asset_que_falta_es_404_y_no_el_html(abierta):
    # Devolver index.html aqui daria un 200 cuyo unico sintoma es un error de
    # sintaxis en la consola: el fallo silencioso.
    assert abierta.get("/assets/no-existe.js").status_code == 404
    assert abierta.get("/assets/index-abc.js").status_code == 200


def test_el_front_se_sirve_de_verdad(abierta):
    assert "<div id=root>" in abierta.get("/").text


# ------------------------------------------------------- negarse a arrancar


def test_sin_build_no_arranca(tmp_path):
    with pytest.raises(WebError) as e:
        create_web_app(dist=tmp_path / "no-hay-nada")
    assert e.value.code == "web_build_missing"
    assert "npm run build" in e.value.hint


def test_expuesto_sin_token_se_niega(monkeypatch, capsys):
    """La barrera que importa: `--host 0.0.0.0` sin token NO arranca."""
    from fv.api.__main__ import main
    monkeypatch.delenv("FV_WEB_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["fv-api", "--host", "0.0.0.0"])
    assert main() == 2
    assert "web_token_required" in capsys.readouterr().err


def test_expuesto_sin_front_construido_tampoco_arranca(monkeypatch, capsys, tmp_path):
    from fv.api.__main__ import main
    monkeypatch.setenv("FV_WEB_DIST", str(tmp_path / "vacio"))
    monkeypatch.setattr("sys.argv", ["fv-api", "--host", "0.0.0.0",
                                     "--token", TOKEN, "--web"])
    assert main() == 2
    assert "web_build_missing" in capsys.readouterr().err


def test_local_sin_token_si_arranca(monkeypatch):
    """No se pide token para 127.0.0.1: si lo pidiera, el flujo de desarrollo
    quedaria peor que antes de este cambio y nadie lo usaria."""
    from fv.api import __main__ as entry
    monkeypatch.delenv("FV_WEB_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["fv-api", "--host", "127.0.0.1"])
    arrancado = {}
    monkeypatch.setattr("uvicorn.run",
                        lambda app, **kw: arrancado.update(kw) or None)
    assert entry.main() == 0
    assert arrancado["host"] == "127.0.0.1" and arrancado["port"] == 8010


def test_el_api_pelado_expuesto_tambien_lleva_puerta(monkeypatch):
    """`--host 0.0.0.0` SIN `--web` sigue siendo el API entero publicado. Que
    exista un token no basta: hay que instalarle la puerta a esa app tambien."""
    from fv.api import __main__ as entry
    monkeypatch.setattr("sys.argv", ["fv-api", "--host", "0.0.0.0", "--token", TOKEN])
    servida = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: servida.update(app=app))
    assert entry.main() == 0
    c = TestClient(servida["app"], raise_server_exceptions=False)
    assert c.get("/runs").status_code == 401
    assert c.get("/runs", headers={HEADER: TOKEN}).status_code == 200


# ------------------------------------------------------ lo que ve el freno


def test_cerrable_puede_ver_los_jobs_por_loopback(dist):
    """`cerrable.mjs` decide si este server se puede apagar preguntando por
    /api/jobs desde 127.0.0.1. Si eso dejara de contestar, un entrenamiento
    lanzado desde la web app se perderia con un veredicto en verde."""
    c = TestClient(create_web_app(dist=dist, token=TOKEN), client=("127.0.0.1", 1))
    r = c.get("/api/jobs")
    assert r.status_code == 200
    assert isinstance(json.loads(r.text)["jobs"], list)
