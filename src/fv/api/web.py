"""The built front and the API on ONE port, behind ONE door.

WHY ONE PORT. In development this is two processes (vite :5173 + api :8010) and
that does not change. On a server it is a systemd unit, and there it has to be
one process: the launcher REFUSES two services out of the same repo -- they
would share the directory and its .env, and `selected_services` in
`do_droplet.py` dies on purpose -- and vite's dev server is a development tool,
not a server. So uvicorn serves `web/dist` at `/` and mounts the API at `/api`.
Single origin, so no CORS allowlist is involved either.

WHY `/api` AND NOT THE ROOT, which is the part that bites: the front's routes
and the API's resources COLLIDE -- `/runs`, `/sweeps`, `/studies`, `/networks`
and `/recipes` are each both a screen and a resource. Mounted at the root,
opening "Runs" in the browser would answer the API's JSON. `/api` is also
exactly the prefix `web/src/api.ts` already sends, and the one vite's proxy
strips in development, so nothing in the front changes.

WHY A DOOR. This runs on the public IP of a throwaway droplet, and the API
DELETES window datasets, runs, sweeps and studies without asking -- the very
artefacts that cost money to measure. An open port with no door is a
destructive capability published on the internet.

⚠ Loopback is trusted WITHOUT a token, and that is only valid while NOTHING
proxies to this process: with no proxy in front, `request.client.host` is the
real peer, so 127.0.0.1 means "already inside the machine" (that is how
`cerrable.mjs` asks it whether a training is running). If a reverse proxy is
ever added, this rule has to go with it -- behind a proxy every request looks
local, and the door would be open to everybody.

⚠ The token also travels as `?t=`, which is the only way to hand it to a phone
browser. That leaves it in the access log of this process; the redirect below
gets it out of the address bar as soon as the cookie is set, and the token is
per machine and dies with it.
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

COOKIE = "fv_web_token"
QUERY = "t"
HEADER = "x-fv-token"
COOKIE_MAX_AGE = 30 * 24 * 3600
LOOPBACK = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


class WebError(Exception):
    """Same {code, message, hint} shape the API uses for its own errors, so a
    failure to start reads like every other refusal in this project."""

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def dist_dir() -> Path:
    """Where the built front is. `FV_WEB_DIST` wins; `web/dist` of this repo is
    only the default (R4: the coupling is declared, discovery is the
    convenience)."""
    declared = os.environ.get("FV_WEB_DIST", "").strip()
    if declared:
        return Path(declared).expanduser()
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def is_loopback(host: str) -> bool:
    """True for the addresses that mean "this machine". Used for two different
    questions -- which host uvicorn may bind without a token, and which peer may
    skip the door -- and they must not drift apart."""
    return (host or "").strip().lower() in LOOPBACK


class _FrontFiles(StaticFiles):
    """`web/dist`, with the fallback a BrowserRouter needs.

    A deep link (`/runs/mi-run`) is resolved by the router in the browser, so a
    file that is not there has to answer `index.html` instead of 404.

    ⚠ Except under `/assets/`: there a 404 is a file that really is missing (a
    half-finished build), and answering the HTML would turn it into a 200 whose
    only symptom is a syntax error in the console -- the silent failure.
    """

    async def get_response(self, path, scope):
        # ⚠ StaticFiles RAISES the 404 (it does not return it), so catching the
        # exception is not belt-and-braces: without it the fallback never runs
        # and every deep link answers `{"detail":"Not Found"}`. The returned-404
        # branch stays for the cases where it does return one.
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or self._es_asset(path):
                raise
            return await super().get_response("index.html", scope)
        if response.status_code == 404 and not self._es_asset(path):
            return await super().get_response("index.html", scope)
        return response

    @staticmethod
    def _es_asset(path: str) -> bool:
        return path.lstrip("/").startswith("assets/")


def _authorized(request: Request, token: str) -> bool:
    for candidate in (request.query_params.get(QUERY),
                      request.cookies.get(COOKIE),
                      request.headers.get(HEADER)):
        if candidate and hmac.compare_digest(candidate, token):
            return True
    return False


def _closed_door(request: Request):
    """401 that says the same thing in the two languages this app speaks: a plain
    page for a browser, the {code, message, hint} envelope for everyone else.

    It keys off `Accept` and not off the `/api` prefix because the door is also
    installed on the bare API (`fv-api --host 0.0.0.0` with no front), where
    nothing is under `/api`.
    """
    if "text/html" not in request.headers.get("accept", ""):
        return JSONResponse(status_code=401, content={"detail": {
            "code": "web_token_required",
            "message": "esta instancia esta expuesta y pide token",
            "hint": "manda la cabecera X-FV-Token, o abre la URL con ?t=<token>"}})
    return HTMLResponse(status_code=401, content=(
        "<!doctype html><meta charset='utf-8'><title>foveal-vision</title>"
        "<body style='font:16px system-ui;padding:2rem;max-width:34rem'>"
        "<h1>Hace falta el token</h1>"
        "<p>Este servidor esta abierto a Internet, asi que la puerta pide token.</p>"
        "<p>Abre la direccion con <code>?t=&lt;token&gt;</code>. En la maquina, el "
        "token lo imprime <code>python3 scripts/web_app.py url</code>; desde "
        "Telegram, el ejecutor <code>fvweb</code>.</p></body>"))


def install_door(app: FastAPI, token: str) -> None:
    """The door, as its own function because it guards TWO apps: the whole web
    app and the bare API served without a front. One implementation, or the
    exposed-without-a-door case comes back through the second entry point."""

    @app.middleware("http")
    async def door(request: Request, call_next):
        peer = request.client.host if request.client else ""
        if is_loopback(peer):
            return await call_next(request)
        if not _authorized(request, token):
            return _closed_door(request)
        # Token accepted from the query string: set the cookie and bounce to the
        # same URL without it, so it stops travelling in links, bookmarks and
        # Referer headers from here on.
        if request.query_params.get(QUERY):
            rest = [(k, v) for k, v in request.query_params.multi_items() if k != QUERY]
            tail = "&".join(f"{k}={v}" for k, v in rest)
            target = request.url.path + (f"?{tail}" if tail else "")
            response = RedirectResponse(target, status_code=303)
        else:
            response = await call_next(request)
        response.set_cookie(COOKIE, token, max_age=COOKIE_MAX_AGE,
                            httponly=True, samesite="lax", path="/")
        return response


def create_web_app(dist: Path | None = None, token: str = "") -> FastAPI:
    """The outer app: `/api` is the API of `app.py`, everything else is the front.

    Refuses to build without a build (R2: either degrade with a declared default
    or fail BEFORE starting; a server that answers 404 to every screen is the
    third option, which is not allowed).
    """
    from fv.api.app import create_app

    dist = Path(dist) if dist is not None else dist_dir()
    if not (dist / "index.html").is_file():
        raise WebError(
            "web_build_missing",
            f"no hay front construido en {dist}",
            "constrúyelo con: cd web && npm ci && npm run build "
            "(o `python3 scripts/web_app.py preparar`)")

    # ⚠ EL CICLO DE VIDA VA AQUI, en el app EXTERIOR, y no en `create_app`.
    # Los eventos de una sub-app MONTADA no se propagan: un `shutdown` puesto
    # dentro del API no corre NUNCA cuando se sirve asi. Costo un rato el
    # 2026-09-01 -- el hook funcionaba con TestClient sobre el API a secas y no
    # en el servicio, que es donde importa.
    #
    # Al apagar se vuelcan las repeticiones de errores que quedan en memoria: sin
    # esto el contador va con hasta una ventana de retraso y un `systemctl
    # restart` se la lleva (medido: 30 peticiones fallidas se quedaban en 1).
    @asynccontextmanager
    async def ciclo(_app):
        yield
        from fv import errores                          # noqa: PLC0415
        n = errores.cerrar_ventanas()
        if n:
            print(f"errores: {n} ventana(s) de repeticiones volcada(s) al apagar",
                  flush=True)

    app = FastAPI(title="foveal-vision", lifespan=ciclo)
    if token:
        install_door(app, token)
    app.mount("/api", create_app())


    app.mount("/", _FrontFiles(directory=str(dist), html=True), name="front")
    return app
