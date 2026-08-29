"""fv-api: serve the backend. Explicit host/port so nothing collides.

With `--web` it also serves the built front, so the whole app is ONE process on
ONE port -- which is how it runs as a service on a server. See `fv/api/web.py`
for why the API moves under `/api` in that mode.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="foveal-vision API server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--web", action="store_true",
                    help="sirve tambien el front construido (web/dist) en el mismo puerto")
    ap.add_argument("--token", default=os.environ.get("FV_WEB_TOKEN", ""),
                    help="token de la puerta; obligatorio si --host no es local")
    args = ap.parse_args()

    import uvicorn
    from fv.api.web import WebError, create_web_app, install_door, is_loopback

    token = args.token.strip()
    # R2(b): binding outside loopback publishes an API that DELETES datasets,
    # runs, sweeps and studies to whoever scans the port. If the door has no
    # key, this refuses to start instead of starting open -- the failure has to
    # be at boot, in the log, and not the day someone finds the port.
    if not is_loopback(args.host) and not token:
        print(f"[web_token_required] --host {args.host} expone este API y no hay token.\n"
              "  El API borra datasets, runs, recorridos y estudios sin preguntar.\n"
              "  Ponle uno:  --token <secreto>   o  FV_WEB_TOKEN=<secreto>\n"
              "  Lo genera y lo guarda:  python3 scripts/web_app.py preparar\n"
              "  O sirve solo para esta maquina:  --host 127.0.0.1", file=sys.stderr)
        return 2

    if args.web:
        try:
            app = create_web_app(token=token)
        except WebError as e:
            print(f"[{e.code}] {e.message}\n  {e.hint}", file=sys.stderr)
            return 2
    else:
        from fv.api.app import create_app
        app = create_app()
        # The door guards this mode too: `--host 0.0.0.0` without `--web` is
        # still the whole API published, and the check above only proved a token
        # EXISTS. Forgetting this line is how "we asked for a token" turns into
        # "we asked for a token and let everyone in anyway".
        if token:
            install_door(app, token)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
