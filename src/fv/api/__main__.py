"""fv-api: serve the backend. Explicit host/port so nothing collides."""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description="foveal-vision API server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    import uvicorn
    from fv.api.app import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
