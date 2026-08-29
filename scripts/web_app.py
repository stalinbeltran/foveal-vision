#!/usr/bin/env python3
"""La web app como SERVICIO: prepararla, servirla, mirarla y cerrarla.

Por que existe
--------------
La app se documentaba como dos terminales en la maquina de desarrollo (backend
`:8010` + vite `:5173`). En un server desechable eso no sirve: nadie va a abrir
dos terminales despues de cada `lanzar launch dev`, y vite es una herramienta de
desarrollo. Aqui la app es UN proceso -- `fv.api --web`, que sirve `web/dist` y
monta el API en `/api` (el porque, en `src/fv/api/web.py`).

Este script es lo que el descriptor de servicio del lanzador invoca
(`services/foveal-vision-web.json`: `install` -> `preparar`, `start` ->
`servir`), y es tambien lo que se ejecuta a mano en una maquina que ya esta
viva. Un solo sitio decide como se prepara y como se arranca, para que la
maquina de hoy y la que la sustituya no diverjan.

⚠ `preparar` corre con `python3` del sistema (el venv puede no existir todavia):
de aqui para abajo, solo biblioteca estandar. `servir` corre con
`.venv/bin/python`, que es el unico subcomando que importa `fv`.

    python3 scripts/web_app.py preparar     # venv + npm ci + build + token + puerto
    python3 scripts/web_app.py estado       # esta viva? se llega? donde?
    python3 scripts/web_app.py url          # la URL con el token, para pegar
    python3 scripts/web_app.py cerrar       # cierra el puerto en el firewall
    .venv/bin/python scripts/web_app.py servir --host 0.0.0.0 --port 8010

Codigo de salida: 0 si lo que se pedia esta en pie, 1 si no.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = "foveal-vision-web"
PUERTO = int(os.environ.get("FV_WEB_PORT", "8010"))
# El token vive FUERA del repo, en la config del usuario: es un secreto de esta
# maquina, no del proyecto, y el repo se copia a los workspaces (§ "Varias
# sesiones a la vez" de telegram-coordinator/CLAUDE.md). Modo 600.
TOKEN_FILE = Path.home() / ".config" / "fv-web.env"


# --------------------------------------------------------------------- utiles


def corre(cmd: list[str], cwd: Path | None = None, timeout: int = 900) -> tuple[int, str]:
    """Ejecuta y devuelve (codigo, salida). El timeout no es opcional: un `npm`
    colgado dejaria el aprovisionamiento esperando en silencio."""
    try:
        p = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True,
                           text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"se agoto el tiempo ({timeout}s) en: {' '.join(cmd)}"
    except OSError as e:
        return 127, str(e)
    return p.returncode, (p.stdout + p.stderr).strip()


def lee_env(path: Path) -> dict[str, str]:
    """`CLAVE=valor` por linea, aceptando `export`. Cinco lineas en vez de una
    dependencia: esto tiene que correr con el python3 pelado del sistema."""
    out: dict[str, str] = {}
    try:
        texto = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        if linea.startswith("export "):
            linea = linea[7:]
        clave, sep, valor = linea.partition("=")
        if sep:
            out[clave.strip()] = valor.strip().strip("'\"")
    return out


def token(crear: bool = False) -> str:
    """El token de la puerta, con su orden de precedencia DECLARADO:

    1. `FV_WEB_TOKEN` del entorno -- lo que manda si alguien lo pone a mano.
    2. `<repo>/.env` (`FV_WEB_TOKEN` o `WEB_TOKEN`): es lo que escribe el
       lanzador con `env_prefix: "FVW_"`, y es el unico camino por el que un
       token SOBREVIVE a rehacer la maquina (vive en el .env del lanzador).
    3. `~/.config/fv-web.env`, generado aqui la primera vez. Efimero como la
       maquina: al rehacerla hay uno nuevo, y se pregunta por Telegram.

    ⚠ Sin `crear` no inventa nada: quien solo pregunta (`estado`, `url`) tiene
    que poder distinguir "no hay token" de "hay uno nuevo porque preguntaste".
    """
    del_entorno = os.environ.get("FV_WEB_TOKEN", "").strip()
    if del_entorno:
        return del_entorno
    del_repo = lee_env(ROOT / ".env")
    for clave in ("FV_WEB_TOKEN", "WEB_TOKEN"):
        if del_repo.get(clave):
            return del_repo[clave]
    guardado = lee_env(TOKEN_FILE).get("FV_WEB_TOKEN", "")
    if guardado:
        return guardado
    if not crear:
        return ""
    nuevo = secrets.token_urlsafe(18)
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(f"FV_WEB_TOKEN={nuevo}\n", encoding="utf-8")
    TOKEN_FILE.chmod(0o600)
    return nuevo


def ip_publica() -> str:
    """La IP por la que se llega desde fuera. Metadatos de DigitalOcean primero
    (es la respuesta correcta en un droplet), la del socket como defecto."""
    try:
        with urllib.request.urlopen(
                "http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address",
                timeout=2) as r:
            ip = r.read().decode().strip()
            if ip:
                return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def escuchando(puerto: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.5)
        return s.connect_ex(("127.0.0.1", puerto)) == 0


def quien_escucha(puerto: int) -> dict | None:
    """Que proceso tiene el puerto, con su CWD -- que es lo que dice de QUIEN es.

    ⚠ Existe porque paso: el 2026-08-29 el :8010 lo tenia un `fv.api` lanzado a
    mano con `setsid nohup` desde `~/ws/tema-2`, huerfano de la sesion que lo
    arranco. Un servicio con `Restart=always` contra un puerto ocupado no falla
    de una vez: se reinicia en bucle, y en el log pone "address already in use"
    con la pinta de un error del propio servicio. Decir de quien es el puerto lo
    convierte en una linea accionable.

    De quien es un proceso lo dice su CWD y no su linea de comando: con varios
    workspaces la orden es identica en todos (misma leccion que `cerrable.mjs`).
    """
    code, salida = corre(["ss", "-ltnp"], timeout=15)
    if code != 0:
        return None
    for linea in salida.splitlines():
        campos = linea.split()
        if len(campos) < 5 or not campos[3].endswith(f":{puerto}"):
            continue
        pid = ""
        marca = "pid="
        if marca in linea:
            pid = linea.split(marca, 1)[1].split(",", 1)[0].strip(") ")
        cwd = ""
        if pid.isdigit():
            try:
                cwd = os.readlink(f"/proc/{pid}/cwd")
            except OSError:
                cwd = "(no se pudo leer)"
        return {"pid": pid, "cwd": cwd}
    return None


def sudo_ufw(*args: str) -> tuple[int, str]:
    """ufw con `sudo -n`: si no hay sudo sin contrasena, se dice y se sigue. El
    firewall es de la maquina, no del repo, y no poder tocarlo no es motivo para
    abortar una instalacion que por lo demas queda bien."""
    if not shutil.which("ufw"):
        return 127, "no hay ufw en esta maquina"
    return corre(["sudo", "-n", "ufw", *args], timeout=30)


def puerto_abierto(puerto: int) -> bool | None:
    """True/False si se pudo mirar, None si no. None NO es False: no poder
    comprobar el firewall y creerlo cerrado seria justo el error tranquilizador."""
    code, salida = sudo_ufw("status")
    if code != 0:
        return None
    return any(str(puerto) in l for l in salida.splitlines())


def unidad_estado() -> str:
    """"activo" | "instalado, parado" | "no instalado" | "no se".

    Las cuatro y no un booleano: "no esta instalado" y "esta instalado y parado"
    piden cosas distintas -- la primera un `install-service`, la segunda un
    `systemctl start`-- y con un si/no las dos se leian igual.
    """
    code, carga = corre(["systemctl", "show", "-p", "LoadState", "--value", UNIT],
                        timeout=20)
    if code != 0:
        return "no se"
    if carga.strip() != "loaded":
        return "no instalado"
    activo, _ = corre(["systemctl", "is-active", "--quiet", UNIT], timeout=20)
    return "activo" if activo == 0 else "instalado, parado"


def ayuda_instalar() -> str:
    """El remedio de "no instalado", escrito UNA vez.

    Lo imprimen `estado` y `arrancar`. Copiado serian dos textos que divergen, y
    el que se depura luego es siempre el que no escribiste tu.
    """
    return ("\n  Instalarlo:  python3 scripts/do_droplet.py install-service "
            f"--service {UNIT}\n  (desde ~/src/digital-ocean-dropplet-auto-launching; "
            "desde Telegram, el ejecutor `instalar-servicio`)")


# ------------------------------------------------------------------ preparar


def cmd_preparar(args: argparse.Namespace) -> int:
    fallos = []

    venv = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / \
        ("python.exe" if os.name == "nt" else "python")
    if venv.exists():
        print(f"  venv: ya esta ({venv})")
    else:
        print("  venv: creando (torch CPU + extras api/dev), tarda unos minutos...")
        code, salida = corre([sys.executable, "-m", "venv", ".venv"], timeout=300)
        if code == 0:
            code, salida = corre([str(venv), "-m", "pip", "install", "-q",
                                  "--index-url", "https://download.pytorch.org/whl/cpu",
                                  "torch"], timeout=1800)
        if code == 0:
            code, salida = corre([str(venv), "-m", "pip", "install", "-q", "-e", ".[api,dev]"],
                                 timeout=1800)
        if code != 0:
            fallos.append(f"venv: {salida[-400:]}")
        else:
            print("  venv: creado")

    if not shutil.which("npm"):
        fallos.append("npm: no esta instalado, no se puede construir el front")
    else:
        # `npm ci` y no `install`: el lockfile esta commiteado y lo que tiene que
        # llegar al server es EXACTAMENTE lo medido, no lo que resuelva hoy npm.
        print("  front: npm ci...")
        code, salida = corre(["npm", "ci"], cwd=ROOT / "web", timeout=900)
        if code != 0:
            fallos.append(f"npm ci: {salida[-400:]}")
        else:
            print("  front: npm run build...")
            code, salida = corre(["npm", "run", "build"], cwd=ROOT / "web", timeout=900)
            if code != 0:
                fallos.append(f"npm run build: {salida[-400:]}")
            else:
                print(f"  front: construido en {ROOT / 'web' / 'dist'}")

    t = token(crear=True)
    print(f"  token: {'listo' if t else 'NO SE PUDO CREAR'} ({TOKEN_FILE})")
    if not t:
        fallos.append("token: no se pudo crear ni leer")

    # El puerto se abre aqui y no en cloud-init: cloud-init vale para TODAS las
    # maquinas, y este puerto solo tiene sentido donde corre este servicio.
    # Se avisa en voz alta porque es lo unico de este script que se ve desde
    # fuera de la maquina.
    if args.sin_abrir:
        print(f"  firewall: --sin-abrir, el puerto {args.port} queda cerrado")
    else:
        code, salida = sudo_ufw("allow", f"{args.port}/tcp")
        if code == 0:
            print(f"  firewall: puerto {args.port}/tcp ABIERTO a Internet "
                  f"(se cierra con: python3 scripts/web_app.py cerrar)")
        else:
            print(f"  AVISO firewall: no pude abrir {args.port}/tcp ({salida[:150]}).\n"
                  f"    Hazlo a mano:  sudo ufw allow {args.port}/tcp")

    if fallos:
        print("\nNO quedo preparada:")
        for f in fallos:
            print(f"  x {f}")
        return 1
    print("\nPreparada. Arrancala con el servicio, o a mano:\n"
          f"  .venv/bin/python scripts/web_app.py servir --host 0.0.0.0 --port {args.port}")
    return 0


# --------------------------------------------------------------------- servir


def cmd_servir(args: argparse.Namespace) -> int:
    """Arranca el proceso. Es el `start` del servicio, asi que aqui NO se falla
    por algo que se pueda resolver: si no hay token se crea (es aleatorio, no
    hay decision que tomar). Lo que si falla antes de empezar es no tener front
    construido o querer exponerse sin puerta -- eso lo decide `fv.api`."""
    # Layout src/: con el venv del repo `fv` ya esta instalado en editable, pero
    # esto deja funcionar tambien a un venv que solo tenga uvicorn.
    sys.path.insert(0, str(ROOT / "src"))
    t = token(crear=not is_local(args.host))
    if t:
        os.environ["FV_WEB_TOKEN"] = t
    from fv.api.__main__ import main as api_main

    argv = ["--host", args.host, "--port", str(args.port)]
    if not args.sin_front:
        argv.append("--web")
    sys.argv = ["fv-api", *argv]
    return api_main()


def is_local(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "::1", "localhost"}


# ------------------------------------------------------------- mirar y cerrar


def url(args: argparse.Namespace) -> str:
    t = token()
    base = f"http://{ip_publica()}:{args.port}"
    return f"{base}/?t={t}" if t else base


def cmd_url(args: argparse.Namespace) -> int:
    if not token():
        print("No hay token todavia. Crealo con: python3 scripts/web_app.py preparar")
        return 1
    print(url(args))
    return 0


def cmd_estado(args: argparse.Namespace) -> int:
    estado_unidad = unidad_estado()
    activa = estado_unidad == "activo"
    viva = escuchando(args.port)
    abierto = puerto_abierto(args.port)
    dist = (ROOT / "web" / "dist" / "index.html").is_file()

    print(f"servicio {UNIT}: {estado_unidad}")
    print(f"puerto {args.port} en esta maquina: {'escuchando' if viva else 'nadie escucha'}")
    if viva and not activa:
        duenio = quien_escucha(args.port)
        if duenio:
            print(f"  AVISO: NO es el servicio, lo tiene el pid {duenio['pid']} "
                  f"(cwd {duenio['cwd']}).")
            print(f"     Mientras siga ahi, {UNIT} no puede arrancar en este puerto.")
    print("firewall: " + {True: f"{args.port}/tcp abierto",
                          False: f"{args.port}/tcp CERRADO (no se llega desde fuera)",
                          None: "NO SE (no pude preguntarle a ufw)"}[abierto])
    print(f"front construido: {'si' if dist else 'NO (falta npm run build)'}")
    print(f"token: {'si' if token() else 'NO'}  ({TOKEN_FILE})")
    # ⚠ La URL solo se imprime si el que escucha es EL SERVICIO. Un `fv-api`
    # puesto a mano escucha en 127.0.0.1 y desde fuera esa direccion no contesta:
    # darla igual seria mandar al usuario a una pagina que no carga, con todo lo
    # de arriba diciendo "si". Un dato que parece bueno y no lo es cuesta mas que
    # no darlo.
    if activa and viva and token():
        print(f"\n  {url(args)}")
    elif estado_unidad == "no instalado":
        print(ayuda_instalar())
    elif viva:
        print(f"\n  Libera el puerto {args.port} y luego:  sudo systemctl start {UNIT}")
    else:
        print(f"\n  Arrancar:  sudo systemctl start {UNIT}")
    return 0 if (activa and viva and dist) else 1


def cmd_cerrar(args: argparse.Namespace) -> int:
    """El freno: deja de estar disponible desde fuera, sin tocar el servicio.
    Cierra el puerto en vez de parar el proceso porque lo que se quiere quitar es
    la EXPOSICION; un entrenamiento en curso dentro del proceso no tiene por que
    morir para eso."""
    code, salida = sudo_ufw("delete", "allow", f"{args.port}/tcp")
    if code != 0:
        print(f"No pude cerrar el puerto ({salida[:200]}).\n"
              f"  A mano:  sudo ufw delete allow {args.port}/tcp")
        return 1
    print(f"Puerto {args.port}/tcp cerrado. El servicio sigue corriendo para esta maquina.\n"
          f"  Volver a abrir:  python3 scripts/web_app.py abrir")
    return 0


def cmd_abrir(args: argparse.Namespace) -> int:
    code, salida = sudo_ufw("allow", f"{args.port}/tcp")
    if code != 0:
        print(f"No pude abrir el puerto ({salida[:200]}).")
        return 1
    print(f"Puerto {args.port}/tcp abierto.\n  {url(args)}")
    return 0


def cmd_parar(args: argparse.Namespace) -> int:
    # Sin unidad no hay nada que parar, o sea que el estado que pediste ya se
    # cumple: se sale con 0 y NO se imprime el remedio de instalar. Pediste
    # apagar; ofrecerte encender seria contestar a otra pregunta.
    if unidad_estado() == "no instalado":
        print(f"{UNIT}: no instalado (nada que parar)")
        return 0
    code, salida = corre(["sudo", "-n", "systemctl", "stop", UNIT], timeout=60)
    print(f"{UNIT}: {'parado' if code == 0 else 'no pude pararlo: ' + salida[:200]}")
    return 0 if code == 0 else 1


def cmd_arrancar(args: argparse.Namespace) -> int:
    # Preguntar ANTES de llamar a systemctl. "no instalado" y "instalado y
    # parado" piden cosas distintas -- que es exactamente para lo que existe
    # `unidad_estado()`-- y este era el unico subcomando que no la miraba: en una
    # maquina sin la unidad devolvia el error crudo de systemd ("Unit ... not
    # found") y ningun remedio, mientras `estado` -- a un mensaje de distancia --
    # si lo daba. Medido el 2026-08-29 en un dev que nacio sin el servicio porque
    # la lanzadora leyo un types/dev.json viejo.
    #
    # ⚠ Corta SOLO con "no instalado", nunca con "no se": si systemctl no
    # contesta no sabemos nada, y un falso negativo que impida arrancar algo que
    # si funciona es peor que el error crudo. Ahi se intenta y habla el error de
    # verdad. Mismo criterio que el `requiere` de los ejecutores, que avisa pero
    # no bloquea.
    if unidad_estado() == "no instalado":
        print(f"{UNIT}: no instalado{ayuda_instalar()}")
        return 1
    code, salida = corre(["sudo", "-n", "systemctl", "start", UNIT], timeout=60)
    if code != 0:
        print(f"{UNIT}: no pude arrancarlo: {salida[:200]}")
        return 1
    print(f"{UNIT}: arrancado")
    return cmd_estado(args)


def cmd_log(args: argparse.Namespace) -> int:
    # `-q` calla el aviso de tres lineas que journalctl imprime cuando quien
    # pregunta no esta en `adm`/`systemd-journal` -- y `deploy` no lo esta, asi
    # que salia SIEMPRE. Desde Telegram eso son tres de las cinco lineas de la
    # respuesta, en la pantalla mas pequena que tenemos.
    # Y arregla de paso el fallback de abajo: sin `-q`, journalctl escribe
    # "-- No entries --", que NO es vacio, asi que `sin log para ...` no se
    # imprimia nunca. Comprobado el 2026-08-29 con la unidad sin instalar.
    code, salida = corre(["journalctl", "-u", UNIT, "-n", str(args.lineas),
                          "--no-pager", "-q"], timeout=60)
    print(salida or f"sin log para {UNIT}")
    return 0 if code == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # `--port` va en CADA subcomando y no antes de el: el descriptor del servicio
    # escribe `servir --host 0.0.0.0 --port 8010`, que es el orden natural, y una
    # opcion global solo se acepta ANTES del subcomando.
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--port", type=int, default=PUERTO)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preparar", parents=[comun],
                       help="venv + front construido + token + puerto")
    p.add_argument("--sin-abrir", action="store_true",
                   help="no toques el firewall: la app queda solo para esta maquina")
    p.set_defaults(func=cmd_preparar)

    p = sub.add_parser("servir", parents=[comun],
                       help="arranca el proceso (es el 'start' del servicio)")
    p.add_argument("--host", default=os.environ.get("FV_WEB_HOST", "127.0.0.1"))
    p.add_argument("--sin-front", action="store_true", help="solo el API, sin web/dist")
    p.set_defaults(func=cmd_servir)

    for nombre, fn, ayuda in (
            ("estado", cmd_estado, "esta viva? se llega desde fuera? donde?"),
            ("url", cmd_url, "la URL con el token, para pegar en el movil"),
            ("abrir", cmd_abrir, "abre el puerto en el firewall"),
            ("cerrar", cmd_cerrar, "cierra el puerto: deja de estar disponible fuera"),
            ("parar", cmd_parar, "para el servicio"),
            ("arrancar", cmd_arrancar, "arranca el servicio")):
        sub.add_parser(nombre, parents=[comun], help=ayuda).set_defaults(func=fn)

    p = sub.add_parser("log", parents=[comun], help="ultimas lineas del log del servicio")
    p.add_argument("--lineas", type=int, default=40)
    p.set_defaults(func=cmd_log)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
