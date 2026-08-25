#!/usr/bin/env python3
r"""Cada hora: ¿le falta algo a los estudios de prioridad? Pues se relanza.

Por que existe
--------------
Una flota de 44 maquinas alquiladas a desconocidos pierde maquinas: sshd que no
contesta, hosts que se caen, inquilinos que se comen los nucleos, plazos que se
agotan. Cada una de esas se lleva los puntos que le quedaban. El resultado es un
estudio **incompleto**, y un barrido incompleto que no dice que le falta es
indistinguible de uno terminado (CLAUDE.md, `reportes/`).

Relanzar es barato y seguro: `estudio_flota.py` lee `runs/` al arrancar y **salta
todo punto cuyo `status.json` diga `done`**, asi que solo alquila para lo que
falta. Un recorrido entero terminado ni siquiera pide maquina.

LO QUE HAY QUE RESPETAR SI SE TOCA ESTO
---------------------------------------
1. **No puede haber dos flotas a la vez.** Los cerrojos de `estudio_flota.py` son
   `threading.Lock`, o sea DENTRO del proceso: dos procesos no se ven. Dos flotas
   sobre los mismos recorridos alquilarian las dos para los mismos puntos y se
   pagaria dos veces. Por eso lo PRIMERO de cada vuelta es mirar si ya hay una
   corriendo, y si la hay no se hace nada.

2. **Tiene tope de relanzamientos** (`--max-relanzamientos`). Un estudio que
   falla siempre -- por codigo, no por maquina -- relanzado cada hora para
   siempre es una factura que crece sola sin arreglar nada. Al agotarse el tope
   se avisa y se para.

3. **Vive desacoplado o no vive.** Un vigilante armado dentro de un turno se
   muere con el turno (CLAUDE.md: paso el 2026-08-14 con tres vigilantes a la
   vez). Se lanza con `scripts/desacoplar.sh`, que le da su propio cgroup y lo
   salva tambien de un `systemctl restart` del coordinador.

4. **El aviso es una comodidad; el log y `runs/` son la fuente de verdad.** Si
   `notify.mjs` falla no hay quien lo lea, asi que todo lo que decide queda
   escrito en su log.

    scripts/desacoplar.sh .venv/bin/python scripts/vigilante_prioridades.py \
        --sweep borde-ancho --sweep pw-fov ... --cada 3600
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

COORD = Path(os.environ.get("COORD_HOME", Path.home() / "src" / "telegram-coordinator"))


def ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"{ahora()}  {msg}", flush=True)


def flota_viva() -> int:
    """PID de una flota ya corriendo, o 0. Ver la regla 1 del docstring."""
    try:
        salida = subprocess.run(
            ["pgrep", "-f", "estudio_flota.py"],
            capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    mios = {os.getpid(), os.getppid()}
    for linea in salida.split():
        try:
            pid = int(linea)
        except ValueError:
            continue
        if pid not in mios:
            return pid
    return 0


def pendientes_por_recorrido(nombres: list) -> dict:
    """{recorrido: (pendientes, total)} leyendo el libro de a bordo de runs/.

    Reusa `cargar_sweeps` de `estudio_flota.py` a proposito: si el vigilante
    contara los puntos por su cuenta, tarde o temprano contaria distinto que
    quien los corre, y entonces relanzaria de mas o de menos sin que se note.
    """
    import estudio_flota as F
    from fv.sweeps.store import SweepStore
    store = SweepStore()
    fuera = {}
    for n in nombres:
        if not store.exists(n):
            fuera[n] = (0, 0)
            continue
        try:
            spec = store.spec(n)
            valid, _ = F.expand_points(spec, spec["base_network_value"])
            pend, hechos = F.puntos_pendientes(n, valid)
            fuera[n] = (len(pend), len(valid))
        except Exception as e:                       # noqa: BLE001
            log(f"  ! no se pudo leer '{n}': {type(e).__name__}: {e}")
            fuera[n] = (-1, -1)
    return fuera


def avisar(texto: str) -> None:
    notify = COORD / "scripts" / "notify.mjs"
    if not notify.exists():
        log(f"  (sin notify.mjs en {notify}; solo queda en este log)")
        return
    try:
        subprocess.run(["node", str(notify), texto], timeout=60,
                       capture_output=True)
    except (OSError, subprocess.SubprocessError) as e:
        log(f"  (el aviso fallo: {e}; el log es la fuente de verdad)")


def relanzar(nombres: list, args) -> None:
    cmd = [".venv/bin/python", "scripts/estudio_flota.py"]
    for n in nombres:
        cmd += ["--sweep", n]
    cmd += ["--reparto", "seed", "--cpu", args.cpu, "--max-price", str(args.max_price),
            "--criba", str(args.criba), "--git", "--horas-max", str(args.horas_max),
            "--yes"]
    salida = Path(args.log_flota)
    log(f"  relanzando: {' '.join(cmd)}")
    log(f"  su log: {salida}")
    with open(salida, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== relanzado por el vigilante {ahora()} =====\n")
        fh.flush()
        subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=fh,
                         start_new_session=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="append", required=True)
    ap.add_argument("--cada", type=int, default=3600, help="segundos entre vueltas")
    ap.add_argument("--vueltas", type=int, default=24, help="tope de vueltas")
    ap.add_argument("--max-relanzamientos", type=int, default=6,
                    help="ver la regla 2: un estudio que falla SIEMPRE no se "
                         "relanza para siempre")
    ap.add_argument("--cpu", default="E5-26")
    ap.add_argument("--max-price", type=float, default=0.12)
    ap.add_argument("--criba", type=int, default=4)
    ap.add_argument("--horas-max", type=float, default=6.0)
    ap.add_argument("--log-flota", default="/tmp/estudio-prioridades.log")
    args = ap.parse_args()

    log(f"vigilante en marcha: {len(args.sweep)} recorridos, cada "
        f"{args.cada} s, {args.vueltas} vueltas como mucho")
    relanzamientos = 0
    for vuelta in range(1, args.vueltas + 1):
        time.sleep(args.cada)
        log(f"--- vuelta {vuelta}/{args.vueltas} ---")

        pid = flota_viva()
        if pid:
            log(f"  hay una flota viva (pid {pid}): no se toca nada")
            continue

        estado = pendientes_por_recorrido(args.sweep)
        faltan = [n for n, (p, _) in estado.items() if p > 0]
        for n, (p, t) in sorted(estado.items()):
            if p > 0:
                log(f"  {n:14s} FALTAN {p}/{t}")
        if not faltan:
            total = sum(t for _, t in estado.values())
            log(f"  todo terminado ({total} puntos). El vigilante se retira.")
            avisar(f"prioridades: los {len(args.sweep)} recorridos estan "
                   f"completos ({total} runs). Toca escribir los reportes. "
                   f"Log: {args.log_flota}")
            return 0

        if relanzamientos >= args.max_relanzamientos:
            log(f"  tope de {args.max_relanzamientos} relanzamientos agotado y "
                f"aun faltan {sum(estado[n][0] for n in faltan)} puntos en "
                f"{len(faltan)} recorridos. NO se relanza mas.")
            avisar(f"prioridades: agotado el tope de {args.max_relanzamientos} "
                   f"relanzamientos y siguen faltando puntos en "
                   f"{', '.join(faltan)}. Ya no es un problema de maquinas: "
                   f"mira {args.log_flota}.")
            return 1

        relanzamientos += 1
        log(f"  faltan puntos en {len(faltan)} recorridos "
            f"(relanzamiento {relanzamientos}/{args.max_relanzamientos})")
        relanzar(faltan, args)
        avisar(f"prioridades: se cayeron puntos en {', '.join(faltan)}. "
               f"Relanzada la flota solo para lo que falta "
               f"({relanzamientos}/{args.max_relanzamientos}).")

    log("se acabaron las vueltas del vigilante")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
