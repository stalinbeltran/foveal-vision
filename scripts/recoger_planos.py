r"""Recoge lo que quedo escrito en la RAIZ PLANA del repo de datos y lo pone en su mes.

    .venv/bin/python scripts/recoger_planos.py            # dice que haria
    .venv/bin/python scripts/recoger_planos.py --aplicar  # lo hace

Para que hace falta
-------------------
`fv.artefactos` escribe fechado (`<anio>/<mes>/sweeps/<rec>/runs/<run>/`) desde
el 2026-08-28. Lo que se escribio ANTES de ese arreglo esta en `<data>/sweeps/`
y `<data>/runs/`, sin fecha. La cascada de `path()` lo sigue encontrando ahi --
por eso nada se rompio-- pero ahi se queda mientras alguien no lo mueva, y dos
formas conviviendo es como se acaba mirando en la carpeta equivocada.

No sirve `migrar_data.py` del repo de datos: aquel COPIA (deja el original en su
sitio: lo plano seguiria estando) y reescribe `index.json` entero desde lo que
encuentra, con lo que se cargaria el mapa de los 851 artefactos ya migrados.

Que mes le toca a cada cosa
---------------------------
El de GENERACION, leido de su propio JSON, nunca el mtime -- en un clon limpio
el mtime es la fecha del checkout y mentiria en bloque. Y con los mismos tres
criterios que ya usan el README del repo de datos y `fv.artefactos`:

  1. Un run vive DENTRO de su recorrido (`provenance.sweep`), y hereda su mes.
  2. Un recorrido no se parte por el mes: se fecha por su run mas antiguo.
  3. Un estudio no se reparte: todos sus recorridos van al mismo mes, el del mas
     antiguo. Si el estudio YA tiene mes (porque algo suyo se archivo antes o se
     escribio ya fechado), los planos se van con ellos -- que es el caso que de
     verdad importa, porque es como se juntan las dos mitades de un estudio que
     empezo plano y siguio fechado.

`index.json` NO se toca: es el mapa de la migracion de agosto, y lo que se mueve
aqui se encuentra recorriendo las carpetas de mes (`artefactos._dirs_por_mes`),
que es lo que hace la cascada con todo lo escrito despues de aquella.

Lo que NO hace, a proposito
---------------------------
No mueve nada de un recorrido VIVO. Mover el directorio bajo los pies de una
flota que esta escribiendo ahi deja los runs a medias en el sitio viejo y la
flota escribiendo en un directorio que ya no es el que nadie lee: la forma cara
de perder datos ya pagados. Se comprueba de dos maneras y basta una para negarse
(§ `motivos_para_no_tocar`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fv import artefactos, settings           # noqa: E402

# Estados de los que todavia se puede esperar una escritura. Un run en
# cualquiera de estos es un run que alguien podria estar tocando ahora mismo.
VIVOS = {"running", "queued", "pending"}


def _leer(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mes_de_ts(ts: float) -> str:
    d = dt.datetime.fromtimestamp(ts, dt.UTC)
    return f"{d.year}/{artefactos.MESES_ES[d.month]}"


def fecha_de_run(run: Path) -> float | None:
    """La fecha de un run, de su propio JSON. `status.json` manda."""
    v = _leer(run / "status.json").get("updated_at")
    if isinstance(v, (int, float)):
        return float(v)
    for fichero in ("summary.json", "config.json"):
        for clave in ("updated_at", "created_at", "finished_at"):
            v = _leer(run / fichero).get(clave)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _yo_y_mis_padres() -> set[str]:
    """Mi pid y el de todos mis ancestros.

    `pgrep -f` casa contra la linea de comando ENTERA, asi que casa tambien con
    el shell desde el que se lanzo esto si esa linea menciona `estudio_flota.py`
    -- un `&&` con un `ps | grep` basta. Ese falso positivo se lee como "hay una
    flota viva" y bloquea la recogida para siempre, que es peor que no
    comprobar. Yo no soy una flota, y mi padre tampoco.
    """
    pids, pid = set(), os.getpid()
    while pid and pid > 1 and str(pid) not in pids:
        pids.add(str(pid))
        try:
            texto = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError:
            break
        padre = [l for l in texto.splitlines() if l.startswith("PPid:")]
        pid = int(padre[0].split()[1]) if padre else 0
    return pids


def motivos_para_no_tocar(datos: Path) -> list[str]:
    """Por que NO se puede recoger ahora. Vacia = adelante.

    Dos comprobaciones, y basta una para negarse, porque ninguna de las dos ve
    lo que ve la otra:

    - **Un proceso vivo**: quien escribe es `estudio_flota.py`, y de quien es un
      proceso lo dice su CWD y no su linea de comando (se lanza con ruta
      relativa: filtrar por `ps` clasifica los propios como ajenos). Pero solo
      se ve si corre en ESTA maquina.
    - **Un run en estado no terminal**: eso si se ve aunque quien lo escribiera
      corra en otro sitio -- que es lo normal aqui, donde entrenan maquinas
      alquiladas. A cambio, un run que murio sin cerrar su estado se queda
      "running" para siempre (para eso existe `RunStore.reconcile`), asi que
      esta comprobacion puede dar un falso "no". Ese es el error barato.
    """
    motivos = []
    try:
        salida = subprocess.run(["pgrep", "-f", "estudio_flota.py"],
                                capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        salida = ""
    mios = _yo_y_mis_padres()
    for pid in salida.split():
        if pid in mios:
            continue
        try:
            cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
        except OSError:
            continue                       # murio entre el pgrep y esto
        if cwd.is_relative_to(ROOT.parent):
            motivos.append(f"hay una flota VIVA en este workspace (pid {pid}, "
                           f"cwd {cwd}): esta escribiendo justo lo que se iba a mover")

    for run in sorted((datos / "runs").glob("*/")) if (datos / "runs").is_dir() else []:
        estado = _leer(run / "status.json").get("status")
        if estado in VIVOS:
            motivos.append(f"el run plano {run.name} esta en estado '{estado}': "
                           f"o lo esta escribiendo alguien, o quedo sin cerrar "
                           f"(mira `estudio_progreso.py` antes de forzar)")
    return motivos


def planear(datos: Path) -> tuple[list, list]:
    """(movimientos, avisos). Un movimiento es (origen, destino)."""
    avisos: list[str] = []
    sw_planos = sorted(p for p in (datos / "sweeps").iterdir()
                       if p.is_dir()) if (datos / "sweeps").is_dir() else []
    runs_planos = sorted(p for p in (datos / "runs").iterdir()
                         if p.is_dir()) if (datos / "runs").is_dir() else []

    # de quien es cada run plano: el config manda; el prefijo del nombre es el
    # respaldo (asi lo hizo la migracion, y hay runs viejos sin provenance)
    def duenno(run: Path) -> str | None:
        s = ((_leer(run / "config.json").get("provenance") or {}).get("sweep"))
        if s:
            return s
        for cand in sorted((p.name for p in sw_planos), key=len, reverse=True):
            if run.name.startswith(cand + "-"):
                return cand
        return None

    fechas: dict[str, list[float]] = {}
    de_quien: dict[Path, str | None] = {}
    for r in runs_planos:
        s = duenno(r)
        de_quien[r] = s
        ts = fecha_de_run(r)
        if s and ts is not None:
            fechas.setdefault(s, []).append(ts)

    # criterio 2: un recorrido se fecha por su run mas antiguo; si no tiene runs
    # con fecha, por su propio state.json
    def mes_del_recorrido(nombre: str) -> str:
        v = fechas.get(nombre)
        if v:
            return _mes_de_ts(min(v))
        ts = _leer(datos / "sweeps" / nombre / "state.json").get("updated_at")
        if isinstance(ts, (int, float)):
            return _mes_de_ts(float(ts))
        avisos.append(f"{nombre}: sin fecha en ningun JSON; va al mes de hoy")
        return artefactos.mes_actual()

    # criterio 3: un estudio no se reparte. Se agrupan sus recorridos planos y
    # todos toman el mismo mes -- el que el estudio ya tenga, o el mas antiguo.
    por_estudio: dict[str, list[str]] = {}
    sueltos: list[str] = []
    for s in sw_planos:
        est = _leer(s / "spec.json").get("study")
        if est:
            por_estudio.setdefault(est, []).append(s.name)
        else:
            sueltos.append(s.name)

    mes_de_sweep: dict[str, str] = {}
    for est, nombres in por_estudio.items():
        ya = artefactos.mes_del_estudio(est)
        if ya:
            avisos.append(f"estudio '{est}': ya vive en {ya}; sus recorridos "
                          f"planos se van con el")
        mes = ya or min(mes_del_recorrido(n) for n in nombres)
        for n in nombres:
            mes_de_sweep[n] = mes
    for n in sueltos:
        mes_de_sweep[n] = mes_del_recorrido(n)

    movimientos = [(datos / "sweeps" / n, datos / mes / "sweeps" / n)
                   for n, mes in sorted(mes_de_sweep.items())]

    for r in runs_planos:
        s = de_quien[r]
        if s and s in mes_de_sweep:                      # criterio 1: dentro de el
            destino = datos / mes_de_sweep[s] / "sweeps" / s / "runs" / r.name
        elif s:
            # su recorrido no esta plano: o ya esta fechado, o no esta
            d = artefactos._archivado("sweeps", s) or artefactos._dir_por_mes("sweeps", s)
            if d is None:
                avisos.append(f"{r.name}: dice ser de '{s}', que no aparece en "
                              f"ninguna parte. Se queda donde esta")
                continue
            destino = d / "runs" / r.name
        else:                                            # suelto (un benchmark)
            ts = fecha_de_run(r)
            if ts is None:
                avisos.append(f"{r.name}: suelto y sin fecha. Se queda donde esta")
                continue
            destino = datos / _mes_de_ts(ts) / "runs" / r.name
        movimientos.append((r, destino))

    return movimientos, avisos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aplicar", action="store_true",
                    help="mueve de verdad; sin esto solo dice que haria")
    ap.add_argument("--aunque-corra", action="store_true",
                    help="⚠ ignora que haya algo vivo. Solo si SABES que el "
                         "estado que se ve es de un run que ya murio")
    args = ap.parse_args()

    datos = settings.data_root()
    if datos == settings.project_root():
        print(f"⚠ `foveal-vision-data` no esta clonado: data_root() cae a "
              f"{datos}.\n  Clonalo antes de recoger nada:\n"
              f"    git clone https://github.com/stalinbeltran/foveal-vision-data.git "
              f"{ROOT.parent}/foveal-vision-data")
        return 2
    print(f"repo de datos: {datos}")

    movimientos, avisos = planear(datos)
    if not movimientos:
        print("nada plano que recoger: todo esta ya bajo su <anio>/<mes>/.")
        return 0

    for origen, destino in movimientos:
        print(f"  {origen.relative_to(datos)}  ->  {destino.relative_to(datos)}")
    for a in avisos:
        print(f"  ⚠ {a}")
    print(f"\n{len(movimientos)} objeto(s)")

    motivos = motivos_para_no_tocar(datos)
    if motivos and not args.aunque_corra:
        print("\n🔴 NO se mueve nada:")
        for m in motivos:
            print(f"  - {m}")
        print("\n  Espera a que termine y repite. Si de verdad no queda nada "
              "vivo, --aunque-corra.")
        return 1
    if motivos:
        print("\n⚠ --aunque-corra: se mueve IGNORANDO esto:")
        for m in motivos:
            print(f"  - {m}")

    if not args.aplicar:
        print("\n[simulacro] no se ha movido nada. Repite con --aplicar")
        return 0

    hechos = 0
    for origen, destino in movimientos:
        if destino.exists():
            print(f"  ! {destino.relative_to(datos)} ya existe: NO se pisa. "
                  f"{origen.relative_to(datos)} se queda donde esta")
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(origen), str(destino))
        hechos += 1
    for vacio in ("runs", "sweeps", "studies"):
        d = datos / vacio
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    print(f"\nmovidos {hechos} de {len(movimientos)}. `index.json` sin tocar "
          f"(es el mapa de la migracion; esto se encuentra recorriendo los meses).")
    print(f"Ahora committea en {datos}: git add -A && git commit && git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
