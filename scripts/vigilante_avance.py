#!/usr/bin/env python3
r"""Cada 10 min: ¿avanza cada máquina alquilada? La que no, se destruye.

Por qué existe (y por qué no basta con `vigilante_prioridades.py`)
------------------------------------------------------------------
El vigilante de prioridades mira **una cosa**: si a los recorridos les faltan
puntos y no hay flota viva, relanza. Su regla 1 dice, con razón, que si ya hay
una flota corriendo no se toca nada — dos flotas alquilarían dos veces para los
mismos puntos.

El hueco está justo ahí: **mientras la flota está viva, nadie mira si las
máquinas que ya pagó hacen algo.** Y no siempre lo hacen. Una máquina de Vast se
alquila y factura desde el segundo uno, pero puede no llegar nunca a ser
utilizable: sshd que no acepta la clave, host saturado, inquilino que se come
los núcleos, disco que no monta. La flota tiene su propio plazo (`--horas-max`)
y hasta que se agota **espera**.

MEDIDO el 2026-08-26 (`/tmp/estudio-p2b.log`, flota de prioridad 2):

    15:20:55  [ov-fov-s3] FALLO en el intento 1/2: RuntimeError: plazo agotado
              (4.0 h) y la maquina no contesta

Cuatro horas de una máquina alquilada sin producir una sola época. A los
0,0713 $/h de esa misma flota son ~0,29 $ tirados, y —lo que importa más— cuatro
horas de reloj perdidas en un estudio que se creía en marcha. Ése es el «servidor
creado sin hacer nada» que hay que cortar en 10 minutos y no en cuatro horas.

Este vigilante es el complemento, no el sustituto: **mira el avance máquina a
máquina y destruye la que no avanza, esté viva la flota o no.** Al destruirla, la
flota ve caer el SSH y reintenta en otra en vez de esperar su plazo entero.

Cómo decide (los dos síntomas, y por qué esos umbrales)
-------------------------------------------------------
El latido de una máquina es el libro de a bordo: `runs/<run>/status.json`, que la
flota reescribe cada vez que consigue traerse algo de ella. Si la máquina se
congela, deja de actualizarse. No hace falta entrar por SSH ni preguntar nada.

  A. **Alquilada y muda**: pasada la gracia, ni un `status.json` suyo escrito.
  B. **Congelada**: tiene runs, pero el último latido es más viejo que
     `--sin-avance`.
  C. **Huérfana**: su lote está entero `done` y la máquina sigue facturando.
  D. **No llega**: avanza, pero a un ritmo que no termina antes de su propio
     plazo (`--horas-max`). Es la que A, B y C no ven: escribe su latido
     puntualmente y por eso parece sana.

Los umbrales salen de lo medido el 2026-08-26 en ese mismo log:

  * arranque real: alquiler → SSH listo **2,7-3,5 min**, instalación ~50 s,
    primera época a los **~5 min**. La gracia por defecto son **25 min**, muy por
    encima, porque destruir una máquina sana cuesta más que esperar a una tonta.
  * época más lenta observada en estos recorridos: **161,6 s**
    (`ov-fov-0012-overlap_fovea_px2_seed3`). Con `--sin-avance` 20 min caben ~7
    épocas de la peor, así que una máquina lenta no se confunde con una muerta.

LO QUE HAY QUE RESPETAR SI SE TOCA ESTO
---------------------------------------
1. **Sólo toca lo que es suyo.** Actúa únicamente sobre instancias con etiqueta
   `estudio-*`, que es la que pone `estudio_flota.py`. Cualquier otra cosa que
   haya en la cuenta —un bench, algo lanzado a mano— se lista y **no se toca**.
   Un vigilante que destruye lo que no entiende es peor que no tenerlo.

2. **La gracia no es opcional y va por instancia, no por vuelta.** Se mide contra
   `start_date` de la propia instancia, así que una máquina recién alquilada en
   mitad de una vuelta no se juzga con el reloj de otra.

3. **Ante la duda, NO se destruye.** El latido es el **más reciente** de dos
   relojes (el `updated_at` que escribe la máquina y el mtime del fichero, que es
   cuando lo recibimos aquí): si cualquiera de los dos dice «hace poco», está
   viva. El reloj remoto puede ir desviado; el error se paga esperando de más, no
   matando de más.

4. **No puede haber dos flotas a la vez** — misma regla que
   `vigilante_prioridades.py`, y por lo mismo: sus cerrojos son `threading.Lock`,
   o sea dentro del proceso. Antes de relanzar se comprueba que no hay ninguna.

5. **Tiene tope de relanzamientos.** Un estudio que falla por código y no por
   máquina, relanzado cada 10 min para siempre, es una factura que crece sola sin
   arreglar nada.

6. **Vive desacoplado o no vive.** Un vigilante armado dentro de un turno se muere
   con el turno (CLAUDE.md: pasó el 2026-08-14 con tres a la vez). Se lanza con
   `scripts/desacoplar.sh`, que le da su propio cgroup y lo salva también de un
   `systemctl restart` del coordinador.

7. **`--dry-run` tiene que seguir siendo verdad.** Es lo único que permite probar
   los umbrales sin destruir ni alquilar nada. Si se añade una acción nueva, va
   detrás de esa bandera igual que las de ahora.

    scripts/desacoplar.sh .venv/bin/python scripts/vigilante_avance.py \
        --sweep ov-fov --cada 600
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANZADOR = ROOT.parent / "digital-ocean-dropplet-auto-launching"
COORD = Path(os.environ.get("COORD_HOME", Path.home() / "src" / "telegram-coordinator"))

sys.path.insert(0, str(ROOT / "src"))

from fv import datarepo
sys.path.insert(0, str(Path(__file__).resolve().parent))

# El espacio de nombres de las instancias de ESTE estudio. Va en una lista para
# que `--prefijo` pueda fijarlo sin `global`.
#
# Por que es parametro: la cuenta es UNA y puede haber dos estudios a la vez.
# COMPROBADO el 2026-08-27 con 8 maquinas `estudio-c*` de otro estudio vivas: con
# el prefijo cableado, este vigilante las cuenta como suyas. Tiene que coincidir
# con el `--prefijo` de estudio_flota.py.
PREFIJO_DEF = "estudio-"       # la etiqueta que pone estudio_flota.py
PREFIJO: list = [PREFIJO_DEF]


# ------------------------------------------------------------------ secretos
#
# Un proceso desacoplado nace SIN tokens a propósito (`desacoplar.sh` no le pasa
# credenciales: sudo escribiría la lista de --preserve-env en el journal). Y son
# DOS ficheros, no uno; el segundo es el que trae el de Vast. Sin esto, el
# vigilante arranca, no puede listar instancias y no destruye nada -- callado.
# Es el mismo fallo que dejó a claude-resumer.mjs diciendo «Not logged in».

def cargar_secretos() -> list:
    cargados = []
    for fichero in (COORD / ".env", Path.home() / ".config" / "dev-secrets.env"):
        if not fichero.exists():
            continue
        try:
            for linea in fichero.read_text(encoding="utf-8").splitlines():
                linea = linea.strip()
                if not linea or linea.startswith("#") or "=" not in linea:
                    continue
                clave, valor = linea.split("=", 1)
                # `dev-secrets.env` lo genera do_droplet.py con `export VAR=...`,
                # así que sin quitar el prefijo la clave sería "export VAR" y el
                # token estaría delante sin que nadie lo viera.
                clave = re.sub(r"^export\s+", "", clave.strip())
                valor = valor.strip().strip('"').strip("'")
                os.environ.setdefault(clave, valor)   # lo heredado manda
            cargados.append(str(fichero))
        except OSError as e:
            print(f"[secretos] no pude leer {fichero}: {e}", file=sys.stderr)
    return cargados


def ahora() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"{ahora()}  {msg}", flush=True)


def avisar(texto: str) -> None:
    """El aviso es una comodidad; este log es la fuente de verdad."""
    notify = COORD / "scripts" / "notify.mjs"
    if not notify.exists():
        log(f"  (sin notify.mjs en {notify}; sólo queda en este log)")
        return
    try:
        subprocess.run(["node", str(notify), texto], timeout=60, capture_output=True)
    except (OSError, subprocess.SubprocessError) as e:
        log(f"  (el aviso falló: {e}; el log es la fuente de verdad)")


# -------------------------------------------------------- lote -> runs -> latido


def descomponer(etiqueta: str) -> tuple:
    """`ov-fov-s3` -> ('ov-fov', ('seed', 3)). Ver `particion()` en estudio_flota.

    Se parte por el ÚLTIMO guion porque los nombres de recorrido llevan guiones
    (`ov-fov`, `pl-t-bs`): partir por el primero daría el recorrido equivocado.
    """
    if "-" in etiqueta:
        base, suf = etiqueta.rsplit("-", 1)
        if suf == "todo":
            return base, ("todo", None)
        if re.fullmatch(r"s\d+", suf):
            return base, ("seed", int(suf[1:]))
        if re.fullmatch(r"p\d+", suf):
            return base, ("punto", int(suf[1:]))
    return etiqueta, ("todo", None)


def runs_del_lote(etiqueta: str):
    """Los runs que a esa máquina le tocaba mover, o None si no se sabe.

    Reusa `expand_points` y `point_run_name` de quien los corre, a propósito: si
    el vigilante nombrara los runs por su cuenta, tarde o temprano los nombraría
    distinto y creería muda a una máquina que está trabajando.
    """
    import estudio_flota as F
    from fv.sweeps.store import SweepStore
    from fv.sweeps.runner import point_run_name

    sweep, (modo, valor) = descomponer(etiqueta)
    store = SweepStore()
    if not store.exists(sweep):
        return None
    try:
        spec = store.spec(sweep)
        valid, _ = F.expand_points(spec, spec["base_network_value"])
    except Exception as e:                                      # noqa: BLE001
        # Un recorrido viejo con un spec que el codigo de hoy ya no sabe
        # expandir (visto el 2026-08-26: `proxy-c-d` da KeyError 'd') no puede
        # llevarse la vuelta por delante. Si se propagara, UNA maquina rara
        # dejaria sin juzgar a TODAS las demas y el vigilante no destruiria
        # nada, callado. Se salta con aviso, como hace el registry con un JSON
        # roto: no se sabe que runs son suyos, asi que no se toca.
        log(f"    ! no pude expandir '{sweep}' ({type(e).__name__}: {e}); "
            f"no juzgo esa maquina")
        return None
    nombres = []
    for i, p in enumerate(valid):
        if modo == "seed" and p["overrides"].get("seed") != valor:
            continue
        if modo == "punto" and i != valor:
            continue
        nombres.append(point_run_name(sweep, i, p["overrides"]))
    return nombres


def latido(nombres: list) -> tuple:
    """(instante del último signo de vida, cuántos runs siguen pendientes).

    El instante es el MÁS RECIENTE de los dos relojes disponibles (regla 3): el
    `updated_at` que escribió la máquina y el mtime de cuando lo recibimos aquí.
    """
    ultimo, pendientes = 0.0, 0
    for n in nombres:
        st = datarepo.resolve("runs", n) / "status.json"
        if not st.exists():
            pendientes += 1
            continue
        try:
            d = json.loads(st.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            d = {}
        if d.get("status") not in ("done", "cancelled"):
            pendientes += 1
        try:
            mt = st.stat().st_mtime
        except OSError:
            mt = 0.0
        try:
            ua = float(d.get("updated_at") or 0)
        except (TypeError, ValueError):
            ua = 0.0
        ultimo = max(ultimo, ua, mt)
    return ultimo, pendientes


# Mediana de epocas hasta parar, MEDIDA el 2026-08-27 sobre los 767 runs con
# summary.json en disco: mediana 41, p90 66, max 150, min 3. Se usa la MEDIANA y
# no el p90 a proposito: aqui sirve para decidir si una maquina NO llega, y el
# error hay que pagarlo esperando de mas (regla 3), asi que se estima el trabajo
# que queda por lo BAJO. Si ni siquiera el caso optimista cabe, no cabe.
EPOCAS_TIPICAS = 41
# Suelo de epocas que le quedan a un run ya empezado: con patience=10 no puede
# parar en la siguiente, asi que estimar "le quedan 0" seria falso.
EPOCAS_MINIMAS = 5


def ritmo(nombres: list) -> tuple:
    """(epoca actual, s/epoca reciente) del run que la maquina tiene a medias.

    El s/epoca sale del propio `metrics.jsonl` -- que ya esta ahi y no cuesta
    nada-- y de las 3 ultimas epocas, igual que la vigilancia de degradacion de
    `estudio_flota.py`. Tres y no una: una epoca lenta suelta es ruido normal.
    """
    for n in nombres:
        st = datarepo.resolve("runs", n) / "status.json"
        mj = datarepo.resolve("runs", n) / "metrics.jsonl"
        if not st.exists() or not mj.exists():
            continue
        try:
            d = json.loads(st.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("status") in ("done", "cancelled"):
            continue
        try:
            filas = [json.loads(x) for x in mj.read_text().splitlines() if x.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        segs = [f.get("seconds") for f in filas[-3:] if f.get("seconds")]
        if not segs:
            continue
        segs.sort()
        return int(d.get("epoch") or len(filas)), segs[len(segs) // 2]
    return 0, 0.0


def juzgar(inst: dict, args, t: float) -> tuple:
    """(veredicto, motivo). Veredicto: ajena | arrancando | ok | danada."""
    etiqueta = (inst.get("label") or "")[len(PREFIJO[0]):]
    try:
        edad = (t - float(inst.get("start_date") or t)) / 60.0
    except (TypeError, ValueError):
        edad = 0.0                      # sin fecha fiable: se trata como recién nacida

    nombres = runs_del_lote(etiqueta)
    if nombres is None:
        return "ajena", f"no sé de qué recorrido es '{etiqueta}'; no la toco"
    if not nombres:
        return "ajena", f"'{etiqueta}' no resuelve a ningún run; no la toco"

    ultimo, pendientes = latido(nombres)

    if pendientes == 0:
        if edad >= args.gracia:
            return "danada", (f"su lote está completo ({len(nombres)} runs done) y "
                              f"sigue facturando desde hace {edad:.0f} min")
        return "ok", "su lote está completo; aún dentro de la gracia"

    if edad < args.gracia:
        return "arrancando", (f"{edad:.0f} min de vida, gracia {args.gracia} min "
                              f"(arrancar cuesta ~5 min)")
    if ultimo <= 0:
        return "danada", (f"{edad:.0f} min alquilada y no ha escrito ni una época "
                          f"de sus {pendientes} runs pendientes")
    quieta = (t - ultimo) / 60.0
    if quieta > args.sin_avance:
        return "danada", (f"sin avance desde hace {quieta:.0f} min "
                          f"(tope {args.sin_avance} min), {pendientes} runs a medias")

    # D. NO LLEGA: avanza, pero tan despacio que no termina antes de su propio
    # plazo. MEDIDO el 2026-08-27: una maquina de `bp-r26` fue a 238 s/epoca
    # contra los 35-50 normales; a las 5,5 h de sus 6 iba por la epoca 19 y le
    # faltaban ~87 min de trabajo para 26 de plazo. Los sintomas A, B y C no la
    # veian -- escribia su latido puntualmente-- asi que habria seguido
    # facturando hasta que el plazo la matara con el run a medias, y el punto
    # habria que rehacerlo igual. Destruirla ANTES no pierde nada y libera el
    # dinero y la ranura.
    #
    # Se estima por lo BAJO (ver EPOCAS_TIPICAS) y ademas se exige margen: solo
    # se declara si NI SIQUIERA el caso optimista cabe en el plazo. Una maquina
    # lenta que llega justo se deja en paz.
    epoca, s_epoca = ritmo(nombres)
    if s_epoca > 0 and args.horas_max:
        quedan_epocas = max(EPOCAS_MINIMAS, EPOCAS_TIPICAS - epoca)
        falta_min = quedan_epocas * s_epoca / 60.0
        plazo_min = args.horas_max * 60.0 - edad
        if falta_min > plazo_min:
            return "danada", (
                f"NO LLEGA: va por la época {epoca} a {s_epoca:.0f} s/época, o sea "
                f"~{falta_min:.0f} min mas de trabajo (y eso siendo optimista: "
                f"{quedan_epocas} épocas), pero solo le quedan {plazo_min:.0f} min "
                f"de sus {args.horas_max} h de plazo")

    return "ok", (f"último avance hace {quieta:.0f} min, {pendientes} runs a medias"
                  + (f", época {epoca} a {s_epoca:.0f} s/época" if s_epoca else ""))


# --------------------------------------------------------------------- acciones


def flota_viva(sweeps: "list | None" = None) -> int:
    """PID de una flota corriendo SOBRE MIS RECORRIDOS, o 0. Ver la regla 4.

    Tres condiciones, y las tres hacen falta. Cada una tapa un fallo MEDIDO:

    1. **Tiene que SER el script, no mencionarlo.** `pgrep -f estudio_flota.py`
       casa con cualquier proceso que lleve esa cadena en su linea de comando --
       incluido el propio `pgrep`, un `while pgrep ...` de un avisador, o el
       comando de quien esta mirando. MEDIDO el 2026-08-27: un avisador armado
       con `while pgrep -f "estudio_flota.py --sweep stride-01"` se conto como
       flota, y este vigilante paso **19 vueltas (~3 h) sin relanzar** los 14
       puntos que faltaban, diciendo en cada una "hay una flota viva". Es
       exactamente el sintoma que la regla existe para evitar: un estudio que
       parece vigilado y no avanza. Asi que se exige que `argv[0]` sea un python
       y que algun argumento SEA el script.
    2. **De quien es lo dice el CWD**, no la linea de comando (CLAUDE.md del
       coordinador). La flota se lanza con ruta relativa, asi que su linea no
       contiene el workspace por ningun lado.
    3. **Y que sea de MIS recorridos**: la cuenta puede tener dos estudios, y
       "no relanzar si hay flota" solo vale si es flota de los mismos puntos.

    Sin `sweeps` se conserva el comportamiento de antes para el resto de
    llamadas: cualquier flota de este workspace.
    """
    try:
        salida = subprocess.run(["pgrep", "-f", "estudio_flota.py"],
                                capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    mios = {os.getpid(), os.getppid()}
    for linea in salida.split():
        try:
            pid = int(linea)
        except ValueError:
            continue
        if pid in mios:
            continue
        try:
            crudo = Path(f"/proc/{pid}/cmdline").read_bytes()
            argv = [a.decode("utf-8", "replace") for a in crudo.split(b"\0") if a]
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue                      # murio entre el pgrep y esto
        if not argv:
            continue
        if "python" not in Path(argv[0]).name:
            continue                      # (1) un envoltorio que solo lo menciona
        if not any(a.endswith("estudio_flota.py") for a in argv):
            continue                      # (1) lo lleva de argumento suelto
        if cwd != str(ROOT):
            continue                      # (2) es de otro workspace
        if sweeps and not any(s in argv for s in sweeps):
            continue                      # (3) es de otro estudio
        return pid
    return 0


def pendientes_por_recorrido(nombres: list) -> dict:
    """{recorrido: (pendientes, total)} leyendo el libro de a bordo de runs/."""
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
            pend, _hechos = F.puntos_pendientes(n, valid)
            fuera[n] = (len(pend), len(valid))
        except Exception as e:                                  # noqa: BLE001
            log(f"  ! no se pudo leer '{n}': {type(e).__name__}: {e}")
            fuera[n] = (-1, -1)
    return fuera


def relanzar(nombres: list, args) -> None:
    cmd = [".venv/bin/python", "scripts/estudio_flota.py"]
    for n in nombres:
        cmd += ["--sweep", n]
    cmd += ["--reparto", "seed", "--cpu", args.cpu, "--max-price", str(args.max_price),
            "--criba", str(args.criba), "--git", "--horas-max", str(args.horas_max),
            "--prefijo", args.prefijo, "--yes"]
    log(f"  relanzando: {' '.join(cmd)}")
    log(f"  su log: {args.log_flota}")
    if args.dry_run:
        log("  (--dry-run: NO se relanza)")
        return
    with open(args.log_flota, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n===== relanzado por vigilante_avance {ahora()} =====\n")
        fh.flush()
        subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=fh,
                         start_new_session=True)


def una_vuelta(args, V, estado: dict) -> str:
    """Una pasada. Devuelve 'seguir' | 'fin' | 'tope'."""
    try:
        vivas = V.instancias()
    except Exception as e:                                      # noqa: BLE001
        log(f"  ! no pude listar instancias ({type(e).__name__}: {e}); "
            f"esta vuelta no juzga nada")
        return "seguir"

    t = time.time()
    mias = [i for i in vivas if (i.get("label") or "").startswith(PREFIJO[0])]
    ajenas = len(vivas) - len(mias)
    gasto = sum(float(i.get("dph_total") or 0) for i in vivas)
    log(f"  {len(vivas)} instancias vivas ({len(mias)} del estudio, {ajenas} ajenas), "
        f"{gasto:.4f} $/h")

    danadas, ajenas_por_nombre = [], []
    for i in mias:
        v, motivo = juzgar(i, args, t)
        if v == "ajena":
            ajenas_por_nombre.append(i)
        marca = {"danada": "DAÑADA", "ok": "ok", "arrancando": "arrancando",
                 "ajena": "ajena"}[v]
        log(f"    [{i.get('label')}] {marca}: {motivo}")
        if v == "danada":
            danadas.append(i)

    for i in danadas:
        iid = int(i["id"])
        if args.dry_run:
            log(f"    (--dry-run: NO destruyo {iid})")
            continue
        try:
            V.destruir(iid)
            estado["destruidas"] += 1
            log(f"    destruida {i.get('label')} ({iid}); deja de facturar "
                f"{float(i.get('dph_total') or 0):.4f} $/h")
        except Exception as e:                                  # noqa: BLE001
            # Un 404 aquí es una buena noticia: ya no existe, no factura.
            if "no_such_instance" in str(e) or "404" in str(e):
                log(f"    {iid} ya no existía (404): nada que pagar")
            else:
                log(f"    AVISO GRAVE: no pude destruir {iid}: {e}")
                log(f"    SIGUE FACTURANDO. Destrúyela ya:\n"
                    f"    python3 {LANZADOR}/scripts/vast_instance.py destroy {iid} --yes")

    if danadas and not args.dry_run:
        avisar(f"vigilante: destruidas {len(danadas)} máquina(s) que no avanzaban "
               f"({', '.join(i.get('label') or '?' for i in danadas)}). "
               f"Log: {args.log}")

    # -- ¿falta algo por medir?
    est = pendientes_por_recorrido(args.sweep)
    faltan = [n for n, (p, _) in est.items() if p > 0]
    for n, (p, tot) in sorted(est.items()):
        if p > 0:
            log(f"    {n:14s} FALTAN {p}/{tot}")

    if not faltan:
        total = sum(tot for _, tot in est.values())
        log(f"  todo terminado ({total} puntos).")
        # Nada que medir y máquinas vivas = huérfanas puras. Se cortan.
        #
        # ⚠ PERO sólo las que son SUYAS. `juzgar` ya declaró "ajena" a la que no
        # resuelve a ningún recorrido de este vigilante ("no sé de qué recorrido
        # es; no la toco") — y esta rama se saltaba ese veredicto y destruía TODA
        # instancia con el prefijo, incluidas las de otro estudio que corriera a
        # la vez. El síntoma habría sido de los peores: runs cortados a media
        # época en el estudio del vecino, sin error propio, indistinguibles de
        # una máquina que se muere sola.
        #
        # Encontrado el 2026-08-27 al preparar el barrido de stride, con 8
        # máquinas de otro estudio vivas en la cuenta (docs/plan-stride-2026-08-27.md 5.3).
        sobrantes = [i for i in mias
                     if i not in danadas and i not in ajenas_por_nombre]
        if ajenas_por_nombre:
            log(f"    {len(ajenas_por_nombre)} instancia(s) con el prefijo pero de "
                f"otro estudio: NO se tocan "
                f"({', '.join(i.get('label') or '?' for i in ajenas_por_nombre)})")
        if sobrantes and not args.dry_run:
            for i in sobrantes:
                try:
                    V.destruir(int(i["id"]))
                    log(f"    destruida sobrante {i.get('label')} ({i['id']})")
                except Exception as e:                          # noqa: BLE001
                    log(f"    AVISO: no pude destruir {i['id']}: {e}")
        avisar(f"vigilante: los {len(args.sweep)} recorridos están completos "
               f"({total} runs) y no queda nada facturando. Toca escribir el "
               f"reporte en reportes/. Log: {args.log}")
        return "fin"

    pid = flota_viva(args.sweep)
    if pid:
        log(f"  hay una flota viva (pid {pid}): no se relanza. Reintentará ella "
            f"en las máquinas que acabo de liberar." if danadas else
            f"  hay una flota viva (pid {pid}): no se relanza (regla 4)")
        return "seguir"

    if estado["relanzamientos"] >= args.max_relanzamientos:
        log(f"  tope de {args.max_relanzamientos} relanzamientos agotado y aún "
            f"faltan puntos en {', '.join(faltan)}. NO se relanza más (regla 5).")
        avisar(f"vigilante: agotado el tope de {args.max_relanzamientos} "
               f"relanzamientos y siguen faltando puntos en {', '.join(faltan)}. "
               f"Ya no es un problema de máquinas: mira {args.log_flota}.")
        return "tope"

    estado["relanzamientos"] += 1
    log(f"  no hay flota y faltan puntos en {len(faltan)} recorridos "
        f"(relanzamiento {estado['relanzamientos']}/{args.max_relanzamientos})")
    relanzar(faltan, args)
    if not args.dry_run:
        avisar(f"vigilante: no había flota y faltaban puntos en "
               f"{', '.join(faltan)}. Relanzada sólo para lo que falta "
               f"({estado['relanzamientos']}/{args.max_relanzamientos}).")
    return "seguir"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", action="append", required=True)
    ap.add_argument("--prefijo", default=PREFIJO_DEF,
                    help="espacio de nombres de las instancias de este estudio. "
                         "Tiene que coincidir con el --prefijo de estudio_flota.py")
    ap.add_argument("--cada", type=int, default=600,
                    help="segundos entre vueltas (tope 600: el encargo es "
                         "'cada 10 min o menos')")
    ap.add_argument("--gracia", type=float, default=25.0,
                    help="minutos que se le perdonan a una máquina recién "
                         "alquilada (arrancar cuesta ~5 min, medido 2026-08-26)")
    ap.add_argument("--sin-avance", type=float, default=20.0,
                    help="minutos sin latido a partir de los cuales se destruye")
    ap.add_argument("--vueltas", type=int, default=144, help="tope de vueltas")
    ap.add_argument("--max-relanzamientos", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true",
                    help="dice qué haría y no destruye ni alquila nada")
    ap.add_argument("--una-vuelta", action="store_true",
                    help="una sola pasada y sale (para probar o para cron)")
    ap.add_argument("--cpu", default="E5-26")
    ap.add_argument("--max-price", type=float, default=0.12)
    ap.add_argument("--criba", type=int, default=3)
    ap.add_argument("--horas-max", type=float, default=6.0)
    ap.add_argument("--log-flota", default="/tmp/estudio-vigilado.log")
    ap.add_argument("--log", default="/tmp/vigilante-avance.log",
                    help="sólo para citarlo en los avisos; la salida va a stdout")
    args = ap.parse_args()
    PREFIJO[0] = args.prefijo

    if args.cada > 600:
        log(f"AVISO: --cada {args.cada}s es más de 10 min; lo bajo a 600.")
        args.cada = 600
    args.cada = max(60, args.cada)

    cargados = cargar_secretos()
    if not (LANZADOR / "scripts" / "vast_instance.py").exists():
        log(f"ERROR: no está el lanzador en {LANZADOR}; sin él no se puede ni "
            f"listar ni destruir nada.")
        return 2
    sys.path.insert(0, str(LANZADOR / "scripts"))
    import vast_instance as V                                   # noqa: E402

    if not (os.environ.get("VAST_AI_API_TOKEN") or os.environ.get("VAST_API_KEY")):
        log(f"ERROR: no hay token de Vast en el entorno. Leídos: "
            f"{', '.join(cargados) or '(ninguno)'}. Sin token no puede destruir "
            f"nada, y callarse sería lo peor que podría hacer.")
        return 2

    log(f"vigilante de avance en marcha: {len(args.sweep)} recorridos, cada "
        f"{args.cada} s, gracia {args.gracia} min, sin-avance {args.sin_avance} min"
        + (", DRY-RUN (no destruye ni alquila)" if args.dry_run else ""))
    log(f"  secretos leídos de: {', '.join(cargados) or '(ninguno)'}")

    estado = {"relanzamientos": 0, "destruidas": 0}
    vueltas = 1 if args.una_vuelta else args.vueltas
    for vuelta in range(1, vueltas + 1):
        log(f"--- vuelta {vuelta}/{vueltas} ---")
        try:
            que = una_vuelta(args, V, estado)
        except Exception as e:                                  # noqa: BLE001
            # Una vuelta que revienta no puede llevarse el vigilante por delante:
            # sin él, nadie mira las máquinas y vuelven las cuatro horas mudas.
            log(f"  ! la vuelta falló ({type(e).__name__}: {e}); sigo en la siguiente")
            que = "seguir"
        if que == "fin":
            log(f"el vigilante se retira. Destruidas {estado['destruidas']} "
                f"máquinas paradas en total.")
            return 0
        if que == "tope":
            return 1
        if vuelta < vueltas:
            time.sleep(args.cada)

    log(f"se acabaron las vueltas. Destruidas {estado['destruidas']} máquinas "
        f"paradas en total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
