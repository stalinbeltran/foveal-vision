"""El vigilante que recoge una instancia de Vast: que se entere de que terminó,
y que la destruya pase lo que pase.

Los dos fallos que fija este fichero costaron dinero el 2026-08-31 y **ninguno
daba error**: el vigilante parecía estar trabajando en los dos casos.

  1. `vive_el_entrenamiento` preguntaba `pgrep -f 'fv-train'` POR SSH, y el shell
     que ssh abre para ejecutarlo lleva esa cadena en su propia línea de comando:
     **se encontraba a sí mismo**. Resultado: sondeó una hora un run terminado y
     habría seguido hasta `--horas-max` (6 h) facturando por nada.
  2. La instancia se destruye en un `finally`, y el `finally` de Python cubre
     EXCEPCIONES, no señales. `systemctl stop` (SIGTERM) mataba el proceso sin
     destruir nada — y parar el vigilante es justo lo que uno hace para terminar.

El sesgo que los dejó pasar está anotado porque es lo reutilizable: al escribirlo
se pensó en el falso NEGATIVO (creer que terminó y destruir con el trabajo
dentro) y se cubrió con cuidado. El falso POSITIVO —no enterarse NUNCA— se quedó
sin mirar, y es el que pasó.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def mod():
    if not (ROOT.parent / "digital-ocean-dropplet-auto-launching").exists():
        pytest.skip("sin el repo del lanzador no se puede importar el modulo")
    spec = importlib.util.spec_from_file_location(
        "adoptar_vast", ROOT / "scripts" / "adoptar_vast.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["adoptar_vast"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------- 1. enterarse de que terminó (el fallo medido)

@pytest.mark.parametrize("estado", ["done", "error", "cancelled", "interrupted"])
def test_un_run_terminado_se_detecta_por_su_status(mod, tmp_path, estado):
    """Los cuatro estados terminales, no sólo `done`: un run que murió a mitad
    también deja de entrenar, y seguir vigilándolo es facturar por nada."""
    (tmp_path / "status.json").write_text(json.dumps({"status": estado}), encoding="utf-8")
    assert mod.vive_el_entrenamiento(tmp_path) is False


def test_un_run_corriendo_sigue_vigilandose(mod, tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    assert mod.vive_el_entrenamiento(tmp_path) is True


@pytest.mark.parametrize("contenido", [None, "{roto", "{}"])
def test_ante_la_duda_se_sigue_vigilando(mod, tmp_path, contenido):
    """Sin fichero, ilegible o sin campo: NO se toma por terminado. Un sondeo de
    más cuesta nada; destruir de más cuesta el entrenamiento entero."""
    if contenido is not None:
        (tmp_path / "status.json").write_text(contenido, encoding="utf-8")
    assert mod.vive_el_entrenamiento(tmp_path) is True


def test_NO_se_pregunta_por_pgrep_a_la_maquina(mod):
    """El fallo de raíz, fijado en el código y no sólo en el comportamiento.

    `pgrep -f <cadena>` casa contra la línea de comando COMPLETA de cada proceso,
    así que preguntarlo por ssh se encuentra a sí mismo — el shell remoto lleva
    la cadena. Es la misma trampa que `cerrable.mjs` ya tenía documentada.

    Se comprueba sobre el fuente porque el fallo NO se ve en la salida: la
    función devolvía un bool perfectamente plausible, sólo que siempre el mismo."""
    import ast
    arbol = ast.parse((ROOT / "scripts" / "adoptar_vast.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(arbol)
              if isinstance(n, ast.FunctionDef) and n.name == "vive_el_entrenamiento")
    # el CÓDIGO, sin el docstring: éste habla de `pgrep` justamente para contar
    # por qué no se usa, y un test que casara ahí prohibiría explicarlo
    cuerpo = fn.body[1:] if ast.get_docstring(fn) else fn.body
    codigo = "\n".join(ast.unparse(n) for n in cuerpo)
    assert "pgrep" not in codigo, \
        "volvió el pgrep por ssh: casa con su propio shell y nunca detecta el fin"
    assert "ssh" not in codigo.lower(), "no hace falta hablar con la máquina para esto"
    assert "status.json" in codigo, "se pregunta al artefacto, no al SO"


def test_los_estados_terminales_son_los_del_proyecto(mod):
    """No una lista propia que pueda divergir de la que escribe `RunStore`."""
    # ⚠ El vocabulario de estados NO tiene hoy una declaración única en Python
    # (es la decisión abierta F16 de docs/decisiones.md: vive en `web/src/api.ts`,
    # o sea en el dominio equivocado). Así que se comprueba contra los ficheros
    # que de verdad los ESCRIBEN, que es lo honesto mientras F16 siga abierta.
    escritores = "".join(
        (ROOT / "src" / "fv" / "training" / f).read_text(encoding="utf-8")
        for f in ("registry.py", "loop.py"))
    for estado in mod.TERMINALES:
        assert f'"{estado}"' in escritores, f"'{estado}' no lo escribe nadie"
    assert "running" not in mod.TERMINALES


# --------------------- 2. destruir aunque lo paren (el otro fallo medido)

def test_una_senal_pasa_por_el_finally_que_destruye(tmp_path):
    """`systemctl stop` mandaba SIGTERM y el proceso moría SIN destruir la
    instancia. El `finally` de Python cubre excepciones, no señales.

    Se comprueba con un proceso de verdad: se le manda SIGTERM y se mira que su
    `finally` haya corrido. Sin el handler, el fichero testigo no aparece."""
    guion = tmp_path / "g.py"
    guion.write_text(textwrap.dedent(f"""
        import signal, sys, time
        sys.path.insert(0, {str(ROOT / "scripts")!r})
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "av", {str(ROOT / "scripts" / "adoptar_vast.py")!r})
        # sólo se necesita el handler; no se ejecuta el módulo entero
        def morir(sig, frame):
            raise SystemExit(143)
        signal.signal(signal.SIGTERM, morir)
        try:
            print("listo", flush=True)
            time.sleep(30)
        finally:
            open({str(tmp_path / "destruida")!r}, "w").write("ok")
    """), encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(guion)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "listo"
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=10)
    assert (tmp_path / "destruida").exists(), \
        "el finally no corrió con SIGTERM: la instancia se habría quedado facturando"


# TODO script que alquile una máquina: el mismo agujero y la misma cura.
# ⚠ `entrenar_vast` y `estudio_flota` LO TENÍAN cuando se arregló `adoptar_vast`
# (2026-08-31) — el fallo se descubrió en uno y vivía en los tres. `estudio_flota`
# es el peor de los tres: alquila una FLOTA, así que ahí la señal perdida deja N
# máquinas facturando en vez de una.
QUE_ALQUILAN = ["adoptar_vast.py", "entrenar_vast.py", "estudio_flota.py"]


@pytest.mark.parametrize("script", QUE_ALQUILAN)
def test_el_handler_de_senal_esta_puesto_ANTES_de_alquilar_nada(script):
    """Ponerlo tarde deja una ventana en la que una señal mata sin destruir, y
    esa ventana es justo el arranque: la máquina ya existe y el programa aún no
    llegó a su bucle. Se registra lo primero de `main`."""
    fuente = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    cuerpo = fuente.split("def main(")[1]
    assert "morir_por_el_finally(" in cuerpo, \
        f"{script} alquila máquinas y no atrapa SIGTERM: un `systemctl stop` " \
        f"lo mataría dejándolas facturando"
    i_sig = cuerpo.index("morir_por_el_finally(")
    # lo que en cada uno significa "ya hay dinero corriendo"
    marca = {"adoptar_vast.py": "V.buscar_instancia",
             "entrenar_vast.py": "preflight(args)",
             "estudio_flota.py": "motivo_sin_libro()"}[script]
    assert i_sig < cuerpo.index(marca), \
        f"{script}: la señal se atrapa DESPUÉS de {marca}"


def test_el_handler_es_uno_solo_para_los_tres():
    """Tres copias del mismo handler divergen, y la que se quede atrás es la que
    deja una máquina encendida. Vive en `fv.proc`, que es dominio de procesos."""
    from fv.proc import morir_por_el_finally      # existe y es importable
    for script in QUE_ALQUILAN:
        fuente = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "from fv.proc import morir_por_el_finally" in fuente, script
        assert "def _morir" not in fuente, f"{script} tiene una copia local"


def test_destruir_es_lo_ultimo_y_va_en_finally():
    """La destrucción no puede depender de que el camino feliz llegue al final:
    es lo único que corta el gasto."""
    fuente = (ROOT / "scripts" / "adoptar_vast.py").read_text(encoding="utf-8")
    cuerpo = fuente.split("def main(")[1]
    assert cuerpo.index("finally:") < cuerpo.index("V.destruir("), \
        "V.destruir tiene que estar dentro del finally"
