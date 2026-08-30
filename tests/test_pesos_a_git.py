"""Que los PESOS de un entrenamiento sobrevivan a la maquina que los produjo.

Por que existe este fichero
---------------------------
La noche del 2026-08-29 al 30 se entreno `fov-optimo-p20` (69 epocas en Vast,
f1 0,9430). Se commitearon sus cuatro ficheros de descripcion y NINGUN peso,
porque asi estaba disenado, y al rehacer la maquina la red entrenada dejo de
existir -- no estaba en ninguna rama de ninguno de los cinco repos. Hubo que
reentrenarla desde cero.

La razon por la que ahora SI van a git la puso el dueno: **hay que poder probar a
mano un entrenamiento**. Sin pesos la web app ensena las imagenes sin cajas, y
"la red detecta mal" no se distingue de "no hay red".

⚠ La cadena tiene TRES puertas, y basta que una siga cerrada para volver a
perderlos. Este fichero fija las tres, porque el fallo original era justamente
que dos de ellas estaban cerradas y nadie lo notaba: no hay error, no hay aviso,
sencillamente el fichero no llega.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PESOS = ("best.pt", "last.pt")


@pytest.fixture()
def flota():
    if not (ROOT.parent / "digital-ocean-dropplet-auto-launching").exists():
        pytest.skip("sin el repo del lanzador no se puede importar el modulo")
    spec = importlib.util.spec_from_file_location(
        "estudio_flota", ROOT / "scripts" / "estudio_flota.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------- puerta 1: salir de la maquina
def test_el_tiron_con_pesos_los_PIDE_y_el_normal_no(flota):
    """La sonda corre `find ... -name X` EN la maquina alquilada: lo que no se
    nombre ahi no sube al tar y no sale de la maquina nunca.

    Y el normal NO los pide a proposito: corre cada `--cada` segundos, y cada
    version que llega es una version que git guarda para siempre."""
    normal, con = flota._pull(False), flota._pull(True)
    for n in PESOS:
        assert f"-name {n}" in con
        assert f"-name {n}" not in normal
    # los cuatro pequenos siguen en los dos
    for n in ("metrics.jsonl", "status.json", "config.json", "summary.json"):
        assert f"-name {n}" in normal and f"-name {n}" in con


# ------------------------------------------------- puerta 2: entrar en el disco
def test_LIBRO_deja_extraer_los_pesos(flota):
    """Aunque el tar los traiga, `guardar()` solo extrae los nombres de `LIBRO`.
    Esta era la puerta que los tiraba SIN decir nada."""
    for n in PESOS:
        assert n in flota.LIBRO
    for n in ("metrics.jsonl", "status.json", "config.json", "summary.json"):
        assert n in flota.LIBRO


# ------------------------------------------------- puerta 3: entrar en git
def test_el_gitignore_del_repo_de_datos_los_deja_pasar():
    """La tercera, y la que hace que las otras dos no sirvan de nada si falla.

    Se comprueba contra el `.gitignore` DE VERDAD, no una copia: una copia en el
    test se queda vieja y entonces el test dice que si mientras el repo dice que
    no."""
    # ⚠ La ruta se calcula a mano y NO con `settings.data_root()`: conftest tiene
    # un autouse que apunta el repo de datos a un temporal en todos los tests
    # (`_nunca_el_repo_de_datos_real`), asi que preguntarle devolveria un
    # directorio vacio y este test se saltaria solo -- justo el que importa.
    datos = ROOT.parent / "foveal-vision-data"
    if not (datos / ".gitignore").exists() or not (datos / ".git").exists():
        pytest.skip("el repo de datos no esta clonado aqui")

    def ignorado(rel: str) -> bool:
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel],
                           cwd=str(datos), capture_output=True)
        return r.returncode == 0

    base = "2026/08-agosto/runs/un-run"
    for n in PESOS:
        assert not ignorado(f"{base}/{n}"), f"{n} no llegaria a git"
    # ...y tambien dentro de un recorrido, que es donde caen los de un barrido
    for n in PESOS:
        assert not ignorado(f"2026/08-agosto/sweeps/sw/runs/r1/{n}")
    # cualquier OTRO .pt sigue fuera: son 2 MB por epoca y por run
    assert ignorado(f"{base}/epoch7.pt")
    assert ignorado(f"{base}/optuna.db")


# ------------------------------------------------- la cadencia, que es la regla
def test_la_cadencia_es_un_dato_y_tiene_defecto(flota, monkeypatch):
    """git guarda TODAS las versiones commiteadas: `last.pt` en cada epoca son
    2 MB x 70 epocas x N runs. La cadencia es lo que separa "dos o tres
    versiones por run" de "gigabytes por barrido", asi que es parte de la regla y
    no un ajuste fino."""
    assert flota.EPOCAS_POR_PESOS == 25

    monkeypatch.setenv("FV_EPOCAS_POR_PESOS", "5")
    spec = importlib.util.spec_from_file_location(
        "estudio_flota_2", ROOT / "scripts" / "estudio_flota.py")
    m2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m2)
    assert m2.EPOCAS_POR_PESOS == 5
