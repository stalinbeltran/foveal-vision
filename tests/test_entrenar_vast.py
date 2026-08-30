"""Los frenos de `entrenar_vast.py`: todos ANTES de alquilar.

No se toca Vast aqui. Lo que se fija es que cada motivo para no poder entrenar se
descubra en el preflight, porque descubrirlo despues es una maquina facturando
para nada (R11) -- y que lo diga con la frase que lo deja claro.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANZADOR = ROOT.parent / "digital-ocean-dropplet-auto-launching"


@pytest.fixture()
def mod(world, monkeypatch):
    if not (LANZADOR / "scripts" / "vast_instance.py").exists():
        pytest.skip("sin el repo del lanzador no se puede importar el modulo")
    spec = importlib.util.spec_from_file_location(
        "entrenar_vast", ROOT / "scripts" / "entrenar_vast.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # el token es lo ULTIMO que mira el preflight: aqui se neutraliza para que
    # los tests fijen los frenos de datos, no la configuracion de la maquina
    monkeypatch.setattr(m.V, "load_env", lambda: None)
    monkeypatch.setattr(m.V, "token", lambda: "x")
    return m


class Args:
    def __init__(self, **kw):
        self.name = "nuevo"
        self.dataset = "mini-b8"
        self.network = "fov16-optimo"
        self.recipe = "plan40"
        self.continuar = False
        self.__dict__.update(kw)


def test_sin_windows_npz_no_se_alquila(mod, world):
    with pytest.raises(SystemExit):
        mod.preflight(Args(dataset="no-existe"))


def test_un_run_que_ya_existe_no_se_pisa(mod, world):
    """`fv-train` se niega a sobrescribir; aqui tiene que verse ANTES de gastar,
    no cuando la maquina ya esta alquilada y el comando falla alli."""
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    train("ya-esta", world["dataset"], "n",
          __import__("tests.conftest", fromlist=["TINY_NET"]).TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=RunStore())
    with pytest.raises(SystemExit):
        mod.preflight(Args(name="ya-esta", dataset=world["dataset"]))
    # ...y con --continuar SI se deja, porque tiene last.pt
    ctx = mod.preflight(Args(name="ya-esta", dataset=world["dataset"], continuar=True))
    assert ctx["continuar"] is True


def test_continuar_algo_que_no_existe_no_se_alquila(mod, world):
    with pytest.raises(SystemExit):
        mod.preflight(Args(name="no-existe", dataset=world["dataset"], continuar=True))


def test_continuar_sin_last_pt_no_se_alquila(mod, world):
    """Sin `last.pt` no hay desde donde seguir, y subir el run igualmente seria
    alquilar para que `fv-continue` falle alli."""
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    st = RunStore()
    train("sin-last", world["dataset"], "n",
          __import__("tests.conftest", fromlist=["TINY_NET"]).TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=st)
    (st.path("sin-last") / "last.pt").unlink()
    with pytest.raises(SystemExit):
        mod.preflight(Args(name="sin-last", dataset=world["dataset"], continuar=True))


def test_los_PESOS_estan_en_lo_que_se_baja_cada_sonda(mod):
    """La diferencia entera con el libro de la flota, y el motivo de que este
    script exista: alli los `.pt` se quedan en la maquina hasta el tar final."""
    assert "best.pt" in mod.TRAER and "last.pt" in mod.TRAER
    assert "metrics.jsonl" in mod.TRAER


def test_el_destino_SSH_se_RE_PREGUNTA_en_cada_intento(mod):
    """Las dos trampas que costaron los dos primeros intentos (2026-08-30), y que
    este repo ya tenia medidas en `estudio_flota`: el banner no es el login, y el
    `host:puerto` que da la API al arrancar puede no ser el definitivo."""
    fuente = (ROOT / "scripts" / "entrenar_vast.py").read_text(encoding="utf-8")
    assert "def conectar(" in fuente
    # el destino se resuelve DENTRO del bucle, no una vez antes
    cuerpo = fuente[fuente.index("def conectar("):fuente.index("def dir_remoto(")]
    assert "while time.time() < fin:" in cuerpo
    assert "V.instancia(iid)" in cuerpo and "V.ssh_destino(info)" in cuerpo
    assert cuerpo.index("while ") < cuerpo.index("V.ssh_destino(info)")
    # y se comprueba con un comando AUTENTICADO, no con el banner
    assert "ssh_capture" in cuerpo
    # ...y NO se llama a `esperar_ssh`, que es la que solo mira el banner. Se
    # comprueba la LLAMADA (con parentesis): el docstring la nombra a proposito
    # para explicar por que no se usa, y buscar el nombre a secas casaria con eso.
    assert "V.esperar_ssh(" not in cuerpo


def test_la_ruta_del_run_en_la_maquina_NO_esta_cableada(mod):
    """Un run suelto no se escribe en `data/runs/<name>` sino bajo la carpeta del
    mes. Cablearlo dejaba la maquina entrenando bien y este lado bajando cero
    ficheros -- el fallo que parece "no entreno" y cuesta el alquiler entero."""
    fuente = (ROOT / "scripts" / "entrenar_vast.py").read_text(encoding="utf-8")
    assert "def dir_remoto(" in fuente
    assert "RunStore().path(" in fuente, "se le pregunta al store, no se adivina"
    assert "/root/bench/data/runs/{name}" not in fuente


def test_no_viaja_ningun_secreto_a_la_maquina(mod):
    """Son ordenadores de desconocidos alquilados por minutos."""
    assert set(mod.ENVIA) == {"src", "scripts", "configs", "pyproject.toml"}
    assert {".git", ".venv"} <= mod.EXCLUYE
    fuente = (ROOT / "scripts" / "entrenar_vast.py").read_text(encoding="utf-8")
    for secreto in ("BOT_TOKEN", "VAST_AI_API_TOKEN", "GITHUB_TOKEN", "DO_TOKEN"):
        assert secreto not in fuente


def test_no_se_sube_un_run_que_esta_corriendo_aqui(mod, world):
    """Dos escrituras sobre el mismo run, y ademas su `status.json` viaja con un
    `pid` de ESTA maquina: en la alquilada ese numero puede existir por
    coincidencia y `reconcile` lo leeria como "sigue vivo"."""
    from fv.training.loop import train
    from fv.training.recipe import Recipe
    from fv.training.registry import RunStore
    st = RunStore()
    train("corriendo", world["dataset"], "n",
          __import__("tests.conftest", fromlist=["TINY_NET"]).TINY_NET, "r",
          Recipe(epochs=1, batch_size=32), store=st)
    st.set_status("corriendo", "running", pid=1)
    with pytest.raises(SystemExit):
        mod.preflight(Args(name="corriendo", dataset=world["dataset"], continuar=True))


def test_un_rechazo_de_clave_NO_se_reintenta_a_ciegas(mod):
    """`estudio_flota.sellar` ya fijó la asimetría: el transporte mejora
    esperando, la autenticación no. Aquí faltaba, y son 12 minutos de reintentos
    ciegos con el diagnóstico apuntando al sitio equivocado (2026-08-30)."""
    cuerpo = _cuerpo(mod, "def conectar(", "def dir_remoto(")
    assert "permission denied" in cuerpo.lower()
    assert "deniegos" in cuerpo
    assert "register-key" in cuerpo, "el error tiene que decir el arreglo"


def test_el_stderr_de_ssh_NO_se_tira(mod):
    """Lo que escondía las dos trampas: `V.ssh_capture` devuelve solo stdout, y
    el motivo del fallo viaja por stderr. Sin él, "rechaza la clave" y "aún no
    levanta sshd" se ven igual: rc=255 y nada más."""
    cuerpo = _cuerpo(mod, "def _ssh(", "def conectar(")
    assert "proc.stderr" in cuerpo
    assert "capture_output=True" in cuerpo
    # y se usa: `conectar` mira el stderr, no solo el codigo
    con = _cuerpo(mod, "def conectar(", "def dir_remoto(")
    assert "err" in con and "_ssh(" in con


def _cuerpo(mod, desde: str, hasta: str) -> str:
    f = (ROOT / "scripts" / "entrenar_vast.py").read_text(encoding="utf-8")
    return f[f.index(desde):f.index(hasta)]


# --- cuando cambiar de maquina (R13: el criterio, escrito antes de mirar) ----

def test_una_maquina_que_se_DEGRADA_se_cambia(mod):
    """El caso "iba bien y se puso lenta": se mide contra SI MISMA. El umbral
    1.35 no es inventado, es el que ya usa `estudio_flota --umbral-degradacion`."""
    assert mod.UMBRAL_DEGRADACION == 1.35
    v = mod.veredicto_maquina([80, 82, 81, 130, 135, 132], mejor=80)
    assert v and v["motivo"] == "degradada"
    assert "se puso lenta" in v["detalle"]


def test_una_maquina_ESTABLE_no_se_toca(mod):
    v = mod.veredicto_maquina([80, 82, 81, 83, 79, 84], mejor=80)
    assert v is None


def test_una_maquina_que_NACE_lenta_tambien_se_cambia(mod):
    """El otro caso, y es distinto: el marketplace da maquinas muy distintas por
    el mismo precio. Eso no se ve contra si misma --es lenta desde la primera
    epoca-- sino contra la MEJOR de esta corrida."""
    v = mod.veredicto_maquina([200, 205, 198], mejor=80)
    assert v and v["motivo"] == "lenta"
    assert "nacio lenta" in v["detalle"]


def test_sin_una_MEJOR_con_la_que_comparar_no_se_juzga_lenta(mod):
    """La primera maquina de una corrida no puede ser 'lenta': no hay contra que.
    Cambiarla seria tirar la unica referencia que tenemos."""
    assert mod.veredicto_maquina([200, 205, 198], mejor=None) is None


def test_una_epoca_suelta_NO_decide(mod):
    """Una epoca mide el arranque (cache fria, primer batch), no la maquina."""
    assert mod.veredicto_maquina([500], mejor=80) is None
    assert mod.veredicto_maquina([500, 90], mejor=80) is None
    # con 3 ya se puede juzgar 'lenta', pero no 'degradada' (necesita 6)
    assert mod.veredicto_maquina([90, 91, 92], mejor=88) is None


def test_el_aviso_NUNCA_rompe_el_entrenamiento(mod, monkeypatch):
    """Es una comodidad; la fuente de verdad es el log y el run en disco. Un
    fallo del avisador no puede tumbar horas de entrenamiento."""
    def explota(*a, **k):
        raise OSError("no hay node")
    monkeypatch.setattr(mod.subprocess, "run", explota)
    mod.avisar("lo que sea")          # no lanza


def test_el_techo_de_gasto_se_mira_ANTES_de_alquilar(mod):
    """Descubrir que no cabe con la maquina ya encendida es justo el gasto que el
    techo existe para evitar."""
    fuente = (ROOT / "scripts" / "entrenar_vast.py").read_text(encoding="utf-8")
    bucle = fuente[fuente.index("    while True:\n        # "):fuente.index("        r = una_maquina(")]
    assert "presupuesto" in bucle and "break" in bucle


def test_la_destruccion_no_depende_de_la_decision_de_seguir(mod):
    """`una_maquina` destruye SIEMPRE en su `finally`; quien decide si alquila
    otra es el bucle de fuera. Si la destruccion viviera en el bucle, cada camino
    de salida nuevo seria una fuga posible."""
    fuente = (ROOT / "scripts" / "entrenar_vast.py").read_text(encoding="utf-8")
    cuerpo = fuente[fuente.index("def una_maquina("):fuente.index("def _epocas(")]
    assert "finally:" in cuerpo and "V.destruir(iid)" in cuerpo


def test_los_ficheros_se_colocan_de_forma_ATOMICA(mod):
    """`best.pt` se lee MIENTRAS se reemplaza: la pantalla de revision usa el
    modelo con el entrenamiento en marcha. Con un rename atomico, quien lee
    obtiene la version vieja o la nueva, nunca media.

    Y el temporal tiene que ir AL LADO del destino: `os.replace` solo es atomico
    dentro del mismo sistema de ficheros. En /tmp --que en muchas maquinas es un
    tmpfs aparte-- daria EXDEV y la descarga fallaria en silencio."""
    cuerpo = _cuerpo(mod, "def traer(", "def main(")
    assert "dir=str(destino)" in cuerpo, "el temporal va junto al destino"
    assert ".replace(local)" in cuerpo
    # y un fallo al colocar se DICE, no se traga
    assert "AVISO: no pude colocar" in cuerpo
