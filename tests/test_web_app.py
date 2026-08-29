"""El envoltorio de Telegram del servicio web (`scripts/web_app.py`).

Metodo de tests.md §2: se testea la costura, no la funcion. La costura aqui es
`unidad_estado()` -- que distingue "no instalado" (pide `install-service`) de
"instalado, parado" (pide `systemctl start`) -- contra los subcomandos que
deberian consultarla. `estado` lo hacia; `arrancar` y `parar` no, asi que el
comando al que llegas cuando quieres levantar la app era justo el que perdia la
distincion y devolvia el error crudo de systemd. Medido el 2026-08-29 en un dev
que nacio sin el servicio.

El caso "no se" tiene su propio test a proposito: es la mitad que se rompe al
arreglar la otra. Un atajo que corte tambien ahi convierte "no pude comprobarlo"
en "no se puede", que es un falso negativo que impide arrancar algo que si
funciona.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _modulo(nombre: str):
    import importlib.util
    ruta = Path(__file__).resolve().parents[1] / "scripts" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(f"_{nombre}_test", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kw) -> argparse.Namespace:
    base = {"port": 8010, "lineas": 40}
    base.update(kw)
    return argparse.Namespace(**base)


def _espia(mod, monkeypatch, code=0, salida=""):
    """Sustituye `corre` y devuelve la lista de comandos que se intentaron."""
    llamadas: list[list[str]] = []

    def falso(cmd, cwd=None, timeout=900):
        llamadas.append(list(cmd))
        return code, salida

    monkeypatch.setattr(mod, "corre", falso)
    return llamadas


# ------------------------------------------------- arrancar sin la unidad puesta


def test_arrancar_sin_unidad_da_el_remedio(monkeypatch, capsys):
    mod = _modulo("web_app")
    monkeypatch.setattr(mod, "unidad_estado", lambda: "no instalado")
    _espia(mod, monkeypatch)

    assert mod.cmd_arrancar(_args()) == 1
    salida = capsys.readouterr().out
    assert "no instalado" in salida
    assert "install-service" in salida
    # Y NO puede haber dicho que arranco: el codigo viejo llegaba al remedio
    # igual, pero por `cmd_estado` y despues de anunciar un arranque que no
    # ocurrio. Sin esta linea el test pasaba con el fallo puesto.
    assert "arrancado" not in salida


def test_arrancar_sin_unidad_ni_llama_a_systemctl(monkeypatch, capsys):
    """No es solo el texto: pedirle a systemd algo que no existe y traducir su
    error despues es como se perdio el remedio la primera vez."""
    mod = _modulo("web_app")
    monkeypatch.setattr(mod, "unidad_estado", lambda: "no instalado")
    llamadas = _espia(mod, monkeypatch)

    mod.cmd_arrancar(_args())
    assert llamadas == []


def test_arrancar_con_estado_desconocido_lo_intenta_igual(monkeypatch, capsys):
    """"no se" NO es "no instalado". Si systemctl no contesta, se intenta y
    habla el error de verdad; cortar aqui seria negarse a arrancar algo sano."""
    mod = _modulo("web_app")
    monkeypatch.setattr(mod, "unidad_estado", lambda: "no se")
    llamadas = _espia(mod, monkeypatch, code=1, salida="boom")

    assert mod.cmd_arrancar(_args()) == 1
    assert any("start" in c for c in llamadas)
    assert "boom" in capsys.readouterr().out


# ---------------------------------------------------- parar sin la unidad puesta


def test_parar_sin_unidad_no_es_un_fallo(monkeypatch, capsys):
    """Pediste que no corriera y no corre: el estado pedido se cumple."""
    mod = _modulo("web_app")
    monkeypatch.setattr(mod, "unidad_estado", lambda: "no instalado")
    llamadas = _espia(mod, monkeypatch)

    assert mod.cmd_parar(_args()) == 0
    assert llamadas == []
    salida = capsys.readouterr().out
    assert "nada que parar" in salida
    # Pediste apagar: ofrecerte instalar seria contestar a otra pregunta.
    assert "install-service" not in salida


# ------------------------------------------------------- el remedio, en un sitio


def test_el_remedio_lo_escribe_un_solo_sitio(monkeypatch, capsys):
    """`estado` y `arrancar` tienen que decir LO MISMO. Dos copias divergen y la
    que se depura luego es siempre la que no escribiste tu."""
    mod = _modulo("web_app")
    monkeypatch.setattr(mod, "unidad_estado", lambda: "no instalado")
    monkeypatch.setattr(mod, "escuchando", lambda p: False)
    monkeypatch.setattr(mod, "puerto_abierto", lambda p: False)
    _espia(mod, monkeypatch)

    mod.cmd_estado(_args())
    de_estado = capsys.readouterr().out
    mod.cmd_arrancar(_args())
    de_arrancar = capsys.readouterr().out

    assert mod.ayuda_instalar().strip() in de_estado
    assert mod.ayuda_instalar().strip() in de_arrancar


# --------------------------------------------------------------------- el log


def test_log_calla_el_aviso_de_journalctl(monkeypatch, capsys):
    """Sin `-q`, journalctl mete tres lineas de "Hint:" cuando quien pregunta no
    esta en `adm`/`systemd-journal` -- y el usuario del servicio no lo esta. Son
    tres de las cinco lineas de la respuesta, en una pantalla de movil."""
    mod = _modulo("web_app")
    llamadas = _espia(mod, monkeypatch)

    mod.cmd_log(_args())
    assert llamadas, "cmd_log tiene que preguntarle a journalctl"
    assert "-q" in llamadas[0]


def test_log_sin_entradas_lo_dice_con_palabras(monkeypatch, capsys):
    """El fallback ya estaba escrito y era codigo muerto: sin `-q`, journalctl
    imprime "-- No entries --", que no es vacio, asi que el `or` nunca corria."""
    mod = _modulo("web_app")
    _espia(mod, monkeypatch, code=0, salida="")

    mod.cmd_log(_args())
    assert "sin log para" in capsys.readouterr().out
