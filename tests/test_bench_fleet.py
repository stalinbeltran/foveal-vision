"""La clave con la que `bench_fleet.py` entra en sus droplets: se PREGUNTA al lanzador.

Hasta el 2026-10-01 estaba cableada `~/.ssh/do_droplet`, y en una máquina de la
flota ese fichero no existe -la flota entra con la clave de flota desde el
2026-09-11-: el benchmark de vCPU moría al empezar en cualquier dev nuevo. Ahora
la ruta la da `do_droplet.py clave-de-entrada`, la misma elección con la que
entran `ssh` y `launch` del lanzador.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FUENTE = ROOT / "scripts" / "bench_fleet.py"


@pytest.fixture()
def bf():
    spec = importlib.util.spec_from_file_location("bench_fleet", FUENTE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_no_cablea_la_clave():
    """Cablear la ruta es volver al fallo la próxima vez que el lanzador cambie de clave."""
    # Sobre el CÓDIGO, no sobre el texto: el comentario que cuenta la historia la nombra.
    assert '".ssh" / "do_droplet"' not in FUENTE.read_text(encoding="utf-8")


def test_ssh_y_scp_usan_la_que_da_el_lanzador(bf, tmp_path, monkeypatch):
    clave = tmp_path / "do_flota"
    clave.write_text("privada de mentira\n")
    # La ruta va en la ÚLTIMA línea: antes puede venir el aviso de la caída a la flota.
    monkeypatch.setattr(bf, "lanzador", lambda *a, **k: f"  AVISO: se usa la de flota\n{clave}\n")
    bf.CLAVE = bf.clave_de_entrada()
    assert bf.CLAVE == clave
    base = bf.ssh_base("203.0.113.7", 22)
    assert base[base.index("-i") + 1] == str(clave)
    assert 'str(CLAVE)' in FUENTE.read_text(encoding="utf-8").split('"scp", "-P"', 1)[1][:200], \
        "el scp del dataset tiene que entrar con la misma clave que el ssh"


def test_sin_clave_se_niega_antes_de_crear_nada(bf, monkeypatch):
    monkeypatch.setattr(bf, "lanzador",
                        lambda *a, **k: "ERROR: No hay clave con la que entrar en los droplets\n")
    with pytest.raises(SystemExit):
        bf.clave_de_entrada()


def test_una_ruta_que_no_existe_tambien_se_niega(bf, tmp_path, monkeypatch):
    """Un fichero que no está no autentica nada: mejor no crear el droplet."""
    monkeypatch.setattr(bf, "lanzador", lambda *a, **k: f"{tmp_path / 'no-esta'}\n")
    with pytest.raises(SystemExit):
        bf.clave_de_entrada()
