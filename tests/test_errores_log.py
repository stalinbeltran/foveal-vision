"""El log de errores: lo que se rompio cuando nadie estaba mirando.

Repartido por la consecuencia del fallo (R10), que aqui es muy desigual:

  1. **que registrar ROMPA a quien lo llama** -> se llama desde el manejador de
     excepciones: si lanza, la peticion que ya iba mal muere de otra forma y
     encima se pierde el motivo original. Es lo unico de aqui sin segunda
     oportunidad.
  2. **que el log se coma el disco** -> el front sondea cada 3 s; un fallo
     permanente son 28.800 lineas al dia en un repo de git.
  3. **que un secreto acabe en el fichero** -> el repo es privado, pero privado
     es un permiso, no un borrado: git no olvida y hay que ROTARLO.
"""
from __future__ import annotations

import json

import pytest

from fv import errores, settings


@pytest.fixture(autouse=True)
def _log_limpio(tmp_path, monkeypatch):
    monkeypatch.setenv("FV_DATA_ROOT", str(tmp_path))
    errores._repeticiones.clear()
    yield
    errores._repeticiones.clear()


def _lineas() -> list[dict]:
    p = errores.mes_path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------- 1. no puede romper a quien llama

def test_registrar_NUNCA_lanza(monkeypatch):
    """Lo unico de este modulo que no tiene segunda oportunidad."""
    monkeypatch.setattr(errores, "_escribir",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disco lleno")))
    assert errores.registrar("x", "y") is None          # y no revienta

    # ni siquiera con basura que no se puede serializar
    class Raro:
        def __str__(self): raise RuntimeError("ni str tengo")
    monkeypatch.undo()
    assert errores.registrar("x", Raro()) is None or True


def test_el_manejador_del_api_registra_y_la_peticion_sigue_contestando(tmp_path,
                                                                       monkeypatch):
    """El 400 sigue siendo un 400 con su razon: el log es un efecto, no un cambio
    de contrato."""
    monkeypatch.setenv("FV_ROOT", str(tmp_path / "codigo"))
    (tmp_path / "codigo").mkdir(parents=True, exist_ok=True)
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    c = TestClient(create_app(), raise_server_exceptions=False)
    r = c.get("/sources/no-existe-jamas/samples/0")
    assert r.status_code in (400, 404)
    assert r.json()["detail"]["code"]                   # la respuesta no cambia
    reg = [l for l in _lineas() if l["origen"] == "api"]
    assert reg and reg[-1]["nivel"] == "rechazo"        # y queda registrado
    assert reg[-1]["donde"].startswith("GET /sources/")


# --------------------------------------------------- 2. no se come el disco

def test_las_repeticiones_se_agrupan():
    """Un fallo permanente en algo que se sondea no puede escribir una linea por
    sondeo. La PRIMERA si se escribe en el acto: agrupar no es perder la unica
    noticia de algo que acaba de tirar el proceso."""
    for _ in range(50):
        errores.registrar("en_bucle", "otra vez", origen="api", donde="GET /runs")
    assert len(_lineas()) == 1                          # 50 sucesos, 1 linea

    # ...y al cerrarse la ventana se dice CUANTAS fueron, en una sola linea mas
    clave = ("error", "en_bucle", "api", "GET /runs")
    errores._repeticiones[clave]["desde"] -= errores.VENTANA_S + 1
    errores.registrar("en_bucle", "otra vez", origen="api", donde="GET /runs")
    ls = _lineas()
    assert len(ls) == 3                                 # 1 + el resumen + la nueva
    assert ls[1]["repeticiones"] == 49
    assert "49 vez/veces mas" in ls[1]["message"]


def test_errores_DISTINTOS_no_se_agrupan_entre_si():
    """Agrupar de mas esconde: dos codigos distintos son dos noticias."""
    errores.registrar("uno", "a", origen="api", donde="GET /runs")
    errores.registrar("otro", "b", origen="api", donde="GET /runs")
    errores.registrar("uno", "a", origen="api", donde="GET /sweeps")
    assert len(_lineas()) == 3


# ------------------------------------------------ 3. no se cuela un secreto

@pytest.mark.parametrize("secreto, marca", [
    ("sk-ant-oat01-" + "A" * 40, "[CLAVE-ANTHROPIC]"),
    ("ghp_" + "B" * 36, "[TOKEN-GITHUB]"),
    ("github_pat_" + "C" * 40, "[TOKEN-GITHUB]"),
    ("dop_v1_" + "d" * 64, "[TOKEN-DIGITALOCEAN]"),
    ("123456789:AAE" + "e" * 32, "[TOKEN-TELEGRAM]"),
])
def test_los_secretos_se_redactan_en_mensaje_y_traza(secreto, marca):
    errores.registrar("con_secreto", f"fallo con {secreto} dentro",
                      hint=f"y en el hint: {secreto}",
                      traza=f"Traceback ... {secreto} ...")
    crudo = errores.mes_path().read_text(encoding="utf-8")
    assert secreto not in crudo
    assert marca in crudo


def test_el_token_de_la_url_no_queda_en_el_log():
    """`?t=` es como viaja el token de la web app a un navegador de movil, asi
    que aparece en la RUTA -- que es justo lo que se registra en `donde`."""
    errores.registrar("x", "y", donde="GET /predict?t=Sup3rSecret0Token")
    crudo = errores.mes_path().read_text(encoding="utf-8")
    assert "Sup3rSecret0Token" not in crudo and "?t=[TOKEN]" in crudo


# ------------------------------------------------------- el sitio y la consulta

def test_vive_en_el_repo_de_datos_con_las_carpetas_de_siempre():
    """R7 (lo produce el sistema, no quien lo transporta) y la misma forma de
    carpetas que `conversaciones/` y los runs. Un fichero por mes (R8)."""
    errores.registrar("donde_vivo", "x")
    p = errores.mes_path()
    assert p.parent.parent.parent == settings.errores_root()
    assert p.parent.name.endswith(("-enero", "-febrero", "-marzo", "-abril", "-mayo",
                                   "-junio", "-julio", "-agosto", "-septiembre",
                                   "-octubre", "-noviembre", "-diciembre"))
    assert p.name.endswith(".jsonl") and len(p.stem) == 7      # 2026-09


def test_la_consulta_filtra_cuenta_y_pagina_en_el_servidor():
    for i in range(30):
        errores.registrar(f"code{i % 3}", f"mensaje {i}", origen="job" if i % 2 else "api",
                          nivel="rechazo" if i % 5 else "error")
        errores._repeticiones.clear()
    d = errores.consultar(limit=10)
    assert len(d["errores"]) == 10 and d["total"] == 30
    # las facetas son lo que hace el filtro usable con muchos: dicen QUE hay
    assert sum(d["facetas"]["code"].values()) == 30
    assert set(d["facetas"]["origen"]) == {"api", "job"}
    # y filtrar de verdad reduce
    solo = errores.consultar(nivel="error")
    assert solo["total"] < 30 and all(e["nivel"] == "error" for e in solo["errores"])
    assert solo["total_sin_filtro"] == 30      # ...y dice cuanto esconde
    # busqueda libre sobre los campos que se leen
    assert errores.consultar(q="mensaje 7")["total"] == 1


def test_una_linea_rota_no_tumba_el_log():
    """Misma regla que el registro del coordinador: se salta, no se muere."""
    errores.registrar("bueno", "x")
    with errores.mes_path().open("a", encoding="utf-8") as fh:
        fh.write("{esto no es json\n")
    assert errores.consultar()["total"] == 1


def test_el_endpoint_devuelve_facetas_y_pagina(tmp_path, monkeypatch):
    monkeypatch.setenv("FV_ROOT", str(tmp_path / "codigo"))
    (tmp_path / "codigo").mkdir(parents=True, exist_ok=True)
    errores.registrar("desde_el_api", "hola", origen="prueba")
    from fastapi.testclient import TestClient
    from fv.api.app import create_app
    c = TestClient(create_app(), raise_server_exceptions=False)
    d = c.get("/errores?limit=5").json()
    assert d["total"] >= 1 and "facetas" in d and "meses" in d
    assert d["facetas"]["origen"].get("prueba") == 1
