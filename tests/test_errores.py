"""Que toda negativa se pueda PRODUCIR, no sólo declarar.

Por qué existe
--------------
El proyecto ya obliga a que cada error lleve `code + message + hint` (U5.2,
api.md R4) y hay validador que lo comprueba. Pero eso mira la FORMA del error, no
su VERDAD -- y la forma se cumple perfectamente con un diagnóstico falso.

*Medido el 2026-09-01:* de los 109 códigos declarados en `src/`, **44 (40 %) no
los nombraba ningún test**. Uno de ellos era `checkpoint_incompatible`, que ese
mismo día le dijo al dueño *«reentrena el run»* cuando lo que hacía falta era
reiniciar el servicio: el mensaje mandaba a gastar en Vast para arreglar un
modelo que estaba perfecto. Un `raise` que ningún test ejecuta es una conjetura
sobre un estado que nadie ha visto.

⚠ Esto es un TRINQUETE, no una nota de deuda
---------------------------------------------
Exigir los 44 de golpe dejaría la suite en rojo hasta que alguien los cubriera
todos, y una prueba que está roja desde hace un mes deja de leerse -- que es
exactamente el fallo que este proyecto llama «el aviso que sale siempre».

Así que la lista de descubiertos se CONGELA abajo y el test comprueba dos cosas,
y la segunda es la que impide que la lista se pudra:

  1. **ningún código NUEVO** puede quedarse sin test;
  2. **ningún código de la lista puede estar YA cubierto** -- si lo cubriste,
     sácalo de la lista en el mismo commit. Sin esto, `SIN_TEST` sería un
     cementerio que sólo crece de nombre.

⚠ Lo que este test NO garantiza
--------------------------------
Comprueba que un test **nombra** el código en una aserción, no que la rama se
haya ejecutado de verdad, y mucho menos que el diagnóstico sea CIERTO. Lo
primero es un proxy barato; lo segundo es juicio y no lo compra ninguna
herramienta. Lo que sí compra este test es que **exista un sitio donde ese juicio
se pudo ejercer**: la técnica que encuentra el diagnóstico falso es escribir el
test de la OTRA causa, y para eso hay que sentarse a producir el error.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Los dos sitios donde este proyecto declara un codigo: el dict de `problems`
# (validacion, que se DEVUELVE) y el primer argumento de una *Error (que se LANZA).
DECLARA = (re.compile(r'"code":\s*"([a-z0-9_]+)"'),
           re.compile(r'(?:Error|error)\(\s*\n?\s*"([a-z0-9_]+)"'))
# Un test lo CUBRE si el codigo aparece como STRING DE VERDAD en el test -- no en
# un comentario y no en un docstring. La distincion no es tiquismiquis: la primera
# version de esto casaba formas de asercion con una regex y daba por descubiertos
# `unknown_regions` y `mask_needs_border`, que estan asertados en la forma
# `[p["code"] for p in problems] == ["unknown_regions"]`. Un trinquete con falsos
# positivos bloquea trabajo legitimo y acaba desactivado, que es peor que uno laxo.
#
# Y al reves: `sweep_exists` sale en un DOCSTRING de tests/test_studies.py y no
# esta asertado en ninguna parte. Contarlo seria un falso aprobado.
#
# `ast` distingue las dos cosas sin ambiguedad, y una regex no.


# ⚠ CONGELADA el 2026-09-01 con 44 nombres. SOLO PUEDE ENCOGER: cubrir uno es
# borrarlo de aqui en el mismo commit, y el test lo exige. Si al anadir un error
# nuevo te dan ganas de meterlo aqui, eso es justo lo que esta lista existe para
# impedir -- escribe el test.
SIN_TEST = {
    "already_published", "base_network_xor_value", "channels_must_be_positive",
    "crop_size_mismatch", "empty_source", "images_budget_exceeded",
    "images_not_uniform", "job_not_found", "kernel_exceeds_band",
    "legacy_geometry_incomplete", "name_required", "network_exists",
    "network_not_found", "no_aprobada", "no_feasible_border",
    "no_points_selected", "no_reviewable_datasets", "no_scored_trials",
    "no_step_awaiting", "point_out_of_range", "point_required", "recipe_exists",
    "sweep_exists",
    "recipe_in_use", "recipe_not_found", "run_not_found",
    "run_without_provenance", "sample_not_found", "source_empty",
    "study_exists", "study_not_found", "sweep_not_found",
    "unknown_cost_metric", "unknown_merge", "unknown_optimizer",
    "unknown_pad_mode", "unknown_pool_mode", "unknown_scheduler",
    "window_dataset_exists", "window_dataset_in_use", "window_dataset_missing",
    "window_dataset_required", "window_not_found",
}


def _declarados() -> dict[str, str]:
    """codigo -> el primer fichero de src/ que lo declara."""
    out: dict[str, str] = {}
    for f in sorted((RAIZ / "src").rglob("*.py")):
        txt = f.read_text(encoding="utf-8")
        for patron in DECLARA:
            for m in patron.finditer(txt):
                out.setdefault(m.group(1), str(f.relative_to(RAIZ)))
    return out


def _cubiertos() -> set[str]:
    """Los codigos que algun test nombra como dato, no como prosa.

    ⚠ Este fichero se excluye a proposito: su docstring y su lista `SIN_TEST`
    nombran codigos, y contarlos haria que el test se diera cobertura a si mismo.
    """
    vistos: set[str] = set()
    for f in sorted((RAIZ / "tests").rglob("*.py")):
        if f.name == Path(__file__).name:
            continue
        arbol = ast.parse(f.read_text(encoding="utf-8"))
        docs = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and ast.get_docstring(nodo):
                cuerpo = nodo.body[0]
                if isinstance(cuerpo, ast.Expr):
                    docs.add(id(cuerpo.value))
        for nodo in ast.walk(arbol):
            if (isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
                    and id(nodo) not in docs):
                vistos.add(nodo.value)
    return vistos


def test_ningun_codigo_de_error_nuevo_se_queda_sin_test():
    """La mitad que mira hacia adelante: un `raise` nuevo trae su test."""
    declarados = _declarados()
    descubiertos = sorted(set(declarados) - _cubiertos() - SIN_TEST)
    assert not descubiertos, (
        "estos codigos de error no los produce ningun test, y no estan en la "
        "lista congelada:\n" +
        "\n".join(f"    {c:38s} {declarados[c]}" for c in descubiertos) +
        "\n\n  Escribe el test que lo produce. Y si el error dice una CAUSA, "
        "escribe tambien el de la otra causa que da el mismo sintoma: es lo que "
        "encuentra los diagnosticos falsos (2026-09-01, checkpoint_incompatible).")


def test_la_lista_congelada_solo_puede_encoger():
    """La mitad que impide que la lista se pudra.

    Sin esto `SIN_TEST` seria un cementerio: se cubre un codigo, nadie lo saca, y
    el numero de la deuda deja de significar nada."""
    cubiertos = _cubiertos()
    ya_cubiertos = sorted(c for c in SIN_TEST if c in cubiertos)
    assert not ya_cubiertos, (
        "estos ya tienen test y siguen en la lista congelada; sacalos de "
        "SIN_TEST en el mismo commit:\n" +
        "\n".join(f"    {c}" for c in ya_cubiertos))
    # y ningun nombre fantasma: un codigo que se borro de src/ no puede seguir
    # contando como deuda
    declarados = set(_declarados())
    fantasmas = sorted(c for c in SIN_TEST if c not in declarados)
    assert not fantasmas, (
        "estos ya no existen en src/ y siguen en la lista:\n" +
        "\n".join(f"    {c}" for c in fantasmas))


def test_la_deuda_esta_medida_y_es_la_que_dice_el_documento():
    """El numero que se cita en 5-invariantes.md sale de aqui, no de la memoria."""
    declarados = _declarados()
    assert len(declarados) >= 100, len(declarados)
    assert len(SIN_TEST) < len(declarados) / 2, (
        f"{len(SIN_TEST)} de {len(declarados)} sin test: la deuda crecio en vez "
        f"de encoger")
