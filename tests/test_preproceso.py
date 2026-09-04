"""`aplicaKernel`: la primera capa de las tres planas de un kernel, como preproceso.

Por que existe este fichero
---------------------------
El encargo (2026-09-04) es que la salida de estos kernels se pueda usar
**opcionalmente como pre-procesador de las imagenes de entrada**. En cuanto algo
es un preproceso, sus fallos dejan de ser ruidosos: una entrada mal escalada, un
canal que falta o un relleno metido de mas no revientan -- entrenan una red
distinta y nadie se entera hasta que los numeros no cuadran con nada.

Asi que lo que se fija aqui no es «la funcion corre», es cada suposicion que la
funcion tiene que hacer sobre la entrada:

  · que es LITERALMENTE la capa L1 de esa red, no algo parecido
  · que sin relleno la salida encoge k-1 px, y eso es visible en la forma
  · que una imagen de UN canal recibe el relleno declarado y no uno inventado
  · que un float en 0..255 se NIEGA en vez de multiplicar la salida por 255
  · que un 3-D ambiguo se niega en vez de elegir por su cuenta

⚠ Y uno que no es de la funcion sino del encargo: que los pesos de los tres
experimentos esten TRACKEADOS por git. Un `.pt` que solo existe en esta maquina
se pierde con ella, y es exactamente lo que le habia pasado al 1k3.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

RAIZ = Path(__file__).resolve().parents[1]
COMUN = RAIZ / "experimentos" / "comun"
sys.path.insert(0, str(COMUN))

from preproceso import (CARPETAS, PreprocesoError, aplicaKernel,      # noqa: E402
                        aplicaKernel_1k3, aplicaKernel_1k5, aplicaKernel_1k7,
                        cargar_kernel)

LIGADAS = {"1k3": aplicaKernel_1k3, "1k5": aplicaKernel_1k5, "1k7": aplicaKernel_1k7}
K_DE = {"1k3": 3, "1k5": 5, "1k7": 7}
TODOS = sorted(CARPETAS)


@pytest.fixture(scope="module")
def kernels():
    return {n: cargar_kernel(n) for n in TODOS}


# --------------------------------------------------------- es la capa de la red
@pytest.mark.parametrize("nombre", TODOS)
def test_el_kernel_leido_a_pelo_es_el_de_la_red_construida(nombre):
    """`cargar_kernel` NO construye la red: saca el tensor del `state_dict`.

    Es lo que permite usar el preproceso sin `fv` ni `red_local.py` -- o sea lo
    que hace que la carpeta del experimento se pueda abrir dentro de un ano. El
    precio es que el nombre del parametro (`center_convs.0.weight`) pasa a ser un
    contrato: si el builder renombra esa capa, esto tiene que romperse AQUI y no
    en un entrenamiento seis meses despues.
    """
    import yaml
    sys.path.insert(0, str(RAIZ / "src"))
    sys.path.insert(0, str(RAIZ / "experimentos" / CARPETAS[nombre] / "nn"))
    from fv.models.builder import full_config
    red_local = importlib.util.spec_from_file_location(
        f"red_local_{nombre}",
        RAIZ / "experimentos" / CARPETAS[nombre] / "nn" / "red_local.py")
    mod = importlib.util.module_from_spec(red_local)
    red_local.loader.exec_module(mod)

    cfg = full_config(yaml.safe_load(
        (RAIZ / "configs" / "networks" / f"plana-20-{nombre}.yaml").read_text()))
    m = mod.PlanaSinPadding(cfg)
    ck = torch.load(RAIZ / "experimentos" / CARPETAS[nombre] / "nn" / "pesos" / "best.pt",
                    map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])

    k = cargar_kernel(nombre)
    assert torch.equal(k.peso, m.center_convs[0].weight.detach())
    assert torch.equal(k.sesgo, m.center_convs[0].bias.detach())
    assert m.center_convs[0].padding == (0, 0), "el experimento es SIN relleno"


@pytest.mark.parametrize("nombre", TODOS)
def test_da_lo_mismo_que_pasar_la_entrada_por_la_convolucion_de_la_red(nombre, kernels):
    """Sobre ruido, no sobre una imagen bonita: si difieren, difieren aqui."""
    sys.path.insert(0, str(RAIZ / "src"))
    x = torch.rand(3, 2, 20, 20)
    conv = torch.nn.Conv2d(2, 1, K_DE[nombre], padding=0)
    with torch.no_grad():
        conv.weight.copy_(kernels[nombre].peso)
        conv.bias.copy_(kernels[nombre].sesgo)
        esperado = conv(x)
    assert torch.allclose(aplicaKernel(x, kernels[nombre]), esperado, atol=1e-6)


@pytest.mark.parametrize("nombre", TODOS)
def test_reproduce_el_mapas_npy_que_dejo_escrito_su_experimento(nombre, monkeypatch):
    """La prueba que ata la funcion a lo MEDIDO, no a otra copia del codigo.

    `stop-04/mapas.npy` se calculo en su momento con el modelo vivo y esta
    commiteado. Si `aplicaKernel` da otra cosa, o la funcion miente o el
    experimento no es lo que dice.

    ⚠ EXCEPCION AL FIXTURE GLOBAL, con motivo: `conftest.py` apunta
    `FV_DATA_ROOT` a un temporal vacio para que ningun test toque el repo de
    datos real. Aqui hay que reconstruir las 10 ventanas de DOS canales, y eso
    sale del `windows.npz` commiteado. Se apunta al repo hermano y se LEE; no se
    escribe nada. Si no esta clonado, se salta -- no se inventa un dato.
    """
    datos = RAIZ.parent / "foveal-vision-data"
    ds = datos / "window-datasets" / "dirty1000-80px-16px-r20260827" / "windows.npz"
    if not ds.exists():
        pytest.skip(f"no esta {ds}: sin el dataset no se pueden rehacer las entradas")
    monkeypatch.setenv("FV_DATA_ROOT", str(datos))
    sys.path.insert(0, str(RAIZ / "src"))
    import aplicar_kernels as ak
    ak.RED = RAIZ / "configs" / "networks" / f"plana-20-{nombre}.yaml"
    x, _, _ = ak.entradas(ak.set_visualizacion(10, 2026))

    esperado = np.load(RAIZ / "experimentos" / CARPETAS[nombre] / "evaluacion"
                       / "stop-04-37epocas" / "mapas.npy")
    # los stops se pintaron con `last.pt`, no con `best.pt`
    salida = aplicaKernel(x.numpy(), cargar_kernel(nombre, pesos="last"))
    assert salida.shape == esperado.shape
    assert np.allclose(salida, esperado, atol=1e-6)


# ------------------------------------------------------------------- la forma
@pytest.mark.parametrize("nombre", TODOS)
def test_sin_relleno_la_salida_encoge_k_menos_1_por_lado(nombre):
    """20x20 -> 18x18 / 16x16 / 14x14. Es el punto del encargo: SIN padding."""
    k = K_DE[nombre]
    y = LIGADAS[nombre](np.zeros((2, 20, 20), dtype=np.float32))
    assert y.shape == (1, 20 - k + 1, 20 - k + 1)


@pytest.mark.parametrize("nombre", TODOS)
def test_acepta_cualquier_tamano_no_solo_la_ventana_de_20(nombre):
    """«una entrada cualquiera», dice el encargo: no esta atado a 20x20."""
    k = K_DE[nombre]
    y = LIGADAS[nombre](np.zeros((37, 51), dtype=np.float32))
    assert y.shape == (1, 37 - k + 1, 51 - k + 1)


@pytest.mark.parametrize("nombre", TODOS)
def test_el_lote_entra_y_sale_como_lote(nombre):
    y = LIGADAS[nombre](np.zeros((4, 2, 20, 20), dtype=np.float32))
    assert y.shape[:2] == (4, 1)


@pytest.mark.parametrize("nombre", TODOS)
def test_mas_pequena_que_el_kernel_se_NIEGA(nombre):
    """Sin relleno no cabe ni una posicion. `conv2d` ya se queja, pero con un
    mensaje sobre tensores; este dice cuanto hace falta."""
    with pytest.raises(PreprocesoError, match="no cabe"):
        LIGADAS[nombre](np.zeros((2, 2), dtype=np.float32))


# ------------------------------------------------------ lo que NO se adivina
def test_una_imagen_de_un_canal_recibe_el_relleno_DECLARADO(kernels):
    """El defecto es 0 = «todo pixel es real», que es lo que vale en el interior
    de una pagina (`input_stack`: el canal es `1 - coverage`).

    Se comprueba que es EXACTAMENTE eso y no otra cosa: la salida con un canal
    tiene que ser identica a la de dos canales con el segundo a ceros. Si algun
    dia se cambiara el defecto, este test lo dice en vez de dejar que un
    preproceso entrene sobre otra entrada.
    """
    k = kernels["1k5"]
    gris = np.random.default_rng(7).random((20, 20)).astype(np.float32)
    dos = np.stack([gris, np.zeros_like(gris)])
    assert np.allclose(aplicaKernel(gris, k), aplicaKernel(dos, k), atol=1e-7)

    unos = np.stack([gris, np.ones_like(gris)])
    assert np.allclose(aplicaKernel(gris, k, relleno=1.0), aplicaKernel(unos, k), atol=1e-7)
    # y que el canal de relleno IMPORTA: si no, este test no probaria nada
    assert not np.allclose(aplicaKernel(gris, k), aplicaKernel(gris, k, relleno=1.0))


def test_uint8_se_divide_por_255_igual_que_build_view(kernels):
    k = kernels["1k7"]
    b = (np.random.default_rng(1).random((20, 20)) * 255).astype(np.uint8)
    assert np.allclose(aplicaKernel(b, k),
                       aplicaKernel(b.astype(np.float32) / 255.0, k), atol=1e-6)


def test_un_float_en_0_255_se_NIEGA_en_vez_de_multiplicar_la_salida_por_255(kernels):
    """EL fallo caro de un preprocesador: no revienta, sale 255x y entrena algo.

    El kernel se entreno sobre vistas en [0,1]. Un float fuera de ese rango es
    ambiguo --niveles sin normalizar, o un dato ya centrado-- y adivinar es
    justamente lo que no se puede hacer aqui.
    """
    malo = np.random.default_rng(2).random((20, 20)).astype(np.float32) * 255
    with pytest.raises(PreprocesoError, match="0,1|\\[0,1\\]|escala"):
        aplicaKernel(malo, kernels["1k5"])
    # y se puede decir explicitamente, que es la salida que el error ofrece
    y = aplicaKernel(malo, kernels["1k5"], escala="0-255")
    assert np.allclose(y, aplicaKernel(malo / 255.0, kernels["1k5"]), atol=1e-6)


def test_un_3d_que_no_es_ni_1_ni_2_canales_se_NIEGA(kernels):
    """(C,H,W) y (B,H,W) son indistinguibles por la forma, y elegir mal cambia
    la salida sin fallar. Se niega y dice como desambiguar."""
    with pytest.raises(PreprocesoError, match="canales"):
        aplicaKernel(np.zeros((5, 20, 20), dtype=np.float32), kernels["1k3"])


def test_la_familia_del_dato_se_conserva(kernels):
    """numpy entra, numpy sale; torch entra, torch sale. Un preproceso metido en
    un `Dataset` de torch y otro en un `np.stack` no pueden pedir conversiones
    distintas."""
    k = kernels["1k3"]
    assert isinstance(aplicaKernel(np.zeros((20, 20), dtype=np.float32), k), np.ndarray)
    assert torch.is_tensor(aplicaKernel(torch.zeros(20, 20), k))


def test_sin_sesgo_la_diferencia_es_EXACTAMENTE_el_sesgo(kernels):
    """`con_sesgo=False` no es «casi lo mismo»: es el mapa menos una constante."""
    k = kernels["1k7"]
    x = np.random.default_rng(3).random((2, 20, 20)).astype(np.float32)
    dif = aplicaKernel(x, k) - aplicaKernel(x, k, con_sesgo=False)
    assert np.allclose(dif, float(k.sesgo[0]), atol=1e-6)


def test_no_hace_falta_fv_para_cargar_el_kernel():
    """R3: una pieza que no se puede usar sola no es una pieza.

    `preproceso.py` se importa y carga los tres kernels sin `fv` en el path. Si
    alguien mete ahi un import del repo, esto lo dice: el motivo entero de leer
    el `state_dict` a pelo era poder abrir la carpeta del experimento sin nada mas.
    """
    guion = (
        "import sys; sys.path.insert(0, %r)\n"
        "import preproceso\n"
        "assert 'fv' not in sys.modules, sorted(m for m in sys.modules if m.startswith('fv'))\n"
        "for n in preproceso.CARPETAS: preproceso.cargar_kernel(n)\n"
        "print('ok')\n" % str(COMUN))
    r = subprocess.run([sys.executable, "-c", guion], capture_output=True, text=True,
                       cwd=str(RAIZ))
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


# --------------------------------------------- lo que el encargo pidio aparte
@pytest.mark.parametrize("nombre", TODOS)
@pytest.mark.parametrize("fichero", ["best.pt", "last.pt", "config.json"])
def test_los_pesos_estan_TRACKEADOS_por_git(nombre, fichero):
    """No que existan en disco: que git los tenga.

    El 1k3 termino sus 37 epocas, dejo sus cuatro stops y sus `.pt` se quedaron
    SOLO en el run de `foveal-vision-data` --que ahi ni siquiera esta trackeado--.
    En disco parecia completo. "Lo que no esta empujado, no existe", y en una
    maquina que se rehace sin aviso eso es la diferencia entre tener la red y
    tener que reentrenarla.
    """
    rel = f"experimentos/{CARPETAS[nombre]}/nn/pesos/{fichero}"
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                       capture_output=True, text=True, cwd=str(RAIZ))
    assert r.returncode == 0, f"{rel} NO esta trackeado por git:\n{r.stderr}"
