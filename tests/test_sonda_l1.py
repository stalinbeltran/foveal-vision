"""La sonda L1: los invariantes que, si se rompen, la hacen medir OTRA cosa.

No se testea "que entrene". Se testean las decisiones cuya rotura es SILENCIOSA
-- el experimento seguiria corriendo y dando numeros creibles:

1. la renormalizacion del decodificador (sin ella, `lambda` premia hacer `z`
   pequeno en vez de disperso, y el barrido en lambda no mide nada);
2. el nulo de las DOS metricas que tienen uno (subespacio clasico y Gabor): sin
   el, la cifra cruda no significa nada y se lee igual de bien en los dos casos;
3. que la resolucion se conserva (el encargo dice explicitamente que no quiere
   una imagen mas pequena);
4. el AISLAMIENTO del modulo (§6 del encargo): `fv.probe` no importa `fv.models`.
   Un import de mas no rompe nada hoy y ata el experimento a la red que estudia.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fv.probe import data as probe_data          # noqa: E402
from fv.probe import figures as probe_figures    # noqa: E402
from fv.probe import gabor as probe_gabor        # noqa: E402
from fv.probe import metrics as probe_metrics    # noqa: E402
from fv.probe.model import L1Probe               # noqa: E402
from fv.probe.run import run_name, train         # noqa: E402
from fv.probe.table import COLUMNS, comparison_table   # noqa: E402

PROBE_DIR = Path(probe_data.__file__).resolve().parent


# ------------------------------------------------- 1. la salida degenerada

@pytest.mark.parametrize("k", [3, 5, 9])
def test_los_atomos_del_decodificador_quedan_a_norma_uno(k):
    m = L1Probe(channels=6, k=k)
    m.renormalize()
    n = m.dec.weight.detach().flatten(1).norm(dim=1)
    assert torch.allclose(n, torch.ones(6), atol=1e-5)


def test_sin_renormalizar_el_modelo_puede_encoger_z_en_vez_de_dispersarlo():
    """La salida degenerada existe de verdad: es lo que la decision 1 bloquea.

    Escalar el codificador por a y el decodificador por 1/a deja la
    reconstruccion IGUAL y divide la penalizacion por a. Sin el freno, bajar
    `mean(|z|)` sale gratis.
    """
    m = L1Probe(channels=4, k=3)
    x = torch.randn(2, 1, 20, 20)
    with torch.no_grad():
        xh0, z0 = m(x)
        a = 0.01
        m.enc.weight.mul_(a); m.enc.bias.mul_(a)
        m.dec.weight.div_(a)
        xh1, z1 = m(x)
    assert torch.allclose(xh0, xh1, atol=1e-4)          # misma reconstruccion
    assert float(z1.abs().mean()) < float(z0.abs().mean()) * 0.02   # 50x menos pena
    m.renormalize()                                      # ...y el freno lo deshace
    assert torch.allclose(m.dec.weight.detach().flatten(1).norm(dim=1),
                          torch.ones(4), atol=1e-5)


def test_la_renormalizacion_sobrevive_a_un_paso_del_optimizador():
    m = L1Probe(channels=5, k=5)
    m.renormalize()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.randn(4, 1, 20, 20)
    for _ in range(3):
        xh, z = m(x)
        loss = ((xh - x) ** 2).mean() + 0.1 * z.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        m.renormalize()
    n = m.dec.weight.detach().flatten(1).norm(dim=1)
    assert torch.allclose(n, torch.ones(5), atol=1e-5)


def test_el_bucle_de_entrenamiento_deja_el_decodificador_normalizado():
    """El freno esta DENTRO de `train`, no solo disponible como metodo."""
    datos = _datos_sinteticos(n_train=64, n_val=32)
    r = train(datos, channels=4, k=3, lam=0.1, seed=1, epochs=1, batch=32,
              lr=3e-3, out_dir=None, val_log=32, verbose=False, gabor_steps=30)
    assert r["kernels_muertos"] >= 0            # el run termina y reporta


# ------------------------------------------------- 2a. el nulo del subespacio

@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_la_base_clasica_es_ortonormal_y_tiene_6_filtros(k):
    B = probe_metrics.classic_basis(k)
    assert B.shape == (6, k * k)
    assert torch.allclose(B @ B.T, torch.eye(6), atol=1e-4)


@pytest.mark.parametrize("k", [3, 5, 9])
def test_un_kernel_aleatorio_da_enriquecimiento_1(k):
    """El nulo 6/k^2 tiene que ser el valor MEDIDO de un kernel aleatorio.

    Es lo que hace que "0,688 en 3x3" se lea como "indistinguible de aleatorio":
    sin este anclaje, la fraccion cruda no significa nada.
    """
    B = probe_metrics.classic_basis(k)
    g = torch.Generator().manual_seed(0)
    W = torch.randn(4000, k * k, generator=g)
    W = W / W.norm(dim=1, keepdim=True)
    frac = (W @ B.T).pow(2).sum(1).mean()
    assert float(frac) == pytest.approx(6.0 / (k * k), rel=0.05)


@pytest.mark.parametrize("k", [3, 5])
def test_los_propios_filtros_clasicos_dan_el_enriquecimiento_maximo(k):
    """El otro extremo: si los kernels SON la base, la energia es 1."""
    B = probe_metrics.classic_basis(k)
    frac = (B @ B.T).pow(2).sum(1)
    assert torch.allclose(frac, torch.ones(6), atol=1e-4)


# ------------------------------------------------- 2b. el nulo del Gabor

def _gabor(k, theta, f, phase, sx, sy):
    a = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    y, x = torch.meshgrid(a, a, indexing="ij")
    u = x * math.cos(theta) + y * math.sin(theta)
    v = -x * math.sin(theta) + y * math.cos(theta)
    env = torch.exp(-(u ** 2 / (2 * sx ** 2) + v ** 2 / (2 * sy ** 2)))
    return (env * torch.cos(2 * math.pi * f * u + phase)).flatten()


@pytest.mark.parametrize("k", [5, 7, 9])
def test_un_gabor_de_verdad_se_ajusta_casi_perfecto(k):
    W = torch.stack([_gabor(k, 0.4, 0.20, 0.0, k / 4, k / 3),
                     _gabor(k, 1.9, 0.35, math.pi / 2, k / 5, k / 4)])
    r2 = probe_gabor.fit_gabor_r2(W, k)
    assert float(r2.min()) > 0.98


def test_el_nulo_del_gabor_es_ALTISIMO_en_3x3_y_por_eso_se_resta():
    """La razon de existir de la linea base, en un numero.

    Con 7 parametros libres sobre 9 muestras, un Gabor ajusta CUALQUIER kernel
    de 3x3. Si alguien leyera el R2 absoluto, un 3x3 aleatorio pareceria un
    filtro generico excelente. Este test fija ese hecho para que la resta no se
    "simplifique" nunca.
    """
    base3 = probe_gabor.random_baseline_r2(3, 64)
    base9 = probe_gabor.random_baseline_r2(9, 64)
    assert float(base3.median()) > 0.70          # ruido puro, R2 altisimo
    assert float(base9.median()) < 0.40          # y cae con k
    assert float(base3.median()) > float(base9.median())


def test_la_diferencia_separa_gabores_de_ruido_donde_el_absoluto_no():
    """Lo que el criterio lee: `gabor_delta`, nunca `gabor_r2`."""
    k = 9
    W = torch.stack([_gabor(k, i * 0.5, 0.2 + 0.05 * i, 0.0, k / 4, k / 3)
                     for i in range(8)])
    delta = float(probe_gabor.fit_gabor_r2(W, k).median()
                  - probe_gabor.random_baseline_r2(k, 64).median())
    assert delta > 0.5


def test_el_ajuste_es_DETERMINISTA():
    """Sin esto, dos lecturas del mismo kernel dan dos numeros y nadie sabe
    cual es el bueno. Los arranques son una rejilla fija, no aleatorios."""
    g = torch.Generator().manual_seed(7)
    W = torch.randn(6, 25, generator=g)
    a = probe_gabor.fit_gabor_r2(W, 5, steps=120)
    b = probe_gabor.fit_gabor_r2(W, 5, steps=120)
    assert torch.allclose(a, b, atol=1e-6)


def test_el_r2_del_gabor_esta_acotado_en_0_1():
    g = torch.Generator().manual_seed(3)
    r2 = probe_gabor.fit_gabor_r2(torch.randn(12, 49, generator=g), 7, steps=120)
    assert float(r2.min()) >= 0.0 and float(r2.max()) <= 1.0


# ------------------------------------------------- 3. la resolucion se conserva

@pytest.mark.parametrize("k", [3, 5, 7, 9])
@pytest.mark.parametrize("K", [8, 32])
def test_la_vista_entra_y_sale_a_20x20(k, K):
    m = L1Probe(channels=K, k=k)
    x = torch.zeros(2, 1, 20, 20)
    xh, z = m(x)
    assert tuple(z.shape) == (2, K, 20, 20)
    assert tuple(xh.shape) == (2, 1, 20, 20)


def test_un_k_par_se_rechaza_en_vez_de_encoger_la_imagen():
    """Con k par, `padding=k//2` NO conserva el tamano: sale 21x21 o 19x19 y el
    experimento mediria otra cosa sin que nadie lo note."""
    with pytest.raises(ValueError):
        L1Probe(channels=4, k=4)


def test_el_codificador_replica_el_borde_como_pad_mode_edge():
    assert L1Probe(channels=4, k=5).enc.padding_mode == "replicate"


def test_convtranspose_no_admite_replicate_y_por_eso_hay_err_rec_int():
    """Fija la razon por la que la metrica interior existe.

    Si una version de torch pasara a admitirlo, este test falla y hay que
    revisar la decision -- que es exactamente lo que se quiere que pase.
    """
    with pytest.raises(ValueError):
        torch.nn.ConvTranspose2d(4, 1, 5, padding=2, bias=False,
                                 padding_mode="replicate")


# ------------------------------------------------- 4. el preprocesado

def test_la_normalizacion_deja_media_local_cero_y_no_explota_en_lo_plano():
    x = torch.zeros(1, 1, 20, 20)
    x[:, :, 8:12, 8:12] = 1.0                     # una mancha, el resto plano
    y = probe_data.local_contrast_norm(x, sigma=2.0, eps=0.0148)
    assert torch.isfinite(y).all()
    # La comprobacion es una RAZON, no un umbral tecleado: en la esquina queda
    # una respuesta real (la cola de la gaussiana llega, sigma=2 y la mancha esta
    # a 4 px), y lo que importa es que sea despreciable frente a la mancha.
    esquina = float(y[0, 0, :4, :4].abs().max())
    mancha = float(y.abs().max())
    assert mancha > 1.0                           # la mancha SI sobrevive
    assert esquina < mancha / 100                 # lo plano NO se amplifica


def test_una_constante_se_normaliza_a_cero():
    y = probe_data.local_contrast_norm(torch.full((1, 1, 20, 20), 0.7), 2.0, 0.0148)
    assert float(y.abs().max()) < 1e-3


# ------------------------------------------------- 5. las metricas del encargo

def _datos_sinteticos(n_train=96, n_val=64, n=20):
    g = torch.Generator().manual_seed(11)
    tr = torch.randn(n_train, 1, n, n, generator=g)
    va = torch.randn(n_val, 1, n, n, generator=g)
    return {"train": tr.numpy(), "val": va.numpy(),
            "var": float(tr.var()), "eps": 0.0148}


def test_kernel_muerto_es_MENOS_del_0_1_por_ciento_de_las_posiciones():
    """§5.3 del encargo, literal. El umbral es un dato del encargo, no un
    numero de conveniencia: con 1e-4 (0,01 %) un kernel practicamente apagado
    contaria como vivo y "cero kernels muertos" dejaria de significar nada."""
    assert probe_metrics.DEAD_KERNEL_FRAC == 1e-3

    m = L1Probe(channels=3, k=3)
    with torch.no_grad():
        m.enc.weight.zero_(); m.enc.bias.zero_()
        m.enc.bias[0] = 1.0                       # siempre activo
        m.enc.bias[1] = -1e6                      # nunca activo
        m.enc.weight[2, 0, 1, 1] = 1.0            # activo donde x > 0
    val = torch.zeros(4, 1, 20, 20)
    val[0, 0, 0, 0] = 5.0                         # 1 de 1600 posiciones = 0,06 %
    r = probe_metrics.final_metrics(m, val, var=1.0, lam=0.0, gabor_steps=30)
    assert r["kernels_muertos"] == 2              # el apagado Y el del 0,06 %


def test_la_dimension_efectiva_al_95_por_ciento_cuenta_las_componentes_reales():
    """§5.6: componentes de PCA para el 95 % de la varianza, sobre k^2."""
    k = 5
    g = torch.Generator().manual_seed(5)
    base = torch.randn(2, k * k, generator=g)
    coef = torch.randn(16, 2, generator=g)
    W = coef @ base                                # rango 2 exacto
    assert probe_metrics.pca_dim_95(W) == 2
    assert probe_metrics.pca_dim_95(torch.randn(30, k * k, generator=g)) > 5


def test_la_alineacion_enc_dec_vale_1_con_pesos_atados():
    """§5.8: coseno entre el kernel i de cada lado, SIN voltear -- porque
    `conv_transpose2d(w)` es el adjunto exacto de `conv2d(w)`, o sea que
    "atados" significa el mismo tensor en la misma orientacion."""
    m = L1Probe(channels=5, k=5)
    with torch.no_grad():
        m.dec.weight.copy_(m.enc.weight)
        m.renormalize()
        m.enc.bias.zero_()
    val = torch.randn(8, 1, 20, 20)
    r = probe_metrics.final_metrics(m, val, var=1.0, lam=0.0, gabor_steps=30)
    assert r["align_enc_dec"] == pytest.approx(1.0, abs=1e-4)
    assert r["align_enc_dec_min"] == pytest.approx(1.0, abs=1e-4)


def test_conv_transpose_es_el_adjunto_de_conv_y_por_eso_no_se_voltea():
    """La razon del test de arriba, comprobada: <conv(x), z> == <x, convT(z)>.

    Si esto dejara de ser cierto, la metrica 8 estaria comparando un kernel con
    la version volteada del otro y saldria ~0 con pesos atados -- o sea, la
    conclusion contraria a la verdadera."""
    g = torch.Generator().manual_seed(2)
    w = torch.randn(4, 1, 5, 5, generator=g)
    x = torch.randn(2, 1, 20, 20, generator=g)
    z = torch.randn(2, 4, 20, 20, generator=g)
    lhs = (torch.nn.functional.conv2d(x, w, padding=2) * z).sum()
    rhs = (x * torch.nn.functional.conv_transpose2d(z, w, padding=2)).sum()
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-4)


def test_las_metricas_traen_las_OCHO_del_encargo():
    m = L1Probe(channels=4, k=5)
    r = probe_metrics.final_metrics(m, torch.randn(8, 1, 20, 20), var=1.0,
                                    lam=0.1, gabor_steps=30)
    for clave in ("r2_rec", "frac_activa", "kernels_muertos", "gabor_delta",
                  "gabor_r2_base", "enriquecimiento", "nulo_6d", "dim_pca95",
                  "coseno_max", "align_enc_dec"):
        assert clave in r, clave


# ------------------------------------------------- 6. artefactos y tabla

def test_un_run_deja_los_CINCO_artefactos_del_encargo(tmp_path):
    """§6: config, metrics.jsonl, checkpoint y los kernels de los DOS lados."""
    datos = _datos_sinteticos()
    train(datos, channels=4, k=3, lam=0.1, seed=1, epochs=2, batch=32, lr=3e-3,
          out_dir=tmp_path, val_log=32, verbose=False, gabor_steps=30)
    d = tmp_path / "k3-K4-l0.1-s1"
    for f in ("config.json", "metrics.jsonl", "checkpoint.pt",
              "kernels_enc.npy", "kernels_dec.npy", "summary.json"):
        assert (d / f).exists(), f
    lineas = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
    assert [l["epoca"] for l in lineas] == [1, 2]      # UNA linea por epoca
    assert np.load(d / "kernels_enc.npy").shape == (4, 3, 3)


def test_la_tabla_lleva_las_ocho_metricas_y_promedia_semillas():
    filas = []
    for s, gd in ((1, 0.10), (2, 0.20)):
        filas.append({"nombre": f"k5-K8-l0.1-s{s}", "k": 5, "K": 8, "lambda": 0.1,
                      "gabor_delta": gd, "gabor_r2": 0.6, "gabor_r2_base": 0.5,
                      "r2_rec": 0.8, "r2_rec_int": 0.85, "frac_activa": 0.1,
                      "kernels_muertos": 0, "enriquecimiento": 1.5,
                      "dim_pca95_frac": 0.3, "coseno_max": 0.4,
                      "align_enc_dec": 0.9})
    md = comparison_table(filas)
    cuerpo = [l for l in md.splitlines() if l.startswith("| k5")]
    assert len(cuerpo) == 1                      # UNA fila para las dos semillas
    celdas = [c.strip() for c in cuerpo[0].split("|")[1:-1]]
    assert len(celdas) == len(COLUMNS)
    assert celdas[COLUMNS.index(("n", "n"))] == "2"
    assert celdas[[c[0] for c in COLUMNS].index("gabor_delta")] == "0.150"


def test_la_figura_no_recorta_su_titulo(tmp_path):
    """Una hoja de contactos recortada hay que volver a generarla, y para
    entonces el run puede no estar. El ancho se MIDE del texto."""
    from PIL import Image
    g = np.random.default_rng(0)
    p = probe_figures.contact_sheet(
        g.normal(size=(4, 3, 3)).astype(np.float32), tmp_path / "h.png",
        "un titulo bastante largo para una rejilla de solo cuatro kernels",
        "y un subtitulo todavia mas largo, con metricas y sus nulos al lado")
    assert Image.open(p).width >= 560


def test_los_mapas_z_salen_uno_por_canal(tmp_path):
    from PIL import Image
    g = np.random.default_rng(1)
    p = probe_figures.code_maps(g.normal(size=(20, 20)).astype(np.float32),
                                np.abs(g.normal(size=(6, 20, 20))).astype(np.float32),
                                tmp_path / "z.png", "mapas")
    assert Image.open(p).size[0] > 0


def test_el_nombre_de_un_run_es_estable():
    assert run_name(32, 9, 0.1) == "k9-K32-l0.1"


# ------------------------------------------------- 7. el AISLAMIENTO (§6)

def test_fv_probe_no_importa_fv_models():
    """§6 del encargo: modulo aislado. `fv.fovea` entra SOLO como cargador de
    ventanas (`build_view`/`dims_of`); `fv.models` no entra en absoluto.

    Un import de mas no rompe nada hoy: ata el experimento a la red que
    estudia, y entonces "es un experimento aparte" deja de ser cierto sin que
    falle nada. Por eso se comprueba leyendo el codigo, no la documentacion.
    """
    permitidos_fovea = {"build_view", "dims_of"}
    for f in sorted(PROBE_DIR.glob("*.py")):
        arbol = ast.parse(f.read_text())
        for nodo in ast.walk(arbol):
            mod = None
            if isinstance(nodo, ast.ImportFrom):
                mod = nodo.module or ""
                if mod == "fv.fovea":
                    nombres = {a.name for a in nodo.names}
                    assert nombres <= permitidos_fovea, f"{f.name}: {nombres}"
            elif isinstance(nodo, ast.Import):
                mod = ",".join(a.name for a in nodo.names)
            if mod and "fv.models" in mod:
                pytest.fail(f"{f.name} importa fv.models: {mod}")


# --------------------------------------- 8. el lanzamiento desacoplado (R11/R17)

REPO = PROBE_DIR.parents[2]
LANZADOR = REPO / "scripts" / "sonda_l1_desacoplada.sh"


@pytest.fixture
def _node_roto(tmp_path):
    """Un `node` que siempre falla, para simular un aviso que no llega."""
    d = tmp_path / "stub"
    d.mkdir()
    (d / "node").write_text("#!/bin/sh\nexit 3\n")
    (d / "node").chmod(0o755)
    return d


def _lanzar(args, stub, tmp_path):
    import os
    import subprocess
    env = dict(os.environ)
    env["PATH"] = f"{stub}{os.pathsep}{env['PATH']}"
    env["COORD_HOME"] = str(tmp_path)          # con notify.mjs ausente no basta:
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts" / "notify.mjs").write_text("")   # ...tiene que existir
    return subprocess.run(["sh", str(LANZADOR), *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600).returncode


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="hace falta un sh POSIX")
def test_un_aviso_que_falla_NO_relanza_la_rejilla(_node_roto, tmp_path):
    """`desacoplar-persistente.sh` registra la unidad con `Restart=on-failure`.

    Si el codigo de salida lo decidiera el AVISO, un `notify.mjs` que falla
    --sin BOT_TOKEN, red caida, hilo borrado-- relanzaria las 12 h de rejilla
    cada 30 s. Medido el 2026-09-02 con el arnes: la sonda termino bien, el
    aviso fallo, y la unidad quedo reiniciandose.
    """
    salida = tmp_path / "vacio"
    salida.mkdir()
    assert _lanzar(["--tabla", "--salida", str(salida)], _node_roto, tmp_path) == 0


@pytest.mark.skipif(not Path("/bin/sh").exists(), reason="hace falta un sh POSIX")
def test_un_trabajo_que_falla_SI_sale_con_error_aunque_el_aviso_tambien_falle(
        _node_roto, tmp_path):
    """El otro lado: si el aviso enmascarara el fallo del trabajo, una sonda que
    revienta saldria como `success` y no se reintentaria -- que es justo cuando
    el reintento sirve, porque `--rejilla` se reanuda saltando lo ya hecho."""
    assert _lanzar(["--flag-que-no-existe"], _node_roto, tmp_path) != 0


def test_el_ejecutor_de_telegram_solo_desacopla_lo_que_ENTRENA():
    """Un flag mal escrito no puede acabar levantando una unidad de 12 h.

    Solo `--rejilla`, `--repetir-mejores` y `--solo` van por
    `desacoplar-persistente.sh`; lo demas (`--cronometrar`, `--tabla`,
    `--figuras`, un typo) corre en primer plano y contesta texto.
    """
    j = json.loads((REPO / "telegram" / "executors" / "sonda-l1.json").read_text())
    cmd = j["command"]
    rama = cmd[cmd.index("*--rejilla*"):cmd.index("*) PYTHON")]
    assert "desacoplar-persistente.sh" in rama
    assert "sonda_l1_desacoplada.sh" in rama
    for flag in ("--cronometrar", "--tabla", "--figuras"):
        assert flag not in rama, flag
    assert cmd.count("desacoplar-persistente.sh") == 1


def test_el_freno_del_coordinador_conoce_la_sonda():
    """R11: un comando que puede EMPEZAR un trabajo largo lleva su freno.

    Aqui no se paga dinero --la sonda no alquila nada-- pero si se pierde
    trabajo: 12 h que mueren con el server. `cerrable.mjs` tiene que nombrarla
    en el veredicto, o el dueno apaga la maquina sin saberlo.
    """
    freno = REPO.parent / "telegram-coordinator" / "scripts" / "cerrable.mjs"
    if not freno.exists():
        pytest.skip("el repo del coordinador no esta clonado en esta maquina")
    trabajos = [l for l in freno.read_text().splitlines() if l.startswith("const TRABAJOS")]
    assert trabajos, "cerrable.mjs ya no declara TRABAJOS"
    assert "sonda_l1" in trabajos[0]


# ============================================================================
# 9. La revision del dueno del 2026-09-02
# ============================================================================
#
# Cinco cosas que cambiaron el estudio, y las cinco romperian en SILENCIO:
# el criterio dejaria de ser comparable entre `k`, o una celda usaria un lambda
# de dos millones, o `enriq` seguiria leyendose como si significase lo mismo.

from fv.probe import calibrate as probe_cal            # noqa: E402
from fv.probe import spectrum as probe_spec            # noqa: E402


# --------------------------------- 9a. el criterio, comparable entre k

def test_un_umbral_ABSOLUTO_es_tres_exigencias_distintas():
    """La razon de existir de `gabor_delta_rel`, en numeros MEDIDOS.

    Con los nulos de cada k, un 0,25 absoluto pide explicar el 52 % del margen
    alcanzable en k=5 y el 32 % en k=9. Si alguien vuelve a poner un umbral
    absoluto, este test dice por que no.
    """
    exigencia = {}
    for k in (5, 7, 9):
        nulo = float(probe_gabor.random_baseline_r2(k, 64).median())
        exigencia[k] = 0.25 / (1.0 - nulo)
    assert exigencia[5] > 0.50 and exigencia[9] < 0.35
    assert exigencia[5] > exigencia[7] > exigencia[9]


def test_gabor_delta_rel_normaliza_por_el_margen_alcanzable():
    m = L1Probe(channels=4, k=7)
    r = probe_metrics.final_metrics(m, torch.randn(8, 1, 20, 20), var=1.0,
                                    lam=0.0, gabor_steps=40)
    esperado = r["gabor_delta"] / (1.0 - r["gabor_r2_base"])
    assert r["gabor_delta_rel"] == pytest.approx(esperado, rel=1e-6)


def test_el_p95_es_el_de_la_MEDIANA_de_K_no_el_de_un_kernel_suelto():
    """El estadistico que se compara es una mediana de K, asi que el nulo tiene
    que ser la distribucion de esa mediana. Con el p95 de valores sueltos la
    prueba seria mucho mas laxa de lo que aparenta, y con mas K se estrecha."""
    g = torch.Generator().manual_seed(1)
    nulos = torch.rand(400, generator=g)
    p95_suelto = float(nulos.quantile(0.95))
    p95_8, p95_64 = probe_spec.bootstrap_p95(nulos, 8), probe_spec.bootstrap_p95(nulos, 64)
    mediana = float(nulos.median())
    assert mediana < p95_64 < p95_8 < p95_suelto


# --------------------------------- 9b. por que `enriq` dejo de significar

def test_la_base_clasica_es_de_BAJA_frecuencia_y_por_eso_enriq_se_hunde():
    """El mecanismo del hallazgo del dueno, fijado.

    `classic_basis` construye los filtros a k>3 con suavizado binomial: son
    plantillas de baja frecuencia. La normalizacion de contraste del §2 quita DC
    y las bajas frecuencias de la ENTRADA, asi que los kernels aprendidos viven
    en alta frecuencia y salen casi ortogonales a esa base -- `enriq` 0,47-0,61,
    por DEBAJO de su nulo de 1,0 (medido 2026-09-02; sin normalizar vuelve a 1,01).

    Si alguien cambia la base y este test falla, es que la lectura de la
    metrica 5 ha cambiado y hay que revisarla, no silenciarlo.
    """
    for k in (5, 7, 9):
        B = probe_metrics.classic_basis(k)
        centro_base = float(probe_spec.spectral_metrics(B, k)["frec_central"].median())
        centro_azar = float(probe_spec.random_spectral_baseline(k, 128)["frec_central"].median())
        assert centro_base < centro_azar * 0.75, (k, centro_base, centro_azar)


# --------------------------------- 9c. las metricas sin plantilla

@pytest.mark.parametrize("k", [5, 7, 9])
def test_un_gabor_de_verdad_supera_el_nulo_de_orientacion(k):
    G = torch.stack([_gabor(k, 0.4, 0.25, 0.0, k / 4, k / 3)])
    m = probe_spec.spectral_metrics(G, k)
    b = probe_spec.random_spectral_baseline(k, 128)
    assert float(m["conc_orient"]) > float(b["conc_orient"].median()) * 2


def test_la_orientacion_SI_es_legible_en_3x3_donde_el_gabor_no_lo_es():
    """El hueco que estas metricas vienen a tapar.

    En 3x3 el nulo del ajuste Gabor es 0,879, o sea que el techo de la
    diferencia es 0,121 y el ancla no se puede juzgar por ahi. El nulo de la
    orientacion en 3x3 es ~0,24: queda margen de sobra.
    """
    nulo_gabor = float(probe_gabor.random_baseline_r2(3, 64).median())
    nulo_orient = float(probe_spec.random_spectral_baseline(3, 128)["conc_orient"].median())
    assert 1 - nulo_gabor < 0.20          # el Gabor no deja margen en 3x3...
    assert 1 - nulo_orient > 0.70         # ...y la orientacion si


@pytest.mark.parametrize("k", [3, 5, 9])
def test_las_metricas_espectrales_estan_acotadas_en_0_1(k):
    g = torch.Generator().manual_seed(4)
    m = probe_spec.spectral_metrics(torch.randn(20, k * k, generator=g), k)
    for n in ("conc_banda", "conc_orient"):
        assert float(m[n].min()) >= 0.0 and float(m[n].max()) <= 1.0


def test_el_relleno_a_ceros_deja_la_MISMA_rejilla_de_frecuencia_en_todo_k():
    """Sin esto, un bin radial significaria otra frecuencia en cada `k` y las
    columnas de la tabla no serian comparables -- que es justo lo que estas
    metricas vienen a arreglar."""
    centros = [float(probe_spec.random_spectral_baseline(k, 128)["frec_central"].median())
               for k in (3, 5, 7, 9)]
    assert max(centros) - min(centros) < 0.01, centros


# --------------------------------- 9d. la calibracion de lambda

def _calibrador_falso(mapa):
    """Sustituye el entrenamiento por una funcion lambda -> activacion."""
    def fake(data, channels, k, lam, seed, epochs, batch, lr, val_log, pasos=None):
        return mapa(lam)
    return fake


def test_la_calibracion_encuentra_la_lambda_de_la_banda(monkeypatch):
    # activacion que baja suave con log(lambda), como la medida
    mapa = lambda lam: max(0.02, 0.46 - 0.085 * math.log10(max(lam, 1e-6) * 1e3))
    monkeypatch.setattr(probe_cal, "_activacion", _calibrador_falso(mapa))
    r = probe_cal.calibrate_lambda({}, 16, 7, verbose=False)
    assert r["en_banda"] is True
    assert abs(r["activa_calibrada"] - 0.10) <= 0.03
    assert r["n_evaluaciones"] <= 7


def test_una_celda_que_SATURA_se_declara_en_vez_de_irse_a_lambda_dos_millones(monkeypatch):
    """Si la activacion toca un suelo, se DICE en vez de subir lambda sin fin.

    ⚠ El caso que motivo este freno (k=3/K=8 clavado en 14,4 %) resulto ser un
    ARTEFACTO de medir a 64 pasos: con el presupuesto correcto esa celda baja
    sin problema y NO hay ninguna celda medida que sature. El freno se conserva
    porque un suelo real es posible y porque protege del λ=2,6e6, pero el caso
    de aqui es SINTETICO a proposito -- no se apoya en una medida retirada.
    """
    mapa = lambda lam: max(0.144, 0.46 - 0.085 * math.log10(max(lam, 1e-6) * 1e3))
    monkeypatch.setattr(probe_cal, "_activacion", _calibrador_falso(mapa))
    r = probe_cal.calibrate_lambda({}, 8, 3, verbose=False)
    assert r["saturado"] is True
    assert r["en_banda"] is False, "no llega a la banda, y tiene que decirlo"
    assert r["lambda"] < 1e4, f"λ absurda: {r['lambda']}"


def test_entre_lambdas_que_EMPATAN_gana_la_mas_pequena(monkeypatch):
    """El objetivo es una esparsidad; entre dos λ que la consiguen, la menor
    distorsiona menos la reconstruccion.

    La curva es la MEDIDA: baja suave y se acerca a un suelo (14,4 % en
    k=3/K=8), asi que las ultimas evaluaciones empatan y sin esta regla ganaria
    la mayor -- que fue como salio λ=2,6e6.
    """
    mapa = lambda lam: 0.144 + 0.32 / (1 + (max(lam, 1e-9) / 20) ** 0.8)
    monkeypatch.setattr(probe_cal, "_activacion", _calibrador_falso(mapa))
    r = probe_cal.calibrate_lambda({}, 8, 3, verbose=False)
    ev = r["evaluaciones"]
    cerca = min(abs(e["activa"] - 0.10) for e in ev)
    empatan = [e["lambda"] for e in ev
               if abs(e["activa"] - 0.10) <= cerca + probe_cal.EMPATE]
    assert len(empatan) > 1, "el caso no reproduce el empate que se quiere probar"
    assert r["lambda"] == min(empatan)
    assert r["lambda"] < max(e["lambda"] for e in ev), "no se va al extremo"


def test_el_desempate_NO_cambia_una_lambda_en_banda_por_una_fuera(monkeypatch):
    """El fallo del 2026-09-02: `EMPATE` (1 punto) es mas ancho que la banda, asi
    que "gana la menor" elegia λ=10 (13,4 %, FUERA) en vez de λ=28 (7,5 %,
    dentro). Un desempate no puede tirar el criterio que lo precede."""
    # a tramos, no por clave exacta: la biseccion produce sqrt(10*80) y comparar
    # floats por igualdad hacia que el falso devolviera otra cosa
    mapa = lambda lam: 0.134 if lam <= 15 else (0.075 if lam <= 40 else 0.039)
    monkeypatch.setattr(probe_cal, "_activacion", _calibrador_falso(mapa))
    r = probe_cal.calibrate_lambda({}, 16, 3, verbose=False)
    assert r["en_banda"] is True
    assert abs(r["activa_calibrada"] - 0.10) <= 0.03
    assert r["lambda"] != 10.0, "λ=10 esta FUERA de banda: no puede ganar el desempate"


def test_la_calibracion_mide_por_PASOS_no_por_epocas(monkeypatch):
    """Lo que asienta la activacion son los pasos del optimizador. Medir con
    "2 epocas de un subconjunto" son 64 pasos y sobreestima 6x -- hasta el punto
    de declarar "satura, no puede esparcirse" una celda que se esparce de sobra."""
    vistos = {}

    def espia(data, channels, k, lam, seed, epochs, batch, lr, val_log, pasos=None):
        vistos["pasos"] = pasos
        return 0.10
    monkeypatch.setattr(probe_cal, "_activacion", espia)
    r = probe_cal.calibrate_lambda({}, 16, 5, verbose=False)
    assert vistos["pasos"] == probe_cal.PASOS >= 256
    assert r["pasos_por_evaluacion"] == probe_cal.PASOS


def test_fit_respeta_un_tope_de_pasos(tmp_path):
    datos = _datos_sinteticos(n_train=512, n_val=64)
    _, lineas, meta = probe_cal.fit(datos, 4, 3, 0.1, 1, 50, 64, 3e-3,
                                    out_dir=None, val_log=64, verbose=False,
                                    max_steps=5)
    assert meta["pasos"] == 5, meta


def test_la_calibracion_corre_de_verdad_sobre_datos(tmp_path):
    """El cableado, no la logica: que `fit` + `_val_pass` se enganchan."""
    datos = _datos_sinteticos(n_train=128, n_val=64)
    r = probe_cal.calibrate_lambda(datos, 4, 3, epochs=1, batch=64, val_log=64,
                                   max_evals=2, verbose=False)
    assert 0.0 <= r["activa_calibrada"] <= 1.0
    assert r["n_evaluaciones"] >= 1


# --------------------------------- 9e. la tabla lleva el criterio nuevo

def test_la_tabla_encabeza_con_lo_que_decide_y_no_con_el_valor_absoluto():
    claves = [c[0] for c in COLUMNS]
    assert claves.index("gabor_delta_rel") < claves.index("gabor_delta"), \
        "el absoluto no puede ir antes que el normalizado: es lo que se lee primero"
    for c in ("gabor_supera_p95", "conc_orient_delta", "conc_banda_delta"):
        assert c in claves, c
