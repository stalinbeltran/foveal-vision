"""The comparison table of the whole grid, with the eight metrics (section 6).

Two formats on purpose: markdown for the report (which is what gets read) and
CSV for whoever wants to sort it. Rows are grouped by (k, K, lambda) and
averaged over seeds, with n printed -- a mean over 3 seeds and a single run must
not look alike in the same column.

⚠ Column 5 carries its own null in the SAME cell (`x/nulo`). The raw fraction is
not comparable across `k`, and a table is exactly where someone reads down a
column without noticing the scale changed.
"""

from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = [
    ("run", "run"),
    ("k", "k"),
    ("K", "K"),
    ("lambda", "lam"),
    ("n", "n"),
    # -- el criterio, en el orden en que se lee (revision del dueno 2026-09-02)
    ("gabor_delta_rel", "[4] GaborD/margen"),
    ("gabor_supera_p95", "  >p95"),
    ("conc_orient_delta", "[4c] orient D"),
    ("conc_orient_supera_p95", "  >p95"),
    ("conc_banda_delta", "[4c] banda D"),
    ("conc_banda_supera_p95", "  >p95"),
    # -- la cifra cruda, que NO es comparable entre k por si sola
    ("gabor_delta", "  Gabor D"),
    ("gabor_r2_base", "  nulo"),
    ("r2_rec_int", "[1] R2 rec int"),
    ("frac_activa", "[2] activa % med"),
    ("frac_activa_vivos", "  de los vivos"),
    ("kernels_vivos", "[3] vivos"),
    ("kernels_muertos", "  muertos"),
    ("kernels_saturados", "  saturados"),
    ("enriquecimiento", "[5] enriq x"),
    ("dim_pca95_frac", "[6] dim95/k2"),
    ("coseno_max", "[7] cos max"),
    ("align_enc_dec", "[8] align"),
]


def _group(rows: list[dict]) -> list[dict]:
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["k"], r["K"], r["lambda"]), []).append(r)
    out = []
    for (k, K, lam), rs in by.items():
        agg = {"run": f"k{k}-K{K}-l{lam}", "k": k, "K": K, "lambda": lam, "n": len(rs)}
        for key, _ in COLUMNS[5:]:
            vals = [float(r[key]) for r in rs if key in r]
            agg[key] = sum(vals) / len(vals) if vals else float("nan")
            if len(rs) > 1 and vals:
                mean = agg[key]
                agg[key + "_sd"] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        out.append(agg)
    out.sort(key=lambda r: (r["k"], r["K"], r["lambda"]))
    return out


def _fmt(key: str, v) -> str:
    if isinstance(v, str):
        return v
    if key.endswith("_supera_p95"):
        return "SI" if v >= 0.999 else ("si" if v > 0 else "no")   # >0 = alguna semilla
    if key in ("k", "K", "n"):
        return f"{v:.0f}"
    if key.startswith("kernels_"):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if key.startswith("frac_activa"):
        return f"{v*100:.1f}"
    if key in ("lambda", "enriquecimiento"):
        return f"{v:.2f}"
    return f"{v:.3f}"


def comparison_table(rows: list[dict], out_md: Path | None = None,
                     out_csv: Path | None = None, note: str = "") -> str:
    """Returns the markdown; writes both files if given a path."""
    grouped = _group(rows)
    heads = [h for _, h in COLUMNS]
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "|".join(["---"] * len(heads)) + "|"]
    for g in grouped:
        lines.append("| " + " | ".join(_fmt(k, g.get(k, float("nan")))
                                       for k, _ in COLUMNS) + " |")
    md = "\n".join(lines)
    legend = (
        "\n\n**El criterio se lee en las seis primeras columnas de metrica**, y "
        "ninguna es un valor absoluto:\n"
        "· `GaborD/margen` = (R2 del ajuste - su nulo) / (1 - su nulo). Dividir "
        "por el margen ALCANZABLE es lo que lo hace comparable entre `k`: con los "
        "nulos medidos, un 0,25 absoluto es el 52 % del margen en k=5, el 38 % en "
        "k=7 y el 32 % en k=9, o sea tres exigencias distintas.\n"
        "· `>p95` = la mediana del run supera el p95 de la mediana de K kernels "
        "ALEATORIOS (bootstrap). Es la prueba, sin unidades; la magnitud la da la "
        "columna de al lado.\n"
        "· `orient D` y `banda D` no dependen de ninguna plantilla, y por eso "
        "sobreviven a que la entrada este normalizada en contraste -- que es lo "
        "que rompe a `enriq` (ver `fv/probe/spectrum.py`).\n"
        "⚠ `Gabor D` en crudo se conserva porque es lo que nombra el encargo, "
        "pero NO es comparable entre `k` por si solo.\n"
        "⚠ `enriq x` vale 1 cuando el kernel es indistinguible de uno aleatorio, "
        "y esta MEDIDO que cae a 0,47-0,61 en toda la sonda por la normalizacion "
        "de contraste: leelo como diagnostico, no como criterio.\n"
        "**[1] R2 rec int** es la cifra limpia: el anillo exterior de k//2 px lo "
        "reconstruye un decodificador que ve ceros donde el codificador vio "
        "borde replicado (torch no admite `padding_mode` en `ConvTranspose2d`).\n"
        "⚠ `activa %` es la MEDIA sobre los canales, y con lambda=0 esa media "
        "esconde dos poblaciones -- medido: nueve canales al 99,97 % y siete "
        "muertos dan una media de 56,2 %, que no describe a ninguno. Por eso van "
        "al lado los VIVOS (ni muertos ni saturados) y su activacion.\n"
        "`n` = semillas promediadas.\n")
    md = md + legend + (f"\n{note}\n" if note else "")
    if out_md:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(md)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(out_csv).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([k for k, _ in COLUMNS])
            for g in grouped:
                w.writerow([g.get(k, "") for k, _ in COLUMNS])
    return md
