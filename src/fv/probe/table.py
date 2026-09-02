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
    ("gabor_delta", "[4] Gabor D"),
    ("gabor_r2", "  R2"),
    ("gabor_r2_base", "  nulo"),
    ("r2_rec_int", "[1] R2 rec int"),
    ("r2_rec", "  R2 rec"),
    ("frac_activa", "[2] activa %"),
    ("kernels_muertos", "[3] muertos"),
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
    if key in ("k", "K", "n", "kernels_muertos"):
        return f"{v:.0f}" if key != "kernels_muertos" else f"{v:.1f}".rstrip("0").rstrip(".")
    if key == "frac_activa":
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
        "\n\n**[4] es la metrica principal**: `Gabor D` = R2 del ajuste a Gabor "
        "menos el de kernels ALEATORIOS del mismo tamano. El valor absoluto no "
        "significa nada -- un Gabor tiene 7 parametros libres y ajusta ruido "
        "mejor de lo que uno espera.\n"
        "**[5] lleva su nulo dentro**: `enriq x` ya es `energia_6d / (6/k2)`, "
        "que vale 1 cuando el kernel es indistinguible de uno aleatorio. La "
        "fraccion cruda NO es comparable entre columnas de `k` distinto.\n"
        "**[1] R2 rec int** es la cifra limpia: el anillo exterior de k//2 px lo "
        "reconstruye un decodificador que ve ceros donde el codificador vio "
        "borde replicado (torch no admite `padding_mode` en `ConvTranspose2d`).\n"
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
