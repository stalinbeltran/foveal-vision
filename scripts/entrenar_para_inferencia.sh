#!/usr/bin/env bash
# Lanza UN entrenamiento cuyo producto es EL MODELO, y lo hace RE-ARRANCABLE.
#
#   scripts/entrenar_para_inferencia.sh <run> <red> [args extra de entrenar_vast]
#
# POR QUE EXISTE, y no basta con llamar a entrenar_vast.py
# --------------------------------------------------------
# La unidad de systemd que lo envuelve lleva `Restart=on-failure`
# (desacoplar-persistente.sh), y eso es lo que hace que el trabajo sobreviva a que
# muera quien lo lanzo. Pero un re-arranque a ciegas es PEOR que no re-arrancar:
#
#   - `entrenar_vast.py` sin `--continuar` llama a `fv-train`, que se NIEGA si el
#     run ya existe ("no se sobrescribe nunca"). Fallo -> systemd reintenta a los
#     30 s -> alquila, falla, destruye, repite. Un bucle que gasta.
#   - `entrenar_vast.py --continuar` ALQUILA OTRA MAQUINA. Si la de antes sigue
#     viva porque lo que murio fue el vigilante, ahora hay DOS facturas y la
#     primera sin nadie que la destruya.
#
# Asi que aqui se mira el estado real antes de decidir, y hay tres:
#
#   A) hay una instancia viva con nuestra etiqueta  -> ADOPTARLA (no se alquila)
#   B) el run ya existe con metricas                -> `--continuar`
#   C) nada de lo anterior                          -> arranque normal
#
# ⚠ El caso A es el que costo dinero el 2026-08-31 y el que un reintento ingenuo
# empeora: el trabajo estaba intacto en la maquina y lo que faltaba era quien lo
# recogiera. `adoptar_vast.py` es exactamente eso.
#
# Se lanza SIEMPRE asi (regla 1 de docs/entrenar-para-inferencia.md):
#   "$COORD_HOME/scripts/desacoplar-persistente.sh" entrenar-<run> \
#       /bin/bash -lc 'cd ~/src/foveal-vision && scripts/entrenar_para_inferencia.sh <run> <red>'
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "uso: $0 <run> <red> [args extra]" >&2
  exit 2
fi
RUN="$1"; RED="$2"; shift 2
cd "$(dirname "$0")/.."

# los secretos se cargan de DISCO: por la unidad no viaja ninguno a proposito
set -a
[ -f "$HOME/.config/dev-secrets.env" ] && . "$HOME/.config/dev-secrets.env"
[ -n "${COORD_HOME:-}" ] && [ -f "$COORD_HOME/.env" ] && . "$COORD_HOME/.env"
set +a

PREFIJO="${PREFIJO:-mk-}"
ETIQUETA="${PREFIJO}${RUN}"

# --- A) ¿hay ya una maquina nuestra viva, huerfana de vigilante? -------------
IID="$(.venv/bin/python - "$ETIQUETA" <<'PY'
import sys
from pathlib import Path
raiz = Path(__file__).resolve().parents[0] if False else Path.cwd()
sys.path.insert(0, str(raiz.parent / "digital-ocean-dropplet-auto-launching" / "scripts"))
try:
    import vast_instance as V
    for i in V.instancias():
        if (i.get("label") or "") == sys.argv[1] and \
           str(i.get("actual_status") or "") != "exited":
            print(i.get("id")); break
except Exception:
    # ⚠ si NO se puede preguntar, NO se imprime nada y se sigue al caso B/C.
    # Es la unica decision de aqui que puede acabar alquilando de mas, y se
    # prefiere a la contraria: quedarse parado dejaria la maquina viva sin
    # vigilante, que es el estado que no se destruye NUNCA.
    pass
PY
)" || IID=""

if [ -n "${IID:-}" ]; then
  echo "hay una instancia viva ($IID) con etiqueta '$ETIQUETA': la ADOPTO en vez de alquilar"
  exec .venv/bin/python scripts/adoptar_vast.py --iid "$IID" --name "$RUN" --horas-max 6
fi

# --- B) ¿el run ya existe con metricas? -> continuar -------------------------
CONT=""
# ⚠ por RunStore y no por `runs_root()`: aquel resuelve el mes en que se creo el
# run y este solo apunta al mes ACTUAL. Un entrenamiento que cruza la medianoche
# del dia 1 se veria como "no existe" y empezaria de cero. (Visto el 2026-09-01.)
if .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from fv import settings
from fv.training.registry import RunStore
m = RunStore(settings.runs_root()).path('$RUN')/'metrics.jsonl'
sys.exit(0 if m.exists() and m.stat().st_size > 0 else 1)
" 2>/dev/null; then
  CONT="--continuar"
  echo "el run '$RUN' ya tiene metricas: continuo en vez de empezar de cero"
fi

# --- C) arranque (o continuacion) -------------------------------------------
exec .venv/bin/python scripts/entrenar_vast.py \
  --name "$RUN" --network "$RED" $CONT \
  --prefijo "$PREFIJO" --yes "$@"
