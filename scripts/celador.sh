#!/usr/bin/env bash
# Vigila AL VIGILANTE: corre `vigilar_entrenamiento.py` cada N segundos y avisa a
# Telegram cuando el veredicto CAMBIA.
#
#   scripts/celador.sh <run> [segundos] [max-edad-pesos]
#
# POR QUE HACE FALTA UNO MAS
# --------------------------
# `entrenar_vast.py` avisa cada `--aviso-cada` horas MIENTRAS vive. La pregunta
# que no puede contestar es la unica que importa cuando algo va mal: "¿sigues
# vivo?". Un proceso muerto no manda el aviso de que ha muerto.
#
# Y el dueno opera desde el movil y puede no estar delante. Sin esto, "el
# entrenamiento se cayo a los diez minutos" y "el entrenamiento va bien" se ven
# exactamente igual: silencio.
#
# ⚠ AVISA SOLO AL CAMBIAR DE ESTADO, no en cada vuelta. Un aviso que sale cada 5
# minutos se deja de leer en una hora, y entonces no avisa de nada (patron B de
# docs/revision-2026-08-22.md). Verde -> rojo avisa; rojo -> rojo calla; rojo ->
# verde avisa ("se arreglo solo").
#
# ⚠ Y NO INTENTA ARREGLAR NADA. Reiniciar el entrenamiento es de
# `Restart=on-failure` + `entrenar_para_inferencia.sh`, que sabe adoptar la
# instancia viva. Un celador que ademas relanzara seria un segundo sitio con esa
# decision, y los dos podrian lanzar a la vez.
set -uo pipefail
RUN="${1:?uso: $0 <run> [segundos] [max-edad]}"
CADA="${2:-300}"
MAX_EDAD="${3:-300}"
cd "$(dirname "$0")/.."
set -a; [ -f "$HOME/.config/dev-secrets.env" ] && . "$HOME/.config/dev-secrets.env"
[ -n "${COORD_HOME:-}" ] && [ -f "$COORD_HOME/.env" ] && . "$COORD_HOME/.env"; set +a

avisar() { [ -n "${COORD_HOME:-}" ] && node "$COORD_HOME/scripts/notify.mjs" "$1" >/dev/null 2>&1 || true; }

previo="arranque"
while true; do
  salida="$(.venv/bin/python scripts/vigilar_entrenamiento.py --name "$RUN" \
            --max-edad "$MAX_EDAD" 2>&1 | grep -v NNPACK)"
  if echo "$salida" | grep -q '^🟢'; then estado=ok; else estado=roto; fi
  echo "[$(date -u +%H:%M:%S)] $estado"
  echo "$salida" | sed 's/^/    /'

  # el entrenamiento TERMINO: el celador ya no pinta nada
  if .venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from fv import settings
from fv.training.registry import RunStore
import json
st = (RunStore(settings.runs_root()).status('$RUN') or {}).get('status')
sys.exit(0 if st in ('done','error','cancelled') else 1)" 2>/dev/null; then
    avisar "$RUN: el entrenamiento TERMINO. $(echo "$salida" | head -1)"
    echo "terminado; el celador se retira"
    exit 0
  fi

  if [ "$estado" != "$previo" ] && [ "$previo" != "arranque" ]; then
    if [ "$estado" = roto ]; then
      avisar "🔴 $RUN: algo va mal.$(printf '\n')$salida"
    else
      avisar "🟢 $RUN: se arreglo solo. $(echo "$salida" | head -1)"
    fi
  elif [ "$previo" = "arranque" ] && [ "$estado" = roto ]; then
    avisar "🔴 $RUN: el celador arranca y ya lo ve mal.$(printf '\n')$salida"
  fi
  previo="$estado"
  sleep "$CADA"
done
