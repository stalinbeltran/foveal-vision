#!/bin/sh
# El punto de entrada de la UNIDAD de systemd. Se lanza asi:
#
#   COORD_HOME="$HOME/src/telegram-coordinator" \
#   "$COORD_HOME/scripts/desacoplar-persistente.sh" soft-argmax \
#     /home/deploy/src/foveal-vision/experimentos/2026-09-04-soft-argmax/nn/cadena.sh
#
# ⚠⚠ MANDA EL CODIGO DEL TRABAJO, NO EL DEL AVISO.
#    La unidad lleva `Restart=on-failure`. Si el `notify.mjs` del final decidiera
#    el codigo de salida, un aviso fallido relanzaria la cadena entera -- medido
#    en este repo el 2026-09-02 (sonda L1) y el 2026-09-04 (37 epocas, 62
#    reinicios). Por eso el aviso va con `|| true` y el `exit` es el del trabajo.
set -u
REPO=/home/deploy/src/foveal-vision
COORD_HOME="${COORD_HOME:-/home/deploy/src/telegram-coordinator}"

cd "$REPO"
"$REPO/.venv/bin/python" "$REPO/experimentos/2026-09-04-soft-argmax/nn/cadena.py" "$@"
CODIGO=$?

if [ "$CODIGO" -eq 0 ]; then
  MSG="soft-argmax: los 3 brazos terminados. Resultados en foveal-vision/experimentos/2026-09-04-soft-argmax/nn/pesos/{A,B,C}/metrics.jsonl"
else
  MSG="soft-argmax: la cadena FALLO (codigo $CODIGO). Log: journalctl -u soft-argmax"
fi
node "$COORD_HOME/scripts/notify.mjs" "$MSG" || true

exit "$CODIGO"
