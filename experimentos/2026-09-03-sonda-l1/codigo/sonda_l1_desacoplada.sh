#!/bin/sh
# La sonda L1 tal como la lanza el ejecutor de Telegram, como UNIDAD de systemd.
#
# POR QUE ESTO ES UN FICHERO Y NO UNA LINEA DENTRO DEL JSON DEL EJECUTOR
# ----------------------------------------------------------------------
# Porque tiene semantica que hay que poder PROBAR, y una tuberia escapada dentro
# de un `case` de un JSON no se prueba. Lo que decide es el codigo de salida, y
# `desacoplar-persistente.sh` registra la unidad con `Restart=on-failure`:
#
#   · si el AVISO decidiera el codigo, un fallo del aviso --sin BOT_TOKEN, red
#     caida, hilo borrado-- relanzaria la rejilla ENTERA cada 30 s. Medido el
#     2026-09-02 con el arnes `test-executor.mjs`, que no pasa BOT_TOKEN: la
#     sonda TERMINO BIEN, `notify.mjs` fallo, y la unidad quedo en
#     `Result=exit-code` reiniciandose. 12 h de trabajo re-lanzadas por un aviso.
#
#   · y al reves: si el aviso enmascarara el fallo del trabajo, una sonda que
#     revienta saldria como `success` y NO se reintentaria, que es justo cuando
#     el reintento sirve (`--rejilla` se reanuda saltando los runs ya hechos).
#
# Asi que: manda el codigo del TRABAJO, y el aviso no puede cambiarlo. Tiene test.
set -u

PYTHONUNBUFFERED=1 .venv/bin/python scripts/sonda_l1.py "$@"
estado=$?

if [ -n "${COORD_HOME:-}" ] && [ -f "$COORD_HOME/scripts/notify.mjs" ]; then
  if [ "$estado" -eq 0 ]; then
    aviso="sonda L1 terminada. Resultados: foveal-vision-data/sondas/l1/ (tabla.md, resumen.json), log en /tmp/sonda-l1.log"
  else
    aviso="sonda L1 FALLO (codigo $estado). Mira /tmp/sonda-l1.log; lo ya escrito en foveal-vision-data/sondas/l1/ se conserva y --rejilla se reanuda saltando esos runs."
  fi
  node "$COORD_HOME/scripts/notify.mjs" "$aviso" || true
fi

exit $estado
