#!/bin/sh
# Los MISMOS stops que el gemelo de 4 kernels: 0, 3, 11, 24, 37 epocas.
# Un stop es una foto en el tiempo, asi que se evalua ENTRE avance y avance --
# volver luego con `--run` leeria `last.pt`, que ya seria otra epoca (el
# evaluador se niega, pero el orden correcto es este).
set -eu
cd "$(dirname "$0")/../../.."          # raiz del repo
EXP=experimentos/2026-09-03-cnn-plana-4k7-replicate
EV="experimentos/comun/aplicar_kernels.py"
RUN=plana-4k7rep-s1

avance() {   # $1 = epocas a anadir, $2 = etiqueta del stop
  .venv/bin/fv-continue --name "$RUN" --more "$1"
  .venv/bin/python "$EV" --exp "$EXP" --red plana-20-4k7-rep --stop "$2" --run "$RUN"
}

avance 2  01-3epocas
avance 8  02-11epocas
avance 13 03-24epocas
avance 13 04-37epocas
echo "CADENA COMPLETA"
