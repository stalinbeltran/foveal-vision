#!/bin/sh
# Los MISMOS stops que los otros cuatro: 0, 3, 11, 24, 37 epocas.
# Que caigan en las mismas epocas es lo que permite ponerlos uno al lado del otro.
set -eu
cd "$(dirname "$0")/../../.."
EXP=experimentos/2026-09-04-cnn-plana-1k7-sinpadding
avance() {
  .venv/bin/python "$EXP/nn/entrenar_local.py" seguir --more "$1"
  .venv/bin/python "$EXP/nn/evaluar_local.py" --stop "$2"
}
avance 2  01-3epocas
avance 8  02-11epocas
avance 13 03-24epocas
avance 13 04-37epocas
echo "CADENA COMPLETA"
