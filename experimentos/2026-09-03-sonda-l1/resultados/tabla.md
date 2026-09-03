| run | k | K | lam | n | [4] GaborD/margen |   >p95 | [4c] orient D |   >p95 | [4c] banda D |   >p95 |   Gabor D |   nulo | [1] R2 rec int | [2] activa % med |   de los vivos | [3] vivos |   muertos |   saturados | [5] enriq x | [6] dim95/k2 | [7] cos max | [8] align |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3-K16-l0.0 | 3 | 16 | 0.00 | 1 | 0.651 | SI | -0.023 | no | -0.055 | no | 0.079 | 0.879 | 1.000 | 57.7 | 44.5 | 5 | 4 | 7 | 1.14 | 0.333 | 1.000 | 0.551 |
| k3-K16-l28.284271247461902 | 3 | 16 | 28.28 | 1 | 0.360 | SI | -0.019 | no | 0.004 | no | 0.043 | 0.879 | 0.883 | 7.9 | 9.7 | 13 | 3 | 0 | 1.08 | 0.667 | 0.921 | 0.682 |
| k5-K16-l0.0 | 5 | 16 | 0.00 | 1 | 0.615 | SI | 0.155 | SI | -0.012 | no | 0.298 | 0.515 | 1.000 | 56.2 | 0.0 | 0 | 7 | 9 | 0.86 | 0.320 | 0.902 | 0.509 |
| k5-K16-l28.284271247461902 | 5 | 16 | 28.28 | 1 | 0.197 | SI | 0.047 | SI | 0.004 | no | 0.096 | 0.515 | 0.875 | 6.7 | 6.7 | 16 | 0 | 0 | 0.95 | 0.400 | 0.719 | 0.871 |
| k7-K16-l0.0 | 7 | 16 | 0.00 | 1 | 0.786 | SI | 0.058 | SI | -0.004 | no | 0.522 | 0.337 | 1.000 | 50.0 | 50.1 | 2 | 7 | 7 | 1.59 | 0.143 | 0.842 | 0.534 |
| k7-K16-l28.284271247461902 | 7 | 16 | 28.28 | 1 | 0.287 | SI | -0.002 | no | 0.010 | no | 0.190 | 0.337 | 0.883 | 5.1 | 5.1 | 16 | 0 | 0 | 0.76 | 0.245 | 0.732 | 0.832 |
| k9-K16-l0.0 | 9 | 16 | 0.00 | 1 | 0.836 | SI | 0.032 | no | -0.040 | no | 0.646 | 0.228 | 1.000 | 68.6 | 33.5 | 6 | 1 | 9 | 2.27 | 0.049 | 0.949 | 0.555 |
| k9-K16-l28.284271247461902 | 9 | 16 | 28.28 | 1 | 0.349 | SI | 0.015 | no | 0.033 | SI | 0.270 | 0.228 | 0.885 | 5.0 | 5.0 | 16 | 0 | 0 | 0.67 | 0.148 | 0.305 | 0.826 |

**El criterio se lee en las seis primeras columnas de metrica**, y ninguna es un valor absoluto:
· `GaborD/margen` = (R2 del ajuste - su nulo) / (1 - su nulo). Dividir por el margen ALCANZABLE es lo que lo hace comparable entre `k`: con los nulos medidos, un 0,25 absoluto es el 52 % del margen en k=5, el 38 % en k=7 y el 32 % en k=9, o sea tres exigencias distintas.
· `>p95` = la mediana del run supera el p95 de la mediana de K kernels ALEATORIOS (bootstrap). Es la prueba, sin unidades; la magnitud la da la columna de al lado.
· `orient D` y `banda D` no dependen de ninguna plantilla, y por eso sobreviven a que la entrada este normalizada en contraste -- que es lo que rompe a `enriq` (ver `fv/probe/spectrum.py`).
⚠ `Gabor D` en crudo se conserva porque es lo que nombra el encargo, pero NO es comparable entre `k` por si solo.
⚠ `enriq x` vale 1 cuando el kernel es indistinguible de uno aleatorio, y esta MEDIDO que cae a 0,47-0,61 en toda la sonda por la normalizacion de contraste: leelo como diagnostico, no como criterio.
**[1] R2 rec int** es la cifra limpia: el anillo exterior de k//2 px lo reconstruye un decodificador que ve ceros donde el codificador vio borde replicado (torch no admite `padding_mode` en `ConvTranspose2d`).
⚠ `activa %` es la MEDIA sobre los canales, y con lambda=0 esa media esconde dos poblaciones -- medido: nueve canales al 99,97 % y siete muertos dan una media de 56,2 %, que no describe a ninguno. Por eso van al lado los VIVOS (ni muertos ni saturados) y su activacion.
`n` = semillas promediadas.
