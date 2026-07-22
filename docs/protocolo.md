# Protocolo experimental

Cuándo un resultado de este proyecto es **creíble**, y qué hay que hacer antes de gastar CPU (o
GPU) en un recorrido. Heredado del proyecto hermano, donde cada regla se pagó con una medición;
adaptado a la tarea (reconocimiento de párrafos) y al plan CPU → GPU.

Sin protocolo, un recorrido produce un ranking en el que **no se distingue el ganador del
ruido**, y se elige a cara o cruz creyendo que se midió.

---

## 1. Las reglas de comparación

Para poder decir *«X es mejor que Y»*:

1. **Mismo commit de git.** Un cambio en la pérdida o el optimizador mata las comparaciones
   anteriores. El run registra el commit (procedencia).
2. **Misma huella de B** (contrato ⑧). Un dataset reconstruido bajo el mismo nombre **no es**
   el mismo dataset; solo la huella lo detecta.
3. **Mismo entorno declarado.** `environment` (python/torch/plataforma/device) va en la
   procedencia. **Los runs de CPU de hoy son los que se compararán con los de GPU mañana** — sin
   este campo esa comparación no puede ni declararse. Cambiar de versión de torch mueve
   resultados sin mover el commit.
4. **N semillas, media ± sd. Nunca un número suelto.** Un run aislado es una anécdota. N=5 es el
   canon; N=3 ve el signo de un efecto grande.
5. **La diferencia supera la banda de ruido** (§4). Si no, es un empate — y no se rompe con
   «pero es que este subió».
6. **El test se toca una sola vez, al final, y solo el ganador.** El val hace dos trabajos
   (elegir `best.pt` y rankear el recorrido) y por eso el val del ganador está **sesgado al
   alza**; ese número no se reporta. El holdout, una vez, al final del todo.

## 2. La métrica que manda es la de la tarea, por imagen

El objetivo real es **si los párrafos se reconocen bien en la imagen completa** — no la métrica
por ventana que el bucle calcula por época. Del hermano se hereda la estructura:

| | Métrica |
|---|---|
| **Entrenar** (la pérdida) | La de D — tiene que ser diferenciable |
| **Elegir `best.pt`** dentro de un run | val de ventana (barato, por época), `monitor` explícito |
| **Rankear el recorrido** | **métrica de tarea por imagen, en val** ← el objetivo |
| **Reportar** | métrica de tarea en **holdout**, una vez, del ganador |

Con las cabezas de esquina (C9), la métrica de tarea se hereda tal cual del hermano: **F1 de
párrafo** — emparejar las cajas de la reconstrucción TL→BR con los `quad` reales por
IoU ≥ 0,5, más el IoU medio de los emparejados. (Su D7 sigue viva aquí: contra el *bbox* del
quad basta mientras `angle≈0`.) Dos propiedades **no negociables**, las dos por el contrato ⑨:

- **Independiente de los pesos de la pérdida** — o barrer `lambda_*` «gana» bajándolos.
- **Definida en píxeles de la imagen original** — o barrer la geometría foveada (`N`, `d`)
  cambia la regla de medir a la vez que el modelo. Es la extensión propia de este proyecto:
  aquí **se barre C**, así que ninguna métrica de ranking puede depender de la vista.

**Paso obligado antes del primer recorrido grande** (paso 2 del hermano): medir la correlación
de rangos (Spearman) entre la métrica de ventana y la de tarea sobre ~8 runs de calidad variada.
Si es alta, el proxy barato sirve para diagnóstico; si es baja, el eslabón débil es la
recomposición de F y mejorar la red no mejora la tarea — cualquiera de las dos respuestas vale
su coste.

## 3. El instrumento antes que el experimento

**Orden: 0 → sesgos → suelo de ruido → proxy → recorrido.**

### Paso 0a — el holdout, generado lo primero

Heredado (D16 del hermano), con sus tres respuestas:

- **Cuándo**: antes que ningún dataset de entrenamiento. Generado después, la sospecha de
  haberlo elegido no se puede descartar.
- **Cómo**: su **propia fuente** (`<nombre>-holdout`), de la que jamás se extrae entrenamiento —
  hace la fuga físicamente imposible. Misma config del generador, otra semilla.
- **Cuánto**: ~500 imágenes (allí: sd teórico del recall ≈ 0,65 %, y la independencia entre
  ventanas es falsa, así que el suelo real es peor — 500 deja margen). Coste: una corrida del
  generador y cero entrenamiento.

**Se mide por imagen**, y por eso sirve para cualquier configuración de B y de C — incluso
barriendo la geometría.

### Paso 0b — dimensionar el dato como variable, no como default

El dato es sintético y el generador está al lado: **el tamaño del dataset es una variable de
investigación** (D6 del hermano), no una restricción. La lección medida: **train manda en el
coste; val manda en la resolución de la medida** — son knobs independientes, y un split
porcentual único los acopla sin querer. Punto de partida razonable: ~2000 imágenes 80/10/10,
para medir, no como conclusión.

### Paso 0c — quitar los sesgos antes de medir

Los defaults con trampa se ponen a propósito **antes** de la baseline (arreglarlos después
invalida lo ya entrenado): `momentum` explícito, `scheduler` explícito, y los que la pérdida de
F1 traiga (si hay smoothL1: su `beta`). Barrer antes de esto es **medir el bug**.

### Paso 1 — el suelo de ruido

5 runs idénticos variando solo el `seed` de D (el de B no se toca). Media ± sd de la métrica de
tarea. El resultado es **un número: la diferencia mínima creíble** — y la baseline sale gratis.

> **Toda diferencia dentro de la banda es un empate.** Si los 6 primeros puntos de un recorrido
> caben en la banda, el resultado es «seis empatados», no «ganó el primero».

Este número decide también si el paso 0 fue suficiente: si el suelo queda por encima de las
diferencias que importan, se genera más val. Es un bucle.

### Paso 2 — validar el proxy (§2)

## 4. Recorridos: presupuesto y disciplina

- **Fijado antes de mirar**: el criterio de éxito de cada experimento se escribe **antes** de
  lanzarlo (el P4 del hermano lo hizo y es lo que hace creíble su resultado). Un criterio
  escrito después se racionaliza.
- **La poda es la palanca nº1 en CPU** (medido: 14 de 30 puntos podados). Parada temprana
  (per-run, D) ≠ poda (entre runs, H).
- **Presupuesto en épocas o en segundos — se declara.** Barriendo C, el coste por época no es
  constante entre puntos: la poda por tiempo favorece a las redes pequeñas sin decirlo. La
  estimación de coste de instructionsNewNN.md §10 (asimétrica ~10–15 % más rápida que la CNN
  equivalente) es orden de magnitud, no medida: se mide aquí en cuanto exista el bucle.
- **CPU primero, corto**: el mismo spec de recorrido se valida en esta máquina con dataset
  pequeño y pocas épocas (¿corre entero? ¿reanuda? ¿los nombres y la procedencia cuadran?), y
  después se lanza con presupuesto real en la GPU. El spec es el mismo objeto; cambia
  `budget` y `execution`.
- **Desatendido de verdad**: en esta máquina, desactivar la suspensión (`powercfg`) antes de una
  corrida nocturna, o contar con la pausa. Presupuestar el throttling (~5× medido en carga
  sostenida).
- **Réplicas del ganador**: el ganador de un recorrido se confirma con N semillas antes de
  reportarlo (el recorrido corre con 1 semilla por punto; la réplica es posterior y solo del
  ganador o del grupo empatado).

## 5. Resultados medidos

*(Vacío a propósito: aquí se escriben, fechados y con sus salvedades, los resultados de los
experimentos de este proyecto — el equivalente del §5.5/§5.6 del hermano. Un resultado de
investigación no va a pytest: va aquí, y envejece aquí.)*

## 6. La cola de análisis pendientes

*(El equivalente del §9 del hermano: cada entrada fija la pregunta, por qué está sin contestar,
cómo se contesta y con qué se rompe. Se llena cuando el instrumento exista. El primer candidato
ya se conoce: **¿la periferia foveada mejora el reconocimiento respecto a una CNN plana del
mismo coste?** — el análogo del P4 del hermano, cuyo resultado allí (err_px ciego 2,30→1,14,
~17× la sd, N=4) es la razón de ser de esta arquitectura.)*
