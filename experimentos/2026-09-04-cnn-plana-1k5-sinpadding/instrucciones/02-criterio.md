# El criterio, **escrito antes de que terminara ninguno de los dos**

> Congelado el 2026-09-04 a las 00:45 UTC. En ese momento el run de 7×7 iba por la época 22 de
> 37 y **su mejor f1 todavía no se conocía**; de este de 5×5 no había ni una época. Lo que se
> escribe abajo no puede cambiarse al ver los números.
>
> ⚠ **Lo que sí se sabía**, y se dice para no fingir un ciego perfecto: las épocas sueltas del
> 7×7 que iban saliendo en el log rondaban f1 ≈ 0,58–0,61. Por eso el criterio de aquí está
> escrito **en relativo** —contra el f1 que acabe dando el 7×7, sea cual sea— y no contra un
> número absoluto.

## ⚠⚠ Este experimento mueve DOS cosas a la vez, y a propósito

Sin relleno, un kernel más pequeño **recorta menos**, así que bajar de 7×7 a 5×5 no deja la red
igual: le **agranda** la cabeza.

| | mapa (valid) | features | L1 (params) | campo receptivo | total |
|---|---|---:|---:|---:|---:|
| 1k**7** sin relleno | 14×14 | **196** | 99 | 7 px | 2.511 |
| 1k**5** sin relleno | 16×16 | **256** | 51 | 5 px | 3.183 |

Los dos efectos empujan **en sentidos opuestos**:

- **+31 % de features** → por la tendencia medida en la serie (~0,09 de f1 por cada factor 2 de
  features), esto **debería SUBIR** el f1 en ≈ 0,09 × log₂(256/196) = **+0,035**.
- **la mitad de parámetros en L1 y un campo receptivo de 5 px en vez de 7** → esto debería
  **BAJARLO**, en una cantidad que nadie ha medido.

**Eso no es un defecto del diseño: es lo que le permite decidir cuál de los dos manda.** Con el
relleno puesto los dos ejes se podrían separar (el mapa seguiría siendo 20×20), pero entonces
volvería el anillo y el experimento dejaría de ser comparable con su gemelo. Se elige mantener
la comparabilidad y **declarar el confound**, no esconderlo.

## Los tres desenlaces

Sea **F₇** el mejor f1 del run de 7×7 (desconocido al escribir esto). Banda de ruido **0,04**, la
misma que fijó el criterio del 7×7 y que sale de la oscilación de f1 en las últimas 9 épocas de
los cuatro runs anteriores (0,019 · 0,020 · 0,025 · 0,039 — se toma el peor).

| si el mejor f1 del 5×5 cae en | veredicto | qué significa |
|---|---|---|
| **F₇ + 0,035 ± 0,04**, o sea `[F₇ − 0,005, F₇ + 0,075]` | **manda el número de features** | el campo receptivo de 7 px no aportaba nada medible sobre este dato; lo que la serie estaba midiendo todo el rato era el tamaño de la cabeza |
| **< F₇ − 0,005** (peor que el 7×7 **pese a tener más features**) | **manda el campo receptivo** | y sería el resultado **más informativo de toda la serie**: diría que la tendencia de ~0,09 por mitad NO es una ley sobre «features», porque aquí suben y el f1 baja |
| **> F₇ + 0,075** | **hay algo más** | más de lo que la tendencia predice. Candidato obvio: con 51 parámetros en L1 la red sobreajusta menos. Habría que mirar la brecha val/train antes de creerlo |

⚠ **Ningún desenlace mueve producción.** Esto mide una plana de una capa; la foveada tiene 4
capas y `k_center: 3`. Lo que este punto puede informar es hacia dónde seguir mirando.

## Lo que este run NO puede contestar, dicho antes

1. **Una semilla.** Acota, no declara.
2. **Los dos ejes van juntos.** Separarlos pediría un tercer run —5×5 **con** relleno, para tener
   196 features con campo de 5— y ése reintroduce el anillo, así que no es una comparación
   limpia tampoco. La separación de verdad pide recortar la vista de antemano, y **no está
   hecho**.
3. **`k` = 3 no se prueba.** Sería el siguiente punto del eje y es el valor que usa la foveada de
   producción, pero el encargo pidió 5×5.
