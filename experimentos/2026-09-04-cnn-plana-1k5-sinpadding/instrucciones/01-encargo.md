> Encargo del dueño, 2026-09-04 (por Telegram):

Podrias en paralelo (o a continuación) correr otro experimento con 1 kernel con 5x5?

---

**Se hizo A CONTINUACIÓN, no en paralelo, y el motivo se midió.** El run de 7×7 estaba usando
**122 % de CPU en una máquina de 2 vCPU** (`ps -eo pcpu`, 2026-09-04 00:42 UTC), así que dos
entrenamientos a la vez no habrían dado ganancia real de reloj —se habrían repartido los mismos
dos núcleos— y además habrían **contaminado el `seconds_per_epoch` de los dos**. Ese número lo
publican los READMEs de los cinco experimentos anteriores y está medido con la máquina
descargada; compararlo contra uno medido bajo contención sería comparar dos cosas distintas sin
decirlo.

**Lectura del encargo:** es [`cnn-plana-1k7-sinpadding`](../../2026-09-04-cnn-plana-1k7-sinpadding/)
con `k_center: 7 → 5`. Se hereda todo lo demás: `padding=0` de verdad, código local sin tocar
`src/fv/`, receta `plan40`, semilla 1, dataset `dirty1000-80px-16px-r20260827`, las mismas 10
ventanas de `../comun/` y los mismos stops (0, 3, 11, 24, 37 épocas).
