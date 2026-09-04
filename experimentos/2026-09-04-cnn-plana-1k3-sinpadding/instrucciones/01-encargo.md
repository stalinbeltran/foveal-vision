> Encargo del dueño, 2026-09-04 (por Telegram):

Ahora uno con 1 kernel 3x3

---

**Lectura:** tercer punto del eje `k` con un solo kernel, después de
[7×7](../../2026-09-04-cnn-plana-1k7-sinpadding/) y [5×5](../../2026-09-04-cnn-plana-1k5-sinpadding/).
Se hereda todo lo demás: `padding=0` de verdad, código local sin tocar `src/fv/`, receta
`plan40`, semilla 1, dataset `dirty1000-80px-16px-r20260827`, las mismas 10 ventanas de
`../comun/` y los mismos stops (0, 3, 11, 24, 37 épocas).

⚠ **Y `k` = 3 no es un valor cualquiera: es el que usa la foveada de producción**
(`fov16-optimo-mask` tiene `k_center: 3`). Los otros dos puntos del eje no lo eran.
