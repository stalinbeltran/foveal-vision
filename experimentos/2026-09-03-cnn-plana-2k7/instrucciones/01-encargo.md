> Encargo del dueño, 2026-09-03 (por Telegram, tras el cuarto avance del experimento de 4 kernels):

Vamos ahora a realizar otro experimento, idéntico a este, pero con sólo 2 kernels. Usa los mismos
stops, para poder comparar stops.

---

**Lectura:** «idéntico» se toma literal — misma geometría, misma entrada de 2 canales, misma
cabeza de 12 salidas, misma receta `plan40`, misma semilla 1, mismo dataset y **las mismas 10
ventanas** del set de visualización. Lo único que cambia es `channels: [4]` → `[2]`.

«Los mismos stops» se toma como **las mismas épocas acumuladas**: 0 (sin entrenar), 3, 11, 24 y
37 — para poder poner un stop al lado del otro.
