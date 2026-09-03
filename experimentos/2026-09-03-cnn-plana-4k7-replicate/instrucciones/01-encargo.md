> Encargo del dueño, 2026-09-03 (por Telegram, tras documentar el anillo de padding):

Otro experimento, igual a este, pero sin anillo de padding

---

**Lectura:** «igual a este» se toma como **igual a `cnn-plana-4k7`** (el de 4 kernels, que es
donde se midió el anillo), para que la única variable que cambie sea el relleno. «Sin anillo de
padding» se implementa como `conv_pad_mode: replicate`: la convolución replica el borde en vez
de rellenar con ceros.

Mismos stops que los otros dos experimentos: 0 (sin entrenar), 3, 11, 24 y 37 épocas.
