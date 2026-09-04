> Encargo del dueño, 2026-09-04 (por Telegram), **corrigiendo el intento anterior**:

Ajá, hay un error en eso. Detén este experimento donde está. No ejecutes nada más. Y vamos a
crear otro donde los datasets de entrada van a ser construidos antes del entrenamiento

---

## Cómo llegó aquí

El encargo original (2026-09-04, literal en
[`../2026-09-04-preproceso-kernel-congelado/instrucciones/01-encargo.md`](../2026-09-04-preproceso-kernel-congelado/instrucciones/01-encargo.md))
pedía *«3 datasets pre-procesados… y con ellos genera 3 cnn planas para 3 estudios»*.

El primer intento metió el preproceso **dentro del modelo** como capa 0 congelada. La decisión
estaba razonada —coste en disco, y la cobertura real del canal de relleno— pero **contestaba a
otra pregunta**. El dueño lo detectó leyendo la tabla de estructura:

> Las estructuras dicen que las entradas de las cnn son siempre data 20x20???

Sí lo eran, y ése era el fallo. Con el preproceso dentro del modelo:

1. la entrada seguía siendo la vista `(2,20,20)` de siempre, no el dataset preprocesado;
2. las redes tenían **dos** convoluciones, así que **no eran planas** — y «plana» en esta serie
   significa una convolución y su cabeza, como en los siete gemelos;
3. el preproceso dejaba de ser un **paso previo** para ser parte de la red, que es justo la
   distinción que el encargo separaba.

**Un argumento correcto sobre la pregunta equivocada sigue siendo un fallo.** El intento anterior
queda detenido en la época 11 de 37, conservado como el brazo *«mismo preproceso pero con ReLU en
medio»* — su geometría y su ancho de cabeza son idénticos a los de aquí, así que sirve de
comparación en vez de tirarse.

## Lo que este experimento hace distinto, en una línea

**Los tres datasets se construyen y se escriben a disco ANTES de entrenar**, y lo que entrena
encima es una CNN plana de verdad: una convolución y su cabeza.
