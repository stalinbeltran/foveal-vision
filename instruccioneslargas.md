# Encargo: sonda L1 — ¿pueden los kernels de la primera capa aprender filtros genéricos?

## Contexto

En `fov16-optimo-mask` los 16 kernels de L1 no aprendieron filtros genéricos. Medido
sobre `best.pt` del run `fov16-mask-p20`: energía en el subespacio clásico 6D
(DC, Sobel-x/y, laplaciano, dos diagonales) = **0.688**, contra 6/9 = **0.667** de un
kernel aleatorio. Enriquecimiento nulo. Dimensión efectiva por PCA: 5–6 de 9. Y hay un
par duplicado (k5/k7, coseno +0.96), ambos DC negativo puro.

Hipótesis: L1 no está bajo presión. No hay reducción tras ella y detrás hay una cabeza
de 153.660 parámetros que puede extraer las esquinas de casi cualquier proyección.
Además 3×3 tiene 9 dimensiones y el subespacio clásico ya ocupa 6: no hay sitio para
que emerja estructura.

Este experimento es **aparte**. No modifica `src/fv/models/builder.py`, ni los configs
de `configs/networks/`, ni nada que cambie el significado de un checkpoint en disco.

## 1. Estructura

Autoencoder convolucional de una sola capa por lado. El modelo *son* los kernels.

```
x (1,20,20)  →  Conv2d(1, K, k, stride=1, padding=k//2, bias=True)  →  ReLU
             →  z (K,20,20)   ← la "imagen simplificada", el entregable
             →  ConvTranspose2d(K, 1, k, stride=1, padding=k//2, bias=False)
             →  x̂ (1,20,20)
```

No negociable:

- **Decodificador lineal, sin sesgo, una sola capa.** Si puede compensar un `z` malo,
  la presión sobre los kernels desaparece — que es el fallo que esto quiere evitar.
- **Nada entre codificador y decodificador**: sin batchnorm, sin pooling.
- **Stride 1 y padding `k//2`**: la resolución se conserva. No queremos una imagen más
  pequeña, queremos una más genérica al mismo tamaño.
- `pad_mode` replica el borde, igual que la red de producción.

## 2. Preprocesado (obligatorio)

Normalización de contraste local por ventana: restar la media local (gaussiana σ≈2 px)
y dividir por la desviación local con un ε en el denominador.

Sin esto, la componente de mayor varianza es el nivel medio de intensidad y la pérdida
gasta ahí sus primeros grados de libertad — es literalmente lo que produjo k5 y k7.

## 3. Esparsidad

```
loss = mse(x̂, x) / var(x)  +  λ · mean(|z|)
```

- El primer término, normalizado por la varianza, hace que λ signifique lo mismo entre
  configuraciones.
- El segundo penaliza **activaciones**, no pesos. Penalizar pesos hace la red pequeña;
  penalizar activaciones la hace selectiva, que es lo que produce filtros con
  significado. Sin este término el autoencoder converge a algo equivalente a PCA:
  base válida pero difusa, sin estructura local.

**Bloqueo de la salida degenerada:** tal cual, el modelo puede multiplicar el
codificador por 0.01 y el decodificador por 100 — misma reconstrucción, penalización
cien veces menor. Aprendería a hacer `z` pequeño en vez de disperso. Por eso: tras cada
paso del optimizador, **renormalizar cada kernel del decodificador a norma L2 = 1**. El
codificador queda libre. Sin esto el experimento no mide lo que dice medir.

**Diagnóstico de λ:** registrar la fracción de activaciones positivas en `z`. Objetivo
5–15 %. Por encima del 30 %, λ es baja; por debajo del 2 % o con kernels muertos, alta.
Es un diagnóstico, no va en la pérdida.

## 4. Barrido

Rejilla completa (36 combinaciones), no OAT: interesa la interacción entre `k` y λ.

| Eje | Valores | Motivo |
|---|---|---|
| `k` | **5, 7, 9** | Eje principal: en 3×3 no cabe la estructura |
| `K` | **8, 16, 32** | Con k=9 y K=32 el código es 32× sobrecompleto |
| `λ` | **0.0, 0.03, 0.1, 0.3** | λ=0 es el control obligatorio (caso tipo PCA) |

Modelos diminutos (`K·k²·2 + K` parámetros). ~30 épocas por run; cronometra una
combinación antes de lanzar las 36. Semilla fija, y repetir las 3 mejores con 3
semillas para separar señal de ruido.

Datos: `dirty1000-80px-16px-r20260827`, canal imagen únicamente, mismo split de
validación. Sin máscaras, sin ramas, sin cabeza de esquinas.

## 5. Métricas (sobre validación, al final de cada run)

1. **R² de reconstrucción** = 1 − mse/var.
2. **Fracción de activaciones positivas** en `z`.
3. **Kernels muertos**: activos en <0.1 % de las posiciones.
4. **Ajuste a Gabor** — métrica principal. Ajustar un Gabor 2D a cada kernel por
   mínimos cuadrados no lineales; reportar la mediana del R² sobre los K kernels.
   ⚠ Calcular la **misma métrica sobre K kernels aleatorios del mismo tamaño**. Un
   Gabor tiene muchos parámetros libres y ajusta ruido mejor de lo que uno espera. Se
   compara la diferencia, nunca el valor absoluto.
5. **Energía en el subespacio clásico**, generalizada a k×k. Su línea base es 6/k², que
   cambia con `k` y **no es comparable entre columnas**. Reportar valor y línea base
   juntos.
6. **Dimensión efectiva**: componentes de PCA para el 95 % de la varianza, entre k².
7. **Redundancia**: coseno máximo entre pares distintos.
8. **Alineación codificador/decodificador**: coseno entre el kernel i de cada uno. Si
   convergen (>0.8) son el mismo objeto; si no, hay que decidir cuál es el entregable.

## 6. Artefactos

- Módulo aislado `src/fv/probe/`, sin importar `fv.models` ni `fv.fovea` salvo el
  cargador de ventanas.
- Por run: config, `metrics.jsonl`, checkpoint, kernels del codificador y del
  decodificador en `.npy`.
- Hoja de contactos por run: los K kernels del codificador como mapas divergentes,
  escala de color común, norma en el título.
- Para las 3 mejores: figura con la entrada y sus K mapas `z` al lado — es la que
  responde visualmente a "¿la imagen resultante es más genérica?".
- Tabla comparativa de las 36 combinaciones con las 8 métricas.

## 7. Criterios — escribir ANTES de entrenar

Crea `docs/plan-sonda-l1-<fecha>.md` y **párate a que el dueño confirme los umbrales
antes de escribir el entrenamiento**. Propuesta de partida:

- **Éxito** si alguna configuración logra a la vez: mediana de R² Gabor ≥ 0.25 por
  encima de su línea base aleatoria, R² de reconstrucción ≥ 0.80, activación entre 5 % y
  15 %, y cero kernels muertos.
- **Fracaso** si ninguna separa el Gabor de su línea base por más de 0.10. Sería un
  resultado válido: la reconstrucción tampoco produce filtros genéricos en este dominio.
- Si el mejor λ=0 iguala al mejor λ>0 en la métrica Gabor, la esparsidad no aportó nada
  y hay que decirlo.

## 8. Fase 2 (solo si la fase 1 tiene éxito)

Congelar el codificador de la mejor configuración como L1 de la **rama central** de
`fov16-optimo-mask`, con `k_center` y `channels[0]` iguales a sus `k` y `K`. La rama
periférica se entrena normal: la sonda se entrenó con 1 canal y la periferia recibe 2.
Entrenar el resto con la receta `plan40` y comparar contra el f1 **0.954** y el error de
posición **1.05 px** de `fov16-mask-p20`.

Si el f1 aguanta con L1 congelada, los kernels son transferibles. Si se hunde,
aprendieron a reconstruir y nada más — también es un resultado, y sale barato.

⚠ Ese modelo cambia la forma de L2 y **no debe intentar cargar checkpoints anteriores**.

Guarda todos los scripts empleados.