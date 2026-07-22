# Herencia de `image-text-finder`

Qué se extrae del proyecto hermano (`C:\Desarrollo\image-text-finder`), qué se **adapta** y qué
se **descarta**. Ese proyecto está completo y verificado (nueve pantallas, diez contratos con
test, recorridos con optuna, y el experimento P4 de visión periférica medido); es la **evidencia**
que sostiene este diseño. Cuando un documento de aquí afirma «medido», la medición vive allí.

**El problema a resolver es exactamente el mismo que allí** (esquinas de párrafo por ventana →
reconstrucción de párrafos); lo que cambia es **la red** (foveada de dos ramas en vez de CNN
plana) y que **los recorridos barren las configuraciones de la red** además de las recetas. Eso
maximiza lo heredable: etiqueta, pérdida, métrica de tarea, fuentes y diagnóstico se adoptan;
lo que se rediseña orbita alrededor de C y de H.

Regla de uso: **se hereda el mecanismo y las lecciones; el significado se conserva donde la
tarea es la misma.** No se copia código a ciegas — se consulta el suyo (`src/itf/`) como
referencia de algoritmos y de forma, igual que él consultaba su tag `pre-rediseno`.

---

## 1. Compatible tal cual (se adopta sin cambios)

| Pieza | Qué es | Dónde está allí |
|---|---|---|
| **Fuente A y su formato** | `labels.jsonl` + `dataset.json` + imágenes del generador (`image-text-sample-generator/SAMPLE_FORMAT.md`). Mismo generador, mismas fuentes; consumimos `quad` de párrafos — que es justo la etiqueta que el reconocimiento de párrafos necesita | `itf/datasets/loader.py` |
| **Fuentes derivadas (resize)** | Solo reducir; geometría reescalada recursiva (todo o nada); `derived.from` direccionable + `scale` real; segunda raíz en `data/sources/` | `itf/datasets/resize.py`, `itf/imageops.py`, formatos.md §4.6 |
| **Índice de offsets de A** | Mirar una imagen sin parsear 522 MB; caché con fecha+tamaño en la clave | `itf/datasets/index.py` |
| **Estructura del run (E)** | `config.json` + `provenance` (nombre+valor+huella+commit+environment), `metrics.jsonl` append-only, `status.json` explícito, `stop.json` cooperativo, `best.pt`/`last.pt`, escritura atómica con reintento en Windows | `itf/training/registry.py`, formatos.md §4.2 |
| **El validador puro** | `check_run` = función de dos dicts, llamado por **todas** las puertas antes de reservar nombre | `itf/validation/` |
| **La cola de jobs** | Límite de workers (1 en CPU), cancelación cooperativa, persistencia, resume en el `lifespan` | `itf/api/jobs.py` |
| **El motor de recorridos** | Spec nuestro (`spec.json`) + optuna como motor (un trial lanza un run y guarda su nombre); poda; reanudación desde disco | `itf/sweeps/` |
| **`matrixview`** | Matriz → payload (números + min/max/mean + `truncated` + trabajo de color declarado `sequential|diverging`). Sin importar nada del proyecto — está listo para compartirse | `itf/matrixview/` |
| **Reglas del API R1–R7** | Un recurso por dominio; renombrar nombres ambiguos; síncrono vs job (~1 s); error = `code` + `message` + `hint`; polling incremental `?since=`; agregados en el servidor; «si se entrenó con ello, tiene nombre» | api.md |
| **Seguridad local** | Allowlist de raíces (403 fuera), CORS cerrado al origen del front. Crítico aquí: el API acabará en un server con GPU | api.md §6 (D4) |
| **Método de tests** | Contratos = plan de pruebas; `xfail(strict=True)` como lista de tareas ejecutable; testear la costura, no la función; tests de dirección de imports; test de reproducibilidad con control; **nunca** testear resultados de investigación | tests.md |
| **Protocolo experimental** | N semillas y media±sd; suelo de ruido; empates dentro de la banda; test una vez al final solo el ganador; mismo commit + misma huella; holdout fuera de B generado primero | protocolo.md |
| **Reglas de UI** | Una pantalla un dominio; toda vista declara (fija, varía, mide); paleta validada por script, no a ojo; divergente ±0 para datos con signo; jamás doble eje; tabla de números como gemela accesible | ui.md |
| **Stack y entorno** | Python 3.12 + PyTorch + FastAPI + Vite/React; Playwright disponible; consola cp1252 → CLIs en ASCII; observaciones de la máquina (hibernación, throttling) | README/CLAUDE.md |

## 2. Compatible con adaptación (el mecanismo sirve, el contenido cambia)

| Pieza | Allí | Aquí | Por qué cambia |
|---|---|---|---|
| **B (el dato de la red)** | `patches.npz`: el patch horneado es la entrada; `images` se añadió después (D23) para la periferia | **`images` + etiquetas por ventana desde el día 0; la vista foveada se construye en el dataloader** | Es la lección central de su P4: hornear la vista en B obliga a re-extraer millones de ejemplos para mover un parámetro de geometría. Aquí TODA la geometría (`N`, `c_frac`, `d`, `pen_frac`) es barrible, así que la vista no puede vivir en B |
| **C (la red)** | CNN secuencial declarativa (`backbone[]` de bloques) | **Dos ramas por región + máscaras contributivas + fusión** (instructionsNewNN.md §6); config de parámetros fundamentales con derivados calculados | La arquitectura es otra; lo que se conserva es el patrón «config puro → build_model(dict)», la traza de validación previa y el checkpoint autodescriptivo |
| **Contrato ①** | `patch_size == input_size`, desdoblado en ①a/①b al llegar la periferia | ①a (ventana etiquetada) / ①b (vista computable: `original_size` cabe en lo que B guarda) | Mismo patrón, geometría distinta |
| **Contrato ⑤** | La ventana deslizante compartida entre extracción e inferencia (`itf.geometry`) | **La vista foveada** compartida entre dataloader e inferencia (`fv.fovea`) | Misma regla («una fórmula, dos lectores, un test de costura»), otra fórmula |
| **H (recorridos)** | Espacio sobre **D** con B y C fijos; barrer C quedó como decisión abierta (D22) | **Espacio sobre C y/o D con B fijo, desde el diseño** | Barrer la forma de la red es el caso de uso central aquí. Se hereda la maquinaria (spec/store/runner/poda/resume) y se amplía la definición; la comparabilidad la sostienen B y la métrica por imagen, no la red |
| **Rangos del espacio** | Escritos en el spec por el usuario | **Calculados por `fv.fovea.build_search_space`**; el spec puede referenciarlos (`"auto"`) o restringirlos | Principio rector de instructionsNewNN.md: los rangos son funciones de `N`, no constantes |
| **Catálogo de D** | BCE + λ·smoothL1 por esquina; `pos_weight` del desbalance; `smooth_l1_beta` explícito | **Se hereda entero, pérdida incluida** (F1 cerrada: las cabezas de esquina se conservan — decisiones.md C9), con las trampas ya pagadas: `lambda_pos` y el contrato ⑨, `smooth_l1_beta` (el default 1.0 anula el Huber), `pos_weight` medido del manifest | Misma cabeza, mismas trampas |
| **Cabeza de esquinas (`CornerHead`)** | `(4, 3)` = `[exists, x, y]` por tipo (TL/TR/BR/BL) sobre el flatten del backbone | Igual concepto, sobre el `feat` fusionado de las dos ramas. **Sustituye al clasificador de referencia de instructionsNewNN.md §6, y también a su `adaptive_avg_pool2d(feat, 1)`**: un pooling global destruye el «dónde», que es lo que la cabeza de posición predice | La spec foveada define muestreo y backbone; su cabeza es un placeholder. Decidido en F1 (decisiones.md C9). Queda **F1b**: sobre qué ventana se etiquetan las esquinas (centro vs campo completo) |
| **`corner_evidence` / esquinas ciegas / V18** | Evidencia geométrica por esquina, `(1-fx)(1-fy)` y simétricas, **congelada contra la ventana etiquetada** (regla R-b de su P4: no se redefine contra el campo de visión) | Igual, congelada contra la ventana etiquetada que fije F1b. Es además **la métrica que mide si la periferia aporta** — el criterio de éxito del primer experimento (protocolo.md §6) | Vuelve a aplicar porque vuelven las esquinas, y la regla de no redefinirla ya se aprendió allí |
| **Reconstrucción TL→BR (`reconstruct_boxes`)** | Emparejado voraz esquinas → cajas; heurística, no red | Igual, como punto de partida de la recomposición de F | Depende de la cabeza, que ahora es la misma |
| **Diagnóstico (E×B)** | Tabla por patch = **caché** (D1): recomputable, clave con mtime del checkpoint, agregados en servidor, threshold como parámetro de consulta | Igual, por ventana; los agregados concretos dependen de F1 | El patrón caché (recomputable/borrable/huella en la clave) está probado tres veces allí |
| **Métricas** | `fv.metrics`-equivalente (`itf.metrics`): un número se define una vez, arrays puros | Igual; las métricas de tarea se definen **en píxeles de la imagen original** para sobrevivir a barrer la geometría (contrato ⑨-extensión) | Nuevo requisito por barrer C |
| **Máscara de cobertura** | La máscara del canal de contexto: fracción real por celda, nunca binarizada, nunca blanco, se enseña junto a su canal (V19) | Igual para el relleno de la vista foveada en bordes de imagen | La lección es directamente transferible |
| **Vistas de la UI** | Catálogo V1–V19 sobre CNN de un canal | Se hereda el catálogo con adaptaciones (ui.md §4): las vistas por rama (kernels/feature maps de centro y periferia por separado), V19 (vista de entrada canal a canal) pasa a ser **fundamental**, y las sondas de 1 canal necesitan diseño multicanal | En su P4 las sondas de 1 canal se negaron (409) sobre redes multicanal; aquí la entrada es multicanal/multirrama de serie |

## 3. Se descarta (no aplica a este proyecto)

| Pieza | Por qué no |
|---|---|
| **El clasificador de referencia de instructionsNewNN.md §6** (`nn.Linear(ch2, num_classes)` + `adaptive_avg_pool2d`) | Era un placeholder estilo MNIST. F1 (decisiones.md C9) lo sustituye por la cabeza de esquinas; el avg-pool global destruye la posición |
| **Los flags de borde `border_features` y el contrato ② de `border`** | Existían porque el patch podía tocar el borde de la imagen sin más señal. Aquí la vista foveada lleva **máscara de cobertura** por construcción, que dice lo mismo con más resolución. Si un experimento los pide, se rediseñan sobre la máscara |
| **El desbalance 3,9:1 y el valor concreto de `pos_weight`** | Números de su dato. El mecanismo (medir el desbalance en el manifest de B y partir de ahí) sí se hereda |
| **Sondas específicas de entrada 1-canal** (occlusion tal cual, V10) | Ligadas a su entrada. Sobre la entrada compuesta se diseñan cuando toquen — occlusion ocluyendo **en la imagen original, pre-muestreo** (su P4 lo dejó explícitamente como «diseño real, no un fix») |
| **Código de migración / retrocompatibilidad** | Aquí no hay datos viejos. Todo artefacto nace completo (`format_version: 1`), y un campo que falta es un error, no un caso legado |

## 4. Las trampas medidas que este proyecto hereda como vacuna

Todas están **medidas** allí (organizacion.md §3 y CLAUDE.md del hermano; citas resolubles
contra su repo). Las que más probabilidad tienen de volver aquí:

1. **Defaults que deciden solos**: `momentum=0` en SGD, `smooth_l1_beta=1.0` con coords
   normalizadas (MSE pura sin avisar), `lr` constante sin scheduler.
2. **Fallos silenciosos con buena cara**: equivocarse de fuente por el sufijo construye un B
   válido que mide otra cosa; un knob de F en unidades absolutas significa algo distinto en cada
   run (va en unidades de la ventana); un overlay SVG y su imagen que no encuadran igual parecen
   un error del modelo.
3. **Rendimiento con cara de cuelgue**: releer la fuente entera para mirar una imagen (30 s);
   descomprimir 2,5 GB para leer una fila de un `.npz` comprimido (los cachés memmapeados con la
   huella en el **nombre del directorio** — en Windows no se puede borrar un fichero mapeado);
   construir cachés sin lock (cuatro sondas × 4 builds simultáneos).
4. **La vista por ítem sin vectorizar**: 5 ms/ítem = horas/época; vectorizada, 0,1 ms. Medido
   48× en su P4.2.
5. **Windows**: `os.replace` necesita reintento con deadline en escritor y lector; un fichero
   memmapeado no se borra; la consola cp1252 revienta con Unicode en `--help`.
6. **La máquina**: hiberna en corridas nocturnas (matar la 5ª semilla de su P4.4 costó el N=5);
   throttling ~5× en carga sostenida.
7. **Medir con la regla equivocada**: el tamaño de muestra efectivo son las imágenes, no las
   ventanas; un run aislado es una anécdota; el val del ganador de un recorrido está sesgado al
   alza — se reporta test/holdout, una vez.
8. **UI**: verificar con navegador (Playwright está); reiniciar el backend antes de verificar;
   una gráfica vacía tiene cara de gráfica (Plot en escala log descarta barras sin avisar).

## 5. Librerías compartidas

El hermano dejó cuatro candidatos library-shaped **sin extraer** (`matrixview`, `jobq`/cola,
`exp-registry`/runs, `convspec`): `claude-libs/` no existe. **Este proyecto es «la segunda vez»**
que su doctrina de extracción exigía (librerias.md §0 de allí: *se extrae en la segunda vez, no
en la primera*).

Decisión pendiente (decisiones.md **D2** de aquí): extraer al empezar la fase correspondiente
(crear `C:\Desarrollo\claude-libs\` y que ambos proyectos instalen editable) **o** copiar y
divergir. Recomendación registrada allí y aquí: extraer al menos `matrixview` (cero
dependencias, contrato probado) y la cola de jobs cuando la fase del API la toque; no
backportear nada al hermano hasta que le toque mantenimiento.
