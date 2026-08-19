# Benchmark de velocidad por capacidad de vCPU

## El encargo

Medir cuánto tarda **una época de entrenamiento** en droplets de distinta
capacidad de vCPU, para decidir sobre qué máquina conviene entrenar de verdad.

La forma de hacerlo es la de siempre en este montaje: la máquina de trabajo
—la que tiene el bot de Telegram y con la que se conversa— **no se mide a sí
misma**. Crea una máquina por tamaño, le pone el dato y el benchmark, recoge el
número y **la destruye**. Ella sigue viva; las medidas son desechables.

```
coordinador (vive)          droplets de medición (nacen y mueren)
     │
     ├── lanza ──────────►  c-2   → mide → reporte → destruido
     ├── lanza ──────────►  c-4   → mide → reporte → destruido
     └── lanza ──────────►  c-8   → mide → reporte → destruido
```

## Qué se mide, exactamente

`scripts/bench_speed.py`, sin tocar nada: red `bench-16` y receta `bench`
**congeladas**, sobre el window dataset **congelado** `bench-dirty1000-16`. La
métrica es `seconds_per_epoch`, la misma que usan los estudios.

> ⚠️ **Dos reportes sólo se comparan si traen el mismo `window_dataset` y la
> misma huella de dato.** Cambiar la fuente mueve el número: por eso
> `benchmarks/foveal_20260813-134338.json` (que es `bench-synth-16`) no se
> compara con los de `bench-dirty1000-16`. El reporte guarda el nombre del
> dataset justo para que la comparación se filtre en vez de suponerse.

## De dónde sale el dato, y por qué eso era el problema

El benchmark mide sobre el dato **real**: las 1000 imágenes de la receta `dirty`
del generador hermano, reducidas a 80×60 y troceadas en ventanas de 16 con paso
8 y semilla 1. `bench_speed.py` se niega —bien— a fabricar una fuente de juguete
si no la encuentra.

El problema es que en una máquina recién hecha **nunca está**, y reconstruirla
son tres pasos en dos repos. Eso se ha resuelto mal más de una vez: se dio por
imposible y se midió sobre otra fuente, gastando una corrida entera.

Ahora la cadena es un comando por tramo (`scripts/bench_dataset.py`):

| tramo | qué hace | coste |
|---|---|---|
| `build` | mil renders con Chromium → fuente 640×480 → reducida a 80 px → ventanas | ~15-20 min |
| `publish --to /mnt/bench-data` | copia el resultado al volumen y le pone una huella SHA-256 | segundos |
| `install --from /mnt/bench-data` | copia del volumen al `data/` local | segundos |
| `verify` | comprueba que lo que hay en disco es lo que dice la huella | segundos |

**Es reproducible, y está comprobado.** Los specs están congelados en git
(`specs.jsonl`, seed 1) y la extracción tiene su propia semilla. Medido el
2026-08-19: renderizando dos veces el mismo spec, los `sha256` de los PNG salen
**idénticos byte a byte**. Generar el dataset no es un riesgo, es una espera.

### Por qué un volumen, y por qué aun así se copia

El dataset vive en un **volumen de bloques** (`bench-data`, 10 GB, 1 $/mes). Un
volumen es lo único de la cuenta que sobrevive a su droplet: sin él, cada
máquina nueva vuelve a pagar los 15-20 minutos de renders, que es exactamente el
punto en el que la cosa se abandona.

Pero los droplets de medición **no montan el volumen**: se les **copia** el dato
al disco local antes de empezar. Dos razones, y las dos importan:

1. Un volumen de DigitalOcean se conecta a **un** droplet a la vez. No es un
   disco compartido, así que no hay forma de que tres máquinas midan a la vez
   leyendo de él.
2. Aunque la hubiera, no se querría: hay que medir la **máquina**, y leer el
   dataset de un disco de red mediría la red.

La huella se verifica **en el destino**, después de copiar. Un dataset a medias
daría un número más rápido y con exactamente la misma pinta que uno bueno.

## Cómo se corre

Primero, comprobar que la máquina tiene con qué (ver
[el CLAUDE.md del coordinador](https://github.com/stalinbeltran/telegram-coordinator)):

```bash
node ~/src/telegram-coordinator/scripts/bench-preflight.mjs
```

Si el dato aún no está en el volumen, una vez en la vida de ese volumen:

```bash
python3 scripts/bench_dataset.py build
python3 scripts/bench_dataset.py publish --to /mnt/bench-data
```

Y la flota:

```bash
python3 scripts/bench_fleet.py --vcpus 2,4,8
```

**Tarda decenas de minutos, así que no se lanza dentro de un turno de chat.** Se
desacopla y avisa al terminar (la regla está en el CLAUDE.md del coordinador: un
mensaje es un proceso que muere y se lleva a su vigilante):

```sh
setsid sh -c 'cd ~/src/foveal-vision && python3 scripts/bench_fleet.py --vcpus 2,4,8 \
  > /tmp/fleet.log 2>&1; \
  node ~/src/telegram-coordinator/scripts/notify.mjs "flota terminada: /tmp/fleet.log"' &
```

El aviso es una comodidad; **la fuente de verdad son `benchmarks/vcpu_*.json` y
`/tmp/fleet.log`**, y se miran al principio del turno siguiente.

## Lo que se espera encontrar

Esto es una **hipótesis, no una medida** — es justo lo que la corrida tiene que
decidir. Se escribe aquí para que el resultado se pueda contrastar con lo que se
creía, y no al revés:

- **Referencia ya medida**: 59,4 y 61,8 s/época en el droplet de trabajo
  (`s-2vcpu-4gb`, 2 vCPU **compartidas**, sin GPU). Las dos corridas están en
  `benchmarks/foveal_20260814-*.json`.
- **Se espera que baje al subir vCPU, pero menos que proporcionalmente.** La red
  es pequeña (16 y 32 canales, ventanas de 16×16) y el lote es de 64: con más
  hilos, la parte paralelizable de cada paso se reparte, pero el coste por lote
  que no escala (el bucle de Python, el dataloader, la sincronización) se queda
  igual. Duplicar vCPU **no** debería dividir el tiempo por dos.
- **Se espera que `c-2` (dedicada) sea más rápida y sobre todo más ESTABLE que
  `s-2vcpu-4gb` (compartida)**, con el mismo número de vCPU. Lo interesante ahí
  no es la media sino la desviación: en la corrida de referencia hay un ±5,3 s
  sobre 61,8 (un 9%), y esa dispersión es de vecinos, no de la red neuronal.
- **Si el tiempo NO baja al subir vCPU**, la conclusión no es que las máquinas
  grandes no sirvan: es que este benchmark está limitado por algo que no es
  cómputo paralelo, y entonces la pregunta siguiente es cuál (memoria,
  dataloader de un solo hilo, tamaño de lote demasiado pequeño para llenar los
  núcleos).

Ese último caso es un resultado, no un fallo. Conviene anotarlo como tal.

## Coste

Cada droplet factura **por segundo mientras exista**. La corrida completa de
`--vcpus 2,4,8` son tres máquinas vivas unos 20-30 minutos: céntimos. Lo caro
es olvidarse una encendida.

- Todos nacen con el tag `bench-efimero` y **nadie más usa ese tag**.
- Se destruyen en un `finally`: pase lo que pase con la medición.
- Si aun así queda algo vivo: `python3 scripts/bench_fleet.py --reap`.
- Y para mirar: `python3 ~/src/digital-ocean-dropplet-auto-launching/scripts/do_droplet.py list`.

## Al terminar

Los reportes se commitean. Estos servidores son efímeros y lo que no está
empujado no existe: una medición que se queda en el disco de un droplet es una
corrida de benchmark tirada.
