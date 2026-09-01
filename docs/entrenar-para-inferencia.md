# Entrenar una red que se va a USAR: las siete medidas, y por qué cada una

**Qué contesta este documento:** qué tiene que cumplir **todo** entrenamiento cuyo
producto es **el modelo** (no una fila de una tabla), y cómo se comprueba cada
cosa. Es una lista de obligaciones, no una guía.

**Por qué existe:** entrenar un punto de un barrido y entrenar una red para
inferir son **dos encargos distintos**. En el primero lo que se conserva es el
número y perder la máquina cuesta un punto; en el segundo lo que se conserva es
**el `.pt`**, y perder la máquina cuesta **el encargo entero**. Cada regla de aquí
abajo sale de algo que ya pasó, con lo que costó anotado.

> **La regla de entrada:** si el producto es el modelo, se usa
> **`scripts/entrenar_vast.py`**, no `estudio_flota.py`. Aquél se trae los pesos en
> cada sonda; éste los deja en la máquina hasta el tar final — correcto para un
> barrido, ruina para un modelo.

---

## Las siete

### 1. Se lanza como UNIDAD de systemd, nunca como hijo de la sesión

```bash
"$COORD_HOME/scripts/desacoplar-persistente.sh" entrenar-<run> \
  /bin/bash -lc '...'
```

**Nunca `desacoplar.sh`** (que usa `systemd-run --scope`): aquél da cgroup propio
—sobrevive al restart del coordinador— pero **sigue siendo hijo** de quien lo
lanza, y un tree-kill al padre se lo lleva.

*Medido el 2026-08-31:* se lanzó así desde una sesión de Claude Code, la sesión
terminó, el vigilante murió con ella — y **el entrenamiento siguió vivo en Vast 1 h
38 min con la factura corriendo y nadie mirando**.

| | restart del coordinador | muerte de su padre |
|---|---|---|
| `desacoplar.sh` (scope) | ✅ | ❌ |
| `desacoplar-persistente.sh` (unidad) | ✅ | ✅ |

**Comprobación:** `systemctl status entrenar-<run>` y que su `PPID` sea 1.

### 2. Techo de gasto Y techo de reloj, los dos

`--presupuesto <$>` corta por dinero; `--horas-max <h>` corta por tiempo. **No son
redundantes**: un entrenamiento que se cuelga sin avanzar no gasta más rápido, pero
sigue gastando. El presupuesto es el techo duro; el reloj es la red por si algo se
queda quieto.

**Comprobación:** que los dos estén en la línea de lanzamiento. Sin uno de los dos,
no se lanza.

### 3. Los pesos bajan MIENTRAS entrena, y se comprueba que están bajando

`--cada <segundos>` es el intervalo de sonda: en cada una `entrenar_vast.py` se trae
`best.pt` y `last.pt` a la **antesala** (`data/inferencia/<run>/`). Con el defecto
(60 s) lo peor que se puede perder es un minuto de entrenamiento.

⚠ **Y hay que MIRAR que están llegando, no suponerlo.** Un `--cada` configurado no
es un fichero en disco: si el `scp` falla por red, el log lo dice y el
entrenamiento sigue tan campante. La comprobación es el `mtime`:

```bash
.venv/bin/python scripts/vigilar_entrenamiento.py --name <run> --max-edad 300
```

*Medido el 2026-08-30:* no guardar ningún peso costó `fov-optimo-p20` entero — 69
épocas en Vast, reentrenadas desde cero.

⚠ **Y cuando el run TERMINA, la pregunta cambia de sitio.** La antesala vacía pasa
a ser lo normal —al promover, los pesos van al repo de datos y la antesala se
limpia—, así que lo que hay que preguntar deja de ser *«¿están bajando?»* y pasa a
ser *«¿sobrevivió el modelo?»*, que se le pregunta al **catálogo**. El vigilante lo
hace solo. *Se descubrió el 2026-09-01 porque daba rojo sobre `fov16-edge-p20`, que
estaba aprobada y commiteada desde el día anterior: un vigilante que llama avería a
un éxito es un vigilante que se apaga.*

### 4. La red se aprueba para inferencia al terminar, y eso es una ORDEN del dueño

`entrenar_vast.py` promueve al terminar (antesala → repo de datos + catálogo) salvo
`--sin-promover`. **La promoción sólo se hace si el dueño la pidió**: los pesos de
un run no se guardan por defecto y sólo las redes aprobadas puede usarlas la app
([inferencia.md](inferencia.md) §1). Copiar y aprobar son **la misma decisión**.

⚠ **Y promover no es empujar.** El commit y el push son parte del encargo: lo que
no está empujado no existe, y el repo de datos es lo único que sobrevive a rehacer
la máquina.

### 5. Si la máquina de Vast se cae, se cambia de máquina — con tope

`--max-cambios <n>` (6 por defecto): ante una máquina muerta o demasiado lenta,
`entrenar_vast.py` alquila otra y **continúa** desde `last.pt`. El tope existe para
que un problema que no es de la máquina —una config que revienta— no alquile
máquinas en bucle.

⚠ **Un run continuado en otra máquina no es bit a bit el mismo.** Para entrenar un
modelo da igual; para publicar una tabla comparable, no.

### 6. Si el vigilante muere igual, hay dos redes debajo

Contra SIGKILL no hay `finally` que valga. Por eso:

- **`cerrable.mjs` avisa aparte** de una máquina de Vast viva **sin ningún
  vigilante** (`⚠ N máquina(s) Vast SIN VIGILANTE`) — el único estado en que la
  instancia no se destruye nunca, ni con el server encendido;
- **`scripts/adoptar_vast.py`** vuelve a engancharse a una instancia huérfana en vez
  de alquilar otra.

**Comprobación:** `node ~/src/telegram-coordinator/scripts/cerrable.mjs` tiene que
nombrar el entrenamiento mientras corra. Si no lo nombra, el freno no lo ve.

### 6 bis. Si el cambio toca el MODELO, se reinicia la app antes de darla por buena

Un campo nuevo en la red (`mask_channel`, `edge_inputs`, `dropout`…) cambia la
**forma** de los pesos. El servicio `foveal-vision-web` es un proceso de larga
vida: si lleva corriendo desde antes de tu commit, construye la red **vieja** y el
checkpoint nuevo no le encaja.

```bash
sudo systemctl restart foveal-vision-web     # y comprobar que la red carga
```

⚠ **Y el síntoma engaña**: sale `[checkpoint_incompatible] … reentrena el run`, que
manda a **gastar en Vast para arreglar un modelo que está perfecto**. *Pasó el
2026-09-01 con `fov16-mask-p20`: el servicio llevaba vivo desde 2 h 26 min antes
del commit.* Desde entonces el cargador **distingue las dos averías** —si el
checkpoint declara campos que el proceso no conoce, el que se quedó atrás es el
proceso— y lo dice: `[checkpoint_de_codigo_mas_nuevo] … reinicia el servicio. NO
reentrenes`. Un test lo fija.

### 7. El criterio se escribe ANTES de mirar

Un documento de plan con **qué se entrena, con qué se compara y qué desenlace
significa qué** ([protocolo.md](protocolo.md) §1). Sin él, cualquier número que
salga se lee como confirmación de lo que ya se creía.

⚠ **Y lleva escrito que UNA red no declara nada.** Una semilla no distingue de
ruido: una red entrenada para usarla es un artefacto, no un resultado.

---

## La lista, para pegar antes de lanzar

```
[ ] plan escrito con el criterio ANTES de mirar          (7)
[ ] `npm test` / pytest en verde con el cambio           (—)
[ ] prueba de humo LOCAL: entrena 1 época de verdad      (—)
[ ] se lanza con desacoplar-persistente.sh               (1)
[ ] --presupuesto y --horas-max, los dos                 (2)
[ ] --cada <= 300 s                                      (3)
[ ] promoción pedida por el dueño (o --sin-promover)     (4)
[ ] --max-cambios con tope                               (5)
[ ] si el cambio toca el modelo, reiniciar fv-api        (6 bis)
[ ] cerrable.mjs lo nombra tras lanzar                   (6)
[ ] vigilar_entrenamiento.py dice que los pesos bajan    (3)
```

⚠ **La prueba de humo local no es opcional y no está en ninguna de las siete
porque no es una medida de seguridad: es la que impide gastar en un fallo de
código.** *Pasó el 2026-09-01 al implementar `mask_channel`: la sonda de
introspección reventaba con `expected input to have 1 channels` porque el reparto
de ramas estaba escrito dos veces. Con un solo canal las dos copias daban lo
mismo, así que la divergencia no existía hasta que existió.*
