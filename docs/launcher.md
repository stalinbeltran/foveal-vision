# Registro en el App Launcher (`launcher.json`)

`launcher.json` en la raíz registra esta app en el App Launcher. **El launcher solo
entiende el esquema de abajo**; cualquier otro campo se ignora y la app no arranca.

## Esquema exacto (no inventar campos)

```json
{
  "name": "Nombre visible de la app",
  "description": "Una línea de qué hace.",
  "processes": [
    { "name": "backend",  "cmd": "<comando con {PORT_API}>" },
    { "name": "frontend", "cwd": "web", "cmd": "<comando con {PORT_WEB}>",
      "env": { "VAR_AL_BACKEND": "http://127.0.0.1:{PORT_API}" } }
  ],
  "ports": [
    { "name": "PORT_API", "preferred": 8010 },
    { "name": "PORT_WEB", "preferred": 5173 }
  ],
  "open": "http://localhost:{PORT_WEB}"
}
```

## Reglas obligatorias

- `processes` es un **array**, no un objeto. El launcher ejecuta cada `cmd` en orden.
- La clave es **`cmd`**, no `command`.
- Puertos con **llaves**: `{PORT_API}`, `{PORT_WEB}`. **Nunca** `${VAR}` ni `%VAR%`.
  Cada nombre entre llaves debe estar declarado en `ports`.
- Declara cada puerto en `ports` con `preferred`. El launcher usa el preferido si está
  libre, o busca otro; el mismo `{PORT_API}` referenciado en el frontend recibe el puerto
  **real** del backend.
- Si un proceso corre en subcarpeta (frontend en `web/`), usa `"cwd": "web"`. Los comandos
  **no** usan rutas absolutas.
- **Prohibido**: `services`, `placeholders`, `dependsOn`, `command`, `url`, `port` sueltos
  o cualquier otro campo.

## Cómo se derivaron los comandos de ESTA app

Fuente: [README.md](../README.md) §"Correr la app" + [web/vite.config.ts](../web/vite.config.ts).

- **backend** — dominio X. `python -m fv.api --host 127.0.0.1 --port {PORT_API}`.
  Corre en la raíz (usa `.\.venv\Scripts\python.exe`). Sin env propias. Puerto por defecto
  **8010** (evita el 8000 del proyecto hermano).
- **frontend** — Vite en `web/`. `npm run dev -- --port {PORT_WEB}`. El proxy de `/api` lee
  **`FV_API_URL`** (`vite.config.ts`, fallback `http://127.0.0.1:8010`) → por eso el `env`
  del frontend es `FV_API_URL: http://127.0.0.1:{PORT_API}`.

### Caveat de puerto del frontend

`vite.config.ts` fija `port: 5173` con `strictPort: true`, y **5173 está en la allowlist de
CORS del backend**. El `-- --port {PORT_WEB}` deja que el launcher elija el puerto real, pero
si no es 5173, CORS lo rechazará hasta actualizar la allowlist del backend. Mantener 5173
como `preferred` evita el problema en el caso normal.

## Verificar

```powershell
py -3.12 - <<'EOF'
import json, re
d = json.load(open('launcher.json'))
declared = {p['name'] for p in d['ports']}
used = set()
for p in d['processes']:
    used |= set(re.findall(r'\{(\w+)\}', p['cmd']))
    for v in p.get('env', {}).values(): used |= set(re.findall(r'\{(\w+)\}', v))
used |= set(re.findall(r'\{(\w+)\}', d['open']))
assert not (used - declared), used - declared
print("OK", declared, used)
EOF
```
