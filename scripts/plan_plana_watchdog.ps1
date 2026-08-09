# Watchdog de los estudios de la CNN plana (docs/plan-plana.md §5): relanza
# `plan_plana.py` si no esta corriendo. El script es reanudable Y lleva su propia
# guarda (no entrena nada mientras `p40-lr-L4` siga vivo), asi que relanzarlo es
# SIEMPRE seguro: si no toca hacer nada, sale en un par de segundos.
#
# Cubre las dos formas en que esto ya murio antes:
#   - la sesion que lo lanzo se cerro y se llevo al hijo (paso el 2026-08-06)
#   - el equipo se apaga por falta de energia (confirmado por el usuario)
#
# A DIFERENCIA del watchdog de lr-L4, este NO rota el log en cada relanzamiento:
# mientras espera despierta cada 10 min durante ~34 h, y rotar dejaria cientos de
# ficheros. El log de verdad es `plan-plana.log`, que el script escribe en modo
# append; lo de aqui son solo las ultimas stdout/stderr del proceso.
$root = "C:\Desarrollo\foveal-vision"
$wlog = Join-Path $root "plan-plana-watchdog.log"
$py = Join-Path $root ".venv\Scripts\python.exe"

# Sonda de una sola vez: comprueba que la cuenta que ejecuta esta tarea puede
# LEER la carpeta y EJECUTAR el venv. Un watchdog que no puede lanzar nada es
# peor que ninguno: da falsa seguridad. (El watchdog anterior fallo asi.)
$probe = Join-Path $root "plan-plana-watchdog.probe"
if (-not (Test-Path $probe)) {
    try {
        $who = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        # el mecanismo EXACTO del relanzamiento: Start-Process con working dir y
        # redireccion dentro del proyecto. Probar '& python' no vale: no ejerce
        # ni el cwd ni los ficheros de redireccion, que es donde una cuenta de
        # servicio falla.
        $t = Join-Path $root "plan-plana-watchdog.probe.out"
        $e = Join-Path $root "plan-plana-watchdog.probe.err"
        # el script de -c va como UN argumento: sin las comillas internas
        # PowerShell lo parte por los espacios y python recibe basura
        $p = Start-Process -FilePath $py `
            -ArgumentList '-c', '"import os,sys;print(sys.version.split()[0],os.getcwd())"' `
            -WorkingDirectory $root -RedirectStandardOutput $t -RedirectStandardError $e `
            -WindowStyle Hidden -PassThru -Wait
        $out = (Get-Content $t -Raw -ErrorAction SilentlyContinue)
        $err = (Get-Content $e -Raw -ErrorAction SilentlyContinue)
        if ($out) { $out = $out.Trim() } else { $out = "(sin salida) stderr=$err" }
        Remove-Item $t, $e -Force -ErrorAction SilentlyContinue
        "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')  SONDA ok  usuario=$who  exit=$($p.ExitCode)  " +
        "start-process=[$out]  (este log lo escribio esa misma cuenta: prueba de escritura)" |
            Add-Content $wlog
    } catch {
        "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')  SONDA FALLO: $_" | Add-Content $wlog
    }
    New-Item -ItemType File -Path $probe -Force | Out-Null
}

# ya corriendo -> no duplicar. Dos ejecutores a la vez se pisarian los runs.
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*plan_plana.py*" }
if ($alive) { exit 0 }

# no corriendo -> lanzarlo. El decide si toca esperar, seguir o no hacer nada:
# el estado vive en los estudios y en el recorrido, no aqui (una segunda copia
# del mismo dato es como se rompen las cosas en este proyecto).
$p = Start-Process -FilePath $py -ArgumentList "scripts\plan_plana.py" `
    -WorkingDirectory $root -RedirectStandardOutput (Join-Path $root "plan-plana.run.log") `
    -RedirectStandardError (Join-Path $root "plan-plana.err") `
    -WindowStyle Hidden -PassThru
