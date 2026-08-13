# Watchdog del recorrido p40-lr-L4 (docs/plan-lr-L4.md §4): lo relanza si no
# esta corriendo. El recorrido es reanudable, asi que relanzar es SIEMPRE seguro:
# `run_sweep` salta lo done/cancelled y rehace lo demas.
#
# Cubre las dos formas en que el plan de 40 h ya murio:
#   - la sesion que lo lanzo se cerro y se llevo al hijo (paso el 2026-08-06)
#   - el equipo se apaga por falta de energia (confirmado por el usuario)
#
# ⚠ NO relanza con --stop-after: la sonda de §2 se corre a mano y se mira. Este
# watchdog es para la etapa B, y solo debe registrarse cuando esa etapa empiece.
$root = "C:\Desarrollo\foveal-vision"
$wlog = Join-Path $root "plan-lr-L4-watchdog.log"
$py = Join-Path $root ".venv\Scripts\python.exe"

# Sonda de una sola vez: comprueba que la cuenta que ejecuta esta tarea puede
# LEER la carpeta y EJECUTAR el venv. Corriendo como SYSTEM eso no es obvio, y
# un watchdog que no puede lanzar nada es peor que ninguno: da falsa seguridad.
# (Leccion del watchdog anterior, que fallo exactamente asi.)
$probe = Join-Path $root "plan-lr-L4-watchdog.probe"
if (-not (Test-Path $probe)) {
    try {
        $who = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        # el mecanismo EXACTO del relanzamiento: Start-Process con working dir y
        # redireccion dentro del proyecto. Probar '& python' no vale: no ejerce
        # ni el cwd ni los ficheros de redireccion, que es donde SYSTEM falla.
        $t = Join-Path $root "plan-lr-L4-watchdog.probe.out"
        $e = Join-Path $root "plan-lr-L4-watchdog.probe.err"
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

# ya terminado -> no hay nada que relanzar. El estado lo dice el propio
# recorrido, no un fichero aparte: una segunda copia del mismo dato es como se
# rompen las cosas en este proyecto.
$state = Join-Path $root "sweeps\p40-lr-L4\state.json"
if (Test-Path $state) {
    $s = Get-Content $state -Raw | ConvertFrom-Json
    if ($s.status -eq "done") {
        "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')  recorrido terminado, watchdog no hace nada" |
            Add-Content $wlog
        exit 0
    }
}

# ya corriendo -> no duplicar
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*plan_lr_L4.py*" }
if ($alive) { exit 0 }

# muerto -> relanzar (se reanuda solo)
$log = Join-Path $root "plan-lr-L4.run.log"
if (Test-Path $log) {
    $stamp = Get-Date -f "yyyyMMdd-HHmmss"
    Move-Item $log (Join-Path $root "plan-lr-L4.$stamp.log") -Force
}
$p = Start-Process -FilePath $py -ArgumentList "scripts\plan_lr_L4.py" `
    -WorkingDirectory $root -RedirectStandardOutput $log `
    -RedirectStandardError (Join-Path $root "plan-lr-L4.err") `
    -WindowStyle Hidden -PassThru
"$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')  relanzado, PID $($p.Id)" | Add-Content $wlog
