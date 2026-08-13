# Watchdog del plan de 40 h: lo relanza si no esta corriendo.
# El plan es reanudable, asi que relanzar es siempre seguro: salta lo hecho.
# Cubre las dos formas en que ya murio o puede morir:
#   - la sesion que lo lanzo se cerro y se llevo al hijo (paso el 2026-08-06 23:28)
#   - el equipo se apaga por falta de energia (confirmado por el usuario)
$root = "C:\Desarrollo\foveal-vision"
$report = Join-Path $root "plan-40h-report.json"
$wlog = Join-Path $root "plan-40h-watchdog.log"

# Sonda de una sola vez: comprueba que la cuenta que ejecuta esta tarea puede
# LEER la carpeta y EJECUTAR el venv. Corriendo como SYSTEM eso no es obvio, y
# un watchdog que no puede lanzar nada es peor que ninguno: da falsa seguridad.
$probe = Join-Path $root "plan-40h-watchdog.probe"
if (-not (Test-Path $probe)) {
    $py = Join-Path $root ".venv\Scripts\python.exe"
    try {
        $who = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        # el mecanismo EXACTO del relanzamiento: Start-Process con working dir y
        # redireccion dentro del proyecto. Probar '& python' no vale: no ejerce
        # ni el cwd ni los ficheros de redireccion, que es donde SYSTEM falla.
        $t = Join-Path $root "plan-40h-watchdog.probe.out"
        $e = Join-Path $root "plan-40h-watchdog.probe.err"
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

# ya terminado -> no hay nada que relanzar
if (Test-Path $report) {
    $r = Get-Content $report -Raw | ConvertFrom-Json
    if ($r.PSObject.Properties.Name -contains "confirm") {
        "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')  plan terminado, watchdog no hace nada" |
            Add-Content $wlog
        exit 0
    }
}

# ya corriendo -> no duplicar
$alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*plan_40h.py*" }
if ($alive) { exit 0 }

# muerto -> relanzar (se reanuda solo)
$log = Join-Path $root "plan-40h.log"
if (Test-Path $log) {
    $stamp = Get-Date -f "yyyyMMdd-HHmmss"
    Move-Item $log (Join-Path $root "plan-40h.$stamp.log") -Force
}
$p = Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") `
    -ArgumentList "scripts\plan_40h.py" -WorkingDirectory $root `
    -RedirectStandardOutput $log -RedirectStandardError (Join-Path $root "plan-40h.err") `
    -WindowStyle Hidden -PassThru
"$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')  relanzado, PID $($p.Id)" |
    Add-Content (Join-Path $root "plan-40h-watchdog.log")
