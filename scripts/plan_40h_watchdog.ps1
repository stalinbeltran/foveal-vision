# Watchdog del plan de 40 h: lo relanza si no esta corriendo.
# El plan es reanudable, asi que relanzar es siempre seguro: salta lo hecho.
# Cubre las dos formas en que ya murio o puede morir:
#   - la sesion que lo lanzo se cerro y se llevo al hijo (paso el 2026-08-06 23:28)
#   - el equipo se apaga por falta de energia (confirmado por el usuario)
$root = "C:\Desarrollo\foveal-vision"
$report = Join-Path $root "plan-40h-report.json"

# ya terminado -> no hay nada que relanzar
if (Test-Path $report) {
    $r = Get-Content $report -Raw | ConvertFrom-Json
    if ($r.PSObject.Properties.Name -contains "confirm") {
        "$(Get-Date -f 'yyyy-MM-dd HH:mm:ss')  plan terminado, watchdog no hace nada" |
            Add-Content (Join-Path $root "plan-40h-watchdog.log")
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
