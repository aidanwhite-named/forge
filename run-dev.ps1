$ErrorActionPreference = "Stop"

$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
  throw "Virtual environment not found. Create it with: python -m venv .venv"
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  throw "Node.js/npm is not installed."
}

if (-not (Test-Path "frontend/node_modules")) {
  npm --prefix frontend install
}
if (-not (Test-Path "node_modules")) {
  npm install
}

$ports = @(5374, 8330)
foreach ($port in $ports) {
  $connections = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
  foreach ($connection in $connections) {
    $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
    if ($process) {
      Write-Host "Stopping process on port $port`: $($process.ProcessName) ($($process.Id))" -ForegroundColor Yellow
      Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
  }
}

Write-Host "Starting frontend on 5374 and backend on 8330 using .venv." -ForegroundColor Green
npm run dev:raw
