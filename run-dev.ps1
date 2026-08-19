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

# 백엔드는 --reload 없이 띄웁니다. uvicorn --reload는 소켓을 만드는 리로더 부모와 요청을
# 처리하는 워커 자식으로 나뉘는데, Windows에서 Ctrl+C가 부모만 죽이면 자식이 포트를 물고
# 살아남습니다. 이때 TCP 표의 소유 PID는 소켓을 만든 **죽은 부모**로 남기 때문에, 아래
# 정리 루프의 Get-Process가 $null을 받아 아무것도 죽이지 않습니다. 살아 있는 워커는 자기
# PID가 표에 없어 발견되지도 않습니다. 결과는 조용한 고착입니다 — 코드를 고치고 재시작해도
# 옛 코드가 계속 응답하고, 화면에는 아무 경고도 뜨지 않습니다(실측: 11초 차이로 저장한
# 수정이 반영되지 않았고 원인을 찾는 데 한참 걸렸습니다).
# 단일 프로세스면 concurrently -k가 확실히 정리하므로 이 상태 자체가 생기지 않습니다.
# 핫리로드를 포기하는 대신 "재시작하면 반드시 새 코드"라는 보장을 얻습니다.
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
