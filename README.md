# Evidence Forge

청구항과 최대 7개 PDF를 입력받아 PDF 텍스트/문단을 추출하고, 문서 유형 분류·문단/페이지 인용·BM25+벡터 유사도 RRF 검색·구성요소별 비교 리포트를 제공하는 React + FastAPI 애플리케이션입니다.

## 실행

가장 간단한 방법은 루트 디렉터리에서 아래 명령 하나를 실행하는 것입니다.

```powershell
npm install
npm run dev
```

실행 전 포트 `5374`, `8330`을 점유한 기존 프로세스가 있으면 자동 종료하고 재시작합니다. Windows PowerShell에서는 `./run-dev.ps1`도 사용할 수 있습니다. 프론트엔드는 `http://localhost:5374`, 백엔드는 `http://localhost:8330/docs`에서 확인합니다. 종료하려면 `Ctrl+C`를 누릅니다.

### 개발 환경

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

개별 실행이 필요하면 백엔드는 `uvicorn app.main:app --app-dir backend --reload`, 프론트엔드는 `npm --prefix frontend run dev`를 사용합니다.

### Docker

`docker compose up --build`로 프론트엔드와 백엔드를 함께 실행합니다. 분석 완료물은 `backend/data/history/`, 작업 로그는 `backend/data/logs/`에 저장됩니다.

## agy 연동

실제 CLI만 사용합니다. `LLM_PROVIDER=agy|claude|gpt`와 `LLM_COMMAND`를 설정하십시오. agy는 `agy -p`, Claude는 `claude -p`, GPT는 `gpt -p` 형식으로 호출되며, CLI 명령은 `backend/app/agy.py`의 단일 어댑터에서 관리합니다. `shell=False`, 인자 배열, timeout, retry, stdout/stderr 분리를 사용합니다.

`-p`는 값을 받는 플래그이므로 프롬프트는 항상 인자 배열의 마지막에 둡니다. Windows 명령줄은 32767자로 제한되므로 프롬프트가 24000자를 넘으면 임시 디렉터리의 파일로 저장해 `--add-dir`로 전달하고 CLI가 직접 읽게 합니다. agy 모델명(`gemini-3.6-flash-medium` 등)은 reasoning effort를 접미사로 포함하므로 `--effort`를 함께 주지 않고 모델명 접미사를 교체합니다.

## API

`POST /api/jobs`, `GET /api/jobs/{job_id}/result`, `GET /api/jobs/{job_id}/download?format=md|txt`, `GET/DELETE /api/history`, `GET/PUT /api/logs`, `GET/PUT /api/settings`, `GET /api/settings/models`를 제공합니다. `DELETE /api/history/{job_id}`는 해당 분석의 결과·리포트·작업 로그를 함께 지우고, `DELETE /api/history`는 저장된 전체 기록과 로그를 한 번에 지웁니다(로그를 남기려면 `?logs=false`). `GET /api/settings/models`는 `agy models` 출력을 캐시해 설정 화면 모델 드롭다운에 제공하며(`?refresh=true`로 재조회), 조회에 실패하면 빈 목록을 반환해 UI가 직접 입력으로 넘어갑니다. 현재 작업은 요청 단위로 처리하며, 다음 단계에서 SSE 작업 큐로 교체할 수 있도록 작업 상태 모델을 분리했습니다.

## 테스트

```powershell
pytest backend/tests
cd frontend
npm run build
```

보안상 업로드는 PDF 확장자·파일 크기·개수·전체 크기를 검증하고, 파일명은 경로 요소를 제거해 저장합니다. 기본 로그에는 PDF 원문·청구항 원문·agy 원문 응답을 기록하지 않습니다.
