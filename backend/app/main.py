import json
import os
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from .config import (DATA_DIR, HISTORY_DIR, JOBS_DIR, JOB_RECORD_TTL_MINUTES, LOG_DIR,
                     LOG_MAX_BYTES, MAX_PDF_SIZE_MB, MAX_TOTAL_UPLOAD_SIZE_MB, AGY_MODEL,
                     load_runtime_settings, save_runtime_settings)
from . import agy, cache, priorart
from .agy import _build_command, available_models
from .claims import ancestry, parse_claims
from .models import AnalysisResult, DependentClaimsAdd, Document
from .pdf import extract_pdf
from .pipeline import analyze, extend_with_dependent_claims, summarize_matrix, uncovered_elements
from .report import to_markdown

app = FastAPI(title="Patent Evidence Analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5374"], allow_methods=["*"], allow_headers=["*"])
jobs: dict[str, dict] = {}
# 작업이 아직 돌고 있어 결과를 건드리면 안 되는 상태. 후속 작업 엔드포인트가 **같은 집합**을
# 봐야 합니다. 종전에는 선행기술 검색만 여기에 "cancelled"를 더 넣어 두어서, 종속항 대비를
# 취소해 보고서가 보존된 뒤에도(그 취소는 히스토리를 일부러 남깁니다) 그 보고서에서는
# 검색을 다시 누를 수 없었습니다. 종속항 추가는 같은 상태에서 그대로 허용되었으므로,
# 한 상태를 두 엔드포인트가 반대로 해석하고 있었습니다.
_ACTIVE_STATUSES = {"preparing", "running", "cancelling"}
# 걷어내면 안 되는 상태는 이보다 좁습니다. "preparing"은 후속 작업을 막아야 하는 상태이면서
# 동시에 **버려진 작업이 영원히 머무는 상태**이기도 합니다(prepare만 하고 탭을 닫은 경우).
# 업로드가 실제로 들어오는 중인지는 _staging으로 따로 봅니다.
_UNSWEEPABLE_STATUSES = {"running", "cancelling"}

def safe_job_id(job_id: str) -> str:
    try: return str(uuid.UUID(job_id))
    except ValueError: raise HTTPException(400, "job_id 형식이 올바르지 않습니다.")

def reject_if_active(job_id: str, message: str) -> None:
    """분석이 진행 중인 작업 위에서 후속 작업을 시작하지 못하게 막습니다."""
    if jobs.get(job_id, {}).get("status") in _ACTIVE_STATUSES:
        raise HTTPException(409, message)

_LOG_TRIM_MARKER = "… (앞부분은 크기 제한으로 잘렸습니다)\n"

def write_log(job_id: str, message: str):
    path = LOG_DIR / f"{job_id}.log"
    _trim_log(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

def _trim_log(path: Path) -> None:
    """상한을 넘으면 뒤쪽 절반만 남깁니다.

    진행 로그는 (청구항 × 문헌) 셀 수에 비례해 늘어나므로 상한이 없으면 한 작업이 디스크를
    계속 먹습니다. 파일을 나누지 않고 한 job = 한 파일을 유지하는 편이 로그 탭과 맞습니다.
    최근 기록이 진단에 쓰이므로 앞쪽을 버립니다.
    """
    try:
        if path.stat().st_size <= LOG_MAX_BYTES:
            return
        tail = path.read_text(encoding="utf-8", errors="replace")[-(LOG_MAX_BYTES // 2):]
        path.write_text(_LOG_TRIM_MARKER + tail.split("\n", 1)[-1], encoding="utf-8")
    except OSError:
        pass  # 로그 정리 실패로 분석을 멈추지 않습니다.

def write_json(path: Path, payload) -> None:
    """같은 디렉터리의 임시 파일에 쓴 뒤 원자적으로 바꿔 답니다.

    write_text는 대상 파일을 먼저 비우고 씁니다. 그 사이에 취소·크래시가 끼면 잘린 JSON이
    남고, meta.json이 그렇게 되면 히스토리 목록 전체를 읽을 수 없게 됩니다.
    """
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def read_json(path: Path):
    """깨진 기록 하나가 목록 전체를 막지 않도록, 읽지 못하면 None을 돌려줍니다."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

_PUBLIC_JOB_FIELDS = ("job_id", "status", "stage", "created_at", "error", "progress")

def public_job(record: dict) -> dict:
    """화면과 디스크에 내보내는 필드만 남깁니다. 문헌·결과·내부 핸들은 뺍니다."""
    return {key: record[key] for key in _PUBLIC_JOB_FIELDS if key in record}

def save_job_state(job_id: str) -> None:
    """작업 상태를 디스크에도 남깁니다.

    상태가 메모리에만 있으면 서버가 재시작되는 순간(reload, 포트 정리, 크래시) 폴링 중인
    화면이 404를 받아 "작업을 찾을 수 없습니다"만 보게 됩니다. 분석이 왜 멈췄는지, 다시
    돌려도 되는지 알 수 없는 것이 문제입니다. 상태 전이에서만 쓰므로 I/O는 몇 번뿐입니다.
    """
    record = jobs.get(job_id)
    if record is None:
        return
    try:
        write_json(JOBS_DIR / f"{job_id}.json", public_job(record))
    except OSError:
        pass  # 상태 기록 실패로 분석을 멈추지 않습니다.

def set_job_status(job_id: str, **fields) -> None:
    """상태를 바꾸고 곧바로 디스크에 남깁니다. 상태 전이는 반드시 이 함수를 거칩니다."""
    record = jobs.get(job_id)
    if record is None:
        return
    record.update(**fields)
    save_job_state(job_id)

def record_progress(job_id: str, stage: str, done: int | None, total: int | None) -> None:
    """진행 단계와, 있으면 셀 진행 수를 레코드에 남깁니다.

    done/total은 화면이 진행 바를 그리는 데 씁니다. 상태 전이가 아니므로 디스크에는 쓰지
    않습니다 — 셀마다 파일을 쓰면 I/O가 판정 수만큼 늘고, 재시작 뒤에 남은 진행률은 어차피
    의미가 없습니다(그 작업은 중단된 것입니다).
    """
    record = jobs.get(job_id)
    if record is None:
        return
    record["stage"] = stage
    if total:
        record["progress"] = {"done": done, "total": total}


def drop_job_state(job_id: str) -> None:
    (JOBS_DIR / f"{job_id}.json").unlink(missing_ok=True)

def recover_interrupted_jobs() -> int:
    """서버가 내려갈 때 돌고 있던 작업을 '중단됨'으로 표시해 되살립니다.

    다시 실행하라고 안내할 수 있어야 합니다. 이미 끝난 (청구항 × 문헌) 판정은 캐시에
    남아 있으므로, 같은 입력으로 다시 돌리면 남은 셀만 새로 대비합니다.
    """
    recovered = 0
    for path in sorted(JOBS_DIR.glob("*.json")):
        record = read_json(path)
        if not isinstance(record, dict) or not record.get("job_id"):
            path.unlink(missing_ok=True)
            continue
        if record.get("status") in _ACTIVE_STATUSES:
            record.update(
                status="interrupted", stage="중단됨",
                error="분석 도중 서버가 재시작되어 중단되었습니다. 같은 청구항과 문헌으로 다시 "
                      "실행하면 이미 끝난 판정은 캐시에서 재사용됩니다.",
            )
            recovered += 1
        jobs.setdefault(record["job_id"], dict(record))
        write_json(path, record)
    return recovered

def release_job_memory(job_id: str) -> None:
    """끝난 작업이 들고 있던 무거운 값을 놓아 줍니다.

    documents에는 업로드한 PDF에서 뽑은 본문이 통째로 들어 있어, 완료된 작업이 이것을
    계속 붙들고 있으면 서버를 켜 둔 만큼 메모리가 늘기만 합니다. 결과·문헌·청구항 원문은
    모두 히스토리에 저장되어 있고 _load_result / _load_analysis_context / get_result가
    없으면 디스크에서 되살리므로, 메모리 사본은 캐시로만 둡니다.
    """
    record = jobs.get(job_id)
    if record is None:
        return
    # _worker는 남깁니다. 끝난 Thread 객체 자체는 작고(run()이 인자 참조를 스스로 끊습니다)
    # 취소를 누른 쪽이 아직 join으로 종료를 기다리고 있을 수 있습니다.
    for key in ("documents", "result", "claims", "claims_text"):
        record.pop(key, None)

def sweep_job_records() -> None:
    """버려진 작업 레코드를 걷어냅니다. 새 작업을 준비할 때마다 한 번씩 돕니다.

    prepare만 하고 start를 하지 않으면(브라우저 탭을 닫으면) 그 레코드와 취소 토큰이
    영구히 남습니다. 진행 중인 작업과 업로드를 받고 있는 작업은 건드리지 않습니다.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=JOB_RECORD_TTL_MINUTES)
    for job_id, record in list(jobs.items()):
        if record.get("status") in _UNSWEEPABLE_STATUSES or record.get("_staging"):
            continue
        try:
            created = datetime.fromisoformat(str(record.get("created_at", "")))
        except ValueError:
            created = cutoff                  # 형식이 깨진 레코드는 곧바로 정리 대상입니다.
        if created <= cutoff:
            jobs.pop(job_id, None)
            agy.finish_job(job_id)
            drop_job_state(job_id)

@app.on_event("startup")
def _on_startup() -> None:
    recovered = recover_interrupted_jobs()
    if recovered:
        for job_id, record in jobs.items():
            if record.get("status") == "interrupted":
                write_log(job_id, "job marked interrupted after server restart")

@app.get("/api/health")
def health(): return {"status": "ok", "llm": "agy-cli", "model": AGY_MODEL}

@app.post("/api/jobs/prepare", status_code=201)
def prepare_job():
    """Reserve a cancellable job id before the browser begins its upload."""
    sweep_job_records()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "preparing",
        "stage": "파일 준비 중",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    agy.register_job(job_id)
    save_job_state(job_id)
    write_log(job_id, "job prepared")
    return {"job_id": job_id, "status": "preparing"}


@app.post("/api/jobs/{job_id}/start", status_code=202)
async def start_job(job_id: str, claims: str = Form(...),
                    pdf_files: list[UploadFile] = File(...),
                    analysis_prompt: str = Form(""),
                    priority_date: str = Form("")):
    """Stage uploads, then run the report in a cancellable worker thread."""
    job_id = safe_job_id(job_id)
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    if record["status"] in {"cancelling", "cancelled"} or agy.is_cancelled(job_id):
        raise HTTPException(409, "이미 취소된 작업입니다.")
    if record["status"] != "preparing":
        raise HTTPException(409, "이미 시작된 작업입니다.")
    if not claims.strip():
        raise HTTPException(400, "청구항을 입력하세요.")
    if not 1 <= len(pdf_files) <= 7:
        raise HTTPException(400, "PDF는 1~7개만 업로드할 수 있습니다.")

    work = Path(tempfile.mkdtemp(prefix=f"patent-{job_id}-"))
    documents: list[Document] = []
    total = 0
    started = False
    record["_staging"] = True
    record.update(status="preparing", stage="문서 준비 중", claims=claims, documents=documents)
    try:
        for index, upload in enumerate(pdf_files, 1):
            agy.raise_if_cancelled(job_id)
            filename = Path(upload.filename or f"document-{index}.pdf").name
            if not filename.lower().endswith(".pdf"):
                raise HTTPException(400, "PDF 파일만 업로드할 수 있습니다.")
            target = work / f"{index}.pdf"
            content = await upload.read()
            total += len(content)
            if len(content) > MAX_PDF_SIZE_MB * 1024 * 1024:
                raise HTTPException(413, "개별 PDF 크기 제한을 초과했습니다.")
            if total > MAX_TOTAL_UPLOAD_SIZE_MB * 1024 * 1024:
                raise HTTPException(413, "전체 업로드 크기 제한을 초과했습니다.")
            target.write_bytes(content)
            agy.raise_if_cancelled(job_id)
            document = extract_pdf(target, str(index))
            document.filename = filename
            document.source_file = f"sources/{index}-{filename}"
            documents.append(document)

        effective_prompt = analysis_prompt.strip() or load_runtime_settings().get("prompt") or ""
        set_job_status(job_id, status="running", stage="구성대비 준비 중")
        worker = threading.Thread(
            target=_run_async_analysis,
            args=(job_id, claims, documents, effective_prompt, work, priority_date.strip()),
            name=f"forge-{job_id[:8]}", daemon=True,
        )
        record["_worker"] = worker
        worker.start()
        started = True
        return {"job_id": job_id, "status": "running"}
    except agy.AnalysisCancelled:
        set_job_status(job_id, status="cancelled", stage="취소됨")
        write_log(job_id, "job cancelled during upload")
        agy.finish_job(job_id)
        raise HTTPException(409, "보고서 생성을 취소했습니다.")
    except HTTPException as exc:
        set_job_status(job_id, status="failed", stage="실패", error=str(exc.detail))
        write_log(job_id, f"job staging failed: {exc.status_code}: {exc.detail}")
        agy.finish_job(job_id)
        raise
    except Exception as exc:
        set_job_status(job_id, status="failed", stage="실패", error=str(exc))
        write_log(job_id, f"job staging failed: {type(exc).__name__}: {exc}")
        agy.finish_job(job_id)
        raise HTTPException(500, str(exc)) from exc
    finally:
        record.pop("_staging", None)
        if not started:
            shutil.rmtree(work, ignore_errors=True)


def _run_async_analysis(job_id: str, claims: str, documents: list[Document],
                        analysis_prompt: str, work: Path, priority_date: str = "") -> None:
    agy.bind_job(job_id)
    try:
        agy.raise_if_cancelled(job_id)

        def progress(stage: str, done: int | None = None, total: int | None = None):
            agy.raise_if_cancelled(job_id)
            record_progress(job_id, stage, done, total)
            write_log(job_id, stage)

        decomposition: dict = {}
        cache_keys: set[str] = set()
        result = analyze(job_id, claims, documents, analysis_prompt, progress, decomposition,
                         cache_keys, priority_date)
        agy.raise_if_cancelled(job_id)
        _persist_initial_analysis(job_id, result, claims, documents, analysis_prompt, work,
                                  decomposition, cache_keys, priority_date)
        # A cancellation that lands during persistence must not leave a searchable report.
        if agy.is_cancelled(job_id):
            shutil.rmtree(HISTORY_DIR / job_id, ignore_errors=True)
            raise agy.AnalysisCancelled("보고서 생성을 취소했습니다.")
        # 결과를 레코드에 담아 두지 않습니다. _persist_initial_analysis가 이미 히스토리에
        # 썼고, get_result / _load_result가 없으면 거기서 읽습니다.
        if job_id in jobs:
            set_job_status(job_id, status="completed", stage="완료")
        write_log(job_id, "analysis completed")
    except agy.AnalysisCancelled:
        shutil.rmtree(HISTORY_DIR / job_id, ignore_errors=True)
        if job_id in jobs:
            set_job_status(job_id, status="cancelled", stage="취소됨")
            jobs[job_id].pop("result", None)
        write_log(job_id, "analysis cancelled; process tree killed")
    except Exception as exc:
        if job_id in jobs:
            set_job_status(job_id, status="failed", stage="실패", error=str(exc))
        write_log(job_id, f"analysis failed: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        agy.finish_job(job_id)
        release_job_memory(job_id)


def _persist_initial_analysis(job_id: str, result: AnalysisResult, claims: str,
                              documents: list[Document], analysis_prompt: str,
                              work: Path, decomposition: dict | None = None,
                              cache_keys: set[str] | None = None,
                              priority_date: str = "") -> None:
    """최초 분석에만 필요한 것(원문 PDF 보존·구성분해)을 남기고 나머지는 공통 경로에 맡깁니다."""
    history = HISTORY_DIR / job_id
    history.mkdir(parents=True, exist_ok=True)
    _save_decomposition(job_id, decomposition)
    _save_cache_keys(job_id, cache_keys)
    (history / "sources").mkdir(exist_ok=True)
    for index, document in enumerate(documents, 1):
        shutil.copy2(work / f"{index}.pdf", history / document.source_file)
    _persist_analysis(job_id, result, claims, documents, analysis_prompt,
                      created_at=datetime.now(timezone.utc).isoformat(),
                      priority_date=priority_date)


@app.delete("/api/jobs/{job_id}")
def cancel_running_job(job_id: str):
    job_id = safe_job_id(job_id)
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    if record["status"] in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": record["status"], "kill_requested": False}
    set_job_status(job_id, status="cancelling", stage="취소 중")
    killed = agy.cancel_job(job_id)
    write_log(job_id, "cancellation requested")
    worker = record.get("_worker")
    if (not worker or not worker.is_alive()) and not record.get("_staging"):
        set_job_status(job_id, status="cancelled", stage="취소됨")
        agy.finish_job(job_id)
    return {"job_id": job_id, "status": record["status"], "kill_requested": killed}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """진행 상태. 메모리에 없으면 디스크에 남긴 상태를 돌려줍니다.

    서버가 재시작되면 메모리 레코드는 사라집니다. 그때 404를 주면 폴링 중인 화면에는
    "작업을 찾을 수 없습니다"만 남아, 분석이 왜 멈췄는지도 다시 돌려도 되는지도 알 수
    없습니다. 디스크 상태에는 interrupted와 그 사유가 들어 있습니다.
    """
    job_id = safe_job_id(job_id)
    record = jobs.get(job_id)
    if record is not None:
        return public_job(record)
    stored = read_json(JOBS_DIR / f"{job_id}.json")
    if not isinstance(stored, dict): raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return stored

@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    """완료된 결과. 메모리 사본이 없으면 히스토리에서 읽습니다.

    끝난 작업은 결과를 메모리에 붙들지 않으므로(release_job_memory), 여기가 히스토리를
    되짚어야 서버를 재시작한 뒤에도 같은 job_id로 결과를 받을 수 있습니다.
    """
    job_id = safe_job_id(job_id)
    stored = jobs.get(job_id, {}).get("result")
    if stored is None:
        stored = read_json(HISTORY_DIR / job_id / "result.json")
    if stored is None:
        raise HTTPException(404, "결과를 찾을 수 없습니다.")
    return stored


@app.post("/api/jobs/{job_id}/dependent-claims", status_code=202)
def add_dependent_claims(job_id: str, payload: DependentClaimsAdd):
    """완료된 보고서에 복수 종속항을 추가합니다. 대비는 취소 가능한 워커에서 돕니다."""
    job_id = safe_job_id(job_id)
    loaded = _load_analysis_context(job_id)
    if loaded is None:
        raise HTTPException(404, "기존 분석 결과 또는 인용발명 캐시를 찾을 수 없습니다.")
    result, existing_text, documents, analysis_prompt = loaded
    incoming = parse_claims(payload.claims)
    if not incoming:
        raise HTTPException(400, "추가할 종속항을 인식하지 못했습니다.")
    existing_claims = parse_claims(existing_text)
    existing_numbers = {claim.number for claim in existing_claims}
    incoming_numbers = [claim.number for claim in incoming]
    if len(incoming_numbers) != len(set(incoming_numbers)) or existing_numbers.intersection(incoming_numbers):
        raise HTTPException(409, "이미 존재하거나 중복된 청구항 번호가 포함되어 있습니다.")

    # 최초 분석이 헤더 없는 단일항 입력이었다면, 뒤에 번호 있는 종속항을 붙일 때
    # 파서가 헤더 앞의 청구항 1을 버리지 않도록 명시적 헤더를 보충합니다.
    existing_for_merge = existing_text
    if (len(existing_claims) == 1 and existing_claims[0].number == 1
            and existing_claims[0].raw.strip() == existing_text.strip()):
        existing_for_merge = f"【청구항 1】\n{existing_text.strip()}"
    combined_text = existing_for_merge.rstrip() + "\n" + payload.claims.strip()
    all_claims = parse_claims(combined_text)
    by_number = {claim.number: claim for claim in all_claims}
    if 1 not in by_number:
        raise HTTPException(409, "기존 보고서에 청구항 1이 없어 종속 관계를 연결할 수 없습니다.")
    invalid = [number for number in incoming_numbers
               if number not in by_number or 1 not in ancestry(all_claims, number)]
    if invalid:
        joined = ", ".join(str(number) for number in invalid)
        raise HTTPException(400, f"청구항 1에 종속되지 않은 항이 있습니다: {joined}")

    reject_if_active(job_id, "이 작업은 아직 실행 중입니다.")
    record = jobs.setdefault(job_id, {"job_id": job_id,
                                      "created_at": datetime.now(timezone.utc).isoformat()})

    # 종속항 대비도 초기 분석과 같은 길이의 작업입니다. 요청 스레드에서 끝까지 돌리면
    # 진행률도 보이지 않고 취소도 닿지 않으며, 중간에 멈추면 그때까지의 판정이 통째로
    # 사라집니다. 초기 분석과 같은 워커 + 폴링 구조로 맞춥니다.
    agy.register_job(job_id)
    set_job_status(job_id, status="running", stage="종속항 구성대비 준비 중")
    write_log(job_id, f"dependent claims batch started: {incoming_numbers}")
    worker = threading.Thread(
        target=_run_dependent_claims,
        args=(job_id, result, combined_text, documents, analysis_prompt, incoming_numbers),
        name=f"forge-dep-{job_id[:8]}", daemon=True,
    )
    record["_worker"] = worker
    worker.start()
    return {"job_id": job_id, "status": "running", "added_claims": incoming_numbers}


def _run_dependent_claims(job_id: str, result: AnalysisResult, combined_text: str,
                          documents: list[Document], analysis_prompt: str,
                          numbers: list[int]) -> None:
    """종속항 대비를 취소 가능한 워커에서 돌리고, 확정된 항은 그때그때 저장합니다."""
    agy.bind_job(job_id)
    # 저장된 분해를 먼저 읽습니다. 같은 분해를 다시 쓰면 취소 후 재시도에서 이미 끝난
    # 셀의 판정 캐시가 그대로 맞습니다(cache.cache_key가 분해 결과를 키에 포함합니다).
    decomposition = _load_decomposition(job_id)
    cache_keys: set[str] = set()
    try:
        def progress(stage: str, done: int | None = None, total: int | None = None):
            agy.raise_if_cancelled(job_id)
            record_progress(job_id, stage, done, total)
            write_log(job_id, stage)

        def checkpoint(partial: AnalysisResult):
            # 히스토리에만 씁니다. 메모리 사본을 함께 두면 끝난 작업이 문헌 본문을 계속
            # 붙들게 되고, 결과를 읽는 쪽은 이미 전부 히스토리 폴백을 갖고 있습니다.
            claims_text = _claims_text_for(combined_text, partial.reports)
            _persist_analysis(job_id, partial, claims_text, documents, analysis_prompt)
            added = sorted({report.claim_number for report in partial.reports} & set(numbers))
            write_log(job_id, f"dependent claims saved: {added}")

        extend_with_dependent_claims(result, combined_text, set(numbers), documents,
                                     analysis_prompt, progress, decomposition, checkpoint,
                                     cache_keys)
        if job_id in jobs:
            set_job_status(job_id, status="completed", stage="완료")
        write_log(job_id, f"dependent claims batch completed: {numbers}")
    except agy.AnalysisCancelled:
        # checkpoint가 이미 확정된 항까지 저장했습니다. 여기서 히스토리를 지우면
        # 그 판정이 사라져 다시 눌렀을 때 같은 항을 또 대비하게 됩니다.
        if job_id in jobs:
            set_job_status(job_id, status="cancelled", stage="취소됨")
        write_log(job_id, "dependent claims cancelled; completed claims kept")
    except Exception as exc:
        if job_id in jobs:
            set_job_status(job_id, status="failed", stage="실패", error=str(exc))
        write_log(job_id, f"dependent claims batch failed: {type(exc).__name__}: {exc}")
    finally:
        # 분해는 판정보다 먼저 끝나므로, 취소·실패로 끝나도 남겨 둡니다. 다음 실행이 같은
        # 분해를 재사용해야 이미 받아 둔 셀 판정이 캐시에서 그대로 살아납니다.
        _save_decomposition(job_id, decomposition)
        # 취소·실패로 끝나도 남깁니다. 이미 받아 둔 판정도 삭제 대상에 들어가야 '삭제'가
        # 실제 삭제가 됩니다.
        _save_cache_keys(job_id, cache_keys)
        agy.finish_job(job_id)
        release_job_memory(job_id)


def _claims_text_for(combined_text: str, reports: list) -> str:
    """보고서에 실제로 올라간 항만 남긴 청구항 원문.

    취소로 일부만 추가된 경우에도 전체 입력을 메타데이터에 적으면, 남은 항을 다시
    추가하려 할 때 "이미 존재하는 청구항 번호"로 거부됩니다.
    """
    numbers = {report.claim_number for report in reports}
    kept = [claim for claim in parse_claims(combined_text) if claim.number in numbers]
    return "\n".join(f"【청구항 {claim.number}】\n{claim.raw.strip()}" for claim in kept)


def _load_decomposition(job_id: str) -> dict:
    value = read_json(HISTORY_DIR / job_id / "claim_elements.json")
    return value if isinstance(value, dict) else {}


def _save_cache_keys(job_id: str, cache_keys: set[str] | None) -> None:
    """이 분석이 사용한 판정 캐시 키. 삭제할 때 그 판정까지 지우기 위한 목록입니다."""
    if not cache_keys:
        return
    history = HISTORY_DIR / job_id
    history.mkdir(parents=True, exist_ok=True)
    stored = read_json(history / "cache_keys.json")
    merged = sorted(set(stored if isinstance(stored, list) else []) | set(cache_keys))
    try:
        write_json(history / "cache_keys.json", merged)
    except OSError:
        pass  # 목록 저장 실패로 분석을 멈추지 않습니다.


def _save_decomposition(job_id: str, decomposition: dict | None) -> None:
    if not decomposition:
        return
    history = HISTORY_DIR / job_id
    history.mkdir(parents=True, exist_ok=True)
    try:
        write_json(history / "claim_elements.json", decomposition)
    except OSError:
        pass  # 저장 실패로 분석을 멈추지 않습니다. 다음 실행에서 다시 분해합니다.

_prior_art_lock = threading.Lock()
_prior_art_running: set[str] = set()

def _prior_art_token(job_id: str) -> str:
    """선행기술 검색용 취소 토큰. 분석 job_id와 네임스페이스를 나눕니다.

    같은 키를 쓰면 검색을 취소했을 때 완료된 보고서의 취소 토큰까지 세워져, 이후
    같은 job에서 검색을 다시 누를 수 없게 됩니다. 검색은 완료된 보고서 위에서 도는
    별도 작업이므로 취소 범위도 검색으로만 한정합니다.
    """
    return f"{job_id}:prior-art"

@app.post("/api/jobs/{job_id}/prior-art")
def search_prior_art(job_id: str):
    """미커버로 남은 구성만 웹에서 검색합니다. 외부 호출이므로 자동 실행하지 않습니다."""
    job_id = safe_job_id(job_id)
    # 종속항 추가와 **같은 게이트**를 봅니다. 검색 가능 여부를 가르는 것은 상태 문자열이
    # 아니라 저장된 보고서의 유무이므로, 그 판단은 아래 _load_result에 맡깁니다.
    reject_if_active(job_id, "분석이 진행 중인 보고서에서는 선행기술 검색을 실행할 수 없습니다.")
    stored = _load_result(job_id)
    if stored is None: raise HTTPException(404, "결과를 찾을 수 없습니다.")
    result, claims_text = stored
    targets = uncovered_elements(result, claims_text)
    if not targets: return {"hits": [], "message": "결합 후 미커버로 남은 구성이 없습니다."}

    token = _prior_art_token(job_id)
    with _prior_art_lock:
        if token in _prior_art_running:
            raise HTTPException(409, "이미 선행기술을 검색하고 있습니다.")
        _prior_art_running.add(token)
    agy.register_job(token)
    # run_cli는 띄운 CLI 프로세스를 current_job()에 달아 둡니다. 이 스레드를 묶어 두지
    # 않으면 프로세스가 어디에도 등록되지 않아, 취소를 눌러도 죽일 대상을 찾지 못하고
    # 검색이 타임아웃까지 계속 돕니다.
    agy.bind_job(token)
    try:
        hits = priorart.search(targets)
        # 검색은 웹으로 나가고 결과는 모델이 적어 준 문헌번호·URL입니다. 파이프라인의 다른
        # 산출물과 같은 기준으로, 코드가 URL을 열어 문헌번호를 대조합니다.
        agy.raise_if_cancelled(token)
        priorart.verify_hits(hits)
    except agy.AnalysisCancelled:
        write_log(job_id, "prior art search cancelled")
        raise HTTPException(409, "선행기술 검색을 취소했습니다.")
    except priorart.SearchFailed as exc:
        # 취소와 마찬가지로 보고서를 건드리지 않고 끝냅니다. CLI가 답을 못 낸 것은
        # "0건"이 아니므로, 저장했다면 지난 검색 결과만 지우는 꼴이 됩니다.
        write_log(job_id, f"prior art search failed: {exc}")
        raise HTTPException(502, str(exc))
    finally:
        agy.finish_job(token)
        with _prior_art_lock:
            _prior_art_running.discard(token)

    result.prior_art = hits
    history = HISTORY_DIR / job_id
    if history.exists():
        write_json(history / "result.json", result.model_dump())
        markdown = to_markdown(result)
        (history / "report.md").write_text(markdown, encoding="utf-8")
        (history / "report.txt").write_text(markdown, encoding="utf-8")
    # 메모리 사본이 남아 있으면 함께 갱신합니다. 없으면 그대로 둡니다 — get_result가
    # 히스토리에서 방금 쓴 result.json을 읽습니다.
    if "result" in jobs.get(job_id, {}): jobs[job_id]["result"] = result.model_dump()
    confirmed = sum(1 for hit in hits if hit.verify == "verified")
    write_log(job_id, f"prior art search: {len(hits)} hits ({confirmed} url-verified) "
                      f"for {len(targets)} uncovered elements")
    return {"hits": [hit.model_dump() for hit in hits], "searched": targets,
            "verified": confirmed}

@app.delete("/api/jobs/{job_id}/prior-art")
def cancel_prior_art(job_id: str):
    """진행 중인 선행기술 검색만 멈춥니다.

    보고서 상태는 completed 그대로 둡니다. 검색을 취소했다고 보고서까지 취소 처리하면
    다시 검색할 수도, 내려받을 수도 없게 됩니다.
    """
    job_id = safe_job_id(job_id)
    token = _prior_art_token(job_id)
    with _prior_art_lock:
        running = token in _prior_art_running
    if not running:
        return {"job_id": job_id, "running": False, "kill_requested": False}
    killed = agy.cancel_job(token)
    write_log(job_id, "prior art cancellation requested")
    return {"job_id": job_id, "running": True, "kill_requested": killed}

def _load_result(job_id: str) -> tuple[AnalysisResult, str] | None:
    """메모리에 없으면 히스토리에서 결과와 청구항 원문을 되살립니다."""
    record = jobs.get(job_id, {})
    raw, claims_text = record.get("result"), record.get("claims_text", "")
    if raw is None:
        raw = read_json(HISTORY_DIR / job_id / "result.json")
        if raw is None: return None
    if not claims_text:
        meta = read_json(HISTORY_DIR / job_id / "meta.json")
        claims_text = meta.get("claims", "") if isinstance(meta, dict) else ""
    if not claims_text: return None
    try:
        return AnalysisResult.model_validate(raw), claims_text
    except ValueError:
        return None


def _load_analysis_context(job_id: str) -> tuple[AnalysisResult, str, list[Document], str] | None:
    """후속 종속항 분석에 필요한 결과·청구항·추출 문헌·판단 지침을 복원합니다."""
    stored = _load_result(job_id)
    if stored is None:
        return None
    result, claims_text = stored
    record = jobs.get(job_id, {})
    documents = record.get("documents") or []
    history = HISTORY_DIR / job_id
    if not documents:
        raw = read_json(history / "documents.json")
        if not isinstance(raw, list):
            return None
        try:
            documents = [Document.model_validate(item) for item in raw]
        except ValueError:
            return None
    meta = read_json(history / "meta.json")
    prompt = str(meta.get("analysis_prompt") or "") if isinstance(meta, dict) else ""
    return result, claims_text, documents, prompt


def _persist_analysis(job_id: str, result: AnalysisResult, claims_text: str,
                      documents: list[Document], analysis_prompt: str,
                      created_at: str | None = None, priority_date: str | None = None) -> None:
    """결과·리포트·감사 데이터와 원 입력 메타데이터를 히스토리에 한 벌로 남깁니다.

    최초 분석과 종속항 추가가 **같은 경로**를 씁니다. 종전에는 같은 6개 파일을 두 함수가
    따로 썼고, 그 사이에서 이미 mkdir 인자가 갈라져 있었습니다.
    """
    history = HISTORY_DIR / job_id
    history.mkdir(parents=True, exist_ok=True)
    write_json(history / "result.json", result.model_dump())
    markdown = to_markdown(result)
    (history / "report.md").write_text(markdown, encoding="utf-8")
    (history / "report.txt").write_text(markdown, encoding="utf-8")
    write_json(history / "judgment.json", summarize_matrix(result))
    write_json(history / "documents.json", [document.model_dump() for document in documents])
    meta = read_json(history / "meta.json")
    meta = meta if isinstance(meta, dict) else {}
    meta.update(
        job_id=job_id,
        # 종속항을 덧붙여도 최초 분석 시각을 유지합니다. 히스토리 정렬 기준입니다.
        created_at=created_at or meta.get("created_at") or datetime.now(timezone.utc).isoformat(),
        # 빈 값으로는 덮어쓰지 않습니다. _claims_text_for는 보고서에 오른 항만 남기는데,
        # 확정된 항이 하나도 없는 시점에 체크포인트가 걸리면 빈 문자열이 됩니다. 그것을
        # 그대로 쓰면 청구항 원문이 지워지고, result.json이 멀쩡해도 그 보고서로는
        # 종속항 추가도 선행기술 검색도 다시 할 수 없게 됩니다.
        claims=claims_text or meta.get("claims", ""),
        claims_summary=(claims_text or meta.get("claims", ""))[:200],
        analysis_prompt=analysis_prompt,
        # 종속항 추가·재실행에서도 같은 기준으로 적격성을 가려야 하므로 남깁니다. 빈 값으로는
        # 덮어쓰지 않습니다(체크포인트가 우선일을 지워 버리면 후속 실행이 기준을 잃습니다).
        priority_date=(priority_date if priority_date is not None
                       else meta.get("priority_date", "")) or meta.get("priority_date", ""),
        documents=[document.filename for document in documents],
    )
    write_json(history / "meta.json", meta)

@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, format: str = "md"):
    if format not in {"md", "txt"}: raise HTTPException(400, "format은 md 또는 txt만 지원합니다.")
    job_id = safe_job_id(job_id)
    path = HISTORY_DIR / job_id / f"report.{format}"
    if not path.exists(): raise HTTPException(404, "리포트를 찾을 수 없습니다.")
    return PlainTextResponse(path.read_text(encoding="utf-8"), headers={"Content-Disposition": f'attachment; filename="{job_id}.{format}"'})


@app.get("/api/jobs/{job_id}/sources/{document_id}")
def download_source(job_id: str, document_id: str):
    """히스토리에 보존한 분석 당시 원문 PDF를 문헌 내부 ID로 돌려줍니다."""
    job_id = safe_job_id(job_id)
    context = _load_analysis_context(job_id)
    if context is None:
        raise HTTPException(404, "분석 기록을 찾을 수 없습니다.")
    document = next((item for item in context[2] if item.id == document_id), None)
    if document is None or not document.source_file:
        raise HTTPException(404, "보존된 원문 PDF를 찾을 수 없습니다.")
    history = (HISTORY_DIR / job_id).resolve()
    source = (history / document.source_file).resolve()
    source_root = (history / "sources").resolve()
    if source_root not in source.parents or not source.is_file():
        raise HTTPException(404, "보존된 원문 PDF를 찾을 수 없습니다.")
    return FileResponse(source, media_type="application/pdf", filename=document.filename)

@app.get("/api/history")
def history():
    """저장된 분석 목록.

    읽지 못하는 meta.json은 건너뜁니다. 한 건이 깨졌다고 목록 전체가 500이 되면 멀쩡한
    나머지 보고서까지 화면에서 사라지고, 사용자에게는 히스토리가 통째로 날아간 것처럼
    보입니다. created_at도 없을 수 있다고 보고 정렬합니다.
    """
    items = []
    for path in sorted(HISTORY_DIR.iterdir()):
        if not path.is_dir():
            continue
        meta = read_json(path / "meta.json")
        if isinstance(meta, dict):
            items.append({**meta, "job_id": meta.get("job_id") or path.name})
    return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

@app.get("/api/history/{job_id}")
def history_detail(job_id: str):
    job_id = safe_job_id(job_id)
    meta = read_json(HISTORY_DIR / job_id / "meta.json")
    if not isinstance(meta, dict): raise HTTPException(404, "히스토리를 찾을 수 없습니다.")
    result = jobs.get(job_id, {}).get("result")
    if result is None:
        result = read_json(HISTORY_DIR / job_id / "result.json")
    return {"meta": meta, "result": result}

def remove_job_record(job_id: str) -> int:
    """분석을 **실제로** 지웁니다. 로그는 로그 탭에서 별도로 관리합니다.

    지워야 할 것이 히스토리 폴더만이 아닙니다.
      - backend/data/history의 레거시 사본: 남겨 두면 _migrate_legacy_storage가 다음 기동에
        그대로 되살립니다. 사용자가 지웠는데 재시작하면 돌아오는 상태였습니다.
      - 판정 캐시: 그 안에 문헌 **원문 발췌**가 들어 있습니다. 히스토리만 지우면 사용자가
        지웠다고 생각한 문장이 디스크에 그대로 남습니다.
      - 작업 상태 파일.
    """
    purged = cache.discard(read_json(HISTORY_DIR / job_id / "cache_keys.json") or [])
    shutil.rmtree(HISTORY_DIR / job_id, ignore_errors=True)
    shutil.rmtree(DATA_DIR / "history" / job_id, ignore_errors=True)
    drop_job_state(job_id)
    jobs.pop(job_id, None)
    return purged

@app.delete("/api/history")
def clear_history():
    removed = {path.name for path in HISTORY_DIR.iterdir() if path.is_dir()}
    legacy = DATA_DIR / "history"
    if legacy.is_dir():
        removed |= {path.name for path in legacy.iterdir() if path.is_dir()}
    purged = sum(remove_job_record(job_id) for job_id in removed)
    return {"ok": True, "removed": len(removed), "cache_purged": purged}

@app.delete("/api/history/{job_id}")
def delete_history(job_id: str):
    job_id = safe_job_id(job_id)
    return {"ok": True, "cache_purged": remove_job_record(job_id)}

@app.delete("/api/cache")
def clear_cache():
    """판정 캐시를 비웁니다. 프롬프트나 모델을 바꾼 뒤 전부 다시 판정하고 싶을 때 씁니다.

    캐시에는 업로드한 문헌의 원문 발췌가 남으므로, 저장물을 완전히 비우고 싶을 때도 씁니다.
    """
    return {"ok": True, "removed": cache.clear()}

@app.get("/api/logs")
def logs():
    items = []
    for path in LOG_DIR.glob("*.log"):
        stat = path.stat()
        items.append({
            "job_id": path.stem,
            "size": stat.st_size,
            "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    return sorted(items, key=lambda item: item["updated_at"], reverse=True)

@app.get("/api/logs/{job_id}")
def get_logs(job_id: str):
    job_id = safe_job_id(job_id)
    path = LOG_DIR / f"{job_id}.log"
    if not path.exists(): raise HTTPException(404, "로그를 찾을 수 없습니다.")
    return {"job_id": job_id, "content": path.read_text(encoding="utf-8")}

@app.delete("/api/logs")
def clear_logs():
    """로그를 지웁니다. 레거시 사본까지 지워야 재시작 때 되살아나지 않습니다."""
    paths = list(LOG_DIR.glob("*.log"))
    for path in paths:
        path.unlink(missing_ok=True)
    legacy = DATA_DIR / "logs"
    if legacy.is_dir():
        for path in legacy.glob("*.log"):
            path.unlink(missing_ok=True)
    return {"ok": True, "removed": len(paths)}

@app.get("/api/settings")
def get_settings(): return load_runtime_settings()

@app.get("/api/settings/models")
def settings_models(provider: str | None = None, refresh: bool = False):
    settings = load_runtime_settings()
    provider = (provider or settings["provider"]).lower()
    if provider not in {"agy", "claude", "gpt"}: raise HTTPException(400, "지원하지 않는 provider입니다.")
    return {"provider": provider, "models": available_models({"provider": provider}, refresh)}

@app.put("/api/settings")
def update_settings(payload: dict):
    provider = str(payload.get("provider", "agy")).lower()
    if provider not in {"agy", "claude", "gpt"}: raise HTTPException(400, "지원하지 않는 provider입니다.")
    model = str(payload.get("model", AGY_MODEL)).strip()
    if not model: raise HTTPException(400, "모델을 입력해 주세요.")
    prompt = str(payload.get("prompt", "")).strip()
    return save_runtime_settings({"provider": provider, "model": model, "prompt": prompt})

@app.post("/api/settings/test")
def test_settings(payload: dict):
    provider = str(payload.get("provider", "agy")).lower()
    try:
        _build_command("연결 테스트", {"provider": provider, "model": str(payload.get("model", AGY_MODEL))})
        return {"ok": True, "message": f"{provider} CLI를 찾았습니다."}
    except Exception as exc: return {"ok": False, "message": str(exc)}
