import json
import shutil
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from .config import HISTORY_DIR, LOG_DIR, MAX_PDF_SIZE_MB, MAX_TOTAL_UPLOAD_SIZE_MB, AGY_MODEL, load_runtime_settings, save_runtime_settings
from . import agy, cache, priorart
from .agy import _build_command, available_models
from .claims import ancestry, parse_claims
from .models import AnalysisResult, DependentClaimsAdd, Document
from .pdf import extract_pdf
from .pipeline import analyze, extend_with_dependent_claims, summarize_matrix, to_markdown, uncovered_elements

app = FastAPI(title="Patent Evidence Analyzer")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5374"], allow_methods=["*"], allow_headers=["*"])
jobs: dict[str, dict] = {}

def safe_job_id(job_id: str) -> str:
    try: return str(uuid.UUID(job_id))
    except ValueError: raise HTTPException(400, "job_id 형식이 올바르지 않습니다.")

def write_log(job_id: str, message: str):
    with (LOG_DIR / f"{job_id}.log").open("a", encoding="utf-8") as file:
        file.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

@app.get("/api/health")
def health(): return {"status": "ok", "llm": "agy-cli", "model": AGY_MODEL}

@app.post("/api/jobs")
async def create_job(claims: str = Form(...), pdf_files: list[UploadFile] = File(...), analysis_prompt: str = Form("")):
    if not claims.strip(): raise HTTPException(400, "청구항을 입력하세요.")
    if not 1 <= len(pdf_files) <= 7: raise HTTPException(400, "PDF는 1~7개만 업로드할 수 있습니다.")
    job_id = str(uuid.uuid4()); work = Path(tempfile.mkdtemp(prefix=f"patent-{job_id}-"))
    documents = []; total = 0
    jobs[job_id] = {"job_id": job_id, "status": "running", "claims": claims, "documents": documents, "stage": "문서 추출 중"}
    write_log(job_id, "analysis started")
    try:
        for index, upload in enumerate(pdf_files, 1):
            filename = Path(upload.filename or f"document-{index}.pdf").name
            if not filename.lower().endswith(".pdf"): raise HTTPException(400, "PDF 파일만 업로드할 수 있습니다.")
            target = work / f"{index}.pdf"; content = await upload.read(); total += len(content)
            if len(content) > MAX_PDF_SIZE_MB * 1024 * 1024: raise HTTPException(413, "개별 PDF 크기 제한을 초과했습니다.")
            if total > MAX_TOTAL_UPLOAD_SIZE_MB * 1024 * 1024: raise HTTPException(413, "전체 업로드 크기 제한을 초과했습니다.")
            target.write_bytes(content)
            document = extract_pdf(target, str(index)); document.filename = filename
            document.source_file = f"sources/{index}-{filename}"
            documents.append(document)

        def progress(stage: str):
            jobs[job_id]["stage"] = stage
            write_log(job_id, stage)

        effective_prompt = analysis_prompt.strip() or load_runtime_settings().get("prompt") or ""
        result = analyze(job_id, claims, documents, effective_prompt, progress)
        jobs[job_id]["result"] = result.model_dump()
        jobs[job_id]["claims_text"] = claims
        history = HISTORY_DIR / job_id; history.mkdir(exist_ok=True)
        source_dir = history / "sources"; source_dir.mkdir(exist_ok=True)
        for index, document in enumerate(documents, 1):
            shutil.copy2(work / f"{index}.pdf", history / document.source_file)
        (history / "result.json").write_text(json.dumps(jobs[job_id]["result"], ensure_ascii=False), encoding="utf-8")
        (history / "report.md").write_text(to_markdown(result), encoding="utf-8")
        (history / "report.txt").write_text(to_markdown(result), encoding="utf-8")
        # 판정 추적 데이터. 별도 감사 리포트는 만들지 않고 이 JSON으로만 남깁니다.
        (history / "judgment.json").write_text(json.dumps(summarize_matrix(result), ensure_ascii=False), encoding="utf-8")
        # 후속 종속항은 PDF 재업로드·재추출 없이 최초 인용발명을 그대로 사용합니다.
        (history / "documents.json").write_text(
            json.dumps([document.model_dump() for document in documents], ensure_ascii=False), encoding="utf-8")
        (history / "meta.json").write_text(json.dumps({"job_id": job_id, "created_at": datetime.now(timezone.utc).isoformat(), "claims_summary": claims[:200], "claims": claims, "documents": [d.filename for d in documents],
                                                   "analysis_prompt": effective_prompt}, ensure_ascii=False), encoding="utf-8")
        jobs[job_id]["status"] = "completed"; jobs[job_id]["stage"] = "완료"
        write_log(job_id, "analysis completed")
        return {"job_id": job_id, "status": "completed"}
    except HTTPException as exc:
        jobs[job_id].update(status="failed", error=str(exc.detail))
        write_log(job_id, f"analysis failed: {exc.status_code}: {exc.detail}")
        raise
    except Exception as exc:
        jobs[job_id].update(status="failed", error=str(exc))
        write_log(job_id, f"analysis failed: {type(exc).__name__}: {exc}")
        # CLI 실패 원인이 500 본문에 묻히지 않도록 detail로 내려보낸다.
        raise HTTPException(502, str(exc)) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.post("/api/jobs/prepare", status_code=201)
def prepare_job():
    """Reserve a cancellable job id before the browser begins its upload."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "preparing",
        "stage": "파일 준비 중",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    agy.register_job(job_id)
    write_log(job_id, "job prepared")
    return {"job_id": job_id, "status": "preparing"}


@app.post("/api/jobs/{job_id}/start", status_code=202)
async def start_job(job_id: str, claims: str = Form(...),
                    pdf_files: list[UploadFile] = File(...),
                    analysis_prompt: str = Form("")):
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
        record.update(status="running", stage="구성대비 준비 중")
        worker = threading.Thread(
            target=_run_async_analysis,
            args=(job_id, claims, documents, effective_prompt, work),
            name=f"forge-{job_id[:8]}", daemon=True,
        )
        record["_worker"] = worker
        worker.start()
        started = True
        return {"job_id": job_id, "status": "running"}
    except agy.AnalysisCancelled:
        record.update(status="cancelled", stage="취소됨")
        write_log(job_id, "job cancelled during upload")
        agy.finish_job(job_id)
        raise HTTPException(409, "보고서 생성을 취소했습니다.")
    except HTTPException as exc:
        record.update(status="failed", stage="실패", error=str(exc.detail))
        write_log(job_id, f"job staging failed: {exc.status_code}: {exc.detail}")
        agy.finish_job(job_id)
        raise
    except Exception as exc:
        record.update(status="failed", stage="실패", error=str(exc))
        write_log(job_id, f"job staging failed: {type(exc).__name__}: {exc}")
        agy.finish_job(job_id)
        raise HTTPException(500, str(exc)) from exc
    finally:
        record.pop("_staging", None)
        if not started:
            shutil.rmtree(work, ignore_errors=True)


def _run_async_analysis(job_id: str, claims: str, documents: list[Document],
                        analysis_prompt: str, work: Path) -> None:
    agy.bind_job(job_id)
    try:
        agy.raise_if_cancelled(job_id)

        def progress(stage: str):
            agy.raise_if_cancelled(job_id)
            if job_id in jobs:
                jobs[job_id]["stage"] = stage
            write_log(job_id, stage)

        result = analyze(job_id, claims, documents, analysis_prompt, progress)
        agy.raise_if_cancelled(job_id)
        _persist_initial_analysis(job_id, result, claims, documents, analysis_prompt, work)
        # A cancellation that lands during persistence must not leave a searchable report.
        if agy.is_cancelled(job_id):
            shutil.rmtree(HISTORY_DIR / job_id, ignore_errors=True)
            raise agy.AnalysisCancelled("보고서 생성을 취소했습니다.")
        if job_id in jobs:
            jobs[job_id].update(
                result=result.model_dump(), claims_text=claims,
                status="completed", stage="완료",
            )
        write_log(job_id, "analysis completed")
    except agy.AnalysisCancelled:
        shutil.rmtree(HISTORY_DIR / job_id, ignore_errors=True)
        if job_id in jobs:
            jobs[job_id].update(status="cancelled", stage="취소됨")
            jobs[job_id].pop("result", None)
        write_log(job_id, "analysis cancelled; process tree killed")
    except Exception as exc:
        if job_id in jobs:
            jobs[job_id].update(status="failed", stage="실패", error=str(exc))
        write_log(job_id, f"analysis failed: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        agy.finish_job(job_id)


def _persist_initial_analysis(job_id: str, result: AnalysisResult, claims: str,
                              documents: list[Document], analysis_prompt: str,
                              work: Path) -> None:
    dumped = result.model_dump()
    history = HISTORY_DIR / job_id
    history.mkdir(exist_ok=True)
    source_dir = history / "sources"
    source_dir.mkdir(exist_ok=True)
    for index, document in enumerate(documents, 1):
        shutil.copy2(work / f"{index}.pdf", history / document.source_file)
    (history / "result.json").write_text(
        json.dumps(dumped, ensure_ascii=False), encoding="utf-8")
    markdown = to_markdown(result)
    (history / "report.md").write_text(markdown, encoding="utf-8")
    (history / "report.txt").write_text(markdown, encoding="utf-8")
    (history / "judgment.json").write_text(
        json.dumps(summarize_matrix(result), ensure_ascii=False), encoding="utf-8")
    (history / "documents.json").write_text(
        json.dumps([document.model_dump() for document in documents], ensure_ascii=False),
        encoding="utf-8")
    (history / "meta.json").write_text(json.dumps({
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claims_summary": claims[:200],
        "claims": claims,
        "documents": [document.filename for document in documents],
        "analysis_prompt": analysis_prompt,
    }, ensure_ascii=False), encoding="utf-8")


@app.delete("/api/jobs/{job_id}")
def cancel_running_job(job_id: str):
    job_id = safe_job_id(job_id)
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(404, "작업을 찾을 수 없습니다.")
    if record["status"] in {"completed", "failed", "cancelled"}:
        return {"job_id": job_id, "status": record["status"], "kill_requested": False}
    record.update(status="cancelling", stage="취소 중")
    killed = agy.cancel_job(job_id)
    write_log(job_id, "cancellation requested")
    worker = record.get("_worker")
    if (not worker or not worker.is_alive()) and not record.get("_staging"):
        record.update(status="cancelled", stage="취소됨")
        agy.finish_job(job_id)
    return {"job_id": job_id, "status": record["status"], "kill_requested": killed}

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job_id = safe_job_id(job_id)
    if job_id not in jobs: raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return {k: v for k, v in jobs[job_id].items()
            if k not in {"documents", "result"} and not k.startswith("_")}

@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    if job_id not in jobs or "result" not in jobs[job_id]: raise HTTPException(404, "결과를 찾을 수 없습니다.")
    return jobs[job_id]["result"]


@app.post("/api/jobs/{job_id}/dependent-claims")
def add_dependent_claims(job_id: str, payload: DependentClaimsAdd):
    """완료된 보고서에 복수 종속항을 한 번의 구성대비 호출로 추가합니다."""
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

    write_log(job_id, f"dependent claims batch started: {incoming_numbers}")
    try:
        def progress(stage: str):
            if job_id in jobs:
                jobs[job_id]["stage"] = stage
            write_log(job_id, stage)

        result = extend_with_dependent_claims(result, combined_text, set(incoming_numbers),
                                               documents, analysis_prompt, progress)
        _persist_analysis(job_id, result, combined_text, documents, analysis_prompt)
        if job_id in jobs:
            jobs[job_id].update(result=result.model_dump(), claims_text=combined_text,
                                documents=documents, status="completed", stage="완료")
        write_log(job_id, f"dependent claims batch completed: {incoming_numbers}")
        return {"job_id": job_id, "added_claims": incoming_numbers, "result": result.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        write_log(job_id, f"dependent claims batch failed: {type(exc).__name__}: {exc}")
        raise HTTPException(502, str(exc)) from exc

@app.post("/api/jobs/{job_id}/prior-art")
def search_prior_art(job_id: str, payload: dict | None = None):
    """미커버로 남은 구성만 웹에서 검색합니다. 외부 호출이므로 자동 실행하지 않습니다."""
    job_id = safe_job_id(job_id)
    if jobs.get(job_id, {}).get("status") in {"preparing", "running", "cancelling", "cancelled"}:
        raise HTTPException(409, "완료되지 않은 보고서에서는 선행기술 검색을 실행할 수 없습니다.")
    stored = _load_result(job_id)
    if stored is None: raise HTTPException(404, "결과를 찾을 수 없습니다.")
    result, claims_text = stored
    targets = uncovered_elements(result, claims_text)
    if not targets: return {"hits": [], "message": "결합 후 미커버로 남은 구성이 없습니다."}
    hits, warnings = priorart.search(targets)
    result.prior_art = hits; result.validation += warnings
    history = HISTORY_DIR / job_id
    if history.exists():
        (history / "result.json").write_text(json.dumps(result.model_dump(), ensure_ascii=False), encoding="utf-8")
        (history / "report.md").write_text(to_markdown(result), encoding="utf-8")
        (history / "report.txt").write_text(to_markdown(result), encoding="utf-8")
    if job_id in jobs: jobs[job_id]["result"] = result.model_dump()
    write_log(job_id, f"prior art search: {len(hits)} hits for {len(targets)} uncovered elements")
    return {"hits": [hit.model_dump() for hit in hits], "searched": targets, "warnings": warnings}

def _load_result(job_id: str) -> tuple[AnalysisResult, str] | None:
    """메모리에 없으면 히스토리에서 결과와 청구항 원문을 되살립니다."""
    record = jobs.get(job_id, {})
    raw, claims_text = record.get("result"), record.get("claims_text", "")
    if raw is None:
        path = HISTORY_DIR / job_id / "result.json"
        if not path.exists(): return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not claims_text:
        meta = HISTORY_DIR / job_id / "meta.json"
        if meta.exists(): claims_text = json.loads(meta.read_text(encoding="utf-8")).get("claims", "")
    if not claims_text: return None
    return AnalysisResult.model_validate(raw), claims_text


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
        path = history / "documents.json"
        if not path.exists():
            return None
        try:
            documents = [Document.model_validate(item)
                         for item in json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, json.JSONDecodeError, ValueError):
            return None
    prompt = ""
    meta_path = history / "meta.json"
    if meta_path.exists():
        try:
            prompt = str(json.loads(meta_path.read_text(encoding="utf-8")).get("analysis_prompt") or "")
        except (OSError, json.JSONDecodeError):
            pass
    return result, claims_text, documents, prompt


def _persist_analysis(job_id: str, result: AnalysisResult, claims_text: str,
                      documents: list[Document], analysis_prompt: str) -> None:
    """추가된 항을 결과·리포트·감사 데이터와 원 입력 메타데이터에 함께 반영합니다."""
    history = HISTORY_DIR / job_id
    history.mkdir(exist_ok=True)
    dumped = result.model_dump()
    (history / "result.json").write_text(json.dumps(dumped, ensure_ascii=False), encoding="utf-8")
    markdown = to_markdown(result)
    (history / "report.md").write_text(markdown, encoding="utf-8")
    (history / "report.txt").write_text(markdown, encoding="utf-8")
    (history / "judgment.json").write_text(
        json.dumps(summarize_matrix(result), ensure_ascii=False), encoding="utf-8")
    (history / "documents.json").write_text(
        json.dumps([document.model_dump() for document in documents], ensure_ascii=False), encoding="utf-8")
    meta_path = history / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(claims=claims_text, claims_summary=claims_text[:200],
                analysis_prompt=analysis_prompt, documents=[document.filename for document in documents])
    (meta_path).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

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
    items = []
    for path in HISTORY_DIR.iterdir():
        if path.is_dir() and (path / "meta.json").exists(): items.append(json.loads((path / "meta.json").read_text(encoding="utf-8")))
    return sorted(items, key=lambda item: item["created_at"], reverse=True)

@app.get("/api/history/{job_id}")
def history_detail(job_id: str):
    job_id = safe_job_id(job_id)
    path = HISTORY_DIR / job_id / "meta.json"
    if not path.exists(): raise HTTPException(404, "히스토리를 찾을 수 없습니다.")
    result_path = path.parent / "result.json"
    result = jobs.get(job_id, {}).get("result")
    if result is None and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return {"meta": json.loads(path.read_text(encoding="utf-8")), "result": result}

def remove_job_record(job_id: str):
    """분석 결과와 리포트를 지웁니다. 로그는 로그 탭에서 별도로 관리합니다."""
    shutil.rmtree(HISTORY_DIR / job_id, ignore_errors=True)
    jobs.pop(job_id, None)

@app.delete("/api/history")
def clear_history():
    removed = [path.name for path in HISTORY_DIR.iterdir() if path.is_dir()]
    for job_id in removed: remove_job_record(job_id)
    return {"ok": True, "removed": len(removed)}

@app.delete("/api/history/{job_id}")
def delete_history(job_id: str):
    job_id = safe_job_id(job_id)
    remove_job_record(job_id); return {"ok": True}

@app.delete("/api/cache")
def clear_cache():
    """판정 캐시를 비웁니다. 프롬프트나 모델을 바꾼 뒤 전부 다시 판정하고 싶을 때만 씁니다."""
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

@app.put("/api/logs/{job_id}")
def update_logs(job_id: str, payload: dict):
    job_id = safe_job_id(job_id)
    (LOG_DIR / f"{job_id}.log").write_text(str(payload.get("content", "")), encoding="utf-8"); return {"ok": True}


@app.delete("/api/logs")
def clear_logs():
    paths = list(LOG_DIR.glob("*.log"))
    for path in paths:
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
