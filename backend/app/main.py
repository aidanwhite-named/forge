import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from .config import HISTORY_DIR, LOG_DIR, MAX_PDF_SIZE_MB, MAX_TOTAL_UPLOAD_SIZE_MB, AGY_MODEL, load_runtime_settings, save_runtime_settings
from .agy import _build_command, available_models
from .pdf import classify, extract_pdf
from .pipeline import analyze, to_markdown

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
async def create_job(claims: str = Form(...), pdf_files: list[UploadFile] = File(...), agy_reasoning_effort: str = Form("medium"), analysis_prompt: str = Form("")):
    if not claims.strip(): raise HTTPException(400, "청구항을 입력하세요.")
    if not 1 <= len(pdf_files) <= 7: raise HTTPException(400, "PDF는 1~7개만 업로드할 수 있습니다.")
    if agy_reasoning_effort not in {"low", "medium", "high"}: raise HTTPException(400, "reasoning effort가 올바르지 않습니다.")
    job_id = str(uuid.uuid4()); work = Path(tempfile.mkdtemp(prefix=f"patent-{job_id}-"))
    documents = []; total = 0
    jobs[job_id] = {"job_id": job_id, "status": "running", "claims": claims, "documents": documents, "effort": agy_reasoning_effort}
    write_log(job_id, "analysis started")
    try:
        for index, upload in enumerate(pdf_files, 1):
            filename = Path(upload.filename or f"document-{index}.pdf").name
            if not filename.lower().endswith(".pdf"): raise HTTPException(400, "PDF 파일만 업로드할 수 있습니다.")
            target = work / f"{index}.pdf"; content = await upload.read(); total += len(content)
            if len(content) > MAX_PDF_SIZE_MB * 1024 * 1024: raise HTTPException(413, "개별 PDF 크기 제한을 초과했습니다.")
            if total > MAX_TOTAL_UPLOAD_SIZE_MB * 1024 * 1024: raise HTTPException(413, "전체 업로드 크기 제한을 초과했습니다.")
            target.write_bytes(content); extracted = extract_pdf(target, str(index))
            documents.append({"id": str(index), "filename": filename, "type": classify(extracted["text"]), **extracted})
        result = analyze(job_id, claims, documents, agy_reasoning_effort, analysis_prompt)
        jobs[job_id]["result"] = result.model_dump()
        history = HISTORY_DIR / job_id; history.mkdir(exist_ok=True)
        (history / "result.json").write_text(json.dumps(jobs[job_id]["result"], ensure_ascii=False), encoding="utf-8")
        (history / "report.md").write_text(to_markdown(result), encoding="utf-8")
        (history / "report.txt").write_text(to_markdown(result), encoding="utf-8")
        (history / "meta.json").write_text(json.dumps({"job_id": job_id, "created_at": datetime.now(timezone.utc).isoformat(), "claims_summary": claims[:200], "documents": [d["filename"] for d in documents], "agy_reasoning_effort": agy_reasoning_effort,
                                                   "analysis_prompt": analysis_prompt.strip()}, ensure_ascii=False), encoding="utf-8")
        jobs[job_id]["status"] = "completed"
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

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs: raise HTTPException(404, "작업을 찾을 수 없습니다.")
    return {k: v for k, v in jobs[job_id].items() if k not in {"documents", "result"}} 

@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    if job_id not in jobs or "result" not in jobs[job_id]: raise HTTPException(404, "결과를 찾을 수 없습니다.")
    return jobs[job_id]["result"]

@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, format: str = "md"):
    if format not in {"md", "txt"}: raise HTTPException(400, "format은 md 또는 txt만 지원합니다.")
    job_id = safe_job_id(job_id)
    path = HISTORY_DIR / job_id / f"report.{format}"
    if not path.exists(): raise HTTPException(404, "리포트를 찾을 수 없습니다.")
    return PlainTextResponse(path.read_text(encoding="utf-8"), headers={"Content-Disposition": f'attachment; filename="{job_id}.{format}"'})

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
    """분석 결과·리포트·작업 로그를 한 작업 단위로 함께 지웁니다."""
    shutil.rmtree(HISTORY_DIR / job_id, ignore_errors=True)
    (LOG_DIR / f"{job_id}.log").unlink(missing_ok=True)
    jobs.pop(job_id, None)

@app.delete("/api/history")
def clear_history(logs: bool = True):
    removed = [path.name for path in HISTORY_DIR.iterdir() if path.is_dir()]
    for job_id in removed: remove_job_record(job_id)
    if logs:
        for path in LOG_DIR.glob("*.log"): path.unlink(missing_ok=True)
    return {"ok": True, "removed": len(removed)}

@app.delete("/api/history/{job_id}")
def delete_history(job_id: str):
    job_id = safe_job_id(job_id)
    remove_job_record(job_id); return {"ok": True}

@app.get("/api/logs")
def logs(): return [{"job_id": p.stem, "size": p.stat().st_size} for p in LOG_DIR.glob("*.log")]

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

@app.get("/api/settings")
def get_settings(): return load_runtime_settings()

@app.get("/api/settings/models")
def settings_models(provider: str | None = None, command: str | None = None, refresh: bool = False):
    settings = load_runtime_settings()
    provider = (provider or settings["provider"]).lower()
    if provider not in {"agy", "claude", "gpt"}: raise HTTPException(400, "지원하지 않는 provider입니다.")
    command = command or (settings["command"] if provider == settings["provider"] else provider)
    return {"provider": provider, "models": available_models({"provider": provider, "command": command}, refresh)}

@app.put("/api/settings")
def update_settings(payload: dict):
    provider = str(payload.get("provider", "agy")).lower()
    if provider not in {"agy", "claude", "gpt"}: raise HTTPException(400, "지원하지 않는 provider입니다.")
    command, model = str(payload.get("command", provider)).strip(), str(payload.get("model", AGY_MODEL)).strip()
    if not command or not model: raise HTTPException(400, "명령어와 모델을 입력해 주세요.")
    prompt = str(payload.get("prompt", "")).strip()
    return save_runtime_settings({"provider": provider, "command": command, "model": model, "prompt": prompt})

@app.post("/api/settings/test")
def test_settings(payload: dict):
    provider = str(payload.get("provider", "agy")).lower()
    try:
        _build_command("연결 테스트", "low", {"provider": provider, "command": str(payload.get("command", provider)), "model": str(payload.get("model", AGY_MODEL))})
        return {"ok": True, "message": f"{provider} CLI를 찾았습니다."}
    except Exception as exc: return {"ok": False, "message": str(exc)}
