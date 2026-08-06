import fitz
import json
import threading
import time
from fastapi.testclient import TestClient

from app import agy, main
from app.models import AnalysisResult

client = TestClient(main.app)


def pdf_bytes(text: str = "메모리 컨트롤러는 쓰기 요청을 큐에 저장한다.") -> bytes:
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), text)
        return doc.tobytes()


def start_job(filename: str = "prior.pdf", content_type: str = "application/pdf"):
    """준비 → 시작 경로로 작업 하나를 만듭니다. 프런트엔드가 쓰는 경로와 같습니다."""
    job_id = client.post("/api/jobs/prepare").json()["job_id"]
    response = client.post(
        f"/api/jobs/{job_id}/start",
        data={"claims": "(A) 쓰기 요청을 큐에 저장하는 것"},
        files={"pdf_files": (filename, pdf_bytes(), content_type)},
    )
    worker = main.jobs.get(job_id, {}).get("_worker")
    if worker is not None:
        worker.join(timeout=5)
    return job_id, response


def post_job():
    job_id, _ = start_job()
    return job_id


def fake_result(job_id, claims_text, documents, analysis_prompt="", progress=None,
                decomposition=None):
    if progress:
        progress("구성대비 1/1")
    if decomposition is not None:
        decomposition.update({"version": 1, "claims": {"1": [{"label": "A", "text": "쓰기 요청을 큐에 저장하는 것"}]}})
    return AnalysisResult(job_id=job_id, claim_mapping=[], reports=[], validation=[])


def finish_dependent(job_id: str):
    """종속항 대비 워커가 끝날 때까지 기다립니다. 초기 분석과 같은 비동기 구조입니다."""
    worker = main.jobs.get(job_id, {}).get("_worker")
    if worker is not None:
        worker.join(timeout=5)
    return main.jobs.get(job_id, {})


def test_completed_job_is_marked_completed(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id, response = start_job()
    assert response.status_code == 202
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "completed"
    client.delete(f"/api/history/{job_id}")


def test_async_job_can_be_cancelled_without_leaving_a_report(monkeypatch):
    entered = threading.Event()

    def slow_result(job_id, claims_text, documents, analysis_prompt="", progress=None,
                    decomposition=None):
        entered.set()
        while not agy.is_cancelled(job_id):
            time.sleep(0.01)
        agy.raise_if_cancelled(job_id)

    monkeypatch.setattr(main, "analyze", slow_result)
    job_id = client.post("/api/jobs/prepare").json()["job_id"]
    response = client.post(
        f"/api/jobs/{job_id}/start",
        data={"claims": "(A) 쓰기 요청을 큐에 저장하는 것"},
        files={"pdf_files": ("prior.pdf", pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 202
    assert entered.wait(timeout=2)

    cancelled = client.delete(f"/api/jobs/{job_id}")
    assert cancelled.status_code == 200
    assert cancelled.json()["kill_requested"] is True
    main.jobs[job_id]["_worker"].join(timeout=2)

    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    assert not (main.HISTORY_DIR / job_id).exists()
    assert client.post(f"/api/jobs/{job_id}/prior-art").status_code == 409
    main.remove_job_record(job_id)


def test_uploaded_source_pdf_is_preserved_with_the_history(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    history = main.HISTORY_DIR / job_id
    raw_documents = json.loads((history / "documents.json").read_text(encoding="utf-8"))
    source_file = raw_documents[0]["source_file"]
    saved = history / source_file

    assert source_file == "sources/1-prior.pdf"
    assert saved.exists() and saved.read_bytes().startswith(b"%PDF")
    response = client.get(f"/api/jobs/{job_id}/sources/1")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")

    client.delete(f"/api/history/{job_id}")
    assert not history.exists()


def test_cli_failure_returns_the_reason_and_marks_the_job_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("agy CLI 명령줄이 Windows 길이 제한을 초과했습니다.")

    monkeypatch.setattr(main, "analyze", boom)
    job_id, _ = start_job()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "길이 제한" in job["error"]


def test_history_can_be_cleared_at_once(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_ids = [post_job() for _ in range(2)]
    assert len(client.get("/api/history").json()) == 2
    response = client.delete("/api/history")
    assert response.json() == {"ok": True, "removed": 2}
    assert client.get("/api/history").json() == []
    assert len(client.get("/api/logs").json()) == 2
    assert all(job_id not in main.jobs for job_id in job_ids)


def test_history_and_logs_are_managed_independently(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    client.delete(f"/api/history/{job_id}")
    assert client.get("/api/history").json() == []
    assert client.get(f"/api/logs/{job_id}").status_code == 200


def test_all_logs_can_be_viewed_and_cleared(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_ids = [post_job() for _ in range(2)]

    listed = client.get("/api/logs").json()
    assert {item["job_id"] for item in listed} == set(job_ids)
    assert all(item["size"] > 0 and item["updated_at"] for item in listed)
    assert "analysis completed" in client.get(f"/api/logs/{job_ids[0]}").json()["content"]

    response = client.delete("/api/logs")
    assert response.json() == {"ok": True, "removed": 2}
    assert client.get("/api/logs").json() == []


def test_settings_models_are_listed_for_the_dropdown(monkeypatch):
    monkeypatch.setattr(main, "available_models", lambda settings, refresh: ["gemini-3.6-flash-medium"])
    response = client.get("/api/settings/models", params={"provider": "agy"})
    assert response.json() == {"provider": "agy", "models": ["gemini-3.6-flash-medium"]}
    assert client.get("/api/settings/models", params={"provider": "nope"}).status_code == 400


def test_prior_art_search_is_not_run_automatically(monkeypatch):
    """외부 웹에 나가는 단계라 분석 중에는 호출되지 않고, 별도 요청에서만 실행된다."""
    called = []
    monkeypatch.setattr(main, "analyze", fake_result)
    monkeypatch.setattr(main.priorart, "search", lambda targets: (called.append(targets) or ([], [])))
    job_id = post_job()
    assert called == []
    response = client.post(f"/api/jobs/{job_id}/prior-art")
    assert response.status_code == 200
    assert called == []                      # 미커버 구성이 없으면 CLI를 부르지 않는다
    assert "미커버로 남은 구성이 없습니다" in response.json()["message"]


def test_cache_can_be_cleared(monkeypatch):
    monkeypatch.setattr(main.cache, "clear", lambda: 3)
    assert client.delete("/api/cache").json() == {"ok": True, "removed": 3}


def test_dependent_claims_reuse_saved_documents_and_are_sent_as_one_batch(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    captured = {}

    def fake_extend(result, claims_text, numbers, documents, prompt="", progress=None,
                    decomposition=None, checkpoint=None):
        captured.update(claims_text=claims_text, numbers=numbers,
                        decomposition=decomposition,
                        filenames=[document.filename for document in documents])
        if progress:
            progress("종속항 2개 일괄 구성대비")
        if checkpoint:
            checkpoint(result)
        return result

    monkeypatch.setattr(main, "extend_with_dependent_claims", fake_extend)
    response = client.post(f"/api/jobs/{job_id}/dependent-claims", json={"claims": (
        "【청구항 2】\n제1항에 있어서, (A) 우선순위 큐\n"
        "【청구항 3】\n제1항에 있어서, (A) 순환형 큐"
    )})

    assert response.status_code == 202
    assert response.json()["added_claims"] == [2, 3]
    assert finish_dependent(job_id)["status"] == "completed"
    assert captured["numbers"] == {2, 3}
    assert captured["filenames"] == ["prior.pdf"]
    assert captured["claims_text"].startswith("【청구항 1】")
    # 초기 분석이 남긴 구성분해를 그대로 물려받아야 판정 캐시 키가 유지된다.
    assert captured["decomposition"]["claims"]["1"][0]["label"] == "A"
    client.delete(f"/api/history/{job_id}")


def test_a_cancelled_dependent_run_keeps_the_claims_it_already_judged(monkeypatch):
    """종속항 대비는 수 분이 걸린다. 취소했다고 그때까지의 판정까지 버리면,
    다시 눌렀을 때 같은 항을 처음부터 다시 대비하게 된다.
    """
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    entered = threading.Event()

    def slow_extend(result, claims_text, numbers, documents, prompt="", progress=None,
                    decomposition=None, checkpoint=None):
        entered.set()
        while not agy.is_cancelled(job_id):
            time.sleep(0.01)
        result.reports = [report for report in result.reports]     # 확정된 항만 남긴 상태
        if checkpoint:
            checkpoint(result)
        raise agy.AnalysisCancelled("보고서 생성을 취소했습니다.")

    monkeypatch.setattr(main, "extend_with_dependent_claims", slow_extend)
    response = client.post(f"/api/jobs/{job_id}/dependent-claims",
                           json={"claims": "【청구항 2】\n제1항에 있어서, (A) 우선순위 큐"})
    assert response.status_code == 202
    assert entered.wait(timeout=2)

    cancelled = client.delete(f"/api/jobs/{job_id}")
    assert cancelled.status_code == 200 and cancelled.json()["kill_requested"] is True
    assert finish_dependent(job_id)["status"] == "cancelled"
    # 초기 분석의 보고서는 그대로 남고, 히스토리도 지워지지 않는다.
    assert (main.HISTORY_DIR / job_id / "result.json").exists()
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 200
    client.delete(f"/api/history/{job_id}")


def test_dependent_claim_endpoint_rejects_an_independent_claim(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    response = client.post(f"/api/jobs/{job_id}/dependent-claims",
                           json={"claims": "【청구항 2】\n(A) 독립적인 장치"})
    assert response.status_code == 400
    assert "청구항 1에 종속되지 않은" in response.json()["detail"]
    client.delete(f"/api/history/{job_id}")


def test_rejected_upload_marks_the_job_failed():
    job_id, response = start_job("prior.txt", "text/plain")
    assert response.status_code == 400
    assert main.jobs[job_id]["status"] == "failed"
