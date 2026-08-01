import fitz
from fastapi.testclient import TestClient

from app import main
from app.models import AnalysisResult

client = TestClient(main.app)


def pdf_bytes(text: str = "메모리 컨트롤러는 쓰기 요청을 큐에 저장한다.") -> bytes:
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), text)
        return doc.tobytes()


def post_job():
    return client.post("/api/jobs", data={"claims": "(A) 쓰기 요청을 큐에 저장하는 것", "agy_reasoning_effort": "low"},
                       files={"pdf_files": ("prior.pdf", pdf_bytes(), "application/pdf")})


def fake_result(job_id, claims_text, documents, effort, analysis_prompt=""):
    return AnalysisResult(job_id=job_id, claim_mapping=[], claims=[], summary="요약", validation=[])


def test_completed_job_is_marked_completed(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    response = post_job()
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "completed"
    client.delete(f"/api/history/{job_id}")


def test_cli_failure_returns_the_reason_and_marks_the_job_failed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("agy CLI 명령줄이 Windows 길이 제한을 초과했습니다.")

    monkeypatch.setattr(main, "analyze", boom)
    response = post_job()
    assert response.status_code == 502
    assert "길이 제한" in response.json()["detail"]
    job = client.get(f"/api/jobs/{list(main.jobs)[-1]}").json()
    assert job["status"] == "failed"
    assert "길이 제한" in job["error"]


def test_history_can_be_cleared_at_once(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_ids = [post_job().json()["job_id"] for _ in range(2)]
    assert len(client.get("/api/history").json()) == 2
    response = client.delete("/api/history")
    assert response.json() == {"ok": True, "removed": 2}
    assert client.get("/api/history").json() == []
    assert client.get("/api/logs").json() == []
    assert all(job_id not in main.jobs for job_id in job_ids)


def test_deleting_one_history_entry_removes_its_log(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job().json()["job_id"]
    client.delete(f"/api/history/{job_id}")
    assert client.get("/api/history").json() == []
    assert client.get(f"/api/logs/{job_id}").status_code == 404


def test_settings_models_are_listed_for_the_dropdown(monkeypatch):
    monkeypatch.setattr(main, "available_models", lambda settings, refresh: ["gemini-3.6-flash-medium"])
    response = client.get("/api/settings/models", params={"provider": "agy", "command": "agy"})
    assert response.json() == {"provider": "agy", "models": ["gemini-3.6-flash-medium"]}
    assert client.get("/api/settings/models", params={"provider": "nope"}).status_code == 400


def test_rejected_upload_marks_the_job_failed():
    response = client.post("/api/jobs", data={"claims": "(A) 무언가", "agy_reasoning_effort": "low"},
                           files={"pdf_files": ("prior.txt", b"not a pdf", "text/plain")})
    assert response.status_code == 400
    assert main.jobs[list(main.jobs)[-1]]["status"] == "failed"
