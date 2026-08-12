import fitz
import json
import threading
import time
from fastapi.testclient import TestClient

from app import agy, cache, main
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
                decomposition=None, cache_keys=None, priority_date=""):
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
                    decomposition=None, cache_keys=None, priority_date=""):
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
    # 보고서가 남지 않았으므로 그 위에서 도는 작업은 "찾을 수 없음"입니다. 상태 문자열이
    # 아니라 저장된 보고서의 유무가 기준입니다 — 종속항 대비를 취소한 경우에는 보고서가
    # 보존되므로 같은 cancelled 상태에서도 검색이 되어야 합니다(아래 전용 테스트).
    assert client.post(f"/api/jobs/{job_id}/prior-art").status_code == 404
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
    assert response.json() == {"ok": True, "removed": 2, "cache_purged": 0}
    assert client.get("/api/history").json() == []
    assert len(client.get("/api/logs").json()) == 2
    assert all(job_id not in main.jobs for job_id in job_ids)


def test_deleting_history_also_removes_the_legacy_copy(monkeypatch):
    """레거시 사본을 남기면 다음 기동에 마이그레이션이 그대로 되살립니다.

    사용자가 지웠는데 서버를 재시작하면 돌아오는 상태였습니다.
    """
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    legacy = main.DATA_DIR / "history" / job_id
    legacy.mkdir(parents=True)
    (legacy / "meta.json").write_text('{"job_id": "%s"}' % job_id, encoding="utf-8")

    client.delete(f"/api/history/{job_id}")
    assert not (main.HISTORY_DIR / job_id).exists()
    assert not legacy.exists()


def test_deleting_history_purges_that_jobs_judgment_cache(monkeypatch):
    """판정 캐시에는 문헌 원문 발췌가 들어 있습니다.

    히스토리만 지우고 캐시를 남기면, 사용자가 지웠다고 생각한 문장이 디스크에 남습니다.
    """
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    # 이 분석이 쓴 판정 2건이 캐시에 있는 상태를 만든다.
    keys = ["cafe1234", "beef5678"]
    for key in keys:
        (cache.CACHE_DIR / f"{key}.json").write_text("[]", encoding="utf-8")
    other = cache.CACHE_DIR / "unrelated.json"
    other.write_text("[]", encoding="utf-8")
    main._save_cache_keys(job_id, set(keys))

    response = client.delete(f"/api/history/{job_id}")
    assert response.json() == {"ok": True, "cache_purged": 2}
    assert not any((cache.CACHE_DIR / f"{key}.json").exists() for key in keys)
    assert other.exists()                      # 다른 분석의 판정은 건드리지 않는다


def test_clearing_logs_also_removes_the_legacy_copies(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    legacy = main.DATA_DIR / "logs"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / f"{job_id}.log").write_text("old", encoding="utf-8")

    client.delete("/api/logs")
    assert client.get("/api/logs").json() == []
    assert not (legacy / f"{job_id}.log").exists()
    client.delete(f"/api/history/{job_id}")


def test_an_interrupted_job_reports_why_instead_of_404(monkeypatch):
    """서버가 재시작되면 메모리 레코드가 사라집니다.

    404를 주면 폴링 중인 화면에는 "작업을 찾을 수 없습니다"만 남아, 분석이 왜 멈췄는지도
    다시 돌려도 되는지도 알 수 없습니다.
    """
    job_id = client.post("/api/jobs/prepare").json()["job_id"]
    main.set_job_status(job_id, status="running", stage="구성대비 3/8")
    main.jobs.clear()                          # 서버 재시작과 같은 상태

    assert main.recover_interrupted_jobs() == 1
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "interrupted"
    assert "다시 실행하면" in job["error"]
    main.remove_job_record(job_id)


def test_cell_progress_is_exposed_as_numbers(monkeypatch):
    """진행률을 문자열에만 담으면 화면이 진행 바를 그릴 수 없습니다."""
    def analyze_with_progress(job_id, claims_text, documents, analysis_prompt="", progress=None,
                              decomposition=None, cache_keys=None, priority_date=""):
        progress("구성대비 3/8 — 청구항 1 × prior.pdf", 3, 8)
        assert client.get(f"/api/jobs/{job_id}").json()["progress"] == {"done": 3, "total": 8}
        return AnalysisResult(job_id=job_id, claim_mapping=[], reports=[], validation=[])

    monkeypatch.setattr(main, "analyze", analyze_with_progress)
    job_id = post_job()
    assert main.jobs[job_id]["status"] == "completed"
    client.delete(f"/api/history/{job_id}")


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
    monkeypatch.setattr(main.priorart, "search", lambda targets: (called.append(targets) or []))
    job_id = post_job()
    assert called == []
    response = client.post(f"/api/jobs/{job_id}/prior-art")
    assert response.status_code == 200
    assert called == []                      # 미커버 구성이 없으면 CLI를 부르지 않는다
    assert "미커버로 남은 구성이 없습니다" in response.json()["message"]


def prepare_uncovered_search(monkeypatch) -> str:
    """미커버 구성이 하나 남은 완료 보고서를 만들어 검색이 CLI까지 가게 한다."""
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    monkeypatch.setattr(main, "uncovered_elements",
                        lambda result, claims_text: [{"claim_number": 1, "label": "A", "text": "쓰기 요청"}])
    return job_id


def test_prior_art_search_can_be_cancelled_while_the_cli_is_running(monkeypatch):
    """취소 요청이 검색 중인 CLI까지 닿는다.

    엔드포인트가 스레드를 작업에 묶어 두지 않으면 run_cli가 띄운 프로세스가 어디에도
    등록되지 않아, 취소를 눌러도 죽일 대상을 찾지 못하고 타임아웃까지 계속 돈다.
    """
    job_id = prepare_uncovered_search(monkeypatch)
    entered, released = threading.Event(), threading.Event()

    def blocking_cli(prompt, expect="claims"):
        entered.set()
        released.wait(timeout=5)
        agy.raise_if_cancelled()             # 실제 run_cli가 프로세스 종료 뒤 확인하는 자리
        return {"hits": []}

    monkeypatch.setattr("app.priorart.run_cli", blocking_cli)

    outcome = {}
    worker = threading.Thread(
        target=lambda: outcome.setdefault("response", client.post(f"/api/jobs/{job_id}/prior-art")))
    worker.start()
    assert entered.wait(timeout=5)

    cancelled = client.delete(f"/api/jobs/{job_id}/prior-art")
    released.set()
    worker.join(timeout=5)

    assert cancelled.json() == {"job_id": job_id, "running": True, "kill_requested": True}
    assert outcome["response"].status_code == 409
    assert "취소" in outcome["response"].json()["detail"]


def seed_existing_prior_art(job_id: str):
    """지난 검색에서 찾아 둔 선행기술이 이미 저장된 상태를 만든다."""
    stored = main.HISTORY_DIR / job_id / "result.json"
    saved = json.loads(stored.read_text(encoding="utf-8"))
    saved["prior_art"] = [{"claim_number": 1, "label": "A", "document_number": "US 1 A",
                           "title": "기존 문헌", "published": "", "correspondence": "",
                           "remaining_difference": "", "url": ""}]
    stored.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    main.jobs[job_id]["result"] = saved
    return stored


def test_cancelled_prior_art_search_keeps_the_existing_report(monkeypatch):
    """취소는 실패가 아니다. 이미 찾아 둔 선행기술을 빈 목록으로 덮어쓰지 않는다."""
    job_id = prepare_uncovered_search(monkeypatch)
    stored = seed_existing_prior_art(job_id)

    def self_cancelling_cli(prompt, expect="claims"):
        agy.cancel_job(agy.current_job())
        agy.raise_if_cancelled()

    monkeypatch.setattr("app.priorart.run_cli", self_cancelling_cli)

    assert client.post(f"/api/jobs/{job_id}/prior-art").status_code == 409
    kept = json.loads(stored.read_text(encoding="utf-8"))["prior_art"]
    assert [hit["document_number"] for hit in kept] == ["US 1 A"]

    # 취소한 job의 상태를 건드리지 않으므로 곧바로 다시 검색할 수 있다.
    monkeypatch.setattr(main.priorart, "search", lambda targets: [])
    assert client.post(f"/api/jobs/{job_id}/prior-art").status_code == 200


def test_failed_prior_art_search_keeps_the_existing_report(monkeypatch):
    """CLI가 답을 못 낸 것은 '0건'이 아니다. 보고서를 건드리지 않고 오류로 끝낸다."""
    job_id = prepare_uncovered_search(monkeypatch)
    stored = seed_existing_prior_art(job_id)

    def failing_cli(prompt, expect="claims"):
        raise RuntimeError("exit code 1: provider unreachable")

    monkeypatch.setattr("app.priorart.run_cli", failing_cli)

    response = client.post(f"/api/jobs/{job_id}/prior-art")
    assert response.status_code == 502
    assert "선행기술 검색에 실패했습니다" in response.json()["detail"]

    saved = json.loads(stored.read_text(encoding="utf-8"))
    assert [hit["document_number"] for hit in saved["prior_art"]] == ["US 1 A"]
    # 실패 기록은 로그에만 남깁니다. 보고서의 검증 참고는 분석상의 유보사항 자리입니다.
    assert not any("검색에 실패" in note for note in saved["validation"])
    assert "prior art search failed" in (main.LOG_DIR / f"{job_id}.log").read_text(encoding="utf-8")

    # 실패해도 상태가 남지 않으므로 곧바로 다시 검색할 수 있다.
    monkeypatch.setattr(main.priorart, "search", lambda targets: [])
    assert client.post(f"/api/jobs/{job_id}/prior-art").status_code == 200


def test_cancelling_an_idle_prior_art_search_is_a_no_op(monkeypatch):
    job_id = prepare_uncovered_search(monkeypatch)
    assert client.delete(f"/api/jobs/{job_id}/prior-art").json() == {
        "job_id": job_id, "running": False, "kill_requested": False}


def test_cache_can_be_cleared(monkeypatch):
    monkeypatch.setattr(main.cache, "clear", lambda: 3)
    assert client.delete("/api/cache").json() == {"ok": True, "removed": 3}


def test_dependent_claims_reuse_saved_documents_and_are_sent_as_one_batch(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    captured = {}

    def fake_extend(result, claims_text, numbers, documents, prompt="", progress=None,
                    decomposition=None, checkpoint=None, cache_keys=None):
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
                    decomposition=None, checkpoint=None, cache_keys=None):
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

    # 보존된 보고서 위에서 후속 작업이 계속 가능해야 한다. 종전에는 선행기술 검색만
    # "cancelled"를 실행 중으로 취급해서, 종속항 추가는 되는데 검색만 영구히 막혔다.
    monkeypatch.setattr(main, "uncovered_elements",
                        lambda result, claims_text: [{"claim_number": 1, "label": "A", "text": "쓰기 요청"}])
    monkeypatch.setattr(main.priorart, "search", lambda targets: [])
    assert client.post(f"/api/jobs/{job_id}/prior-art").status_code == 200
    client.delete(f"/api/history/{job_id}")


def test_history_list_survives_an_unreadable_meta(monkeypatch):
    """meta.json 한 건이 깨져도 나머지 보고서는 목록에 남아야 한다.

    meta.json은 취소·크래시가 쓰기 도중에 끼면 잘린 채 남을 수 있다. 그때 목록 전체가
    500이 되면 멀쩡한 보고서까지 화면에서 사라져, 히스토리가 통째로 날아간 것처럼 보인다.
    """
    monkeypatch.setattr(main, "analyze", fake_result)
    good, broken = post_job(), post_job()
    (main.HISTORY_DIR / broken / "meta.json").write_text('{"job_id": "x", "crea',
                                                         encoding="utf-8")
    listed = client.get("/api/history")
    assert listed.status_code == 200
    assert [item["job_id"] for item in listed.json()] == [good]

    # created_at이 없는 기록도 정렬에서 터지지 않는다.
    (main.HISTORY_DIR / broken / "meta.json").write_text('{"job_id": "%s"}' % broken,
                                                         encoding="utf-8")
    assert client.get("/api/history").status_code == 200
    client.delete("/api/history")


def test_finished_jobs_release_the_extracted_document_text(monkeypatch):
    """완료된 작업이 PDF 본문을 계속 붙들고 있으면 서버를 켜 둔 만큼 메모리가 늘기만 한다."""
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id = post_job()
    assert not (set(main.jobs[job_id]) & {"documents", "result", "claims", "claims_text"})
    # 놓아준 뒤에도 결과 조회는 히스토리에서 그대로 된다.
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 200
    client.delete(f"/api/history/{job_id}")


def test_abandoned_prepared_jobs_are_swept(monkeypatch):
    """prepare만 하고 start를 하지 않으면(탭을 닫으면) 레코드와 취소 토큰이 영구히 남는다."""
    monkeypatch.setattr(main, "JOB_RECORD_TTL_MINUTES", 0)
    abandoned = client.post("/api/jobs/prepare").json()["job_id"]
    assert abandoned in main.jobs and agy.is_cancelled(abandoned) is False
    fresh = client.post("/api/jobs/prepare").json()["job_id"]
    assert abandoned not in main.jobs
    assert abandoned not in agy._cancel_events
    main.jobs.pop(fresh, None)
    agy.finish_job(fresh)


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
