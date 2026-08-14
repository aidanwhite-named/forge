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


def join_worker(job_id: str, timeout: float = 5) -> None:
    worker = main.jobs.get(job_id, {}).get("_worker")
    if worker is not None:
        worker.join(timeout=timeout)


def confirm_decomposition(job_id: str, decomposition: dict | None = None, join: bool = True):
    """분해를 확정해 구성대비를 시작합니다. 확정 전에는 한 셀도 판정하지 않습니다."""
    response = client.post(f"/api/jobs/{job_id}/decomposition/confirm",
                           json={"decomposition": decomposition or {}})
    if join:
        join_worker(job_id)
    return response


def start_job(filename: str = "prior.pdf", content_type: str = "application/pdf",
              confirm: bool = True):
    """준비 → 시작 → (분해 확정) 경로로 작업 하나를 만듭니다.

    시작만으로는 구성대비가 돌지 않습니다. 분해 확정이 실제 분석의 방아쇠라, 프런트엔드가
    쓰는 경로도 두 단계입니다.
    """
    job_id = client.post("/api/jobs/prepare").json()["job_id"]
    response = client.post(
        f"/api/jobs/{job_id}/start",
        data={"claims": "(A) 쓰기 요청을 큐에 저장하는 것"},
        files={"pdf_files": (filename, pdf_bytes(), content_type)},
    )
    join_worker(job_id)                                   # 분해 제안
    if confirm and main.jobs.get(job_id, {}).get("status") == "awaiting_decomposition":
        response = confirm_decomposition(job_id)
    return job_id, response


def post_job():
    job_id, _ = start_job()
    return job_id


def fake_result(job_id, claims_text, documents, analysis_prompt="", progress=None,
                decomposition=None, cache_keys=None, priority_date="",
                pinned_decomposition=None):
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


def test_lifespan_recovers_interrupted_jobs(monkeypatch):
    logs: list[tuple[str, str]] = []
    monkeypatch.setattr(main, "recover_interrupted_jobs", lambda: 1)
    monkeypatch.setattr(main, "jobs", {"job-1": {"status": "interrupted"}})
    monkeypatch.setattr(main, "write_log", lambda job_id, message: logs.append((job_id, message)))

    with TestClient(main.app) as lifespan_client:
        assert lifespan_client.get("/api/health").status_code == 200

    assert logs == [("job-1", "job marked interrupted after server restart")]


def test_completed_job_is_marked_completed(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id, response = start_job()
    assert response.status_code == 202
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "completed"
    client.delete(f"/api/history/{job_id}")


def test_async_job_can_be_cancelled_without_leaving_a_report(monkeypatch):
    entered = threading.Event()

    def slow_result(job_id, claims_text, documents, analysis_prompt="", progress=None,
                    decomposition=None, cache_keys=None, priority_date="",
                    pinned_decomposition=None):
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
    join_worker(job_id)                                   # 분해 제안
    assert confirm_decomposition(job_id, join=False).status_code == 202
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
                              decomposition=None, cache_keys=None, priority_date="",
                              pinned_decomposition=None):
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
    # 종속항 추가는 이미 확정된 분해 위에서 도는 후속 작업이라 확정 단계를 다시 거치지
    # 않습니다. 분해 확정은 최초 분석에만 있는 관문입니다.
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


# --- 분해 확정 관문 --------------------------------------------------------------
# 분해는 한정 문언·검색어를 정하고, 그 둘이 비교 캐시 키와 문헌에서 읽어 올 청크를 좌우한다.
# 확정 전에 구성대비를 시작하면 사람이 고칠 기회를 갖기도 전에 판정이 끝나 있고, 고치는
# 순간 그 판정은 전부 버려진다.

def test_start_stops_at_the_decomposition_and_judges_nothing(monkeypatch):
    """확정 전에는 **한 셀도** 판정하지 않는다."""
    called: list[str] = []
    monkeypatch.setattr(main, "analyze", lambda *args, **kwargs: called.append("analyze"))

    job_id, response = start_job(confirm=False)

    assert response.status_code == 202
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "awaiting_decomposition"
    assert called == []
    main.remove_job_record(job_id)


def test_the_proposal_is_returned_next_to_the_original_claim_text():
    """원문 없이 분해만 보여 주면 확인이 성립하지 않는다 — 모델이 보탠 한정인지 알 수 없다."""
    job_id, _ = start_job(confirm=False)

    payload = client.get(f"/api/jobs/{job_id}/decomposition").json()

    assert payload["status"] == "awaiting_decomposition"
    assert payload["claims_text"] == "(A) 쓰기 요청을 큐에 저장하는 것"
    assert payload["decomposition"]["claims"]["1"][0]["label"] == "A"
    assert payload["confirmed"] is False
    main.remove_job_record(job_id)


def test_the_confirmed_decomposition_is_what_the_analysis_runs_on(monkeypatch):
    """확정본은 공유 캐시와 LLM 재분해를 모두 이겨야 한다. 그러지 않으면 확인이 무의미하다."""
    seen: dict = {}

    def capture(job_id, claims_text, documents, analysis_prompt="", progress=None,
                decomposition=None, cache_keys=None, priority_date="",
                pinned_decomposition=None):
        seen["pinned"] = pinned_decomposition
        return AnalysisResult(job_id=job_id, claim_mapping=[], reports=[], validation=[])

    monkeypatch.setattr(main, "analyze", capture)
    job_id, _ = start_job(confirm=False)
    edited = {"version": "test", "claims": {"1": [{
        "label": "A", "text": "쓰기 요청을 큐에 저장하는 것", "importance": 5, "is_sub": False,
        "search_terms": ["우선순위 큐"],
        "limitations": [{"text": "쓰기 요청을 우선순위 큐에 저장함", "kind": "core",
                         "alternative_group": ""}]}]}}

    confirm_decomposition(job_id, edited)

    assert seen["pinned"]["claims"] == edited["claims"]
    client.delete(f"/api/history/{job_id}")


def test_the_edit_between_proposal_and_confirmation_is_recorded(monkeypatch):
    """무엇을 고쳤는지가 다음 개선의 자료다. 확정본만 남기면 그 사실이 사라진다."""
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id, _ = start_job(confirm=False)
    edited = {"version": "test", "claims": {"1": [{
        "label": "A", "text": "쓰기 요청을 큐에 저장하는 것", "importance": 5, "is_sub": False,
        "search_terms": ["우선순위 큐"],
        "limitations": [{"text": "쓰기 요청을 우선순위 큐에 저장함", "kind": "core",
                         "alternative_group": ""}]}]}}

    confirm_decomposition(job_id, edited)

    review = json.loads((main.HISTORY_DIR / job_id / "decomposition_review.json")
                        .read_text(encoding="utf-8"))
    assert review["edited"] is True
    assert review["edits"][0]["label"] == "A"
    assert review["edits"][0]["after"]["importance"] == 5
    assert review["confirmed"]["claims"] == edited["claims"]
    client.delete(f"/api/history/{job_id}")


def test_confirming_without_edits_keeps_the_proposal(monkeypatch):
    """대부분의 실행은 '이대로 확정' 한 번이다. 그때 전체 분해를 되돌려 보내게 하면 안 된다."""
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id, _ = start_job(confirm=False)

    confirm_decomposition(job_id)

    review = json.loads((main.HISTORY_DIR / job_id / "decomposition_review.json")
                        .read_text(encoding="utf-8"))
    assert review["edited"] is False and review["edits"] == []
    assert review["confirmed"]["claims"] == review["proposed"]["claims"]
    client.delete(f"/api/history/{job_id}")


def test_confirming_twice_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "analyze", fake_result)
    job_id, _ = start_job(confirm=False)
    assert confirm_decomposition(job_id).status_code == 202
    assert client.post(f"/api/jobs/{job_id}/decomposition/confirm", json={}).status_code == 409
    client.delete(f"/api/history/{job_id}")


def test_cancelling_while_waiting_removes_the_staged_upload():
    """확정을 기다리다 취소하면 돌고 있는 워커가 없어 아무도 임시 파일을 지우지 않는다."""
    job_id, _ = start_job(confirm=False)
    work = main.jobs[job_id]["_work"]
    assert work.exists()

    client.delete(f"/api/jobs/{job_id}")

    assert not work.exists()
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    main.remove_job_record(job_id)


def test_a_failed_decomposition_leaves_no_history(monkeypatch):
    """아직 아무것도 쓰지 않은 단계다. 실패했다고 지울 히스토리가 있으면 안 된다."""
    def boom(claims_text):
        raise RuntimeError("청구항을 인식하지 못했습니다.")

    monkeypatch.setattr(main, "propose_decomposition", boom)
    job_id, _ = start_job(confirm=False)

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed" and "인식하지" in job["error"]
    assert not (main.HISTORY_DIR / job_id).exists()
    main.remove_job_record(job_id)


def test_a_rejected_confirmation_keeps_the_job_waiting(monkeypatch):
    """검증에 걸렸다고 상태를 옮기면 오타 하나에 업로드부터 다시 해야 한다."""
    called: list[str] = []
    monkeypatch.setattr(main, "analyze", lambda *args, **kwargs: called.append("analyze"))
    job_id, _ = start_job(confirm=False)
    broken = {"version": "test", "claims": {"1": [{
        "label": "A", "text": "사용자가 바꿔 버린 구성 원문", "importance": 4, "is_sub": False,
        "search_terms": [], "limitations": [{"text": "무엇을 함", "kind": "core",
                                             "alternative_group": ""}]}]}}

    response = client.post(f"/api/jobs/{job_id}/decomposition/confirm",
                           json={"decomposition": broken})

    assert response.status_code == 400
    assert "구성 원문은 고칠 수 없습니다" in response.json()["detail"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "awaiting_decomposition"
    assert called == []
    # 고쳐서 다시 보내면 그대로 진행된다.
    assert confirm_decomposition(job_id).status_code == 202
    main.remove_job_record(job_id)


def test_what_lands_in_claim_elements_is_exactly_what_was_confirmed(monkeypatch):
    """확정본이 실제로 쓰였는지는 claim_elements.json으로만 확인할 수 있다.

    _restore_elements가 조용히 실패하면 파이프라인은 LLM 재분해로 넘어가고, 저장되는 분해는
    사용자가 확인한 것이 아니게 된다. 이 테스트는 실제 assign_importance를 태워 그 경로를
    검사한다 — block_the_real_cli가 켜져 있으므로 재분해로 넘어가면 곧바로 실패한다.
    """
    from app.claims import assign_importance, parse_claims

    def analyze_with_the_real_decomposition(
            job_id, claims_text, documents, analysis_prompt="", progress=None,
            decomposition=None, cache_keys=None, priority_date="", pinned_decomposition=None):
        parsed = parse_claims(claims_text)
        assign_importance(parsed, decomposition, claims_text,
                          pinned_decomposition=pinned_decomposition)
        return AnalysisResult(job_id=job_id, claim_mapping=[], reports=[], validation=[])

    monkeypatch.setattr(main, "analyze", analyze_with_the_real_decomposition)
    job_id, _ = start_job(confirm=False)
    edited = {"version": "test", "claims": {"1": [{
        "label": "A", "text": "쓰기 요청을 큐에 저장하는 것", "importance": 5, "is_sub": False,
        "search_terms": ["우선순위 큐", "priority queue"],
        "limitations": [{"text": "쓰기 요청을 받음", "kind": "core", "alternative_group": ""},
                        {"text": "저장 대상을 우선순위 큐로 한정함", "kind": "qualifier",
                         "alternative_group": ""}]}]}}

    confirm_decomposition(job_id, edited)

    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "completed"
    saved = json.loads((main.HISTORY_DIR / job_id / "claim_elements.json")
                       .read_text(encoding="utf-8"))
    assert saved["claims"] == edited["claims"]
    client.delete(f"/api/history/{job_id}")


def test_only_one_worker_starts_when_confirmations_race(monkeypatch):
    """두 확정이 겹치면 둘 다 awaiting을 읽고 각자 워커를 띄울 수 있다.

    그러면 같은 job에 분석이 두 벌 돌면서 같은 히스토리 디렉터리에 서로의 결과를 덮어쓴다.
    """
    started = threading.Semaphore(0)
    running = threading.Event()

    def slow_analyze(job_id, claims_text, documents, analysis_prompt="", progress=None,
                     decomposition=None, cache_keys=None, priority_date="",
                     pinned_decomposition=None):
        started.release()
        running.wait(timeout=2)
        return AnalysisResult(job_id=job_id, claim_mapping=[], reports=[], validation=[])

    monkeypatch.setattr(main, "analyze", slow_analyze)
    job_id, _ = start_job(confirm=False)

    results: list[int] = []
    barrier = threading.Barrier(2)

    def confirm():
        barrier.wait(timeout=2)
        results.append(client.post(f"/api/jobs/{job_id}/decomposition/confirm",
                                   json={}).status_code)

    racers = [threading.Thread(target=confirm) for _ in range(2)]
    for racer in racers:
        racer.start()
    for racer in racers:
        racer.join(timeout=5)
    running.set()
    join_worker(job_id)

    assert sorted(results) == [202, 409]                 # 하나만 통과한다
    assert started.acquire(blocking=False) is True       # 워커는 한 번만 돌았다
    assert started.acquire(blocking=False) is False
    client.delete(f"/api/history/{job_id}")


def test_a_confirmation_that_loses_to_a_cancel_never_starts_a_worker(monkeypatch):
    """취소가 상태를 옮기는 사이에 확정이 끼어들면 취소된 작업 위에서 워커가 뜬다."""
    called: list[str] = []
    monkeypatch.setattr(main, "analyze", lambda *args, **kwargs: called.append("analyze"))
    job_id, _ = start_job(confirm=False)

    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    response = client.post(f"/api/jobs/{job_id}/decomposition/confirm", json={})

    assert response.status_code == 409
    assert called == []
    main.remove_job_record(job_id)


def test_a_restart_while_waiting_says_the_upload_is_gone():
    """확정 전 재시작은 복구할 것이 없다. 캐시 재사용을 약속하는 문구를 쓰면 사용자는 이어서
    돌 수 있다고 읽고 같은 화면에서 기다린다."""
    job_id, _ = start_job(confirm=False)
    main.jobs.clear()                                    # 서버 재시작과 같은 상태

    assert main.recover_interrupted_jobs() == 1

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "interrupted"
    assert "처음부터 다시 시작" in job["error"] and "저장되기 전" in job["error"]
    main.remove_job_record(job_id)


def test_the_stored_decomposition_is_the_normalised_one(monkeypatch):
    """검증을 통과해도 복원 과정에서 값이 달라진다. 저장본과 분석 입력이 갈리면 안 된다."""
    seen: dict = {}

    def capture(job_id, claims_text, documents, analysis_prompt="", progress=None,
                decomposition=None, cache_keys=None, priority_date="",
                pinned_decomposition=None):
        seen["pinned"] = pinned_decomposition
        return AnalysisResult(job_id=job_id, claim_mapping=[], reports=[], validation=[])

    monkeypatch.setattr(main, "analyze", capture)
    job_id, _ = start_job(confirm=False)
    messy = {"version": "test", "claims": {"1": [{
        "label": "A", "text": "쓰기 요청을 큐에 저장하는 것", "importance": 4, "is_sub": False,
        "search_terms": ["  우선순위   큐 "],
        "limitations": [{"text": " 쓰기 요청을  받음 ;", "kind": "core",
                         "alternative_group": ""}]}]}}

    response = confirm_decomposition(job_id, messy)

    canonical = response.json()["decomposition"]
    element = canonical["claims"]["1"][0]
    assert element["search_terms"] == ["우선순위 큐"]
    # 한정 문언은 다듬지 않는다. 사용자가 확정한 그대로가 캐시 키와 프롬프트로 간다.
    assert element["limitations"][0]["text"] == " 쓰기 요청을  받음 ;"
    # 저장본·분석 입력·응답이 모두 같은 값이어야 한다.
    assert seen["pinned"] == canonical
    review = json.loads((main.HISTORY_DIR / job_id / "decomposition_review.json")
                        .read_text(encoding="utf-8"))
    assert review["confirmed"] == canonical
    client.delete(f"/api/history/{job_id}")


def test_staged_uploads_live_under_the_managed_directory():
    """tempfile 기본 위치에 만들면 서버가 죽었을 때 경로가 메모리와 함께 사라져 아무도 그
    폴더를 찾지 못한다. 폴더 자체는 디스크에 그대로 남는다."""
    job_id, _ = start_job(confirm=False)
    work = main.jobs[job_id]["_work"]

    assert work.parent == main.STAGING_DIR and work.exists()

    client.delete(f"/api/jobs/{job_id}")
    main.remove_job_record(job_id)


def test_orphan_staging_folders_are_swept_at_startup():
    """기동 시점에는 돌고 있는 작업이 없으므로 남은 것은 정의상 전부 고아다."""
    main.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    orphan = main.STAGING_DIR / "left-over-from-a-crash"
    orphan.mkdir()
    (orphan / "1.pdf").write_bytes(b"%PDF-1.4")

    main._sweep_orphan_staging()

    assert not orphan.exists()
    assert main.STAGING_DIR.exists()
