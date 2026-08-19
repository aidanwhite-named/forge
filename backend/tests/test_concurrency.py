"""병렬 실행이 지켜야 하는 것들.

이 파이프라인의 산출물은 결정론적이어야 합니다 — 같은 비교 매트릭스에서는 항상 같은 조합과
보고서가 나와야 하고, 그래서 병렬화는 **속도만** 바꿔야 합니다. 여기 있는 테스트는 그 경계를
지킵니다: 순서·상한·취소 전파. 속도 자체는 검사하지 않습니다(느린 CI에서 깨지는 단언이 되므로).
"""
import threading
import time

from app import agy, compare, entailment
from app.models import (Chunk, Claim, ClaimElement, Document, ElementMatch, Limitation,
                        LimitationCheck)
from app.verify import verify_matches

_QUOTE = ("We optimize the global alignment of all local models and generate the final 3D model "
          "in the absolute coordinate frame under positional and geometric constraints.")


def _document(document_id: str = "1") -> Document:
    return Document(id=document_id, filename=f"doc-{document_id}.pdf", type="paper", chunks=[
        Chunk(document_id=document_id, chunk_id=f"D{document_id}-B-p001-01", page=1, text=_QUOTE)])


def _match(document_id: str, limitations: int) -> ElementMatch:
    return ElementMatch(
        claim_number=1, label="E", document_id=document_id, judgment="실질적 동일",
        directness="direct", quote=_QUOTE, chunk_id=f"D{document_id}-B-p001-01",
        limitation_checks=[
            LimitationCheck(index=index, kind="core", limitation=f"한정 {index}",
                            disclosed=True, quote=_QUOTE,
                            chunk_id=f"D{document_id}-B-p001-01")
            for index in range(limitations)])


def _supported(item_id: str) -> dict:
    return {"item_id": item_id, "supported": True, "relation": "explicit",
            "directness": "direct", "reason": "근거가 한정을 뒷받침한다."}


# --- 순서 -----------------------------------------------------------------------

def test_entailment_notes_follow_batch_order_not_completion_order(monkeypatch):
    """느리게 끝난 배치의 노트가 뒤로 밀려서는 안 된다.

    노트 순서가 곧 보고서의 줄 순서다. 완료 순서로 모으면 같은 입력에서 실행마다 다른 보고서가
    나오는데, 그것은 이 파이프라인이 결정론을 내세우는 근거 자체를 무너뜨린다.
    """
    document = _document()
    match = _match("1", 4)                       # 배치 2개(한 배치에 2건)로 갈린다
    verify_matches([match], {"1": document})

    def staggered(prompt, expect="entailments"):
        import json
        payload = json.loads(prompt.split("CONTEXT:\n", 1)[1])
        wanted = [item["item_id"] for item in payload["items"]]
        # 첫 배치를 **더 느리게** 만든다. 완료 순서로 모으면 순서가 뒤집힌다.
        time.sleep(0.20 if wanted[0] == "1:E:0" else 0.01)
        return {"entailments": [
            {**_supported(item_id), "supported": False, "relation": "unsupported",
             "reason": f"{item_id} 기각"} for item_id in wanted]}

    monkeypatch.setattr(entailment, "run_cli", staggered)
    notes = entailment.validate_entailment([match], {"1": document})

    assert [note.split("한정 ")[1][0] for note in notes] == ["0", "1", "2", "3"]


def test_compare_samples_keep_their_submission_order(monkeypatch):
    """표본 순서는 표본 번호다. consensus가 '앞선 두 표본이 일치했는가'를 그 번호로 센다."""
    order: list[int] = []
    delays = iter([0.20, 0.01, 0.05])
    lock = threading.Lock()

    def staggered(prompt, expect="matches"):
        with lock:
            index = len(order)
            order.append(index)
        time.sleep(next(delays))
        return {"matches": [{"label": "A", "sample": index}]}

    monkeypatch.setattr(compare, "run_cli", staggered)
    responses, error = compare._sample("prompt", 3)

    assert error == ""
    # 제출 순서 0,1,2 그대로. 완료 순서라면 1,2,0이 된다.
    assert [response[0]["sample"] for response in responses] == [0, 1, 2]


def test_a_failed_sample_is_dropped_and_the_rest_still_vote(monkeypatch):
    """표본 하나가 실패해도 나머지로 다수결을 낸다. 전부 실패했을 때만 미판정이다."""
    calls = iter([0, 1, 2])

    def flaky(prompt, expect="matches"):
        index = next(calls)
        if index == 1:
            raise RuntimeError("두 번째 표본 실패")
        return {"matches": [{"label": "A", "sample": index}]}

    monkeypatch.setattr(compare, "run_cli", flaky)
    responses, error = compare._sample("prompt", 3)

    assert [response[0]["sample"] for response in responses] == [0, 2]
    assert "두 번째 표본 실패" in error


# --- 상한 -----------------------------------------------------------------------

def test_run_cli_never_exceeds_the_global_slot_limit(monkeypatch):
    """단계별 병렬도를 곱한 만큼 프로세스가 뜨면 provider 한도에 걸린다.

    한도에 걸린 호출은 실패해 재시도가 붙으므로 병렬화의 이득이 그대로 사라진다. 상한은 성능
    제한이 아니라 성능 보호다. 세마포어가 있다는 것이 아니라 **run_cli가 그것을 지나간다는
    것**을 검사해야 하므로 프로세스 생성 자체를 가짜로 세운다.
    """
    live = 0
    peak = 0
    lock = threading.Lock()

    class FakeProcess:
        returncode = 0
        pid = 0

        def poll(self):
            return 0

        def communicate(self, input=None, timeout=None):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.05)
            with lock:
                live -= 1
            return '{"claims": ["ok"]}', ""

    monkeypatch.setattr(agy, "_cli_slots", threading.BoundedSemaphore(2))
    monkeypatch.setattr(agy, "_build_command", lambda *args, **kwargs: ["noop"])
    monkeypatch.setattr(agy.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    outcomes = agy.run_parallel([lambda: agy.run_cli("prompt")] * 8, 8)
    assert [error for _, error in outcomes] == [None] * 8
    assert peak == 2


# --- 취소 -----------------------------------------------------------------------

def test_parallel_workers_inherit_the_job_so_cancellation_reaches_them():
    """워커 스레드는 부모의 job을 물려받지 않는다. 매어 두지 않으면 취소가 닿지 않는다."""
    agy.register_job("job-1")
    agy.bind_job("job-1")
    try:
        seen = agy.run_parallel([lambda: agy.current_job()] * 4, 4)
        assert [value for value, _ in seen] == ["job-1"] * 4
    finally:
        agy.finish_job("job-1")


def test_run_parallel_hands_back_exceptions_instead_of_raising():
    """실패를 다루는 방식은 호출부마다 다르다 — 표본은 버리고, 배치는 노트를 남긴다."""
    def boom():
        raise RuntimeError("실패")

    outcomes = agy.run_parallel([boom, lambda: "ok"], 2)
    assert outcomes[0][0] is None and isinstance(outcomes[0][1], RuntimeError)
    assert outcomes[1] == ("ok", None)


def test_cancellation_in_one_entailment_batch_stops_the_stage(monkeypatch):
    """취소는 실패가 아니다. 삼키면 사용자가 멈춘 뒤에도 다음 단계가 그대로 진행된다."""
    document = _document()
    match = _match("1", 4)
    verify_matches([match], {"1": document})

    def cancelling(prompt, expect="entailments"):
        raise agy.AnalysisCancelled("보고서 생성을 취소했습니다.")

    monkeypatch.setattr(entailment, "run_cli", cancelling)
    try:
        entailment.validate_entailment([match], {"1": document})
        assert False, "취소가 올라오지 않았습니다"
    except agy.AnalysisCancelled:
        pass
