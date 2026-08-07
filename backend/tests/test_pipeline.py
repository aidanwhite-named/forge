"""파이프라인 전체. CLI는 고정 응답으로 대체하고, 그 뒤 단계가 전부 코드인지 확인한다."""
import json

import pytest

from app import agy, cache, claims as claims_module, compare, pipeline, priorart
from app.models import (ChainInfo, Chunk, ClaimResult, Document, DocumentMapping,
                        ElementMatch, EvidenceSpan)
from app.pdf import classify, detect_paragraph_pattern, extract_document_number
from app.report import to_markdown
from app.report import (_closest_related, _difference, _narrative, _reason_clause,
                        _summary_difference, refresh_mappings)

CLAIMS = "전자장치에 있어서, (A) 쓰기 요청을 큐에 저장하는 메모리 컨트롤러; (B) 상태 변경 시 알림을 전송하는 통신부"
QUOTE_A = "메모리 컨트롤러는 데이터 쓰기 요청을 큐에 저장한 후 순차적으로 처리한다."
QUOTE_B = "제어부는 장치 상태 변경 시 사용자 단말로 알림 메시지를 전송한다."


def document(document_id: str, text: str, paragraph: str) -> Document:
    return Document(id=document_id, filename=f"{document_id}.pdf", type="patent",
                    document_number=f"10-2020-000{document_id}",
                    chunks=[Chunk(document_id=document_id, chunk_id=f"D{document_id}-P-{paragraph}",
                                  page=1, paragraph=paragraph, text=text)])


DOCUMENTS = [document("1", QUOTE_A, "0021"), document("2", QUOTE_B, "0012")]

def element(report, label: str):
    """라벨로 구성을 찾는다. 전제부(P0)가 0번이라 위치 인덱스는 쓰지 않는다."""
    return next(item for item in report.claims if item.label == label)


def checks(disclosed: bool, quote: str = "", chunk_id: str = "") -> list[dict]:
    """구성당 하위 제한 1개(중요도 응답에 limitations가 없으면 구성 원문 1개로 잡힌다)."""
    return [{"index": 0, "disclosed": disclosed, "quote": quote, "chunk_id": chunk_id}]


# 전제부(P0)도 판정 대상이므로 응답에 포함되어야 한다. 빠지면 '미판정'으로 처리된다.
RESPONSES = {
    "1": {"matches": [
        {"label": "P0", "judgment": "대응 없음", "directness": "absent", "reason": "전제부 대응 기재 없음",
         "quote": "", "chunk_id": "", "missing_limitations": [], "limitation_checks": checks(False)},
        {"label": "A", "judgment": "실질적 동일", "directness": "direct", "reason": "쓰기 요청 관련 구성이 대응함",
         "quote": QUOTE_A, "chunk_id": "D1-P-0021", "missing_limitations": [],
         "limitation_checks": checks(True, QUOTE_A, "D1-P-0021")},
        {"label": "B", "judgment": "대응 없음", "directness": "absent", "reason": "알림 전송 기재 없음",
         "quote": "", "chunk_id": "", "missing_limitations": ["알림 전송"], "limitation_checks": checks(False)},
    ]},
    "2": {"matches": [
        {"label": "P0", "judgment": "대응 없음", "directness": "absent", "reason": "전제부 대응 기재 없음",
         "quote": "", "chunk_id": "", "missing_limitations": [], "limitation_checks": checks(False)},
        {"label": "A", "judgment": "일부 유사", "directness": "inferred", "reason": "저장 관련 기재만 있음",
         "quote": "", "chunk_id": "", "limitation_checks": checks(False)},
        {"label": "B", "judgment": "동일", "directness": "direct", "reason": "상태 변경 시 알림 전송을 개시함",
         "quote": QUOTE_B, "chunk_id": "D2-P-0012", "missing_limitations": [],
         "limitation_checks": checks(True, QUOTE_B, "D2-P-0012")},
    ]},
}


@pytest.fixture
def stub_cli(monkeypatch):
    """LLM 호출은 중요도·구성대비 단계뿐이며, 그 밖에는 코드가 계산한다."""
    calls: list[str] = []

    def fake(prompt: str, expect: str = "claims"):
        calls.append(expect)
        if expect == "elements":
            return {"elements": [{"claim_number": 1, "label": "P0", "importance": 2},
                                 {"claim_number": 1, "label": "A", "importance": 5},
                                 {"claim_number": 1, "label": "B", "importance": 4}]}
        for document_id, response in RESPONSES.items():
            if f'"id": "{document_id}"' in prompt:
                return response
        raise AssertionError("알 수 없는 비교 요청")

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", fake)
    return calls


def test_pipeline_combines_two_documents_and_calls_the_cli_once_per_cell(stub_cli):
    result = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    # 중요도 1회 + 2×1 비교. 별도의 심사 판단 호출은 수행하지 않습니다.
    assert stub_cli == ["elements", "matches", "matches"]
    report = result.reports[0]
    assert report.track == "inventive_step_combination"
    assert report.chain.primary == "1" and report.chain.secondaries == ["2"]
    assert [item.label for item in report.claims] == ["P0", "A", "B"]
    assert element(report, "A").similarity == 94 and element(report, "A").emoji == "🟢"
    assert element(report, "B").similarity == 99                # 보조 문헌의 동일 판정이 채택됨


def test_an_undisclosed_preamble_is_reported_without_deciding_the_conclusion(stub_cli):
    """전제부가 한정적인지는 법적 판단이라 코드가 정하지 않는다.

    빼놓으면(종전) 제한적 전제부가 인용발명에 없어도 신규성이 부정되고, 하드 게이트로
    삼으면 "…장치에 있어서" 같은 범주 기재 때문에 거의 모든 청구항이 판단 불가가 된다.
    대비·보고는 하되 결론은 막지 않고 쟁점만 드러낸다.
    """
    report = pipeline.analyze("job", CLAIMS, DOCUMENTS).reports[0]
    assert report.chain.uncovered == ["P0"]
    assert report.chain.preamble_undisclosed == ["P0"]
    assert report.track == "inventive_step_combination"          # 결론을 막지는 않는다
    assert element(report, "P0").is_preamble is True
    assert element(report, "P0").similarity is None
    assert "추가 검색 필요" in element(report, "P0").narrative


def test_reference_numbers_follow_the_selected_role_not_upload_order(stub_cli):
    result = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    assert [(mapping.reference_number, mapping.document_id, mapping.role) for mapping in result.claim_mapping] == [
        (1, "1", "주 인용발명"), (2, "2", "보조 인용발명")]


def test_evidence_carries_the_verified_quote_and_its_location(stub_cli):
    result = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    evidence = element(result.reports[0], "B").evidence[0]
    assert evidence.reference_number == 2 and evidence.paragraph == "0012"
    assert evidence.excerpt == QUOTE_B and evidence.quality == "HIGH"
    assert evidence.chunk_id == "D2-P-0012"


def test_narrative_is_one_sentence_with_excerpt_location_and_reason():
    """구성대비는 발췌·인용 위치·판단 이유를 한 문장에 담는다. 판단 이유는 생략할 수 없다."""
    original = "The edge device initiates a virtual encryption session before content arrives."
    translation = "에지 장치는 콘텐츠 도착 전에 가상 암호화 세션을 개시한다."
    document = Document(id="1", filename="prior.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0037", page=5, paragraph="0037", text=original),
    ])
    match = ElementMatch(
        claim_number=1, label="D", document_id="1", judgment="일부 차이", directness="direct",
        quote=original, quote_translation=translation, chunk_id="D1-P-0037", verify="verified",
        reason="콘텐츠와 분리된 세션을 미리 만들어 두고 있음",
    )
    narrative = _narrative(
        "D", "가상 세션을 생성하고 세션 토큰을 발급함", match, match, False,
        [DocumentMapping(reference_number=1, filename="prior.pdf",
                         document_id="1", document_number="US 2023/0362144 A1")],
        {"1": document},
    )

    assert narrative == (
        '인용발명 1 (US 2023/0362144 A1)에는 "에지 장치는 콘텐츠 도착 전에 가상 암호화 세션을 개시한다." '
        '(단락 [0037])("The edge device initiates a virtual encryption session before content arrives.")'
        '는 구성이 기재되어 있으며, 콘텐츠와 분리된 세션을 미리 만들어 두고 있으므로 '
        '청구항의 "가상 세션을 생성하고 세션 토큰을 발급함" 구성과 대응됩니다.'
    )
    assert "\n" not in narrative


def test_an_element_with_no_correspondence_is_flagged_for_further_search():
    """대응 문헌이 없으면 유사도를 붙이지 않고 추가 검색 대상으로 표시한다."""
    match = ElementMatch(claim_number=1, label="B", document_id="1", judgment="대응 없음",
                         directness="absent")
    narrative = _narrative("B", "유효 수요 지표를 산출함", match, match, False, [], {})

    assert narrative == "(B) 구성에 대응되는 인용발명이 확인되지 않음 — 추가 검색 필요"


def test_the_closest_verified_passage_survives_a_no_correspondence_verdict():
    """대응으로 인정하지 않더라도 원문 대조를 통과한 인접 기재는 버리지 않는다.

    감추면 심사관이 이미 확인된 문단을 처음부터 다시 찾게 되고, 어디까지 검토된 상태인지도
    알 수 없게 된다. 등급을 올리지 않고 위치만 함께 남긴다.
    """
    quote = "The request is invalidated after the time-to-live expires."
    documents = {"1": Document(id="1", filename="d1.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0617", page=24, paragraph="0617", text=quote)])}
    match = ElementMatch(
        claim_number=1, label="H", document_id="1", judgment="차이", directness="absent",
        evidence=[EvidenceSpan(chunk_id="D1-P-0617", quote=quote,
                               quote_translation="TTL이 만료되면 요청이 무효화된다.", verify="verified")])
    mappings = [DocumentMapping(reference_number=3, filename="d1.pdf", document_id="1",
                                document_number="US 2003/0097564 A1")]

    related = _closest_related("H", {"1": {"H": match}}, mappings, documents)
    narrative = _narrative("H", "세션 토큰을 파기함", match, match, False, mappings, documents, related)

    assert narrative.splitlines() == [
        "(H) 구성에 대응되는 인용발명이 확인되지 않음 — 추가 검색 필요",
        '(가장 가까운 기재: 인용발명 3 (US 2003/0097564 A1) "TTL이 만료되면 요청이 무효화된다." '
        "(단락 [0617]) — 청구항 한정 전체를 개시하는 근거는 아님)",
    ]


def test_hallucinated_quote_is_downgraded_before_selection(monkeypatch, stub_cli):
    """검증에 실패한 발췌 위에서 결정론적 계산이 돌지 않아야 한다."""
    fabricated = json.loads(json.dumps(RESPONSES))
    # P0가 아니라 실제로 결과를 확인하는 구성 A의 발췌를 변조합니다.
    fabricated["1"]["matches"][1].update(judgment="동일", quote="이 문장은 문헌 어디에도 존재하지 않는 발췌입니다")
    monkeypatch.setitem(RESPONSES, "1", fabricated["1"])
    result = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    item = element(result.reports[0], "A")
    # 검증에 실패한 발췌는 '차이' 이하로 내려가고, '차이'는 대응 구간(80% 이상)에 들지 못한다.
    # 지어낸 문장 위에 유사도 퍼센트를 얹지 않는 것이 이 게이트의 목적이다.
    assert item.similarity is None and item.status == "미개시"
    assert "추가 검색 필요" in item.narrative
    assert any("→" in note or "낮췄" in note for note in result.verify_notes)
    monkeypatch.setitem(RESPONSES, "1", RESPONSES["1"])


def test_second_run_reuses_the_cache_without_calling_the_cli(stub_cli):
    pipeline.analyze("job", CLAIMS, DOCUMENTS)
    stub_cli.clear()
    result = pipeline.analyze("job2", CLAIMS, DOCUMENTS)
    assert stub_cli == ["elements"]                             # 비교 호출 없음
    assert result.cached_claims == [1]


def test_multiple_dependent_claims_use_one_batch_comparison_and_stay_separate(monkeypatch, stub_cli):
    existing = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    calls: list[str] = []

    def batch_cli(prompt: str, expect: str = "claims"):
        calls.append(expect)
        if expect == "elements":
            return {"elements": [
                {"claim_number": 2, "label": "A", "importance": 4,
                 "limitations": ["큐가 우선순위를 가짐"]},
                {"claim_number": 3, "label": "A", "importance": 5,
                 "limitations": ["큐가 순환형임"]},
            ]}
        return {"matches": [
            {"claim_number": claim_number, "document_id": document_id, "label": "A",
             "judgment": "동일" if document_id == "1" else "대응 없음",
             "directness": "direct" if document_id == "1" else "absent",
             "reason": f"청구항 {claim_number}의 별도 판정",
             "quote": QUOTE_A if document_id == "1" else "",
             "chunk_id": "D1-P-0021" if document_id == "1" else "",
             "limitation_checks": [{
                 "index": 0, "disclosed": document_id == "1",
                 "quote": QUOTE_A if document_id == "1" else "",
                 "chunk_id": "D1-P-0021" if document_id == "1" else "",
             }]}
            for claim_number in (2, 3) for document_id in ("1", "2")
        ]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", batch_cli)
    combined = (
        f"【청구항 1】\n{CLAIMS}\n"
        "【청구항 2】\n제1항에 있어서, (A) 큐가 우선순위를 갖는 전자장치\n"
        "【청구항 3】\n제1항에 있어서, (A) 큐가 순환형인 전자장치"
    )
    result = pipeline.extend_with_dependent_claims(existing, combined, {2, 3}, DOCUMENTS)

    assert calls == ["elements", "matches"]  # 중요도 1회 + 구성대비 1회
    assert [report.claim_number for report in result.reports] == [1, 2, 3]
    assert result.reports[1].depends_on == 1 and result.reports[2].depends_on == 1
    assert result.reports[1].claims[0].claim != result.reports[2].claims[0].claim
    assert "큐가 우선순위를 갖는 전자장치" in result.reports[1].claims[0].narrative
    assert "큐가 순환형인 전자장치" in result.reports[2].claims[0].narrative


def test_an_incomplete_batch_response_only_re_asks_the_cells_that_were_missing(monkeypatch, stub_cli):
    """일괄 응답에서 한 칸이 빠졌다고 나머지 온전한 칸까지 다시 물어보면 안 된다.

    한 응답에 (종속항 × 문헌) 수십 칸과 하위 제한 점검 수백 줄을 담아야 하므로 어딘가
    빠지는 일은 드물지 않다. 그때마다 전 칸을 단건으로 다시 받으면 일괄 호출은 늘 헛돈이
    되고 소요 시간은 칸 수에 그대로 비례한다.
    """
    existing = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    calls: list[str] = []
    single_cells: list[str] = []

    def cell(claim_number: int, document_id: str) -> dict:
        return {"claim_number": claim_number, "document_id": document_id, "label": "A",
                "judgment": "동일", "directness": "direct", "reason": "큐 구성을 개시함",
                "quote": QUOTE_A if document_id == "1" else QUOTE_B,
                "chunk_id": f"D{document_id}-P-{'0021' if document_id == '1' else '0012'}",
                "limitation_checks": checks(True, QUOTE_A if document_id == "1" else QUOTE_B,
                                            f"D{document_id}-P-{'0021' if document_id == '1' else '0012'}")}

    def partial_batch(prompt: str, expect: str = "claims"):
        calls.append(expect)
        if expect == "elements":
            return {"elements": [{"claim_number": 2, "label": "A", "importance": 4},
                                 {"claim_number": 3, "label": "A", "importance": 5}]}
        if '"claims"' in prompt:
            # 4칸 중 (청구항 3 × 문헌 2) 한 칸만 빠뜨린 응답.
            return {"matches": [cell(2, "1"), cell(2, "2"), cell(3, "1")]}
        single_cells.append(prompt)
        return {"matches": [{"label": "A", "judgment": "동일", "directness": "direct",
                             "reason": "순환형 큐를 개시함", "quote": QUOTE_B,
                             "chunk_id": "D2-P-0012",
                             "limitation_checks": checks(True, QUOTE_B, "D2-P-0012")}]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", partial_batch)
    combined = (f"【청구항 1】\n{CLAIMS}\n"
                "【청구항 2】\n제1항에 있어서, (A) 큐가 우선순위를 갖는 전자장치\n"
                "【청구항 3】\n제1항에 있어서, (A) 큐가 순환형인 전자장치")
    result = pipeline.extend_with_dependent_claims(existing, combined, {2, 3}, DOCUMENTS)

    # 일괄 1회 + 빠진 한 칸만 단건 1회. 예전에는 여기서 네 칸을 전부 다시 물어봤다.
    assert calls == ["elements", "matches", "matches"]
    assert len(single_cells) == 1 and '"claim_number": 3' in single_cells[0]
    assert [report.claim_number for report in result.reports] == [1, 2, 3]
    assert all(report.track != "analysis_incomplete" for report in result.reports)


def test_dependent_cells_do_not_resend_the_whole_document(monkeypatch, stub_cli):
    """종속항 행렬에는 "에 있어서" 뒤의 추가 한정만 들어 있다.

    한 줄짜리 한정을 판정하려고 문헌 전문을 실으면 같은 문헌을 종속항 수만큼 다시 읽히게
    된다. 이번 실측에서는 문헌 3건(14.5만 자)이 종속항 7개에 걸쳐 101만 자로 불어났다.
    """
    budgets: list[int | None] = []
    real_select = compare.select_chunks

    def record(claim, document, budget=None):
        budgets.append(budget)
        return real_select(claim, document, budget)

    monkeypatch.setattr(compare, "select_chunks", record)
    existing = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    assert budgets == [None, None]                   # 독립항은 문헌 전문을 그대로 쓴다

    budgets.clear()
    monkeypatch.setattr(compare, "compare_claims_documents",
                        lambda claims, documents, guideline="": ({}, ["일괄 실패"]))
    combined = f"【청구항 1】\n{CLAIMS}\n【청구항 2】\n제1항에 있어서, (A) 큐가 우선순위를 갖는 전자장치"
    pipeline.extend_with_dependent_claims(existing, combined, {2}, DOCUMENTS)
    assert budgets == [compare.DEPENDENT_DOCUMENT_BUDGET_CHARS] * 2


def test_a_failed_batch_call_falls_back_to_per_cell_comparison(monkeypatch, stub_cli):
    """일괄 호출 한 번의 실패로 종속항 **전부**가 판정을 잃어서는 안 된다.

    일괄 프롬프트는 모든 종속항과 모든 문헌을 함께 싣기 때문에 길이 초과 한 번이
    보고서 전체를 '판정 불가'로 만든다. 속도를 위한 최적화가 결과를 못 내는 쪽으로
    기울면 안 되므로, 실패하면 셀 단위로 다시 물어본다.
    """
    existing = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    calls: list[str] = []

    def failing_batch(prompt: str, expect: str = "claims"):
        calls.append(expect)
        if expect == "elements":
            return {"elements": [{"claim_number": 2, "label": "A", "importance": 4}]}
        if '"claims"' in prompt:                       # 일괄 경로만 실패시킨다
            raise RuntimeError("agy CLI의 response 필드가 JSON이 아닙니다")
        return {"matches": [
            {"label": "A", "judgment": "동일", "directness": "direct", "reason": "우선순위 큐를 개시함",
             "quote": QUOTE_A, "chunk_id": "D1-P-0021",
             "limitation_checks": checks(True, QUOTE_A, "D1-P-0021")}]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", failing_batch)
    combined = (f"【청구항 1】\n{CLAIMS}\n"
                "【청구항 2】\n제1항에 있어서, (A) 큐가 우선순위를 갖는 전자장치")
    result = pipeline.extend_with_dependent_claims(existing, combined, {2}, DOCUMENTS)

    report = result.reports[1]
    assert calls == ["elements", "matches", "matches", "matches"]  # 일괄 1회 실패 + 셀 2회
    assert report.track != "analysis_incomplete"
    assert report.claims[0].similarity == 99


def test_changing_the_guideline_invalidates_the_cache(stub_cli):
    pipeline.analyze("job", CLAIMS, DOCUMENTS, "기본 지침")
    stub_cli.clear()
    pipeline.analyze("job2", CLAIMS, DOCUMENTS, "다른 지침")
    assert stub_cli.count("matches") == 2


def test_markdown_carries_only_the_comparison_itself(stub_cli):
    """보고서에는 구성대비만 남는다. 내부 선정 지표와 도구 동작 로그는 감사 JSON으로 간다."""
    markdown = to_markdown(pipeline.analyze("job", CLAIMS, DOCUMENTS))
    assert "| 인용발명 1 | 10-2020-0001 | 1.pdf | - | 주 인용발명 |" in markdown
    assert "**인용발명 조합**: 인용발명 1 (10-2020-0001) + 인용발명 2 (10-2020-0002)" in markdown
    assert "### (A) 쓰기 요청을 큐에 저장하는 메모리 컨트롤러" in markdown
    assert "유사도: 94% 🟢 실질적 동일" in markdown
    assert "단락 [0021]" in markdown
    for noise in ("판정 대표값", "검토 트랙", "결합 후 구성대비 지표", "주지관용",
                  "결합에 채택하지 않은 대응", "주 인용발명 개시:", "결합 후 남는 차이:",
                  "단독 적합도"):
        assert noise not in markdown


def test_the_report_states_which_rejection_the_claim_faces(stub_cli):
    """구성별 유사도만 늘어놓고 결론을 적지 않으면 읽는 사람이 표에서 결론을 추정하게 된다."""
    result = pipeline.analyze("job", CLAIMS, DOCUMENTS)

    assert result.reports[0].conclusion.startswith("진보성 검토 (인용발명 결합) — ")
    assert "**결론**: 진보성 검토 (인용발명 결합)" in to_markdown(result)


def test_a_novelty_conclusion_says_when_it_rests_on_equivalence(monkeypatch, stub_cli):
    """신규성 부정이 문언 그대로의 개시가 아니라 등가 판단에 서 있으면 그렇다고 적는다.

    '실질적 동일'은 용어가 다른 것을 같다고 본 **판단**이다. 이를 밝히지 않으면 보고서가
    문언이 그대로 있었던 경우와 구별되지 않는 모습으로 나가고, 정작 다투어야 할 등가
    여부가 검토 대상에서 빠진다.
    """
    preamble_quote = "본 실시예의 전자장치는 메모리 컨트롤러와 통신부를 포함한다."
    single = document("1", f"{preamble_quote} {QUOTE_A} {QUOTE_B}", "0021")
    everything = {"matches": [
        {"label": "P0", "judgment": "실질적 동일", "directness": "direct", "reason": "전자장치를 개시함",
         "quote": preamble_quote, "chunk_id": "D1-P-0021", "missing_limitations": [],
         "limitation_checks": checks(True, preamble_quote, "D1-P-0021")},
        {"label": "A", "judgment": "실질적 동일", "directness": "direct", "reason": "쓰기 요청을 큐에 저장함",
         "quote": QUOTE_A, "chunk_id": "D1-P-0021", "missing_limitations": [],
         "limitation_checks": checks(True, QUOTE_A, "D1-P-0021")},
        {"label": "B", "judgment": "실질적 동일", "directness": "direct", "reason": "상태 변경 시 알림을 전송함",
         "quote": QUOTE_B, "chunk_id": "D1-P-0021", "missing_limitations": [],
         "limitation_checks": checks(True, QUOTE_B, "D1-P-0021")},
    ]}
    monkeypatch.setattr(compare, "run_cli",
                        lambda prompt, expect="claims": (
                            {"elements": []} if expect == "elements" else everything))

    report = pipeline.analyze("job", CLAIMS, [single]).reports[0]

    assert report.track == "novelty_single"
    assert report.conclusion.startswith("신규성 없음 (단일 인용발명) — ")
    assert "구성 A, B은 문언 그대로의 개시가 아니라 '실질적 동일'" in report.conclusion
    # 전제부는 한정 여부 자체가 미정이라 등가 확인 대상으로 세우지 않는다.
    assert "P0" not in report.conclusion


def test_each_limitation_keeps_the_sentence_that_proved_it(monkeypatch):
    """대표 발췌 한 문장만 남기면 어느 한정을 무엇으로 개시했는지가 보고서에서 사라진다."""
    decomposition = {"elements": [
        {"claim_number": 1, "label": "A", "importance": 5, "limitations": [
            {"text": "쓰기 요청을 큐에 저장함", "kind": "core"},
            {"text": "저장된 요청을 순차적으로 처리함", "kind": "qualifier"},
        ]},
    ]}
    matches = {"matches": [
        {"label": "P0", "judgment": "대응 없음", "directness": "absent", "quote": "",
         "chunk_id": "", "limitation_checks": checks(False)},
        {"label": "A", "judgment": "실질적 동일", "directness": "direct", "reason": "큐에 저장함",
         "quote": QUOTE_A, "chunk_id": "D1-P-0021", "missing_limitations": [],
         "limitation_checks": [
             {"index": 0, "disclosed": True, "quote": QUOTE_A, "chunk_id": "D1-P-0021"},
             {"index": 1, "disclosed": True, "quote": QUOTE_A, "chunk_id": "D1-P-0021"}]},
        {"label": "B", "judgment": "대응 없음", "directness": "absent", "quote": "",
         "chunk_id": "", "limitation_checks": checks(False)},
    ]}
    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli",
                            lambda prompt, expect="claims": (
                                decomposition if expect == "elements" else matches))

    result = pipeline.analyze("job", CLAIMS, [DOCUMENTS[0]])
    markdown = to_markdown(result)

    assert "근거:" in markdown
    assert f'- (core · 쓰기 요청을 큐에 저장함) "{QUOTE_A}" (단락 [0021])' in markdown
    assert f'- (qualifier · 저장된 요청을 순차적으로 처리함) "{QUOTE_A}" (단락 [0021])' in markdown
    # 구성 원문 한 줄을 통째로 점검한 셀은 한정별 근거가 아니므로 반복해 적지 않는다.
    assert markdown.count("근거:") == 1


def test_summary_states_the_common_ground_and_the_sharpest_difference(stub_cli):
    report = pipeline.analyze("job", CLAIMS, DOCUMENTS).reports[0]
    assert report.summary_similarity.startswith("청구항과 인용발명 1, 인용발명 2는 ")
    assert report.summary_similarity.endswith("기술적 목적과 핵심 메커니즘이 공통됩니다.")
    assert "\n" not in report.summary_similarity
    assert element(report, "B").adopted_reference == 2


def test_a_gap_that_no_document_fills_is_named_in_the_summary_difference():
    results = [
        ClaimResult(label="A", claim="재생 요청을 수신함", similarity=94, status="개시됨",
                    adopted_reference=1),
        ClaimResult(label="B", claim="유효 수요 지표를 산출함", similarity=None, status="미개시"),
    ]
    difference = _summary_difference(ChainInfo(claim_number=1, primary="1",
                                               track="inventive_step_combination"), results)

    assert "구성 B" in difference and "추가 검색" in difference


def test_supplement_is_written_as_one_combined_sentence():
    """주 인용발명에 없는 부분을 보완 인용발명이 채우면, 두 문헌을 한 문장에 이어서 적는다."""
    primary_quote = "제어부는 센서 온도를 기준값과 비교하여 냉각팬 구동 여부를 결정한다."
    documents = {"1": document("1", primary_quote, "0034"), "2": document("2", QUOTE_B, "0012")}
    primary = ElementMatch(claim_number=1, label="B", document_id="1", judgment="일부 차이",
                           directness="direct", quote=primary_quote, chunk_id="D1-P-0034",
                           verify="verified", missing_limitations=["알림 전송"])
    adopted = ElementMatch(claim_number=1, label="B", document_id="2", judgment="동일",
                           directness="direct", quote=QUOTE_B, chunk_id="D2-P-0012",
                           verify="verified", reason="상태 변경 시 알림을 보내고 있음")
    mappings = [DocumentMapping(reference_number=1, filename="1.pdf", document_id="1",
                                document_number="10-2020-0001"),
                DocumentMapping(reference_number=2, filename="2.pdf", document_id="2",
                                document_number="10-2020-0002")]

    narrative = _narrative("B", "냉각팬 구동 여부 결정 및 알림 전송", adopted, primary, True,
                           mappings, documents)
    difference = _difference(adopted, primary, True, mappings, documents)

    assert narrative == (
        f'인용발명 1 (10-2020-0001)에는 "{primary_quote}" (단락 [0034])는 구성이 기재되어 있으나 '
        f'알림 전송에 대한 기재는 없고, 인용발명 2 (10-2020-0002)에는 "{QUOTE_B}" (단락 [0012])는 '
        '구성이 기재되어 있어 이를 결합하면 청구항의 "냉각팬 구동 여부 결정 및 알림 전송" 구성과 대응됩니다.'
    )
    assert difference == ("인용발명 1 (10-2020-0001)은 알림 전송에 대한 기재가 없으나 "
                          "인용발명 2 (10-2020-0002) (단락 [0012])의 결합으로 해소됨")


def test_uncovered_elements_are_offered_for_prior_art_search(monkeypatch, stub_cli):
    only_partial = {"1": {"matches": [
        {"label": "P0", "judgment": "대응 없음", "directness": "absent", "quote": "", "chunk_id": "",
         "limitation_checks": checks(False)},
        {"label": "A", "judgment": "동일", "directness": "direct", "quote": QUOTE_A, "chunk_id": "D1-P-0021",
         "limitation_checks": checks(True, QUOTE_A, "D1-P-0021")},
        {"label": "B", "judgment": "대응 없음", "directness": "absent", "quote": "", "chunk_id": "",
         "limitation_checks": checks(False)}]}}
    monkeypatch.setattr(compare, "run_cli",
                        lambda prompt, expect="claims": (
                            {"elements": []} if expect == "elements" else only_partial["1"]))
    result = pipeline.analyze("job", CLAIMS, [DOCUMENTS[0]])
    assert result.reports[0].track == "rejection_impossible"
    assert [target["label"] for target in pipeline.uncovered_elements(result, CLAIMS)] == ["B"]


def test_residual_limitation_is_offered_for_prior_art_search():
    from app.models import AnalysisResult, ChainInfo, ClaimReport, ElementCoverage

    result = AnalysisResult(job_id="job", claim_mapping=[], reports=[ClaimReport(
        claim_number=1,
        chain=ChainInfo(
            claim_number=1,
            residual=["B"],
            element_coverage=[ElementCoverage(
                label="B",
                residual_difference=[
                    "상태 변경 시 관리자 단말에 즉시 알림을 전송하는 조건",
                    "일부 차이 판정에 그쳐 하위 한정까지 동일하다고 보기 어렵습니다",
                ],
            )],
        ),
    )])

    targets = pipeline.uncovered_elements(result, CLAIMS)

    assert targets == [{
        "claim_number": 1,
        "label": "B",
        "text": "상태 변경 시 관리자 단말에 즉시 알림을 전송하는 조건",
        "claim_context": "상태 변경 시 알림을 전송하는 통신부",
    }]


def test_prior_art_hit_keeps_the_claim_number(monkeypatch):
    monkeypatch.setattr(priorart, "run_cli", lambda prompt, expect="hits": {"hits": [{
        "claim_number": 12,
        "label": "A",
        "document_number": "US 2024/0123456 A1",
        "title": "Spline Camera Path Rendering",
        "published": "2024-01-01",
        "correspondence": "B-스플라인 카메라 경로를 개시함",
        "remaining_difference": "동적 시야각 변경은 미개시",
        "url": "https://example.com/patent",
    }]})

    hits, warnings = priorart.search([{"claim_number": 12, "label": "A", "text": "B-스플라인 경로"}])

    assert warnings == []
    assert hits[0].claim_number == 12 and hits[0].label == "A"


def test_document_classification_and_number_extraction():
    numbered = "\n".join(f"[{index:04d}] 이 단락은 발명의 구성을 설명한다." for index in range(1, 12))
    assert classify(numbered, detect_paragraph_pattern(numbered)) == "patent"
    assert classify("Abstract Introduction References DOI 10.1000/x") == "paper"
    assert extract_document_number("United States Patent US 11,456,887 B1") == "US 11,456,887 B1"
    assert extract_document_number("공개특허 10-2020-0012345 공보") == "10-2020-0012345"


def test_cache_key_changes_with_the_model(monkeypatch):
    from app.claims import parse_claims
    claim = parse_claims(CLAIMS)[0]
    base = {"provider": "agy", "model": "m1", "prompt": ""}
    monkeypatch.setattr(cache, "load_runtime_settings", lambda: base)
    first = cache.cache_key(claim, DOCUMENTS[0])
    monkeypatch.setattr(cache, "load_runtime_settings", lambda: {**base, "model": "m2"})
    assert cache.cache_key(claim, DOCUMENTS[0]) != first


def test_cache_reuses_entries_that_contain_the_removed_similarity_field():
    """구성대비 사실은 같으므로 기존 캐시를 버리고 LLM을 다시 부르지 않는다."""
    key = "legacy-with-similarity-point"
    (cache.CACHE_DIR / f"{key}.json").write_text(json.dumps([{
        "claim_number": 1,
        "label": "A",
        "document_id": "1",
        "judgment": "동일",
        "directness": "direct",
        "reason": "기술 구성이 대응함",
        "similarity_point": "과거 요약 전용 값",
        "quote": QUOTE_A,
    }], ensure_ascii=False), encoding="utf-8")

    loaded = cache.load(key)

    assert loaded is not None and loaded[0].judgment == "동일"
    assert not hasattr(loaded[0], "similarity_point")


def test_reason_clause_drops_temporary_numbers_and_connects_to_the_conclusion():
    """모델이 쓴 임시 문헌 번호는 확정 매핑 번호와 충돌하므로 지운다.

    이유는 "…므로 청구항의 … 구성과 대응됩니다"에 이어 붙으므로 어미도 함께 맞춘다.
    """
    assert _reason_clause("인용발명 1에는 동기화 기술이 개시됨") == "동기화 기술이 개시되므로"
    assert _reason_clause("인용문헌 2에는 병합 단계만 개시된다") == "병합 단계만 개시되므로"
    assert _reason_clause("큐에 저장해 순차 처리하고 있으므로") == "큐에 저장해 순차 처리하고 있으므로"
    assert _reason_clause("토큰을 검증합니다") == "토큰을 검증하므로"
    # 이유는 생략할 수 없는 항목이므로, 비어 있을 때도 문장이 끊기지 않게 중립 문구로 잇는다.
    assert _reason_clause("").endswith("므로")


def test_refresh_mappings_keeps_numbers_and_updates_newly_adopted_role():
    from app.models import ChainInfo, DocumentMapping
    mappings = [
        DocumentMapping(reference_number=1, filename="1.pdf", document_id="1", role="주 인용발명"),
        DocumentMapping(reference_number=2, filename="2.pdf", document_id="2", role="미채택"),
    ]
    chains = [ChainInfo(claim_number=1, primary="1"),
              ChainInfo(claim_number=2, primary="1", secondaries=["2"], added="2")]

    refreshed = refresh_mappings(mappings, chains)

    assert [mapping.reference_number for mapping in refreshed] == [1, 2]
    assert refreshed[1].role == "보조 인용발명"


# --- 종속항 대비의 속도와 중단 안전성 -------------------------------------------

def test_missing_cells_are_compared_in_parallel_but_reassembled_in_order(monkeypatch, stub_cli):
    """셀은 서로를 참조하지 않고 시간의 거의 전부가 CLI 응답 대기다.

    직렬로 돌면 (종속항 × 문헌) 수만큼 그대로 곱해진다. 실측 로그에서 21셀이 셀당 22초로
    7분 반이 걸렸다. 다만 병렬로 돌리더라도 매트릭스는 제출 순서대로 다시 모아야, 완료
    순서가 실행마다 달라져도 같은 보고서가 나온다.
    """
    import threading

    existing = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    running = 0
    peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)

    def slow_cell(prompt: str, expect: str = "claims"):
        nonlocal running, peak
        if expect == "elements":
            return {"elements": [{"claim_number": 2, "label": "A", "importance": 4}]}
        if '"claims"' in prompt:
            return {"matches": []}                     # 일괄은 한 칸도 채우지 못한다
        with lock:
            running += 1
            peak = max(peak, running)
        barrier.wait()                                 # 두 셀이 실제로 동시에 떠야 통과한다
        with lock:
            running -= 1
        quote = QUOTE_A if '"id": "1"' in prompt else QUOTE_B
        chunk = "D1-P-0021" if '"id": "1"' in prompt else "D2-P-0012"
        return {"matches": [{"label": "A", "judgment": "동일", "directness": "direct",
                             "reason": "큐 구성을 개시함", "quote": quote, "chunk_id": chunk,
                             "limitation_checks": checks(True, quote, chunk)}]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", slow_cell)
    combined = f"【청구항 1】\n{CLAIMS}\n【청구항 2】\n제1항에 있어서, (A) 큐가 우선순위를 갖는 전자장치"
    result = pipeline.extend_with_dependent_claims(existing, combined, {2}, DOCUMENTS)

    assert peak == 2
    report = result.reports[1]
    assert report.claim_number == 2 and report.track != "analysis_incomplete"
    # 매트릭스는 제출 순서(문헌 1 → 2)대로 모인다.
    assert [item.document_id for item in report.chain.candidates] == ["1", "2"]


def test_a_stored_decomposition_keeps_the_comparison_cache_valid(monkeypatch, stub_cli):
    """분해를 저장해 두지 않으면 취소 후 재시도가 처음부터 다시 돈다.

    같은 청구항을 다시 분해하면 문장이 조금 달라지고, 그 문장이 캐시 키에 들어 있으므로
    이미 받아 둔 셀 판정이 전부 미스가 된다. 실제 캐시 디렉터리에 같은 (청구항 1, 문헌 1)
    항목이 31개 쌓여 있었던 이유다.
    """
    import copy

    base = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    drift = iter(["큐에 저장함", "큐에다 저장함", "큐에 담아 둠"])
    cells: list[str] = []

    def drifting(prompt: str, expect: str = "claims"):
        if expect == "elements":
            return {"elements": [{"claim_number": 2, "label": "A", "importance": 4,
                                  "limitations": [{"text": next(drift), "kind": "core"}]}]}
        if '"claims"' in prompt:
            return {"matches": []}
        cells.append(prompt)
        quote = QUOTE_A if '"id": "1"' in prompt else QUOTE_B
        chunk = "D1-P-0021" if '"id": "1"' in prompt else "D2-P-0012"
        return {"matches": [{"label": "A", "judgment": "동일", "directness": "direct",
                             "reason": "큐 구성을 개시함", "quote": quote, "chunk_id": chunk,
                             "limitation_checks": checks(True, quote, chunk)}]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", drifting)
    combined = f"【청구항 1】\n{CLAIMS}\n【청구항 2】\n제1항에 있어서, (A) 큐가 우선순위를 갖는 전자장치"

    store: dict = {}
    pipeline.extend_with_dependent_claims(copy.deepcopy(base), combined, {2}, DOCUMENTS,
                                          decomposition=store)
    assert len(cells) == 2                                        # 문헌 2건을 처음 판정

    # 같은 분해를 물려주면 캐시가 그대로 맞아 셀을 한 번도 다시 부르지 않는다.
    pipeline.extend_with_dependent_claims(copy.deepcopy(base), combined, {2}, DOCUMENTS,
                                          decomposition=store)
    assert len(cells) == 2

    # 분해를 물려주지 않으면 같은 청구항인데도 키가 달라져 전부 다시 판정한다.
    pipeline.extend_with_dependent_claims(copy.deepcopy(base), combined, {2}, DOCUMENTS)
    assert len(cells) == 4


def test_a_cancelled_extension_keeps_the_claims_it_already_judged(monkeypatch, stub_cli):
    """취소했다고 몇 분치 판정을 통째로 버리면, 다시 눌렀을 때 같은 항을 또 대비한다.

    판정이 모두 모인 항만 보고서에 올린다. 절반만 대비된 항을 넣으면 아직 읽지도 않은
    문헌을 그 항에 대해 "대응 없음"으로 단정하게 된다.
    """
    existing = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    saved: list[list[int]] = []

    def cancel_on_claim_three(prompt: str, expect: str = "claims"):
        if expect == "elements":
            return {"elements": [{"claim_number": 2, "label": "A", "importance": 4},
                                 {"claim_number": 3, "label": "A", "importance": 5}]}
        if '"claims"' in prompt:
            return {"matches": []}
        if '"claim_number": 3' in prompt:
            raise agy.AnalysisCancelled("보고서 생성을 취소했습니다.")
        quote = QUOTE_A if '"id": "1"' in prompt else QUOTE_B
        chunk = "D1-P-0021" if '"id": "1"' in prompt else "D2-P-0012"
        return {"matches": [{"label": "A", "judgment": "동일", "directness": "direct",
                             "reason": "큐 구성을 개시함", "quote": quote, "chunk_id": chunk,
                             "limitation_checks": checks(True, quote, chunk)}]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", cancel_on_claim_three)
    combined = (f"【청구항 1】\n{CLAIMS}\n"
                "【청구항 2】\n제1항에 있어서, (A) 큐가 우선순위를 갖는 전자장치\n"
                "【청구항 3】\n제1항에 있어서, (A) 큐가 순환형인 전자장치")

    with pytest.raises(agy.AnalysisCancelled):
        pipeline.extend_with_dependent_claims(
            existing, combined, {2, 3}, DOCUMENTS,
            checkpoint=lambda partial: saved.append([r.claim_number for r in partial.reports]))

    assert saved == [[1, 2]]                    # 판정이 끝난 2항까지 저장하고 3항은 남기지 않는다
    assert [report.claim_number for report in existing.reports] == [1, 2]


def test_the_initial_analysis_compares_documents_in_parallel_without_extra_calls(monkeypatch, stub_cli):
    """문헌마다 소요 시간의 거의 전부가 CLI 응답 대기라, 직렬로 돌면 문헌 수만큼 곱해진다.

    병렬화는 **호출 횟수도 프롬프트도 바꾸지 않는다.** 셀당 1회 그대로이므로 토큰 비용은
    같고 대기 시간만 줄어든다. 대신 완료 순서가 실행마다 달라지므로, 매트릭스는 반드시
    (청구항, 문헌) 순서로 다시 모아야 같은 보고서가 나온다.
    """
    import threading
    import time

    baseline = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    assert stub_cli.count("matches") == 2
    cache.clear()

    running = 0
    peak = 0
    lock = threading.Lock()
    sequential = compare.run_cli

    def out_of_order(prompt: str, expect: str = "claims"):
        nonlocal running, peak
        if expect == "elements":
            return sequential(prompt, expect)
        with lock:
            running += 1
            peak = max(peak, running)
        # 먼저 제출한 문헌 1을 늦게 끝내 제출 순서와 완료 순서를 뒤집는다.
        time.sleep(0.20 if '"id": "1"' in prompt else 0.02)
        with lock:
            running -= 1
        return sequential(prompt, expect)

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", out_of_order)
    shuffled = pipeline.analyze("job", CLAIMS, DOCUMENTS)

    assert peak == 2                                    # 두 문헌이 실제로 동시에 떴다
    assert stub_cli.count("matches") == 4               # 셀당 1회 그대로. 호출이 늘지 않는다
    assert shuffled.model_dump() == baseline.model_dump()
