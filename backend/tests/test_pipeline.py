"""파이프라인 전체. CLI는 고정 응답으로 대체하고, 그 뒤 단계가 전부 코드인지 확인한다."""
import json

import pytest

from app import agy, cache, claims as claims_module, compare, pipeline, priorart
from app.chain import build_chain
from app.models import (ChainInfo, Chunk, Claim, ClaimElement, ClaimReport, ClaimResult, Document,
                        DocumentMapping, ElementCoverage, ElementMatch, EvidenceSpan,
                        LimitationCheck, SupplementCandidate)
from app.claims import parse_claims
from app.compare import DOCUMENT_BUDGET_CHARS
from app.pdf import classify, detect_paragraph_pattern, extract_document_number
from app.report import to_markdown
from app.report import pipeline_invariants, report_invariants
from app.report import (_closest_related, _difference, _narrative, _reason_clause,
                        _summary_difference, _summary_similarity, build_claim_report,
                        build_mappings, refresh_mappings)

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
        # 등급은 코드가 산출하므로 스텁도 judgment가 아니라 terminology를 넘긴다.
        # 한정이 전부 개시된 상태에서 identical이면 "동일"이 산출된다.
        {"label": "B", "terminology": "identical", "directness": "direct",
         "reason": "상태 변경 시 알림 전송을 개시함",
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
    assert element(report, "A").emoji == "🟢" and element(report, "A").corresponded
    assert element(report, "A").disclosed_limitations == 1 and element(report, "A").total_limitations == 1
    assert element(report, "B").grade == "동일"                 # 보조 문헌의 동일 판정이 채택됨


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
    assert element(report, "P0").corresponded is False
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

    # '일부 차이'는 부분 대응이므로 문장도 부분 대응으로 끝난다. 전부 "대응됩니다"로 끝내면
    # 모델이 이유에 "…부분은 개시되어 있지 않다"고 적은 경우 "…개시되어 있지 않으므로 …
    # 대응됩니다"라는 자기모순 문장이 그대로 보고서에 나간다.
    assert narrative == (
        '인용발명 1 (US 2023/0362144 A1)에는 "에지 장치는 콘텐츠 도착 전에 가상 암호화 세션을 개시한다." '
        '(단락 [0037])("The edge device initiates a virtual encryption session before content arrives.")'
        '는 구성이 기재되어 있으며, 콘텐츠와 분리된 세션을 미리 만들어 두고 있으므로 '
        '청구항의 "가상 세션을 생성하고 세션 토큰을 발급함" 구성과 부분적으로 대응됩니다.'
    )
    assert "\n" not in narrative

    full = _narrative(
        "D", "가상 세션을 생성하고 세션 토큰을 발급함",
        match.model_copy(update={"judgment": "실질적 동일"}), match, False,
        [DocumentMapping(reference_number=1, filename="prior.pdf",
                         document_id="1", document_number="US 2023/0362144 A1")],
        {"1": document},
    )
    assert full.endswith('구성과 대응됩니다.')


def test_an_element_with_no_correspondence_is_flagged_for_further_search():
    """대응 문헌이 없으면 정량 지표를 붙이지 않고 추가 검색 대상으로 표시한다."""
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
    # 지어낸 문장 위에 정량 지표를 얹지 않는 것이 이 게이트의 목적이다.
    assert item.corresponded is False and item.status == "미개시"
    assert "추가 검색 필요" in item.narrative
    assert any("→" in note or "낮췄" in note for note in result.verify_notes)
    monkeypatch.setitem(RESPONSES, "1", RESPONSES["1"])


def test_report_keeps_the_comparison_when_no_document_qualifies_as_primary():
    """조합을 세우지 못했다는 이유로 구성대비 본문을 비우지 않는다.

    주 인용발명 자격 게이트는 차별적 핵심 구성(중요도 4 이상)의 **직접** 개시량만 본다.
    핵심 구성이 '일부 유사·inferred'에 머물면 그 값이 전 문헌에서 0이 되어 조합은 서지
    않지만, 문헌별 대비 결과는 그대로 남아 있다. 종전에는 채택 문헌 목록이 비었다는 이유로
    본문을 통째로 비워, 원문 대조를 통과한 '실질적 동일·direct' 대응까지 "대응되는 인용발명이
    확인되지 않음 — 추가 검색 필요"로 나갔다. 구성대비를 수행하고도 하지 않은 것처럼 보고한
    셈이고, 그 진술은 출원인에게 유리한 방향이라 검토에서 이의가 제기되지도 않는다.
    """
    target = Claim(number=1, elements=[
        ClaimElement(label="A", text="쓰기 요청을 큐에 저장하는 메모리 컨트롤러", importance=2),
        ClaimElement(label="B", text="상태 변경 시 알림을 전송하는 통신부", importance=5)])
    matrix = {"1": {
        "A": ElementMatch(claim_number=1, label="A", document_id="1", judgment="실질적 동일",
                          directness="direct", quote=QUOTE_A, chunk_id="D1-P-0021",
                          verify="verified", reason="큐에 저장한 뒤 순차 처리하므로"),
        "B": ElementMatch(claim_number=1, label="B", document_id="1", judgment="일부 유사",
                          directness="inferred", quote=QUOTE_B, chunk_id="D1-P-0021",
                          verify="verified", reason="알림 전송이 기재되어 있으므로"),
    }}
    chain = build_chain(target, matrix, {}, [target])
    assert chain.primary is None and chain.track == "rejection_impossible"

    mappings = build_mappings([DOCUMENTS[0]], [chain])
    report = build_claim_report(target, chain, matrix, {"1": DOCUMENTS[0]}, mappings)
    disclosed = element(report, "A")
    assert disclosed.corresponded is True and disclosed.status == "개시됨"
    assert disclosed.adopted_reference == 1
    assert "추가 검색 필요" not in disclosed.narrative
    assert disclosed.evidence                        # 근거 발췌가 본문에 남는다
    assert element(report, "B").status == "부분 개시"
    assert "대응되는 기술 내용이 확인되지 않았습니다" not in report.summary_similarity
    # 채택은 하지 않았다는 사실은 그대로 드러나야 한다.
    assert mappings[0].role == "미채택"
    assert "채택된 인용발명 없음" in to_markdown(pipeline.AnalysisResult(
        job_id="job", claim_mapping=mappings, reports=[report]))


def test_second_run_reuses_the_cache_without_calling_the_cli(stub_cli):
    """같은 입력이면 LLM을 **한 번도** 부르지 않는다.

    종전에는 분해만 다시 불렀다("elements"). 그런데 분해 결과가 비교 캐시 키에 들어가므로,
    다시 분해해서 문장이 한 글자라도 달라지면 그 뒤 판정 캐시가 전량 미스가 된다. 분해까지
    입력 해시로 캐시해야 "같은 입력이면 같은 결과"가 실제로 성립한다.
    """
    pipeline.analyze("job", CLAIMS, DOCUMENTS)
    stub_cli.clear()
    result = pipeline.analyze("job2", CLAIMS, DOCUMENTS)
    assert stub_cli == []
    assert result.cached_claims == [1]


def test_force_redecompose_bypasses_the_decomposition_cache(monkeypatch, stub_cli):
    """분해 프롬프트를 손볼 때는 캐시를 무시하고 다시 분해할 수 있어야 한다."""
    pipeline.analyze("job", CLAIMS, DOCUMENTS)
    stub_cli.clear()
    monkeypatch.setattr(claims_module, "FORCE_REDECOMPOSE", True)
    pipeline.analyze("job2", CLAIMS, DOCUMENTS)
    assert stub_cli == ["elements"]                             # 분해만 다시, 비교는 캐시


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
            {"label": "A", "terminology": "identical", "directness": "direct",
             "reason": "우선순위 큐를 개시함",
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
    assert report.claims[0].grade == "동일" and report.claims[0].corresponded


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
    assert "한정 1/1 개시 · 🟢 실질적 동일" in markdown
    assert "단락 [0021]" in markdown
    for noise in ("판정 대표값", "검토 트랙", "결합 후 구성대비 지표", "주지관용",
                  "결합에 채택하지 않은 대응", "주 인용발명 개시:", "결합 후 남는 차이:",
                  "단독 적합도"):
        assert noise not in markdown


def test_the_report_states_which_rejection_the_claim_faces(stub_cli):
    """구성별 등급만 늘어놓고 결론을 적지 않으면 읽는 사람이 표에서 결론을 추정하게 된다."""
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
    # 근거 줄에는 출처 인용발명 번호가 반드시 함께 붙는다. 구성 하나의 근거 목록에는 결합된
    # 여러 문헌의 발췌가 섞이므로, 번호가 없으면 바로 위 구성대비 문장이 지목한 문헌이
    # 목록 전체의 출처인 것처럼 읽힌다.
    assert f'- (core · 쓰기 요청을 큐에 저장함) "{QUOTE_A}" (인용발명 1 · 단락 [0021])' in markdown
    assert (f'- (qualifier · 저장된 요청을 순차적으로 처리함) "{QUOTE_A}" '
            "(인용발명 1 · 단락 [0021])") in markdown
    # 구성 원문 한 줄을 통째로 점검한 셀은 한정별 근거가 아니므로 반복해 적지 않는다.
    assert markdown.count("근거:") == 1
    # 발췌가 한정 문언을 그대로 담고 있으면(explicit) 판단 줄을 덧붙이지 않는다.
    assert "발췌 문언 그대로는 아니며" not in markdown


def test_a_disclosure_bridged_by_semantic_review_says_so_next_to_the_excerpt():
    """발췌 문언 그대로가 아니라 의미검증의 판단으로 인정한 개시는 그 판단을 함께 적는다.

    의미검증은 발췌 한 문장이 아니라 그 문장이 속한 청크 원문과 형제 한정의 인용문까지 읽고
    판단하는데, 보고서에 찍히는 것은 짧은 대표 발췌 하나뿐이다. 그래서 실측 보고서에서
    "HoloNet은 sRGB 이미지를 입력으로 받는다"가 "균일도 보정 이미지를 획득함"의 근거로
    제시됐다. 인정의 실제 근거는 같은 청크의 광원 강도 보정 서술이었지만 보고서에는 없었고,
    발췌만 읽은 심사관에게는 도구가 개시를 잘못 인정한 것으로 보일 수밖에 없다.
    """
    limitation = "쓰기 요청을 큐에 저장함"
    note = "버퍼에 적재하는 구성이 큐 저장과 역할이 같음"
    target = Claim(number=1, elements=[
        ClaimElement(label="A", text=limitation, importance=5)])
    match = ElementMatch(
        claim_number=1, label="A", document_id="1", judgment="실질적 동일",
        directness="direct", quote=QUOTE_A, chunk_id="D1-P-0021", verify="verified",
        limitation_checks=[LimitationCheck(
            index=0, limitation=limitation, kind="core", disclosed=True,
            quote=QUOTE_A, chunk_id="D1-P-0021", verify="verified",
            semantic_status="accepted", semantic_relation="functional_equivalent",
            semantic_note=note)])
    matrix = {"1": {"A": match}}
    chain = build_chain(target, matrix, {}, [target])
    mappings = build_mappings([DOCUMENTS[0]], [chain])
    report = build_claim_report(target, chain, matrix, {"1": DOCUMENTS[0]}, mappings)
    result = pipeline.AnalysisResult(job_id="job", claim_mapping=mappings, reports=[report])

    proof = next(value for value in element(report, "A").evidence if value.kind == "core")
    assert proof.semantic_relation == "기능적 동등으로 인정"
    assert proof.semantic_note == note
    markdown = to_markdown(result)

    assert ("  ↳ 발췌 문언 그대로는 아니며 기능적 동등으로 인정 — "
            "버퍼에 적재하는 구성이 큐 저장과 역할이 같음") in markdown


def test_summary_states_the_common_ground_and_the_sharpest_difference(stub_cli):
    report = pipeline.analyze("job", CLAIMS, DOCUMENTS).reports[0]
    summary = report.summary_similarity
    # 문헌 번호는 **대표 구성이 대응된 문헌**만 적는다. 대응된 모든 구성의 채택 문헌을 합쳐
    # 적으면, 그 구성을 개시하지 않은 문헌까지 공통점의 주어가 된다(구성 B는 인용발명 2가
    # 대응했고 대표 구성 A는 인용발명 1이 대응했다).
    assert summary.startswith("청구항과 인용발명 1은 ")
    assert "인용발명 2" not in summary
    assert "\n" not in summary
    # 유사점은 한 줄 요약이다. 구성 원문을 이어 붙이면 "…단계 및에 관한"처럼 연결어미에서
    # 문장이 끊기고, 이미 위에 구성별로 적힌 내용을 다시 나열하는 것에 그친다.
    assert " 및에" not in summary and " 및 " not in summary
    assert summary.count("구성에서 공통되며") == 1
    # 무엇이 공통인지(가장 중요한 대응 구성)와 어디까지 공통인지(대응 범위)를 함께 적는다.
    assert "쓰기 요청을 큐에 저장하는" in summary
    assert "청구항의 구성 2개 전부에 대응 기재가 확인됩니다" in summary
    assert element(report, "B").adopted_reference == 2


def test_a_gap_that_no_document_fills_is_named_in_the_summary_difference():
    results = [
        ClaimResult(label="A", claim="재생 요청을 수신함", corresponded=True, status="개시됨",
                    adopted_reference=1),
        ClaimResult(label="B", claim="유효 수요 지표를 산출함", status="미개시"),
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


def test_a_limitation_only_supplement_is_counted_and_marked_as_a_combination():
    """대표 셀이 주 문헌에 남아도 한정을 댄 부 문헌은 결합으로 표시되어야 한다."""
    limitation = "출력 이미지를 도파관 광학 시스템을 통해 출력함"
    quote_1 = "장치는 출력 이미지를 획득하여 표시한다."
    quote_2 = "회절 도파관은 광학 이미지를 출력한다."
    target = Claim(number=1, elements=[
        ClaimElement(label="A", text="이미지를 획득하여 도파관으로 출력함", importance=5)])
    primary = ElementMatch(
        claim_number=1, label="A", document_id="1", judgment="일부 차이",
        directness="direct", quote=quote_1, chunk_id="D1-P-0001", verify="verified",
        missing_limitations=[limitation], limitation_checks=[
            LimitationCheck(index=0, limitation="출력 이미지를 획득함", kind="core",
                            disclosed=True, quote=quote_1, chunk_id="D1-P-0001",
                            verify="verified"),
            LimitationCheck(index=1, limitation=limitation, kind="qualifier", disclosed=False),
        ])
    secondary = ElementMatch(
        claim_number=1, label="A", document_id="2", judgment="차이",
        directness="inferred", quote=quote_2, chunk_id="D2-P-0002", verify="verified",
        limitation_checks=[
            LimitationCheck(index=0, limitation=limitation, kind="qualifier", disclosed=True,
                            quote=quote_2, chunk_id="D2-P-0002", verify="verified")])
    matrix = {"1": {"A": primary}, "2": {"A": secondary}}
    chain = build_chain(target, matrix, {}, [target])
    documents = [document("1", quote_1, "0001"), document("2", quote_2, "0002")]
    mappings = build_mappings(documents, [chain])
    report = build_claim_report(target, chain, matrix,
                                {item.id: item for item in documents}, mappings)
    item = element(report, "A")

    assert chain.secondaries == ["2"]
    assert item.combination is True
    assert (item.disclosed_limitations, item.total_limitations) == (2, 2)
    assert "인용발명 2" in (item.difference or "") and "결합으로 해소" in item.difference
    candidate = next(value for value in chain.element_coverage[0].candidates
                     if value.document_id == "2")
    assert candidate.adopted is True and candidate.gain > 0.0


def test_a_remaining_limitation_names_the_unadopted_document_that_discloses_it():
    """남은 한정을 미채택 문헌이 개시했다면 차이점 줄에 그 사실을 함께 적는다.

    적지 않으면 "→ 차이점: 길 안내 정보를 제공함"만 남아 추가 검색 대상으로 읽히는데, 그
    기재는 이미 업로드된 문헌 안에 있다. 채택 여부는 거절 이유를 어떻게 세울지의 문제이지
    문헌에 기재가 있느냐의 문제가 아니므로 둘을 같은 문장에 뭉치지 않는다.

    **빠진 이유는 지어내지 않는다.** 상한이 실제로 걸렸을 때만 상한 탓으로 적고, 자리가
    남아 있었다면 그렇게 적는다(report._unadopted_note).
    """
    documents = {"1": document("1", QUOTE_A, "0025")}
    match = ElementMatch(claim_number=1, label="P0", document_id="1", judgment="일부 차이",
                         directness="direct", quote=QUOTE_A, chunk_id="D1-P-0025",
                         verify="verified", missing_limitations=["길 안내 정보를 제공함"])
    mappings = [DocumentMapping(reference_number=1, filename="1.pdf", document_id="1",
                                document_number="10-2020-0001"),
                DocumentMapping(reference_number=3, filename="3.pdf", document_id="3",
                                document_number="US 2009/0005961 A1")]

    assert _difference(match, None, False, mappings, documents) == "길 안내 정보를 제공함"
    assert _difference(match, None, False, mappings, documents,
                       {"길 안내 정보를 제공함": ["3"]}, "결합 문헌 수 상한(2건)을 넘어") == (
        "길 안내 정보를 제공함 (인용발명 3 (US 2009/0005961 A1)에 대응 기재가 있으나 "
        "결합 문헌 수 상한(2건)을 넘어 이 거절 이유에는 세우지 않음)")
    assert _difference(match, None, False, mappings, documents,
                       {"길 안내 정보를 제공함": ["3"]}, "보완 후보 평가에서 채택되지 않아") == (
        "길 안내 정보를 제공함 (인용발명 3 (US 2009/0005961 A1)에 대응 기재가 있으나 "
        "보완 후보 평가에서 채택되지 않아 이 거절 이유에는 세우지 않음)")


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


def test_a_residual_limitation_already_found_in_an_over_limit_document_is_not_searched():
    """이미 손에 든 문헌에 있는 기재를 웹에서 다시 찾지 않는다. 구성 전체가 그럴 때와 같다."""
    from app.models import AnalysisResult, ChainInfo, ClaimReport, ElementCoverage

    limitation = "상태 변경 시 관리자 단말에 즉시 알림을 전송하는 조건"
    result = AnalysisResult(job_id="job", claim_mapping=[], reports=[ClaimReport(
        claim_number=1,
        chain=ChainInfo(
            claim_number=1,
            residual=["B"],
            beyond_limit_residual={"B": {limitation: ["3"]}},
            element_coverage=[ElementCoverage(
                label="B",
                residual_difference=[
                    limitation,
                    "일부 차이 판정에 그쳐 하위 한정까지 동일하다고 보기 어렵습니다",
                ],
            )],
        ),
    )])

    # 남은 한정이 그것뿐이었으므로 이 구성은 검색 대상에서 통째로 빠진다.
    assert pipeline.uncovered_elements(result, CLAIMS) == []


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

    hits = priorart.search([{"claim_number": 12, "label": "A", "text": "B-스플라인 경로"}])

    assert hits[0].claim_number == 12 and hits[0].label == "A"


def test_prior_art_search_failure_is_raised_not_returned_as_zero_hits(monkeypatch):
    """CLI 실패를 빈 결과로 돌려주면 호출부가 그것을 '0건'으로 저장해 버린다."""
    def failing_cli(prompt, expect="hits"):
        raise RuntimeError("exit code 1: provider unreachable")

    monkeypatch.setattr(priorart, "run_cli", failing_cli)

    with pytest.raises(priorart.SearchFailed, match="선행기술 검색에 실패했습니다"):
        priorart.search([{"claim_number": 1, "label": "A", "text": "쓰기 요청"}])


def test_prior_art_cancellation_is_not_mistaken_for_a_failure(monkeypatch):
    """AnalysisCancelled도 RuntimeError라 SearchFailed로 뭉개지기 쉽다."""
    def cancelling_cli(prompt, expect="hits"):
        raise agy.AnalysisCancelled("보고서 생성을 취소했습니다.")

    monkeypatch.setattr(priorart, "run_cli", cancelling_cli)

    with pytest.raises(agy.AnalysisCancelled):
        priorart.search([{"claim_number": 1, "label": "A", "text": "쓰기 요청"}])


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


def test_cache_key_separates_the_narrow_dependent_context_from_the_full_one(monkeypatch):
    """좁은 문맥에서 나온 판정이 넓은 문맥으로 볼 경로에서 재사용되면 안 된다.

    같은 (청구항 × 문헌)이라도 최초 분석은 문헌 전문에 가까운 예산으로, 종속항 추가는 그보다
    훨씬 좁은 예산으로 판정한다. 예산을 키에서 빼면 좁은 문맥에서 "대응 없음"으로 떨어진 셀이
    넓은 문맥의 실행에서 그대로 재사용되어, 문헌에 기재가 있는데도 없다는 판정이 고착된다.
    """
    from app.claims import parse_claims
    from app.compare import DEPENDENT_DOCUMENT_BUDGET_CHARS, DOCUMENT_BUDGET_CHARS
    claim = parse_claims(CLAIMS)[0]
    monkeypatch.setattr(cache, "load_runtime_settings",
                        lambda: {"provider": "agy", "model": "m1", "prompt": ""})

    full = cache.cache_key(claim, DOCUMENTS[0], "", DOCUMENT_BUDGET_CHARS)
    narrow = cache.cache_key(claim, DOCUMENTS[0], "", DEPENDENT_DOCUMENT_BUDGET_CHARS)

    assert full != narrow


def test_cache_key_changes_with_the_parent_claim_text(monkeypatch):
    """같은 문언의 종속항이라도 부모항이 다르면 "상기 …"의 대상이 달라 판정이 달라진다."""
    from app.claims import parse_claims
    claim = parse_claims(CLAIMS)[0]
    monkeypatch.setattr(cache, "load_runtime_settings",
                        lambda: {"provider": "agy", "model": "m1", "prompt": ""})

    without = cache.cache_key(claim, DOCUMENTS[0], "", 0, [])
    with_parent = cache.cache_key(claim, DOCUMENTS[0], "", 0,
                                  [{"claim_number": 1, "preamble": "장치에 있어서,",
                                    "elements": [{"label": "A", "text": "이미지를 수신하는 입력부"}]}])

    assert without != with_parent


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

    # 분해를 물려주지 않아도 입력 해시 캐시가 같은 분해를 돌려주므로 키가 유지된다.
    # 종전에는 여기서 청구항 문언이 그대로인데도 분해가 흔들려 전부 다시 판정했다(4).
    pipeline.extend_with_dependent_claims(copy.deepcopy(base), combined, {2}, DOCUMENTS)
    assert len(cells) == 2

    # 강제 재분해를 켜면 그 안전장치가 풀리고 분해가 다시 흔들린다.
    monkeypatch.setattr(claims_module, "FORCE_REDECOMPOSE", True)
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


def test_summary_does_not_claim_an_undisclosed_limitation_as_common_ground():
    """부분 개시 구성을 대표로 쓸 때는 "부분적으로 공통"이라고 적는다.

    부분 개시 구성의 문언에는 개시되지 않은 한정까지 들어 있다. 그것을 "모두 …에서
    공통된다"고 적으면 정작 없는 개시를 공통점으로 단언하게 된다.
    """
    claim = Claim(number=1, elements=[
        ClaimElement(label="A", text="제1 반사부재와 제2 반사부재 사이에 배치되는 광원", importance=5),
        ClaimElement(label="B", text="영상을 출력하는 디스플레이부", importance=3),
    ])
    partial_only = [
        ClaimResult(label="A", claim=claim.elements[0].text, corresponded=True,
                    status="부분 개시", adopted_reference=1),
    ]
    summary = _summary_similarity(claim, partial_only)
    assert "부분적으로 공통되며" in summary

    # 완전 개시 구성이 하나라도 있으면 중요도가 낮아도 그쪽을 대표로 삼는다.
    mixed = partial_only + [
        ClaimResult(label="B", claim=claim.elements[1].text, corresponded=True,
                    status="개시됨", adopted_reference=1),
    ]
    summary = _summary_similarity(claim, mixed)
    assert "영상을 출력하는 디스플레이부" in summary
    assert "부분적으로" not in summary


def test_closest_passage_prefers_the_document_that_came_nearest():
    """미대응 구성의 "가장 가까운 기재"는 번호 순이 아니라 근접한 정도로 고른다.

    대표 발췌만 있고 보조 발췌가 없는 셀을 건너뛰면, 정작 그 구성을 가장 잘 개시한 문헌이
    사라지고 무관한 문헌의 총론 문장이 남는다. 강등되기 전 판정으로 재지 않으면 상한에
    걸린 문헌들이 같은 등급으로 납작해져 번호 순서가 승부를 가른다.
    """
    boilerplate = "실시예들은 하나 이상의 컴퓨팅 장치와 관련하여 수행될 수 있다."
    on_point = "사용자단말기는 통신모듈을 통하여 ID를 전송받고 서버에 관련 컨텐츠를 요청한다."
    documents = {
        "1": Document(id="1", filename="a.pdf", chunks=[
            Chunk(document_id="1", chunk_id="D1-P-0039", page=7, paragraph="0039", text=boilerplate)]),
        "2": Document(id="2", filename="b.pdf", chunks=[
            Chunk(document_id="2", chunk_id="D2-P-0151", page=13, paragraph="0151", text=on_point)]),
    }
    matrix = {
        "1": {"B": ElementMatch(claim_number=5, label="B", document_id="1", judgment="차이",
                                downgraded_from="일부 유사", directness="direct", quote=boilerplate,
                                chunk_id="D1-P-0039", verify="verified")},
        "2": {"B": ElementMatch(claim_number=5, label="B", document_id="2", judgment="차이",
                                downgraded_from="실질적 동일", directness="direct", quote=on_point,
                                chunk_id="D2-P-0151", verify="verified")},
    }
    mappings = [DocumentMapping(reference_number=1, filename="a.pdf", document_id="1"),
                DocumentMapping(reference_number=2, filename="b.pdf", document_id="2")]

    related = _closest_related("B", matrix, mappings, documents)

    assert "인용발명 2" in related and "단락 [0151]" in related
    assert "컴퓨팅 장치" not in related


# --- 문헌 축 일괄 구성대비 -------------------------------------------------------
# 문헌 본문이 청구항 수만큼 반복해 실리는 것이 이 파이프라인 토큰 비용의 대부분입니다.
# 문헌 축으로 묶으면 그 반복이 사라지지만, 응답 하나에 담기는 판정 수가 늘어 셀이 빠질
# 위험이 커집니다. 아래 두 테스트가 지키는 것은 "빠진 셀이 조용히 사라지지 않는다"입니다.

_BATCH_CLAIMS = ("【청구항 1】 요청을 큐에 저장하는 저장부를 포함하는 장치.\n"
                 "【청구항 2】 제1항에 있어서, 상기 큐를 우선순위로 정렬하는 정렬부.")
_BATCH_QUOTE = "The controller stores the request in a processing queue by priority order."
_BATCH_DOCUMENTS = [Document(id="1", filename="prior.pdf", chunks=[
    Chunk(document_id="1", chunk_id="D1-P-0001", page=1, paragraph="0001", text=_BATCH_QUOTE)])]


def _batch_match(number: int) -> dict:
    return {"claim_number": number, "label": "A", "directness": "direct", "quote": _BATCH_QUOTE,
            "chunk_id": "D1-P-0001", "terminology": "equivalent", "different_purpose": False,
            "limitation_checks": [{"index": 0, "disclosed": True, "quote": _BATCH_QUOTE,
                                   "chunk_id": "D1-P-0001"}]}


def _batch_importance() -> dict:
    return {"elements": [{"claim_number": number, "label": "A", "importance": 5}
                         for number in (1, 2)]}


def _enable_batch(monkeypatch, size: int = 2) -> None:
    monkeypatch.setattr(pipeline, "COMPARE_CLAIM_BATCH", size)


def test_document_batch_sends_the_document_once_for_all_claims(monkeypatch):
    """문헌 축으로 묶으면 문헌 본문이 청구항 수만큼 반복 전송되지 않는다."""
    carrying_document = []

    def fake(prompt: str, expect: str = "claims"):
        if expect == "elements":
            return _batch_importance()
        if _BATCH_QUOTE in prompt:
            carrying_document.append(prompt)
        return {"matches": [_batch_match(1), _batch_match(2)]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", fake)
    _enable_batch(monkeypatch)

    result = pipeline.analyze("job", _BATCH_CLAIMS, _BATCH_DOCUMENTS)

    # 청구항 축이었다면 문헌 본문이 청구항마다 한 번씩, 즉 두 번 실렸다.
    assert len(carrying_document) == 1
    assert [report.claim_number for report in result.reports] == [1, 2]


def test_a_claim_missing_from_the_batch_reply_falls_back_to_a_single_call(monkeypatch):
    """일괄 응답에서 빠진 청구항은 미판정으로 흘리지 않고 단건으로 다시 받는다."""
    modes: list[str] = []

    def fake(prompt: str, expect: str = "claims"):
        if expect == "elements":
            return _batch_importance()
        if '"claims":' in prompt:                 # 문헌 축 일괄 프롬프트
            modes.append("batch")
            return {"matches": [_batch_match(1)]}  # 청구항 2를 일부러 뺀다
        modes.append("single")
        return {"matches": [_batch_match(2)]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", fake)
    _enable_batch(monkeypatch)

    result = pipeline.analyze("job", _BATCH_CLAIMS, _BATCH_DOCUMENTS)

    assert modes == ["batch", "single"]
    assert [report.claim_number for report in result.reports] == [1, 2]
    assert all(report.chain.track != "analysis_incomplete" for report in result.reports)


def test_batch_and_single_cells_are_cached_under_different_keys(monkeypatch):
    """일괄로 받은 셀을 단건 키로 저장하면 두 프롬프트의 판정이 한 파일에서 섞인다."""
    stored: list[str] = []

    def fake(prompt: str, expect: str = "claims"):
        if expect == "elements":
            return _batch_importance()
        return {"matches": [_batch_match(1), _batch_match(2)]}

    for module in (compare, claims_module):
        monkeypatch.setattr(module, "run_cli", fake)
    monkeypatch.setattr(cache, "store", lambda key, cell: stored.append(key))
    _enable_batch(monkeypatch)

    pipeline.analyze("job", _BATCH_CLAIMS, _BATCH_DOCUMENTS)

    claim = parse_claims(_BATCH_CLAIMS)[0]
    single = cache.cache_key(claim, _BATCH_DOCUMENTS[0], "", DOCUMENT_BUDGET_CHARS, [],
                             mode="single", samples=1)
    assert stored and single not in stored


# --- 선행기술 적격성 분류 ---------------------------------------------------------
# 날짜는 오래 "나열만" 되었습니다. 그래서 대상 우선일 이후에 나온 문헌도, 후공개 선출원도
# 통상 선행기술과 같은 칸에 들어갔습니다. 여기서도 자동 탈락은 시키지 않습니다 — 적격성은
# 법역과 신규성·진보성 구분까지 봐야 정해지고 날짜 추출 실패도 흔하기 때문입니다.

def _dated(document_id: str, published: str = "", filed: str = "") -> Document:
    return Document(id=document_id, filename=f"D{document_id}.pdf",
                    publication_date=published, filing_date=filed,
                    chunks=[Chunk(document_id=document_id, chunk_id=f"D{document_id}-P-0001",
                                  text="본문")])


def test_a_document_published_after_the_priority_date_is_named_as_a_later_document():
    from app import eligibility
    category, detail = eligibility.classify_document(
        _dated("1", published="2023-01-01", filed="2022-12-01"), "2022-06-01")
    assert category == eligibility.LATER
    assert "선행기술로 쓸 수 없습니다" in detail


def test_a_secret_prior_application_is_not_filed_next_to_ordinary_prior_art():
    """공개는 뒤지만 출원이 앞선 문헌은 신규성 근거로만 쓸 수 있어 칸을 나눠야 한다."""
    from app import eligibility
    category, detail = eligibility.classify_document(
        _dated("1", published="2023-01-01", filed="2021-05-01"), "2022-06-01")
    assert category == eligibility.SECRET_PRIOR_APPLICATION
    assert "신규성 근거로만" in detail and "진보성" in detail


def test_missing_dates_are_unknown_rather_than_eligible_or_ineligible():
    """추출 실패를 적격으로 흘리면 없는 근거 위에 거절이 서고, 부적격으로 흘리면 문헌이 사라진다."""
    from app import eligibility
    category, _ = eligibility.classify_document(_dated("1"), "2022-06-01")
    assert category == eligibility.UNKNOWN


def test_eligibility_is_reported_but_never_filters_the_matrix(monkeypatch, stub_cli):
    """분류는 보고서에 남기되 문헌을 선정에서 빼지는 않는다."""
    later = Document(id="2", filename="later.pdf", publication_date="2030-01-01",
                     filing_date="2029-01-01", chunks=DOCUMENTS[1].chunks)
    result = pipeline.analyze("job", CLAIMS, [DOCUMENTS[0], later],
                              priority_date="2022-06-01")
    assert any("선행기술로 쓸 수 없습니다" in line for line in result.validation)
    # 그래도 판정 자체는 수행되어 매트릭스에 남는다.
    assert any(row.document_id == "2"
               for coverage in result.reports[0].chain.element_coverage
               for row in coverage.candidates)


def test_an_unparseable_priority_date_suspends_classification_instead_of_guessing():
    from app import eligibility
    lines = eligibility.warnings([_dated("1", published="2020-01-01")], "2022년 6월")
    assert len(lines) == 1 and "형식을 인식하지 못했습니다" in lines[0]


def test_a_combination_that_leaves_a_gap_does_not_claim_it_was_resolved():
    """본문의 "결합으로 해소됨"과 결론의 "차이가 남습니다"가 같은 구성을 두고 어긋나면 안 된다.

    실측 보고서에서 구성 (B)가 정확히 그랬다. 본문은 주 인용발명의 누락만 보고 "해소됨"을
    적었고, 결론은 채택 셀의 누락을 보고 잔존 차이로 적었다. 실제로는 두 한정 중 하나를
    어느 문헌도 메우지 못한 상태였다.
    """
    mappings = [DocumentMapping(document_id="1", filename="primary.pdf", reference_number=1),
                DocumentMapping(document_id="2", filename="supplement.pdf", reference_number=2)]
    documents = {"2": Document(id="2", filename="supplement.pdf", chunks=[
        Chunk(document_id="2", chunk_id="D2-P-0007", page=3, paragraph="0007", text="근거 원문")])}
    primary = ElementMatch(claim_number=1, label="B", document_id="1", judgment="일부 차이",
                           directness="direct", quote="주 인용발명 발췌", chunk_id="D1-P-0001",
                           verify="verified",
                           missing_limitations=["트레이닝 세트로 한정함", "도파관 모델링으로 한정함"])
    # 보완 문헌이 앞의 한정만 메우고 "도파관 모델링"은 그대로 남긴 경우.
    adopted = ElementMatch(claim_number=1, label="B", document_id="2", judgment="일부 차이",
                           directness="direct", quote="보완 발췌", chunk_id="D2-P-0007",
                           verify="verified", missing_limitations=["도파관 모델링으로 한정함"])

    line = _difference(adopted, primary, True, mappings, documents)
    assert "도파관 모델링으로 한정함" in line and "남음" in line
    assert not line.endswith("결합으로 해소됨")

    # 보완 문헌이 공백을 전부 메운 경우에는 종전 문장 그대로.
    adopted.missing_limitations = []
    assert _difference(adopted, primary, True, mappings, documents).endswith("결합으로 해소됨")


def test_an_antecedent_cap_does_not_swallow_the_remaining_limitations():
    """지시 관계 상한 문장이 실제 누락 한정을 대신해서는 안 된다.

    종전 구현은 antecedent_note가 있으면 곧바로 반환했다. "이 경우 하위 한정은 전부 개시로
    남아 있다(예: 2/2)"를 전제한 것인데, 실측에서 1/5인 셀이 나왔다. 그러면 보고서는
    "선행 구성이 없어 완전 개시로 보지 않았다"만 적고 실제로 빠진 네 한정은 한 줄도 남기지
    않아, 읽는 사람이 집계와 차이점 줄을 대조할 수 없게 된다.
    """
    mappings = [DocumentMapping(document_id="1", filename="d.pdf", reference_number=1)]
    documents = {"1": Document(id="1", filename="d.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", page=1, text="근거")])}
    match = ElementMatch(
        claim_number=1, label="C", document_id="1", judgment="일부 유사", directness="direct",
        quote="발췌", chunk_id="D1-P-0001", verify="verified",
        antecedent_note="같은 인용발명에서 구성 B의 대응이 확인되지 않아, 이를 참조하는 이 구성을 "
                        "완전 개시로 보지 않았습니다",
        missing_limitations=["학습된 뉴럴 네트워크에 입력될 이미지로 한정함",
                             "타겟 균일도 이미지의 출력을 목적으로 한정함"])

    line = _difference(match, None, False, mappings, documents)
    assert "구성 B의 대응이 확인되지 않아" in line          # 상한 사유는 그대로 남고
    assert "학습된 뉴럴 네트워크에 입력될 이미지로 한정함" in line   # 누락 한정도 함께 나온다
    assert "타겟 균일도 이미지의 출력을 목적으로 한정함" in line

    # 누락 한정이 없으면 종전처럼 상한 사유만 적는다.
    match.missing_limitations = []
    assert _difference(match, None, False, mappings, documents) == match.antecedent_note


def test_an_antecedent_bridge_does_not_hide_inferred_directness():
    """선행 구성 보완 뒤에도 해당 셀의 '추론 대응' 차이는 보고서에 남아야 한다."""
    match = ElementMatch(
        claim_number=1, label="D", document_id="2", judgment="실질적 동일",
        directness="inferred", quote="두 깊이 맵을 융합한다.", verify="verified",
        antecedent_resolved_by=["1"])
    bridge = ElementMatch(
        claim_number=1, label="A", document_id="1", judgment="실질적 동일",
        directness="direct", quote="3D 센서 데이터를 획득한다.", verify="verified")
    mappings = [DocumentMapping(reference_number=1, filename="a.pdf", document_id="1"),
                DocumentMapping(reference_number=2, filename="b.pdf", document_id="2")]

    difference = _difference(match, None, False, mappings, {}, bridge=bridge)

    assert difference == "직접 개시가 아니라 추론에 의한 대응입니다"


def test_batch_cells_judged_in_a_different_grouping_do_not_share_a_cache_key():
    """일괄 프롬프트에는 형제 청구항이 함께 실린다. 묶음이 다르면 같은 셀도 다른 프롬프트다."""
    claim = parse_claims(_BATCH_CLAIMS)[0]
    document = _BATCH_DOCUMENTS[0]
    args = (claim, document, "", DOCUMENT_BUDGET_CHARS, [])
    alone = cache.cache_key(*args, mode="document-batch", samples=1, cohort=[1, 2])
    wider = cache.cache_key(*args, mode="document-batch", samples=1, cohort=[1, 2, 3])
    assert alone != wider


def test_claim_grouping_does_not_depend_on_what_is_already_cached(monkeypatch):
    """묶음이 캐시 상태를 따라 달라지면 같은 셀이 실행마다 다른 프롬프트로 판정된다.

    그러면 키에 적은 cohort와 실제 프롬프트가 어긋나, 서로 다른 프롬프트의 판정이 한 키를
    공유하게 된다.
    """
    monkeypatch.setattr(pipeline, "COMPARE_CLAIM_BATCH", 2)
    claims = parse_claims(_BATCH_CLAIMS)
    assert [[c.number for c in g] for g in pipeline._claim_groups(claims)] == [[1, 2]]
    # 청구항이 하나 더 붙어도 앞 묶음은 그대로다(고정 규칙이므로).
    more = claims + [Claim(number=3, elements=[ClaimElement(label="A", text="구성")])]
    assert [[c.number for c in g] for g in pipeline._claim_groups(more)] == [[1, 2], [3]]


# --- 보고서 자기모순 자동 감지 ------------------------------------------------------
# 이 파이프라인의 버그 두 건은 계산이 틀린 것이 아니라 본문과 결론이 서로 다른 자료를 본
# 것이었고, 둘 다 테스트가 아니라 보고서를 눈으로 읽다가 발견됐다. 조립 직후에 기계적으로
# 맞춰 보면 같은 유형이 다시 새어 나가지 않는다.

def _assembled(difference: str | None, disclosed: int, total: int,
               residual: list[str], missing: list[str] | None = None) -> ClaimReport:
    return ClaimReport(
        claim_number=1, track="inventive_step_combination",
        chain=ChainInfo(claim_number=1, primary="1", residual=residual),
        claims=[ClaimResult(label="B", claim="구성 B", corresponded=True,
                            disclosed_limitations=disclosed, total_limitations=total,
                            missing_limitations=missing or [],
                            grade="일부 차이", emoji="🟠", narrative="서술",
                            difference=difference, status="부분 개시")])


def test_a_conclusion_that_contradicts_the_body_is_flagged():
    """실제로 나간 보고서: 결론은 'B에 차이가 남습니다', 본문은 '결합으로 해소됨'."""
    notes = report_invariants([_assembled("인용발명 2의 결합으로 해소됨", 3, 3, ["B"])])
    assert any("결합으로 해소되었다고 적었는데" in note for note in notes)


def test_a_missing_limitation_with_no_difference_line_is_flagged():
    """실제로 나간 보고서: 집계는 1/5인데 차이점 줄에 빠진 한정이 하나도 없었다."""
    notes = report_invariants([_assembled("같은 인용발명에서 구성 A의 대응이 확인되지 않았습니다",
                                          1, 5, [], missing=["학습된 NN에 입력될 이미지로 한정함"])])
    assert any("본문에 빠진 한정이 적히지 않았습니다" in note for note in notes)


def test_a_consistent_report_produces_no_note():
    assert report_invariants([_assembled("도파관 모델링 한정은 결합 후에도 남음", 2, 3, ["B"],
                                        missing=["도파관 모델링 한정"])]) == []
    assert report_invariants([_assembled(None, 3, 3, [])]) == []


# --- 사건 무관 불변식 ----------------------------------------------------------
# 사건별 기대값은 사람이 문헌을 통독해야 쓸 수 있어 사건이 늘지 않는다. 늘지 않으면 다음
# 사건은 여전히 처음 보는 사건이고, 그래서 "새 청구항·새 인용발명을 넣으면 또 안 된다"가
# 반복된다. 아래 성질들은 어떤 청구항·어떤 문헌에서도 참이어야 하므로 기대값이 필요 없다.

def _matrix(*matches) -> dict:
    result: dict = {}
    for match in matches:
        result.setdefault(match.document_id, {})[match.label] = match
    return result


def _report(chain: ChainInfo) -> ClaimReport:
    return ClaimReport(claim_number=1, track=chain.track, chain=chain, claims=[])


def _evidenced(document_id: str, label: str, *, rejected: bool = False) -> ElementMatch:
    """원문 대조를 통과한 개시 근거를 가진 셀. rejected면 의미검증이 축 결손으로만 뺀 상태."""
    return ElementMatch(
        claim_number=1, label=label, document_id=document_id,
        judgment="차이" if rejected else "실질적 동일",
        directness="direct", quote="원문 발췌", chunk_id=f"D{document_id}-P-0001",
        verify="verified",
        limitation_checks=[LimitationCheck(
            index=0, limitation="핵심 동작을 수행함", kind="core",
            disclosed=not rejected, quote="원문 발췌", chunk_id=f"D{document_id}-P-0001",
            verify="verified",
            semantic_status="rejected" if rejected else "accepted",
            semantic_note="대상 축 결손" if rejected else "")])


def test_p1_catches_a_gap_claimed_over_evidence_that_exists():
    """"어느 인용발명에서도 확인되지 않았다"는 진짜 공백에만 쓸 수 있다.

    이 진술을 만드는 경로가 여러 개(uncovered·rejection_impossible·미채택)라 한 곳을 막아도
    다른 곳으로 새어 나왔다. 그래서 경로가 아니라 결과를 본다.
    """
    matrix = _matrix(_evidenced("1", "A"))
    chain = ChainInfo(claim_number=1, track="rejection_impossible", primary="1", uncovered=["A"])
    notes = pipeline_invariants([_report(chain)], {1: matrix})
    assert any("[불변식 P1]" in note and "문헌 1" in note for note in notes)

    # 같은 공백이라도 유보로 갈라 두었으면 사실과 어긋나지 않는다.
    chain.combination_pending = ["A"]
    assert not [note for note in pipeline_invariants([_report(chain)], {1: matrix})
                if "[불변식 P1]" in note]


def test_p1_treats_an_axis_rejected_document_as_evidence_too():
    """축 결손으로 기각된 근거도 '문헌에 원문이 있다'는 사실은 그대로다."""
    matrix = _matrix(_evidenced("1", "A", rejected=True))
    chain = ChainInfo(claim_number=1, track="rejection_impossible", primary="1", uncovered=["A"])
    notes = pipeline_invariants([_report(chain)], {1: matrix})
    assert any("[불변식 P1]" in note for note in notes)


def test_p1_allows_a_real_gap_when_only_one_of_several_limitations_has_evidence():
    """E처럼 일부 한정의 근접 기재만 있어도 구성 전체는 정확히 미대응일 수 있다."""
    partial = _evidenced("1", "E")
    partial.limitation_checks.append(LimitationCheck(
        index=1, limitation="융합 점군에서 가시 영역을 추출함", kind="core",
        disclosed=False, semantic_status="not_run"))
    partial.judgment = "차이"
    chain = ChainInfo(claim_number=1, track="rejection_impossible", primary="1",
                      uncovered=["E"])

    notes = pipeline_invariants([_report(chain)], {1: _matrix(partial)})
    narrative = _narrative("E", "가시 영역을 추출함", partial, partial, False, [], {},
                           related="문헌 1, 문단 1")

    assert not [note for note in notes if "[불변식 P1]" in note]
    assert "일부 하위 한정에는 원문 근거가 있으나" in narrative
    assert "구성 전체를 충족" in narrative


def test_p3_catches_a_grade_higher_than_its_own_limitation_checks():
    """등급은 한정별 개시에서 유도된 값을 넘을 수 없다.

    넘은 등급은 그대로 has_correspondence·신규성 게이트·문헌 순위로 들어간다. 낮은 쪽은 보지
    않는다 — 발췌 검증·지시 관계·결합 결과 상한이 모두 등급을 의도적으로 내리는 장치다.
    """
    inflated = _evidenced("1", "A", rejected=True)      # 한정은 미개시인데
    inflated.judgment = "실질적 동일"                     # 등급만 높게 남은 상태
    chain = ChainInfo(claim_number=1, track="inventive_step_combination", primary="1")
    notes = pipeline_invariants([_report(chain)], {1: _matrix(inflated)})
    assert any("[불변식 P3]" in note for note in notes)


def test_invariants_stay_quiet_while_the_comparison_is_incomplete():
    """미판정 상태에서는 이 성질들이 애초에 성립하지 않는다.

    미판정을 '대응 없음'과 같은 칸에 넣지 않는 것이 이 파이프라인의 규율이고, 불변식 검사도
    같은 규율을 따라야 한다. 그러지 않으면 분석이 중단될 때마다 위반이 무더기로 찍힌다.
    """
    matrix = _matrix(_evidenced("1", "A"))
    chain = ChainInfo(claim_number=1, track="analysis_incomplete", uncovered=["A"])
    assert pipeline_invariants([_report(chain)], {1: matrix}) == []


def _candidate(document_id: str, **overrides) -> SupplementCandidate:
    row = {"document_id": document_id, "judgment": "실질적 동일", "directness": "direct",
           "verify": "verified", "has_quote": True, "eligible": True}
    return SupplementCandidate(**{**row, **overrides})


def test_p2_stays_quiet_for_a_duplicate_candidate_the_combination_already_covers():
    """채택 조합이 같은 기여를 이미 확보했다면 그 후보는 정상 제외다.

    종전 P2는 주 인용발명 대비 이득(row.gain)과 limit_binding을 맞댔는데 둘은 기준선이
    다르다 — 앞은 주 인용발명 단독, 뒤는 채택 조합 전체다. 실측에서 두 문헌이 같은 구성에
    정확히 같은 이득을 냈고, 하나가 채택되자 다른 하나가 매 회차 위반으로 보고됐다.
    """
    matrix = _matrix(_evidenced("2", "A"))
    chain = ChainInfo(
        claim_number=1, track="inventive_step_combination", primary="1", secondaries=["3"],
        residual=["A"],
        element_coverage=[ElementCoverage(label="A", candidates=[
            _candidate("2", gain=0.44, merged_gain=0.0,
                       excluded_reason="채택 조합이 같은 기여를 이미 확보함(증분 0)")])])

    assert not [note for note in pipeline_invariants([_report(chain)], {1: matrix})
                if "[불변식 P2]" in note]


def test_p2_catches_a_candidate_that_vanished_without_a_reason():
    """조합에 더 보탤 것이 있는데 사유 없이 빠졌다면 게이트 하나가 조용히 막고 있는 것이다."""
    matrix = _matrix(_evidenced("2", "A"))
    chain = ChainInfo(
        claim_number=1, track="inventive_step_combination", primary="1", secondaries=["3"],
        residual=["A"],
        element_coverage=[ElementCoverage(label="A", candidates=[
            _candidate("2", gain=0.44, merged_gain=0.44, excluded_reason="")])])

    notes = pipeline_invariants([_report(chain)], {1: matrix})
    assert any("[불변식 P2]" in note and "제외 사유도 기록되지 않았습니다" in note for note in notes)


def test_p2_still_checks_when_the_combination_limit_binds():
    """상한이 걸린 실행이라고 통째로 건너뛰면, 그 안에서 사유 없이 사라진 후보를 놓친다.

    상한 자체가 이제 후보 행에 사유로 적히므로 미리 걸러 낼 필요가 없다.
    """
    matrix = _matrix(_evidenced("2", "A"))
    chain = ChainInfo(
        claim_number=1, track="inventive_step_combination", primary="1", secondaries=["3"],
        residual=["A"], limit_binding=True,
        element_coverage=[ElementCoverage(label="A", candidates=[
            _candidate("2", gain=0.44, merged_gain=0.44, excluded_reason="")])])

    assert any("[불변식 P2]" in note for note in pipeline_invariants([_report(chain)], {1: matrix}))


def test_a_primary_without_correspondence_is_not_dressed_up_as_a_combination():
    """'차이'는 core가 하나도 개시되지 않았다는 뜻이다. 그 문헌을 결합 상대로 세우면 안 된다.

    실측: 점군 융합 구성을 인용발명 2가 단독으로 개시했는데, 주 인용발명의 **텍스처 이미지
    스티칭** 문장이 결합 상대로 실려 "…는 구성이 기재되어 있으나 … 이를 결합하면"으로
    나갔다. 그 문헌이 융합을 가르친 것처럼 읽히고, 인용하면 곧바로 반박당한다.
    """
    documents = {
        "1": Document(id="1", filename="primary.pdf", chunks=[
            Chunk(document_id="1", chunk_id="D1-P-0262", page=11, paragraph="0262",
                  text="Instead of fusing all input images we use only three images for the texture map.")]),
        "2": Document(id="2", filename="fusion.pdf", chunks=[
            Chunk(document_id="2", chunk_id="D2-B-p003", page=3,
                  text="The ToF and stereo depth measurements are fused using the confidence measures.")]),
    }
    mappings = [DocumentMapping(reference_number=1, filename="primary.pdf", document_id="1",
                                document_number="US 2019/0035149 A1"),
                DocumentMapping(reference_number=2, filename="fusion.pdf", document_id="2")]
    primary = ElementMatch(
        claim_number=1, label="D", document_id="1", judgment="차이", directness="absent",
        quote="Instead of fusing all input images we use only three images for the texture map.",
        chunk_id="D1-P-0262", verify="verified")
    adopted = ElementMatch(
        claim_number=1, label="D", document_id="2", judgment="실질적 동일", directness="direct",
        quote="The ToF and stereo depth measurements are fused using the confidence measures.",
        chunk_id="D2-B-p003", verify="verified", reason="두 센서의 신뢰도로 가중해 융합하고 있음")

    narrative = _narrative("D", "두 점군을 신뢰도에 기초하여 융합함", adopted, primary, True,
                           mappings, documents)

    assert "이를 결합하면" not in narrative
    assert "텍스처" not in narrative and "texture" not in narrative
    assert narrative.endswith('구성과 대응됩니다.')


def test_the_summary_names_only_the_documents_that_disclose_the_representative_element():
    """주어와 술어가 다른 출처에서 오면 요약이 본문보다 넓게 말한다."""
    results = [
        ClaimResult(label="A", claim="두 점군을 신뢰도에 기초하여 융합함", corresponded=True,
                    status="개시됨", adopted_reference=2),
        ClaimResult(label="B", claim="영상을 획득함", corresponded=True, status="개시됨",
                    adopted_reference=1),
    ]
    claim = Claim(number=1, elements=[
        ClaimElement(label="A", text="두 점군을 신뢰도에 기초하여 융합함", importance=5),
        ClaimElement(label="B", text="영상을 획득함", importance=2)])

    summary = _summary_similarity(claim, results)

    assert summary.startswith("청구항과 인용발명 2는 ")     # 대표 구성 A를 개시한 문헌만
    assert "인용발명 1" not in summary
