"""파이프라인 전체. CLI는 고정 응답으로 대체하고, 그 뒤 단계가 전부 코드인지 확인한다."""
import json

import pytest

from app import cache, claims as claims_module, compare, pipeline, priorart
from app.models import (ChainInfo, Chunk, Claim, ClaimElement, ClaimResult, Document,
                        DocumentMapping, ElementMatch, EvidenceSpan, LimitationCheck)
from app.pdf import classify, extract_document_number
from app.report import to_markdown
from app.report import (_narrative, _reason_sentence, _summary, _summary_similarity,
                        _supporting_passages, refresh_mappings)

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
    assert element(report, "A").similarity == 92 and element(report, "A").emoji == "🟢"
    assert element(report, "B").similarity == 97                # 보조 문헌의 동일 판정이 채택됨


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
    assert "전제부 대응 미확인" in report.rejection_basis
    assert "전제부" in report.summary_difference
    assert element(report, "P0").is_preamble is True


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


def test_narrative_maps_each_atomic_limitation_to_its_own_quote():
    """대표 발췌 하나가 서로 다른 하위 한정의 근거인 것처럼 표시되면 안 된다."""
    from app.report import _narrative

    storage_quote = "저장소에 보관된 콘텐츠 조각을 재생 요청에 따라 검색한다."
    token_quote = "서버는 생성된 세션에 대응하는 세션 토큰을 사용자에게 발급한다."
    document = Document(id="1", filename="prior.pdf", chunks=[
        Chunk(document_id="1", chunk_id="storage", page=4, text=storage_quote),
        Chunk(document_id="1", chunk_id="token", page=5, text=token_quote),
    ])
    match = ElementMatch(
        claim_number=1, label="D", document_id="1", judgment="일부 차이",
        directness="inferred", quote=storage_quote, chunk_id="storage", verify="verified",
        missing_limitations=["가상 세션을 생성함"],
        limitation_checks=[
            LimitationCheck(index=0, limitation="콘텐츠를 검색함", disclosed=True,
                            quote=storage_quote, chunk_id="storage", verify="verified"),
            LimitationCheck(index=1, limitation="세션 토큰을 발급함", disclosed=True,
                            quote=token_quote, chunk_id="token", verify="verified"),
        ],
    )
    narrative = _narrative(
        "콘텐츠를 검색하고 가상 세션을 생성하여 세션 토큰을 발급함", match,
        ChainInfo(claim_number=1, primary="1"), {"1": {"D": match}},
        [DocumentMapping(reference_number=1, filename="prior.pdf", document_id="1")],
        {"1": document},
    )

    assert narrative.count(storage_quote) == 1
    assert narrative.count(token_quote) == 1
    assert "콘텐츠를 검색함" in narrative and "세션 토큰을 발급함" in narrative
    assert "4 페이지" in narrative and "5 페이지" in narrative
    assert "가상 세션을 생성함 기재는 확인되지 않았습니다" in narrative


def test_hallucinated_quote_is_downgraded_before_selection(monkeypatch, stub_cli):
    """검증에 실패한 발췌 위에서 결정론적 계산이 돌지 않아야 한다."""
    fabricated = json.loads(json.dumps(RESPONSES))
    # P0가 아니라 실제로 결과를 확인하는 구성 A의 발췌를 변조합니다.
    fabricated["1"]["matches"][1].update(judgment="동일", quote="이 문장은 문헌 어디에도 존재하지 않는 발췌입니다")
    monkeypatch.setitem(RESPONSES, "1", fabricated["1"])
    result = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    item = element(result.reports[0], "A")
    assert item.similarity is not None and item.similarity < 97
    assert any("→" in note or "낮췄" in note for note in [item.note] + result.validation)
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
    assert "큐가 우선순위를 가짐" in result.reports[1].claims[0].narrative
    assert "큐가 순환형임" in result.reports[2].claims[0].narrative


def test_changing_the_guideline_invalidates_the_cache(stub_cli):
    pipeline.analyze("job", CLAIMS, DOCUMENTS, "기본 지침")
    stub_cli.clear()
    pipeline.analyze("job2", CLAIMS, DOCUMENTS, "다른 지침")
    assert stub_cli.count("matches") == 2


def test_markdown_reports_the_chain_and_the_deterministic_grades(stub_cli):
    result = pipeline.analyze("job", CLAIMS, DOCUMENTS)
    markdown = to_markdown(result)
    assert "| 인용발명 1 | 10-2020-0001 | 1.pdf | - | patent |" in markdown
    # 조문은 검토 트랙의 근거로만 인용하고, 이 파이프라인이 평가하지 않은 항목을 함께 적는다.
    assert "**검토 트랙**: 진보성 검토 후보 (제29조제2항) — 인용발명 2건 결합" in markdown
    assert "결합 동기·용이성 미평가" in markdown
    assert "선행기술 적격성" in markdown
    assert "**인용발명 조합**: 인용발명 1 (10-2020-0001) + 인용발명 2 (10-2020-0002)" in markdown
    assert "### (A) 쓰기 요청을 큐에 저장하는 메모리 컨트롤러" in markdown
    assert "판정 대표값: 92% 🟢 실질적 동일" in markdown
    assert "단락 [0021]" in markdown
    assert "심사 판단" not in markdown


def test_summary_describes_overall_similarity_in_one_line(stub_cli):
    """구성별 유사 문장을 반복하지 않고 전체 공통 내용을 한 문장으로 적는다."""
    report = pipeline.analyze("job", CLAIMS, DOCUMENTS).reports[0]
    assert report.summary_similarity == (
        "청구항과 인용발명 1 및 인용발명 2는 쓰기 요청을 큐에 저장하는 메모리 컨트롤러 및 "
        "상태 변경 시 알림을 전송하는 통신부에 관한 기술 내용이 전체적으로 유사합니다."
    )
    assert "\n" not in report.summary_similarity
    assert report.summary_similarity.count("유사합니다") == 1
    assert "나머지 구성 B은 인용발명 2 (10-2020-0002)에서 확인됩니다" in report.summary
    assert element(report, "B").adopted_reference == 2
    assert "인용발명 2" in element(report, "B").supplement_disclosure

    markdown = to_markdown(pipeline.analyze("job-summary", CLAIMS, DOCUMENTS))
    assert "인용발명 선정 근거(구성대비 단계)" not in markdown
    assert "- 최종 판단:" not in markdown


def test_partial_correspondence_is_not_summarized_as_disclosed():
    target = Claim(number=1, elements=[
        ClaimElement(label="A", text="재생 요청을 수신함"),
        ClaimElement(label="D", text="가상 세션을 생성하고 세션 토큰을 발급함"),
    ])
    chain = ChainInfo(claim_number=1, primary="1", track="inventive_step_combination")
    results = [
        ClaimResult(label="A", claim=target.elements[0].text, status="개시됨", adopted_reference=1),
        ClaimResult(label="D", claim=target.elements[1].text, status="부분 개시", adopted_reference=1),
    ]
    merged = {
        "A": ElementMatch(claim_number=1, label="A", document_id="1", judgment="동일"),
        "D": ElementMatch(claim_number=1, label="D", document_id="1", judgment="일부 차이"),
    }
    mappings = [DocumentMapping(reference_number=1, filename="d1.pdf", document_id="1")]

    summary = _summary(target, chain, results, merged, mappings)
    similarity = _summary_similarity(results)

    assert "구성 A이 개시되어 있고" in summary
    assert "구성 D에는 일부 대응 기재만" in summary
    assert "구성 A, D이 개시" not in summary
    assert "구성 D에는 일부 대응 기재만" in similarity


def test_supplement_narrative_is_ordered_without_repeating_the_quote(stub_cli):
    """주 문헌의 한계 다음에 보완 근거와 판단을 쓰고, 문장마다 줄을 바꾼다."""
    narrative = element(pipeline.analyze("job", CLAIMS, DOCUMENTS).reports[0], "B").narrative
    lines = narrative.splitlines()

    assert len(lines) == 5
    assert lines[0].startswith("인용발명 1 (10-2020-0001)만으로는")
    assert lines[1] == "누락된 한정은 알림 전송입니다."
    assert lines[2] == "인용발명 2 (10-2020-0002)에서 다음 대응 기재를 확인했습니다."
    assert lines[3].endswith(f'"{QUOTE_B}" (단락 [0012])')
    assert lines[4].startswith('따라서 청구항의 "상태 변경 시 알림을 전송하는 통신부"')
    assert narrative.count(QUOTE_B) == 1

    markdown = to_markdown(pipeline.analyze("job-2", CLAIMS, DOCUMENTS))
    assert "보기 어렵습니다.  \n누락된 한정은" in markdown


def test_verified_related_evidence_is_shown_without_calling_it_full_disclosure():
    match = ElementMatch(
        claim_number=1, label="C", document_id="1", judgment="대응 없음", directness="absent",
        missing_limitations=["수요 지표의 임계값에 따라 저장 위치를 정함"],
        evidence=[EvidenceSpan(chunk_id="D1-P-0021", quote=QUOTE_A,
                               quote_translation="관련 저장 처리", verify="verified")],
    )
    passages = _supporting_passages(match, {"1": DOCUMENTS[0]})

    assert passages == [
        (["관련 기재(청구항 한정 전체를 개시하는 근거는 아님)"],
         "관련 저장 처리", QUOTE_A, "단락 [0021]")
    ]

    narrative = _narrative(
        "수요 지표에 따라 저장 위치를 변경함", match,
        ChainInfo(claim_number=1, primary="1"), {"1": {"C": match}},
        [DocumentMapping(reference_number=1, filename="1.pdf", document_id="1")],
        {"1": DOCUMENTS[0]},
    )
    assert "관련 기재(청구항 한정 전체를 개시하는 근거는 아님)" in narrative
    assert QUOTE_A in narrative
    assert narrative.endswith("관련 기재가 있으나 청구항 한정 전체에 대응하는 개시로 보기 어렵습니다.")


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
    assert classify("[0001] 어떤 기재 [0002] 다른 기재 [0003] 또 다른 기재") == "patent"
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


def test_reason_sentence_removes_temporary_reference_number():
    assert _reason_sentence("인용발명 1에는 동기화 기술이 개시됨") == "이는 동기화 기술이 개시된다는 점을 보여 줍니다."
    assert _reason_sentence("인용발명 4는 XR 표시를 개시함") == "이는 XR 표시를 개시한다는 점을 보여 줍니다."
    assert _reason_sentence("인용문헌 2에는 병합 단계만 개시됨") == "이는 병합 단계만 개시된다는 점을 보여 줍니다."


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
