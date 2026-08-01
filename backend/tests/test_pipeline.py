from app.pdf import classify
from app.pipeline import build_claims, build_prompt, resolve_mapping, split_claims, to_markdown
from app.models import AnalysisResult
from app.prompts import DEFAULT_ANALYSIS_PROMPT
from app.search import retrieve

DOCUMENTS = [
    {"id": "1", "filename": "a.pdf", "type": "patent", "chunks": []},
    {"id": "2", "filename": "b.pdf", "type": "patent", "chunks": []},
]

def test_classification():
    assert classify("[0001] A patent claim") == "patent"
    assert classify("Abstract Introduction Methods DOI") == "paper"

def test_rrf_retrieval():
    chunks = [{"text":"memory controller queues write requests", "page":1}, {"text":"unrelated optics", "page":2}]
    assert retrieve("write requests", chunks)[0]["page"] == 1

def test_multiline_element_stays_one_component_and_keeps_its_label():
    """줄바꿈으로 쪼개면 (B)가 둘로 갈려 이후 라벨이 전부 밀린다."""
    preamble, claims = split_claims("청구항 1.\r\n(A) 네트워크 모듈;\r\n(B) 영상, 음성을\r\n수집하는 입력 모듈;\r\n(C) 저장 모듈")
    assert preamble == "청구항 1."
    assert [c["label"] for c in claims] == ["A", "B", "C"]
    assert claims[1]["text"] == "영상, 음성을 수집하는 입력 모듈;"

def test_unlabeled_claims_fall_back_to_sequential_labels():
    _, claims = split_claims("첫 번째 구성\n두 번째 구성")
    assert [c["label"] for c in claims] == ["A", "B"]

def test_reference_numbers_follow_the_model_ranking_not_upload_order():
    mappings, warnings = resolve_mapping([{"reference_number": 1, "document_id": "2", "document_number": "US 11,456,887"}], DOCUMENTS)
    assert [(m.reference_number, m.document_id) for m in mappings] == [(1, "2"), (2, "1")]
    assert mappings[0].role == "주 인용발명" and mappings[0].document_number == "US 11,456,887"
    assert warnings  # 누락된 문헌을 업로드 순서로 붙였다는 경고

def test_mapping_falls_back_to_upload_order_without_model_ranking():
    mappings, warnings = resolve_mapping(None, DOCUMENTS)
    assert [m.document_id for m in mappings] == ["1", "2"]
    assert warnings

def test_claims_are_aligned_by_label_and_evidence_gets_the_reference_number():
    mappings, _ = resolve_mapping([{"document_id": "2"}], DOCUMENTS)
    claims = [{"label": "A", "text": "네트워크 모듈"}, {"label": "B", "text": "저장 모듈"}]
    raw = [{"label": "B", "similarity": 92, "narrative": "인용발명 1에는\n기재되어 있습니다.", "difference": "용어 차이",
            "evidence": [{"document_id": "2", "paragraph": "0021", "excerpt": "저장한다", "quality": "HIGH"}]}]
    results, warnings = build_claims(raw, claims, mappings, DOCUMENTS)
    assert [r.label for r in results] == ["A", "B"]
    assert results[1].similarity == 92 and results[1].emoji == "🟢" and results[1].grade == "실질적 동일"
    assert "\n" not in results[1].narrative
    assert results[1].evidence[0].reference_number == 1 and results[1].evidence[0].filename == "b.pdf"
    assert results[0].similarity is None and "추가 검색 필요" in results[0].narrative
    assert warnings

def test_prompt_carries_the_user_instruction_and_the_labels():
    prompt = build_prompt(DEFAULT_ANALYSIS_PROMPT, "청구항 1.", [{"label": "A", "text": "네트워크 모듈"}], DOCUMENTS)
    assert "[분석 지시]" in prompt and "인용발명 1(주 인용발명)" in prompt
    assert '"label": "A"' in prompt and "[출력 규약]" in prompt

def test_markdown_follows_the_requested_format():
    mappings, _ = resolve_mapping([{"document_id": "1", "document_number": "US 10,987,654 A1"}], DOCUMENTS)
    claims = [{"label": "A", "text": "네트워크 모듈"}]
    raw = [{"label": "A", "similarity": 88, "narrative": "인용발명 1 (US 10,987,654 A1)에는 … 대응됩니다.", "difference": "용어 차이"}]
    results, _ = build_claims(raw, claims, mappings, DOCUMENTS)
    report = to_markdown(AnalysisResult(job_id="j", claim_mapping=mappings, claims=results,
                                        summary_similarity="공통 목적", summary_difference="세부 차이"))
    assert "| 인용발명 1 (주 인용발명) | US 10,987,654 A1 | a.pdf | patent |" in report
    assert "## (A) 네트워크 모듈" in report
    assert "유사도: 88% 🟠 기술 사상 동일, 세부 구현 방식의 단순 변경" in report
    assert "→ 차이점: 용어 차이" in report
    assert "- 유사점: 공통 목적" in report
