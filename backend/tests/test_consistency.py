"""구성 간 정합성. 셀은 독립 판정이라 청구항 전체로 보면 성립할 수 없는 조합이 남는다."""
from app.consistency import (anaphora, antecedents, cross_document_notes,
                             enforce_antecedents)
from app.models import Claim, ClaimElement, ElementMatch, LimitationCheck


def claim() -> Claim:
    return Claim(number=3, elements=[
        ClaimElement(label="A", importance=5,
                     text="상기 반사 심도부는 상호 이격되어 평행하게 배치되어 각각 부분 반사 특성을 "
                          "갖는 제1반사부재 및 제2반사부재"),
        ClaimElement(label="B", importance=4,
                     text="상기 제1 반사부재 및 상기 제2 반사부재 사이에 배치되는 광원을 포함하는 것"),
    ])


def cell(label: str, judgment: str, *, quote: str = "원문 발췌 문장입니다",
         verify: str = "verified", direct: bool = True, document_id: str = "1") -> ElementMatch:
    return ElementMatch(claim_number=3, label=label, document_id=document_id, judgment=judgment,
                        directness="direct" if direct else "absent",
                        quote=quote, chunk_id="D1-P-0001" if quote else "",
                        verify=verify if quote else "empty")


def test_anaphora_ignores_spacing_between_claim_and_specification():
    """같은 용어를 띄어쓰기만 달리 적는 일이 흔하다("제1반사부재" ↔ "제1 반사부재")."""
    assert anaphora("상기 제1 반사부재 및 상기 제2 반사부재 사이에 배치되는 광원") == [
        "제1반사부재", "제2반사부재"]
    assert antecedents(claim()) == {"B": ["A"]}


def test_element_referring_to_an_undisclosed_antecedent_is_capped():
    """문헌에 제1·제2 반사부재가 없으면 "그 둘 사이의 광원"도 그 문헌에 있을 수 없다.

    (B) 셀만 떼어 놓고 보면 "광원"이라는 문장이 있으므로 부분 대응이 나온다. 그러면 남는 것은
    "광원을 포함한다"뿐이고, 그 분야의 어떤 장치나 만족하는 문장이 개시 근거로 보고된다.
    """
    matrix = {"1": {"A": cell("A", "차이", quote="", direct=False),
                    "B": cell("B", "실질적 동일")}}

    notes = enforce_antecedents(claim(), matrix)

    # 완전 개시로는 세지 않되, 원문 대조를 통과한 발췌가 있으므로 부분 대응까지는 남긴다.
    # '차이'까지 내리면 그 문헌이 조합에서 빠져 근거 목록에서도 사라진다.
    assert matrix["1"]["B"].judgment == "일부 유사"
    assert matrix["1"]["B"].downgraded_from == "실질적 동일"
    assert notes and "참조 구성 A" in notes[0]


def test_a_disclosed_antecedent_leaves_the_referring_element_alone():
    matrix = {"1": {"A": cell("A", "실질적 동일"), "B": cell("B", "실질적 동일")}}

    assert enforce_antecedents(claim(), matrix) == []
    assert matrix["1"]["B"].judgment == "실질적 동일"


def test_the_cap_is_per_document_so_combinations_survive():
    """문헌 A가 앞 구성을, 문헌 B가 뒤 구성을 개시한 경우는 정상적인 결합이다."""
    matrix = {
        "1": {"A": cell("A", "동일"), "B": cell("B", "대응 없음", quote="", direct=False)},
        "2": {"A": cell("A", "대응 없음", quote="", direct=False), "B": cell("B", "동일")},
    }

    enforce_antecedents(claim(), matrix)

    assert matrix["1"]["A"].judgment == "동일"          # 문헌 1의 앞 구성은 그대로
    assert matrix["2"]["B"].judgment == "일부 유사"     # 문헌 2 안에서는 완전 개시로 보지 않는다


def test_a_selected_document_combination_restores_only_the_antecedent_cap():
    """앞 구성과 뒤 구성을 문헌 둘이 나눠 개시하면 결합 결과에는 단독문헌 경고를 남기지 않는다."""
    from app.chain import build_chain

    target = claim()
    matrix = {
        "1": {"A": cell("A", "대응 없음", quote="", direct=False, document_id="1"),
              "B": cell("B", "동일", document_id="1")},
        "2": {"A": cell("A", "동일", document_id="2"),
              "B": cell("B", "대응 없음", quote="", direct=False, document_id="2")},
    }
    enforce_antecedents(target, matrix)
    assert matrix["1"]["B"].judgment == "일부 유사"
    assert matrix["1"]["B"].antecedent_capped_from == "동일"

    chain = build_chain(target, matrix, {}, [target])
    coverage = next(item for item in chain.element_coverage if item.label == "B")

    assert chain.primary == "2" and chain.secondaries == ["1"]
    assert coverage.adopted_judgment == "동일"
    assert coverage.residual_difference == []
    # 감사용 단독문헌 셀은 여전히 상한이 걸린 상태여야 합니다.
    assert matrix["1"]["B"].judgment == "일부 유사"


def test_an_unjudged_cell_is_never_capped():
    """판정을 받지 못한 셀은 '대응 없음'과 다르므로 건드리지 않는다."""
    matrix = {"1": {"A": cell("A", "대응 없음", quote="", direct=False),
                    "B": cell("B", "실질적 동일")}}
    matrix["1"]["B"].error = "응답에 (B) 판정이 없습니다."

    enforce_antecedents(claim(), matrix)

    assert matrix["1"]["B"].judgment == "실질적 동일"


# --- 교차문헌 일관성 --------------------------------------------------------------
# 의미검증은 문헌별로 따로 호출된다. 어떤 문헌을 심사하는 호출은 다른 문헌에 무엇을
# 인정했는지 볼 수 없으므로, 같은 성격의 기재가 한쪽에서 인정되고 다른 쪽에서 기각되는 일이
# 구조적으로 생긴다. 실측에서 한 문헌의 "보정 행렬"은 인정되고 같은 한정에 대해 다른 문헌의
# "보정 맵"은 기각됐다. 프롬프트로는 닿지 않으므로 코드가 갈린 지점을 찾아 남긴다.

def _judged(document_id: str, status: str, quote: str) -> ElementMatch:
    return ElementMatch(
        claim_number=1, label="C", document_id=document_id, judgment="일부 차이",
        directness="direct", quote=quote, chunk_id=f"D{document_id}-P-0001", verify="verified",
        limitation_checks=[LimitationCheck(
            index=0, limitation="균일도 보정 이미지를 획득함", kind="core",
            disclosed=(status == "accepted"), quote=quote,
            chunk_id=f"D{document_id}-P-0001", verify="verified",
            semantic_status=status, semantic_note="사유")])


def test_a_limitation_judged_both_ways_across_documents_is_reported():
    notes = cross_document_notes(
        [_judged("2", "rejected", "obtaining arrays of scaling factors"),
         _judged("3", "accepted", "to obtain a plurality of correction matrices")],
        {"2": "maps.pdf", "3": "matrices.pdf"})
    assert len(notes) == 1
    assert "갈렸습니다" in notes[0]
    assert "maps.pdf(기각" in notes[0] and "matrices.pdf(인정" in notes[0]


def test_documents_that_agree_produce_no_note():
    """정상적으로 일치한 판정까지 남기면 이 노트가 구성대비 결과보다 길어진다."""
    assert cross_document_notes(
        [_judged("2", "accepted", "quote a"), _judged("3", "accepted", "quote b")], {}) == []
    assert cross_document_notes(
        [_judged("2", "rejected", "quote a"), _judged("3", "rejected", "quote b")], {}) == []


def test_divergence_is_reported_but_never_corrected():
    """어느 쪽이 옳은지는 원문을 읽어야 정해진다. 코드가 한쪽으로 맞추면 안 된다."""
    rejected = _judged("2", "rejected", "quote a")
    accepted = _judged("3", "accepted", "quote b")
    cross_document_notes([rejected, accepted], {})
    assert rejected.limitation_checks[0].disclosed is False       # 그대로 둔다
    assert accepted.limitation_checks[0].disclosed is True
