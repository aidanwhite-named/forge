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


def chained_claim() -> Claim:
    """같은 대상을 여러 구성이 되풀이 참조하는 청구항. 실무에서는 이쪽이 기본값이다.

    구성 두 개짜리 청구항으로는 지시 관계 버그가 하나도 드러나지 않는다. 되풀이 참조도,
    "…로부터" 형태의 참조도, 전제부와 겹치는 용어도 구성이 셋 이상이어야 나타난다.
    """
    return Claim(number=1, elements=[
        ClaimElement(label="P0", importance=1, is_preamble=True,
                     text="결함 인지형 건축물 외벽 3차원 모델링 시스템에 있어서"),
        ClaimElement(label="A", importance=2,
                     text="카메라로 촬영된 건물 외벽 영상 데이터를 수집하는 데이터수집부"),
        ClaimElement(label="B", importance=5,
                     text="상기 데이터수집부에 수집된 건물 외벽 영상 데이터로부터 균열 특징을 "
                          "산출하는 결함탐지부"),
        ClaimElement(label="C", importance=4,
                     text="상기 데이터수집부에 수집된 건물 외벽 영상 데이터와 상기 결함탐지부로부터 "
                          "생성된 균열정보를 근거로 3차원 모델을 생성하는 3차원모델생성부"),
        ClaimElement(label="D", importance=4,
                     text="상기 데이터수집부에 수집된 건물 외벽 영상 데이터에서 수평 구조물을 "
                          "검출하는 수평구조물검출부"),
        ClaimElement(label="E", importance=5,
                     text="상기 3차원 모델의 결함 인스턴스에 층 인덱스를 할당하는 층인덱스할당부"),
    ])


def test_anaphora_ignores_spacing_between_claim_and_specification():
    """같은 용어를 띄어쓰기만 달리 적는 일이 흔하다("제1반사부재" ↔ "제1 반사부재")."""
    assert anaphora("상기 제1 반사부재 및 상기 제2 반사부재 사이에 배치되는 광원") == [
        "제1반사부재", "제2반사부재"]
    assert antecedents(claim()) == {"B": ["A"]}


def test_stacked_particles_do_not_swallow_the_referenced_term():
    """"…로부터"는 조사가 겹쳐 붙는다. 하나만 끊으면 지시 어구가 "결함탐지부로"가 된다.

    앞 구성의 문언은 "…결함탐지부"라 그 어구는 어디에도 걸리지 않고, 그 구성은 지시 관계가
    아예 없는 것으로 처리된다. 상한이 걸려야 할 자리에서 조용히 안 걸리는 쪽이라 보고서만
    보고는 알 수 없다.
    """
    assert anaphora("상기 결함탐지부로부터 생성된 균열정보를 입력받고") == ["결함탐지부"]
    assert anaphora("상기 수평구조물검출부로부터 검출된 수평 구조물을") == ["수평구조물검출부"]
    assert anaphora("상기 문자인식부로부터 인식된 층수와") == ["문자인식부"]


def test_only_the_element_that_introduced_the_term_is_the_antecedent():
    """뒤 구성이 같은 대상을 되풀이 참조해도 지시 대상은 그것을 도입한 구성 하나다.

    B·C·D가 모두 "상기 데이터수집부에 수집된 …"으로 시작한다. 어구를 담은 앞 구성을 모두
    이으면 D의 지시 대상이 [A, B, C]가 되고, enforce_antecedents가 min(선행 구성 판정)으로
    상한을 잡으므로 D와 아무 관계 없는 B의 미대응이 D의 등급을 끌어내린다. 실측에서 한정이
    2/2 전부 개시된 구성이 그렇게 강등된 채 결론의 '차이가 남는 구성'에 실렸다.
    """
    links = antecedents(chained_claim())

    assert links["D"] == ["A"]                  # 데이터수집부를 도입한 구성만
    assert links["B"] == ["A"]
    assert links["C"] == ["A", "B"]             # 결함탐지부는 실제로 B가 도입했다


def test_the_preamble_is_the_antecedent_only_when_nothing_else_introduced_the_term():
    """전제부는 발명의 명칭이라 뒤 구성이 쓰는 용어의 부분 문자열을 거의 언제나 품는다.

    "…3차원 모델링 시스템에 있어서"가 "3차원 모델"을 품는다. 전제부를 지시 대상으로 잡으면
    상한이 사실상 풀린다 — 전제부는 "컴퓨터로 실행되는 …시스템"이라 어느 문헌에서나 완전
    개시로 나오기 때문이다.
    """
    links = antecedents(chained_claim())

    assert links["E"] == ["C"]                  # 그 3차원 모델을 실제로 생성한 구성


def test_an_unrelated_earlier_element_never_caps_the_grade():
    """지시 관계가 없는 구성의 미대응이 등급을 끌어내리면 안 된다.

    이 방향의 오류는 P3 불변식이 잡지 못한다(등급이 유도값보다 **높은** 쪽만 본다). 여기서
    새면 보고서가 "한정 2/2 개시"와 "차이가 남는 구성"을 동시에 적고도 전부 통과한다.
    """
    target = chained_claim()
    matrix = {"1": {"P0": cell("P0", "실질적 동일"), "A": cell("A", "실질적 동일"),
                    "B": cell("B", "대응 없음", quote="", direct=False),
                    "C": cell("C", "일부 유사"), "D": cell("D", "실질적 동일"),
                    "E": cell("E", "대응 없음", quote="", direct=False)}}
    for label, match in matrix["1"].items():
        match.claim_number, match.label = 1, label

    enforce_antecedents(target, matrix)

    assert matrix["1"]["D"].judgment == "실질적 동일"      # D가 참조하는 것은 A뿐이다
    assert matrix["1"]["D"].antecedent_note == ""
    assert matrix["1"]["C"].judgment == "일부 유사"        # C는 실제로 B를 참조한다


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
