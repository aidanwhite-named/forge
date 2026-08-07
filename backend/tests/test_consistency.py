"""구성 간 정합성. 셀은 독립 판정이라 청구항 전체로 보면 성립할 수 없는 조합이 남는다."""
from app.consistency import anaphora, antecedents, enforce_antecedents
from app.models import Claim, ClaimElement, ElementMatch


def claim() -> Claim:
    return Claim(number=3, elements=[
        ClaimElement(label="A", importance=5,
                     text="상기 반사 심도부는 상호 이격되어 평행하게 배치되어 각각 부분 반사 특성을 "
                          "갖는 제1반사부재 및 제2반사부재"),
        ClaimElement(label="B", importance=4,
                     text="상기 제1 반사부재 및 상기 제2 반사부재 사이에 배치되는 광원을 포함하는 것"),
    ])


def cell(label: str, judgment: str, *, quote: str = "원문 발췌 문장입니다",
         verify: str = "verified", direct: bool = True) -> ElementMatch:
    return ElementMatch(claim_number=3, label=label, document_id="1", judgment=judgment,
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


def test_an_unjudged_cell_is_never_capped():
    """판정을 받지 못한 셀은 '대응 없음'과 다르므로 건드리지 않는다."""
    matrix = {"1": {"A": cell("A", "대응 없음", quote="", direct=False),
                    "B": cell("B", "실질적 동일")}}
    matrix["1"]["B"].error = "응답에 (B) 판정이 없습니다."

    enforce_antecedents(claim(), matrix)

    assert matrix["1"]["B"].judgment == "실질적 동일"
