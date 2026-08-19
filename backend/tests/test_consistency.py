"""구성 간 정합성. 셀은 독립 판정이라 청구항 전체로 보면 성립할 수 없는 조합이 남는다."""
from app.consistency import (antecedent_terms, antecedents, cross_document_notes,
                             enforce_antecedents, pending_aliases, reference_warnings,
                             references)
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


def _referenced(introduces: str, refers: str) -> list[str]:
    """앞 구성이 도입한 용어를 뒤 구성이 "상기 …"로 받을 때, 잡히는 지시 어구.

    지시 어구가 어디서 끝나는지는 문법만으로 정해지지 않으므로(consistency._candidates),
    후보를 앞 구성의 문언에 대조해 고른다. 그래서 이 성질은 문장 하나만으로는 확인할 수
    없고 반드시 앞 구성과 함께 봐야 한다.
    """
    target = Claim(number=1, elements=[
        ClaimElement(label="A", importance=3, text=introduces),
        ClaimElement(label="B", importance=3, text=refers)])
    return antecedent_terms(target).get("B", [])


def test_anaphora_ignores_spacing_between_claim_and_specification():
    """같은 용어를 띄어쓰기만 달리 적는 일이 흔하다("제1반사부재" ↔ "제1 반사부재")."""
    assert antecedents(claim()) == {"B": ["A"]}
    assert _referenced("제1반사부재 및 제2반사부재를 포함하고",
                       "상기 제1 반사부재 및 상기 제2 반사부재 사이에 배치되는 광원") == [
        "제1 반사부재", "제2 반사부재"]


def test_stacked_particles_do_not_swallow_the_referenced_term():
    """"…로부터"는 조사가 겹쳐 붙는다. 하나만 끊으면 지시 어구가 "결함탐지부로"가 된다.

    앞 구성의 문언은 "…결함탐지부"라 그 어구는 어디에도 걸리지 않고, 그 구성은 지시 관계가
    아예 없는 것으로 처리된다. 상한이 걸려야 할 자리에서 조용히 안 걸리는 쪽이라 보고서만
    보고는 알 수 없다.
    """
    assert _referenced("균열 특징을 산출하는 결함탐지부",
                       "상기 결함탐지부로부터 생성된 균열정보를 입력받고") == ["결함탐지부"]
    assert _referenced("수평 구조물을 검출하는 수평구조물검출부",
                       "상기 수평구조물검출부로부터 검출된 수평 구조물을") == ["수평구조물검출부"]
    assert _referenced("층수를 읽는 문자인식부",
                       "상기 문자인식부로부터 인식된 층수와") == ["문자인식부"]


def test_a_quantifier_inside_the_phrase_does_not_cut_the_referent():
    """실측: "상기 복수의 3차원 기준점들"이 '복수의'의 '의'에서 잘려 대상이 "복수"가 됐다.

    그런 낱말을 도입한 구성은 없으므로 참조가 통째로 유실된다. 조사로 어구의 끝을 찍는 방식은
    조사가 어구 **안에** 있으면 반드시 이렇게 틀린다.
    """
    found = _referenced(
        "복수의 3차원 기준점들을 포함하는 가시 두상 영역을 추출하는 단계",
        "상기 복수의 3차원 기준점들의 위치 정보를 포함하는 메타데이터를 저장하는 단계")

    assert found == ["복수의 3차원 기준점들"]
    assert "복수" not in found


def _reworded_claim() -> Claim:
    """(E)가 세운 "가시 두상 영역"을 (G)가 "가시 두상 영상"으로 받아 적은 실측 청구항."""
    return Claim(number=1, elements=[
        ClaimElement(label="E", importance=4, text="가시 두상 영역을 추출하는 단계"),
        ClaimElement(label="F", importance=4, text="비가시 두피 영역을 추정하는 단계"),
        ClaimElement(label="G", importance=4,
                     text="상기 가시 두상 영상과 상기 비가시 두피 영역을 결합하는 단계")])


def test_a_reworded_reference_is_found_but_marked_a_guess():
    """표기가 흔들린 참조는 **찾되 확정하지 않는다.**

    어구 전체를 맞춰야 하는 방식에서는 그 참조가 없는 것이 되어, (E)가 미대응인데 (G)는 완전
    개시로 나갔다. 그렇다고 공통 부분만으로 확정하면 "가시 두상 영역/색상/영상"처럼 앞 구성이
    여럿일 때 엉뚱한 곳에 붙는다. 찾아서 보여 주되 등급은 건드리지 않는 자리가 맞다.
    """
    found = {item.source: item for item in references(_reworded_claim())["G"]}

    assert (found["E"].term, found["E"].quality) == ("가시 두상", "fuzzy")
    assert not found["E"].confirmed
    assert found["F"].quality == "direct"       # 문언이 그대로면 확정된 연결이다
    # 판정을 바꾸는 자리는 전부 antecedents()를 본다. 추측은 거기 들어가지 않는다.
    assert antecedents(_reworded_claim())["G"] == ["F"]


def test_a_common_prefix_with_two_introducers_is_ambiguous():
    """"가시 두상"만으로는 A인지 B인지 알 수 없다. 첫 번째를 골라 확정하면 안 된다."""
    target = Claim(number=1, elements=[
        ClaimElement(label="A", importance=3, text="가시 두상 영역을 추출함"),
        ClaimElement(label="B", importance=3, text="가시 두상 색상을 산출함"),
        ClaimElement(label="C", importance=3, text="상기 가시 두상 영상을 처리함")])

    found = references(target)["C"][0]

    assert found.quality == "ambiguous"
    assert found.candidates == ["A", "B"]
    assert not found.confirmed


def test_a_phrase_never_introduced_does_not_borrow_a_common_prefix():
    """"사용자 영상 모델"은 앞에서 도입된 적이 없다. "사용자 영상"이 겹친다고 확정하면 안 된다."""
    target = Claim(number=1, elements=[
        ClaimElement(label="A", importance=3, text="사용자 영상 센서 데이터를 획득함"),
        ClaimElement(label="B", importance=3, text="상기 사용자 영상 모델을 저장함")])

    found = references(target)["B"][0]

    assert (found.term, found.quality) == ("사용자 영상", "fuzzy")
    assert not found.confirmed


def test_a_longer_phrase_found_only_in_a_re_referencing_element_is_not_the_antecedent():
    """되받는 구성은 앞 구성이 쓴 어구를 통째로 품는다. 길이만 보면 그쪽이 지시 대상이 된다.

    (B)의 문언은 (D)가 쓴 "데이터수집부에 수집된 건물 외벽 영상 데이터"를 그대로 품지만, B는
    그것을 "상기 …"로 되받았을 뿐이다. 도입한 자리에는 "상기"가 없다는 것이 둘을 가른다.
    """
    links = antecedents(chained_claim())

    assert links["D"] == ["A"]
    assert antecedent_terms(chained_claim())["D"] == ["데이터수집부"]


def _reworded_matrix(uncovered: str) -> dict:
    matrix = {"1": {label: cell(label, "대응 없음" if label == uncovered else "실질적 동일",
                                quote="" if label == uncovered else "원문 발췌 문장입니다",
                                direct=label != uncovered)
                    for label in ("E", "F", "G")}}
    for label, match in matrix["1"].items():
        match.claim_number, match.label = 1, label
    return matrix


def test_a_confirmed_reference_whose_antecedent_is_uncovered_gets_capped():
    """문언이 그대로 이어진 참조(G→F)는 상한의 근거가 된다."""
    matrix = _reworded_matrix("F")

    notes = enforce_antecedents(_reworded_claim(), matrix)

    assert matrix["1"]["G"].judgment == "일부 유사"
    assert matrix["1"]["G"].downgraded_from == "실질적 동일"
    assert any("참조 구성 F" in note for note in notes)


def test_a_guessed_reference_warns_but_never_caps():
    """추측으로 등급을 내리면 맞게 개시된 구성이 근거 없이 강등된다.

    참조를 놓치는 쪽은 상한이 안 걸릴 뿐이지만, 잘못 이으면 없는 결격을 만들어 낸다. 다만
    청구항 문언 자체가 어긋나 있을 수도 있으므로 "추정했다"는 사실은 숨기지 않는다.
    """
    matrix = _reworded_matrix("E")

    assert enforce_antecedents(_reworded_claim(), matrix) == []
    assert matrix["1"]["G"].judgment == "실질적 동일"      # 건드리지 않는다
    assert matrix["1"]["G"].antecedent_note == ""
    # 다만 추정했다는 사실은 **보고서 본문에** 남는다. verify_notes로 보내면 보이지 않는다.
    warnings = reference_warnings([_reworded_claim()])
    assert any("추정" in note and "가시 두상" in note for note in warnings)


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


# --- 추측이 판정으로 새는 경로 --------------------------------------------------
# 등급 상한만 막아서는 부족하다. antecedent_terms()는 의미검증의 reference_terms가 되고,
# 프롬프트는 거기 실린 낱말을 "이미 세워 둔 대상"으로 놓은 뒤 **그 낱말이 근거에 없다는
# 이유로 기각하지 말라**고 지시한다. 추측을 실으면 오류의 방향이 근거 없는 강등에서 근거
# 없는 인정으로 바뀔 뿐이다.

def _reference_terms(claim: Claim) -> list[str]:
    """의미검증이 실제로 받는 reference_terms. 관측용 값이 아니라 평가 입력이다."""
    from app.entailment import _references
    return [term for terms in _references([claim]).values() for term in terms]


def test_a_confirmed_reference_reaches_the_entailment_input():
    assert "비가시 두피 영역" in _reference_terms(_reworded_claim())


def test_a_guessed_reference_never_reaches_the_entailment_input():
    """fuzzy 연결이 실리면 D1에 "가시 두상"이 없어도 (G)를 개시로 인정할 길이 열린다."""
    terms = _reference_terms(_reworded_claim())

    assert "가시 두상" not in terms
    assert not any("가시 두상" in term for term in terms)


def test_an_ambiguous_reference_never_reaches_the_entailment_input():
    target = Claim(number=1, elements=[
        ClaimElement(label="A", importance=3, text="가시 두상 영역을 추출함"),
        ClaimElement(label="B", importance=3, text="가시 두상 색상을 산출함"),
        ClaimElement(label="C", importance=3, text="상기 가시 두상 영상을 처리함")])

    assert _reference_terms(target) == []
    assert references(target)["C"][0].quality == "ambiguous"   # 관측은 그대로 남는다


def test_a_phrase_never_introduced_does_not_relax_the_entailment_input():
    """"사용자 영상 모델"은 도입된 적이 없다. "사용자 영상"이 겹친다고 주어진 대상이 되면 안 된다."""
    target = Claim(number=1, elements=[
        ClaimElement(label="A", importance=3, text="사용자 영상 센서 데이터를 획득함"),
        ClaimElement(label="B", importance=3, text="상기 사용자 영상 모델을 저장함")])

    assert _reference_terms(target) == []


def test_a_guess_still_warns_even_though_it_changes_nothing():
    """판정을 건드리지 않는다고 감추지는 않는다. 청구항 문언이 어긋난 신호일 수 있다."""
    warnings = reference_warnings([_reworded_claim()])

    assert any("가시 두상" in note and "추정" in note for note in warnings)


# --- 사람이 확정한 별칭 ----------------------------------------------------------
# 문언이 어긋난 참조를 도구가 자동으로 이어서는 안 된다. "가시 두상 영역" ↔ "가시 두상 영상"은
# 동의어가 아니라 오기일 수 있고, 모델에게 물으면 문맥상 거의 언제나 "같다"고 답한다. 그 답을
# 받아 이으면 도구가 청구항의 기재 문제를 대신 덮어 주는 것이지 해소가 아니다.

def _confirm(claim: Claim, term: str, source: str = "") -> Claim:
    """분해 확정 기록을 왕복시켜 별칭 하나를 확정한 청구항을 만든다.

    source를 주면 애매한 참조에서 **해소기의 기본값이 아닌 후보**를 고른 경우를 흉내 낸다.
    """
    from app.claims import dump_decomposition
    from app.models import ReferenceAlias
    record = dump_decomposition([claim])
    aliases = []
    for item in record["aliases"][str(claim.number)]:
        picked = item["term"] == term
        aliases.append(ReferenceAlias.model_validate({
            **item, "confirmed": picked,
            "selected_source": (source or item["candidates"][0]) if picked else ""}))
    return claim.model_copy(update={"aliases": aliases})


def test_a_guessed_reference_is_offered_for_confirmation_unconfirmed():
    """후보는 결정론적 해소가 확정하지 못한 것만 오르고, 기본값은 언제나 미확정이다."""
    pending = pending_aliases(_reworded_claim())

    # 후보가 하나면 고를 것이 없으므로 미리 채워 둔다 — 비워 두면 화면에서 확정 자체를
    # 할 수 없다. 채우는 것은 선택이지 확정이 아니므로 confirmed는 그대로 거짓이다.
    assert [(item.target, item.term, item.candidates, item.selected_source, item.confirmed)
            for item in pending] == [("G", "가시 두상", ["E"], "E", False)]


def test_an_unconfirmed_alias_changes_nothing():
    """확정 화면에 올랐다는 사실과 사람이 그렇다고 답했다는 사실은 다르다."""
    offered = _reworded_claim().model_copy(update={"aliases": pending_aliases(_reworded_claim())})

    assert antecedents(offered)["G"] == ["F"]
    assert "가시 두상" not in _reference_terms(offered)
    assert references(offered)["G"][0].quality == "fuzzy"


def test_a_confirmed_alias_promotes_the_link_everywhere():
    """확정하면 판정 경로 셋이 함께 열린다 — 등급 상한·의미검증 입력·차이점 서술."""
    settled = _confirm(_reworded_claim(), "가시 두상")

    assert antecedents(settled)["G"] == ["E", "F"]
    assert "가시 두상" in _reference_terms(settled)
    assert references(settled)["G"][0].quality == "confirmed_alias"
    assert references(settled)["G"][0].confirmed


def test_a_confirmed_alias_lets_the_cap_fire():
    matrix = _reworded_matrix("E")

    notes = enforce_antecedents(_confirm(_reworded_claim(), "가시 두상"), matrix)

    assert matrix["1"]["G"].judgment == "일부 유사"
    assert any("참조 구성 E" in note for note in notes)


def test_a_confirmed_alias_stops_warning_but_the_others_keep_warning():
    settled = _confirm(_reworded_claim(), "가시 두상")

    assert reference_warnings([settled]) == []
    assert reference_warnings([_reworded_claim()])            # 확정 전에는 경고가 있었다


def test_an_unreadable_alias_record_defaults_to_unconfirmed():
    """잘못 저장된 값이 확정으로 읽히면 사람이 승인하지 않은 연결이 판정을 바꾼다."""
    from app.claims import _restore_elements, dump_decomposition

    record = dump_decomposition([_reworded_claim()])
    record["aliases"]["1"] = [{"term": "가시 두상"}, "쓰레기", {"confirmed": True}]
    target = _reworded_claim()

    assert _restore_elements(target, record, strict_version=False)
    assert target.aliases == []
    assert antecedents(target)["G"] == ["F"]


def test_an_ambiguous_reference_carries_every_candidate_for_the_user_to_pick():
    """후보를 버리면 사용자가 고를 것이 사라져, 애매한 참조가 확정 아니면 폐기로 눌린다."""
    target = Claim(number=1, elements=[
        ClaimElement(label="A", importance=3, text="가시 두상 영역을 추출함"),
        ClaimElement(label="B", importance=3, text="가시 두상 색상을 산출함"),
        ClaimElement(label="C", importance=3, text="상기 가시 두상 영상을 처리함")])

    pending = pending_aliases(target)

    assert [(item.target, item.term, item.candidates) for item in pending] == [
        ("C", "가시 두상", ["A", "B"])]


def test_the_user_can_pick_a_candidate_the_resolver_did_not_default_to():
    """해소기가 기본으로 집은 후보와 사람이 고른 후보가 다를 수 있다. 물어본 이유가 그것이다."""
    target = Claim(number=1, elements=[
        ClaimElement(label="A", importance=3, text="가시 두상 영역을 추출함"),
        ClaimElement(label="B", importance=3, text="가시 두상 색상을 산출함"),
        ClaimElement(label="C", importance=3, text="상기 가시 두상 영상을 처리함")])

    assert references(target)["C"][0].source == "A"          # 해소기의 기본값
    settled = _confirm(target, "가시 두상", source="B")       # 사람은 B를 골랐다

    assert references(settled)["C"][0].source == "B"
    assert antecedents(settled)["C"] == ["B"]


def test_confirming_without_picking_a_candidate_changes_nothing():
    """확정 표시만 있고 고른 후보가 없으면 어느 구성을 가리키는지 여전히 모른다."""
    from app.models import ReferenceAlias

    half = _reworded_claim().model_copy(update={"aliases": [
        ReferenceAlias(target="G", term="가시 두상", candidates=["E"], confirmed=True)]})

    assert antecedents(half)["G"] == ["F"]
    assert "가시 두상" not in _reference_terms(half)


def test_a_selection_that_left_the_candidate_list_is_dropped():
    """청구항이 바뀌어 후보에서 빠진 선택을 되살리면, 보지 않은 연결이 확정된 채로 들어간다."""
    from app.models import ReferenceAlias

    stale = _reworded_claim().model_copy(update={"aliases": [
        ReferenceAlias(target="G", term="가시 두상", candidates=["Z"],
                       selected_source="Z", confirmed=True)]})

    assert antecedents(stale)["G"] == ["F"]
    assert pending_aliases(stale)[0].confirmed is False
