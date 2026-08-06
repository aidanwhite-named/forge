"""인용발명 선정 알고리즘. 비교 매트릭스가 같으면 항상 같은 조합이 나와야 한다."""
from app.chain import build_chain, matrix_for
from app.coverage import score_document
from app.models import Claim, ClaimElement, ElementMatch


def claim(number: int = 1, depends_on: int | None = None, importances=(5, 5, 3)) -> Claim:
    labels = ["A", "B", "C", "D"][:len(importances)]
    return Claim(number=number, depends_on=depends_on,
                 elements=[ClaimElement(label=label, text=f"구성 {label}", importance=importance)
                           for label, importance in zip(labels, importances)])


def cell(document_id: str, label: str, judgment: str, *, number: int = 1, direct: bool = True,
         missing: list[str] | None = None, quote: str | None = None,
         verify: str | None = None) -> ElementMatch:
    has_quote = quote if quote is not None else ("원문 발췌 문장입니다" if judgment != "대응 없음" else "")
    return ElementMatch(claim_number=number, label=label, document_id=document_id, judgment=judgment,
                        directness="direct" if direct else "inferred",
                        quote=has_quote,
                        chunk_id=f"D{document_id}-P-0001" if judgment != "대응 없음" else "",
                        verify=verify or ("verified" if has_quote else "empty"),
                        missing_limitations=missing or [])


def coverage_of(chain, label: str):
    return next(item for item in chain.element_coverage if item.label == label)


def build(target: Claim, matches: list[ElementMatch], parents=None, all_claims=None):
    return build_chain(target, matrix_for(matches), parents or {}, all_claims or [target])


def test_single_document_full_disclosure_stops_before_any_combination():
    """이 게이트를 통과하면 문헌 결합과 이후 판정 단계를 전부 건너뛴다."""
    matches = [cell("1", label, "동일") for label in "ABC"] + [cell("2", label, "일부 유사") for label in "ABC"]
    chain = build(claim(), matches)
    assert chain.track == "novelty_single"
    assert chain.primary == "1" and chain.secondaries == []


def test_a_missing_limitation_blocks_the_novelty_gate():
    matches = [cell("1", "A", "동일"), cell("1", "B", "동일", missing=["온도 임계값 한정"]), cell("1", "C", "동일")]
    chain = build(claim(), matches)
    assert chain.track != "novelty_single"
    assert chain.novelty.missing_by_document["1"] == ["B"]


def test_inferred_disclosure_does_not_satisfy_the_novelty_gate():
    matches = [cell("1", label, "동일", direct=(label != "B")) for label in "ABC"]
    assert build(claim(), matches).track != "novelty_single"


def test_secondary_is_chosen_by_increment_not_absolute_strength():
    """전체 점수가 높아도 주 인용발명의 공백을 못 채우면 채택하지 않는다."""
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "대응 없음")]
        + [cell("2", "A", "실질적 동일"), cell("2", "B", "실질적 동일"), cell("2", "C", "대응 없음")]
        + [cell("3", "A", "대응 없음"), cell("3", "B", "대응 없음"), cell("3", "C", "동일")]
    )
    chain = build(claim(), matches)
    assert chain.primary == "1"
    assert chain.secondaries == ["3"]          # 2가 아니라 공백 C를 메우는 3
    assert chain.uncovered == []
    assert chain.track == "inventive_step_combination"


def test_core_gap_that_no_document_fills_makes_the_rejection_impossible():
    """'차이' 판정은 라벨 정의상 개시로 보지 않으므로 공백이 그대로 남는다."""
    matches = ([cell("1", "A", "동일"), cell("1", "B", "대응 없음"), cell("1", "C", "동일")]
               + [cell("2", "A", "일부 유사"), cell("2", "B", "차이"), cell("2", "C", "일부 유사")])
    chain = build(claim(), matches)
    assert chain.track == "rejection_impossible"
    assert chain.uncovered == ["B"]
    assert chain.secondaries == []


def test_a_weak_but_real_correspondence_is_a_difference_not_a_gap():
    """세부 구현이 다르다는 이유로 "대응 기재가 없다"고 보고하지 않는다.

    주 인용발명이 통째로 놓친 구성의 유일한 대응 기재를 가진 문헌은, 판정이 '일부 차이'라도
    결합 후보에서 탈락하지 않는다. 남은 차이는 uncovered가 아니라 residual로 넘어간다.
    """
    matches = ([cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "대응 없음")]
               + [cell("2", "C", "일부 차이", direct=False, missing=["디스패리티 추정 방식"])])
    chain = build(claim(), matches)
    assert chain.secondaries == ["2"]
    assert chain.uncovered == []
    assert "C" in chain.residual
    assert chain.track == "inventive_step_combination"


def test_a_low_importance_gap_is_still_reported_as_a_gap():
    """중요도가 낮다는 이유만으로 미개시 구성을 결론에서 빼내지 않는다.

    종전에는 중요도 2 이하를 '주지관용 검토'로 분리했다. 분리 기준이 중요도 하나뿐이라
    실제로 인정된 것은 아무것도 없는데, 보고서에는 별도 절이 생겨 미개시 사실이 흐려졌다.
    """
    target = claim(importances=(5, 5, 2))
    matches = [cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "대응 없음")]
    chain = build(target, matches, all_claims=[target])
    assert chain.track == "rejection_impossible"
    assert chain.uncovered == ["C"]


def test_dependent_claim_does_not_pass_novelty_on_its_added_limitation_alone():
    """종속항 추가 한정만 단일 문헌에 있어도 부모 구성 없이 신규성을 부정하지 않는다."""
    parent = claim(1, importances=(5,))
    child = claim(2, depends_on=1, importances=(5,))
    parent_chain = build(parent, [cell("1", "A", "동일")], all_claims=[parent, child])
    child_matches = [
        cell("1", "A", "대응 없음", number=2),
        cell("2", "A", "동일", number=2),
    ]

    chain = build(child, child_matches, {1: parent_chain}, [parent, child])

    assert chain.track != "novelty_single"
    assert chain.inherited == ["1"]
    assert chain.added == "2"


def test_a_partially_disclosed_element_records_what_is_still_missing():
    """부분 대응이 있어도 남은 하위 한정은 그대로 적어 둔다. 요약에서 사라지면 안 된다.

    대응 기재가 있으므로 공백(uncovered)이 아니라 차이점(residual)으로 넘어가고,
    누락된 하위 한정은 차이점 목록에 문장으로 남는다.
    """
    target = claim(importances=(5, 5, 2))
    matches = [cell("1", "A", "동일"), cell("1", "B", "동일"),
               cell("1", "C", "일부 차이", missing=["가상현실 환경", "제스처 수집"])]
    chain = build(target, matches, all_claims=[target])
    assert chain.uncovered == [] and chain.residual == ["C"]

    residual = coverage_of(chain, "C").residual_difference
    assert "가상현실 환경" in residual and "제스처 수집" in residual


def test_dependent_claim_inherits_the_parent_chain_and_adds_one_document():
    parent, child = claim(1), claim(2, depends_on=1)
    parent_matches = ([cell("1", label, "동일") for label in "AB"] + [cell("1", "C", "대응 없음")]
                      + [cell("2", label, "대응 없음") for label in "AB"] + [cell("2", "C", "동일")])
    parent_chain = build(parent, parent_matches, all_claims=[parent, child])

    child_matches = ([cell("1", label, "동일", number=2) for label in "AB"] + [cell("1", "C", "대응 없음", number=2)]
                     + [cell("2", label, "대응 없음", number=2) for label in "AB"] + [cell("2", "C", "대응 없음", number=2)]
                     + [cell("3", label, "대응 없음", number=2) for label in "AB"] + [cell("3", "C", "동일", number=2)])
    chain = build(child, child_matches, {1: parent_chain}, [parent, child])
    assert chain.inherited == ["1", "2"]
    assert chain.added == "3"
    assert chain.uncovered == []


def test_dependent_claim_never_adds_two_new_documents():
    """하나의 종속항 거절을 위해 새 문헌을 2개 이상 추가하지 않는다."""
    parent, child = claim(1, importances=(5, 5, 5)), claim(2, depends_on=1, importances=(5, 5, 5))
    parent_chain = build(parent, [cell("1", label, "동일") for label in "AB"] + [cell("1", "C", "대응 없음")],
                         all_claims=[parent, child])
    child_matches = (
        [cell("1", label, "동일", number=2) for label in "AB"] + [cell("1", "C", "대응 없음", number=2)]
        + [cell("2", "A", "대응 없음", number=2), cell("2", "B", "동일", number=2), cell("2", "C", "대응 없음", number=2)]
        + [cell("3", "A", "대응 없음", number=2), cell("3", "B", "대응 없음", number=2), cell("3", "C", "동일", number=2)]
    )
    chain = build(child, child_matches, {1: parent_chain}, [parent, child])
    assert chain.added in {None, "3"}
    assert len([document for document in chain.secondaries if document not in chain.inherited]) <= 1


def test_a_document_that_directly_discloses_a_core_element_stays_a_candidate():
    """평균 점수가 낮아도 핵심 구성을 원문으로 직접 개시한 문헌은 후보에서 탈락시키지 않는다."""
    target = claim(importances=(5, 3, 3))
    matches = ([cell("1", "A", "일부 유사"), cell("1", "B", "동일"), cell("1", "C", "동일")]
               + [cell("2", "A", "동일"), cell("2", "B", "대응 없음"), cell("2", "C", "대응 없음")])
    chain = build(target, matches, all_claims=[target])
    assert "2" in [candidate.document_id for candidate in chain.candidates]
    assert chain.primary == "2"                # 핵심 A를 직접 개시한 문헌이 주 인용발명


def test_the_same_matrix_always_produces_the_same_chain():
    matches = ([cell("1", "A", "동일"), cell("1", "B", "실질적 동일"), cell("1", "C", "대응 없음")]
               + [cell("2", "A", "일부 차이"), cell("2", "B", "일부 차이"), cell("2", "C", "동일")])
    first, second = build(claim(), matches), build(claim(), matches)
    assert first.model_dump() == second.model_dump()


# --- 보완 인용발명 반영 --------------------------------------------------------

def test_partially_disclosed_element_is_supplemented_by_another_document():
    """사례 1. '일부 차이'로 커버된 구성도 더 나은 대응이 있으면 보완 문헌으로 채운다."""
    matches = ([cell("1", "A", "동일"), cell("1", "B", "일부 차이"), cell("1", "C", "대응 없음")]
               + [cell("2", "A", "대응 없음"), cell("2", "B", "동일"), cell("2", "C", "실질적 동일")])
    chain = build(claim(), matches)
    assert chain.primary == "1"
    assert chain.secondaries == ["2"]
    assert chain.supplement_needed == ["B", "C"]           # 공백 C뿐 아니라 부분 대응 B도 검토 대상
    assert coverage_of(chain, "B").adopted_document == "2"
    assert coverage_of(chain, "C").adopted_document == "2"
    assert coverage_of(chain, "A").adopted_document == "1"
    assert chain.uncovered == []


def test_a_document_with_fewer_missing_limitations_wins_at_the_same_judgment():
    """사례 2. 판정이 같아도 누락 한정이 적은 문헌이 더 나은 보완 근거다."""
    matches = ([cell("1", "A", "동일"), cell("1", "B", "일부 차이", missing=["온도 범위", "주기 조건"]),
                cell("1", "C", "동일")]
               + [cell("2", "A", "대응 없음"), cell("2", "B", "일부 차이"), cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)
    assert chain.primary == "1"
    assert coverage_of(chain, "B").adopted_document == "2"
    assert coverage_of(chain, "B").primary_missing == ["온도 범위", "주기 조건"]
    assert "2" in chain.secondaries


def test_a_weaker_judgment_is_never_adopted_as_a_supplement():
    """사례 3. 점수가 더 낮은 문헌을 억지로 보완 문헌으로 끌어오지 않는다."""
    matches = ([cell("1", "A", "동일"), cell("1", "B", "일부 차이"), cell("1", "C", "동일")]
               + [cell("2", "A", "대응 없음"), cell("2", "B", "일부 유사"), cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)
    assert chain.secondaries == []
    assert coverage_of(chain, "B").adopted_document == "1"


def test_an_unverified_supplement_is_rejected_but_its_reason_is_recorded():
    """사례 4. 판정만 높고 발췌·검증이 없는 문헌은 채택하지 않되 탈락 사유를 남긴다."""
    matches = ([cell("1", "A", "동일"), cell("1", "B", "일부 차이"), cell("1", "C", "동일")]
               + [cell("2", "A", "대응 없음"),
                  cell("2", "B", "동일", quote="", verify="empty"),
                  cell("2", "C", "동일", quote="지어낸 발췌", verify="not_found")])
    chain = build(claim(), matches)
    assert chain.secondaries == []
    assert coverage_of(chain, "B").adopted_document == "1"
    rejected = {candidate.document_id: candidate for candidate in coverage_of(chain, "B").candidates}
    assert rejected["2"].eligible is False
    assert rejected["2"].rejected_reason == "원문 발췌 없음"
    # 판정 라벨만 보면 더 높지만 근거 자격이 없다는 사실을 둘 다 남긴다. 채택은 자격이 막는다.
    assert rejected["2"].better_than_primary is True
    assert rejected["2"].adopted is False
    assert {candidate.document_id: candidate.rejected_reason
            for candidate in coverage_of(chain, "C").candidates}["2"] == "발췌 검증 실패(not_found)"


def test_an_unverified_stronger_label_never_displaces_a_verified_match():
    """근거 없는 '동일'이 검증된 '일부 차이'를 밀어내면 안 된다."""
    from app.coverage import best_match
    verified = cell("1", "B", "일부 차이")
    unverified = cell("2", "B", "동일", quote="지어낸 발췌", verify="not_found")
    assert best_match([verified, unverified]).document_id == "1"
    assert best_match([unverified, verified]).document_id == "1"


def test_an_unsupported_difference_does_not_displace_the_current_gap():
    """둘 다 근거 자격이 없으면 보조 문헌의 라벨만 보고 채택 문헌을 바꾸지 않는다."""
    from app.coverage import best_match
    current = cell("1", "B", "대응 없음")
    unsupported = cell("2", "B", "차이", quote="", verify="empty")
    unsupported.directness = "absent"

    assert best_match([current, unsupported]).document_id == "1"


def test_documents_filling_different_gaps_are_all_combined():
    """사례 5. 서로 다른 공백의 유일한 근거를 가진 문헌은 개수와 무관하게 모두 결합한다.

    결합 문헌 수를 2건으로 묶어 두면, 세 번째 문헌이 어떤 구성의 **유일한** 검증 근거를
    가지고 있어도 통째로 버려지고 그 구성이 "어느 인용발명에도 대응이 없다"로 보고된다.
    업로드된 문헌에 기재가 있는데 그렇게 적는 것은 사실과 다르다.
    """
    target = claim(importances=(5, 5, 4, 4))
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "대응 없음"), cell("1", "D", "대응 없음")]
        + [cell("2", label, "대응 없음") for label in "ABD"] + [cell("2", "C", "동일")]
        + [cell("3", label, "대응 없음") for label in "ABC"] + [cell("3", "D", "동일")]
    )
    chain = build(target, matches, all_claims=[target])
    assert chain.primary == "1" and sorted(chain.secondaries) == ["2", "3"]
    assert chain.uncovered == []
    assert coverage_of(chain, "C").adopted_document == "2"
    assert coverage_of(chain, "D").adopted_document == "3"
    # 모든 문헌의 구성별 대응은 채택 여부와 무관하게 전부 분석된다.
    assert {candidate.document_id for candidate in coverage_of(chain, "D").candidates} == {"1", "2", "3"}


def test_a_core_gap_is_filled_by_a_third_document_too():
    """차별적 핵심 구성이라고 해서 세 번째 문헌의 직접 근거를 버리지 않는다."""
    target = claim(importances=(5, 3, 3, 4))                    # D가 핵심, 마지막에 남도록 구성
    matches = (
        [cell("1", "A", "동일")] + [cell("1", label, "대응 없음") for label in "BCD"]
        + [cell("2", "A", "대응 없음"), cell("2", "B", "동일"), cell("2", "C", "동일"),
           cell("2", "D", "대응 없음")]
        + [cell("3", label, "대응 없음") for label in "ABC"] + [cell("3", "D", "동일")]
    )
    chain = build(target, matches, all_claims=[target])
    assert chain.primary == "1" and chain.secondaries == ["2", "3"]  # B·C를 한 번에 메우는 2가 먼저
    assert chain.uncovered == []


def test_combination_stops_when_a_document_adds_nothing():
    """이득이 없으면 문헌 수 상한이 없어도 결합이 멈춘다. 상한 대신 이득이 제동을 건다."""
    matches = ([cell("1", label, "동일") for label in "ABC"]
               + [cell("2", label, "일부 유사") for label in "ABC"]
               + [cell("3", label, "차이") for label in "ABC"])
    chain = build(claim(), matches)
    assert chain.secondaries == []


def test_supplement_analysis_runs_even_when_nothing_is_uncovered():
    """공백이 없어도 보완 검토를 끝내지 않는다. 이전 구현은 여기서 조기 종료했다."""
    matches = [cell("1", label, "일부 차이") for label in "ABC"]
    chain = build(claim(), matches)
    assert chain.uncovered == []
    assert chain.supplement_needed == ["A", "B", "C"]
    assert chain.residual == ["A", "B", "C"]
    assert "전 구성이 커버" not in chain.rationale
    assert "완전 개시되지 않아" in chain.rationale
    assert "A, B, C" in chain.rationale


def test_core_direct_disclosure_outranks_broad_shallow_coverage():
    target = claim(importances=(5, 5, 1))
    strong_core = {label: cell("1", label, judgment) for label, judgment
                   in zip("ABC", ["동일", "동일", "대응 없음"])}
    broad_shallow = {label: cell("2", label, judgment) for label, judgment
                     in zip("ABC", ["일부 차이", "일부 차이", "동일"])}
    assert score_document(target, strong_core)[0] > score_document(target, broad_shallow)[0]


def test_sparse_core_matches_do_not_collapse_to_a_document_id_tie():
    """핵심 공백이 많아도 부분 대응의 개수 차이는 주 인용발명 순위에 남아야 한다."""
    target = claim(importances=(5, 5, 4, 4, 4, 4))
    one_core_match = {label: cell("1", label, "일부 차이" if label == "A" else "대응 없음")
                      for label in "ABCDEF"}
    two_core_matches = {label: cell("2", label, "일부 차이" if label in "AB" else "대응 없음")
                        for label in "ABCDEF"}

    first_score = score_document(target, one_core_match)[0]
    second_score = score_document(target, two_core_matches)[0]

    assert 0 < first_score < second_score
    chain = build(target, list(one_core_match.values()) + list(two_core_matches.values()))
    assert chain.primary == "2"


# --- 미판정과 '대응 없음'의 분리 -----------------------------------------------

def test_unjudged_cell_never_becomes_a_no_correspondence_conclusion():
    """LLM 호출 실패는 '선행문헌에 그런 기재가 없다'가 아니다.

    이 둘을 섞으면 분석 실패가 출원인에게 유리한 결론으로 나가고, 유리한 결론이라
    검토 과정에서 이의가 제기되지도 않는다.
    """
    matches = [cell("1", label, "대응 없음") for label in "ABC"]
    for match in matches:
        match.error = "D1.pdf 비교 호출 실패: CLI 실행 시간이 초과되었습니다."
    chain = build(claim(), matches)
    assert chain.track == "analysis_incomplete"
    assert chain.uncovered == []          # 대응 기재가 없다고 단정하지 않는다
    assert chain.primary is None
    assert chain.incomplete_reasons


def test_partial_verification_never_satisfies_the_novelty_gate():
    """partial은 다분절 인용문의 일부만 원문에서 확인됐다는 뜻이다.

    verify.py가 partial을 반드시 inferred로 강등하므로 정상 경로에서는 directness에서도
    걸리지만, 신규성 게이트가 그 불변식에 의존하지 않는지를 여기서 고정한다.
    """
    matches = [cell("1", label, "동일", verify="partial" if label == "B" else "verified")
               for label in "ABC"]
    assert build(claim(), matches).track != "novelty_single"


def test_no_primary_is_named_when_nothing_is_directly_disclosed():
    """0점 문헌이 마진만으로 자격을 얻어 무관한 문헌이 주 인용발명으로 찍히면 안 된다."""
    chain = build(claim(), [cell("1", label, "대응 없음") for label in "ABC"]
                  + [cell("2", label, "대응 없음") for label in "ABC"])
    assert chain.primary is None
    assert chain.track == "rejection_impossible"


def test_incomplete_parent_blocks_the_dependent_conclusion():
    parent_matches = [cell("1", label, "대응 없음") for label in "ABC"]
    for match in parent_matches:
        match.error = "비교 호출 실패"
    parent_claim, child_claim = claim(1), claim(2, depends_on=1, importances=(4,))
    parents = {1: build(parent_claim, parent_matches)}
    child = build(child_claim, [cell("1", "A", "동일", number=2)], parents=parents,
                  all_claims=[parent_claim, child_claim])
    assert child.track == "analysis_incomplete"


# --- 종속항 트랙 ---------------------------------------------------------------

def test_single_document_covering_the_dependent_addition_stays_on_novelty():
    """부모항을 단독 개시한 문헌이 추가 한정까지 개시하면 결합을 논할 단계가 아니다.

    무조건 진보성으로 넘기면 "제29조제2항 — 단일 인용발명"이라는 자기모순 라벨이 나온다.
    """
    parent_claim, child_claim = claim(1), claim(2, depends_on=1, importances=(4,))
    parents = {1: build(parent_claim, [cell("1", label, "동일") for label in "ABC"])}
    assert parents[1].track == "novelty_single"
    child = build(child_claim, [cell("1", "A", "동일", number=2)], parents=parents,
                  all_claims=[parent_claim, child_claim])
    assert child.track == "novelty_single"
    assert child.primary == "1" and child.secondaries == []


def test_dependent_addition_absent_from_the_parent_document_moves_to_combination():
    parent_claim, child_claim = claim(1), claim(2, depends_on=1, importances=(4,))
    parents = {1: build(parent_claim, [cell("1", label, "동일") for label in "ABC"])}
    child = build(child_claim,
                  [cell("1", "A", "대응 없음", number=2), cell("2", "A", "동일", number=2)],
                  parents=parents, all_claims=[parent_claim, child_claim])
    assert child.track == "inventive_step_combination"
    assert child.added == "2"
