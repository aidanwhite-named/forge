"""인용발명 선정 알고리즘. 비교 매트릭스가 같으면 항상 같은 조합이 나와야 한다."""
from app.chain import build_chain, matrix_for, merge_selected
from app.coverage import evidence_locations, limitation_counts, report_grade, score_document
from app.models import Claim, ClaimElement, ElementMatch, LimitationCheck  # noqa: F401


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


WAVEGUIDE = "출력 광학 이미지를 도파관 광학 시스템을 통해 출력되는 이미지로 한정함"


def disclosing(document_id: str, label: str, judgment: str, discloses: list[str], *,
               missing: list[str] | None = None, direct: bool = True) -> ElementMatch:
    """하위 한정 개시까지 채운 셀. disclosed_limitations가 읽는 것은 limitation_checks다."""
    match = cell(document_id, label, judgment, direct=direct, missing=missing)
    match.limitation_checks = [
        LimitationCheck(index=index, limitation=text, kind="qualifier", disclosed=True,
                        quote=match.quote, chunk_id=match.chunk_id, verify="verified")
        for index, text in enumerate(discloses)
    ] + [
        LimitationCheck(index=len(discloses) + index, limitation=text, kind="qualifier",
                        disclosed=False, verify="empty")
        for index, text in enumerate(missing or [])
    ]
    return match


def test_a_secondary_that_only_fills_a_missing_limitation_is_still_adopted():
    """부 인용발명은 구성 전체를 주 인용발명보다 잘 개시하는 문헌이 아니다.

    진보성 결합에서 부 인용발명을 데려오는 이유는 주 인용발명이 빠뜨린 한정을 대는 것이다.
    그러므로 구성 단위 우열(is_better_match) 하나로 문을 지키면, 주 인용발명이 '일부 차이'만
    되어도 보완이 사실상 불가능해진다 — 한정 하나를 대는 문헌은 구성 전체로는 등급이 낮은
    것이 정상이기 때문이다.

    실측 사건: 주 인용발명이 (A)를 '일부 차이'(rank 3)로 덮고 도파관 한정 하나만 빠뜨렸는데,
    그 한정을 원문으로 개시한 문헌 3건이 구성 전체로는 '차이'(rank 1)라 전부 탈락했다.
    결과는 1건짜리 조합과 "다른 문헌에서도 이 차이를 완전히 해소하는 더 강한 직접 근거는
    확인하지 못했습니다"라는 결론이었다.
    """
    matches = ([disclosing("1", "A", "일부 차이", ["이미지 세트를 획득함"], missing=[WAVEGUIDE]),
                cell("1", "B", "일부 차이"), cell("1", "C", "일부 차이")]
               + [disclosing("2", "A", "차이", [WAVEGUIDE], direct=False),
                  cell("2", "B", "대응 없음"), cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)
    assert chain.primary == "1"
    assert chain.secondaries == ["2"]
    candidate = next(item for item in coverage_of(chain, "A").candidates
                     if item.document_id == "2")
    assert candidate.adopted is True
    assert candidate.gain > 0.0
    assert candidate.better_than_primary is False


def test_a_limitation_filled_by_the_combination_is_not_reported_as_a_remaining_difference():
    """조합 안의 다른 문헌이 그 한정을 원문으로 댔다면 남은 차이로 적지 않는다.

    best_match는 구성 하나를 문헌 하나에 통째로 넘기므로, 진 쪽 셀이 개시한 한정까지 함께
    버려진다. 그러면 그 한정을 개시한 문헌을 같은 조합 안에 세워 두고도 "결합 후에도 남는
    차이"로 적히고, 선행기술 검색 대상까지 된다.

    등급은 올리지 않는다. 다만 한정이 전부 메워졌다면 커버리지 기준의 잔여 차이에서는 뺀다.
    결합 동기·용이성은 이 도구가 확인하지 않으며 보고서 결론에서 별도로 유보한다.
    """
    matches = ([disclosing("1", "A", "일부 차이", ["이미지 세트를 획득함"], missing=[WAVEGUIDE]),
                cell("1", "B", "일부 차이"), cell("1", "C", "일부 차이")]
               + [disclosing("2", "A", "차이", [WAVEGUIDE], direct=False),
                  cell("2", "B", "대응 없음"), cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)

    coverage = coverage_of(chain, "A")
    assert WAVEGUIDE not in coverage.residual_difference
    assert coverage.adopted_judgment == "일부 차이"      # 한정은 메워도 등급은 그대로
    assert "A" not in chain.residual
    merged = merge_selected(claim(), matrix_for(matches), [chain.primary, *chain.secondaries])
    assert limitation_counts(merged["A"]) == (2, 2)
    # 조합이 메운 한정은 미채택 문헌 목록(추가 검색 신호)에도 남지 않는다.
    assert WAVEGUIDE not in chain.beyond_limit_residual.get("A", {})


def test_a_later_winning_cell_keeps_limitations_resolved_by_the_existing_combination():
    """세 번째 문헌이 대표 셀을 바꿔도 앞선 조합이 해소한 한정은 다시 살아나지 않는다.

    종속항은 부모 조합 두 건에 새 문헌 한 건을 더할 수 있습니다. 1+2 병합에서 해소한 한정을
    3번 문헌도 빠뜨렸지만 직접성이 더 높아 대표 셀이 3번으로 바뀌는 경우, 2번의 해소 출처를
    넘기지 않으면 missing이 되살아납니다.
    """
    target = claim(importances=(5,))
    matches = [
        disclosing("1", "A", "일부 차이", ["이미지 세트를 획득함"],
                   missing=[WAVEGUIDE], direct=False),
        disclosing("2", "A", "차이", [WAVEGUIDE], direct=False),
        disclosing("3", "A", "일부 차이", ["이미지 세트를 획득함"],
                   missing=[WAVEGUIDE], direct=True),
    ]
    merged = merge_selected(target, matrix_for(matches), ["1", "2", "3"])["A"]

    assert merged.document_id == "3"
    assert merged.missing_limitations == []
    assert merged.combination_resolved == {WAVEGUIDE: "2"}
    assert limitation_counts(merged) == (2, 2)


def test_equal_contributors_are_broken_by_closeness_to_the_claim_not_document_order():
    """같은 한정을 여러 문헌이 개시하면 증분 이득이 같아진다. 그때 번호로 고르지 않는다.

    실측 사건에서 도파관 한정이 문헌 3건에 있었다. 번호가 빠른 것을 집으면 청구항과 거의
    무관한 문헌(격자 구조 특허)이 번호만 앞선다는 이유로 부 인용발명이 되고, 같은 한정을
    개시하면서 다른 구성까지 걸치는 문헌은 밀려난다. 기여가 같다면 청구항에 전체적으로 더
    가까운 문헌을 세우는 쪽이 더 방어 가능한 거절 이유다.
    """
    matches = ([disclosing("1", "A", "일부 차이", ["이미지 세트를 획득함"], missing=[WAVEGUIDE]),
                cell("1", "B", "일부 차이"), cell("1", "C", "일부 차이")]
               # 2번: 도파관 한정만 대고 나머지는 대응 없음 — 청구항과 먼 문헌.
               + [disclosing("2", "A", "차이", [WAVEGUIDE], direct=False),
                  cell("2", "B", "대응 없음"), cell("2", "C", "대응 없음")]
               # 3번: 같은 도파관 한정을 대면서 나머지 구성에도 대응이 있는 문헌.
               + [disclosing("3", "A", "차이", [WAVEGUIDE], direct=False),
                  cell("3", "B", "일부 유사"), cell("3", "C", "일부 유사")])
    chain = build(claim(), matches)
    assert chain.primary == "1"
    assert chain.secondaries == ["3"]

    # 동률이 완전히 같으면(적합도까지) 문헌 번호 순이라 결과는 항상 재현된다.
    twins = ([disclosing("1", "A", "일부 차이", ["이미지 세트를 획득함"], missing=[WAVEGUIDE]),
              cell("1", "B", "일부 차이"), cell("1", "C", "일부 차이")]
             + [disclosing("2", "A", "차이", [WAVEGUIDE], direct=False),
                cell("2", "B", "대응 없음"), cell("2", "C", "대응 없음")]
             + [disclosing("3", "A", "차이", [WAVEGUIDE], direct=False),
                cell("3", "B", "대응 없음"), cell("3", "C", "대응 없음")])
    assert build(claim(), twins).secondaries == ["2"]


def test_the_report_only_blames_the_combination_limit_when_it_actually_bound():
    """자리가 남아 있는데 빠진 문헌을 두고 "상한을 넘었다"고 적지 않는다.

    실측 보고서가 1건짜리 조합을 세워 놓고 "결합 문헌 수 상한(2건)을 넘어 이 거절 이유에는
    세우지 않음"이라고 적었다. 상한은 2건인데 조합에는 1건뿐이었으니 넘긴 것이 없다. 읽는
    사람은 상한만 올리면 그 문헌이 들어온다고 읽지만, 실제로 문헌을 떨어뜨린 것은 보완 후보
    평가였다.
    """
    # 2번이 공백 C를 메운 뒤에도 3번이 B의 누락 한정을 메울 수 있지만 자리가 없다.
    matches = ([cell("1", "A", "동일"),
                disclosing("1", "B", "일부 차이", ["기본 동작"], missing=[WAVEGUIDE]),
                cell("1", "C", "대응 없음")]
               + [cell("2", "A", "대응 없음"), cell("2", "B", "대응 없음"),
                  cell("2", "C", "동일")]
               + [cell("3", "A", "대응 없음"),
                  disclosing("3", "B", "차이", [WAVEGUIDE], direct=False),
                  cell("3", "C", "대응 없음")])
    assert build(claim(), matches).limit_binding is True

    # 후보가 정확히 두 건이고 둘 다 채택됐다면 상한에 닿았어도 배제된 후보는 없다.
    filled = ([cell("1", "A", "동일"), cell("1", "B", "동일"),
               cell("1", "C", "대응 없음")]
              + [cell("2", "A", "대응 없음"), cell("2", "B", "대응 없음"),
                 cell("2", "C", "동일")])
    assert build(claim(), filled).limit_binding is False

    # 후보가 아예 없어 조합이 1건에 머문 상태 — 상한 탓이 아니다.
    alone = [cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "일부 차이")]
    chain = build(claim(), alone)
    assert chain.secondaries == []
    assert chain.limit_binding is False

    # 문헌이 상한보다 적으면 조합이 짧아도 상한 탓이 아니다.
    two = ([cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "일부 차이")]
           + [cell("2", "A", "차이"), cell("2", "B", "차이"), cell("2", "C", "차이")])
    assert build(claim(), two).limit_binding is False


def test_a_dependent_claim_measures_its_limit_against_added_documents_only():
    """종속항 상한은 새로 더한 문헌 수에 걸린다. 상속한 문헌은 그 상한의 대상이 아니다.

    조합 전체 크기와 비교하면 부모항에서 문헌 2건을 상속받는 것만으로 언제나 "상한을
    넘었다"가 되어, 보고서가 실제로 하지 않은 판단을 한 것처럼 적는다.
    """
    parent = claim(1, importances=(5, 5, 3))
    child = claim(2, depends_on=1, importances=(5,))
    matches = ([cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "대응 없음")]
               + [cell("2", "C", "동일")])
    parent_chain = build(parent, matches, all_claims=[parent, child])
    assert parent_chain.secondaries == ["2"]

    # 종속항은 부모 조합 2건을 상속하지만 새로 더한 문헌은 없다.
    child_chain = build_chain(child, matrix_for([cell("1", "A", "동일")]),
                              {1: parent_chain}, [parent, child])
    assert child_chain.added == []
    assert child_chain.limit_binding is False


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
    assert chain.added == ["2"]


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
    assert chain.added == ["3"]
    assert chain.uncovered == []


def test_dependent_claim_stops_when_no_document_fills_a_remaining_gap():
    """상수 상한이 아니라 보완 이득으로 멈춘다. 공백에 기여하지 못하는 문헌은 붙지 않는다."""
    parent, child = claim(1, importances=(5, 5, 5)), claim(2, depends_on=1, importances=(5, 5, 5))
    parent_chain = build(parent, [cell("1", label, "동일") for label in "AB"] + [cell("1", "C", "대응 없음")],
                         all_claims=[parent, child])
    child_matches = (
        [cell("1", label, "동일", number=2) for label in "AB"] + [cell("1", "C", "대응 없음", number=2)]
        + [cell("2", "A", "대응 없음", number=2), cell("2", "B", "동일", number=2), cell("2", "C", "대응 없음", number=2)]
        + [cell("3", "A", "대응 없음", number=2), cell("3", "B", "대응 없음", number=2), cell("3", "C", "동일", number=2)]
    )
    chain = build(child, child_matches, {1: parent_chain}, [parent, child])
    assert chain.added == ["3"]                  # 유일한 공백 C를 메우는 문헌만 붙는다
    assert "2" not in chain.secondaries          # B는 이미 커버되어 새로 기여하는 것이 없다
    assert chain.uncovered == []


def test_dependent_claim_adds_at_most_one_document():
    """종속항이 부모 조합에 새로 더할 수 있는 문헌은 1건이다.

    한 줄짜리 추가 한정 하나를 위해 문헌을 여러 건 끌어오면 거절 이유가 실무에서 설득력을
    잃는다. 다만 상한 때문에 뺀 문헌의 기재를 "없다"고 적으면 사실과 다르므로, 그 구성은
    beyond_limit에 근거 문헌과 함께 남겨 uncovered의 다른 항목과 구별한다.
    """
    parent, child = claim(1, importances=(5,)), claim(2, depends_on=1, importances=(4, 4))
    parents = {1: build(parent, [cell("1", "A", "동일")], all_claims=[parent, child])}
    child_matches = [cell("1", "A", "대응 없음", number=2), cell("1", "B", "대응 없음", number=2),
                     cell("2", "A", "동일", number=2), cell("2", "B", "대응 없음", number=2),
                     cell("3", "A", "대응 없음", number=2), cell("3", "B", "동일", number=2)]

    chain = build(child, child_matches, parents=parents, all_claims=[parent, child])

    assert chain.added == ["2"]
    assert chain.combination_limit == 1
    assert chain.beyond_limit == ["B"] and chain.beyond_limit_documents["B"] == ["3"]
    # "어느 인용발명에서도 확인되지 않았다"고 적지 않는다. 기재는 있고 한도를 넘었을 뿐이다.
    assert "어느 인용발명에서도 확인되지 않았습니다" not in chain.rationale
    assert "상한(1건)" in chain.rationale


_GUIDANCE = "길 안내 정보를 제공함"


def _partial(document_id: str, label: str, *, found: bool, number: int = 1) -> ElementMatch:
    """한정 2개 중 하나만 개시된 셀. found면 나머지 하나(_GUIDANCE)까지 개시한다."""
    return ElementMatch(
        claim_number=number, label=label, document_id=document_id,
        judgment="실질적 동일" if found else "일부 차이", directness="direct",
        quote="원문 발췌 문장입니다", chunk_id=f"D{document_id}-P-0001", verify="verified",
        missing_limitations=[] if found else [_GUIDANCE],
        limitation_checks=[
            LimitationCheck(index=0, kind="core", limitation="체적 영상을 표시함", disclosed=True,
                            quote="원문 발췌 문장입니다", chunk_id=f"D{document_id}-P-0001",
                            verify="verified"),
            LimitationCheck(index=1, kind="core", limitation=_GUIDANCE, disclosed=found,
                            quote="원문 발췌 문장입니다" if found else "",
                            chunk_id=f"D{document_id}-P-0001" if found else "",
                            verify="verified" if found else "empty"),
        ])


def test_a_remaining_limitation_that_an_over_limit_document_discloses_is_not_a_plain_gap():
    """차이점 줄도 미대응 줄과 같은 규율을 따른다 — 손에 든 문헌의 기재를 "없다"고 적지 않는다.

    구성 전체가 아니라 **하위 한정 하나**가 결합 한도 밖 문헌에 있는 경우다. 실측에서 전제부의
    "길 안내 정보를 제공함"이 그냥 남은 차이로 적혔는데, 그 한정을 원문으로 개시한 내비게이션
    특허가 이미 업로드되어 있었다. 그대로 두면 읽는 사람은 그것을 추가 검색 대상으로 옮겨 적고,
    선행기술 검색도 이미 찾은 것을 웹에서 다시 찾는다.
    """
    parent, child = claim(1, importances=(5,)), claim(2, depends_on=1, importances=(4, 4))
    parents = {1: build(parent, [cell("1", "A", "동일")], all_claims=[parent, child])}
    child_matches = [cell("1", "A", "대응 없음", number=2), cell("1", "B", "대응 없음", number=2),
                     cell("2", "A", "동일", number=2), _partial("2", "B", found=False, number=2),
                     cell("3", "A", "대응 없음", number=2), _partial("3", "B", found=True, number=2)]

    chain = build(child, child_matches, parents=parents, all_claims=[parent, child])

    assert chain.added == ["2"]                   # 공백 A를 메우는 문헌이 유일한 추가 자리를 쓴다
    assert chain.combination_limit == 1
    assert chain.uncovered == [] and chain.residual == ["B"]
    # 구성이 아니라 한정 단위로 남는다. 그 한정을 개시한 것은 채택되지 못한 문헌 3이다.
    assert chain.beyond_limit_residual == {"B": {_GUIDANCE: ["3"]}}
    assert chain.beyond_limit == []               # 구성 B 자체는 대응이 있으므로 미대응이 아니다


def test_a_remaining_limitation_absent_everywhere_stays_a_plain_gap():
    """한도 밖 문헌에도 없는 한정은 종전대로 그냥 남은 차이다. 문장을 늘리지 않는다."""
    parent, child = claim(1, importances=(5,)), claim(2, depends_on=1, importances=(4, 4))
    parents = {1: build(parent, [cell("1", "A", "동일")], all_claims=[parent, child])}
    child_matches = [cell("1", "A", "대응 없음", number=2), cell("1", "B", "대응 없음", number=2),
                     cell("2", "A", "동일", number=2), _partial("2", "B", found=False, number=2),
                     cell("3", "A", "대응 없음", number=2), _partial("3", "B", found=False, number=2)]

    chain = build(child, child_matches, parents=parents, all_claims=[parent, child])

    assert chain.residual == ["B"] and chain.beyond_limit_residual == {}


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


def test_independent_claim_combines_at_most_two_documents():
    """사례 5. 독립항 거절 이유에 세우는 인용발명은 최대 2건(주 1 + 보조 1)이다.

    3건을 결합한 거절 이유는 실무에서 성립하기 어렵다. 다만 상한 때문에 빠진 세 번째 문헌이
    어떤 구성의 유일한 근거를 가지고 있다면 그 사실은 반드시 남는다 — 감추면 "어느 인용발명에도
    대응이 없다"는 거짓 진술이 되고, 이미 손에 든 문헌을 다시 찾게 된다.
    """
    target = claim(importances=(5, 5, 4, 4))
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "대응 없음"), cell("1", "D", "대응 없음")]
        + [cell("2", label, "대응 없음") for label in "ABD"] + [cell("2", "C", "동일")]
        + [cell("3", label, "대응 없음") for label in "ABC"] + [cell("3", "D", "동일")]
    )
    chain = build(target, matches, all_claims=[target])
    assert chain.primary == "1" and chain.secondaries == ["2"]
    assert chain.combination_limit == 2
    assert coverage_of(chain, "C").adopted_document == "2"
    assert chain.beyond_limit == ["D"] and chain.beyond_limit_documents["D"] == ["3"]
    assert "상한(2건)" in chain.rationale
    # 모든 문헌의 구성별 대응은 채택 여부와 무관하게 전부 분석된다.
    assert {candidate.document_id for candidate in coverage_of(chain, "D").candidates} == {"1", "2", "3"}


def test_the_secondary_that_fills_the_most_gaps_wins_the_single_slot():
    """보조 인용발명 자리가 하나뿐이면 공백을 가장 많이 메우는 문헌이 차지한다."""
    target = claim(importances=(5, 3, 3, 4))                    # D가 핵심, 마지막에 남도록 구성
    matches = (
        [cell("1", "A", "동일")] + [cell("1", label, "대응 없음") for label in "BCD"]
        + [cell("2", "A", "대응 없음"), cell("2", "B", "동일"), cell("2", "C", "동일"),
           cell("2", "D", "대응 없음")]
        + [cell("3", label, "대응 없음") for label in "ABC"] + [cell("3", "D", "동일")]
    )
    chain = build(target, matches, all_claims=[target])
    assert chain.primary == "1" and chain.secondaries == ["2"]   # B·C를 한 번에 메우는 2
    assert chain.beyond_limit == ["D"] and chain.beyond_limit_documents["D"] == ["3"]


def test_video_editing_buffer_document_outranks_a_shallow_timing_improvement():
    """실측 회귀: US2010이 (D) 공백을 메우면 (B)만 조금 보강하는 CN보다 먼저 채택한다."""
    target = claim(importances=(2, 5, 5, 4))
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "일부 차이"),
         cell("1", "C", "동일"), cell("1", "D", "대응 없음")]
        + [cell("2", label, "대응 없음") for label in "ABC"]
        + [cell("2", "D", "실질적 동일")]
        + [cell("3", "A", "대응 없음"), cell("3", "B", "동일"),
           cell("3", "C", "대응 없음"), cell("3", "D", "대응 없음")]
    )

    chain = build(target, matches, all_claims=[target])

    assert chain.primary == "1"
    assert chain.secondaries == ["2"]
    assert coverage_of(chain, "D").adopted_document == "2"
    assert chain.uncovered == []


# --- 주지관용기술 -------------------------------------------------------------

def test_a_generic_gap_backed_by_several_documents_becomes_well_known_art():
    """범용 구성 하나 때문에 설 수 있는 거절이 통째로 사라지면 안 된다.

    중요도가 낮고(범용 부품·통상 인터페이스) 업로드된 문헌 **여러 건**이 그 구성을 실제로
    언급하면, 그것은 인용발명이 아니라 주지관용기술로 다룰 수 있다. 실증 문헌을 함께 남겨
    심사관이 다툴 수 있게 한다.
    """
    target = claim(importances=(5, 5, 1))
    matches = ([cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "차이")]
               + [cell("2", "A", "일부 유사"), cell("2", "B", "일부 유사"), cell("2", "C", "차이")])

    chain = build(target, matches, all_claims=[target])

    assert chain.track == "inventive_step_combination"      # 거절 이유가 선다
    assert chain.well_known == ["C"]
    assert chain.well_known_documents["C"] == ["1", "2"]
    assert "주지관용기술" in chain.rationale


def test_a_generic_gap_that_no_document_mentions_is_not_well_known_art():
    """중요도만으로는 주지관용이 되지 않는다. 이 도구가 관용성을 말할 근거가 없다."""
    target = claim(importances=(5, 5, 1))
    matches = ([cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "대응 없음")]
               + [cell("2", "A", "일부 유사"), cell("2", "B", "일부 유사"), cell("2", "C", "대응 없음")])

    chain = build(target, matches, all_claims=[target])

    assert chain.well_known == []
    assert chain.track == "rejection_impossible"
    assert chain.uncovered == ["C"]


def test_a_core_gap_is_never_treated_as_well_known_art():
    """차별적 핵심 구성은 여러 문헌이 언급하더라도 주지관용으로 넘기지 않는다."""
    target = claim(importances=(5, 5, 5))
    matches = ([cell("1", "A", "동일"), cell("1", "B", "동일"), cell("1", "C", "차이")]
               + [cell("2", "A", "일부 유사"), cell("2", "B", "일부 유사"), cell("2", "C", "차이")])

    chain = build(target, matches, all_claims=[target])

    assert chain.well_known == []
    assert chain.track == "rejection_impossible"


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


def test_no_primary_still_reports_the_correspondences_that_were_found():
    """주 인용발명이 서지 않는다고 구성대비 결과를 '대응 없음'으로 덮어쓰지 않는다.

    자격 게이트는 차별적 핵심 구성(중요도 4 이상)의 직접 개시량만 본다. 핵심 구성이
    '일부 유사·inferred'에 머물면 그 값이 전 문헌에서 0이 되어 조합을 세울 수 없는데,
    종전에는 그때 uncovered에 전 구성을 넣고 빈 조합으로 마감했다. 그러면 같은 문헌이
    원문 대조까지 통과해 '실질적 동일·direct'로 개시한 구성까지 "어느 인용발명에서도
    확인되지 않았다"로 보고되고, 보고서 본문에서 구성대비가 통째로 사라졌다.
    """
    target = claim(importances=(2, 5))
    chain = build(target, [cell("1", "A", "실질적 동일"),
                           cell("1", "B", "일부 유사", direct=False)], all_claims=[target])
    assert chain.track == "rejection_impossible"      # 결론은 그대로 접는다
    assert chain.primary is None and chain.secondaries == []
    assert chain.uncovered == []                      # 확인된 대응을 공백으로 적지 않는다
    assert chain.residual == ["B"]
    assert chain.reference_only == ["1"]              # 채택은 아니고 보고 대상일 뿐
    assert coverage_of(chain, "A").adopted_document == "1"
    assert coverage_of(chain, "A").adopted_role == "미채택"


def test_dependent_claim_reports_its_comparison_when_the_parent_built_no_combination():
    """부모항이 조합을 세우지 못해도 종속항의 추가 한정 대비는 그대로 남아야 한다.

    부모가 _no_primary_chain으로 끝나면 상속할 문헌이 없다. 종전에는 그 빈 목록으로
    종속항 결합을 계속 돌아 merged가 {}인 채 끝났고, 추가 한정을 '실질적 동일·direct'로
    개시한 문헌이 있어도 "대응되는 인용발명이 확인되지 않음"으로 보고됐다. 독립항에서
    고친 것과 같은 결손이다.
    """
    parent_claim = claim(1, importances=(2, 5))
    child_claim = claim(2, depends_on=1, importances=(5,))
    parent_chain = build(parent_claim, [cell("1", "A", "실질적 동일"),
                                        cell("1", "B", "일부 유사", direct=False)],
                         all_claims=[parent_claim, child_claim])
    assert parent_chain.primary is None and parent_chain.inherited == []

    child = build(child_claim, [cell("1", "A", "실질적 동일", number=2)],
                  parents={1: parent_chain}, all_claims=[parent_claim, child_claim])
    assert child.track == "rejection_impossible"
    assert child.primary is None and child.secondaries == []
    assert child.uncovered == []                      # 개시된 한정을 공백으로 적지 않는다
    assert child.reference_only == ["1"]
    assert coverage_of(child, "A").adopted_document == "1"
    # 이 항만으로 주 인용발명을 세우지 않았다는 사유가 결론에 남아야 한다.
    assert "부모 청구항 1의 인용발명 조합이 서지 않아" in child.rationale


def test_no_primary_still_names_the_labels_that_are_genuinely_missing():
    """대응이 확인된 구성과 어디에도 없는 구성은 결론에서 구별되어야 한다."""
    target = claim(importances=(2, 5, 3))
    chain = build(target, [cell("1", "A", "실질적 동일"),
                           cell("1", "B", "일부 유사", direct=False),
                           cell("1", "C", "대응 없음")], all_claims=[target])
    assert chain.uncovered == ["C"]
    assert "구성 A, B에는 대응 기재가 확인되었으나" in chain.rationale
    assert "구성 C은 어느 인용발명에서도 대응 기재가 확인되지 않았습니다" in chain.rationale


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
    assert child.added == ["2"]


# --- 보고서 정량 지표 ---------------------------------------------------------

def checked(*chunks: str, judgment: str = "실질적 동일", disclosed: bool = True) -> ElementMatch:
    """하위 한정마다 근거 청크를 지정한 셀. 같은 청크를 되풀이하면 근거가 한 자리에 몰린 것이다."""
    return ElementMatch(
        claim_number=1, label="A", document_id="1", judgment=judgment, directness="direct",
        quote="원문 발췌 문장입니다", chunk_id=chunks[0], verify="verified",
        limitation_checks=[
            LimitationCheck(index=index, limitation=f"한정 {index}", disclosed=disclosed,
                            quote=f"한정 {index}의 근거 문장입니다", chunk_id=chunk, verify="verified")
            for index, chunk in enumerate(chunks)])


def test_limitation_counts_expose_the_numerator_and_denominator():
    """백분율 대신 셀 수 있는 값을 낸다. 독자가 근거 목록과 대조해 검증할 수 있어야 한다."""
    assert limitation_counts(checked("D1-P-0001", "D1-P-0002")) == (2, 2)
    assert limitation_counts(checked("D1-P-0001", "D1-P-0002", disclosed=False)) == (0, 2)
    assert limitation_counts(None) == (0, 0)


def test_alternative_group_counts_as_one_limitation():
    """"A, B 또는 C 중 적어도 하나"는 하나만 개시되면 충족이므로 분모를 키우지 않는다."""
    match = ElementMatch(
        claim_number=1, label="A", document_id="1", judgment="실질적 동일", directness="direct",
        quote="원문 발췌 문장입니다", chunk_id="D1-P-0001", verify="verified",
        limitation_checks=[
            LimitationCheck(index=0, limitation="길이를 포함함", alternative_group="속성",
                            disclosed=True, quote="근거 문장입니다", chunk_id="D1-P-0001",
                            verify="verified"),
            LimitationCheck(index=1, limitation="장르를 포함함", alternative_group="속성"),
        ])
    assert limitation_counts(match) == (1, 1)


def test_evidence_locations_count_distinct_passages():
    """한 문단을 모든 한정의 근거로 되풀이 인용한 대응은 그 사실이 드러나야 한다."""
    assert evidence_locations(checked("D1-P-0001", "D1-P-0002")) == 2
    assert evidence_locations(checked("D1-P-0001", "D1-P-0001")) == 1


def test_grade_carries_no_percentage_band():
    """등급은 이름과 기호만 남긴다. 백분율은 등급을 숫자로 다시 쓴 것에 가까웠다."""
    assert report_grade(checked("D1-P-0001")) == ("실질적 동일", "🟢")
    assert report_grade(checked("D1-P-0001", judgment="차이")) == ("대응 안됨", "⚪")
    assert report_grade(None) == ("대응 안됨", "⚪")


# --- 결합에서의 지시 관계 상한 복원 (fail-closed) --------------------------------

def _anaphora_claim() -> Claim:
    """(B)가 (A)의 대상을 "상기 …"로 참조하는 청구항. consistency.antecedents가 링크를 만든다."""
    return Claim(number=1, elements=[
        ClaimElement(label="A", text="제1 반사부재", importance=5),
        ClaimElement(label="B", text="상기 제1 반사부재의 전방에 배치되는 광원", importance=5),
    ])


def _capped(document_id: str, judgment: str = "일부 유사") -> ElementMatch:
    """(B) 셀. 같은 문헌에 (A)가 없어 consistency가 상한을 씌운 상태를 재현한다."""
    match = cell(document_id, "B", judgment)
    match.antecedent_capped_from = "실질적 동일"
    match.downgraded_from = "실질적 동일"
    match.antecedent_note = "같은 인용발명에서 구성 A의 대응이 확인되지 않아, 완전 개시로 보지 않았습니다"
    return match


def test_a_fully_disclosed_antecedent_in_another_document_lifts_the_cap():
    """결합 문헌이 선행 구성을 완전 개시했으면 단독문헌 상한은 풀려야 한다."""
    matches = [cell("1", "A", "대응 없음"), _capped("1"),
               cell("2", "A", "실질적 동일"), cell("2", "B", "대응 없음")]
    chain = build(_anaphora_claim(), matches)

    adopted = coverage_of(chain, "B")
    assert adopted.adopted_judgment == "실질적 동일"
    assert matrix_for(matches)["1"]["B"].judgment == "일부 유사"   # 원본 셀은 감사용으로 보존된다


def test_a_partially_disclosed_antecedent_does_not_lift_the_cap():
    """fail-closed: 선행 구성이 부분 개시면 결합해도 지시 대상이 세워지지 않는다.

    종전에는 has_correspondence만 요구해 '일부 차이'로도 상한이 풀렸다. 그러면 (A)를 부분적으로만
    개시한 문헌을 끌어와 (B)를 완전 개시로 세우게 되고, 이는 교차문헌 결합 명제를 검증하지 않은
    채 결합 커버리지를 과대평가하는 것이다.
    """
    matches = [cell("1", "A", "대응 없음"), _capped("1"),
               cell("2", "A", "일부 차이", missing=["반사면의 곡률 한정"]), cell("2", "B", "대응 없음")]
    chain = build(_anaphora_claim(), matches)

    adopted = coverage_of(chain, "B")
    assert adopted.adopted_judgment == "일부 유사"                 # 상한 유지
    # 상한을 유지했다는 사실이 보고서에도 남아야 한다. 조용히 낮추면 "한정은 개시인데 등급만
    # 낮은" 결과가 이유 없이 보인다.
    assert matrix_for(matches)["1"]["B"].antecedent_note


def test_the_restored_grade_never_exceeds_the_antecedent_it_references():
    """복원 상한은 선행 구성 중 가장 약한 판정을 넘지 못한다.

    (A)가 '실질적 동일'(rank 4)인데 (B)를 '동일'(rank 5)로 복원하면, 참조하는 대상보다 더
    완전하게 개시되었다고 적는 셈이 된다. enforce_antecedents가 같은 문헌 안에서 쓰는
    limit = min(선행 구성 판정) 규칙을 결합 범위에도 그대로 적용한다.
    """
    capped = _capped("1")
    capped.antecedent_capped_from = "동일"
    matches = [cell("1", "A", "대응 없음"), capped,
               cell("2", "A", "실질적 동일"), cell("2", "B", "대응 없음")]
    chain = build(_anaphora_claim(), matches)

    assert coverage_of(chain, "B").adopted_judgment == "실질적 동일"


# --- 주 인용발명 자격 게이트 ------------------------------------------------------
# 이 게이트는 오래 "있는 척"만 했습니다. 임계가 절대 차(0.20)라 core_direct 최고점이 0.5
# 안팎인 실측 분포에서는 최고점의 60%짜리 문헌까지 통과했고, 그마저도 "핵심 구성을 하나라도
# 직접 개시하면 되살린다"는 예외가 임계와 무관하게 되돌려 놓았습니다. 실측 사건에서는
# 4문헌 중 3문헌이 후보로 남았고 결국 main_score 1위가 뽑혔습니다.

def _shared_core_matrix() -> list[ElementMatch]:
    """실측 사건의 구조: 핵심 구성 C는 세 문헌이 함께 개시하고, D만 문헌 1이 앞선다.

    (C)를 공유한다는 사실은 주 인용발명 자격의 근거가 되지 못합니다 — 후보 전부가 가진
    것이기 때문입니다. 자격을 가르는 것은 아무도 갖지 못한 (D)입니다.
    """
    return (
        [cell("1", "C", "실질적 동일"), cell("1", "D", "일부 차이", direct=False)]
        + [cell("2", "C", "실질적 동일"), cell("2", "D", "차이", direct=False)]
        + [cell("3", "C", "실질적 동일"), cell("3", "D", "대응 없음")]
    )


def _core_claim() -> Claim:
    target = Claim(number=1, elements=[
        ClaimElement(label="C", text="구성 C", importance=4),
        ClaimElement(label="D", text="구성 D", importance=5)])
    return target


def test_a_core_element_every_candidate_discloses_does_not_earn_primary_standing():
    """모두가 가진 핵심 구성을 가졌다는 사실로는 주 인용발명 후보가 되지 못한다."""
    chain = build(_core_claim(), _shared_core_matrix())
    assert chain.primary == "1"
    # 문헌 2·3도 (C)를 '실질적 동일·direct·검증됨'으로 개시하지만 고유 기여가 없다.
    assert chain.candidates[0].document_id == "1"
    roles = {score.document_id: score.detail["role"] for score in chain.candidates}
    assert roles["2"] != "주 인용발명" and roles["3"] != "주 인용발명"


def test_a_document_holding_the_only_route_to_a_core_element_survives_the_gate():
    """임계에 못 미쳐도 다른 후보가 못 가진 핵심 구성을 직접 개시하면 후보로 남긴다."""
    target = Claim(number=1, elements=[
        ClaimElement(label="A", text="구성 A", importance=5),
        ClaimElement(label="B", text="구성 B", importance=5)])
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "대응 없음")]
        + [cell("2", "A", "대응 없음"), cell("2", "B", "동일")]   # 점수는 낮지만 (B)의 유일 경로
    )
    chain = build(target, matches)
    assert chain.primary == "1"
    # 문헌 2는 자격을 잃지 않았으므로 보조 인용발명으로 결합될 수 있다.
    assert chain.secondaries == ["2"]


def test_an_alternative_group_always_occupies_exactly_one_slot():
    """"A, B 또는 C 중 적어도 하나"는 요구사항 하나다. 몇 개를 개시했든 셈이 흔들리면 안 된다.

    종전에는 충족된 묶음에서 미개시 대안만 뺐다. 그래서 모델이 대안을 몇 개나 개시로
    표시했는지에 따라 보고서에 (1,1)·(2,2)·(3,3)이 제각각 찍혔고, 같은 청구항을 충족한
    두 문헌이 서로 다른 개시 수를 달고 나갔다.
    """
    def alternatives(*flags) -> ElementMatch:
        return ElementMatch(
            claim_number=1, label="A", document_id="1", judgment="실질적 동일",
            directness="direct", quote="원문 발췌 문장입니다", chunk_id="D1-P-0001",
            verify="verified",
            limitation_checks=[
                LimitationCheck(index=index, limitation=f"대안 {index}", alternative_group="속성",
                                disclosed=disclosed,
                                quote="근거 문장입니다" if disclosed else "",
                                chunk_id="D1-P-0001" if disclosed else "",
                                verify="verified" if disclosed else "empty")
                for index, disclosed in enumerate(flags)])

    assert limitation_counts(alternatives(True, False, False)) == (1, 1)
    assert limitation_counts(alternatives(True, True, False)) == (1, 1)
    assert limitation_counts(alternatives(True, True, True)) == (1, 1)
    # 아무 대안도 개시되지 않으면 묶음 하나가 통째로 미개시다.
    assert limitation_counts(alternatives(False, False, False)) == (0, 1)


# --- 축 결손 기각과 결합 -------------------------------------------------------
# 의미검증(entailment)은 문헌 하나만 놓고 한정을 본다. "동작은 있는데 그 동작의 대상이 이
# 문헌에 없다"는 축 결손이 그래서 나오고, 빠진 축을 다른 인용발명이 대는 것이 진보성 결합의
# 정의다. 아래 세 테스트는 그 경로가 살아 있는지를 지킨다.

def axis_rejected(document_id: str, label: str, disclosed: list[str],
                  rejected: list[str], *, direct: bool = False) -> ElementMatch:
    """1차 판정은 개시였는데 의미검증이 축 결손으로 뺀 셀.

    entailment는 기각할 때 check.disclosed를 False로 덮어쓰고 판단만 semantic_status에
    남긴다. 그 상태를 그대로 만든다.
    """
    match = cell(document_id, label, "차이", direct=direct, missing=list(rejected))
    match.limitation_checks = [
        LimitationCheck(index=index, limitation=text, kind="qualifier", disclosed=True,
                        quote=match.quote, chunk_id=match.chunk_id, verify="verified",
                        semantic_status="accepted")
        for index, text in enumerate(disclosed)
    ] + [
        LimitationCheck(index=len(disclosed) + index, limitation=text, kind="core",
                        disclosed=False, quote=match.quote, chunk_id=match.chunk_id,
                        verify="verified", semantic_status="rejected",
                        semantic_note="대상 축 결손")
        for index, text in enumerate(rejected)
    ]
    return match


def blank(document_id: str, label: str, *, core: list[str], qualifier: list[str]) -> ElementMatch:
    """같은 분해 결과로 판정됐지만 아무것도 개시하지 못한 셀. 한정 문언과 kind가 맞물려야
    다른 문헌의 셀과 실제로 비교된다."""
    match = cell(document_id, label, "차이", missing=[*core, *qualifier])
    match.limitation_checks = [
        LimitationCheck(index=index, limitation=text, kind=kind, disclosed=False, verify="empty")
        for index, (text, kind) in enumerate(
            [(text, "core") for text in core] + [(text, "qualifier") for text in qualifier])
    ]
    return match


def test_a_document_rejected_only_on_a_missing_axis_is_still_adopted():
    """실측(neareye-waveguide). 이 문헌이 빠지면 청구항 전체가 "거절 이유 구성 곤란"이 된다.

    구성 (B)의 뉴럴 네트워크 학습을 인용발명 3이 원문으로 개시했고 1차 판정은 '실질적 동일
    2/2'였다. 의미검증이 "학습 대상이 도파관 광학 시스템임은 이 문헌에 없다"고 그 한정을
    뺐고, 도파관은 같은 조합의 다른 문헌이 개시하고 있었다. 종전 게이트는 공백 구성에
    has_correspondence(합친 결과)를 요구했는데, 공백이라는 말은 양쪽 라벨이 이미 낮다는
    뜻이라 무엇을 합쳐도 참이 될 수 없었다 — 공백을 메울 후보에게 공백이 이미 메워져
    있기를 요구하는 순환이다.
    """
    # 두 셀은 **같은 분해 결과**로 판정되므로 한정 문언이 서로 같다. 주 인용발명은 그 둘을
    # 모두 놓쳤고, 보완 후보는 하나를 개시하고 하나는 축 결손으로 기각됐다.
    matches = ([disclosing("1", "A", "실질적 동일", ["구성 A를 개시함"]),
                disclosing("1", "B", "차이", [],
                           missing=["뉴럴 네트워크를 학습함", "도파관 광학계를 모델링함"]),
                cell("1", "C", "동일")]
               + [cell("2", "A", "대응 없음"),
                  axis_rejected("2", "B", ["뉴럴 네트워크를 학습함"], ["도파관 광학계를 모델링함"]),
                  cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)
    assert chain.primary == "1"
    assert chain.secondaries == ["2"]


def test_an_axis_rejected_element_is_reserved_not_declared_absent():
    """원문 근거가 있는 구성을 "어느 인용발명에서도 확인되지 않았다"고 적을 수는 없다.

    도구가 확인하지 못한 것과 문헌에 없는 것은 다른 사실이고, 읽는 사람이 취할 다음 행동도
    다르다. 앞은 결합 위에서 다시 묻는 일이고 뒤는 추가 검색이다.

    실측과 같은 모양으로 세운다 — 기각된 한정이 core라서, 결합이 qualifier를 채워도 그
    구성은 여전히 대응이 서지 않는다. 그 상태에서도 공백이 아니라 유보여야 한다.
    """
    matches = ([disclosing("1", "A", "실질적 동일", ["구성 A를 개시함"]),
                blank("1", "B", core=["도파관 광학계를 모델링함"],
                      qualifier=["뉴럴 네트워크를 학습함"]),
                cell("1", "C", "동일")]
               + [cell("2", "A", "대응 없음"),
                  axis_rejected("2", "B", ["뉴럴 네트워크를 학습함"], ["도파관 광학계를 모델링함"]),
                  cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)
    assert "B" in chain.uncovered                      # 대응은 여전히 서지 않는다
    assert chain.combination_pending == ["B"]          # 그러나 공백으로 단정하지 않는다
    assert "대상 축 결손" in " ".join(chain.combination_pending_reasons["B"])
    # 유보는 결론을 막지 않는다. 막으면 유보가 곧 미개시와 같은 값이 된다.
    assert chain.track == "inventive_step_combination"
    assert "유보" in chain.rationale


def test_the_combination_grade_rises_with_the_facts_but_stops_below_identity():
    """결합으로 한정을 채우면 등급도 따라 움직이되 동일급에는 닿지 않는다.

    등급을 옛 값으로 못 박아 두면 같은 셀 안에서 한정과 등급이 서로 모순하고, 그 모순된
    등급이 has_correspondence를 거쳐 "이 구성은 어느 인용발명에도 없다"는 사실 진술로
    나간다. 반대로 끝까지 다시 유도하면 '실질적 동일'이 되는데, 그것은 단일 문헌이 그 구성을
    개시한다는 진술이라 결합 결과에 붙일 수 있는 말이 아니다.
    """
    gap = "도파관 광학계를 통해 출력함"
    matches = ([disclosing("1", "A", "차이", [], missing=[gap]),
                cell("1", "B", "동일"), cell("1", "C", "동일")]
               + [disclosing("2", "A", "차이", [gap], direct=False),
                  cell("2", "B", "대응 없음"), cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)
    merged = merge_selected(claim(), matrix_for(matches), ["1", "2"])
    assert merged["A"].combination_resolved.get(gap) == "2"
    assert merged["A"].judgment == "일부 차이"
    assert "A" not in chain.uncovered


def test_a_wobbling_importance_score_cannot_empty_the_primary_gate():
    """중요도가 한 칸 달라졌다고 인용발명이 전부 미채택이 되어서는 안 된다.

    중요도는 분해와 함께 LLM이 매 실행 새로 매기는 값이다. 실측에서 같은 청구항의 구성 (A)가
    한 실행에서 4, 다음 실행에서 3을 받았다. 3을 받은 실행에서는 핵심 구성이 (B),(C)만 남았고
    둘 다 어느 문헌에서도 직접 개시되지 않아 주 인용발명 자격 게이트가 통째로 비었다 —
    (A)를 '실질적 동일·direct·검증됨·누락 0'으로 개시한 문헌을 눈앞에 두고 전 문헌이
    '미채택'으로 나갔다. 자격 게이트가 답할 질문은 후보 사이의 우열이지 "거절 이유를 세울 수
    있는가"가 아니다.
    """
    target = claim(importances=(3, 4, 5))          # (A)만 잘 개시되는데 (A)가 핵심에서 빠졌다
    matches = ([cell("1", "A", "실질적 동일"), cell("1", "B", "차이"), cell("1", "C", "차이")]
               + [cell("2", "A", "차이"), cell("2", "B", "차이"), cell("2", "C", "차이")])
    chain = build(target, matches, all_claims=[target])
    assert chain.primary == "1"
    assert chain.track != "rejection_impossible" or chain.primary


def test_the_primary_gate_still_refuses_when_nothing_is_disclosed_anywhere():
    """되돌아갈 곳까지 비면 세우지 않는다. 무관한 문헌에 '주 인용발명'을 찍지 않기 위해서다."""
    target = claim(importances=(3, 4, 5))
    matches = [cell(document_id, label, "대응 없음")
               for document_id in ("1", "2") for label in "ABC"]
    chain = build(target, matches, all_claims=[target])
    assert chain.primary is None
    assert chain.track == "rejection_impossible"


def test_a_limitation_accepted_in_combination_stops_being_a_gap():
    """결합 심사가 인정한 한정은 조합 결과에서 메워진 것으로 센다.

    인정은 채택 조합 전체의 근거 위에서 내린 판단이므로(entailment.validate_combination),
    문헌 단독 셀의 disclosed는 그대로 두고 결합 결과에서만 해소로 처리한다. 그러지 않으면
    그 문헌 혼자 그 한정을 개시한 것처럼 감사 데이터에 남는다.
    """
    axis = "도파관 광학계를 모델링함"
    supplement = axis_rejected("2", "B", ["뉴럴 네트워크를 학습함"], [axis])
    # 결합 심사가 문헌 1을 빠진 축의 출처로 지목한 상태.
    rejected_check = next(item for item in supplement.limitation_checks if item.limitation == axis)
    rejected_check.semantic_status = "accepted_in_combination"
    rejected_check.combination_documents = ["1"]

    matches = ([disclosing("1", "A", "실질적 동일", ["구성 A를 개시함"]),
                blank("1", "B", core=[axis], qualifier=["뉴럴 네트워크를 학습함"]),
                cell("1", "C", "동일")]
               + [cell("2", "A", "대응 없음"), supplement, cell("2", "C", "대응 없음")])
    chain = build(claim(), matches)

    assert "B" not in chain.uncovered              # 조합 안에 그 한정의 근거가 있다
    assert chain.combination_pending == []         # 확인이 끝났으므로 유보도 아니다
    merged = merge_selected(claim(), matrix_for(matches), ["1", "2"])
    assert merged["B"].combination_resolved.get(axis) == "1"
    # 문헌 단독 판정은 손대지 않는다.
    assert rejected_check.disclosed is False


# --- 미채택 사유 기록 ----------------------------------------------------------
# 불변식 P2가 물어야 하는 것은 "이득이 있었는가"가 아니라 "왜 빠졌는지 적혀 있는가"다.
# 이득(주 인용발명 대비)과 limit_binding(채택 조합 대비)은 기준선이 달라서, 둘을 맞대면
# 채택 조합이 이미 같은 것을 대고 있는 중복 후보마다 위반이 찍힌다.

def test_a_duplicate_candidate_is_recorded_as_a_normal_exclusion():
    """같은 기여를 내는 두 문헌 중 하나가 채택되면 다른 하나는 '중복'으로 빠진 것이다.

    실측(facade-defect-3d)에서 두 문헌이 같은 구성에 정확히 같은 이득 0.4417을 냈고, 하나가
    채택되자 다른 하나가 매 회차 P2 위반으로 보고됐다. 동률은 흔하므로 그 상태로는 경고가
    늘 켜져 진짜 위반이 묻힌다.
    """
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "대응 없음"), cell("1", "C", "동일")]
        + [cell("2", "A", "대응 없음"), cell("2", "B", "실질적 동일"), cell("2", "C", "대응 없음")]
        + [cell("3", "A", "대응 없음"), cell("3", "B", "실질적 동일"), cell("3", "C", "대응 없음")]
    )
    chain = build(claim(), matches)
    rows = {row.document_id: row for row in coverage_of(chain, "B").candidates}

    adopted = [document_id for document_id, row in rows.items() if row.adopted]
    assert len(adopted) == 1                        # 둘 중 하나만 채택된다
    loser = "3" if adopted == ["2"] else "2"
    assert rows[loser].gain > 0                     # 주 인용발명 대비로는 여전히 이득이 있다
    assert rows[loser].merged_gain == 0.0           # 조합 대비 증분은 0이다
    assert "증분 0" in rows[loser].excluded_reason


def test_every_unadopted_candidate_carries_a_reason():
    """사유 없이 사라진 후보가 있으면 게이트 하나가 조용히 경로를 막고 있다는 뜻이다."""
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "대응 없음"), cell("1", "C", "일부 유사")]
        + [cell("2", "A", "일부 유사"), cell("2", "B", "실질적 동일"), cell("2", "C", "대응 없음")]
        + [cell("3", "A", "대응 없음"), cell("3", "B", "일부 차이"), cell("3", "C", "실질적 동일")]
        + [cell("4", "A", "대응 없음"), cell("4", "B", "대응 없음"), cell("4", "C", "대응 없음")]
    )
    chain = build(claim(), matches)

    for coverage in chain.element_coverage:
        for row in coverage.candidates:
            if row.adopted:
                assert row.excluded_reason == ""
            else:
                assert row.excluded_reason, f"{coverage.label}/문헌 {row.document_id}에 사유가 없다"


def test_the_combination_limit_is_recorded_as_the_reason_when_it_binds():
    """상한이 실제로 걸려 빠진 문헌은 그 사실이 후보 행에 적혀야 한다."""
    matches = (
        [cell("1", "A", "동일"), cell("1", "B", "대응 없음"), cell("1", "C", "대응 없음")]
        + [cell("2", "A", "대응 없음"), cell("2", "B", "실질적 동일"), cell("2", "C", "대응 없음")]
        + [cell("3", "A", "대응 없음"), cell("3", "B", "대응 없음"), cell("3", "C", "실질적 동일")]
    )
    chain = build(claim(), matches)
    assert chain.limit_binding                       # 자리가 하나 더 있었다면 채택됐을 후보가 남아 있다

    dropped = [row for coverage in chain.element_coverage for row in coverage.candidates
               if not row.adopted and row.merged_gain > 0]
    assert dropped and all("상한" in row.excluded_reason for row in dropped)
