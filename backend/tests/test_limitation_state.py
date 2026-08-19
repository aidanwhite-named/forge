"""개시·누락·**미완료** 3분류와 그 네 소비자. 구현보다 먼저 고정한 단위 불변식이다.

의미검증(entailment)이 응답에서 항목을 빠뜨리면 그 한정은 검증을 **받지 못한** 상태가 된다.
실측(GNSS/SfM 사건)에서 7건이 그렇게 빠졌고, 하필 그것들이 🟢 실질적 동일을 받은 구성 D·E의
한정이었다. 두 가지를 동시에 지켜야 한다.

  - 미완료를 **개시로 세지 않는다.** 세면 검증을 건너뛴 한정이 가장 튼튼한 근거로 보인다
    (보고서에서 의미검증을 통과한 근거에만 단서가 붙으므로, 미검증 근거는 무표시 = 원문
    그대로로 읽힌다).
  - 미완료를 **누락으로도 세지 않는다.** 세면 "검증기가 응답을 빠뜨렸다"가 "이 문헌에 그
    한정이 없다"라는 문헌에 대한 사실 주장으로 바뀐다. 방향이 반대인 오류다.

그래서 disclosed를 뒤집지 않고 제3 상태를 중앙에서 한 번 가른 뒤, 등급·신규성·문헌 선정·
근거 출력 네 곳이 **같은 함수**를 본다. 각자 자기 규칙을 들면 같은 한정이 자리마다 다르게
세어진다 — 이 파이프라인에서 되풀이된 실패가 정확히 그 형태다.
"""
from app import coverage
from app.chain import build_chain, matrix_for, screen_novelty
from app.coverage import (DISCLOSED, MISSING, UNVERIFIED, best_match, derive_judgment,
                          disclosed_count, is_complete, limitation_states, quality_key,
                          residual_difference, score_document, supplement_reason,
                          unverified_limitations)
from app.models import (Claim, ClaimElement, ElementMatch, LimitationCheck,
                        missing_limitations)


def _check(index: int, limitation: str, *, kind: str = "core", disclosed: bool = True,
           status: str = "accepted", group: str = "") -> LimitationCheck:
    return LimitationCheck(index=index, limitation=limitation, kind=kind, disclosed=disclosed,
                           alternative_group=group, semantic_status=status,
                           quote="원문 발췌 문장입니다", chunk_id="D2-B-p009-06", verify="verified")


def _match(checks: list[LimitationCheck], *, judgment: str = "실질적 동일",
           document_id: str = "2", label: str = "E") -> ElementMatch:
    return ElementMatch(claim_number=1, label=label, document_id=document_id, judgment=judgment,
                        directness="direct", quote="원문 발췌 문장입니다",
                        chunk_id="D2-B-p009-06", verify="verified", limitation_checks=checks,
                        missing_limitations=missing_limitations(checks))


# 실측 사건의 구성 E를 그대로 옮긴 형태다. core 2개 + qualifier 1개이고, 의미검증 응답에서
# 세 건 모두 빠져 semantic_status가 error로 남았다.
def _element_e() -> list[LimitationCheck]:
    return [_check(0, "전역 좌표 정합을 최적화함", status="error"),
            _check(1, "절대좌표계에 대응하는 최종 3D 모델을 생성함", status="error"),
            _check(2, "최적화 방식을 위치 및 기하학적 제약조건 적용으로 한정함",
                   kind="qualifier", status="error")]


# --- 중앙 분류 ----------------------------------------------------------------

def test_an_unchecked_limitation_is_neither_disclosed_nor_missing():
    states = [state for _, state in limitation_states(_element_e())]
    assert states == [UNVERIFIED, UNVERIFIED, UNVERIFIED]


def test_the_three_states_are_distinguished_in_one_pass():
    checks = [_check(0, "개시된 한정"),
              _check(1, "검증을 받지 못한 한정", status="error"),
              _check(2, "빠진 한정", disclosed=False, status="rejected")]
    assert [state for _, state in limitation_states(checks)] == [DISCLOSED, UNVERIFIED, MISSING]


def test_verification_never_flips_the_disclosure_flag():
    """미완료는 문헌에 대한 사실 주장이 아니다. disclosed도 누락 목록도 건드리지 않는다."""
    checks = _element_e()
    assert all(check.disclosed for check in checks)
    assert missing_limitations(checks) == []
    # 개시 수도 줄이지 않는다. 줄이면 보고서에 "2/3 개시"로 찍혀 한정 하나가 이 문헌에
    # 없다는 뜻이 된다 — 실제로는 있는지 없는지 확인하지 못한 것이다.
    assert disclosed_count(checks) == (3, 3)


def test_a_limitation_that_never_entered_verification_is_not_counted_as_unchecked():
    """semantic_status='not_run'은 미완료가 아니다.

    원문 대조를 통과한 발췌 묶음이 없어 의미검증 대상에 아예 오르지 않은 한정이다. 그쪽은
    quote·verify 게이트(ineligible_reason·_directly_disclosed)가 이미 따로 막는다. 여기까지
    미완료로 묶으면 정상 경로의 사건 대부분이 미완료로 찍혀 상한이 무의미해진다.
    """
    checks = [_check(0, "검증 대상에 오르지 않은 한정", status="not_run")]
    assert [state for _, state in limitation_states(checks)] == [DISCLOSED]


def test_an_alternative_group_is_disclosed_when_any_verified_alternative_holds():
    """대안 묶음은 하나만 확인되면 충족이다. 미완료 대안이 묶음을 끌어내리지 않는다."""
    checks = [_check(0, "대안 A", group="g1", status="error"),
              _check(1, "대안 B", group="g1", status="accepted")]
    assert [state for _, state in limitation_states(checks)] == [DISCLOSED]


def test_an_alternative_group_is_unverified_when_only_unchecked_alternatives_hold():
    checks = [_check(0, "대안 A", group="g1", status="error"),
              _check(1, "대안 B", group="g1", disclosed=False, status="rejected")]
    assert [state for _, state in limitation_states(checks)] == [UNVERIFIED]


def test_a_limitation_supplied_by_the_combination_counts_as_disclosed():
    checks = [_check(0, "다른 인용발명이 댄 한정", disclosed=False, status="rejected")]
    assert [state for _, state in limitation_states(checks, {"다른 인용발명이 댄 한정"})] == [DISCLOSED]


def test_unverified_limitations_lists_the_texts_for_the_report():
    assert unverified_limitations(_match(_element_e())) == [
        "전역 좌표 정합을 최적화함",
        "절대좌표계에 대응하는 최종 3D 모델을 생성함",
        "최적화 방식을 위치 및 기하학적 제약조건 적용으로 한정함"]


# --- 소비자 1: 등급 -----------------------------------------------------------

def test_an_unchecked_limitation_caps_the_grade_below_the_full_judgments():
    """실측 과대판정을 잡는 자리다. 미완료가 있으면 동일급을 줄 수 없다."""
    assert derive_judgment(_element_e(), has_evidence=True) == "일부 차이"


def test_the_cap_does_not_pretend_the_core_is_absent():
    """미완료를 '일부 유사'로 내리면 core가 없다는 뜻이 된다. 상한이지 강등이 아니다.

    core 하나가 실제로 빠진 경우가 '일부 유사'이고, 그것과 "확인하지 못했다"를 같은 칸에
    넣으면 보고서의 차이점 줄이 없는 누락을 지어낸다.
    """
    checks = [_check(0, "개시된 core"), _check(1, "검증을 못 받은 core", status="error")]
    assert derive_judgment(checks, has_evidence=True) == "일부 차이"

    really_missing = [_check(0, "개시된 core"),
                      _check(1, "빠진 core", disclosed=False, status="rejected")]
    assert derive_judgment(really_missing, has_evidence=True) == "일부 유사"


def test_the_cap_never_raises_a_grade_that_was_already_lower():
    """상한은 위에서만 누른다. 이미 '일부 유사'인 셀을 '일부 차이'로 끌어올리지 않는다."""
    checks = [_check(0, "빠진 core", disclosed=False, status="rejected"),
              _check(1, "검증을 못 받은 core", status="error")]
    assert derive_judgment(checks, has_evidence=True) == "일부 유사"


def test_a_fully_verified_element_still_reaches_the_full_judgment():
    checks = [_check(0, "개시된 core"), _check(1, "개시된 qualifier", kind="qualifier")]
    assert derive_judgment(checks, has_evidence=True) == "실질적 동일"


# --- 소비자 2: 신규성 ---------------------------------------------------------

def test_the_novelty_gate_does_not_count_an_unverified_element_as_disclosed():
    """신규성 부정은 청구항을 죽이는 가장 강한 결론이다. 미검증 위에 세울 수 없다."""
    claim = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A", importance=5),
                                      ClaimElement(label="B", text="구성 B", importance=5)])
    verified = _match([_check(0, "개시된 core")], judgment="동일", document_id="1", label="A")
    unchecked = _match(_element_e(), judgment="동일", document_id="1", label="B")

    screen = screen_novelty(claim, matrix_for([verified, unchecked]), [])

    assert screen.selected_document is None
    assert screen.missing_by_document["1"] == ["B"]


def test_a_fully_verified_document_still_passes_the_novelty_gate():
    claim = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A", importance=5)])
    verified = _match([_check(0, "개시된 core")], judgment="동일", document_id="1", label="A")

    screen = screen_novelty(claim, matrix_for([verified]), [])

    assert screen.selected_document == "1"


# --- 소비자 3: 문헌 선정 -------------------------------------------------------

def test_an_unverified_element_is_not_complete_so_supplements_keep_being_searched():
    assert is_complete(_match([_check(0, "개시된 core")])) is True
    assert is_complete(_match(_element_e())) is False


def test_the_supplement_reason_names_the_verification_gap():
    reason = supplement_reason(_match(_element_e()))
    assert "의미검증 미완료" in reason and "3건" in reason


def test_a_verified_cell_outranks_an_unverified_cell_at_the_same_grade():
    """등급·근거 품질이 같으면 검증된 쪽이 결합의 근거가 되어야 한다."""
    verified = _match([_check(0, "개시된 core")], document_id="1")
    unchecked = _match([_check(0, "개시된 core", status="error")], document_id="2")

    assert quality_key(verified) > quality_key(unchecked)
    assert best_match([unchecked, verified]) is verified


def test_document_scoring_does_not_reward_the_unverified_cell():
    """미완료 한정 위에서 문헌 순위가 뒤집히면 주·보조 선정이 검증을 못 받은 셀에 선다."""
    claim = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A", importance=5)])
    verified = _match([_check(0, "개시된 core")], document_id="1", label="A")
    unchecked = _match(_element_e(), document_id="2", label="A")

    better, _ = score_document(claim, {"A": verified})
    worse, _ = score_document(claim, {"A": unchecked})

    assert better > worse


# --- 소비자 4: 근거 출력 -------------------------------------------------------

def test_the_residual_difference_says_the_verification_never_ran():
    """차이점 줄이 비면 결론과 본문이 어긋난다(report_invariants가 실측에서 잡은 형태)."""
    residual = residual_difference(_match(_element_e(), judgment="일부 차이"))
    assert any("의미검증" in line for line in residual)
    # 상한으로 눌러 둔 라벨을 대비 결과인 것처럼 되읽게 하면 안 된다.
    assert not any("일부 차이 판정에 그쳐" in line for line in residual)


def test_a_partly_verified_element_still_reports_its_grade_limit():
    """유보가 아닌 셀에서는 종전 문구가 그대로 남아야 한다."""
    residual = residual_difference(
        _match([_check(0, "확인된 core"), _check(1, "미검증 qualifier", kind="qualifier",
                                              status="error")], judgment="일부 차이"))
    assert any("일부 차이 판정에 그쳐" in line for line in residual)


def test_a_verified_element_gets_no_verification_caveat():
    residual = residual_difference(_match([_check(0, "개시된 core")], judgment="실질적 동일"))
    assert not any("의미검증" in line for line in residual)


def _rendered(match: ElementMatch) -> str:
    """이 셀 하나로 보고서를 조립해 마크다운으로 낸다."""
    from app.chain import build_chain, matrix_for
    from app.report import build_claim_report, build_mappings, to_markdown
    from app.models import AnalysisResult, Chunk, Document

    target = Claim(number=1, elements=[ClaimElement(label=match.label, text="구성 E 원문",
                                                    importance=5)])
    document = Document(id=match.document_id, filename="sensors-21-03939.pdf", type="paper",
                        chunks=[Chunk(document_id=match.document_id, chunk_id=match.chunk_id,
                                      page=9, text="원문 발췌 문장입니다")])
    matrix = matrix_for([match])
    chain = build_chain(target, matrix, {}, [target])
    mappings = build_mappings([document], [chain])
    report = build_claim_report(target, chain, matrix, {match.document_id: document}, mappings)
    return to_markdown(AnalysisResult(job_id="job", claim_mapping=mappings, reports=[report]))


def test_the_report_marks_the_metric_line_the_bullet_and_a_review_section():
    markdown = _rendered(_match(_element_e()))

    assert "↳ ⚠️ 의미검증 미완료" in markdown               # 근거 불릿
    assert "### 검토 필요 — 의미검증 미완료" in markdown     # 결론만 읽는 독자를 위한 절
    assert "절대좌표계에 대응하는 최종 3D 모델을 생성함" in markdown
    # "3/3 개시"에 "미완료 3건"을 나란히 적으면 스스로를 반박한다. 세 칸으로 갈라 적는다.
    assert "한정 3/3 개시" not in markdown
    assert "한정 3개 — 개시 확인 0 · ⚠️ 의미검증 미완료 3 · 미개시 0" in markdown
    # 확인된 개시가 하나도 없는 구성을 '부분 개시'로 집계하면 절반쯤 확인된 것처럼 읽힌다.
    assert "판정유보 1" in markdown
    # 차이점 줄이 비면 결론과 본문이 어긋난다. 등급이 왜 눌렸는지 본문에 반드시 남는다.
    assert "→ 차이점:" in markdown
    difference = next(line for line in markdown.splitlines() if line.startswith("→ 차이점:"))
    assert "의미검증을 수행하지 못해" in difference


def test_the_report_says_nothing_when_every_limitation_was_checked():
    markdown = _rendered(_match([_check(0, "개시된 core")]))

    assert "의미검증 미완료" not in markdown
    assert "검토 필요" not in markdown


def test_an_unnamed_whole_element_check_still_shows_up_for_review():
    """하위 한정으로 분해되지 않은 점검은 문언이 없다. 표시까지 사라지면 안 된다."""
    whole = LimitationCheck(index=0, limitation="구성 E 원문", kind="core", whole_element=True,
                            disclosed=True, semantic_status="error", quote="원문 발췌 문장입니다",
                            chunk_id="D2-B-p009-06", verify="verified")
    markdown = _rendered(_match([whole]))

    assert "### 검토 필요 — 의미검증 미완료" in markdown
    assert "하위 한정으로 분해되지 않은 점검" in markdown


# --- 미완료를 공백으로 바꿔 적지 않는다 ------------------------------------------
# 확정 개시만 세는 지표(atomic_coverage·disclosed_limitations)를 "확인되었는가"가 아니라
# "없다고 판정되었는가"를 묻는 자리에 그대로 쓰면, 원문 발췌가 그대로 있는 구성이 공백으로
# 떨어진다. 이 파이프라인에서 가장 무거운 오류(손에 든 문헌을 다시 찾아 나서게 하는 거짓
# 진술)이고, 불변식 P1이 막으려는 것이다.

def test_a_fully_unverified_element_is_never_reported_as_a_gap():
    from app.chain import build_chain, matrix_for
    from app.report import build_claim_report, build_mappings, to_markdown
    from app.models import AnalysisResult, Chunk, Document

    quote = ("After optimizing the above camera pose, a robust and accurate global camera "
             "pose was obtained for subsequent triangulation.")

    def cell(label, statuses):
        checks = [LimitationCheck(index=index, limitation=f"{label} 한정{index}", kind="core",
                                  disclosed=True, semantic_status=status, quote=quote,
                                  chunk_id="D2-B-p009-06", verify="verified")
                  for index, status in enumerate(statuses)]
        return ElementMatch(claim_number=1, label=label, document_id="2",
                            judgment="일부 차이" if "error" in statuses else "실질적 동일",
                            directness="direct", quote=quote, chunk_id="D2-B-p009-06",
                            verify="verified", limitation_checks=checks)

    labels = ["A", "B", "E"]
    cells = [cell(label, ["accepted", "accepted"]) for label in labels if label != "E"]
    cells.append(cell("E", ["error", "error", "error"]))
    target = Claim(number=1, elements=[ClaimElement(label=label, text=f"구성 {label}",
                                                    importance=5) for label in labels])
    document = Document(id="2", filename="sensors-21-03939.pdf", type="paper",
                        chunks=[Chunk(document_id="2", chunk_id="D2-B-p009-06", page=9,
                                      text=quote)])
    matrix = matrix_for(cells)
    chain = build_chain(target, matrix, {}, [target])
    mappings = build_mappings([document], [chain])
    report = build_claim_report(target, chain, matrix, {"2": document}, mappings)
    markdown = to_markdown(AnalysisResult(job_id="x", claim_mapping=mappings,
                                          reports=[report]))

    assert chain.uncovered == []
    assert "어느 인용발명에서도 확인되지 않았습니다" not in markdown
    assert "미대응 구성" not in markdown
    assert "⚠️ 판정유보 1" in markdown


def test_correspondence_still_fails_when_every_limitation_is_genuinely_missing():
    """미완료를 통과시킨다고 진짜 0/N까지 대응으로 세면 안 된다. 가드의 원래 목적이다."""
    from app.coverage import has_correspondence

    rejected = _match([_check(0, "core 1", disclosed=False, status="rejected"),
                       _check(1, "core 2", disclosed=False, status="rejected")],
                      judgment="일부 유사")
    assert has_correspondence(rejected) is False
    assert has_correspondence(_match(_element_e(), judgment="일부 차이")) is True


def test_the_p1_invariant_can_still_see_an_unchecked_limitation():
    """감시자가 감시할 자리에서 눈을 감으면 안 된다.

    미완료 구성이 공백으로 떨어지는 경우가 바로 P1이 잡아야 할 상황인데, 확정 개시만 보면
    그 자리에서 근거가 없는 것으로 보인다.
    """
    from app.coverage import evidenced_limitations, unchecked_limitations

    match = _match(_element_e())
    assert unchecked_limitations(match) == {check.limitation for check in _element_e()}
    assert evidenced_limitations(match) == unchecked_limitations(match)


# --- 런타임 불변식 P4 ------------------------------------------------------------

def test_a_fully_unverified_element_says_so_in_the_grade_and_the_narrative_too():
    """숫자만 고치면 같은 구성이 한 화면에서 서로 다른 말을 한다.

    독자는 등급 배지와 서술을 먼저 읽는다. 거기에 "기술 사상 동일"과 "구성과 부분적으로
    대응됩니다"가 남아 있으면, 아래 집계의 '판정유보'는 읽히지 않는다.
    """
    markdown = _rendered(_match(_element_e()))

    assert "⚠️ 판정 유보 — 의미검증 미완료" in markdown
    assert "기술 사상 동일" not in markdown
    assert "구성과 부분적으로 대응됩니다" not in markdown
    assert "판단을 받지 못했습니다" in markdown
    # 유사점 요약이 유보 구성을 공통점으로 되살리면 안 된다.
    assert "대응되는 기술 내용이 확인되지 않았습니다" in markdown
    # 미검증은 **차이가 확인된 상태가 아니다.** 차이점 줄과 결론 어디에도 차이로 적지 않는다.
    difference = next(line for line in markdown.splitlines() if line.startswith("→ 차이점:"))
    assert "세부 구현·조건에 차이가 있어" not in difference
    assert "차이가 남는 구성" not in markdown
    assert "차이점 판단이 필요합니다" not in markdown
    assert "차이가 남아 있습니다" not in markdown        # 종합 분석 요약까지
    assert "의미검증 미완료로 판정을 유보한 구성: E." in markdown


def test_the_combination_rationale_separates_reserved_from_differing():
    """결합이 선 청구항에서도 결론 문장이 유보를 차이로 바꿔 적지 않는지."""
    from app.chain import build_chain, matrix_for
    from app.report import build_claim_report, build_mappings, to_markdown
    from app.models import AnalysisResult, Chunk, Document

    quote = ("After optimizing the above camera pose, a robust and accurate global camera "
             "pose was obtained for subsequent triangulation.")

    def cell(label, statuses):
        checks = [LimitationCheck(index=index, limitation=f"{label} 한정{index}", kind="core",
                                  disclosed=True, semantic_status=status, quote=quote,
                                  chunk_id="D2-B-p009-06", verify="verified")
                  for index, status in enumerate(statuses)]
        return ElementMatch(claim_number=1, label=label, document_id="2",
                            judgment="일부 차이" if "error" in statuses else "실질적 동일",
                            directness="direct", quote=quote, chunk_id="D2-B-p009-06",
                            verify="verified", limitation_checks=checks)

    labels = ["A", "B", "C", "E"]
    cells = [cell(label, ["accepted", "accepted"]) for label in labels if label != "E"]
    cells.append(cell("E", ["error", "error", "error"]))
    target = Claim(number=1, elements=[ClaimElement(label=label, text=f"구성 {label}",
                                                    importance=5) for label in labels])
    document = Document(id="2", filename="sensors-21-03939.pdf", type="paper",
                        chunks=[Chunk(document_id="2", chunk_id="D2-B-p009-06", page=9,
                                      text=quote)])
    matrix = matrix_for(cells)
    chain = build_chain(target, matrix, {}, [target])
    report = build_claim_report(target, chain, matrix, {"2": document},
                                build_mappings([document], [chain]))
    markdown = to_markdown(AnalysisResult(job_id="x", claim_mapping=[], reports=[report]))

    assert chain.reserved == ["E"]
    assert "차이가 남는 구성" not in markdown
    assert "차이점 판단이 필요합니다" not in markdown
    assert "차이가 남아 있습니다" not in markdown        # 종합 분석 요약까지
    assert "판정을 유보한 구성: E." in markdown
    assert "차이 판단이 아니라 원문 확인이 필요한 항목입니다" in markdown


def test_a_partly_verified_element_keeps_the_familiar_tally():
    """흔한 경우의 표기는 바꾸지 않는다. 세 칸을 늘 적으면 0이 둘 붙은 줄이 매번 나간다."""
    markdown = _rendered(_match([_check(0, "개시된 core"),
                                 _check(1, "개시된 qualifier", kind="qualifier")]))
    assert "한정 2/2 개시" in markdown and "개시 확인" not in markdown


def test_an_element_with_some_confirmed_disclosure_is_still_partial_not_reserved():
    """일부라도 확인됐으면 '판정 유보'가 아니다. 유보는 확인이 0일 때만이다."""
    # 파이프라인이 남기는 모양 그대로다 — 재산출이 이미 상한을 씌워 '일부 차이'가 되어 있다.
    markdown = _rendered(_match([_check(0, "확인된 core"),
                                 _check(1, "검증을 못 받은 qualifier", kind="qualifier",
                                        status="error")], judgment="일부 차이"))
    assert "한정 2개 — 개시 확인 1 · ⚠️ 의미검증 미완료 1 · 미개시 0" in markdown
    assert "판정유보" not in markdown
    assert "부분개시 1" in markdown


# --- 안전장치를 우회하는 경로가 남아 있지 않은지 ---------------------------------

def test_a_whole_call_failure_also_caps_the_grade():
    """항목 몇 건이 빠진 경우보다 **더 나쁜** 경우(한 건도 검증 못 함)가 더 관대하면 안 된다.

    결손 응답 경로에는 상한이 걸리는데 호출 자체가 실패한 경로에는 걸리지 않으면, 우회로가
    하나 열린 채로 남는다. 안전장치는 가장 나쁜 경로에서 가장 먼저 작동해야 한다.
    """
    from app import entailment
    from app.entailment import validate_entailment
    from app.models import Chunk, Document

    quote = ("After optimizing the above camera pose, a robust and accurate global camera "
             "pose was obtained for subsequent triangulation.")
    checks = [LimitationCheck(index=index, limitation=text, kind="core", disclosed=True,
                              quote=quote, chunk_id="D1-B-p001-01", verify="verified")
              for index, text in enumerate(["전역 좌표 정합을 최적화함",
                                            "절대좌표계에 대응하는 최종 3D 모델을 생성함"])]
    match = ElementMatch(claim_number=1, label="E", document_id="1", judgment="실질적 동일",
                         directness="direct", quote=quote, chunk_id="D1-B-p001-01",
                         verify="verified", limitation_checks=checks)
    document = Document(id="1", filename="sensors-21-03939.pdf", type="paper",
                        chunks=[Chunk(document_id="1", chunk_id="D1-B-p001-01", page=1,
                                      text=quote)])

    def broken(prompt, expect="entailments"):
        raise RuntimeError("CLI 응답을 파싱하지 못했습니다")

    import pytest
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(entailment, "run_cli", broken)
    try:
        notes = validate_entailment([match], {"1": document})
    finally:
        monkeypatch.undo()

    assert all(check.semantic_status == "error" for check in checks)
    assert match.judgment == "일부 차이"          # 🟢가 살아남지 않는다
    assert all(check.disclosed for check in checks)   # 개시는 그대로
    assert match.missing_limitations == []
    assert notes and "건너뛰었습니다" in notes[0]


def test_an_unchecked_limitation_cannot_fill_a_gap_for_the_combination():
    """결합의 확정 근거로 쓰이는 자리. 등급 상한이 막지 못하는 우회로다.

    filled_limitations는 "주 인용발명이 빠뜨린 그 한정을 이 문헌이 댔다"를 정하고, 그 판단이
    보조 인용발명 채택으로 이어진다. 확인하지 못한 한정으로 그 문장을 세우면 미완료가
    표시만 눌린 채 문헌 선정으로 들어온다.
    """
    from app.coverage import disclosed_limitations, filled_limitations

    supplement = _match([_check(0, "확인된 한정"),
                         _check(1, "절대좌표계에 대응하는 최종 3D 모델을 생성함", status="error")],
                        document_id="1")
    adopted = _match([_check(0, "확인된 한정")], document_id="2")
    adopted.missing_limitations = ["절대좌표계에 대응하는 최종 3D 모델을 생성함"]

    assert "절대좌표계에 대응하는 최종 3D 모델을 생성함" not in disclosed_limitations(supplement)
    assert filled_limitations(supplement, adopted) == set()


def test_an_all_unverified_cell_does_not_earn_a_scarce_combination_slot():
    """결합 문헌은 두 건뿐이다. 아무것도 확인하지 못한 셀이 그 슬롯을 가져가면 안 된다.

    등급과 누락 수는 둘 다 미검증을 개시 쪽에 세고 계산하므로, 한정을 하나도 확인하지 못한
    셀이 실제로 한 건을 확인한 셀보다 **더 큰** 보완 기여를 받는다(고치기 전 0.60 대 0.30).
    """
    from app.chain import _supplement_step

    current = _match([_check(0, "core 1", disclosed=False, status="rejected"),
                      _check(1, "core 2", disclosed=False, status="rejected")],
                     judgment="차이", document_id="1")
    current.missing_limitations = ["core 1", "core 2"]
    unchecked = _match([_check(0, "core 1", status="error"),
                        _check(1, "core 2", status="error")],
                       judgment="일부 차이", document_id="2")
    verified = _match([_check(0, "core 1"),
                       _check(1, "core 2", disclosed=False, status="rejected")],
                      judgment="일부 유사", document_id="3")
    verified.missing_limitations = ["core 2"]

    assert _supplement_step(unchecked, current) == 0.0
    assert _supplement_step(verified, current) > 0.0


def test_a_candidate_that_confirms_something_new_keeps_its_structural_gain():
    """미검증이 섞였다는 이유로 실제 기여까지 지우면 결합이 서야 할 사건에서 서지 않는다."""
    from app.chain import _supplement_step

    current = _match([_check(0, "core 1", disclosed=False, status="rejected")],
                     judgment="차이", document_id="1")
    current.missing_limitations = ["core 1"]
    mixed = _match([_check(0, "core 1"), _check(1, "core 2", status="error")],
                   judgment="일부 차이", document_id="2")

    assert _supplement_step(mixed, current) > 0.0


def _five(document_id: str, confirmed: int, unchecked: int, judgment: str) -> ElementMatch:
    """한정 5개짜리 셀. 확인 / 미검증 / 나머지는 미개시."""
    labels = [f"core {index}" for index in range(5)]
    checks = []
    for index, label in enumerate(labels):
        if index < confirmed:
            checks.append(_check(index, label))
        elif index < confirmed + unchecked:
            checks.append(_check(index, label, status="error"))
        else:
            checks.append(_check(index, label, disclosed=False, status="rejected"))
    match = _match(checks, judgment=judgment, document_id=document_id, label="A")
    match.missing_limitations = labels[confirmed + unchecked:]
    return match


def test_unverified_limitations_cannot_ride_along_on_one_confirmed_disclosure():
    """"확정 개시가 하나라도 있으면 통과"는 문턱이 너무 낮다.

    1건 확인 + 4건 미검증인 후보가 0.60을 받고, 2건을 실제로 확인한 후보가 0.28에 그쳤다.
    미검증 넷이 확정 개시 하나에 묻어 들어오는 길이다.
    """
    from app.chain import _supplement_step

    current = _five("1", confirmed=0, unchecked=0, judgment="차이")
    mixed = _five("2", confirmed=1, unchecked=4, judgment="일부 차이")
    really = _five("3", confirmed=2, unchecked=0, judgment="일부 유사")

    assert _supplement_step(mixed, current) < _supplement_step(really, current)


def test_a_document_with_nothing_confirmed_never_outranks_one_with_something():
    """감쇠 계수로는 보장할 수 없는 성질이다 — 한정 수가 늘면 다시 뒤집힌다(4.14 대 4.09).

    크기의 문제가 아니라 순서의 문제라, 계수가 아니라 규칙으로 두어야 한다.
    """
    claim = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A", importance=5)])
    for size in (2, 5):
        labels = [f"core {index}" for index in range(size)]
        nothing = _match([_check(index, label, status="error")
                          for index, label in enumerate(labels)],
                         judgment="일부 차이", document_id="9", label="A")
        something = _match([_check(0, labels[0])]
                           + [_check(index, label, disclosed=False, status="rejected")
                              for index, label in enumerate(labels[1:], start=1)],
                           judgment="일부 유사", document_id="8", label="A")
        something.missing_limitations = labels[1:]

        assert score_document(claim, {"A": nothing})[0] == 0.0, size
        assert score_document(claim, {"A": something})[0] > 0.0, size


def test_a_document_with_only_missing_limitations_is_not_zeroed():
    """미검증이 아니라 누락뿐인 셀은 개시가 없다고 **판정된** 것이다. 규칙 대상이 아니다."""
    claim = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A", importance=5)])
    missing_only = _five("7", confirmed=1, unchecked=0, judgment="일부 유사")

    assert score_document(claim, {"A": missing_only})[0] > 0.0


def test_document_ranking_prefers_the_document_that_actually_confirmed_something():
    """주 인용발명 선정은 보완 경로와 다른 계산을 탄다. 거기도 같은 결론이어야 한다."""
    claim = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A", importance=5)])
    unchecked = _match([_check(0, "core 1", status="error"),
                        _check(1, "core 2", status="error")],
                       judgment="일부 차이", document_id="2", label="A")
    verified = _match([_check(0, "core 1"),
                       _check(1, "core 2", disclosed=False, status="rejected")],
                      judgment="일부 유사", document_id="3", label="A")
    verified.missing_limitations = ["core 2"]

    assert score_document(claim, {"A": verified})[0] > score_document(claim, {"A": unchecked})[0]


def test_the_semantic_discount_is_proportional_not_a_cliff():
    """다섯 중 하나가 미검증인 셀과 전부 미검증인 셀은 같은 것이 아니다."""
    from app.coverage import item_similarity

    mostly = _match([_check(index, f"core {index}") for index in range(4)]
                    + [_check(4, "core 4", status="error")], judgment="일부 차이")
    every = _match([_check(index, f"core {index}") for index in range(5)], judgment="일부 차이")
    none_checked = _match([_check(index, f"core {index}", status="error") for index in range(5)],
                          judgment="일부 차이")

    assert item_similarity(none_checked) < item_similarity(mostly) < item_similarity(every)
    assert item_similarity(mostly) > 0.85 * item_similarity(every)


def test_a_verified_limitation_still_fills_a_gap():
    """미완료를 뺀다고 정상 보완까지 막으면, 결합이 서야 할 사건에서 서지 않는다."""
    from app.coverage import filled_limitations

    supplement = _match([_check(0, "절대좌표계에 대응하는 최종 3D 모델을 생성함")], document_id="1")
    adopted = _match([_check(0, "확인된 한정")], document_id="2")
    adopted.missing_limitations = ["절대좌표계에 대응하는 최종 3D 모델을 생성함"]

    assert filled_limitations(supplement, adopted) == {"절대좌표계에 대응하는 최종 3D 모델을 생성함"}


def test_p4_catches_a_full_grade_standing_on_an_unchecked_limitation():
    """등급 상한은 재산출 경로를 탄 셀에만 걸린다. 경로가 아니라 결과를 본다."""
    from app.report import pipeline_invariants
    from app.models import ChainInfo, ClaimReport

    match = _match(_element_e(), judgment="실질적 동일")
    chain = ChainInfo(claim_number=1, track="inventive_step_combination", primary="2")
    report = ClaimReport(claim_number=1, track=chain.track, chain=chain, claims=[])

    notes = pipeline_invariants([report], {1: {"2": {"E": match}}})

    assert any("[불변식 P4]" in note and "문헌 2" in note for note in notes)


def test_p4_stays_quiet_once_the_cap_has_been_applied():
    from app.report import pipeline_invariants
    from app.models import ChainInfo, ClaimReport

    match = _match(_element_e(), judgment="일부 차이")
    chain = ChainInfo(claim_number=1, track="inventive_step_combination", primary="2")
    report = ClaimReport(claim_number=1, track=chain.track, chain=chain, claims=[])

    notes = pipeline_invariants([report], {1: {"2": {"E": match}}})

    assert not [note for note in notes if "[불변식 P4]" in note]


# --- 네 소비자가 같은 함수를 본다 ------------------------------------------------

def test_every_consumer_reads_the_same_classifier():
    """소비자가 각자 semantic_status를 다시 해석하면 규칙이 갈라진다.

    coverage 밖에서 'error'를 문자열로 직접 비교하는 곳이 있으면, 상태를 하나 더 늘릴 때
    그곳만 빠진다. 실제로 이 파이프라인은 같은 형태로 여러 번 어긋났다 — 판정 사다리를
    모듈마다 따로 들어 같은 조건이 자리마다 다른 등급을 받았다.
    """
    import pathlib
    # 상태를 세우는 쪽(entailment)·형을 정의하는 쪽(models)·가르는 쪽(coverage)만 예외다.
    allowed = {"coverage.py", "entailment.py", "models.py"}
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(pathlib.Path(coverage.__file__).parent.glob("*.py"))
        if path.name not in allowed
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "semantic_status" in line and "error" in line]
    assert offenders == []
