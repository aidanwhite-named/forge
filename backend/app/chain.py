"""인용발명 선정. 비교 매트릭스가 확정된 뒤로는 LLM을 한 번도 부르지 않습니다.

같은 매트릭스가 들어오면 항상 같은 조합이 나옵니다. 순위 결정에 LLM을 쓰면
재실행마다 결론이 흔들려 판정 캐시도 감사 기록도 의미가 없어집니다.
"""
from .claims import ancestry
from .consistency import antecedents
from .coverage import (
    JUDGMENT_RANK, PRIMARY_CANDIDATE_RATIO, SUBSTANTIVE_IMPORTANCE,
    best_match, combination_supported, combined_similarity, core_direct_score, core_elements,
    derive_judgment,
    difference_labels, disclosed_limitations, evidenced_limitations, filled_limitations,
    has_correspondence, ineligible_reason, is_better_match, is_complete,
    is_eligible_supplement, judgment_at_rank, limitation_gain, no_correspondence_labels,
    limitation_state_map, pending_labels, reserved_labels, residual_difference, rows_for,
    score_document,
    supplement_gain, supplement_needed_labels, supplement_reason, unverified_count,
    well_known_labels,
)
from .models import (Claim, ChainInfo, DocumentScore, ElementCoverage, ElementMatch,
                     NoveltyScreen, SupplementCandidate)

# 이 판정들만 단일문헌 신규성의 직접 개시로 인정합니다.
NOVELTY_DIRECT = {"동일", "실질적 동일"}
# 완전 미대응을 메우는 이득에 주는 우선순위. 남은 결합 여유를 품질 개선에 먼저 쓰지 않게 합니다.
GAP_PRIORITY = 3.0
# 보완 문헌을 하나 더 끌어오기 위해 요구하는 최소 이득. 상한에 닿기 전에도 새로 기여하는
# 것이 없으면 여기서 멈춥니다.
MIN_SUPPLEMENT_GAIN = 0.05
# 독립항은 업로드된 문헌 중 실제 증분 기여가 있는 문헌을 모두 채택할 수 있습니다. 고정된 2건
# 상한은 청구항의 서로 다른 공백을 문헌 2와 3이 각각 메우는 사건에서 한쪽 근거를 버렸습니다.
# 결합 동기·용이성은 보고서가 별도로 유보하므로, 여기서는 구성 커버리지를 인위적으로 자르지 않습니다.
# 종속항이 부모항 조합에 **새로 더할 수 있는** 문헌 수. 종속항의 추가 한정은 대개 한 줄이라,
# 그것 하나를 위해 문헌을 여러 건 더 끌어오면 거절 이유가 실무에서 설득력을 잃습니다.
MAX_DEPENDENT_ADDITIONS = 1


Matrix = dict[str, dict[str, ElementMatch]]   # document_id → label → 판정


def build_chain(claim: Claim, matrix: Matrix, parent_chains: dict[int, ChainInfo],
                all_claims: list[Claim]) -> ChainInfo:
    """청구항 1건의 인용발명 조합을 확정합니다."""
    scores = [DocumentScore(document_id=document_id, main_score=score, detail=detail)
              for document_id, (score, detail) in
              ((document_id, score_document(claim, matches)) for document_id, matches in matrix.items())]
    scores.sort(key=lambda item: (-item.main_score, item.document_id))
    chain = ChainInfo(claim_number=claim.number, candidates=scores)

    # 판정을 **받지 못한** 셀이 하나라도 있으면 여기서 멈춥니다. 미판정을 "대응 없음"으로
    # 흘려보내면 분석 실패가 "인용발명에 그런 기재가 없다"는 결론으로 둔갑하고, 그 결론은
    # 출원인에게 유리한 방향이라 검토 과정에서 이의가 제기되지도 않습니다.
    incomplete = _incomplete_reasons(claim, matrix, parent_chains)
    if incomplete:
        return _incomplete_chain(claim, chain, matrix, incomplete)

    # 종속항 행렬에는 부모항이 아니라 "에 있어서" 뒤의 추가 한정만 들어 있다.
    # 이 한정만 단일 문헌에 있다고 종속항 전체의 신규성을 부정하면 안 되므로,
    # 부모 조합 상속을 신규성 게이트보다 먼저 적용한다.
    if claim.depends_on and claim.depends_on in parent_chains:
        return _dependent_chain(claim, matrix, parent_chains[claim.depends_on], chain, all_claims)

    novelty = screen_novelty(claim, matrix, scores)
    chain.novelty = novelty
    if novelty.selected_document:
        chain.track = "novelty_single"
        chain.primary = novelty.selected_document
        chain.combined_similarity = combined_similarity(claim, matrix[novelty.selected_document])
        chain.rationale = "단일 인용발명이 모든 필수 구성을 직접 개시하므로 문헌 결합을 검토하지 않습니다."
        return _finalize(claim, chain, dict(matrix[novelty.selected_document]), matrix)
    return _independent_chain(claim, matrix, chain)


# --- 미판정 처리 --------------------------------------------------------------

def _incomplete_reasons(claim: Claim, matrix: Matrix, parent_chains: dict[int, ChainInfo]) -> list[str]:
    """판정을 받지 못한 셀의 사유. 부모항이 미완료면 종속항도 결론을 낼 수 없습니다."""
    reasons: list[str] = []
    for document_id in sorted(matrix):
        for element in claim.elements:
            match = matrix[document_id].get(element.label)
            if match is not None and match.error and match.error not in reasons:
                reasons.append(match.error)
    parent = parent_chains.get(claim.depends_on) if claim.depends_on else None
    if parent is not None and parent.track == "analysis_incomplete":
        reasons.append(f"부모 청구항 {parent.claim_number}의 구성대비가 완료되지 않았습니다.")
    return reasons


def _incomplete_chain(claim: Claim, chain: ChainInfo, matrix: Matrix, reasons: list[str]) -> ChainInfo:
    """결론을 만들지 않고 미판정 사유만 남깁니다.

    uncovered를 채우지 않는 것이 중요합니다. uncovered는 "어느 인용발명에도 대응 기재가
    없다"는 확정 진술이고, 판정을 받지 못한 상태에서 그렇게 적으면 근거 없는 사실 주장이
    됩니다. 문헌별 원자료는 element_coverage에 그대로 남으므로 정보가 사라지지는 않습니다.
    """
    chain.track = "analysis_incomplete"
    chain.incomplete_reasons = reasons
    chain.rationale = ("구성대비 판정을 받지 못한 셀이 있어 신규성·진보성 판단을 수행하지 않았습니다. "
                       "미판정은 '대응 없음'과 다르므로 이 보고서로 거절 가부를 판단하지 마십시오.")
    return _finalize(claim, chain, {}, matrix)


# --- 신규성 우선 판단 --------------------------------------------------------

def blocking_labels(claim: Claim, labels: list[str]) -> list[str]:
    """결론을 막는 공백만 남깁니다.

    전제부는 대비도 하고 보고서에도 남기지만 여기서는 빼냅니다. 전제부가 한정적 의미를
    갖는지는 사건마다 다른 법적 판단이고, 실무에서 흔한 "…장치에 있어서" 같은 범주 기재를
    하드 게이트로 삼으면 거의 모든 청구항이 "거절 이유 구성 곤란"으로 나옵니다.
    대신 preamble_undisclosed에 남겨 감사 데이터에 그대로 노출합니다.
    """
    preamble = {element.label for element in claim.elements if element.is_preamble}
    return [label for label in labels if label not in preamble]


def preamble_gaps(claim: Claim, labels: list[str]) -> list[str]:
    preamble = {element.label for element in claim.elements if element.is_preamble}
    return [label for label in labels if label in preamble]


def screen_novelty(claim: Claim, matrix: Matrix, scores: list[DocumentScore]) -> NoveltyScreen:
    """모든 문헌을 서로 결합하지 않고 독립적으로 심사합니다."""
    complete: list[str] = []
    missing_by_document: dict[str, list[str]] = {}
    for document_id, matches in matrix.items():
        missing = [element.label for element in claim.elements
                   if not _directly_disclosed(matches.get(element.label))]
        # missing_by_document에는 전제부도 그대로 남깁니다. 게이트에서만 빼냅니다.
        missing_by_document[document_id] = missing
        if claim.elements and not blocking_labels(claim, missing):
            complete.append(document_id)
    by_score = {score.document_id: score.main_score for score in scores}
    selected = max(complete, key=lambda document_id: (by_score.get(document_id, 0.0), document_id)) if complete else None
    return NoveltyScreen(
        selected_document=selected,
        complete_documents=complete,
        missing_by_document=missing_by_document,
        result="single_document_complete" if selected else "no_single_document_complete",
    )


def _directly_disclosed(match: ElementMatch | None) -> bool:
    """직접 근거·동일급 판정·누락 제한 부재가 모두 확인되어야 합니다.

    verify는 "verified"만 받습니다. "partial"은 다분절 인용문 중 일부 분절만 원문에서
    확인됐다는 뜻입니다. 지금은 verify.py가 partial을 반드시 inferred로 강등하므로 아래
    directness 조건에도 걸리지만, 신규성 부정은 청구항을 죽이는 가장 강한 결론이라
    다른 모듈의 불변식에 기대지 않고 이 게이트에서 직접 막습니다.

    의미검증 미완료도 같은 이유로 여기서 직접 막습니다. derive_judgment의 상한이 이미
    동일급을 못 주게 하지만, 그 상한은 판정을 재산출하는 경로를 탄 셀에만 걸립니다. 이
    게이트는 그 경로가 돌았는지와 무관하게 성립해야 합니다.
    """
    return bool(
        match
        and not match.error
        and match.quote
        and match.verify == "verified"
        and match.judgment in NOVELTY_DIRECT
        and match.directness == "direct"
        and not match.missing_limitations
        and not unverified_count(match)
    )


# --- 독립항 결합 -------------------------------------------------------------

def _independent_chain(claim: Claim, matrix: Matrix, chain: ChainInfo) -> ChainInfo:
    eligible = _eligible_primaries(claim, matrix, chain.candidates)
    if not eligible:
        return _no_primary_chain(claim, matrix, chain)

    # 자격 게이트를 통과한 후보 중 단독 적합도 1위를 주 인용발명으로 확정합니다.
    chain.primary = eligible[0]
    chain.track = "inventive_step_combination"
    chain.combination_limit = 0                # 0 = 독립항에는 인위적인 문헌 수 상한 없음
    merged = dict(matrix[chain.primary])
    # 보완 검토 대상은 미커버 구성보다 넓습니다. '일부 차이'로 커버된 구성도 여기 들어옵니다.
    chain.supplement_needed = supplement_needed_labels(claim, merged)

    # 상한 안에서, 새로 기여하는 문헌이 있는 한 결합합니다. 이득이 문턱 아래로 떨어지면
    # 상한에 닿기 전에도 멈춥니다.
    while len(chain.secondaries) + 1 < len(matrix):
        # 전제부는 보고서에 대비하되 법적 한정 여부를 코드가 정하지 않으므로, 그것만 보강하려고
        # 보조 인용발명을 늘리지 않습니다. 실측에서는 핵심 구성에 기여하지 않는 논문이 P0 한 줄
        # 때문에 세 번째 채택 문헌이 되어 보고서만 길어졌습니다.
        targets = blocking_labels(claim, supplement_needed_labels(claim, merged))
        if not targets:
            break
        candidate = _best_secondary(claim, matrix, merged, targets,
                                    no_correspondence_labels(claim, merged),
                                    exclude={chain.primary, *chain.secondaries},
                                    scores=_fitness(chain))
        if candidate is None:
            break
        chain.secondaries.append(candidate)
        merged = _merge(claim, merged, matrix[candidate])

    chain.limit_binding = False

    chain.uncovered = no_correspondence_labels(claim, merged)
    # 대응은 있으나 하위 한정이나 구현 방식에 차이가 남는 구성입니다.
    chain.residual = difference_labels(claim, merged)
    chain.reserved = reserved_labels(claim, merged)
    chain.combined_similarity = combined_similarity(claim, merged)
    _apply_gap_policy(claim, chain, matrix, merged)
    return _finalize(claim, chain, merged, matrix)


def _no_primary_chain(claim: Claim, matrix: Matrix, chain: ChainInfo, cause: str = "") -> ChainInfo:
    """세울 조합이 없는 상태. 결론만 접고 구성대비는 그대로 보고합니다.

    이 게이트가 답하는 것은 **거절 이유를 세울 수 있는가**이지, 문헌에 대응 기재가 있는가가
    아닙니다. 둘을 같은 칸에 넣어 빈 조합(merged={})으로 마감하면 두 가지가 함께 무너집니다.
    원문 대조까지 통과한 '실질적 동일·direct' 대응이 있는 구성까지 "어느 인용발명에서도
    확인되지 않았다"고 단정하게 되고, 보고서 본문에서 구성대비 결과가 통째로 사라집니다
    (report.build_claim_report는 채택 문헌 목록으로 본문을 만듭니다).

    미판정을 '대응 없음'으로 흘려보내지 않는 것(_incomplete_chain)과 같은 이유로, 확인된
    대응을 없는 것으로 적지 않습니다. 채택은 하지 않으므로 primary·secondaries는 비운 채
    두고, 구성별로 가장 강한 대응만 모아 보고용으로 씁니다. 그 대응을 가진 문헌은
    reference_only에 남으므로 역할은 '미채택' 그대로입니다.

    cause는 왜 조합을 세우지 못했는지입니다. 독립항은 자격 게이트를 통과한 문헌이 없어서고,
    종속항은 부모항이 세운 조합이 없어서입니다. 결론이 같아도 다음에 해야 할 일이 다르므로
    사유를 뭉뚱그리지 않습니다.
    """
    chain.track = "rejection_impossible"
    # 동률일 때 항상 같은 문헌이 뽑히도록 문헌 순서를 고정합니다(_best_secondary와 같은 이유).
    reference = {element.label: best_match([matrix[document_id].get(element.label)
                                            for document_id in sorted(matrix)])
                 for element in claim.elements}
    chain.uncovered = no_correspondence_labels(claim, reference)
    chain.residual = difference_labels(claim, reference)
    chain.reserved = reserved_labels(claim, reference)
    chain.combined_similarity = combined_similarity(claim, reference)
    corresponded = [element.label for element in claim.elements
                    if element.label not in chain.uncovered]
    chain.reference_only = sorted({match.document_id for label in corresponded
                                   if (match := reference.get(label)) is not None})
    pending = pending_labels(matrix, chain.uncovered)
    chain.combination_pending = [label for label in chain.uncovered if label in pending]
    chain.combination_pending_reasons = pending
    reasons = [cause or "차별적 핵심 구성을 직접 개시한 인용발명이 없어 주 인용발명을 세우지 못했습니다."]
    if corresponded:
        reasons.append(f"구성 {', '.join(corresponded)}에는 대응 기재가 확인되었으나, "
                       "주 인용발명이 서지 않아 인용발명 조합을 확정하지 않았습니다.")
    blocking = [label for label in blocking_labels(claim, chain.uncovered)
                if label not in chain.combination_pending]
    if blocking:
        reasons.append(f"구성 {', '.join(blocking)}은 어느 인용발명에서도 대응 기재가 확인되지 않았습니다.")
    chain.rationale = " ".join(reasons) + _pending_clause(chain)
    return _finalize(claim, chain, reference, matrix)


# --- 결합 한도 밖의 기재와 주지관용 ------------------------------------------

def _apply_gap_policy(claim: Claim, chain: ChainInfo, matrix: Matrix,
                      merged: dict[str, ElementMatch]) -> None:
    """남은 공백을 세 가지로 갈라 트랙과 결론 문장을 정합니다.

      - 채택되지 않은 문헌에 검증된 대응 기재가 있는 구성 → beyond_limit
      - 주지관용기술로 다룰 수 있는 구성 → well_known (거절 이유는 그대로 성립)
      - 나머지 → 진짜 공백. 이때만 "어느 인용발명에서도 확인되지 않았다"고 적습니다.

    셋을 한 칸에 뭉치면 보고서가 사실과 다른 진술을 하게 됩니다. 채택하지 않은 문헌의 기재를
    "없다"고 적으면 이미 손에 든 문헌을 다시 찾게 되고, 주지관용으로 충분한 범용 구성 하나
    때문에 거절 이유 전체가 "구성 곤란"으로 떨어지면 실제로 설 수 있는 거절이 사라집니다.

    **왜 채택되지 않았는지는 따로 셉니다(limit_binding).** 이 목록이 답하는 것은 "그 기재가
    어디 있는가"까지입니다. 조합에 자리가 남아 있는데도 빠진 문헌을 두고 "상한을 넘었다"고
    적으면, 도구가 하지 않은 판단을 한 것처럼 보고하게 됩니다.
    """
    adopted = {chain.primary, *chain.secondaries}
    overflow: dict[str, list[str]] = {}
    for label in chain.uncovered:
        sources = sorted(document_id for document_id, matches in matrix.items()
                         if document_id not in adopted and has_correspondence(matches.get(label)))
        if sources:
            overflow[label] = sources
    chain.beyond_limit = [label for label in chain.uncovered if label in overflow]
    chain.beyond_limit_documents = overflow
    chain.beyond_limit_residual = _residual_overflow(chain, matrix, merged, adopted)

    # 축 결손으로 기각된 한정이 걸린 구성은 공백과 성격이 다릅니다(models.combination_pending).
    # 결론을 막는 목록에서 먼저 떼어 내야, 결합 위에서 다시 물어야 할 것이 "어느 인용발명에도
    # 없다"는 확정 진술로 굳지 않습니다.
    pending = pending_labels(matrix, chain.uncovered)
    chain.combination_pending = [label for label in chain.uncovered if label in pending]
    chain.combination_pending_reasons = pending

    blocking = blocking_labels(claim, chain.uncovered)
    well_known = well_known_labels(claim, matrix, blocking)
    chain.well_known = [label for label in blocking if label in well_known]
    chain.well_known_documents = well_known

    blocking = [label for label in blocking
                if label not in well_known and label not in chain.combination_pending]
    if not blocking:
        chain.rationale = _combination_rationale(chain)
        return
    chain.track = "rejection_impossible"
    genuine = [label for label in blocking if label not in overflow]
    limited = [label for label in blocking if label in overflow]
    reasons: list[str] = []
    if genuine:
        reasons.append(f"구성 {', '.join(genuine)}의 청구항 한정 전체를 충족하는 기재가 "
                       "어느 인용발명에서도 확인되지 않았습니다.")
    if limited:
        reasons.append(f"구성 {', '.join(limited)}에는 대응 기재를 가진 인용발명이 있으나, "
                       f"{_unadopted_reason(chain)} 이 거절 이유에 세우지 않았습니다.")
    chain.rationale = (" ".join(reasons) + " 이대로는 거절 이유를 구성하기 어렵습니다."
                       + _pending_clause(chain))


def _unadopted_reason(chain: ChainInfo) -> str:
    """채택되지 않은 문헌에 기재가 있을 때, 그 문헌이 빠진 이유를 한 구로 적습니다.

    보고서 세 곳(결론·미대응 줄·차이점 줄)이 같은 사실을 설명하므로 문구를 여기서 한 번만
    정합니다. report._unadopted_note가 같은 값을 씁니다.
    """
    if chain.limit_binding:
        if chain.inherited:
            return f"종속항 추가 인용발명 수 상한({chain.combination_limit}건)을 넘어"
        return f"결합 문헌 수 상한({chain.combination_limit}건)을 넘어"
    return "보완 후보 평가에서 채택되지 않아"


def _residual_overflow(chain: ChainInfo, matrix: Matrix, merged: dict[str, ElementMatch],
                       adopted: set[str | None]) -> dict[str, dict[str, list[str]]]:
    """대응은 되었는데 **남은 차이**를 미채택 문헌이 메우는 경우를 한정 단위로 찾습니다.

    beyond_limit이 지키는 것은 "(X) 구성에 대응되는 인용발명이 확인되지 않음" 줄이고, 이쪽이
    지키는 것은 "→ 차이점: …" 줄입니다. 둘은 같은 사실을 서로 다른 자리에서 부정합니다 —
    구성 전체가 빠졌든 한정 하나가 빠졌든, 업로드된 문헌에 원문이 있는데 없다고 적으면 심사관은
    이미 손에 든 문헌을 다시 찾아 나서게 됩니다. 채택 조합 밖의 문헌이 그 한정을 원문으로
    개시하고 있다면 그 사실을 남은 차이로만 적어서는 안 됩니다.

    **채택 조합이 이미 메운 한정은 여기 오지 않습니다.** merged를 보므로, _absorb_limitations가
    조합 안의 다른 문헌으로 메운 한정은 missing에서 빠져 애초에 후보가 되지 않습니다. 그것은
    미채택 문헌을 가리킬 일이 아니라 결합으로 해소된 것이라고 적을 일입니다.

    구성 단위 대응(has_correspondence)이 아니라 **그 한정 자체**를 개시했는지로 봅니다.
    구성에 대응이 있다는 것만으로는 정작 빠진 그 한정이 있다는 뜻이 아닙니다.
    """
    overflow: dict[str, dict[str, list[str]]] = {}
    for label in chain.residual:
        missing = (merged.get(label).missing_limitations if merged.get(label) else []) or []
        for limitation in missing:
            sources = sorted(document_id for document_id, matches in matrix.items()
                             if document_id not in adopted
                             and limitation in disclosed_limitations(matches.get(label)))
            if sources:
                overflow.setdefault(label, {})[limitation] = sources
    return overflow


def _eligible_primaries(claim: Claim, matrix: Matrix, scores: list[DocumentScore]) -> list[str]:
    """차별적 핵심 구성의 직접 개시량이 최고 문헌에 크게 못 미치면 후보에서 제외합니다.

    다만 전체 점수가 낮아도 **다른 후보가 갖지 못한** 핵심 구성을 원문으로 직접 개시한
    문헌은 남깁니다. 평균 점수에 희석되어 유효한 후보가 탈락하지 않게 하기 위한 예외입니다.

    임계는 절대 차가 아니라 **비율**입니다(PRIMARY_CANDIDATE_RATIO). core_direct는 최고
    문헌도 0.5 안팎이라, 절대값을 빼는 방식으로는 점수 분포에 따라 임계의 뜻이 달라집니다.

    예외는 **고유 기여**에만 적용합니다. 핵심 구성 중에는 후보 대부분이 함께 개시하는 것이
    있어서, 하나라도 직접 개시하면 되살리는 규칙은 사실상 전 문헌을 통과시킵니다. 모두가
    가진 것을 가졌다는 사실은 주 인용발명 자격의 근거가 되지 못하므로, 이미 통과한 후보들이
    **직접 개시하지 못한** 핵심 구성을 이 문헌이 개시할 때만 되살립니다.
    """
    core_labels = [element.label for element in core_elements(claim)]
    # 문헌 순위(score_document)와 **같은 정의**를 씁니다. 한쪽만 중요도 가중을 빼면 같은
    # 이름의 지표가 두 곳에서 다른 값이 되고, 마진도 서로 다른 척도 위에서 비교됩니다.
    core_direct = {document_id: core_direct_score(claim, matches)
                   for document_id, matches in matrix.items()}
    if not core_direct:
        return []
    top = max(core_direct.values())
    if top <= 0.0:
        # 핵심으로 분류된 구성 중 어느 것도, 어느 문헌에서도 직접 개시되지 않은 상태입니다.
        # 그러면 중요도 4 이상이라는 선은 이 사건에서 아무 후보도 남기지 못한 것이므로 실질
        # 구성(3 이상)까지 한 칸 넓혀 다시 잽니다.
        #
        # 중요도는 분해와 함께 LLM이 매 실행 새로 매기는 값이라 같은 청구항에서도 한 칸씩
        # 흔들립니다. 4를 하드 게이트로 두면, 잘 개시된 구성 하나가 3으로 내려앉는 순간
        # 게이트가 통째로 비어 전 문헌이 '미채택'이 됩니다. 자격 게이트가 답할 질문은 후보
        # 사이의 우열이지 "거절 이유를 세울 수 있는가"가 아닙니다.
        #
        # **2 이하로는 내려가지 않습니다.** 분해 프롬프트가 1~2를 "어느 발명에나 나타나는 범용
        # 부품"과 "통상의 인터페이스·입출력 구성"으로 정의하므로, 거기까지 넓히면 범용 구성
        # 하나를 개시했다는 이유로 무관한 문헌이 주 인용발명으로 찍힙니다. 그것은 이 게이트가
        # 애초에 막으려던 것입니다.
        fallback = [element for element in claim.elements
                    if element.importance >= SUBSTANTIVE_IMPORTANCE]
        core_direct = {document_id: core_direct_score(claim, matches, fallback)
                       for document_id, matches in matrix.items()} if fallback else {}
        core_labels = [element.label for element in fallback]
        top = max(core_direct.values(), default=0.0)
    if top <= 0.0:
        # 어느 문헌도 어떤 구성도 직접 개시하지 못했습니다. 비율 임계만 놓고 보면 0점 문헌이
        # 전부 자격을 얻어 무관한 문헌이 "주 인용발명"으로 보고서에 찍힙니다.
        return []
    eligible = [score.document_id for score in scores
                if core_direct.get(score.document_id, 0.0) >= top * PRIMARY_CANDIDATE_RATIO]

    def directly_covered(document_id: str) -> set[str]:
        return {label for label in core_labels
                if _directly_disclosed(matrix.get(document_id, {}).get(label))}

    covered = {label for document_id in eligible for label in directly_covered(document_id)}
    for score in scores:
        if score.document_id in eligible:
            continue
        unique = directly_covered(score.document_id) - covered
        if unique:
            eligible.append(score.document_id)
            covered |= unique
    order = {score.document_id: index for index, score in enumerate(scores)}
    return sorted(eligible, key=lambda document_id: order.get(document_id, len(order)))


def _fitness(chain: ChainInfo) -> dict[str, float]:
    """문헌별 단독 적합도. 보완 후보의 기여가 동률일 때의 차순위 기준입니다."""
    return {score.document_id: score.main_score for score in chain.candidates}


def _best_secondary(claim: Claim, matrix: Matrix, merged: dict[str, ElementMatch],
                    targets: list[str], gaps: list[str], exclude: set[str],
                    scores: dict[str, float] | None = None) -> str | None:
    """주 인용발명 대비 **증분**으로 평가합니다. 세 종류를 함께 봅니다.

      - 완전 미대응 구성: 검증된 대응 기재를 처음 제공하면 이득으로 셉니다(GAP_PRIORITY 가중).
      - 부분 대응 구성: 판정·직접성·근거·누락 한정 중 하나 이상이 실제로 개선될 때만 셉니다.
      - 그 어느 쪽도 아니지만 **채택 셀이 빠뜨린 한정을 원문으로 개시한** 구성.

    절대 성능이 높아도 어느 쪽에도 기여하지 못하면 채택하지 않습니다.

    세 번째가 없으면 결합이 서지 않습니다. 부 인용발명은 구성 전체를 주 인용발명보다 잘
    개시하는 문헌이 아니라 빠진 한정 하나를 대는 문헌이므로, 구성 단위 우열(is_better_match)
    하나로 문을 지키면 주 인용발명이 '일부 차이'만 되어도 보완이 사실상 불가능해집니다
    (coverage.filled_limitations 참조).

    공백에 유사도 하한을 걸지 않는 이유: 그 구성의 유일한 대응 기재를 가진 문헌이라도
    판정이 '일부 차이'면 하한에 걸려 탈락하고, 결과적으로 "어느 문헌에도 대응이 없다"는
    보고서가 나옵니다. 결합해서 남는 차이는 버리지 않고 보고서의 차이점으로 남깁니다.

    **동률 처리.** 같은 한정을 여러 문헌이 개시하면 증분 이득이 정확히 같아집니다. 그때 문헌
    번호가 빠른 것을 집으면 청구항과 거의 무관한 문헌이 번호만 앞선다는 이유로 부 인용발명이
    됩니다. 기여도가 같다면 **청구항에 전체적으로
    더 가까운 문헌**(단독 적합도)을 세우는 것이 더 방어 가능한 거절 이유이므로 그것을 다음
    기준으로 둡니다. 그래도 같으면 문헌 번호 순이라 결과는 항상 재현됩니다.
    """
    fitness = scores or {}
    best, best_key = None, None
    for document_id, (gain, useful) in _supplement_gains(
            claim, matrix, merged, targets, gaps, exclude).items():
        if not useful:
            continue
        key = (gain, useful, fitness.get(document_id, 0.0))
        if best_key is None or key > best_key:
            best, best_key = document_id, key
    return best if best_key and best_key[0] >= MIN_SUPPLEMENT_GAIN else None


def _supplement_gains(claim: Claim, matrix: Matrix, merged: dict[str, ElementMatch],
                      targets: list[str], gaps: list[str],
                      exclude: set[str]) -> dict[str, tuple[float, int]]:
    """문헌별 (증분 이득, 기여한 구성 수). 선정과 감사 기록이 **같은 값**을 보게 합니다.

    이 계산이 _best_secondary 안에만 있으면, 채택되지 않은 문헌이 왜 빠졌는지 나중에 다시
    물을 수 없습니다. 후보 행에는 주 인용발명 대비 이득만 남아 있어서, 이미 채택된 보조
    인용발명이 같은 것을 대고 있는 중복 후보와 아무도 대지 못한 것을 대는 후보가 같아 보입니다.

    문헌 순서를 고정해 넣습니다(dict는 삽입 순서를 지킵니다). 동률일 때 항상 같은 문헌이
    뽑혀야 재실행 결과가 재현됩니다.
    """
    gap_labels = set(gaps)
    gains: dict[str, tuple[float, int]] = {}
    for document_id in sorted(matrix):
        if document_id in exclude:
            continue
        matches = matrix[document_id]
        gain, useful = 0.0, 0
        for label in targets:
            candidate, current = matches.get(label), merged.get(label)
            if not is_eligible_supplement(candidate):
                continue
            step = _supplement_step(candidate, current)
            if step <= 0.0:
                continue
            if label in gap_labels and not _fills_a_gap(candidate, current):
                continue
            weight = GAP_PRIORITY if label in gap_labels else 1.0
            gain += weight * step * _importance(claim, label)
            useful += 1
        gains[document_id] = (round(gain, 6), useful)
    return gains


def _fills_a_gap(candidate: ElementMatch | None, current: ElementMatch | None) -> bool:
    """공백 구성에서 이 후보를 데려올 이유가 실제로 있는지. **한정 단위로** 묻습니다.

    구성 단위 판정 라벨(has_correspondence)로 문을 지켜서는 안 됩니다. 공백이라는 말은 채택
    셀의 라벨이 이미 '차이'·'대응 없음'이라는 뜻이고, 부 인용발명은 구성 전체가 아니라 빠진
    한정 하나를 대는 문헌이라 후보의 라벨도 낮은 것이 정상입니다. 그러면 두 셀을 어떻게 합쳐도
    조건이 참이 될 수 없습니다 — 공백을 메울 후보만 골라 놓고 공백이 이미 메워져 있을 것을
    요구하는 순환이고, 보조 인용발명이 한 건도 서지 못합니다.

    그렇다고 "검증된 발췌가 있으면 통과"로 열어서도 안 됩니다. 그러면 구성이 공백으로 남는데
    문헌만 보조 인용발명으로 이름이 올라갑니다 — 거절 이유에 세운 문헌이 정작 그 구성에는
    아무것도 보태지 못한 상태입니다. 그래서 **개시된 한정이 실제로 있을 것**을 요구합니다.
    개시로 확정된 것이든, 원문 근거를 내고 축 결손으로만 기각된 것이든(결합 위에서 다시 물을
    대상이므로) 둘 다 기여입니다.

    **filled_limitations로 재지 않습니다.** 그것은 후보가 개시한 한정 문언과 채택 셀의 누락
    한정 문언이 **문자열로 같을 것**을 요구하는데, 두 셀을 각각 판정한 결과라 같은 요구사항도
    표현이 갈립니다. 공백 구성에서는 채택 셀이 애초에 아무것도 개시하지 않았으므로 대조할
    상대가 없고, 물어야 할 것은 "후보가 여기에 원문을 보태는가" 하나뿐입니다.

    한정 점검이 아예 없는 셀은 잴 재료가 없으므로 구성 단위 판정으로 되돌아갑니다.
    """
    if candidate is None or not is_eligible_supplement(candidate):
        return False
    if candidate.limitation_checks:
        return bool(evidenced_limitations(candidate))
    return has_correspondence(best_match([current, candidate]))


def _supplement_step(candidate: ElementMatch | None, current: ElementMatch | None) -> float:
    """후보 한 셀의 실제 보완 기여. 선정과 감사 데이터가 같은 값을 쓰게 합니다.

    구성 전체의 우열이 개선되면서 빠진 한정도 메우는 후보가 있을 수 있습니다. 둘을 if/else로
    가르면 우열이 조금 개선됐다는 이유로 더 큰 한정 보완 이득이 가려집니다. 같은 누락 감소를
    두 번 더하지 않도록 두 경로 중 큰 값을 사용합니다.

    **미검증만으로 생긴 우열은 구조적 이득으로 세지 않습니다.** 등급과 누락 수는 둘 다
    미검증 한정을 개시 쪽에 세고 계산합니다(derive_judgment의 상한, missing_limitations).
    그래서 한정을 하나도 확인하지 못한 셀이 실제로 한 건을 확인한 셀보다 **더 큰** 보완
    기여를 받습니다 — 실측 형태로 재현하면 0.60 대 0.30이었습니다. 결합 문헌은
    고정 결합 상한이 있던 버전에서는 그 셀이 유일한 보조 슬롯을 소비해 실제로 한정을
    확인한 문헌을 밀어내고 결론까지 바꾸기도 했습니다.

    **"확정 개시가 하나라도 있으면 통과"로는 부족합니다.** 그 문턱은 미검증 넷이 확정 개시
    하나에 묻어 들어오는 길을 그대로 둡니다 — 1건 확인+4건 미검증인 후보가 0.60을 받고,
    2건을 실제로 확인한 후보가 0.28에 그칩니다. 그래서 미검증이 하나라도 있으면 구조적
    이득 자체를 쓰지 않고, 확정한 한정의 기여(limitation_gain)만 인정합니다. 그쪽은
    disclosed_limitations에서 나오므로 정의상 확인된 것만 셉니다.

    이렇게 해도 실제 기여가 지워지지는 않습니다. 미검증이 섞였다는 이유로 확정 개시까지
    버리면 결합이 서야 할 사건에서 서지 않는데(entailment._mark_unchecked가 directness를
    absent로 내려 같은 사고를 낸 적이 있습니다), limitation_gain이 그 몫을 그대로 냅니다.
    """
    structural = supplement_gain(candidate, current) if is_better_match(candidate, current) else 0.0
    if unverified_count(candidate):
        structural = 0.0
    return max(0.0, structural, limitation_gain(candidate, current))


def _merge(claim: Claim, current: dict[str, ElementMatch], addition: dict[str, ElementMatch]) -> dict[str, ElementMatch]:
    """구성별로 더 강한 판정을 채택합니다. 결합 후 커버리지는 여기서만 정해집니다."""
    merged = {}
    for element in claim.elements:
        cells = [current.get(element.label), addition.get(element.label)]
        merged[element.label] = _absorb_limitations(best_match(cells), cells)
    return _restore_combination_antecedents(claim, merged)


def _absorb_limitations(chosen: ElementMatch | None,
                        cells: list[ElementMatch | None]) -> ElementMatch | None:
    """채택 셀이 빠뜨린 한정을 같은 조합의 다른 문헌이 개시했다면 결합 결과에 반영합니다.

    best_match는 구성 하나를 문헌 하나에 통째로 넘깁니다. 그대로 두면 진 쪽 셀이 원문으로
    개시한 한정까지 함께 버려지고, 이긴 셀의 missing_limitations가 그대로 "결합 후에도 남는
    차이"가 됩니다 — 그 한정을 개시한 문헌을 **같은 조합 안에** 세워 두고도 그렇습니다.
    _best_secondary가 바로 그 기여를 보고 채택한 문헌인데 결과에는 반영되지 않는 셈입니다.

    **판정 라벨은 채운 한정에서 다시 유도합니다.** 등급은 한정별 개시 여부의 결정론적
    함수이므로(coverage.derive_judgment), 한정을 채워 놓고 등급만 그대로 두면 같은 셀 안에서
    두 값이 서로 모순합니다. 그 모순된 등급은 has_correspondence를 거쳐 "이 구성은 어느
    인용발명에도 없다"는 **사실 진술**로 나갑니다. 조합이 그 한정의 원문을 들고 있는데도
    그렇습니다.

    유보해야 할 것은 등급이 아니라 결론입니다. 결합의 동기·용이성·저해 요인은 여전히 평가하지
    않고, 그 사실은 rationale과 보고서 결론이 계속 말합니다(_combination_rationale). 어느 문헌이
    무엇을 댔는지도 combination_resolved에 그대로 남습니다.

    원본 matrix 셀은 감사용 단독 판정이므로 사본에만 기록합니다.
    """
    if chosen is None or not chosen.missing_limitations:
        return chosen
    # chosen 자체가 앞선 병합에서 이미 해소한 한정은 그대로 보존합니다. 또한 진 쪽 셀이
    # 합성 셀이라면 그 셀의 해소 출처도 이번 chosen의 누락을 메울 수 있습니다. 이를 빼면
    # 문헌 1+2가 해소한 한정이 문헌 3과 합칠 때 다시 missing으로 살아납니다.
    resolved: dict[str, str] = dict(chosen.combination_resolved)
    missing = set(chosen.missing_limitations)
    for cell in cells:
        if cell is None:
            continue
        # 결합 심사가 인정한 한정은 그 자리에서 해소된 것으로 봅니다. 이 인정은 채택 조합 전체의
        # 근거 위에서 내린 판단이고(entailment.validate_combination), 어느 문헌이 빠진 축을
        # 댔는지도 함께 확인된 상태입니다. **진 쪽 셀도 봅니다** — 기각은 특정 문헌의 판정에
        # 붙는 사실이라, 그 문헌이 병합에서 지면 인정까지 함께 버려집니다.
        for limitation, sources in combination_supported(cell).items():
            if limitation in missing and sources:
                resolved.setdefault(limitation, sources[0])
        for limitation, document_id in cell.combination_resolved.items():
            if limitation in missing:
                resolved.setdefault(limitation, document_id)
        if cell.document_id == chosen.document_id:
            continue
        if not is_eligible_supplement(cell):
            continue
        for limitation in sorted(filled_limitations(cell, chosen)):
            resolved.setdefault(limitation, cell.document_id)
    if not resolved:
        return chosen
    copy = chosen.model_copy(deep=True)
    copy.missing_limitations = [limitation for limitation in copy.missing_limitations
                                if limitation not in resolved]
    copy.combination_resolved = resolved
    # 채운 한정을 반영해 등급을 다시 유도하되, **동일급으로는 올리지 않습니다.**
    #
    # 끝까지 다시 유도하면 과합니다. '동일'·'실질적 동일'은 "이 인용발명이 이
    # 구성을 개시한다"는 **단일 문헌 진술**이라 결합 결과에 붙일 수 있는 말이 아닙니다. 그래서
    # 상한을 '일부 차이'로 둡니다. 그 아래에서 올라오는 것("관련 기재만 있음" → "대응은 하되
    # 차이가 남음")은 조합 안에 원문이 있다는 사실 진술이므로 안전하고, 정확히 그 구간에서
    # 거짓 공백이 발생합니다.
    for check in copy.limitation_checks:
        if check.limitation in resolved and not check.disclosed:
            check.disclosed = True
    derived = derive_judgment(
        copy.limitation_checks, has_evidence=bool(copy.quote or copy.evidence),
        terminology=copy.terminology, different_purpose=copy.different_purpose)
    ceiling = JUDGMENT_RANK["일부 차이"]
    if JUDGMENT_RANK.get(derived, 0) > JUDGMENT_RANK.get(copy.judgment, 0):
        copy.judgment = judgment_at_rank(min(JUDGMENT_RANK.get(derived, 0), ceiling))
    return copy


def _restore_combination_antecedents(
        claim: Claim, merged: dict[str, ElementMatch]) -> dict[str, ElementMatch]:
    """다른 채택 문헌이 선행 구성을 **완전히** 개시했을 때만 문헌 단독 상한을 풉니다.

    consistency.enforce_antecedents는 단일 문헌이 앞 구성을 놓친 상태에서 뒤 구성만 완전 개시로
    세는 것을 막습니다. 그러나 진보성 결합에서는 문헌 A가 앞 구성을, 문헌 B가 뒤 구성을
    나누어 개시할 수 있습니다. 결합 뒤에도 단독문헌 상한을 그대로 두면 정상적인 보완을
    선택하고도 등급과 차이점에는 "같은 문헌에 없음"이 남습니다.

    **fail-closed입니다.** 이 함수가 하는 일은 "두 문헌을 합치면 지시 대상이 성립한다"는 주장인데,
    그 주장이 참인지 확인할 교차문헌 검증 단계가 아직 없습니다. 확인할 수 없는 것은 풀지 않는
    쪽이 기본값이어야 하므로, 상한을 푸는 조건을 다음 두 가지로 좁힙니다.

      1. 선행 구성이 결합 안에서 **완전 개시**(is_complete)일 것. has_correspondence로는
         부족합니다 — 그것은 '일부 유사'·'일부 차이'도 통과시킵니다. 상한이 걸린 이유가 "지시
         대상이 이 문헌에 세워지지 않았다"인데, 부분적으로만 개시된 대상은 결합에서도 그
         대상을 세우지 못합니다. 그런 상태에서 상한을 풀면 결합 커버리지를 과대평가합니다.
      2. 복원 상한은 **선행 구성 중 가장 약한 판정을 넘지 못할 것.** 어떤 구성도 자기가
         참조하는 대상보다 더 완전하게 개시되었다고 볼 수 없습니다. enforce_antecedents가
         같은 문헌 안에서 쓰는 규칙(limit = min(선행 구성 판정))을 결합 범위로 그대로 옮긴
         것이라, 두 방향이 같은 기준 위에서 움직입니다.

    조건을 채우지 못하면 상한과 antecedent_note를 그대로 둡니다. 그러면 보고서는 "같은
    인용발명에서 선행 구성의 대응이 확인되지 않아 완전 개시로 보지 않았다"고 계속 적습니다.

    원본 matrix 셀은 감사용 단독 판정이므로 수정하지 않고, 결합 결과의 깊은 사본만 복원합니다.
    """
    restored = dict(merged)
    for label, source_labels in antecedents(claim).items():
        match = restored.get(label)
        if (match is None or not match.antecedent_note or not match.antecedent_capped_from
                or not source_labels
                # 조건 1. 부분 개시된 선행 구성으로는 지시 대상이 세워지지 않습니다.
                or not all(is_complete(restored.get(source)) for source in source_labels)):
            continue
        support_documents = sorted({restored[source].document_id for source in source_labels
                                    if restored.get(source) is not None
                                    and restored[source].document_id != match.document_id})
        if not support_documents:
            continue
        # 조건 2. 참조하는 대상보다 더 완전하게 복원하지 않습니다.
        limit = min(JUDGMENT_RANK.get(restored[source].judgment, 0) for source in source_labels)
        target = min(JUDGMENT_RANK.get(match.antecedent_capped_from, 0), limit)
        # 다른 문헌이 지시 대상만 세워 주더라도 이 셀 자신의 근거가 추론이라는 사실은 바뀌지
        # 않습니다. 추론 셀을 동일급으로 복원하면 유사도 배지와 차이점 줄이 서로 모순합니다.
        if match.directness != "direct":
            target = min(target, JUDGMENT_RANK["일부 차이"])
        if target <= JUDGMENT_RANK.get(match.judgment, 0):
            continue
        copy = match.model_copy(deep=True)
        copy.judgment = judgment_at_rank(target)
        if copy.downgraded_from == copy.antecedent_capped_from:
            copy.downgraded_from = ""
        copy.antecedent_note = ""
        copy.antecedent_capped_from = ""
        copy.antecedent_resolved_by = support_documents
        restored[label] = copy
    return restored


def merge_selected(claim: Claim, matrix: Matrix, document_ids: list[str]) -> dict[str, ElementMatch]:
    """채택 문헌들을 선정 단계와 보고서 단계에서 같은 규칙으로 병합합니다."""
    merged: dict[str, ElementMatch] = {}
    for document_id in document_ids:
        addition = matrix.get(document_id, {})
        merged = _merge(claim, merged, addition) if merged else dict(addition)
    return merged


# --- 종속항 결합 -------------------------------------------------------------

def _dependent_chain(claim: Claim, matrix: Matrix, parent: ChainInfo, chain: ChainInfo,
                     all_claims: list[Claim]) -> ChainInfo:
    """부모항 조합을 상속하고, 남은 공백을 모두 채우는 새 문헌 1개까지만 추가합니다."""
    # 부모항을 단독 개시한 문헌이 종속항의 추가 한정까지 직접 개시하면, 그 청구항은 문헌
    # 1건에 전부 개시된 것입니다. 결합을 논할 단계가 아닌데 진보성 트랙으로 넘기면
    # "제29조제2항 — 단일 인용발명"이라는 자기모순 라벨이 나옵니다.
    single = _single_document_novelty(claim, matrix, parent) if parent.track == "novelty_single" else None
    if single:
        chain.track = "novelty_single"
        chain.primary = single
        chain.inherited = [single]
        chain.novelty = NoveltyScreen(selected_document=single, complete_documents=[single],
                                      result="single_document_complete")
        chain.combined_similarity = combined_similarity(claim, matrix[single])
        chain.rationale = (f"부모 청구항 {parent.claim_number}을 단독 개시한 인용발명이 종속항의 추가 "
                           "한정까지 직접 개시하므로 문헌 결합을 검토하지 않습니다.")
        return _finalize(claim, chain, dict(matrix[single]), matrix)

    inherited = [document_id for document_id in [parent.primary, *parent.secondaries] if document_id]
    inherited += [document_id for document_id in parent.inherited if document_id not in inherited]
    if not inherited:
        # 부모항이 조합을 세우지 못했으면(_no_primary_chain) 상속할 것이 없습니다. 그렇다고
        # 빈 조합으로 진행하면 merged가 {}인 채 끝나, 이 항의 추가 한정을 실제로 개시한 문헌이
        # 있어도 "대응되는 인용발명이 확인되지 않음"으로 나갑니다 — 독립항에서 고친 것과 같은
        # 결손입니다. 이 항만으로 주 인용발명을 세우지는 않습니다. 종속항 행렬에는 "…에 있어서"
        # 뒤의 추가 한정만 들어 있어서, 그것 하나로 문헌을 채택하면 부모 구성을 개시하지 않은
        # 문헌이 이 항의 거절 근거로 서게 됩니다(build_chain의 신규성 게이트 주석과 같은 이유).
        return _no_primary_chain(claim, matrix, chain,
                                 f"부모 청구항 {parent.claim_number}의 인용발명 조합이 서지 않아 "
                                 "상속할 조합이 없습니다.")
    chain.inherited = inherited
    chain.primary = parent.primary
    chain.secondaries = [document_id for document_id in inherited if document_id != parent.primary]
    chain.track = parent.track if parent.track != "novelty_single" else "inventive_step_combination"

    # 상속한 문헌들 사이에서도 구성별로 가장 강한 대응을 채택합니다(부모항의 보완이 그대로 이어짐).
    merged: dict[str, ElementMatch] = {}
    for document_id in inherited:
        merged = _merge(claim, merged, matrix.get(document_id, {})) if merged else dict(matrix.get(document_id, {}))
    chain.supplement_needed = supplement_needed_labels(claim, merged)

    # 종속항도 독립항과 **같은 보완 탐색**을 돌립니다. 공백(uncovered)뿐 아니라 '보완 검토
    # 대상'(supplement_needed) 전체를 후보 대상으로 삼습니다.
    #
    # 대상을 공백으로만 한정하면, 상속한 문헌이 어떤 구성을 **약하게라도** 커버한 순간 그
    # 구성은 탐색에서 빠지고 같은 구성을 원문으로 직접 개시한 다른 문헌이 있어도 영원히
    # 채택되지 않습니다. 총론 문단 하나에 '일부 유사'로 붙은 대응이 '실질적 동일·direct·
    # 검증됨'을 밀어내는 상태입니다. README가 독립항에 대해 "보완 검토 대상 ≠ 미커버"라고
    # 정한 것과 같은 이유입니다.
    #
    # 다만 **새로 더하는 문헌은 1건까지**입니다(MAX_DEPENDENT_ADDITIONS). 종속항의 추가 한정은
    # 대개 한 줄인데 그것 하나를 위해 문헌을 여러 건 끌어오면 거절 이유가 실무에서 설득력을
    # 잃습니다. 공백을 메우는 후보와 근거 품질만 올리는 후보가 함께 있으면 _best_secondary가
    # GAP_PRIORITY 가중으로 공백 쪽을 먼저 고릅니다. 상한 때문에 빠진 문헌의 기재는 버리지
    # 않고 beyond_limit에 남깁니다.
    added: list[str] = []
    chain.combination_limit = MAX_DEPENDENT_ADDITIONS
    while len(added) < MAX_DEPENDENT_ADDITIONS and len(chain.secondaries) + 1 < len(matrix):
        gaps = no_correspondence_labels(claim, merged)
        targets = supplement_needed_labels(claim, merged)
        if not targets:
            break
        candidate = _best_secondary(claim, matrix, merged, targets, gaps,
                                    exclude={chain.primary, *chain.secondaries},
                                    scores=_fitness(chain))
        if candidate is None:
            break
        added.append(candidate)
        chain.secondaries.append(candidate)
        merged = _merge(claim, merged, matrix[candidate])
    chain.added = added
    # 종속항의 상한은 **새로 더한 문헌 수**에 걸리며, 실제로 다음 유효 후보가 남아 있을 때만
    # 조합을 막은 것입니다. 상속한 문헌 수나 단순한 상한 도달만으로 참이 되지 않습니다.
    chain.limit_binding = bool(
        len(added) >= MAX_DEPENDENT_ADDITIONS
        and _best_secondary(
            claim, matrix, merged, supplement_needed_labels(claim, merged),
            no_correspondence_labels(claim, merged),
            exclude={chain.primary, *chain.secondaries}, scores=_fitness(chain)))

    chain.uncovered = no_correspondence_labels(claim, merged)
    chain.residual = difference_labels(claim, merged)
    chain.reserved = reserved_labels(claim, merged)
    chain.combined_similarity = combined_similarity(claim, merged)
    parents = ancestry(all_claims, claim.number)
    inherited_text = f"청구항 {', '.join(str(number) for number in parents)}의 인용발명 조합을 상속" if parents else "부모항 조합을 상속"
    _apply_gap_policy(claim, chain, matrix, merged)
    if chain.track == "rejection_impossible":
        chain.rationale = f"{inherited_text}했으나 추가 한정에 대해 {chain.rationale}"
    elif chain.added:
        chain.rationale = (f"{inherited_text}하고, 추가 한정을 개시하는 인용발명 "
                           f"{len(chain.added)}건을 결합했습니다.{_well_known_clause(chain)}")
    else:
        chain.rationale = (f"{inherited_text}했으며 추가 문헌 없이 종속항 한정까지 "
                           f"커버됩니다.{_well_known_clause(chain)}")
    return _finalize(claim, chain, merged, matrix)


def _single_document_novelty(claim: Claim, matrix: Matrix, parent: ChainInfo) -> str | None:
    """부모항을 단독 개시한 문헌 중, 종속항의 추가 한정까지 직접 개시하는 문헌.

    부모항의 신규성 심사에서 완전 개시로 확인된 문헌만 후보로 봅니다. 종속항 행렬에는
    추가 한정만 들어 있어서, 이 제한이 없으면 부모 구성을 개시하지 않은 문헌이 뽑힙니다.
    """
    if not claim.elements:
        return None
    candidates = [document_id for document_id in parent.novelty.complete_documents
                  if all(_directly_disclosed(matrix.get(document_id, {}).get(element.label))
                         for element in claim.elements)]
    if not candidates:
        return None
    return parent.primary if parent.primary in candidates else min(candidates)


# --- 공통 ---------------------------------------------------------------------

def _finalize(claim: Claim, chain: ChainInfo, merged: dict[str, ElementMatch],
              matrix: Matrix) -> ChainInfo:
    rows = rows_for(claim, merged)
    # 전제부 공백은 트랙을 바꾸지 않지만 라벨과 보고서에는 반드시 나가야 합니다.
    # 미판정 상태에서는 "대응이 확인되지 않았다"고 적을 근거가 없으므로 비워 둡니다.
    if chain.track != "analysis_incomplete":
        chain.preamble_undisclosed = preamble_gaps(claim, no_correspondence_labels(claim, merged))
        # 결합을 세운 경로는 _apply_gap_policy가 이미 채웠습니다. 나머지 경로(단일문헌 신규성,
        # 주 인용발명 미성립)도 같은 사실을 남겨야 보고서가 한 곳에서만 유보를 적습니다.
        if not chain.combination_pending:
            gaps = no_correspondence_labels(claim, merged)
            pending = pending_labels(matrix, gaps)
            chain.combination_pending = [label for label in gaps if label in pending]
            chain.combination_pending_reasons = pending
    chain.candidates = sorted(chain.candidates, key=lambda score: (-score.main_score, score.document_id))
    for score in chain.candidates:
        score.detail = {**score.detail, "role": _role_of(score.document_id, chain)}
    chain.combined_similarity = chain.combined_similarity or round(
        sum(row["importance"] * row["similarity"] for row in rows) / (sum(row["importance"] for row in rows) or 1) * 100, 2)
    # 후보 행보다 **먼저** 채워야 합니다. 미채택 사유가 이 값에서 나옵니다.
    if chain.primary:
        chain.unadopted_gain = {
            document_id: gain for document_id, (gain, _) in _supplement_gains(
                claim, matrix, merged, supplement_needed_labels(claim, merged),
                no_correspondence_labels(claim, merged),
                exclude={chain.primary, *chain.secondaries}).items()}
    chain.element_coverage = _element_coverage(claim, chain, merged, matrix)
    return chain


def _element_coverage(claim: Claim, chain: ChainInfo, merged: dict[str, ElementMatch],
                      matrix: Matrix) -> list[ElementCoverage]:
    """구성별 전 문헌 대응. 채택되지 않은 문헌의 판정도 감사용으로 그대로 남깁니다."""
    coverages: list[ElementCoverage] = []
    for element in claim.elements:
        label = element.label
        primary = matrix.get(chain.primary or "", {}).get(label)
        adopted = merged.get(label)
        coverages.append(ElementCoverage(
            label=label,
            importance=element.importance,
            supplement_needed=bool(supplement_reason(primary)),
            supplement_reason=supplement_reason(primary),
            primary_document=chain.primary if primary else None,
            primary_judgment=primary.judgment if primary else "대응 없음",
            primary_missing=list(primary.missing_limitations) if primary else [],
            adopted_document=adopted.document_id if adopted else None,
            adopted_judgment=adopted.judgment if adopted else "대응 없음",
            adopted_role=_role_of(adopted.document_id, chain) if adopted else "미대응",
            residual_difference=residual_difference(adopted),
            candidates=[_candidate_row(document_id, matrix[document_id].get(label), primary,
                                       adopted, chain)
                        for document_id in sorted(matrix)],
        ))
    return coverages


def _candidate_row(document_id: str, match: ElementMatch | None, primary: ElementMatch | None,
                   adopted: ElementMatch | None, chain: ChainInfo) -> SupplementCandidate:
    reason = ineligible_reason(match)
    adopted_sources = ({adopted.document_id, *adopted.combination_resolved.values(),
                        *adopted.antecedent_resolved_by} if adopted else set())
    taken = document_id in adopted_sources
    merged_gain = _supplement_step(match, adopted)
    return SupplementCandidate(
        document_id=document_id,
        judgment=match.judgment if match else "대응 없음",
        directness=match.directness if match else "absent",
        verify=match.verify if match else "empty",
        has_quote=bool(match and match.quote),
        missing_count=len(match.missing_limitations) if match else 0,
        unverified_count=unverified_count(match),
        limitation_states=limitation_state_map(match),
        gain=_supplement_step(match, primary),
        merged_gain=merged_gain,
        better_than_primary=is_better_match(match, primary),
        eligible=not reason,
        rejected_reason=reason,
        excluded_reason="" if taken else _exclusion_reason(document_id, reason, merged_gain, chain),
        adopted=taken,
        sample_count=match.sample_count if match else 0,
        sample_agreement=match.sample_agreement if match else 0.0,
        sample_unanimous=match.sample_unanimous if match else 0,
        sample_requirements=match.sample_requirements if match else 0,
        sample_early_exit=bool(match and match.sample_early_exit),
    )


def _exclusion_reason(document_id: str, ineligible: str, merged_gain: float,
                      chain: ChainInfo) -> str:
    """채택되지 않은 후보가 **왜** 빠졌는지. 빈 문자열이면 사유 없이 사라진 것입니다.

    불변식 P2는 이 값이 비어 있을 때만 발화해야 합니다. 종전에는 주 인용발명 대비 이득만
    보고 발화했는데, 그 이득과 limit_binding은 **기준선이 다릅니다** — 앞은 주 인용발명
    단독, 뒤는 채택 조합 전체입니다. 그래서 채택된 보조 인용발명이 이미 같은 것을 대고 있는
    중복 후보마다 위반이 찍혔습니다. 동률 후보는 흔하므로(실측에서 두 문헌이 같은 구성에
    정확히 같은 이득을 냈습니다) 그 상태로는 경고가 늘 켜져 진짜 위반이 묻힙니다.

    사유는 좁은 것부터 봅니다. 자격 미달이면 그것이 이유이고, 자격은 있는데 조합이 이미
    같은 기여를 확보했다면 중복이며, 새로 보탤 것이 있는데도 문헌 단위 이득이 문턱에
    못 미치면 그것이 이유입니다. 셋 다 아닌데 상한이 실제로 걸렸다면 상한이 이유입니다.
    """
    if ineligible:
        return ineligible
    if merged_gain <= 0.0:
        return "채택 조합이 같은 기여를 이미 확보함(증분 0)"
    remaining = chain.unadopted_gain.get(document_id)
    if remaining is not None and remaining < MIN_SUPPLEMENT_GAIN:
        return f"문헌 단위 보완 이득 {remaining} < 문턱 {MIN_SUPPLEMENT_GAIN}"
    if chain.limit_binding:
        return f"결합 문헌 수 상한({chain.combination_limit}건)"
    return ""


def _role_of(document_id: str, chain: ChainInfo) -> str:
    if document_id == chain.primary:
        return "주 인용발명"
    if document_id in chain.added:
        return "추가 인용발명"
    if document_id in chain.secondaries:
        return "보조 인용발명"
    return "미채택"


def _importance(claim: Claim, label: str) -> int:
    for element in claim.elements:
        if element.label == label:
            return element.importance
    return 3


def _well_known_clause(chain: ChainInfo) -> str:
    """주지관용으로 다룬 구성이 있으면 그 사실과 실증 문헌 수를 덧붙입니다.

    조용히 메우면 보고서만 보고는 그 구성이 인용발명에 개시된 것인지 주지관용으로 넘어간
    것인지 구별할 수 없습니다. 주지관용 인정은 심사관의 판단이므로 반드시 드러냅니다.
    """
    if not chain.well_known:
        return ""
    counts = {len(chain.well_known_documents.get(label, [])) for label in chain.well_known}
    evidence = f"인용발명 {min(counts)}건 이상에 같은 취지의 기재가 있음" if counts else ""
    return (f" 구성 {', '.join(chain.well_known)}은 해당 기술분야의 주지관용기술로 보아 결합에 "
            f"더했습니다({evidence}). 주지관용 인정 여부는 별도로 확인해야 합니다.")


def _combination_rationale(chain: ChainInfo) -> str:
    if not chain.secondaries:
        if chain.supplement_needed or chain.residual:
            labels = chain.residual or chain.supplement_needed
            reserved = set(chain.reserved)
            differing = [label for label in labels if label not in reserved]
            sentences = []
            if differing:
                sentences.append(
                    "주 인용발명 단독으로 모든 구성에 대응 기재는 확인되지만, "
                    f"구성 {', '.join(differing)}은 완전 개시되지 않아 차이점 판단이 필요합니다. "
                    "다른 문헌에서도 이 차이를 완전히 해소하는 더 강한 직접 근거는 확인하지 못했습니다.")
            # 유보 구성에는 "차이점 판단이 필요하다"고 적지 않습니다. 판단할 차이가 있는지
            # 자체가 아직 확인되지 않았고, 필요한 것은 차이 판단이 아니라 원문 확인입니다.
            if reserved:
                sentences.append(
                    f"구성 {', '.join(chain.reserved)}은 의미검증을 수행하지 못해 개시 여부를 "
                    "확정하지 못했습니다.")
            return " ".join(sentences) + _well_known_clause(chain)
        return ("주 인용발명 단독으로 모든 필수 구성이 직접·완전하게 개시됩니다."
                + _well_known_clause(chain))
    base = ("주 인용발명이 완전히 개시하지 않은 구성을 보완 인용발명이 직접 개시하여 결합했습니다. "
            "이 결합은 구성 커버리지만으로 조립한 것이고, 결합의 동기·용이성·결합 방해 요소·"
            "작용효과는 평가하지 않았으므로 진보성 결론이 아닙니다.")
    # 유보 구성은 "차이가 남는다"에 넣지 않습니다. 차이는 대비해 본 결과이고, 유보는 대비
    # 자체를 못 한 것입니다. 한 문장에 뭉치면 결론 줄이 바로 아래 지표('판정 유보')와
    # 어긋납니다.
    differing = [label for label in chain.residual if label not in set(chain.reserved)]
    if differing:
        base += f" 결합 후에도 구성 {', '.join(differing)}에는 차이가 남습니다."
    if chain.reserved:
        base += (f" 구성 {', '.join(chain.reserved)}은 의미검증을 수행하지 못해 "
                 "개시 여부를 확정하지 못했습니다.")
    return base + _pending_clause(chain) + _well_known_clause(chain)


def _pending_clause(chain: ChainInfo) -> str:
    """축 결손으로 유보한 구성이 있으면 그 사실을 결론에 드러냅니다.

    조용히 넘기면 보고서만 보고는 그 구성이 개시된 것인지 판단이 유보된 것인지 알 수
    없습니다. 반대로 이것을 공백으로 적으면 원문 근거가 있는 기재를 없다고 단정하게 됩니다.
    둘 다 피하려면 유보를 유보라고 적는 수밖에 없습니다.
    """
    if not chain.combination_pending:
        return ""
    return (f" 구성 {', '.join(chain.combination_pending)}은 인용발명에 원문 근거가 있으나 "
            "의미검증이 문헌 단독으로는 한정의 일부 축을 확인하지 못했습니다. 그 축을 같은 "
            "조합의 다른 인용발명이 대는지는 이 도구가 판단하지 않았으므로 결론을 유보합니다 — "
            "미개시로 읽지 마십시오.")


def matrix_for(matches: list[ElementMatch]) -> Matrix:
    matrix: Matrix = {}
    for match in matches:
        matrix.setdefault(match.document_id, {})[match.label] = match
    return matrix


def chain_documents(chain: ChainInfo) -> list[str]:
    ordered = [chain.primary] if chain.primary else []
    ordered += [document_id for document_id in chain.secondaries if document_id and document_id not in ordered]
    return ordered
