"""인용발명 선정. 비교 매트릭스가 확정된 뒤로는 LLM을 한 번도 부르지 않습니다.

같은 매트릭스가 들어오면 항상 같은 조합이 나옵니다. 순위 결정에 LLM을 쓰면
재실행마다 결론이 흔들려 판정 캐시도 감사 기록도 의미가 없어집니다.
"""
from .claims import ancestry
from .consistency import antecedents
from .coverage import (
    JUDGMENT_RANK, PRIMARY_CANDIDATE_RATIO,
    best_match, combined_similarity, core_direct_score, core_elements, difference_labels,
    disclosed_limitations, has_correspondence, ineligible_reason, is_better_match,
    is_complete, is_eligible_supplement, judgment_at_rank, no_correspondence_labels,
    residual_difference, rows_for, score_document, supplement_gain, supplement_needed_labels,
    supplement_reason, well_known_labels,
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
# 독립항 거절 이유 하나에 세울 인용발명 수의 상한. 주 인용발명 1 + 보조 인용발명 1입니다.
# 여기에 주지관용기술을 더할 수 있으므로, 실무에서 성립하는 네 형태
# (인용발명 1 / 1 + 주지관용 / 2 / 2 + 주지관용)를 그대로 표현합니다.
#
# 상한을 두면 세 번째 문헌이 어느 구성의 유일한 검증 근거를 가지고 있을 때 그것이 버려집니다.
# 그 사실을 감추면 "어느 인용발명에도 대응이 없다"는 거짓 보고가 되므로, 버리는 대신
# beyond_limit에 그 구성과 문헌을 남깁니다. 상한은 **거절 이유를 몇 건으로 세울지**의 문제이지
# 문헌에 기재가 있느냐의 문제가 아니므로, 둘을 같은 칸에 적지 않습니다.
MAX_COMBINED_DOCUMENTS = 2
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
    """
    return bool(
        match
        and not match.error
        and match.quote
        and match.verify == "verified"
        and match.judgment in NOVELTY_DIRECT
        and match.directness == "direct"
        and not match.missing_limitations
    )


# --- 독립항 결합 -------------------------------------------------------------

def _independent_chain(claim: Claim, matrix: Matrix, chain: ChainInfo) -> ChainInfo:
    eligible = _eligible_primaries(claim, matrix, chain.candidates)
    if not eligible:
        return _no_primary_chain(claim, matrix, chain)

    # 자격 게이트를 통과한 후보 중 단독 적합도 1위를 주 인용발명으로 확정합니다.
    chain.primary = eligible[0]
    chain.track = "inventive_step_combination"
    chain.combination_limit = MAX_COMBINED_DOCUMENTS
    merged = dict(matrix[chain.primary])
    # 보완 검토 대상은 미커버 구성보다 넓습니다. '일부 차이'로 커버된 구성도 여기 들어옵니다.
    chain.supplement_needed = supplement_needed_labels(claim, merged)

    # 상한 안에서, 새로 기여하는 문헌이 있는 한 결합합니다. 이득이 문턱 아래로 떨어지면
    # 상한에 닿기 전에도 멈춥니다.
    while len(chain.secondaries) + 1 < min(len(matrix), MAX_COMBINED_DOCUMENTS):
        targets = supplement_needed_labels(claim, merged)
        if not targets:
            break
        candidate = _best_secondary(claim, matrix, merged, targets,
                                    no_correspondence_labels(claim, merged),
                                    exclude={chain.primary, *chain.secondaries})
        if candidate is None:
            break
        chain.secondaries.append(candidate)
        merged = _merge(claim, merged, matrix[candidate])

    chain.uncovered = no_correspondence_labels(claim, merged)
    # 대응은 있으나 하위 한정이나 구현 방식에 차이가 남는 구성입니다.
    chain.residual = difference_labels(claim, merged)
    chain.combined_similarity = combined_similarity(claim, merged)
    _apply_gap_policy(claim, chain, matrix, merged)
    return _finalize(claim, chain, merged, matrix)


def _no_primary_chain(claim: Claim, matrix: Matrix, chain: ChainInfo, cause: str = "") -> ChainInfo:
    """세울 조합이 없는 상태. 결론만 접고 구성대비는 그대로 보고합니다.

    이 게이트가 답하는 것은 **거절 이유를 세울 수 있는가**이지, 문헌에 대응 기재가 있는가가
    아닙니다. 종전에는 여기서 uncovered에 전 구성을 넣고 빈 조합(merged={})으로 마감했는데,
    그러면 두 가지가 한꺼번에 무너집니다. 하나는 원문 대조까지 통과한 '실질적 동일·direct'
    대응이 있는 구성까지 "어느 인용발명에서도 확인되지 않았다"고 단정하는 것이고, 다른 하나는
    보고서 본문에서 구성대비 결과가 통째로 사라지는 것입니다(report.build_claim_report는
    채택 문헌 목록으로 본문을 만듭니다). 실제로 세 문헌이 같은 구성을 '실질적 동일·direct·
    검증됨'으로 개시한 분석이 "구성 A, B은 제시된 인용발명 어디에서도 대응 기재가 확인되지
    않아 추가 검색이 필요합니다"로 나갔습니다.

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
    chain.combined_similarity = combined_similarity(claim, reference)
    corresponded = [element.label for element in claim.elements
                    if element.label not in chain.uncovered]
    chain.reference_only = sorted({match.document_id for label in corresponded
                                   if (match := reference.get(label)) is not None})
    reasons = [cause or "차별적 핵심 구성을 직접 개시한 인용발명이 없어 주 인용발명을 세우지 못했습니다."]
    if corresponded:
        reasons.append(f"구성 {', '.join(corresponded)}에는 대응 기재가 확인되었으나, "
                       "주 인용발명이 서지 않아 인용발명 조합을 확정하지 않았습니다.")
    blocking = blocking_labels(claim, chain.uncovered)
    if blocking:
        reasons.append(f"구성 {', '.join(blocking)}은 어느 인용발명에서도 대응 기재가 확인되지 않았습니다.")
    chain.rationale = " ".join(reasons)
    return _finalize(claim, chain, reference, matrix)


# --- 결합 한도 밖의 기재와 주지관용 ------------------------------------------

def _apply_gap_policy(claim: Claim, chain: ChainInfo, matrix: Matrix,
                      merged: dict[str, ElementMatch]) -> None:
    """남은 공백을 세 가지로 갈라 트랙과 결론 문장을 정합니다.

      - 결합 한도 밖 문헌에 검증된 대응 기재가 있는 구성 → beyond_limit
      - 주지관용기술로 다룰 수 있는 구성 → well_known (거절 이유는 그대로 성립)
      - 나머지 → 진짜 공백. 이때만 "어느 인용발명에서도 확인되지 않았다"고 적습니다.

    셋을 한 칸에 뭉치면 보고서가 사실과 다른 진술을 하게 됩니다. 상한 때문에 뺀 문헌의 기재를
    "없다"고 적으면 이미 손에 든 문헌을 다시 찾게 되고, 주지관용으로 충분한 범용 구성 하나
    때문에 거절 이유 전체가 "구성 곤란"으로 떨어지면 실제로 설 수 있는 거절이 사라집니다.
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

    blocking = blocking_labels(claim, chain.uncovered)
    well_known = well_known_labels(claim, matrix, blocking)
    chain.well_known = [label for label in blocking if label in well_known]
    chain.well_known_documents = well_known

    blocking = [label for label in blocking if label not in well_known]
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
                       f"결합 문헌 수 상한({chain.combination_limit}건)을 넘어 이 거절 이유에 "
                       "세우지 않았습니다.")
    chain.rationale = " ".join(reasons) + " 이대로는 거절 이유를 구성하기 어렵습니다."


def _residual_overflow(chain: ChainInfo, matrix: Matrix, merged: dict[str, ElementMatch],
                       adopted: set[str | None]) -> dict[str, dict[str, list[str]]]:
    """대응은 되었는데 **남은 차이**를 한도 밖 문헌이 메우는 경우를 한정 단위로 찾습니다.

    beyond_limit이 지키는 것은 "(X) 구성에 대응되는 인용발명이 확인되지 않음" 줄이고, 이쪽이
    지키는 것은 "→ 차이점: …" 줄입니다. 둘은 같은 사실을 서로 다른 자리에서 부정합니다 —
    구성 전체가 빠졌든 한정 하나가 빠졌든, 업로드된 문헌에 원문이 있는데 없다고 적으면 심사관은
    이미 손에 든 문헌을 다시 찾아 나서게 됩니다. 실측에서 전제부의 "길 안내 정보를 제공함"이
    그렇게 적혔습니다. 그 한정을 원문으로 개시한 내비게이션 특허가 업로드되어 있었지만 결합
    상한(2건) 밖이었고, 보고서는 그 사실을 적지 않은 채 남은 차이로만 적었습니다.

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

    두 가지가 종전 구현과 다릅니다.

    1. 임계를 절대 차가 아니라 **비율**로 잡습니다(PRIMARY_CANDIDATE_RATIO). core_direct는
       최고 문헌도 0.5 안팎이라, 0.20을 빼면 임계가 최고점의 60% 수준으로 내려앉았습니다.
    2. 예외를 **고유 기여**로 좁힙니다. 종전에는 핵심 구성을 하나라도 직접 개시하면 무조건
       되살렸는데, 핵심 구성 중에는 후보 대부분이 함께 개시하는 것이 있습니다. 실측 사건에서
       핵심 구성 (C)를 4문헌 중 3문헌이 '실질적 동일·direct·검증됨'으로 개시했고, 그래서 이
       예외가 임계와 무관하게 세 문헌을 전부 되살렸습니다. 모두가 가진 것을 가졌다는 사실은
       주 인용발명 자격의 근거가 되지 못합니다. 이미 통과한 후보들이 **직접 개시하지 못한**
       핵심 구성을 이 문헌이 개시할 때만 되살립니다.
    """
    core_labels = [element.label for element in core_elements(claim)]
    # 문헌 순위(score_document)와 **같은 정의**를 씁니다. 종전에는 여기만 중요도 가중이 아닌
    # 단순 평균이라, 같은 이름의 지표가 두 곳에서 다른 값이었고 마진도 다른 척도 위에서
    # 비교됐습니다.
    core_direct = {document_id: core_direct_score(claim, matches)
                   for document_id, matches in matrix.items()}
    if not core_direct:
        return []
    top = max(core_direct.values())
    if top <= 0.0:
        # 어느 문헌도 핵심 구성을 직접 개시하지 못한 상태. 비율 임계만 놓고 보면 0점 문헌이
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


def _best_secondary(claim: Claim, matrix: Matrix, merged: dict[str, ElementMatch],
                    targets: list[str], gaps: list[str], exclude: set[str]) -> str | None:
    """주 인용발명 대비 **증분**으로 평가합니다. 두 종류를 함께 봅니다.

      - 완전 미대응 구성: 검증된 대응 기재를 처음 제공하면 이득으로 셉니다(GAP_PRIORITY 가중).
      - 부분 대응 구성: 판정·직접성·근거·누락 한정 중 하나 이상이 실제로 개선될 때만 셉니다.

    절대 성능이 높아도 어느 쪽에도 기여하지 못하면 채택하지 않습니다.

    공백에 유사도 하한을 걸지 않는 이유: 그 구성의 유일한 대응 기재를 가진 문헌이라도
    판정이 '일부 차이'면 하한에 걸려 탈락하고, 결과적으로 "어느 문헌에도 대응이 없다"는
    보고서가 나옵니다. 결합해서 남는 차이는 버리지 않고 보고서의 차이점으로 남깁니다.
    """
    gap_labels = set(gaps)
    best, best_gain = None, 0.0
    for document_id in sorted(matrix):                 # 동률일 때 항상 같은 문헌이 뽑히도록 고정
        if document_id in exclude:
            continue
        matches = matrix[document_id]
        gain, useful = 0.0, 0
        for label in targets:
            candidate, current = matches.get(label), merged.get(label)
            if not is_eligible_supplement(candidate) or not is_better_match(candidate, current):
                continue
            if label in gap_labels:
                # 공백에는 '검증된 대응 기재가 실제로 생겼는지'만 요구합니다.
                if not has_correspondence(best_match([current, candidate])):
                    continue
                weight = GAP_PRIORITY
            else:
                weight = 1.0
            gain += weight * supplement_gain(candidate, current) * _importance(claim, label)
            useful += 1
        if useful and gain > best_gain:
            best, best_gain = document_id, gain
    return best if best_gain >= MIN_SUPPLEMENT_GAIN else None


def _merge(claim: Claim, current: dict[str, ElementMatch], addition: dict[str, ElementMatch]) -> dict[str, ElementMatch]:
    """구성별로 더 강한 판정을 채택합니다. 결합 후 커버리지는 여기서만 정해집니다."""
    merged = {element.label: best_match([current.get(element.label), addition.get(element.label)])
              for element in claim.elements}
    return _restore_combination_antecedents(claim, merged)


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

      1. 선행 구성이 결합 안에서 **완전 개시**(is_complete)일 것. 종전에는 has_correspondence만
         요구했는데, 그것은 '일부 유사'·'일부 차이'도 통과시킵니다. 상한이 걸린 이유가 "지시
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
    # 종전에는 대상을 공백으로만 한정했습니다("새 문헌을 끌어오는 이유는 공백뿐"). 그러면
    # 상속한 문헌이 어떤 구성을 **약하게라도** 커버한 순간 그 구성은 탐색에서 빠지고, 같은
    # 구성을 원문으로 직접 개시한 다른 문헌이 있어도 영원히 채택되지 않습니다. 실제로 한
    # 보고서에서 "고유 식별값에 매칭된 영상을 호출하는 제어부"가, 그런 기재가 전혀 없는
    # 문헌의 "컴퓨팅 장치를 사용할 수 있다"는 총론 문단에 '일부 유사'로 붙었습니다. 그
    # 구성을 실제로 개시한 문헌은 '실질적 동일·direct·검증됨'이었는데도 미채택으로 남았습니다.
    # README가 독립항에 대해 "보완 검토 대상 ≠ 미커버"라고 정한 것과 같은 이유입니다.
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
                                    exclude={chain.primary, *chain.secondaries})
        if candidate is None:
            break
        added.append(candidate)
        chain.secondaries.append(candidate)
        merged = _merge(claim, merged, matrix[candidate])
    chain.added = added

    chain.uncovered = no_correspondence_labels(claim, merged)
    chain.residual = difference_labels(claim, merged)
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
    chain.candidates = sorted(chain.candidates, key=lambda score: (-score.main_score, score.document_id))
    for score in chain.candidates:
        score.detail = {**score.detail, "role": _role_of(score.document_id, chain)}
    chain.combined_similarity = chain.combined_similarity or round(
        sum(row["importance"] * row["similarity"] for row in rows) / (sum(row["importance"] for row in rows) or 1) * 100, 2)
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
            candidates=[_candidate_row(document_id, matrix[document_id].get(label), primary, adopted)
                        for document_id in sorted(matrix)],
        ))
    return coverages


def _candidate_row(document_id: str, match: ElementMatch | None, primary: ElementMatch | None,
                   adopted: ElementMatch | None) -> SupplementCandidate:
    reason = ineligible_reason(match)
    return SupplementCandidate(
        document_id=document_id,
        judgment=match.judgment if match else "대응 없음",
        directness=match.directness if match else "absent",
        verify=match.verify if match else "empty",
        has_quote=bool(match and match.quote),
        missing_count=len(match.missing_limitations) if match else 0,
        gain=supplement_gain(match, primary),
        better_than_primary=is_better_match(match, primary),
        eligible=not reason,
        rejected_reason=reason,
        adopted=bool(adopted and adopted.document_id == document_id),
        sample_count=match.sample_count if match else 0,
        sample_agreement=match.sample_agreement if match else 0.0,
        sample_early_exit=bool(match and match.sample_early_exit),
    )


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
            return ("주 인용발명 단독으로 모든 구성에 대응 기재는 확인되지만, "
                    f"구성 {', '.join(labels)}은 완전 개시되지 않아 차이점 판단이 필요합니다. "
                    "다른 문헌에서도 이 차이를 완전히 해소하는 더 강한 직접 근거는 확인하지 못했습니다."
                    + _well_known_clause(chain))
        return ("주 인용발명 단독으로 모든 필수 구성이 직접·완전하게 개시됩니다."
                + _well_known_clause(chain))
    base = ("주 인용발명이 완전히 개시하지 않은 구성을 보완 인용발명이 직접 개시하여 결합했습니다. "
            "이 결합은 구성 커버리지만으로 조립한 것이고, 결합의 동기·용이성·결합 방해 요소·"
            "작용효과는 평가하지 않았으므로 진보성 결론이 아닙니다.")
    if chain.residual:
        base += f" 결합 후에도 구성 {', '.join(chain.residual)}에는 차이가 남습니다."
    return base + _well_known_clause(chain)


def matrix_for(matches: list[ElementMatch]) -> Matrix:
    matrix: Matrix = {}
    for match in matches:
        matrix.setdefault(match.document_id, {})[match.label] = match
    return matrix


def chain_documents(chain: ChainInfo) -> list[str]:
    ordered = [chain.primary] if chain.primary else []
    ordered += [document_id for document_id in chain.secondaries if document_id and document_id not in ordered]
    return ordered
