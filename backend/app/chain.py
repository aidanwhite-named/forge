"""인용발명 선정. 비교 매트릭스가 확정된 뒤로는 LLM을 한 번도 부르지 않습니다.

같은 매트릭스가 들어오면 항상 같은 조합이 나옵니다. 순위 결정에 LLM을 쓰면
재실행마다 결론이 흔들려 판정 캐시도 감사 기록도 의미가 없어집니다.
"""
from .claims import ancestry
from .coverage import (
    PRIMARY_CANDIDATE_MARGIN,
    best_match, combined_similarity, difference_labels, direct_similarity, has_correspondence,
    ineligible_reason, is_better_match, is_core, is_eligible_supplement,
    no_correspondence_labels, quality_key, residual_difference, rows_for, score_document,
    supplement_gain, supplement_needed_labels, supplement_reason,
)
from .models import (Claim, ChainInfo, ConventionalNote, DocumentScore, DroppedSupplement,
                     ElementCoverage, ElementMatch, NoveltyScreen, SupplementCandidate)

# 이 판정들만 단일문헌 신규성의 직접 개시로 인정합니다.
NOVELTY_DIRECT = {"동일", "실질적 동일"}
# 중요도가 이 이하인 미커버 구성은 주지관용 검토로 분리합니다.
# ClaimElement.importance의 기본값이 3이므로(중요도 판정 실패 시에도 3), 이 값을 3으로 올리면
# 중요도를 받지 못한 구성이 전부 주지관용으로 분류됩니다. 여기는 2로 고정합니다.
#
# 이 분류는 중요도 하나만 보고 내리는 **행정적 분리**이지 주지관용 인정이 아닙니다.
# 주지관용 기술은 근거 제시가 원칙이고 이 파이프라인은 그 근거를 찾지 않으므로,
# 분리된 구성마다 무엇이 입증되지 않았는지를 conventional_notes에 남겨 보고서로 내보냅니다.
CONVENTIONAL_IMPORTANCE = 2
# 세 번째 문헌까지 허용할 수 있는 잔여 공백의 중요도 상한. 주지관용 분류와는 별개 기준입니다.
# 차별적 핵심 구성(4~5)을 세 번째 문헌으로 메우는 것은 계속 막습니다.
EXCEPTIONAL_GAP_IMPORTANCE = 3
MAX_COMBINED_DOCUMENTS = 2        # 주 인용발명 1 + 보완 1
MAX_EXCEPTIONAL_DOCUMENTS = 3     # 잔여 공백에 명시적·검증된 직접 근거가 있을 때만
# 완전 미대응을 메우는 이득에 주는 우선순위. 남은 결합 여유를 품질 개선에 먼저 쓰지 않게 합니다.
GAP_PRIORITY = 3.0


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
    대신 preamble_undisclosed에 남겨 검토 트랙 라벨에 함께 표시합니다.
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
        chain.track = "rejection_impossible"
        chain.uncovered = [element.label for element in claim.elements]
        chain.rationale = "주 인용발명 자격을 갖춘 문헌이 없어 거절 이유를 구성하기 어렵습니다."
        return _finalize(claim, chain, {}, matrix)

    # 자격 게이트를 통과한 후보 중 단독 적합도 1위를 주 인용발명으로 확정합니다.
    chain.primary = eligible[0]
    chain.track = "inventive_step_combination"
    merged = dict(matrix[chain.primary])
    # 보완 검토 대상은 미커버 구성보다 넓습니다. '일부 차이'로 커버된 구성도 여기 들어옵니다.
    chain.supplement_needed = supplement_needed_labels(claim, merged)

    limit = MAX_COMBINED_DOCUMENTS
    blocked_by_limit = False
    while len(chain.secondaries) + 1 < MAX_EXCEPTIONAL_DOCUMENTS + 1:
        gaps = no_correspondence_labels(claim, merged)
        targets = supplement_needed_labels(claim, merged)
        if not targets:
            break
        candidate = _best_secondary(claim, matrix, merged, targets, gaps,
                                    exclude={chain.primary, *chain.secondaries})
        if candidate is None:
            break
        if len(chain.secondaries) + 1 >= limit:
            # 예외적 3문헌 결합: 남은 **공백**이 차별적 핵심 구성이 아니고, 세 번째 문헌이
            # 그 공백 전부를 검증된 직접 근거로 개시할 때만. 근거 품질 개선만을 이유로는 안 됩니다.
            remaining_core = [label for label in gaps if _importance(claim, label) > EXCEPTIONAL_GAP_IMPORTANCE]
            if not gaps or remaining_core or not _has_explicit_support(matrix[candidate], gaps):
                blocked_by_limit = True
                break
        chain.secondaries.append(candidate)
        merged = _merge(claim, merged, matrix[candidate])

    gaps = no_correspondence_labels(claim, merged)
    chain.conventional = [label for label in gaps if _importance(claim, label) <= CONVENTIONAL_IMPORTANCE]
    # 중요도가 낮아도 주지관용 근거가 없으면 여전히 미개시 구성이다.
    # 별도 검토 표시(conventional)는 남기되 uncovered에서 제거하지 않는다.
    chain.uncovered = list(gaps)
    # 대응은 있으나 하위 한정이나 구현 방식에 차이가 남는 구성입니다.
    chain.residual = difference_labels(claim, merged)
    chain.combined_similarity = combined_similarity(claim, merged)
    blocking = blocking_labels(claim, chain.uncovered)
    if blocking:
        chain.track = "rejection_impossible"
        chain.rationale = (f"결합 후에도 구성 {', '.join(blocking)}의 청구항 한정 전체를 충족하는 기재가 "
                           "어느 인용발명에서도 확인되지 않아 거절 이유를 구성하기 어렵습니다.")
    else:
        chain.rationale = _combination_rationale(chain)
    return _finalize(claim, chain, merged, matrix, blocked_by_limit)


def _eligible_primaries(claim: Claim, matrix: Matrix, scores: list[DocumentScore]) -> list[str]:
    """차별적 핵심 구성의 직접 개시량이 최고 문헌에 크게 못 미치면 후보에서 제외합니다.

    다만 전체 점수가 낮아도 핵심 구성을 원문으로 직접 개시한 문헌은 남깁니다.
    평균 점수에 희석되어 유효한 후보가 탈락하지 않게 하기 위한 예외입니다.
    """
    core_labels = [element.label for element in claim.elements if is_core(element)] \
        or [element.label for element in claim.elements]
    core_direct = {
        document_id: sum(direct_similarity(matches.get(label)) for label in core_labels) / (len(core_labels) or 1)
        for document_id, matches in matrix.items()
    }
    if not core_direct:
        return []
    top = max(core_direct.values())
    if top <= 0.0:
        # 어느 문헌도 핵심 구성을 직접 개시하지 못한 상태. 마진(0.20)만 놓고 보면 0점 문헌이
        # 전부 자격을 얻어 무관한 문헌이 "주 인용발명"으로 보고서에 찍힙니다.
        return []
    eligible = [score.document_id for score in scores
                if core_direct.get(score.document_id, 0.0) >= top - PRIMARY_CANDIDATE_MARGIN]
    for score in scores:
        if score.document_id in eligible:
            continue
        matches = matrix[score.document_id]
        if any(_directly_disclosed(matches.get(label)) for label in core_labels):
            eligible.append(score.document_id)
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
    return best


def _has_explicit_support(matches: dict[str, ElementMatch], gaps: list[str]) -> bool:
    """예외적 3문헌 결합의 문턱. 신규성 게이트와 같은 이유로 "verified"만 받습니다."""
    return all(
        (matches.get(label) is not None
         and matches[label].quote
         and matches[label].verify == "verified"
         and matches[label].directness == "direct")
        for label in gaps
    )


def _merge(claim: Claim, current: dict[str, ElementMatch], addition: dict[str, ElementMatch]) -> dict[str, ElementMatch]:
    """구성별로 더 강한 판정을 채택합니다. 결합 후 커버리지는 여기서만 정해집니다."""
    return {element.label: best_match([current.get(element.label), addition.get(element.label)])
            for element in claim.elements}


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
    chain.inherited = inherited
    chain.primary = parent.primary
    chain.secondaries = [document_id for document_id in inherited if document_id != parent.primary]
    chain.track = parent.track if parent.track != "novelty_single" else "inventive_step_combination"

    # 상속한 문헌들 사이에서도 구성별로 가장 강한 대응을 채택합니다(부모항의 보완이 그대로 이어짐).
    merged: dict[str, ElementMatch] = {}
    for document_id in inherited:
        merged = _merge(claim, merged, matrix.get(document_id, {})) if merged else dict(matrix.get(document_id, {}))
    chain.supplement_needed = supplement_needed_labels(claim, merged)
    gaps = no_correspondence_labels(claim, merged)
    if gaps:
        # 종속항에서 새 문헌을 끌어오는 이유는 '공백'뿐입니다. 근거 품질 개선은 참고로만 남깁니다.
        candidate = _best_secondary(claim, matrix, merged, gaps, gaps, exclude=set(inherited))
        # 종속항 하나를 거절하기 위해 새 문헌을 2개 이상 추가하지 않습니다.
        if candidate and _fills_all(claim, matrix[candidate], merged, gaps):
            chain.added = candidate
            chain.secondaries.append(candidate)
            merged = _merge(claim, merged, matrix[candidate])
            gaps = no_correspondence_labels(claim, merged)

    chain.conventional = [label for label in gaps if _importance(claim, label) <= CONVENTIONAL_IMPORTANCE]
    chain.uncovered = list(gaps)
    chain.residual = difference_labels(claim, merged)
    chain.combined_similarity = combined_similarity(claim, merged)
    parents = ancestry(all_claims, claim.number)
    inherited_text = f"청구항 {', '.join(str(number) for number in parents)}의 인용발명 조합을 상속" if parents else "부모항 조합을 상속"
    blocking = blocking_labels(claim, chain.uncovered)
    if blocking:
        chain.track = "rejection_impossible"
        chain.rationale = (f"{inherited_text}했으나 추가 한정 {', '.join(blocking)}에 대응하는 기재를 "
                           "어느 인용발명에서도 확인하지 못해 거절 이유를 구성하기 어렵습니다.")
    elif chain.added:
        chain.rationale = f"{inherited_text}하고, 추가 한정을 개시하는 인용발명 1건을 결합했습니다."
    else:
        chain.rationale = f"{inherited_text}했으며 추가 문헌 없이 종속항 한정까지 커버됩니다."
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


def _fills_all(claim: Claim, candidate: dict[str, ElementMatch], merged: dict[str, ElementMatch],
               gaps: list[str]) -> bool:
    """공백을 전부 메우는 문헌만 종속항에 추가합니다. 일부만 메우면 결합 한도를 넘게 됩니다."""
    return all(has_correspondence(best_match([merged.get(label), candidate.get(label)]))
               for label in gaps if _importance(claim, label) > CONVENTIONAL_IMPORTANCE)


# --- 공통 ---------------------------------------------------------------------

def _finalize(claim: Claim, chain: ChainInfo, merged: dict[str, ElementMatch], matrix: Matrix,
              blocked_by_limit: bool = False) -> ChainInfo:
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
    chain.reference_only = [coverage.label for coverage in chain.element_coverage if coverage.reference_document]
    chain.dropped_supplements = _dropped_supplements(chain, blocked_by_limit)
    chain.conventional_notes = _conventional_notes(claim, chain, merged)
    return chain


def _conventional_notes(claim: Claim, chain: ChainInfo, merged: dict[str, ElementMatch]) -> list[ConventionalNote]:
    """주지관용으로 분리한 구성마다 근거 상태를 남깁니다.

    분리 기준은 중요도뿐이므로 여기서 인정되는 것은 아무것도 없습니다. 부분 대응이라도
    검증된 근거가 있는지, 무엇이 아직 개시되지 않았는지를 적어 심사관이 직접 판단하게 합니다.
    """
    notes: list[ConventionalNote] = []
    for label in chain.conventional:
        match = merged.get(label)
        supported = is_eligible_supplement(match)
        missing = list(match.missing_limitations) if match else []
        gap = ", ".join(missing) or "청구항 구성 전체"
        notes.append(ConventionalNote(
            label=label,
            importance=_importance(claim, label),
            judgment=match.judgment if match else "대응 없음",
            partial_support=supported,
            missing=missing,
            note=(f"인용발명에 부분 대응 기재는 있으나 {gap}에 대해서는 주지관용 근거가 제시되지 않았습니다."
                  if supported else
                  "대응 기재가 확인되지 않았고 주지관용 근거도 제시되지 않았습니다."),
        ))
    return notes


def _element_coverage(claim: Claim, chain: ChainInfo, merged: dict[str, ElementMatch],
                      matrix: Matrix) -> list[ElementCoverage]:
    """구성별 전 문헌 대응. **문헌 수 제한을 적용하지 않고** 모든 문헌을 그대로 평가합니다.

    결합 한도 때문에 채택하지 못한 대응도 사유와 함께 남겨, 정보 자체가 사라지지 않게 합니다.
    """
    selected = set(chain_documents(chain))
    coverages: list[ElementCoverage] = []
    for element in claim.elements:
        label = element.label
        primary = matrix.get(chain.primary or "", {}).get(label)
        adopted = merged.get(label)
        candidates = [_candidate_row(document_id, matrix[document_id].get(label), primary, adopted)
                      for document_id in sorted(matrix)]
        reference = _reference_document(matrix, label, adopted, selected)
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
            reference_document=reference,
            reference_judgment=(matrix[reference][label].judgment if reference else "대응 없음"),
            candidates=candidates,
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
    )


def _reference_document(matrix: Matrix, label: str, adopted: ElementMatch | None,
                        selected: set[str]) -> str | None:
    """채택되지 않은 문헌 중 이 구성을 명확히 더 잘 개시하는 문헌. 결합 문헌으로 쓰지는 않습니다."""
    best: str | None = None
    for document_id in sorted(matrix):
        if document_id in selected:
            continue
        match = matrix[document_id].get(label)
        if not is_eligible_supplement(match) or not is_better_match(match, adopted):
            continue
        if best is None or quality_key(match) > quality_key(matrix[best][label]):
            best = document_id
    return best


def _dropped_supplements(chain: ChainInfo, blocked_by_limit: bool = False) -> list[DroppedSupplement]:
    """채택하지 않은 보완을 문헌 단위로 묶습니다. 실제 결합 문헌으로 표기하지 않습니다.

    탈락 사유를 실제로 작동한 규칙으로 적습니다. 문헌 수 한도에 걸린 적이 없는데도
    "결합 문헌 수 제한"이라고 적으면, 보고서를 읽는 사람이 한도만 늘리면 채택된다고
    잘못 판단하게 됩니다.
    """
    grouped: dict[str, list[str]] = {}
    for coverage in chain.element_coverage:
        if coverage.reference_document:
            grouped.setdefault(coverage.reference_document, []).append(coverage.label)
    if chain.track == "analysis_incomplete":
        cause = "구성대비가 완료되지 않아 결합 여부를 판단하지 않았습니다."
    elif chain.track == "novelty_single":
        cause = "단일 인용발명으로 신규성을 판단해 다른 문헌을 결합하지 않았습니다."
    elif blocked_by_limit:
        cause = "결합 문헌 수 제한으로 인용발명 조합에는 넣지 않았습니다."
    else:
        cause = ("주 인용발명 대비 보완 이득이 다른 후보 문헌보다 작아 결합 문헌으로 선정되지 "
                 "않았습니다.")
    return [DroppedSupplement(document_id=document_id, labels=labels,
                              reason=f"{cause} 해당 구성의 대응 근거로만 참고합니다.")
            for document_id, labels in sorted(grouped.items())]


def _role_of(document_id: str, chain: ChainInfo) -> str:
    if document_id == chain.primary:
        return "주 인용발명"
    if document_id == chain.added:
        return "추가 인용발명"
    if document_id in chain.secondaries:
        return "보조 인용발명"
    return "미채택"


def _importance(claim: Claim, label: str) -> int:
    for element in claim.elements:
        if element.label == label:
            return element.importance
    return 3


def _combination_rationale(chain: ChainInfo) -> str:
    if not chain.secondaries:
        if chain.supplement_needed or chain.residual:
            labels = chain.residual or chain.supplement_needed
            return ("주 인용발명 단독으로 모든 구성에 대응 기재는 확인되지만, "
                    f"구성 {', '.join(labels)}은 완전 개시되지 않아 차이점 판단이 필요합니다. "
                    "다른 문헌에서도 이 차이를 완전히 해소하는 더 강한 직접 근거는 확인하지 못했습니다.")
        return "주 인용발명 단독으로 모든 필수 구성이 직접·완전하게 개시됩니다."
    base = ("주 인용발명이 완전히 개시하지 않은 구성을 보완 인용발명이 직접 개시하여 결합했습니다. "
            "이 결합은 구성 커버리지만으로 조립한 것이고, 결합의 동기·용이성·결합 방해 요소·"
            "작용효과는 평가하지 않았으므로 진보성 결론이 아닙니다.")
    if chain.conventional:
        base += (f" 구성 {', '.join(chain.conventional)}은 중요도가 낮아 주지관용 검토 대상으로 분리했을 뿐,"
                 " 주지관용임을 뒷받침하는 근거는 확인하지 않았습니다.")
    if chain.residual:
        base += f" 결합 후에도 구성 {', '.join(chain.residual)}에는 차이가 남습니다."
    return base


# 이 파이프라인이 실제로 확인하지 않은 것들. 조문을 인용하는 라벨에는 반드시 함께 나갑니다.
# 조문만 적고 이 단서를 빼면, 기술적 구성대비 결과가 법적 결론으로 읽힙니다.
NOVELTY_CAVEAT = "선행기술 적격성(공개일 vs 대상 청구항 우선일) 미확인"
INVENTIVE_CAVEAT = "결합 동기·용이성 미평가 · " + NOVELTY_CAVEAT


def rejection_basis(chain: ChainInfo, claim: Claim) -> str:
    """확정된 인용발명 조합과 커버리지로 **검토 트랙** 라벨을 조립합니다.

    이 파이프라인은 기술적 구성대비만 수행합니다. 선행기술 적격성(공개일과 대상 청구항
    우선일의 선후)도, 진보성의 결합 동기·용이성도 평가하지 않으므로, 조문 번호를 확정형
    거절 이유로 적지 않고 검토 후보와 미평가 항목을 함께 표시합니다.
    """
    if chain.track == "analysis_incomplete":
        return "판정 불가 — 구성대비 미완료 (분석 실패 셀 존재)"
    # 전제부 대응이 확인되지 않았다는 사실은 트랙과 무관하게 라벨에 붙습니다. 전제부를
    # 한정으로 볼지는 심사관의 판단이므로, 코드가 대신 정하지 않고 쟁점만 드러냅니다.
    preamble = (" · 전제부 대응 미확인(한정 여부 검토 필요)" if chain.preamble_undisclosed else "")
    if chain.track == "novelty_single":
        return (f"신규성 검토 후보 (제29조제1항제2호) — 단일 인용발명이 전 구성 개시"
                f"{preamble} · {NOVELTY_CAVEAT}")
    if chain.track == "rejection_impossible":
        return f"거절 이유 구성 곤란 — 대응 기재 없는 구성 잔존{preamble}"
    count = 1 + len(chain.secondaries)
    suffix = f" — 인용발명 {count}건 결합" if count > 1 else " — 단일 인용발명"
    if chain.conventional:
        # 주지관용 인정이 아니라 별도 입증이 남았다는 표시입니다. 라벨에서 이를 감추지 않습니다.
        suffix += " + 주지관용 기술(근거 미제시)"
    return f"진보성 검토 후보 (제29조제2항){suffix}{preamble} · {INVENTIVE_CAVEAT}"


def matrix_for(matches: list[ElementMatch]) -> Matrix:
    matrix: Matrix = {}
    for match in matches:
        matrix.setdefault(match.document_id, {})[match.label] = match
    return matrix


def chain_documents(chain: ChainInfo) -> list[str]:
    ordered = [chain.primary] if chain.primary else []
    ordered += [document_id for document_id in chain.secondaries if document_id and document_id not in ordered]
    return ordered
