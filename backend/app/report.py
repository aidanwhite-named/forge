"""보고서 조립. 문장은 확정된 판정 데이터에서 템플릿으로 만들고 LLM에 다시 묻지 않습니다.

LLM이 쓴 마크다운을 정규식으로 되돌려 고치는 코드가 필요 없어지는 대신,
표현은 템플릿이 허용하는 범위로 제한됩니다. 판정 근거의 재현성을 우선한 선택입니다.

구성 하나는 "정량 지표 한 줄 + 구성대비 한 문장 + (있으면) 차이점 한 줄"로 나갑니다.
같은 내용을 서술·역할·잔여차이로 나눠 세 번 반복하던 종전 구조는 읽는 사람이 매번
같은 문장을 다시 읽게 만들 뿐, 새로 알려 주는 것이 없어 걷어냈습니다.
"""
import re

from .chain import chain_documents, merge_selected
from .consistency import antecedents
from .coverage import (FULL_JUDGMENTS, JUDGMENT_RANK, best_match, derive_judgment,
                       evidence_locations, evidenced_limitations, is_reserved, is_unverified_check,
                       limitation_counts, report_grade, residual_difference, unverified_count,
                       unverified_limitations)
from .models import (AnalysisResult, ChainInfo, Claim, ClaimReport, ClaimResult, Document,
                     DocumentMapping, ElementMatch, Evidence, LimitationCheck, SampleTally,
                     SemanticEvent, TrailStep, VerificationTrail)

# 발췌 길이 상한. 요구 형식이 "최대 3줄, 가능하면 1줄"이므로 한 줄 분량으로 자릅니다.
EXCERPT_LIMIT = 150
ORIGINAL_LIMIT = 150
_STATUS = {"동일": "개시됨", "실질적 동일": "개시됨", "일부 차이": "부분 개시", "일부 유사": "부분 개시"}
_QUALITY = {"verified": "HIGH", "partial": "MEDIUM", "not_found": "UNVERIFIED",
            "short": "LOW", "empty": "UNVERIFIED"}


def build_mappings(documents: list[Document], chains: list[ChainInfo]) -> list[DocumentMapping]:
    """인용발명 번호는 채택 역할 순서로 매깁니다. 업로드 순서와 무관합니다."""
    scores: dict[str, float] = {}
    for chain in chains:
        for candidate in chain.candidates:
            scores[candidate.document_id] = max(scores.get(candidate.document_id, 0.0), candidate.main_score)
    ordered: list[str] = []
    for chain in chains:
        for document_id in chain_documents(chain):
            if document_id not in ordered:
                ordered.append(document_id)
    rest = sorted((document.id for document in documents if document.id not in ordered),
                  key=lambda document_id: (-scores.get(document_id, 0.0), document_id))
    by_id = {document.id: document for document in documents}
    mappings: list[DocumentMapping] = []
    for number, document_id in enumerate(ordered + rest, 1):
        document = by_id.get(document_id)
        if document is None:
            continue
        mappings.append(DocumentMapping(
            reference_number=number, filename=document.filename, document_type=document.type,
            document_id=document_id, document_number=document.document_number,
            publication_date=document.publication_date, filing_date=document.filing_date,
            source_file=document.source_file,
            role=_role(document_id, chains), main_score=round(scores.get(document_id, 0.0), 2),
        ))
    return mappings


def refresh_mappings(mappings: list[DocumentMapping], chains: list[ChainInfo]) -> list[DocumentMapping]:
    """후속 종속항을 추가한 뒤에도 기존 인용발명 번호는 유지하면서 역할과 최고 점수를 갱신합니다."""
    scores: dict[str, float] = {}
    for chain in chains:
        for candidate in chain.candidates:
            scores[candidate.document_id] = max(scores.get(candidate.document_id, 0.0), candidate.main_score)
    return [mapping.model_copy(update={
        "role": _role(mapping.document_id, chains),
        "main_score": round(scores.get(mapping.document_id, mapping.main_score), 2),
    }) for mapping in mappings]


def _role(document_id: str, chains: list[ChainInfo]) -> str:
    for chain in chains:
        if chain.primary == document_id:
            return "주 인용발명"
    for chain in chains:
        if document_id in chain.secondaries:
            return "보조 인용발명"
    return "미채택"


def build_claim_report(claim: Claim, chain: ChainInfo, matrix: dict[str, dict[str, ElementMatch]],
                       documents: dict[str, Document], mappings: list[DocumentMapping]) -> ClaimReport:
    selected = _reported_documents(chain)
    merged = merge_selected(claim, matrix, selected)
    results = [_element_result(claim, element.label, merged.get(element.label), chain, matrix,
                               documents, mappings)
               for element in claim.elements]
    return ClaimReport(
        claim_number=claim.number,
        depends_on=claim.depends_on,
        preamble=claim.preamble,
        track=chain.track,
        chain=chain,
        claims=results,
        conclusion=_conclusion(claim, chain, merged),
        coverage_summary=_coverage_summary(results),
        summary_similarity=_summary_similarity(claim, results),
        summary_difference=_summary_difference(chain, results),
    )


def _reported_documents(chain: ChainInfo) -> list[str]:
    """구성대비를 본문에 실을 문헌. 채택 조합이 있으면 그것, 없으면 참고용 문헌입니다.

    주 인용발명 자격을 갖춘 문헌이 없으면 조합은 비지만 문헌별 대비 결과는 그대로 남아
    있습니다(chain._no_primary_chain). 조합이 비었다는 이유로 본문을 비우면 원문 대조까지
    통과한 대응이 "대응되는 인용발명이 확인되지 않음"으로 나갑니다 — 구성대비를 수행하고도
    수행하지 않은 것처럼 보고하는 것입니다.

    역할 표기(_role, _chain_text)는 chain_documents를 그대로 쓰므로 여기 실리는 문헌은
    '미채택'으로 남습니다. 대비 결과를 보이는 것과 거절 이유에 세우는 것은 다른 문제입니다.
    """
    return chain_documents(chain) or list(chain.reference_only)


def _element_result(claim: Claim, label: str, match: ElementMatch | None, chain: ChainInfo,
                    matrix: dict[str, dict[str, ElementMatch]], documents: dict[str, Document],
                    mappings: list[DocumentMapping]) -> ClaimResult:
    element = next((item for item in claim.elements if item.label == label), None)
    text = element.text if element else label
    is_preamble = bool(element and element.is_preamble)
    if chain.track == "analysis_incomplete":
        # 판정을 받지 못한 구성에 "확인되지 않았습니다"라고 쓰면 근거 없는 사실 주장이 됩니다.
        # 대표값·등급도 붙이지 않고 미판정임을 그대로 노출합니다.
        return ClaimResult(
            label=label, is_preamble=is_preamble, claim=text, grade="판정 불가", emoji="⚠️",
            status="판정 불가",
            narrative=f'({label}) 구성은 구성대비 판정을 받지 못했습니다 — 재실행 필요',
        )

    primary = matrix.get(chain.primary or "", {}).get(label)
    corresponded = _corresponded(match)
    grade, emoji = report_grade(match)
    evidence = _collect_evidence(label, chain, matrix, documents, mappings)
    bridge = _antecedent_bridge(claim, label, match, chain, matrix)
    replaced_cell = bool(corresponded and match and chain.primary and (
        (match.document_id != chain.primary and _usable(primary)) or bridge is not None))
    combined = bool(replaced_cell or (match and match.combination_resolved))
    disclosed, total = limitation_counts(match)
    pending = unverified_count(match)
    status = _status_of(match, disclosed, pending)
    # 확인된 개시가 하나도 없으면 표시 등급과 서술도 함께 유보로 바꿉니다. 숫자만 고치고
    # 등급 배지에 "기술 사상 동일"을, 서술에 "구성과 부분적으로 대응됩니다"를 남기면 같은
    # 구성이 한 화면에서 서로 다른 말을 합니다 — 읽는 사람은 등급과 문장을 먼저 봅니다.
    # analysis_incomplete의 '판정 불가'와 같은 처리이고, 사유만 다릅니다.
    reserved = status == "판정 유보"
    if reserved:
        grade, emoji = "판정 유보 — 의미검증 미완료", "⚠️"
    return ClaimResult(
        label=label,
        is_preamble=is_preamble,
        claim=text,                      # 구성 원문은 입력을 그대로 씁니다. 모델이 고쳐 쓴 문장을 쓰지 않습니다.
        corresponded=corresponded,
        disclosed_limitations=disclosed,
        total_limitations=total,
        evidence_locations=evidence_locations(match) if corresponded else 0,
        missing_limitations=list(match.missing_limitations) if match else [],
        unverified_limitations=_unverified_entries(match),
        unverified_limitation_count=pending,
        grade=grade,
        emoji=emoji,
        narrative=_reserved_narrative(match, mappings) if reserved else _narrative(
            label, text, match, primary, replaced_cell, mappings, documents,
            _closest_related(label, matrix, mappings, documents),
            _gap_note(label, chain, mappings), bridge=bridge),
        difference=_difference(
            match, primary, replaced_cell, mappings, documents,
            chain.beyond_limit_residual.get(label), _unadopted_note(chain), bridge=bridge),
        combination=combined,
        evidence=evidence,
        status=status,
        adopted_document=match.document_id if corresponded and match else "",
        adopted_reference=(_reference_number(match.document_id, mappings)
                           if corresponded and match else None),
        trail=_verification_trail(label, matrix, mappings),
    )


def _corresponded(match: ElementMatch | None) -> bool:
    """보고서가 "대응된 구성"으로 다루는지. 등급표에 오르는 판정만 해당합니다.

    유사도 숫자의 유무에 기대지 않고 이름 있는 조건으로 드러냅니다.
    """
    return match is not None and match.judgment in _STATUS


# --- 구성대비 서술 -------------------------------------------------------------

def _gap_note(label: str, chain: ChainInfo, mappings: list[DocumentMapping]) -> str:
    """채택 조합이 커버하지 못한 구성의 첫 줄. 빈 문자열이면 진짜 공백입니다.

    "대응되는 인용발명이 확인되지 않음"은 사실 진술입니다. 결합 한도 때문에 뺀 문헌에 그
    기재가 실제로 있거나 주지관용으로 넘긴 구성에까지 그렇게 적으면, 읽는 사람은 이미 손에
    든 문헌을 다시 찾아 나서게 됩니다. 세 경우가 부르는 후속 조치가 각각 다르므로 문장도
    각각 다르게 씁니다.
    """
    if label in chain.well_known:
        names = ", ".join(_reference_name(document_id, mappings)
                          for document_id in chain.well_known_documents.get(label, []))
        return (f"({label}) 구성은 주지관용기술로 보아 결합에 더했음 — {names}에 같은 취지의 "
                "기재가 있음 (주지관용 인정 여부는 별도 확인 필요)")
    if label in chain.beyond_limit:
        names = ", ".join(_reference_name(document_id, mappings)
                          for document_id in chain.beyond_limit_documents.get(label, []))
        return (f"({label}) 구성은 채택된 인용발명 조합에는 대응 기재가 없음 — {names}에 대응 "
                f"기재가 있으나 {_unadopted_note(chain)} 이 거절 이유에는 세우지 않음")
    # 네 번째 경우. 원문 근거는 있는데 의미검증이 문헌 **단독으로는** 한정의 축 하나를 세우지
    # 못한 구성입니다. 앞의 셋과 달리 이것은 아직 결론이 아니므로 "확인되지 않음"으로 적을 수
    # 없습니다. 빠진 축을 같은 조합의 다른 인용발명이 대는지가 남은 질문이고, 그 질문에
    # 답하는 것이 진보성 결합입니다.
    if label in chain.combination_pending:
        reasons = chain.combination_pending_reasons.get(label) or []
        detail = f"\n  ↳ {reasons[0]}" if reasons else ""
        return (f"({label}) 구성은 인용발명에 원문 근거가 있으나 문헌 단독으로는 한정의 일부 축이 "
                f"확인되지 않아 판단을 유보함 — 결합 위에서 확인 필요 (미개시 아님){detail}")
    return ""


def _narrative(label: str, text: str, match: ElementMatch | None, primary: ElementMatch | None,
               combined: bool, mappings: list[DocumentMapping],
               documents: dict[str, Document], related: str = "", gap_note: str = "",
               bridge: ElementMatch | None = None) -> str:
    """구성 1개의 구성대비를 한 문장으로 조립합니다.

    대응 문헌이 없으면 유사도 없이 미대응 한 줄만 남깁니다. 근거가 약한 대응을 억지로
    문장으로 만들어 섞지 않습니다. 다만 원문 대조를 통과한 인접 기재까지 버리지는 않습니다.
    그것을 감추면 심사관이 이미 확인된 문단을 처음부터 다시 찾게 되고, 어디까지 검토된
    상태인지도 알 수 없게 됩니다.
    """
    if not _corresponded(match) or match is None:
        line = gap_note or f"({label}) 구성에 대응되는 인용발명이 확인되지 않음 — 추가 검색 필요"
        return f"{line}\n(가장 가까운 기재: {related} — 청구항 한정 전체를 개시하는 근거는 아님)" if related else line
    passage = _passage(match, mappings, documents)
    if bridge is not None:
        return (f"{passage}는 구성이 기재되어 있으나 참조되는 선행 구성은 같은 문헌에서 "
                f"확인되지 않고, {_passage(bridge, mappings, documents)}는 그 선행 "
                f"구성이 기재되어 있어 이를 결합하면 청구항의 \"{_clip(text)}\" 구성과 "
                "부분적으로 대응됩니다.")
    # 결합 문장은 **결합 상대가 실제로 기여할 때만** 씁니다. combination_resolved는 조합
    # 전체를 근거로 한정을 되살린 경우에도 참이라, primary가 그 구성을 개시하지 않았는데도
    # 이 가지로 들어올 수 있습니다. 그때는 단독 개시 문장이 사실에 맞습니다.
    if combined and _usable(primary):
        missing = "; ".join(primary.missing_limitations[:2]) or "청구항이 요구하는 세부 구성"
        return (f"{_passage(primary, mappings, documents)}는 구성이 기재되어 있으나 {missing}에 대한 "
                f"기재는 없고, {passage}는 구성이 기재되어 있어 이를 결합하면 "
                f'청구항의 "{_clip(text)}" 구성과 대응됩니다.')
    return (f"{passage}는 구성이 기재되어 있으며, {_reason_clause(match.reason)} "
            f'청구항의 "{_clip(text)}" 구성과 {_correspondence_verb(match)}.')


# 판정 라벨이 부분 대응이면 문장도 부분 대응이라고 끝나야 합니다.
_FULL_CORRESPONDENCE = {"동일", "실질적 동일"}


def _correspondence_verb(match: ElementMatch) -> str:
    """대응의 정도를 문장 끝에 반영합니다.

    모두 "대응됩니다"로 끝내면, 모델이 이유에 "…부분은 개시되어 있지 않다"고 적은 경우
    "…개시되어 있지 않으므로 청구항의 '…' 구성과 대응됩니다"라는 자기모순 문장이 나갑니다.
    등급은 '일부 유사'인데 문장만 완전 대응으로 읽히므로, 표를 보지 않는 사람은 그 구성이
    개시된 것으로 인용하게 됩니다.
    """
    return "대응됩니다" if match.judgment in _FULL_CORRESPONDENCE else "부분적으로 대응됩니다"


def _passage(match: ElementMatch, mappings: list[DocumentMapping],
             documents: dict[str, Document]) -> str:
    """`인용발명 N (문헌번호)에는 "발췌" (단락 [0000])("원문")` 까지를 만듭니다."""
    name = _reference_name(match.document_id, mappings)
    shown = _clip(match.quote_translation or match.quote, EXCERPT_LIMIT)
    if not shown:
        return f"{name}에는 해당 취지의 기재"
    original = (f'("{_clip(match.quote, ORIGINAL_LIMIT)}")'
                if match.quote_translation and match.quote else "")
    return f'{name}에는 "{shown}" ({_location(match, documents)}){original}'


# 판단 이유를 "…므로 청구항의 … 구성과 대응됩니다"에 이어 붙이기 위한 어미 변환.
# 프롬프트가 "므로"로 끝내라고 요구하지만, 모델이 평서형으로 답해도 문장이 깨지지 않게 합니다.
_CONNECTIVE_ENDINGS = ("므로", "때문에", "어서", "아서", "여서", "이라서")
_ENDING_REWRITES = (
    ("하고 있습니다", "하고 있으므로"), ("하고 있다", "하고 있으므로"), ("있습니다", "있으므로"),
    ("있다", "있으므로"), ("합니다", "하므로"), ("한다", "하므로"), ("입니다", "이므로"),
    ("이다", "이므로"), ("됩니다", "되므로"), ("된다", "되므로"),
    ("함", "하므로"), ("됨", "되므로"), ("음", "으므로"),
)


def _reason_clause(reason: str) -> str:
    """모델의 판단 이유를 결과절과 이어지는 원인절로 다듬습니다.

    이유는 생략할 수 없는 항목입니다. 모델이 이유를 주지 않았을 때만 발췌 자체를 가리키는
    중립 문구로 대체하고, 없는 근거를 지어내지 않습니다.
    """
    reason = re.sub(r"\s+", " ", str(reason or "")).strip().rstrip(". ")
    # 모델이 프롬프트 내 문헌 순서를 "인용발명 1"로 써도 보고서의 확정 매핑 번호와
    # 충돌하지 않도록 이유 앞의 임시 번호를 제거합니다.
    reason = re.sub(r"^(?:인용발명|인용문헌|문헌)\s*\d*\s*(?:에는|에서는|에서|은|는|이|가)\s*", "", reason)
    reason = re.sub(r"^청구항의?\s*", "", reason)
    if not reason:
        return "위 기재가 해당 구성의 입력·처리·출력 관계를 그대로 보여 주므로"
    if reason.endswith(_CONNECTIVE_ENDINGS):
        return reason
    for ending, replacement in _ENDING_REWRITES:
        if reason.endswith(ending):
            return reason[: -len(ending)] + replacement
    return f"{reason}에 해당하므로"


def _resolved_note(match: ElementMatch, mappings: list[DocumentMapping]) -> str:
    """채택 셀이 빠뜨린 한정을 조합 안의 다른 인용발명이 댄 경우 그 사실을 적습니다.

    이 줄이 없으면 chain._absorb_limitations가 메운 한정은 차이점에서 그냥 사라집니다.
    그러면 집계(개시 한정 수)는 올라갔는데 이유가 보고서 어디에도 없어, 읽는 사람은 무엇이
    왜 바뀌었는지 대조할 수 없습니다. 어느 문헌이 그 한정을 댔는지가 결합 거절 이유의
    본체이므로 문헌 이름과 함께 적습니다.
    """
    if not match.combination_resolved:
        return ""
    by_document: dict[str, list[str]] = {}
    for limitation, document_id in match.combination_resolved.items():
        by_document.setdefault(document_id, []).append(limitation)
    return ", ".join(
        f"{'; '.join(sorted(limitations)[:2])}은 "
        f"{_reference_name(document_id, mappings)}의 결합으로 해소됨"
        for document_id, limitations in sorted(by_document.items()))


def _difference(match: ElementMatch | None, primary: ElementMatch | None, combined: bool,
                mappings: list[DocumentMapping], documents: dict[str, Document],
                overflow: dict[str, list[str]] | None = None,
                unadopted_note: str = "",
                bridge: ElementMatch | None = None) -> str | None:
    """차이점 한 줄. 누락 사유를 고른 뒤 의미검증 미완료를 뒤에 덧붙입니다.

    분기마다 따로 적지 않고 마지막에 한 번 붙이는 이유: 아래 함수는 지시 관계·결합 해소·
    보완 등 사유가 여러 갈래이고 그중 하나는 **None**을 돌려줍니다(한정이 전부 개시된 동일급).
    미완료를 각 갈래에 적으면 반드시 한 갈래를 빠뜨리게 되고, 빠진 갈래에서는 등급이 왜
    눌렸는지가 본문에서 사라집니다 — 결론과 본문이 어긋나는 실측 사례가 정확히 그 형태였고,
    그때도 원인은 "차이점 줄을 만드는 경로가 여러 개"라는 것이었습니다.
    """
    pending = _unverified_entries(match)
    if not pending:
        return _difference_before_verification(match, primary, combined, mappings, documents,
                                               overflow, unadopted_note, bridge)
    note = "; ".join(pending[:2])
    if len(pending) > 2:
        note += f" 외 {len(pending) - 2}건"
    note += " 한정은 의미검증을 수행하지 못해 개시가 확인되지 않았습니다"
    # 확인된 개시가 하나도 없으면 **차이를 적지 않습니다.** 미검증은 차이가 확인된 상태가
    # 아닙니다. "세부 구현·조건에 차이가 있어 동일하다고 보기 어렵습니다"를 붙이면, 판단을
    # 받지 못한 것을 판단해 본 결과로 옮겨 적게 되고 바로 위의 '판정 유보'와도 어긋납니다.
    if is_reserved(match):
        return note
    base = _difference_before_verification(match, primary, combined, mappings, documents,
                                           overflow, unadopted_note, bridge)
    return f"{base.rstrip('.')}. 또한 {note}" if base else note


def _difference_before_verification(
        match: ElementMatch | None, primary: ElementMatch | None, combined: bool,
        mappings: list[DocumentMapping], documents: dict[str, Document],
        overflow: dict[str, list[str]] | None = None,
        unadopted_note: str = "",
        bridge: ElementMatch | None = None) -> str | None:
    if match is None or not _corresponded(match):
        return None
    # 지시 관계 상한은 다른 어떤 사유보다 먼저 적습니다. 다만 **누락 한정을 대신하지는
    # 않습니다.** "이 경우 하위 한정은 전부 개시로 남아 있다(2/2)"는 전제는 깨질 수 있고
    # (1/5), 그때 이 줄만 내보내면 보고서는 "선행 구성이 없어 완전 개시로 보지 않았다"만 적고
    # 실제로 빠진 나머지 한정은 한 줄도 남기지 않습니다. 읽는 사람은 집계와 차이점 줄을
    # 대조할 수 없게 됩니다.
    if match.antecedent_note:
        if match.missing_limitations:
            residual = _residual_gap(match.missing_limitations[:3], mappings, overflow,
                                     unadopted_note)
            return f"{match.antecedent_note}. 또한 {residual}에 대한 기재가 확인되지 않았습니다"
        return match.antecedent_note
    # 다른 채택 문헌이 선행 구성만 보완한 경우, 이 구성 자체에 남은 시점·조건 차이는
    # 해소된 것으로 쓰지 않습니다. 이번 비디오 편집 사례의 "편집 중" 시점이 여기에 해당합니다.
    if bridge is not None:
        if match.missing_limitations:
            return _residual_gap(match.missing_limitations[:3], mappings, overflow, unadopted_note)
        # 다른 문헌이 선행 구성만 이어 줬다고 이 셀 자체의 추론 직접성까지 사라지지는 않습니다.
        # 결합 서술은 antecedent bridge를 보여 주고, 차이점 줄은 남은 추론 한계를 보여 줍니다.
        if match.judgment not in FULL_JUDGMENTS or match.directness != "direct":
            return "; ".join(residual_difference(match)) or None
        return None
    # 채택 셀 자체는 주 인용발명인데, 그 셀이 빠뜨린 한정을 조합 안의 다른 문헌이 댄 경우.
    # 아래 `combined` 분기는 채택 셀의 **문헌이 바뀐** 경우만 다루므로 여기서 먼저 답합니다.
    resolved_note = _resolved_note(match, mappings)
    if resolved_note:
        if match.missing_limitations:
            residual = _residual_gap(match.missing_limitations[:3], mappings, overflow,
                                     unadopted_note)
            return f"{resolved_note}. 다만 {residual}은 결합 후에도 남음"
        if not match.antecedent_note and not match.different_purpose:
            return resolved_note
        # 한정은 모두 메워졌어도 지시 관계나 목적 차이가 남으면 그 차이는 감추지 않습니다.
        return (f"{resolved_note}. 다만 {match.judgment} 판정에 그쳐 하위 한정까지 "
                "동일하다고 보기는 어렵습니다")
    if combined and primary is not None:
        # 보완 문헌이 주 인용발명의 공백을 **전부** 메웠을 때만 "해소됨"이라고 적습니다.
        #
        # 결합이 일어났다는 사실만으로 이 문장을 쓰면 안 됩니다. 결론(chain.residual)은 채택
        # 셀의 누락을 보는데 이 줄이 주 인용발명의 누락만 보면, 같은 구성이 본문에는 "결합으로
        # 해소됨"으로 결론에는 "결합 후에도 차이가 남습니다"로 적혀 보고서가 스스로를
        # 반박합니다. 실제로는 두 한정 중 하나
        # 를 어느 문헌도 메우지 못한 상태일 수 있습니다.
        resolved = [limitation for limitation in primary.missing_limitations
                    if limitation not in match.missing_limitations]
        supplement = (f"{_reference_name(match.document_id, mappings)} "
                      f"({_location(match, documents)})")
        if not match.missing_limitations:
            gap = "; ".join(resolved[:2]) or "세부 구성"
            return (f"{_reference_name(primary.document_id, mappings)}은 {gap}에 대한 기재가 없으나 "
                    f"{supplement}의 결합으로 해소됨")
        residual = _residual_gap(match.missing_limitations[:3], mappings, overflow,
                                 unadopted_note)
        if resolved:
            return (f"{_reference_name(primary.document_id, mappings)}에 없던 "
                    f"{'; '.join(resolved[:2])}은 {supplement}의 결합으로 해소되었으나, "
                    f"{residual}은 결합 후에도 남음")
        return residual
    if match.missing_limitations:
        return _residual_gap(match.missing_limitations[:3], mappings, overflow, unadopted_note)
    if match.judgment in {"동일", "실질적 동일"} and not match.downgraded_from:
        return None
    return "세부 구현·조건에 차이가 있어 동일하다고 보기 어렵습니다."


def _antecedent_bridge(claim: Claim, label: str, match: ElementMatch | None,
                       chain: ChainInfo, matrix: dict[str, dict[str, ElementMatch]]) -> ElementMatch | None:
    """결합 복원에 사용된 선행 구성의 대표 근거를 찾습니다."""
    if match is None or not match.antecedent_resolved_by:
        return None
    source_labels = antecedents(claim).get(label, [])
    candidates = [matrix.get(document_id, {}).get(source_label)
                  for document_id in match.antecedent_resolved_by
                  if document_id in {chain.primary, *chain.secondaries}
                  for source_label in source_labels]
    return best_match(candidates)


def _unadopted_note(chain: ChainInfo) -> str:
    """채택되지 않은 문헌에 기재가 있을 때 그 이유. chain._unadopted_reason과 같은 값입니다.

    상한이 실제로 걸렸을 때만 상한 탓으로 적습니다. 자리가 남아 있는데도 빠진 문헌을 두고
    "상한을 넘었다"고 적으면 도구가 하지 않은 판단을 한 것처럼 보고하게 되고, 읽는 사람은
    상한만 올리면 그 문헌이 들어온다고 읽습니다.
    """
    if chain.limit_binding:
        if chain.inherited:
            return f"종속항 추가 인용발명 수 상한({chain.combination_limit}건)을 넘어"
        return f"결합 문헌 수 상한({chain.combination_limit}건)을 넘어"
    return "보완 후보 평가에서 채택되지 않아"


def _residual_gap(missing: list[str], mappings: list[DocumentMapping],
                  overflow: dict[str, list[str]] | None, unadopted_note: str) -> str:
    """남은 한정을 적되, 채택되지 않은 문헌이 개시한 것은 그 사실과 함께 적습니다.

    채택 밖에 기재가 있는 한정을 그냥 남은 차이로 적으면, 읽는 사람은 그것을 추가 검색
    대상으로 읽습니다. 그런데 그 기재는 이미 업로드된 문헌 안에 있습니다. 채택 여부는 *거절
    이유를 어떻게 세울지*의 문제이지 *문헌에 기재가 있느냐*의 문제가 아니므로 둘을 같은
    문장에 뭉치지 않습니다(chain.beyond_limit이 미대응 줄에서 하는 것과 같은 구분입니다).
    """
    parts: list[str] = []
    for limitation in missing:
        sources = (overflow or {}).get(limitation)
        if not sources:
            parts.append(limitation)
            continue
        names = ", ".join(_reference_name(document_id, mappings) for document_id in sources)
        parts.append(f"{limitation} ({names}에 대응 기재가 있으나 {unadopted_note} "
                     "이 거절 이유에는 세우지 않음)")
    return "; ".join(parts)


def _closest_related(label: str, matrix: dict[str, dict[str, ElementMatch]],
                     mappings: list[DocumentMapping], documents: dict[str, Document]) -> str:
    """미대응 구성에 대해 원문 대조를 통과한 가장 가까운 기재 하나를 고릅니다.

    **대표 발췌를 먼저 봅니다.** 보조 발췌(evidence)만 훑으면 대표 발췌만 있고 보조 발췌가
    없는 셀이 통째로 건너뛰어집니다. 그 셀이 바로 그 구성을 가장 잘 개시한 문헌인 경우가
    있습니다 — 지시 관계 상한이나 결합 한도로 채택에서 빠진 문헌이 그렇습니다. 그러면
    보고서에는 아무 관련 없는 문헌의 총론 문장이 "가장 가까운 기재"로 남고, 정작 확인된
    원문은 사라집니다.

    문헌 선택은 판정 강도 순입니다. 인용발명 번호 순으로 첫 번째를 집으면 그 구성과 무관한
    문헌이 번호만 빠르다는 이유로 뽑힙니다. 동률이면 번호 순이라 결과는 항상 같습니다.

    강도는 **강등되기 전** 판정으로 잽니다. 지시 관계 상한이나 발췌 검증으로 내려간 값으로
    재면, 원래 그 구성을 가장 잘 개시했던 문헌과 총론 한 줄만 걸린 문헌이 같은 등급으로
    납작해져 번호 순서가 승부를 가릅니다. "가장 가까운 기재"는 말 그대로 어디까지 근접했는지를
    묻는 자리이므로 근접했던 정도를 그대로 씁니다.
    """
    candidates: list[tuple] = []
    for order, mapping in enumerate(mappings):
        match = matrix.get(mapping.document_id, {}).get(label)
        if match is None:
            continue
        spans = [(match.chunk_id, match.quote, match.quote_translation, match.verify)]
        spans += [(span.chunk_id, span.quote, span.quote_translation, span.verify)
                  for span in match.evidence]
        for chunk_id, quote, translation, verify in spans:
            if verify != "verified" or not quote:
                continue
            strength = JUDGMENT_RANK.get(match.downgraded_from or match.judgment, 0)
            candidates.append((strength, -order, mapping.document_id, chunk_id,
                               translation or quote))
            break
    if not candidates:
        return ""
    _, _, document_id, chunk_id, quote = max(candidates)
    location = _chunk_location(document_id, chunk_id, documents)
    return f'{_reference_name(document_id, mappings)} "{_clip(quote, EXCERPT_LIMIT)}" ({location})'


def _chunk_location(document_id: str, chunk_id: str, documents: dict[str, Document]) -> str:
    page, paragraph = _chunk_position(documents.get(document_id), chunk_id)
    if paragraph:
        return f"단락 [{paragraph}]"
    if page:
        return f"{page} 페이지"
    return chunk_id or "출처 미상"


def _usable(match: ElementMatch | None) -> bool:
    """결합 문장에 주 인용발명을 함께 세울 수 있는지. 검증된 발췌 + **대응 판정**이 있어야 합니다.

    "대응 없음"만 걸러서는 부족합니다. "차이"는 core가 하나도 개시되지 않았다는 뜻이고, 그
    문헌에 남은 것은 가장 가까운 인접 기재뿐입니다. 그것을 결합 상대로 세우면 "…에는 …는
    구성이 기재되어 있으나"라는 문장이 붙어, 그 구성을 개시하지 않은 문헌이 기여한 것처럼
    읽힙니다. 실측: 점군 융합 구성에서 **텍스처 이미지 스티칭** 문장이 결합 상대로 실려,
    단독 개시인 구성이 두 문헌의 결합처럼 보고됐습니다.
    """
    return bool(match and match.quote and match.verify in {"verified", "partial"}
                and _corresponded(match))


# --- 근거 위치 ----------------------------------------------------------------

def _collect_evidence(label: str, chain: ChainInfo, matrix: dict[str, dict[str, ElementMatch]],
                      documents: dict[str, Document], mappings: list[DocumentMapping]) -> list[Evidence]:
    by_id = {mapping.document_id: mapping for mapping in mappings}
    evidence: list[Evidence] = []
    for document_id in _reported_documents(chain):
        match = matrix.get(document_id, {}).get(label)
        if not match:
            continue
        document = documents.get(document_id)
        mapping = by_id.get(document_id)
        def add(chunk_id: str, quote: str, translation: str, verify: str,
                limitation: str = "", kind: str = "",
                check: LimitationCheck | None = None) -> None:
            page, paragraph = _chunk_position(document, chunk_id)
            relation, note = _semantic_bridge(check)
            evidence.append(Evidence(
                document_id=document_id,
                filename=document.filename if document else "",
                reference_number=mapping.reference_number if mapping else None,
                document_number=(mapping.document_number if mapping else "") or None,
                paragraph=paragraph, page=page, chunk_id=chunk_id,
                excerpt=translation or quote,
                original_excerpt=quote if translation else None,
                quality=_QUALITY.get(verify, "UNVERIFIED"),
                limitation=limitation, kind=kind,
                semantic_relation=relation, semantic_note=note,
                verification_incomplete=is_unverified_check(check),
            ))

        spans = []
        if match.quote:
            spans.append((match.chunk_id, match.quote, match.quote_translation, match.verify))
        spans.extend((span.chunk_id, span.quote, span.quote_translation, span.verify)
                     for span in match.evidence if span.quote)
        seen: set[tuple[str, str]] = set()
        for chunk_id, quote, translation, verify in spans:
            if (chunk_id, quote) in seen:
                continue
            seen.add((chunk_id, quote))
            add(chunk_id, quote, translation, verify)
        # 하위 한정별 근거는 대표 발췌와 중복되더라도 따로 남깁니다. 대표 발췌 하나만
        # 남기면 "어느 한정을 무엇으로 개시했는가"가 사라지고, 총론 문장 하나로 구성
        # 전체를 개시했다고 적은 판정과 한정마다 원문을 짚은 판정이 보고서에서 구별되지
        # 않습니다. 어느 쪽인지가 곧 그 판정을 다툴 수 있는지를 가릅니다.
        for check in match.limitation_checks:
            if check.disclosed and check.quote and not check.whole_element:
                add(check.chunk_id, check.quote, check.quote_translation, check.verify,
                    limitation=check.limitation, kind=check.kind, check=check)
                for span in check.evidence:
                    if span.quote:
                        add(span.chunk_id, span.quote, span.quote_translation, span.verify,
                            limitation=check.limitation, kind=check.kind, check=check)
    return evidence


# 의미검증이 개시를 인정하며 붙인 관계. explicit은 발췌가 문언 그대로를 담은 경우라
# 따로 적지 않습니다. 나머지 둘은 발췌와 한정 사이를 **판단으로** 이은 것이므로 적습니다.
_BRIDGE_LABELS = {
    "necessary_implicit": "필연적 함의로 인정",
    "functional_equivalent": "기능적 동등으로 인정",
}


# 옛 기록에는 append-only 이벤트 로그가 없고 종단 상태만 있습니다. 그때는 상태 하나에서
# 이벤트 하나를 되짚어 만듭니다 — 경로는 잃었지만 마지막 판정은 남아 있으므로, 아무것도 적지
# 않는 것보다 낫습니다. 새 기록은 semantic_events를 그대로 씁니다.
_TERMINAL_STAGES: dict[str, tuple[str, str]] = {
    "accepted": ("의미검증", "인정"),
    "rejected": ("의미검증", "기각"),
    "accepted_in_combination": ("결합검증", "인정"),
    "rejected_in_combination": ("결합검증", "기각"),
}


def _events(check: LimitationCheck) -> list[SemanticEvent]:
    """이 한정을 건드린 검증 이벤트. 로그가 없는 옛 기록은 종단 상태에서 되짚습니다."""
    if check.semantic_events:
        return list(check.semantic_events)
    found = _TERMINAL_STAGES.get(check.semantic_status)
    if found is None:
        return []
    stage, outcome = found
    return [SemanticEvent(stage=stage, outcome=outcome, note=check.semantic_note,
                          supplied_by=list(check.combination_documents))]


def _reportable(events: list[SemanticEvent]) -> list[SemanticEvent]:
    """경위로 적을 이벤트만 남깁니다.

    의미검증 인정 하나로 끝난 한정은 근거 목록의 "↳ 발췌 문언 그대로는 아니며 …" 줄이 이미
    같은 말을 합니다. 두 자리에 같은 문장을 적으면 경위가 보고서의 기본 소음이 되고, 그러면
    정작 판정이 움직인 한정이 그 소음에 묻힙니다. **단계를 두 번 이상 거친 한정은 그대로
    남깁니다** — 그때는 근거 줄이 마지막 판정만 말하므로 경로가 사라집니다.
    """
    if len(events) == 1 and events[0].stage == "의미검증" and events[0].outcome == "인정":
        return []
    return events


def _verification_trail(label: str, matrix: dict[str, dict[str, ElementMatch]],
                        mappings: list[DocumentMapping]) -> list[VerificationTrail]:
    """이 구성이 어떤 단계를 거쳐 지금 판정이 되었는지를 문헌별로 모읍니다.

    **채택 문헌이 아니라 행렬 전체를 훑습니다.** 미대응 구성에는 채택 문헌이 없어서, 채택된
    셀만 보면 경위가 가장 필요한 구성에서 정확히 아무것도 남지 않습니다. 경위가 있다는 것은
    그 문헌을 실제로 심사했다는 뜻이므로, 채택 여부와 무관하게 남길 값입니다.
    """
    trails: list[VerificationTrail] = []
    for document_id in sorted(matrix):
        match = matrix[document_id].get(label)
        if match is None:
            continue
        steps = [TrailStep(index=check.index, limitation=check.limitation,
                           stage=event.stage, outcome=event.outcome,
                           note=_clip(event.note, 300),
                           combination_documents=list(event.supplied_by))
                 for check in match.limitation_checks
                 for event in _reportable(_events(check))]
        trail = VerificationTrail(
            document_id=document_id,
            reference_number=_reference_number(document_id, mappings),
            sample_count=match.sample_count,
            sample_unanimous=match.sample_unanimous,
            sample_requirements=match.sample_requirements,
            steps=steps,
            # 만장일치가 **아닌** 한정만 싣습니다. 갈린 것뿐 아니라 무응답·판독불가가 섞인
            # 한정도 포함됩니다 — 셋 중 하나만 답한 자리는 갈린 것은 아니어도 확인이 더
            # 필요합니다. 전부 실으면 정작 볼 것이 그 목록에 묻힙니다.
            tallies=[(check.index, check.limitation, check.sample_tally)
                     for check in match.limitation_checks
                     if check.sample_tally.available and not check.sample_tally.unanimous],
        )
        # 적을 것이 없는 문헌은 싣지 않습니다. 만장일치로 통과한 셀까지 한 줄씩 나가면
        # 경위가 보고서의 기본 소음이 되고, 그러면 갈린 셀이 묻힙니다.
        if steps or trail.split or trail.tallies:
            trails.append(trail)
    return trails


def _trail_lines(item: ClaimResult, mappings: list[DocumentMapping]) -> list[str]:
    """판정 경위를 단계별로 적습니다. 초기 비교·의미검증·결합검증을 한 줄에 섞지 않습니다."""
    if not item.trail:
        return []
    lines = ["", "판정 경위:"]
    for trail in item.trail:
        where = _document_name(trail.document_id, trail.reference_number)
        if trail.split:
            # 비율이 아니라 분자·분모를 적습니다. "0.33"은 두 가지로 읽히지만 "3개 중 1개"는
            # 한 가지로만 읽힙니다.
            # **"갈렸다"고 단정하지 않습니다.** 만장일치가 아닌 이유는 판단이 나뉜 것일 수도,
            # 표본이 답하지 못한 것일 수도 있습니다. 어느 쪽인지는 아래 한정별 줄이 말하고,
            # 이 줄은 셀 단위로 확인할 수 있는 사실(만장일치 비율)만 적습니다.
            lines.append(f"- ({where}) ⚠️ 초기 비교 결과가 불안정합니다 — "
                         f"한정 {trail.sample_requirements}개 중 "
                         f"{trail.sample_unanimous}개만 표본 {trail.sample_count}회 만장일치")
        # 어느 한정이 몇 대 몇으로 갈렸는지. 구성 단위 비율만으로는 "2대 1로 갈린 미개시"와
        # "3대 0으로 일치한 미개시"가 같은 값이 되는데, 그 둘은 다음 조치가 다릅니다.
        lines += [f"  · 한정 #{index} 「{_clip(text, 46)}」 {_tally_kind(tally)}: "
                  f"개시 {tally.disclosed}표 · 미개시 {tally.missing}표"
                  + (f" · 무응답 {absent}표" if (absent := tally.count("absent")) else "")
                  + (f" · 판독불가 {invalid}표" if (invalid := tally.count("invalid")) else "")
                  + f" (표본 {tally.total}회)"
                  for index, text, tally in trail.tallies]
        # 단계 순서(의미검증 → 결합검증)로 묶습니다. 한정 번호 순으로 늘어놓으면 두 단계가
        # 번갈아 나와, 어느 단계가 무엇을 걸렀는지가 줄을 세어야 보입니다.
        #
        # 한정 하나가 **두 단계 모두에** 나올 수 있습니다. 문헌 단독으로 기각된 뒤 결합 위에서
        # 다시 심사받은 한정이 그렇고, 그 두 줄이 함께 있어야 "결합까지 보고도 안 됐다"가
        # 읽힙니다(models.SemanticEvent).
        for stage in ("의미검증", "결합검증"):
            for step in [item for item in trail.steps if item.stage == stage]:
                supplied = ""
                if step.combination_documents:
                    names = ", ".join(
                        _document_name(document_id, _reference_number(document_id, mappings))
                        for document_id in step.combination_documents)
                    supplied = f" (빠진 축은 {names}가 개시)"
                note = f" — {step.note}" if step.note else ""
                # 번호는 감사 데이터(judgment.json·verify_notes)의 한정 인덱스와 같은 값이라
                # 0부터입니다. "한정 0"으로 적으면 개수로 읽히므로 식별자임을 표시합니다.
                lines.append(f"- ({where}) {step.stage} {step.outcome}: "
                             f"한정 #{step.index} 「{_clip(step.limitation, 60)}」{supplied}{note}")
    return lines


def _tally_kind(tally: SampleTally) -> str:
    """이 한정이 왜 만장일치가 아닌지. 후속 조치가 갈리므로 한 낱말로 뭉치지 않습니다.

    판정 불일치는 같은 근거를 두고 판단이 나뉜 것이라 사람이 원문을 봐야 하고, 응답 결손은
    도구가 답을 받지 못한 것이라 다시 물으면 됩니다.
    """
    kinds = [name for name, holds in (("판정 불일치", tally.split),
                                      ("응답 결손", tally.incomplete)) if holds]
    return " · ".join(kinds) or "표본 불일치"


def _document_name(document_id: str, reference_number: int | None) -> str:
    return f"인용발명 {reference_number}" if reference_number is not None else f"문헌 {document_id}"


def _semantic_bridge(check: LimitationCheck | None) -> tuple[str, str]:
    """이 한정의 개시가 발췌 문언 그대로가 아니라 의미검증의 판단에 서 있으면 그 근거.

    의미검증(entailment)은 발췌 한 문장이 아니라 그 문장이 속한 청크 원문과 형제 한정의
    인용문까지 함께 읽고 판단합니다. 그런데 보고서에 찍히는 것은 짧은 대표 발췌 하나뿐이라,
    인정의 실제 근거가 그 발췌 밖에 있으면 독자에게는 보이지 않습니다. 그러면 발췌만 읽은
    심사관은 도구가 개시를 잘못 인정했다고 볼 수밖에 없습니다.

    판단을 감추지 않고 관계와 이유를 함께 내보내면, 그 다리가 타당한지를 다툴 수 있습니다.
    """
    if check is None or check.semantic_status != "accepted":
        return "", ""
    label = _BRIDGE_LABELS.get(check.semantic_relation)
    if not label:
        return "", ""
    return label, _clip(check.semantic_note, 160)


def _chunk_position(document: Document | None, chunk_id: str) -> tuple[int | None, str | None]:
    if document is None:
        return None, None
    for chunk in document.chunks:
        if chunk.chunk_id == chunk_id:
            return chunk.page, chunk.paragraph
    return None, None


def _location(match: ElementMatch, documents: dict[str, Document]) -> str:
    """단락번호가 있으면 단락으로, 없으면 페이지로 인용 위치를 적습니다."""
    return _chunk_location(match.document_id, match.chunk_id, documents)


def _reference_name(document_id: str | None, mappings: list[DocumentMapping]) -> str:
    for mapping in mappings:
        if mapping.document_id == document_id:
            detail = mapping.document_number or mapping.filename
            return f"인용발명 {mapping.reference_number} ({detail})" if detail \
                else f"인용발명 {mapping.reference_number}"
    return f"문헌 {document_id}" if document_id else "주 인용발명"


def _reference_number(document_id: str, mappings: list[DocumentMapping]) -> int | None:
    for mapping in mappings:
        if mapping.document_id == document_id:
            return mapping.reference_number
    return None


# --- 결론 --------------------------------------------------------------------

_TRACK_TITLES = {
    "novelty_single": "신규성 없음 (단일 인용발명)",
    "inventive_step_combination": "진보성 검토 (인용발명 결합)",
    "rejection_impossible": "거절 이유 구성 곤란",
    "analysis_incomplete": "구성대비 미완료",
}


def _conclusion(claim: Claim, chain: ChainInfo, merged: dict[str, ElementMatch]) -> str:
    """이 청구항에 어떤 거절 이유가 서는지 한 줄로 확정합니다.

    구성별 유사도만 늘어놓고 결론을 적지 않으면, 읽는 사람은 표를 되짚어 신규성인지
    진보성인지를 스스로 추정하게 됩니다. 추정의 방향은 읽는 사람마다 다르고, 그 추정이
    보고서의 결론으로 인용됩니다. track과 rationale은 이미 확정된 값이므로 그대로 적습니다.
    """
    title = _TRACK_TITLES.get(chain.track, "결론 미확정")
    if chain.track == "analysis_incomplete":
        return f"{title} — 판정을 받지 못한 셀이 있어 신규성·진보성 결론을 만들지 않았습니다."
    detail = [chain.rationale] if chain.rationale else []
    detail += _equivalence_caveat(claim, chain, merged)
    # 미대응 구성을 한 덩어리로 적으면 "기재가 없다"와 "기재는 있으나 한도를 넘었다"가
    # 구별되지 않습니다. 앞의 것만 추가 검색이 필요한 항목입니다.
    genuine = [label for label in chain.uncovered
               if label not in chain.beyond_limit and label not in chain.well_known]
    if genuine:
        detail.append(f"미대응 구성: {', '.join(genuine)}.")
    if chain.beyond_limit:
        detail.append(f"미채택 인용발명에만 기재가 있는 구성: {', '.join(chain.beyond_limit)} "
                      f"({_unadopted_note(chain)} 세우지 않음).")
    if chain.well_known:
        detail.append(f"주지관용기술로 다룬 구성: {', '.join(chain.well_known)}.")
    if chain.residual:
        # 남은 차이 중 미채택 문헌이 메우는 것은 따로 적습니다. 뭉쳐 두면 결론 줄만 읽는
        # 사람이 그 구성 전체를 추가 검토 대상으로 옮겨 적게 됩니다.
        limited = [label for label in chain.residual if label in chain.beyond_limit_residual]
        # 확인된 개시가 하나도 없는 구성은 "차이가 남는다"가 아닙니다. 차이는 대비해 본
        # 결과이고, 이쪽은 대비 자체를 못 한 것입니다. 후속 조치도 다릅니다 — 앞의 것은
        # 보완 문헌 검색, 뒤의 것은 원문 확인과 재실행입니다.
        reserved = set(chain.reserved)
        differing = [label for label in chain.residual if label not in reserved]
        reserved = [label for label in chain.residual if label in reserved]
        if differing:
            detail.append(f"차이가 남는 구성: {', '.join(differing)}.")
        if reserved:
            detail.append(f"의미검증 미완료로 판정을 유보한 구성: {', '.join(reserved)}.")
        if limited:
            detail.append(f"그중 미채택 인용발명이 그 한정을 개시한 구성: {', '.join(limited)}.")
    return f"{title} — {' '.join(detail)}" if detail else title


def _equivalence_caveat(claim: Claim, chain: ChainInfo,
                        merged: dict[str, ElementMatch]) -> list[str]:
    """신규성 부정이 문언 그대로의 개시가 아니라 등가 판단에 서 있으면 그렇다고 적습니다.

    신규성 부정은 청구항을 죽이는 가장 강한 결론인데, `실질적 동일`은 "용어가 다르나
    기술적 의미가 같다"는 **판단**입니다. 판단 근거가 등가성인 구성을 밝히지 않으면
    보고서는 문언이 그대로 있었던 것과 구별되지 않는 모습으로 나가고, 정작 다투어야 할
    등가 여부가 검토 대상에서 빠집니다. 등가로 본 구성이 하나도 없으면 아무것도 적지 않습니다.
    """
    if chain.track != "novelty_single":
        return []
    equivalent = [element.label for element in claim.elements
                  if not element.is_preamble
                  and (match := merged.get(element.label)) and match.judgment == "실질적 동일"]
    if not equivalent:
        return []
    return [f"다만 구성 {', '.join(equivalent)}은 문언 그대로의 개시가 아니라 '실질적 동일'"
            "(용어가 다르나 기술적 의미와 작동 관계가 같음) 판단에 근거하므로, 결론을 확정하기 전에 "
            "각 구성의 등가 여부를 근거 발췌로 확인해야 합니다."]


# --- 종합 분석 요약 -----------------------------------------------------------

def _summary_similarity(claim: Claim, results: list[ClaimResult]) -> str:
    """청구항과 인용발명이 공유하는 내용을 한 줄로 요약합니다.

    구성 원문을 " 및 "로 이어 붙이지 않습니다. 구성 문언은 "…하는 단계 및", "…를 포함하되"
    처럼 다음 구성으로 이어지는 어미로 끝나는 일이 많아 그대로 이으면 문장이 깨지고,
    무엇보다 그것은 요약이 아니라 구성 목록이라 위에 이미 적힌 내용을 다시 읽히게 됩니다.

    요약은 **무엇이 공통인가**(가장 중요한 대응 구성 하나)와 **어디까지 공통인가**(대응 범위)
    두 가지로 만듭니다. 둘 다 확정된 판정 데이터에서 나오므로 LLM을 다시 부르지 않습니다.
    """
    if any(result.status == "판정 불가" for result in results):
        return "구성대비 판정을 받지 못해 유사 내용을 요약할 수 없습니다."
    substantive = [result for result in results if not result.is_preamble]
    # 확인된 개시가 하나도 없는 구성은 공통점 요약에서 뺍니다. 그 구성을 "대응 기재가
    # 확인됨"으로 세면, 본문에서 판정 유보로 적어 둔 것을 요약이 공통점으로 되살립니다.
    corresponded = [result for result in substantive
                    if result.corresponded and result.status != "판정 유보"]
    if not corresponded:
        return "청구항과 인용발명 사이에 대응되는 기술 내용이 확인되지 않았습니다."
    representative = _representative(claim, corresponded)
    # 문헌 번호는 **대표 구성이 실제로 대응된 문헌**만 적습니다. 대응된 모든 구성의 채택
    # 문헌을 합쳐 적으면 주어와 술어가 서로 다른 출처에서 옵니다 — 실측: 융합 구성을
    # 인용발명 2 하나가 개시했는데 요약은 "인용발명 1, 2, 3은 …융합… 구성에서 공통되며"로
    # 나갔습니다. 나머지 문헌이 어디까지 대응했는지는 뒤의 범위 문구가 이미 말합니다.
    references = (f"인용발명 {representative.adopted_reference}"
                  if representative.adopted_reference is not None else "제시된 인용발명")
    common = _summary_phrase(representative.claim)
    # 부분 개시 구성을 "모두 …에서 공통된다"고 적으면, 정작 개시되지 않은 한정을 공통점으로
    # 단언하게 됩니다. 실제로 "제1·제2 반사부재 **사이에** 배치되는 광원"이 그 배치 관계는
    # 개시되지 않은 채 '일부 유사'를 받았는데, 요약은 그 문언 그대로를 공통점으로 적었습니다.
    partial = representative.status != "개시됨"
    ground = "구성에서 부분적으로 공통되며" if partial else "구성에서 공통되며"
    # 구성 문언을 따옴표로 묶습니다. 긴 구성은 잘릴 수밖에 없는데, 묶지 않으면 어디까지가
    # 청구항 문언이고 어디부터가 요약자의 말인지 구별되지 않고 생략 부호도 어색하게 붙습니다.
    return (f"청구항과 {references}{_topic_particle(references)} \"{common}\" "
            f"{ground}, {_scope(substantive, corresponded)}.")


def _representative(claim: Claim, corresponded: list[ClaimResult]) -> ClaimResult:
    """유사점을 대표할 구성 하나. 완전 개시를 먼저 보고, 그 안에서 중요도 순입니다.

    여러 구성을 나열하면 요약이 아니라 목록이 됩니다. 어느 구성이 이 청구항의 변별점인지는
    분해 단계가 매긴 중요도가 이미 말해 주므로 그것을 그대로 씁니다.

    완전 개시를 앞세우는 이유: 부분 개시 구성의 문언에는 **개시되지 않은 한정까지** 들어
    있어서, 그것을 공통점 문장에 그대로 넣으면 없는 개시를 단언하게 됩니다. 완전 개시가
    하나도 없을 때만 부분 개시를 쓰고, 그때는 문장도 "부분적으로 공통"으로 바뀝니다.
    """
    importance = {element.label: element.importance for element in claim.elements}
    order = {result.label: index for index, result in enumerate(corresponded)}
    return max(corresponded, key=lambda result: (result.status == "개시됨",
                                                 importance.get(result.label, 0),
                                                 -order[result.label]))


def _scope(substantive: list[ClaimResult], corresponded: list[ClaimResult]) -> str:
    """대응 범위 한 마디. 몇 개 중 몇 개가, 어느 강도로 대응되는지."""
    total, matched = len(substantive), len(corresponded)
    partial = sum(1 for result in corresponded if result.status == "부분 개시")
    scope = (f"청구항의 구성 {total}개 전부에 대응 기재가 확인됩니다" if matched == total
             else f"청구항의 구성 {total}개 중 {matched}개에 대응 기재가 확인됩니다")
    return f"{scope}(그중 {partial}개는 부분 개시)" if partial else scope


# 구성 문언의 꼬리를 "… 구성에서 공통되며"에 이어 붙일 수 있는 관형형으로 바꿉니다.
# 청구항 구성은 다음 구성으로 이어지는 어미로 끝나는 일이 많아, 그대로 두면 요약 문장이
# 연결어미 자리에서 끊깁니다.
_PHRASE_ENDINGS = (
    ("되고", "되는"), ("되며", "되는"), ("되어", "되는"), ("된다", "되는"), ("됨", "되는"),
    ("하고", "하는"), ("하며", "하는"), ("하여", "하는"), ("한다", "하는"), ("함", "하는"),
    ("이고", "인"), ("이며", "인"), ("이다", "인"),
)
# 관형형 뒤에 붙는 형식 명사. 떼어 내야 "…하는 구성에서"로 이어집니다.
_PHRASE_TAIL_NOUNS = re.compile(r"\s*(?:단계|과정|방법|장치|시스템|것)\s*$")


def _summary_phrase(text: str, limit: int = 80) -> str:
    """구성 문언을 요약 문장 안에 넣을 수 있는 형태로 다듬습니다.

    앞뒤의 "상기"(앞 구성을 가리키는 지시어라 요약문에서는 가리킬 대상이 없습니다), 다음
    구성으로 이어지는 꼬리("… 및", "…를 포함하되"), 관형형 뒤의 형식 명사를 떼어 내고
    연결어미를 관형형으로 되돌립니다.
    """
    phrase = re.sub(r"\s+", " ", str(text or "")).strip()
    phrase = re.sub(r"상기\s*", "", phrase)
    phrase = re.sub(r"(?:을|를)?\s*포함하(?:되|고|며|는|여)\s*$", "", phrase.strip(" ,.;·"))
    phrase = re.sub(r"[,\s]*(?:및|또는)\s*$", "", phrase.strip(" ,.;·")).strip(" ,.;·")
    phrase = _PHRASE_TAIL_NOUNS.sub("", phrase).strip(" ,.;·")
    for ending, adnominal in _PHRASE_ENDINGS:
        if phrase.endswith(ending):
            phrase = phrase[: -len(ending)] + adnominal
            break
    return _clip_words(phrase.strip(" ,.;·"), limit)


def _clip_words(text: str, limit: int) -> str:
    """낱말 경계에서 자릅니다. 글자 수로 끊으면 요약문이 낱말 한가운데서 끊깁니다."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    boundary = head.rfind(" ")
    return (head[:boundary] if boundary > limit // 2 else head).rstrip(" ,.;·") + "…"


# 숫자를 한국어로 읽었을 때 받침이 있는지. 조사 은/는 선택에만 씁니다.
_DIGIT_HAS_FINAL = {"0": True, "1": True, "3": True, "6": True, "7": True, "8": True,
                    "2": False, "4": False, "5": False, "9": False}


def _topic_particle(word: str) -> str:
    """앞 낱말의 받침에 따라 "은"/"는"을 고릅니다.

    인용발명 번호가 그대로 조사 앞에 오므로("인용발명 1", "인용발명 2") 한쪽으로 고정하면
    보고서마다 어느 한쪽이 반드시 틀립니다.
    """
    text = str(word or "").strip()
    if not text:
        return "은"
    last = text[-1]
    if last.isdigit():
        return "은" if _DIGIT_HAS_FINAL[last] else "는"
    if "가" <= last <= "힣":
        return "은" if (ord(last) - 0xAC00) % 28 else "는"
    return "은"


def pipeline_invariants(reports: list[ClaimReport],
                        matrices: dict[int, dict[str, dict[str, ElementMatch]]]) -> list[str]:
    """**사건과 무관하게** 항상 참이어야 하는 성질만 확인합니다. 기대값이 필요 없습니다.

    이 파이프라인의 회귀는 지금까지 전부 "새 청구항·새 인용발명을 넣으니 또 안 된다"는
    형태로 왔고, 그때마다 관측한 셀 하나를 고쳤습니다. 사건별 기대값(cases/expected.json)은
    사람이 문헌을 통독해야 쓸 수 있어서 사건이 늘지 않고, 늘지 않으면 다음 사건은 여전히
    처음 보는 사건입니다. 그래서 채점의 축을 사건별 정답에서 **성질**로 옮깁니다.

    아래 성질들은 어떤 청구항·어떤 문헌 조합에서도 참이어야 하므로, 사람이 아무것도 적지
    않아도 모든 실행에서 자동으로 채점됩니다. 실제 분석 실행에도 그대로 붙습니다 — 회귀는
    하니스보다 실사용에서 먼저 나타나기 때문입니다.

    고치지는 않습니다. 어느 쪽이 맞는지는 사안마다 다르므로 어긋났다는 사실만 남깁니다.
    """
    notes: list[str] = []
    for report in reports:
        matrix = matrices.get(report.claim_number) or {}
        if report.chain.track == "analysis_incomplete":
            # 미판정 상태에서는 아래 성질들이 애초에 성립할 수 없습니다. 미판정을 '대응 없음'과
            # 같은 칸에 넣지 않는 것이 이 파이프라인의 규율이므로 여기서도 가릅니다.
            continue
        notes += _evidence_is_never_erased(report, matrix)
        notes += _every_rejection_has_a_reason(report, matrix)
        notes += _grades_never_exceed_their_own_evidence(report, matrix)
        notes += _no_full_grade_rests_on_an_unchecked_limitation(report, matrix)
        notes += _verified_judgments_keep_their_trail(report, matrix)
        notes += _sample_tallies_add_up(report, matrix)
    return notes


def _sample_tallies_add_up(report: ClaimReport,
                           matrix: dict[str, dict[str, ElementMatch]]) -> list[str]:
    """표본 집계가 원시 투표와 어긋나지 않는지.

    집계는 투표에서 유도하므로 정의상 갈릴 수 없지만, 기록되는 투표 **수**가 물어본 표본
    수와 다르면 유도값 자체가 사실이 아닙니다. "3표 중 2표"라고 적힌 줄이 실제로는 두 표만
    받은 것이면, 읽는 사람은 없는 표본 하나를 셈에 넣습니다.
    """
    notes: list[str] = []
    for document_id, matches in sorted(matrix.items()):
        for label, match in sorted(matches.items()):
            for check in match.limitation_checks:
                tally = check.sample_tally
                if not tally.available or len(tally.votes) == tally.total:
                    continue
                notes.append(f"[불변식 S1] 청구항 {report.claim_number} ({label}) / 문헌 "
                             f"{document_id} 한정 #{check.index}: 표본 {tally.total}회인데 "
                             f"기록된 투표는 {len(tally.votes)}건입니다.")
    return notes


def _verified_judgments_keep_their_trail(
        report: ClaimReport, matrix: dict[str, dict[str, ElementMatch]]) -> list[str]:
    """검증 단계가 움직인 판정은 보고서에도 남아야 합니다.

    행렬에는 "의미검증이 이 한정을 기각했다"가 사유와 함께 기록되는데, 보고서가 그것을 읽지
    않으면 남는 것은 결과 한 줄뿐입니다. 그러면 읽는 사람은 도구가 그 구성을 **검토하지
    않았다**고 읽습니다 — 실제로는 근거를 들어 두 번 기각한 것인데도 그렇습니다. 실측에서
    바로 그 상태의 보고서를 두고 두 사람이 정반대 결론을 냈습니다.

    _evidence_is_never_erased가 **인정된** 근거를 지키는 것과 같은 이유로, 이쪽은 **기각된**
    판단의 사유를 지킵니다. 인정만 남기고 기각을 버리면 보고서는 한쪽으로만 검증 가능해집니다.
    """
    notes: list[str] = []
    by_label = {item.label: item for item in report.claims}
    for label, item in sorted(by_label.items()):
        # 이벤트 단위로 셉니다. 한정 단위로 세면 두 단계를 거친 한정에서 뒤엣것만 남아도
        # 통과합니다 — 앞선 기각이 사라지는 것이 애초에 이 기록을 만든 이유입니다.
        moved = {(document_id, check.index, event.stage, event.outcome)
                 for document_id, matches in matrix.items()
                 for check in (matches.get(label).limitation_checks if matches.get(label) else [])
                 for event in _reportable(_events(check))}
        carried = {(trail.document_id, step.index, step.stage, step.outcome)
                   for trail in item.trail for step in trail.steps}
        if lost := moved - carried:
            notes.append(f"[보고서 정합성] 청구항 {report.claim_number} ({label}): 검증 단계가 "
                         f"판정을 바꾼 기록 {len(lost)}건이 보고서 경위에서 빠졌습니다.")
    return notes


def _no_full_grade_rests_on_an_unchecked_limitation(
        report: ClaimReport, matrix: dict[str, dict[str, ElementMatch]]) -> list[str]:
    """P4. 의미검증을 받지 못한 한정 위에 동일급 판정이 설 수 없습니다.

    실측에서 정확히 이 형태로 새어 나갔습니다. 구성 D·E의 한정 다섯 건이 검증기 응답에서
    빠졌고(entailment._mark_unchecked), 그 다섯 건이 그대로 개시로 남아 두 구성이 🟢
    실질적 동일을 받았습니다. 그중 하나(절대좌표계에 대응하는 최종 3D 모델)는 원문을 읽으면
    그 문헌에 없는 구성이었습니다.

    derive_judgment가 이미 상한을 씌우지만, 상한은 판정을 재산출하는 경로를 탄 셀에만
    걸립니다. 재산출을 부르지 않는 경로가 하나라도 남으면 같은 결과가 다시 나갑니다 —
    P3가 "등급이 한정별 개시보다 높다"를 결과로 확인하는 것과 같은 이유로, 이쪽도 경로가
    아니라 **결과**를 봅니다.
    """
    notes: list[str] = []
    for document_id, matches in sorted(matrix.items()):
        for label, match in sorted(matches.items()):
            if match.error or match.judgment not in FULL_JUDGMENTS:
                continue
            pending = unverified_limitations(match)
            if pending:
                notes.append(
                    f"[불변식 P4] 청구항 {report.claim_number} ({label}) / 문헌 "
                    f"{document_id}: 등급 '{match.judgment}'인데 한정 {len(pending)}건이 "
                    f"의미검증을 받지 못했습니다 ({'; '.join(pending)[:120]}).")
    return notes


def _evidence_is_never_erased(report: ClaimReport,
                              matrix: dict[str, dict[str, ElementMatch]]) -> list[str]:
    """P1. "어느 인용발명에서도 확인되지 않았다"는 진짜 공백에만 쓸 수 있습니다.

    업로드된 문헌 중 하나라도 그 구성의 한정을 원문 대조를 통과한 근거로 개시했거나 축
    결손으로만 기각했다면, 그 구성을 공백으로 적는 것은 **사실과 다른 진술**입니다. 읽는
    사람은 이미 손에 든 문헌을 다시 찾아 나서게 됩니다.

    파이프라인이 이 진술을 만드는 경로가 여러 개(uncovered·rejection_impossible·미채택)라
    한 곳을 막아도 다른 곳으로 새어 나왔습니다. 그래서 경로가 아니라 **결과**를 봅니다.
    """
    notes: list[str] = []
    chain = report.chain
    excused = {*chain.beyond_limit, *chain.well_known, *chain.combination_pending}
    for label in chain.uncovered:
        if label in excused:
            continue
        holders = sorted(document_id for document_id, matches in matrix.items()
                         if evidenced_limitations(matches.get(label)))
        if holders:
            notes.append(f"[불변식 P1] 청구항 {report.claim_number} ({label}): 공백으로 적었으나 "
                         f"문헌 {', '.join(holders)}에 원문 대조를 통과한 개시 근거가 있습니다.")
    return notes


def _every_rejection_has_a_reason(report: ClaimReport,
                                  matrix: dict[str, dict[str, ElementMatch]]) -> list[str]:
    """P2. 결합에서 빠진 문헌에는 빠진 이유가 **기록되어** 있어야 합니다.

    어떤 구성의 근거를 가진 문헌이 채택되지 않았다면, 그 문헌은 보완 후보 평가에서 떨어진
    것이고 그 사유가 element_coverage의 후보 행에 남아 있어야 합니다. 사유 없이 사라지는
    문헌이 있다는 것은 게이트 하나가 조용히 경로를 막고 있다는 뜻입니다 —
    neareye-waveguide에서 gain 0.3짜리 유효 후보가 정확히 그렇게 사라졌고, 그 사실은 어떤
    채점에도 걸리지 않았습니다.

    **묻는 것은 사유의 유무이지 이득의 크기가 아닙니다.** 종전에는 주 인용발명 대비 이득이
    양수인데 limit_binding이 거짓이면 위반으로 봤는데, 그 둘은 기준선이 다릅니다 — 앞은
    주 인용발명 단독 대비, 뒤는 채택 조합 전체 대비입니다. 그래서 채택된 보조 인용발명이
    이미 같은 것을 대고 있는 **중복 후보**마다 위반이 찍혔습니다. 실측에서 두 문헌이 같은
    구성에 정확히 같은 이득(0.4417)을 냈고, 하나가 채택되자 다른 하나가 매 회차 위반으로
    보고됐습니다. 동률은 흔하므로 그 상태로는 경고가 늘 켜져 진짜 위반이 묻힙니다.

    limit_binding으로 전체를 건너뛰지도 않습니다. 상한이 걸린 실행에도 사유 없이 사라진
    후보가 있을 수 있고, 이제는 상한 자체가 후보 행에 사유로 적히므로 걸러 낼 필요가
    없습니다(chain._exclusion_reason).
    """
    notes: list[str] = []
    chain = report.chain
    if not chain.primary:
        return notes
    adopted = {chain.primary, *chain.secondaries}
    by_label = {coverage.label: coverage for coverage in chain.element_coverage}
    for label in [*chain.uncovered, *chain.residual]:
        coverage = by_label.get(label)
        if coverage is None:
            continue
        for document_id, matches in sorted(matrix.items()):
            if document_id in adopted or not evidenced_limitations(matches.get(label)):
                continue
            row = next((item for item in coverage.candidates
                        if item.document_id == document_id), None)
            if row is None:
                notes.append(f"[불변식 P2] 청구항 {report.claim_number} ({label}): 문헌 "
                             f"{document_id}에 개시 근거가 있는데 후보 평가 기록이 없습니다.")
            elif not row.adopted and not row.excluded_reason and label not in chain.beyond_limit:
                notes.append(f"[불변식 P2] 청구항 {report.claim_number} ({label}): 문헌 "
                             f"{document_id}은 조합에 {row.merged_gain}을 더 보탤 수 있는데 "
                             "채택되지 않았고, 제외 사유도 기록되지 않았습니다.")
    return notes


def _grades_never_exceed_their_own_evidence(
        report: ClaimReport, matrix: dict[str, dict[str, ElementMatch]]) -> list[str]:
    """P3. 판정 등급은 한정별 개시 여부에서 유도된 값을 넘을 수 없습니다.

    등급은 derive_judgment의 결정론적 함수입니다(coverage). 그보다 **높은** 등급이 셀에
    남아 있다면 어딘가에서 파생값이 1차 사실과 어긋난 채 굳은 것이고, 그 등급은 그대로
    has_correspondence·신규성 게이트·문헌 순위로 들어갑니다.

    낮은 쪽은 보지 않습니다. 발췌 검증 실패·지시 관계 상한·결합 결과 상한은 모두 등급을
    **의도적으로** 내리는 장치라, 양방향으로 검사하면 정상 동작이 매번 위반으로 찍힙니다.
    """
    notes: list[str] = []
    for matches in matrix.values():
        for label, match in sorted(matches.items()):
            if match.error or not match.limitation_checks:
                continue
            derived = derive_judgment(
                match.limitation_checks, has_evidence=bool(match.quote or match.evidence),
                terminology=match.terminology, different_purpose=match.different_purpose)
            if JUDGMENT_RANK.get(match.judgment, 0) > JUDGMENT_RANK.get(derived, 0):
                notes.append(f"[불변식 P3] 청구항 {report.claim_number} ({label}) / 문헌 "
                             f"{match.document_id}: 등급 '{match.judgment}'이 한정별 개시에서 "
                             f"유도되는 '{derived}'보다 높습니다.")
    return notes


def report_invariants(reports: list[ClaimReport]) -> list[str]:
    """조립된 보고서가 스스로를 반박하지 않는지 확인합니다.

    이 파이프라인의 버그는 계산이 틀리는 형태보다 **본문과 결론이 서로 다른 자료를 보는**
    형태로 나왔습니다. 실제로 나간 두 건이 그랬습니다.
      - 결론은 "구성 B에 차이가 남습니다"인데 본문은 "인용발명 2의 결합으로 해소됨"
        (본문이 주 인용발명의 누락만 보고, 결론은 채택 셀의 누락을 봤습니다)
      - 집계는 "1/5 개시"인데 차이점 줄에는 지시 관계 사유만 있고 누락 한정은 한 줄도 없음
    둘 다 테스트가 아니라 보고서를 눈으로 읽다가 발견됐습니다. 조립 직후에 기계적으로
    맞춰 보면 같은 유형이 다시 새어 나가지 않습니다.

    고치지는 않습니다. 어느 쪽이 맞는지는 사안마다 다르므로, 어긋났다는 사실만 남깁니다.
    """
    notes: list[str] = []
    for report in reports:
        residual = set(report.chain.residual)
        for item in report.claims:
            gap = item.total_limitations - item.disclosed_limitations
            difference = (item.difference or "").strip()
            if item.label in residual and not difference:
                notes.append(f"[보고서 정합성] 청구항 {report.claim_number} ({item.label}): "
                             "결론은 차이가 남는다고 적었는데 본문에 차이점 줄이 없습니다.")
            # 빠진 한정이 있으면 차이점 줄이 **그 한정을 실제로 언급**해야 합니다. 줄이
            # 있기만 하면 통과시키면, 지시 관계 사유 한 줄만 적고 빠진 한정 넷을 통째로
            # 삼킨 실측 사례를 놓칩니다.
            # 누락 목록 자체가 없으면 대조할 것이 없습니다. 그때는 "차이점 줄이 아예 없다"만
            # 봅니다 — 빈 목록으로 "하나도 언급되지 않았다"를 참으로 만들면, 멀쩡한 보고서가
            # 전부 지적됩니다(필드가 없던 시절의 산출물이 정확히 그랬습니다).
            mentioned = any(limitation and limitation in difference
                            for limitation in item.missing_limitations)
            if item.corresponded and gap > 0 and (
                    not difference or (item.missing_limitations and not mentioned)):
                notes.append(f"[보고서 정합성] 청구항 {report.claim_number} ({item.label}): "
                             f"한정 {item.disclosed_limitations}/{item.total_limitations} 개시인데 "
                             "본문에 빠진 한정이 적히지 않았습니다.")
            if item.label in residual and difference.endswith("결합으로 해소됨"):
                notes.append(f"[보고서 정합성] 청구항 {report.claim_number} ({item.label}): "
                             "본문은 결합으로 해소되었다고 적었는데 결론은 차이가 남는다고 "
                             "적었습니다.")
    return notes


def _summary_difference(chain: ChainInfo, results: list[ClaimResult]) -> str:
    """구성별 차이점과 겹치지 않는 범위에서 가장 두드러진 차이를 한 줄로 정리합니다.

    커버되지 않은 구성을 한 덩어리로 적지 않습니다. "어디에서도 확인되지 않았다"는 진짜
    공백에만 해당하고, 결합 한도 밖 문헌에 기재가 있는 구성과 주지관용으로 넘긴 구성은
    각각 다른 후속 조치를 부릅니다 — 앞의 것은 추가 검색, 뒤의 것은 문헌 선택 재검토,
    마지막은 주지관용 인정 여부 확인입니다.
    """
    if chain.track == "analysis_incomplete":
        return "구성대비가 완료되지 않아 차이점을 특정할 수 없습니다."
    uncovered = [result.label for result in results
                 if not result.corresponded and not result.is_preamble]
    lines: list[str] = []
    genuine = [label for label in uncovered
               if label not in chain.beyond_limit and label not in chain.well_known
               and label not in chain.combination_pending]
    limited = [label for label in uncovered if label in chain.beyond_limit]
    well_known = [label for label in uncovered if label in chain.well_known]
    pending = [label for label in uncovered if label in chain.combination_pending]
    if genuine:
        lines.append(f"구성 {', '.join(genuine)}은 제시된 인용발명 어디에서도 대응 기재가 확인되지 않아 "
                     "추가 검색이 필요합니다.")
    if pending:
        lines.append(f"구성 {', '.join(pending)}은 인용발명에 원문 근거가 있으나 문헌 단독으로는 한정의 "
                     "일부 축이 확인되지 않아 판단을 유보했습니다. 추가 검색 대상이 아니라 "
                     "결합 위에서 확인할 대상입니다.")
    if limited:
        lines.append(f"구성 {', '.join(limited)}은 대응 기재를 가진 인용발명이 있으나 "
                     f"{_unadopted_note(chain)} 이 조합에 세우지 않았습니다.")
    if well_known:
        lines.append(f"구성 {', '.join(well_known)}은 주지관용기술로 보아 결합에 더했으며, "
                     "그 인정 여부는 별도로 확인해야 합니다.")
    # 유보 구성은 차이 목록에서 뺍니다. 본문과 결론이 '판정 유보'로 적은 것을 요약이
    # "세부 구현·하위 한정에 차이가 남아 있다"로 되살리면, 요약만 읽는 독자는 대비해 본
    # 결과가 있다고 읽습니다. 여기서만 새어 나가면 앞의 두 곳을 고친 의미가 없습니다.
    if chain.reserved:
        lines.append(f"구성 {', '.join(chain.reserved)}은 의미검증을 수행하지 못해 개시 여부를 "
                     "확정하지 못했습니다. 차이 판단이 아니라 원문 확인이 필요한 항목입니다.")
    if lines:
        return " ".join(lines)
    residual = [label for label in chain.residual
                if label not in chain.uncovered and label not in set(chain.reserved)]
    if not residual:
        return ""
    line = f"구성 {', '.join(residual)}은 결합 후에도 세부 구현·하위 한정에 차이가 남아 있습니다."
    # 그 차이 중 일부가 미채택 문헌에 이미 있다면 여기서도 갈라 적습니다. 요약만 읽고
    # 추가 검색 목록을 만드는 독자에게는 이 줄이 유일한 신호입니다.
    limited = [label for label in residual if label in chain.beyond_limit_residual]
    if limited:
        line += (f" 그중 구성 {', '.join(limited)}의 남은 한정은 채택하지 않은 인용발명에 "
                 f"대응 기재가 있습니다({_unadopted_note(chain)} 세우지 않음).")
    return line


def _clip(text: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# --- 마크다운 ----------------------------------------------------------------

# 선행기술 결과의 실재 확인 표시. 구성대비 발췌를 원문 대조하는 것과 같은 이유로, 검색
# 결과도 코드가 URL을 열어 문헌번호를 확인합니다. 확인하지 못한 것을 지우지는 않습니다.
_PRIOR_ART_VERIFY = {
    "verified": "✅ 확인됨",
    "mismatch": "⚠️ 문헌번호 불일치",
    "unreachable": "❔ 확인 불가",
    "unchecked": "❔ 미확인",
}

def to_markdown(result: AnalysisResult) -> str:
    lines = ["# 구성대비 분석", ""]
    incomplete = [report.claim_number for report in result.reports
                  if report.track == "analysis_incomplete"]
    if incomplete:
        lines += [f"> ⚠️ **구성대비 미완료** — 청구항 "
                  f"{', '.join(str(number) for number in incomplete)}은 판정을 받지 못한 셀이 있어 "
                  "결론을 만들지 않았습니다. 미판정은 '대응 없음'과 다릅니다.", ""]
    lines += ["## 문헌 매핑 테이블", "",
              "| 인용발명 | 문헌번호 | 파일명 | 공개·제출일 | 역할 |",
              "|---|---|---|---|---|"]
    lines += [f"| 인용발명 {mapping.reference_number} | {mapping.document_number or '-'} "
              f"| {mapping.filename} | {mapping.publication_date or mapping.filing_date or '-'} "
              f"| {mapping.role} |"
              for mapping in result.claim_mapping]
    for report in result.reports:
        lines += _claim_section(report, result.claim_mapping)
    if result.prior_art:
        lines += ["", "## 미커버 구성 선행기술 검색", "",
                  "> 이 절의 문헌은 웹 검색 결과입니다. 아래 표시는 제시된 URL을 열어 문헌번호가"
                  " 그 페이지에 있는지만 확인한 것이며, 선행기술 적격성 판단이 아닙니다.", ""]
        for hit in result.prior_art:
            lines.append(f"- {_PRIOR_ART_VERIFY[hit.verify]} ({hit.label}) "
                         f"{hit.document_number or hit.title or '문헌 미상'}"
                         + (f" · {hit.published}" if hit.published else "")
                         + (f" — {hit.correspondence}" if hit.correspondence else "")
                         + (f" (남은 차이: {hit.remaining_difference})" if hit.remaining_difference else "")
                         + (f" {hit.url}" if hit.url else "")
                         + (f" [{hit.verify_note}]" if hit.verify_note else ""))
    if result.validation:
        lines += ["", "## 참고", ""] + [f"- {item}" for item in result.validation]
    return "\n".join(lines) + "\n"


def _claim_section(report: ClaimReport, mappings: list[DocumentMapping]) -> list[str]:
    header = f"청구항 {report.claim_number}"
    if report.depends_on:
        header += f" (청구항 {report.depends_on} 종속)"
    lines = ["", f"## {header}", ""]
    if report.track == "analysis_incomplete":
        lines += ["**인용발명 조합**: 판정을 받지 못해 확정하지 않았습니다.", ""]
        lines += [f"> - {reason}" for reason in report.chain.incomplete_reasons] + [""]
    elif not any(item.adopted_reference for item in report.claims):
        # 종속항은 부모항의 조합을 상속하므로, 추가 구성에 대응이 하나도 없어도 상속된
        # 문헌 이름이 그대로 찍힙니다. 그러면 그 문헌들이 이 청구항의 거절 근거인 것처럼
        # 읽히지만, 실제로는 추가 한정을 개시한 문헌이 없습니다.
        lines += ["**인용발명 조합**: 이 청구항의 구성에 대응하는 인용발명이 확인되지 않았습니다.", ""]
    else:
        lines += [f"**인용발명 조합**: {_chain_text(report.chain, mappings)}", ""]
    if report.conclusion:
        lines += [f"**결론**: {report.conclusion}", ""]
    if report.coverage_summary:
        lines += [f"**구성 집계**: {report.coverage_summary}", ""]
    if report.preamble:
        lines += [f"> {report.preamble}", ""]
    for item in report.claims:
        heading = "전제부" if item.is_preamble else item.label
        lines += ["", f"### ({heading}) {item.claim}", ""]
        if item.corresponded:
            lines.append(_metric_line(item))
        lines.append(item.narrative.replace("\n", "  \n"))
        if item.difference:
            lines.append(f"→ 차이점: {item.difference}")
        lines += _limitation_evidence_lines(item)
        lines += _trail_lines(item, mappings)
    lines += ["", "### 종합 분석 요약", ""]
    if report.summary_similarity:
        lines.append(f"- 유사점: {report.summary_similarity}")
    if report.summary_difference:
        lines.append(f"- 차이점: {report.summary_difference}")
    lines += _review_needed_lines(report)
    return lines


def _reserved_narrative(match: ElementMatch | None, mappings: list[DocumentMapping]) -> str:
    """확인된 개시가 하나도 없는 구성의 서술. 대응을 주장하지 않습니다.

    이 셀에서 확인된 사실은 **발췌가 원문에 실재한다**는 것 하나뿐입니다. 그 이상을 적으면
    판단받지 않은 것을 판단한 결과로 옮겨 적게 됩니다.
    """
    where = _reference_name(match.document_id, mappings) if match else "제시된 인용발명"
    return (f"{where}에 관련 기재가 있고 발췌가 원문에 실재하는 것까지는 확인했으나, "
            "그 발췌가 이 구성의 한정을 뒷받침하는지는 의미검증에서 판단을 받지 못했습니다. "
            "개시 여부를 원문으로 직접 확인하십시오.")


def _status_of(match: ElementMatch | None, disclosed: int, pending: int) -> str:
    """집계에 쓰는 구성 상태. 확인된 개시가 하나도 없으면 '판정 유보'입니다.

    '부분 개시'는 "일부는 이 문헌에 있고 일부는 없다"는 뜻입니다. 한정이 전부 미완료인
    구성을 거기 넣으면, 아무것도 확인하지 못한 상태가 절반쯤 확인된 상태로 집계됩니다.
    '미개시'로 넣는 것도 같은 크기의 오류를 반대 방향으로 냅니다 — 그쪽은 문헌에 없다는
    뜻이 됩니다. 세 칸 어디에도 맞지 않으므로 칸을 하나 더 둡니다.
    """
    if match is None:
        return "미개시"
    if is_reserved(match):
        return "판정 유보"
    return _STATUS.get(match.judgment, "미개시")


def _unverified_entries(match: ElementMatch | None) -> list[str]:
    """보고서에 적을 미완료 항목. 문언이 없어도 **표시는 사라지지 않게** 합니다.

    구성이 하위 한정으로 분해되지 않은 경우(whole_element) 그 점검의 limitation은 구성 원문
    한 줄 전체라 목록에서 빠집니다. 그런데 등급 상한은 그 점검에도 걸리므로, 문언이 없다는
    이유로 아무것도 적지 않으면 "등급이 왜 눌렸는지 본문에 없는" 상태가 됩니다 — 결론과
    본문이 어긋나는, 이 파이프라인이 되풀이한 바로 그 형태입니다.
    """
    texts = unverified_limitations(match)
    if texts or not unverified_count(match):
        return texts
    return ["구성 전체 (하위 한정으로 분해되지 않은 점검)"]


def _review_needed_lines(report: ClaimReport) -> list[str]:
    """도구가 확인하지 못한 것을 한자리에 모읍니다. 판정이 아니라 **작업 지시**입니다.

    구성별 근거 줄에도 같은 사실이 찍히지만, 그것은 그 구성을 읽는 사람에게만 보입니다.
    결론만 읽고 넘기는 독자에게는 "이 보고서의 어느 판정이 검증 위에 서 있지 않은가"가
    한 번도 보이지 않습니다. 실측 사건에서 🟢 두 개가 정확히 그렇게 나갔습니다.

    판정을 여기서 되돌리지 않습니다. 어느 쪽이 옳은지는 원문을 읽어야 정해지고, 이 절의
    목적은 읽어야 할 자리를 지목하는 것입니다.
    """
    pending = [item for item in report.claims if item.unverified_limitations]
    if not pending:
        return []
    lines = ["", "### 검토 필요 — 의미검증 미완료", "",
             "아래 한정은 발췌가 원문에 실재하는 것까지만 확인했고, 그 발췌가 한정을 "
             "뒷받침하는지는 판단을 받지 못했습니다. 개시로 적혀 있으나 확정된 개시가 "
             "아니므로, 해당 구성의 등급은 동일급을 받지 못하도록 상한이 걸려 있습니다. "
             "원문으로 직접 확인하십시오.", ""]
    for item in pending:
        heading = "전제부" if item.is_preamble else item.label
        for limitation in item.unverified_limitations:
            lines.append(f"- ({heading}) {_clip(limitation, 100)}")
    return lines


def _metric_line(item: ClaimResult) -> str:
    """구성 한 줄의 정량 지표. 백분율 대신 셀 수 있는 값만 적습니다.

    분자·분모가 그대로 보이므로 독자가 바로 아래 근거 목록과 대조해 검증할 수 있습니다.
    한정 분해가 되지 않은 구성(총 0개)에는 커버율을 적지 않습니다 — 없는 분모를 지어내면
    "0/0 개시"처럼 읽혀 미개시로 오해됩니다.
    """
    parts = []
    if item.total_limitations:
        parts.append(_limitation_tally(item))
    parts.append(f"{item.emoji} {item.grade}")
    if item.evidence_locations:
        parts.append(f"근거 {item.evidence_locations}곳")
    return " · ".join(parts)


def _limitation_tally(item: ClaimResult) -> str:
    """한정 집계 한 줄. 미완료가 있으면 세 칸으로 갈라 적습니다.

    "한정 3/3 개시 · ⚠️ 의미검증 미완료 3건"은 스스로를 반박합니다 — 셋 다 개시인데 셋 다
    미완료일 수는 없습니다. 내부의 disclosed 값을 보존하는 것과 그 값을 그대로 보여 주는
    것은 다른 문제입니다. 읽는 사람에게 필요한 것은 "확인된 것이 몇 개인가"이고, 그 답은
    미완료를 개시에 포함한 분자로는 나오지 않습니다.

    미완료가 없으면 종전 형식을 그대로 씁니다. 흔한 경우의 표기를 바꾸면 옛 보고서와
    대조가 안 되고, 세 칸을 늘 적으면 0이 두 개 붙은 줄이 매번 나갑니다.
    """
    pending = item.unverified_limitation_count
    if not pending:
        return f"한정 {item.disclosed_limitations}/{item.total_limitations} 개시"
    confirmed = max(item.disclosed_limitations - pending, 0)
    uncovered = max(item.total_limitations - item.disclosed_limitations, 0)
    return (f"한정 {item.total_limitations}개 — 개시 확인 {confirmed} · "
            f"⚠️ 의미검증 미완료 {pending} · 미개시 {uncovered}")


def _coverage_summary(results: list[ClaimResult]) -> str:
    """이 청구항의 구성이 어떻게 갈렸는지 한 줄.

    결론(신규성·진보성·거절 곤란)을 실제로 정하는 것은 구성별 등급이 아니라 이 집계입니다.
    구성별 숫자만 늘어놓으면 읽는 사람이 표를 되짚어 직접 세게 됩니다.
    """
    substantive = [result for result in results if not result.is_preamble]
    if not substantive:
        return ""
    if any(result.status == "판정 불가" for result in substantive):
        return f"구성 {len(substantive)}개 — 판정을 받지 못해 집계하지 않았습니다"
    full = sum(1 for result in substantive if result.status == "개시됨")
    partial = sum(1 for result in substantive if result.status == "부분 개시")
    # 확인된 개시가 하나도 없는 구성은 따로 셉니다. 부분개시에 섞으면 "구성 5개 중 2개가
    # 부분 개시"라는 집계가 실제로는 "2개는 아무것도 확인하지 못함"을 감춥니다. 결론을
    # 정하는 것이 구성별 등급이 아니라 이 집계이므로, 여기서 감추면 결론이 흔들립니다.
    reserved = sum(1 for result in substantive if result.status == "판정 유보")
    uncovered = len(substantive) - full - partial - reserved
    summary = f"구성 {len(substantive)}개 — 완전개시 {full} · 부분개시 {partial}"
    if reserved:
        summary += f" · ⚠️ 판정유보 {reserved}"
    return f"{summary} · 미대응 {uncovered}"


def _limitation_evidence_lines(item: ClaimResult) -> list[str]:
    """하위 한정마다 무엇을 근거로 개시를 인정했는지 적습니다.

    대표 발췌 한 문장만 찍던 종전 형식으로는, 총론 한 줄로 구성 전체를 개시했다고 본
    판정과 한정마다 실시 문장을 짚은 판정이 똑같이 "…는 구성이 기재되어 있으며"로 나갑니다.
    등급이 같아도 다툴 수 있는 판정인지는 여기서 갈리므로, 근거를 한정 단위로 남깁니다.
    """
    proofs = [evidence for evidence in item.evidence if evidence.limitation]
    if not proofs:
        return []
    lines = ["", "근거:"]
    for proof in proofs:
        kind = f"{proof.kind} · " if proof.kind else ""
        lines.append(f"- ({kind}{_clip(proof.limitation, 80)}) "
                     f'"{_clip(proof.excerpt, EXCERPT_LIMIT)}" ({_evidence_source(proof)})')
        # 발췌 문언 그대로가 아니라 의미검증의 판단으로 인정된 개시는 그 판단을 붙여 적습니다.
        # 없으면 발췌만 읽는 독자에게는 근거가 어긋나 보입니다(_semantic_bridge).
        if proof.semantic_relation:
            note = f" — {proof.semantic_note}" if proof.semantic_note else ""
            lines.append(f"  ↳ 발췌 문언 그대로는 아니며 {proof.semantic_relation}{note}")
        # 검증을 받은 근거에만 위 단서가 붙으므로, 받지 못한 근거는 아무 표시가 없으면
        # **가장 튼튼한 근거로** 읽힙니다. 신호가 뒤집히지 않도록 사실을 그대로 적습니다.
        elif proof.verification_incomplete:
            lines.append("  ↳ ⚠️ 의미검증 미완료 — 발췌가 원문에 실재하는 것은 확인했으나, "
                         "이 발췌가 한정을 뒷받침하는지는 판단을 받지 못했습니다")
    return lines


def _evidence_source(evidence: Evidence) -> str:
    """근거 한 줄의 출처. 인용발명 번호를 위치와 함께 반드시 적습니다.

    구성 하나의 근거 목록에는 결합에 쓰인 **여러 문헌**의 발췌가 함께 들어갑니다. 번호가 없으면
    바로 위 구성대비 문장이 지목한 문헌 하나가 목록 전체의 출처인 것처럼 읽히는데, 실제로는
    다른 인용발명의 문장이 섞여 있습니다. 그대로 인용하면 어느 문헌에 없는 기재를 그 문헌의
    개시로 적게 되고, 그것이 곧 거절 이유의 근거로 나갑니다.
    """
    location = _evidence_location(evidence)
    if evidence.reference_number is None:
        return location
    return f"인용발명 {evidence.reference_number} · {location}"


def _evidence_location(evidence: Evidence) -> str:
    if evidence.paragraph:
        return f"단락 [{evidence.paragraph}]"
    if evidence.page:
        return f"{evidence.page} 페이지"
    return evidence.chunk_id or "출처 미상"


def _chain_text(chain: ChainInfo, mappings: list[DocumentMapping]) -> str:
    """이 청구항의 거절 이유가 무엇 위에 서 있는지 한 줄.

    주지관용기술을 함께 세웠으면 그것도 적습니다. 조용히 빼면 "인용발명 1 + 주지관용기술"과
    "인용발명 1 단독"이 보고서에서 구별되지 않는데, 둘은 다른 거절 이유입니다.
    """
    names = [_reference_name(document_id, mappings) for document_id in chain_documents(chain)]
    if chain.well_known:
        names.append(f"주지관용기술 (구성 {', '.join(chain.well_known)})")
    if names:
        return " + ".join(names)
    if chain.reference_only:
        # 조합은 세우지 않았지만 아래 구성대비가 어느 문헌의 대비 결과인지는 밝혀야 합니다.
        # 밝히지 않으면 근거 발췌의 출처가 채택된 인용발명인 것처럼 읽힙니다.
        referenced = ", ".join(_reference_name(document_id, mappings)
                               for document_id in chain.reference_only)
        return (f"채택된 인용발명 없음 — 주 인용발명 자격을 갖춘 문헌이 없어 조합을 세우지 "
                f"않았습니다. 아래 구성대비는 {referenced}에 대한 대비 결과이며 모두 미채택입니다.")
    return "채택된 인용발명 없음"
