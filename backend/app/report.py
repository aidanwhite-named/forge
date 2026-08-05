"""보고서 조립. 문장은 확정된 판정 데이터에서 템플릿으로 만들고 LLM에 다시 묻지 않습니다.

LLM이 쓴 마크다운을 정규식으로 되돌려 고치는 코드가 필요 없어지는 대신,
표현은 템플릿이 허용하는 범위로 제한됩니다. 판정 근거의 재현성을 우선한 선택입니다.
"""
import re

from .chain import chain_documents, rejection_basis
from .coverage import REPORT_GRADE, REPORT_PERCENT, best_match, item_similarity
from .models import (AnalysisResult, ChainInfo, Claim, ClaimReport, ClaimResult, Document,
                     DocumentMapping, ElementCoverage, ElementMatch, Evidence)

_CONCLUSION = {
    "동일": "동일합니다",
    "실질적 동일": "실질적으로 동일합니다",
    "일부 차이": "대체로 대응되나 세부 구현에 차이가 있습니다",
    "일부 유사": "핵심 기능은 유사하나 목적·효과에 차이가 있습니다",
    "차이": "대응된다고 보기 어렵습니다",
    "대응 없음": "대응되는 기재가 확인되지 않습니다",
}
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
    selected = chain_documents(chain)
    merged: dict[str, ElementMatch] = {}
    for element in claim.elements:
        merged[element.label] = best_match([matrix.get(document_id, {}).get(element.label)
                                            for document_id in selected])
    coverages = {coverage.label: coverage for coverage in chain.element_coverage}
    results = [_element_result(claim, element.label, merged.get(element.label), chain, matrix,
                               documents, mappings, coverages.get(element.label))
               for element in claim.elements]
    return ClaimReport(
        claim_number=claim.number,
        depends_on=claim.depends_on,
        preamble=claim.preamble,
        track=chain.track,
        rejection_basis=rejection_basis(chain, claim),
        chain=chain,
        claims=results,
        summary=_summary(claim, chain, results, merged, mappings),
        summary_similarity=_summary_similarity(results),
        summary_difference=_summary_difference(claim, chain, results),
    )


def _element_result(claim: Claim, label: str, match: ElementMatch | None, chain: ChainInfo,
                    matrix: dict[str, dict[str, ElementMatch]], documents: dict[str, Document],
                    mappings: list[DocumentMapping],
                    coverage: ElementCoverage | None = None) -> ClaimResult:
    element = next((item for item in claim.elements if item.label == label), None)
    text = element.text if element else label
    if chain.track == "analysis_incomplete":
        # 판정을 받지 못한 구성에 "확인되지 않았습니다"라고 쓰면 근거 없는 사실 주장이 됩니다.
        # 대표값·등급도 붙이지 않고 미판정임을 그대로 노출합니다.
        errors = [cell.error for document_id in sorted(matrix)
                  if (cell := matrix[document_id].get(label)) is not None and cell.error]
        return ClaimResult(
            label=label, is_preamble=bool(element and element.is_preamble),
            claim=text, similarity=None, grade="판정 불가", emoji="⚠️",
            status="판정 불가",
            narrative=f'청구항의 "{_clip(text)}" 구성은 구성대비 판정을 받지 못했습니다. '
                      "대응 기재의 유무가 확인되지 않은 상태입니다."
                      + (f" (사유: {_clip(errors[0], 160)})" if errors else ""),
            note="; ".join(dict.fromkeys(errors))[:400],
        )
    judgment = match.judgment if match else "대응 없음"
    grade, emoji = REPORT_GRADE[judgment]
    percent = REPORT_PERCENT[judgment]
    supporting = [document_id for document_id in chain_documents(chain)
                  if item_similarity(matrix.get(document_id, {}).get(label)) >= item_similarity(match) > 0]
    evidence = _collect_evidence(label, chain, matrix, documents, mappings)
    return ClaimResult(
        label=label,
        is_preamble=bool(element and element.is_preamble),
        claim=text,                      # 구성 원문은 입력을 그대로 씁니다. 모델이 고쳐 쓴 문장을 쓰지 않습니다.
        similarity=percent,
        grade=grade,
        emoji=emoji,
        narrative=_narrative(text, match, chain, matrix, mappings, documents),
        difference=_difference(match),
        combination=len(supporting) > 1 or (match is not None and match.document_id != chain.primary),
        references=[document_id for document_id in {evidence_item.document_id for evidence_item in evidence}],
        evidence=evidence,
        status={"동일": "개시됨", "실질적 동일": "개시됨", "일부 차이": "부분 개시",
                "일부 유사": "부분 개시"}.get(judgment, "미개시"),
        note=(match.verify_note if match else "") or _downgrade_note(match),
        # 이 구성의 근거가 실제로 어느 문헌에서 왔는지. 전부 인용발명 1로 보이지 않게 하는 기준값입니다.
        adopted_document=match.document_id if match else "",
        adopted_reference=_reference_number(match.document_id, mappings) if match else None,
        primary_disclosure=_disclosure_line(chain.primary, label, matrix, mappings, documents),
        supplement_disclosure=_supplement_line(label, match, chain, matrix, mappings, documents),
        residual_difference=_residual_line(coverage),
        reference_note=_reference_only_line(coverage, mappings, matrix, documents, chain),
    )


def _disclosure_line(document_id: str | None, label: str, matrix: dict[str, dict[str, ElementMatch]],
                     mappings: list[DocumentMapping], documents: dict[str, Document]) -> str:
    """문헌 1건이 이 구성의 무엇을 개시했는지 한 줄로. 판정과 누락 한정을 함께 적습니다."""
    match = matrix.get(document_id or "", {}).get(label)
    if match is None:
        return f"{_reference_name(document_id, mappings)}: 대응 기재가 확인되지 않았습니다." if document_id else ""
    related = next((span for span in match.evidence
                    if span.verify == "verified" and span.quote), None)
    if match.judgment == "대응 없음" and related is None:
        return f"{_reference_name(document_id, mappings)}: 대응 기재가 확인되지 않았습니다."
    shown = match.quote_translation or match.quote
    quoted = f' "{_clip(shown, 120)}" ({_location(match, documents)})' if shown else ""
    if not shown:
        if related:
            related_shown = related.quote_translation or related.quote
            related_location = _chunk_location(match.document_id, related.chunk_id, documents)
            quoted = f' / 관련 기재 "{_clip(related_shown, 120)}" ({related_location})'
    line = f"{_reference_name(document_id, mappings)}: {match.judgment}{quoted}"
    if match.missing_limitations:
        line += f" / 누락 한정: {'; '.join(match.missing_limitations[:3])}"
    return line


def _supplement_line(label: str, match: ElementMatch | None, chain: ChainInfo,
                     matrix: dict[str, dict[str, ElementMatch]], mappings: list[DocumentMapping],
                     documents: dict[str, Document]) -> str:
    """보완 인용발명이 개시한 부분. 주 인용발명이 그대로 채택된 구성에서는 비어 있습니다."""
    if match is None or not chain.primary or match.document_id == chain.primary:
        return ""
    return _disclosure_line(match.document_id, label, matrix, mappings, documents)


def _residual_line(coverage: ElementCoverage | None) -> str:
    if coverage is None or not coverage.residual_difference:
        return ""
    return "; ".join(coverage.residual_difference[:3])


def _reference_only_line(coverage: ElementCoverage | None, mappings: list[DocumentMapping],
                         matrix: dict[str, dict[str, ElementMatch]],
                         documents: dict[str, Document], chain: ChainInfo | None = None) -> str:
    """미채택 문헌 참고. 결합 문헌으로 표기하지 않고 '참고'로만 남깁니다.

    탈락 사유는 chain이 확정한 실제 사유를 그대로 옮깁니다. 여기서 "문헌 수 제한"이라고
    단정하면, 수 제한에 걸린 적이 없는 경우에도 그 문장이 나갑니다.
    """
    if coverage is None or not coverage.reference_document:
        return ""
    match = matrix.get(coverage.reference_document, {}).get(coverage.label)
    shown = (match.quote_translation or match.quote) if match else ""
    quoted = f' "{_clip(shown, 120)}" ({_location(match, documents)})' if shown and match else ""
    reason = next((dropped.reason for dropped in (chain.dropped_supplements if chain else [])
                   if dropped.document_id == coverage.reference_document),
                  "인용발명 조합에는 넣지 않았습니다. 해당 구성의 대응 근거로만 참고합니다.")
    return (f"{_reference_name(coverage.reference_document, mappings)}(미채택)에 "
            f"{coverage.reference_judgment} 수준의 더 강한 대응{quoted}이 있으나, {reason}")


def _narrative(text: str, match: ElementMatch | None, chain: ChainInfo,
               matrix: dict[str, dict[str, ElementMatch]], mappings: list[DocumentMapping],
               documents: dict[str, Document]) -> str:
    related = (next((span for span in match.evidence
                     if span.verify == "verified" and span.quote), None)
               if match else None)
    if match is None or (match.judgment == "대응 없음" and related is None):
        return f'청구항의 "{_clip(text)}" 구성에 대응되는 기재가 인용발명에서 확인되지 않았습니다.'
    name = _reference_name(match.document_id, mappings)
    sentences: list[str] = []

    # 보조 문헌이 최종 근거라면 주 문헌의 한계를 먼저 밝힙니다. 인용문을 먼저 쓴 뒤
    # 같은 내용을 보완 설명에서 다시 반복하던 종전 순서를 피하기 위한 것입니다.
    if chain.primary and match.document_id != chain.primary:
        primary = matrix.get(chain.primary, {}).get(match.label)
        primary_name = _reference_name(chain.primary, mappings)
        shortfall = (f"{primary.judgment} 판정에 그쳐" if primary and primary.judgment != "대응 없음"
                     else "대응 기재가 없어")
        sentences.append(
            f"{primary_name}만으로는 이 구성이 {shortfall} 완전히 개시되었다고 보기 어렵습니다."
        )
        if primary and primary.missing_limitations:
            sentences.append(f"누락된 한정은 {'; '.join(primary.missing_limitations[:3])}입니다.")

    passages = _supporting_passages(match, documents)
    if passages:
        sentences.append(f"{name}에서 다음 대응 기재를 확인했습니다.")
        for limitations, shown, original, location in passages:
            subject = "; ".join(limitations) if limitations else "대표 대응 근거"
            original_text = f' (원문: "{_clip(original)}")' if original else ""
            sentences.append(f'- {_clip(subject, 180)}: "{_clip(shown)}" ({location}){original_text}')
    else:
        sentences.append(f"{name}에서 원문과 대조된 직접 발췌는 확인되지 않았습니다.")

    if match.missing_limitations:
        sentences.append(f"다만 {'; '.join(match.missing_limitations[:3])} 기재는 확인되지 않았습니다.")
    conclusion = ("관련 기재가 있으나 청구항 한정 전체에 대응하는 개시로 보기 어렵습니다"
                  if match.judgment == "대응 없음" and related else _CONCLUSION[match.judgment])
    sentences.append(f'따라서 청구항의 "{_clip(text)}" 구성과 {conclusion}.')

    return "\n".join(sentence for sentence in sentences if sentence).strip()


def _supporting_passages(match: ElementMatch,
                         documents: dict[str, Document]) -> list[tuple[list[str], str, str, str]]:
    """검증된 하위 한정 발췌를 실제 청구항 구성과 일대일로 묶습니다.

    대표 발췌 하나만 보여 주면, 다른 문장에서 입증된 하위 한정까지 그 한 문장이 뒷받침하는
    것처럼 보입니다. 같은 발췌를 쓴 하위 한정은 한 행으로 합쳐 중복도 피합니다.
    """
    grouped: dict[str, dict] = {}
    for check in match.limitation_checks:
        if not check.disclosed or check.verify != "verified" or not check.quote:
            continue
        key = re.sub(r"\s+", " ", check.quote).strip()
        entry = grouped.setdefault(key, {
            "limitations": [],
            "shown": check.quote_translation or check.quote,
            "original": check.quote if check.quote_translation else "",
            "location": _chunk_location(match.document_id, check.chunk_id, documents),
        })
        if check.limitation and check.limitation not in entry["limitations"]:
            entry["limitations"].append(check.limitation)

    main_key = re.sub(r"\s+", " ", match.quote).strip()
    if match.quote and match.verify == "verified" and main_key not in grouped:
        grouped[main_key] = {
            "limitations": [],
            "shown": match.quote_translation or match.quote,
            "original": match.quote if match.quote_translation else "",
            "location": _location(match, documents),
        }
    # evidence는 누락 한정과 가장 가까운 기재이지 그 한정 전체를 충족하는 근거가 아니다.
    # 보고서에서 버리면 실제로 관련 구조가 있는데도 "원문 발췌 없음"으로 보이고, 반대로
    # 일반 대응 근거처럼 표시하면 과대평가된다. 성격을 명시해 검증된 원문만 별도로 보여 준다.
    for span in match.evidence:
        if span.verify != "verified" or not span.quote:
            continue
        key = re.sub(r"\s+", " ", span.quote).strip()
        grouped.setdefault(key, {
            "limitations": ["관련 기재(청구항 한정 전체를 개시하는 근거는 아님)"],
            "shown": span.quote_translation or span.quote,
            "original": span.quote if span.quote_translation else "",
            "location": _chunk_location(match.document_id, span.chunk_id, documents),
        })
    return [(entry["limitations"], entry["shown"], entry["original"], entry["location"])
            for entry in grouped.values()]


def _chunk_location(document_id: str, chunk_id: str,
                    documents: dict[str, Document]) -> str:
    document = documents.get(document_id)
    if document is not None:
        for chunk in document.chunks:
            if chunk.chunk_id != chunk_id:
                continue
            if chunk.paragraph:
                return f"단락 [{chunk.paragraph}]"
            if chunk.page:
                return f"{chunk.page} 페이지"
    return chunk_id or "출처 미상"


def _reason_sentence(reason: str) -> str:
    """모델의 짧은 판정 이유를 독립적으로 읽히는 한 문장으로 정돈합니다."""
    reason = re.sub(r"\s+", " ", reason).strip().rstrip(". ")
    # 모델이 프롬프트 내 문헌 순서를 "인용발명 1"로 써도 보고서의
    # 확정 매핑 번호와 충돌하지 않도록 판단 이유 앞의 임시 번호를 제거한다.
    reason = re.sub(r"^(?:인용발명|인용문헌|문헌)(?:\s*\d+)?(?:에는|에서|은|는|이|가)\s*",
                    "이는 ", reason)
    if reason.endswith("함"):
        reason = reason[:-1] + "한다는 점을 보여 줍니다"
    elif reason.endswith("됨"):
        reason = reason[:-1] + "된다는 점을 보여 줍니다"
    elif not reason.startswith(("이는 ", "청구항", "해당", "위 ")) and not reason.endswith(
        ("합니다", "됩니다", "있습니다", "없습니다", "어렵습니다", "보여 줍니다")
    ):
        reason = f"그 대응 근거는 {reason}입니다"
    return reason + "."


def _difference(match: ElementMatch | None) -> str | None:
    if match is None:
        return None
    if match.missing_limitations:
        return "; ".join(match.missing_limitations[:3])
    if match.judgment in {"동일", "실질적 동일"} and not match.downgraded_from:
        return None
    if match.judgment == "대응 없음":
        return None
    return "세부 구현·조건에 차이가 있어 동일하다고 보기 어렵습니다."


def _downgrade_note(match: ElementMatch | None) -> str:
    if match and match.downgraded_from:
        return f"발췌 검증 결과 {match.downgraded_from} 판정을 {match.judgment}(으)로 낮췄습니다."
    return ""


def _collect_evidence(label: str, chain: ChainInfo, matrix: dict[str, dict[str, ElementMatch]],
                      documents: dict[str, Document], mappings: list[DocumentMapping]) -> list[Evidence]:
    by_id = {mapping.document_id: mapping for mapping in mappings}
    evidence: list[Evidence] = []
    for document_id in chain_documents(chain):
        match = matrix.get(document_id, {}).get(label)
        if not match:
            continue
        document = documents.get(document_id)
        mapping = by_id.get(document_id)
        spans = []
        if match.quote:
            spans.append((match.chunk_id, match.quote, match.quote_translation, match.verify))
        spans.extend((span.chunk_id, span.quote, span.quote_translation, span.verify)
                     for span in match.evidence if span.quote)
        seen: set[tuple[str, str]] = set()
        for chunk_id, quote, translation, verify in spans:
            key = (chunk_id, quote)
            if key in seen:
                continue
            seen.add(key)
            page, paragraph = _chunk_position(document, chunk_id)
            evidence.append(Evidence(
                document_id=document_id,
                filename=document.filename if document else "",
                reference_number=mapping.reference_number if mapping else None,
                document_number=(mapping.document_number if mapping else "") or None,
                paragraph=paragraph, page=page, chunk_id=chunk_id,
                excerpt=translation or quote,
                original_excerpt=quote if translation else None,
                quality=_QUALITY.get(verify, "UNVERIFIED"),
            ))
    return evidence


def _position(match: ElementMatch, document: Document | None) -> tuple[int | None, str | None]:
    return _chunk_position(document, match.chunk_id)


def _chunk_position(document: Document | None, chunk_id: str) -> tuple[int | None, str | None]:
    if document is None:
        return None, None
    for chunk in document.chunks:
        if chunk.chunk_id == chunk_id:
            return chunk.page, chunk.paragraph
    return None, None


def _location(match: ElementMatch, documents: dict[str, Document]) -> str:
    page, paragraph = _position(match, documents.get(match.document_id))
    if paragraph:
        return f"단락 [{paragraph}]"
    if page:
        return f"{page} 페이지"
    return match.chunk_id or "출처 미상"


def _reference_name(document_id: str | None, mappings: list[DocumentMapping]) -> str:
    for mapping in mappings:
        if mapping.document_id == document_id:
            label = f"인용발명 {mapping.reference_number}"
            detail = mapping.document_number or mapping.filename
            return f"{label} ({detail})" if detail else label
    return f"문헌 {document_id}" if document_id else "주 인용발명"


def _reference_number(document_id: str, mappings: list[DocumentMapping]) -> int | None:
    for mapping in mappings:
        if mapping.document_id == document_id:
            return mapping.reference_number
    return None


def _summary_similarity(results: list[ClaimResult]) -> str:
    """채택된 대응을 구성별 문장으로 반복하지 않고 전체 공통 내용을 한 줄로 요약합니다."""
    if any(result.status == "판정 불가" for result in results):
        return "구성대비 판정을 받지 못해 유사 내용을 요약할 수 없습니다."
    disclosed = [result for result in results if result.status == "개시됨"]
    partial = [result for result in results if result.status == "부분 개시"]
    if not disclosed and not partial:
        return "청구항과 인용발명 사이에 전체적으로 유사한 기술 내용이 확인되지 않았습니다."

    references = list(dict.fromkeys(result.adopted_reference for result in [*disclosed, *partial]
                                    if result.adopted_reference is not None))
    reference_text = (" 및 ".join(f"인용발명 {number}" for number in references)
                      if references else "채택 인용발명")
    sentences: list[str] = []
    if disclosed:
        common_content = _clip(" 및 ".join(dict.fromkeys(_clip(result.claim, 100) for result in disclosed)), 240)
        sentences.append(f"청구항과 {reference_text}는 {common_content}에 관한 기술 내용이 전체적으로 유사합니다.")
    if partial:
        labels = ", ".join(result.label for result in partial)
        sentences.append(f"구성 {labels}에는 일부 대응 기재만 확인되며 완전한 개시로 보지 않습니다.")
    return " ".join(sentences)


def _labels_by_document(results: list[ClaimResult], merged: dict[str, ElementMatch],
                        statuses: set[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for result in results:
        match = merged.get(result.label)
        if result.status in statuses and match is not None:
            grouped.setdefault(match.document_id, []).append(result.label)
    return grouped


def _summary_difference(claim: Claim, chain: ChainInfo, results: list[ClaimResult]) -> str:
    if chain.track == "analysis_incomplete":
        return "구성대비가 완료되지 않아 차이점을 특정할 수 없습니다."
    parts: list[str] = []
    uncovered = [label for label in chain.uncovered
                 if label not in chain.conventional and label not in chain.preamble_undisclosed]
    if uncovered:
        parts.append(f"구성 {', '.join(uncovered)}은 인용발명 결합으로도 청구항 한정 전체의 개시가 "
                     "확인되지 않았습니다.")
    if chain.preamble_undisclosed:
        # 전제부가 한정적인지는 사건마다 다릅니다. 코드가 정하지 않고 판단 지점을 드러냅니다.
        parts.append("전제부에 대응하는 기재는 확인되지 않았습니다. 전제부가 한정적 의미를 갖는지, "
                     "아니면 용도·기술분야를 밝힌 기재인지 확인한 뒤 결론을 정하십시오. "
                     "이 분석은 전제부를 결론 판단에서 제외했습니다.")
    if chain.conventional:
        parts.append(f"구성 {', '.join(chain.conventional)}은 중요도가 낮아 주지관용 검토 대상으로 "
                     "분리했을 뿐, 주지관용임을 뒷받침하는 근거는 확인되지 않았습니다.")
    residual = [label for label in chain.residual if label not in chain.uncovered + chain.conventional]
    if residual:
        parts.append(f"구성 {', '.join(residual)}은 결합 후에도 세부 구현·하위 한정에 차이가 남아 있습니다.")
    elif not parts and [result.label for result in results if result.status == "부분 개시"]:
        partial = [result.label for result in results if result.status == "부분 개시"]
        parts.append(f"구성 {', '.join(partial)}에는 세부 구현상의 차이가 남아 있습니다.")
    if chain.reference_only:
        parts.append(f"구성 {', '.join(chain.reference_only)}은 미채택 문헌에 더 강한 대응이 있어 참고로 남겼습니다.")
    return " ".join(parts)


def _summary(claim: Claim, chain: ChainInfo, results: list[ClaimResult],
             merged: dict[str, ElementMatch], mappings: list[DocumentMapping]) -> str:
    primary = _reference_name(chain.primary, mappings) if chain.primary else "주 인용발명"
    if chain.track == "analysis_incomplete":
        return (f"청구항 {claim.number}은 구성대비 판정을 받지 못한 셀이 있어 신규성·진보성 판단을 "
                "수행하지 않았습니다. 미판정은 '대응 없음'과 다르므로, 이 결과를 인용발명에 해당 "
                "기재가 없다는 뜻으로 읽어서는 안 됩니다. 원인을 해소한 뒤 재실행해야 합니다.")
    if chain.track == "novelty_single":
        return (f"청구항 {claim.number}의 모든 필수 구성이 {primary} 하나에 직접 개시되어 있습니다. "
                "신규성 부정 사유에 해당하는지 검토가 필요하나, 이 분석은 해당 문헌이 대상 청구항의 "
                "우선일 전에 공지·공개되어 특허법 제29조의 선행기술로 적격한지를 확인하지 "
                "않았습니다. 적격성 확인 전에는 결론이 아닙니다.")
    if chain.track == "rejection_impossible":
        pending = (f" 구성 {', '.join(chain.conventional)}은 중요도가 낮아 주지관용 검토 대상이지만,"
                   " 근거가 없으므로 현재 보고서에서는 미개시 상태로 유지합니다."
                   if chain.conventional else "")
        return (f"청구항 {claim.number}은 {primary}을 주 인용발명으로 하더라도 구성 "
                f"{', '.join(chain.uncovered) or '일부'}의 청구항 한정 전체를 충족하는 기재가 어느 "
                "인용발명에서도 확인되지 않아 "
                f"제시된 인용발명만으로는 거절 이유를 구성하기 어렵습니다.{pending}")
    by_primary = _labels_by_document(results, merged, {"개시됨"}).get(chain.primary or "", [])
    partial = [result.label for result in results if result.status == "부분 개시"]
    supplemented = [result.label for result in results
                    if (match := merged.get(result.label)) is not None and match.document_id != chain.primary
                    and result.status == "개시됨"]
    lead = (f"청구항 {claim.number}은 {primary}에 의해 구성 {', '.join(by_primary)}이 개시되어 있고, "
            if by_primary else
            f"청구항 {claim.number}의 " if supplemented else
            f"청구항 {claim.number}은 ")
    if supplemented:
        adopted_documents = []
        for label in supplemented:
            document_id = merged[label].document_id
            if document_id not in adopted_documents:
                adopted_documents.append(document_id)
        secondaries = ", ".join(_reference_name(document_id, mappings) for document_id in adopted_documents)
        qualifier = "나머지 " if by_primary else ""
        body = f"{qualifier}구성 {', '.join(supplemented)}은 {secondaries}에서 확인됩니다. "
    else:
        body = "제시된 인용발명에서 청구항의 각 구성이 확인됩니다. "
    conventional = (f" 남은 구성 {', '.join(chain.conventional)}은 주지관용 기술 여부를 별도로 "
                    "입증해야 하며, 이 분석은 그 근거를 제시하지 않았습니다."
                    if chain.conventional else "")
    partial_note = (f" 구성 {', '.join(partial)}에는 일부 대응 기재만 확인되며, 남은 하위 한정 때문에 "
                    "완전한 개시로 보지 않습니다."
                    if partial else "")
    return f"{lead}{body}{partial_note}{conventional}"


def _clip(text: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# --- 마크다운 ----------------------------------------------------------------

def to_markdown(result: AnalysisResult) -> str:
    lines = ["# 구성대비 분석", ""]
    # 실패는 맨 아래 "검증 참고" 각주가 아니라 첫 화면에서 보여야 합니다. 각주로 밀면
    # 각 청구항 헤더의 확신에 찬 "거절 이유 유형"만 읽히고 실패 사실은 전달되지 않습니다.
    incomplete = [report.claim_number for report in result.reports
                  if report.track == "analysis_incomplete"]
    if incomplete:
        lines += [f"> ⚠️ **구성대비 미완료** — 청구항 "
                  f"{', '.join(str(number) for number in incomplete)}은 판정을 받지 못한 셀이 있어 "
                  "신규성·진보성 판단을 수행하지 않았습니다. 미판정은 '대응 없음'과 다릅니다. "
                  "아래 결과를 거절 가부 판단에 사용하지 마십시오.", ""]
    lines += ["## 문헌 매핑 테이블", "",
             "| 인용발명 | 문헌번호 | 파일명 | 공개·제출일 | 문서 유형 | 단독 적합도 | 역할 |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| 인용발명 {mapping.reference_number} | {mapping.document_number or '-'} | {mapping.filename} "
              f"| {mapping.publication_date or mapping.filing_date or '-'} | {mapping.document_type} | {mapping.main_score:.2f} | {mapping.role} |"
              for mapping in result.claim_mapping]
    for report in result.reports:
        lines += _claim_section(report, result.claim_mapping)
    if result.prior_art:
        lines += ["", "## 미커버 구성 선행기술 검색", ""]
        for hit in result.prior_art:
            lines.append(f"- ({hit.label}) {hit.document_number or hit.title or '문헌 미상'}"
                         + (f" · {hit.published}" if hit.published else "")
                         + (f" — {hit.correspondence}" if hit.correspondence else "")
                         + (f" (남은 차이: {hit.remaining_difference})" if hit.remaining_difference else "")
                         + (f" {hit.url}" if hit.url else ""))
    if result.validation:
        lines += ["", "## 검증 참고", ""] + [f"- {item}" for item in result.validation]
    return "\n".join(lines) + "\n"


def _claim_section(report: ClaimReport, mappings: list[DocumentMapping]) -> list[str]:
    header = f"청구항 {report.claim_number}"
    if report.depends_on:
        header += f" (청구항 {report.depends_on} 종속)"
    # "거절 이유 유형"이라고 쓰면 뒤의 단서를 다 붙여도 헤더만 읽고 확정 결론으로 받아들입니다.
    lines = ["", f"## {header}", "", f"**검토 트랙**: {report.rejection_basis}"]
    if report.track == "analysis_incomplete":
        # 결합 유사도를 찍지 않습니다. 판정이 없는 상태의 0.00%는 "유사하지 않다"로 읽힙니다.
        lines += ["**인용발명 조합**: 판정을 받지 못해 확정하지 않았습니다.", ""]
        lines += ["> ⚠️ 이 청구항은 구성대비가 완료되지 않았습니다. 미판정 사유:"]
        lines += [f"> - {reason}" for reason in report.chain.incomplete_reasons] + [""]
    else:
        lines += [f"**인용발명 조합**: {_chain_text(report.chain, mappings)}",
                  f"**결합 후 구성대비 지표**: {report.chain.combined_similarity:.2f}%"
                  " (법적 결론이 아닌 내부 선정 지표)", ""]
    if report.preamble:
        lines += [f"> {report.preamble}", ""]
    for item in report.claims:
        heading = "전제부" if item.is_preamble else item.label
        lines += ["", f"### ({heading}) {item.claim}", ""]
        if item.similarity is None:
            lines.append(item.narrative)
        else:
            # 판정 라벨의 대표값이지 커버 여부가 아닙니다. 상태를 같은 줄에 붙이지 않으면
            # "87%"와 "미개시"가 한 보고서 안에서 서로를 부정하는 것처럼 읽힙니다.
            lines += [f"판정 대표값: {item.similarity}% {item.emoji} {item.grade} · {item.status}"
                      + (f" · 채택 근거: 인용발명 {item.adopted_reference}" if item.adopted_reference else ""),
                      "", item.narrative.replace("\n", "  \n")]
        # 어느 문헌이 무엇을 개시했는지 구성마다 분리해 적습니다.
        # 주 인용발명이 단독으로 채택된 구성은 위 서술과 겹치므로 역할 줄을 따로 적지 않습니다.
        split_roles = bool(item.supplement_disclosure or item.reference_note or item.residual_difference)
        if item.primary_disclosure and split_roles:
            lines.append(f"- 주 인용발명 개시: {item.primary_disclosure}")
        if item.supplement_disclosure:
            lines.append(f"- 보완 인용발명 개시: {item.supplement_disclosure}")
        if item.residual_difference:
            lines.append(f"- 결합 후 남는 차이: {item.residual_difference}")
        if item.reference_note:
            lines.append(f"- 참고(미채택 문헌): {item.reference_note}")
        if item.difference:
            lines.append(f"→ 차이점: {item.difference}")
        if item.note:
            lines.append(f"※ {item.note}")
    if report.chain.conventional_notes:
        lines += ["", "### 주지관용 검토 대상 (근거 미제시)", "",
                  "아래 구성은 결합 후에도 커버 기준에 미치지 못했으나 중요도가 낮아 분리한 것입니다.",
                  "주지관용 기술로 인정하려면 별도 근거가 필요하며, 이 분석은 그 근거를 찾지 않았습니다.", ""]
        lines += [f"- ({note.label}) 중요도 {note.importance} · 결합 후 판정 {note.judgment}"
                  + (f" · 미개시 한정: {', '.join(note.missing)}" if note.missing else "")
                  + f" — {note.note}"
                  for note in report.chain.conventional_notes]
    if report.chain.dropped_supplements:
        lines += ["", "### 결합에 채택하지 않은 대응", ""]
        lines += [f"- {_reference_name(dropped.document_id, mappings)}: 구성 {', '.join(dropped.labels)} — {dropped.reason}"
                  for dropped in report.chain.dropped_supplements]
    lines += ["", "### 종합 분석 요약", ""]
    if report.summary_similarity:
        lines.append(f"- 유사점: {report.summary_similarity}")
    if report.summary_difference:
        lines.append(f"- 차이점: {report.summary_difference}")
    lines += ["", report.summary]
    return lines


def _chain_text(chain: ChainInfo, mappings: list[DocumentMapping]) -> str:
    names = [_reference_name(document_id, mappings) for document_id in chain_documents(chain)]
    return " + ".join(names) if names else "채택된 인용발명 없음"
