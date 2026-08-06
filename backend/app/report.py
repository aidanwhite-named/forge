"""보고서 조립. 문장은 확정된 판정 데이터에서 템플릿으로 만들고 LLM에 다시 묻지 않습니다.

LLM이 쓴 마크다운을 정규식으로 되돌려 고치는 코드가 필요 없어지는 대신,
표현은 템플릿이 허용하는 범위로 제한됩니다. 판정 근거의 재현성을 우선한 선택입니다.

구성 하나는 "유사도 한 줄 + 구성대비 한 문장 + (있으면) 차이점 한 줄"로 나갑니다.
같은 내용을 서술·역할·잔여차이로 나눠 세 번 반복하던 종전 구조는 읽는 사람이 매번
같은 문장을 다시 읽게 만들 뿐, 새로 알려 주는 것이 없어 걷어냈습니다.
"""
import re

from .chain import chain_documents
from .coverage import best_match, report_grade, report_similarity
from .models import (AnalysisResult, ChainInfo, Claim, ClaimReport, ClaimResult, Document,
                     DocumentMapping, ElementCoverage, ElementMatch, Evidence)

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
    selected = chain_documents(chain)
    merged = {element.label: best_match([matrix.get(document_id, {}).get(element.label)
                                         for document_id in selected])
              for element in claim.elements}
    coverages = {coverage.label: coverage for coverage in chain.element_coverage}
    results = [_element_result(claim, element.label, merged.get(element.label), chain, matrix,
                               documents, mappings, coverages.get(element.label))
               for element in claim.elements]
    return ClaimReport(
        claim_number=claim.number,
        depends_on=claim.depends_on,
        preamble=claim.preamble,
        track=chain.track,
        chain=chain,
        claims=results,
        summary_similarity=_summary_similarity(results, mappings),
        summary_difference=_summary_difference(chain, results),
    )


def _element_result(claim: Claim, label: str, match: ElementMatch | None, chain: ChainInfo,
                    matrix: dict[str, dict[str, ElementMatch]], documents: dict[str, Document],
                    mappings: list[DocumentMapping],
                    coverage: ElementCoverage | None = None) -> ClaimResult:
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
    similarity = report_similarity(match)
    grade, emoji = report_grade(match)
    evidence = _collect_evidence(label, chain, matrix, documents, mappings)
    combined = bool(similarity is not None and match and chain.primary
                    and match.document_id != chain.primary and _usable(primary))
    return ClaimResult(
        label=label,
        is_preamble=is_preamble,
        claim=text,                      # 구성 원문은 입력을 그대로 씁니다. 모델이 고쳐 쓴 문장을 쓰지 않습니다.
        similarity=similarity,
        grade=grade,
        emoji=emoji,
        narrative=_narrative(label, text, match, primary, combined, mappings, documents,
                             _closest_related(label, matrix, mappings, documents)),
        difference=_difference(match, primary, combined, mappings, documents),
        combination=combined,
        evidence=evidence,
        status=_STATUS.get(match.judgment, "미개시") if match else "미개시",
        adopted_document=match.document_id if similarity is not None and match else "",
        adopted_reference=(_reference_number(match.document_id, mappings)
                           if similarity is not None and match else None),
    )


# --- 구성대비 서술 -------------------------------------------------------------

def _narrative(label: str, text: str, match: ElementMatch | None, primary: ElementMatch | None,
               combined: bool, mappings: list[DocumentMapping],
               documents: dict[str, Document], related: str = "") -> str:
    """구성 1개의 구성대비를 한 문장으로 조립합니다.

    대응 문헌이 없으면 유사도 없이 미대응 한 줄만 남깁니다. 근거가 약한 대응을 억지로
    문장으로 만들어 섞지 않습니다. 다만 원문 대조를 통과한 인접 기재까지 버리지는 않습니다.
    그것을 감추면 심사관이 이미 확인된 문단을 처음부터 다시 찾게 되고, 어디까지 검토된
    상태인지도 알 수 없게 됩니다.
    """
    if report_similarity(match) is None or match is None:
        line = f"({label}) 구성에 대응되는 인용발명이 확인되지 않음 — 추가 검색 필요"
        return f"{line}\n(가장 가까운 기재: {related} — 청구항 한정 전체를 개시하는 근거는 아님)" if related else line
    passage = _passage(match, mappings, documents)
    if combined and primary is not None:
        missing = "; ".join(primary.missing_limitations[:2]) or "청구항이 요구하는 세부 구성"
        return (f"{_passage(primary, mappings, documents)}는 구성이 기재되어 있으나 {missing}에 대한 "
                f"기재는 없고, {passage}는 구성이 기재되어 있어 이를 결합하면 "
                f'청구항의 "{_clip(text)}" 구성과 대응됩니다.')
    return (f"{passage}는 구성이 기재되어 있으며, {_reason_clause(match.reason)} "
            f'청구항의 "{_clip(text)}" 구성과 대응됩니다.')


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


def _difference(match: ElementMatch | None, primary: ElementMatch | None, combined: bool,
                mappings: list[DocumentMapping], documents: dict[str, Document]) -> str | None:
    if match is None or report_similarity(match) is None:
        return None
    if combined and primary is not None:
        gap = "; ".join(primary.missing_limitations[:2]) or "세부 구성"
        return (f"{_reference_name(primary.document_id, mappings)}은 {gap}에 대한 기재가 없으나 "
                f"{_reference_name(match.document_id, mappings)} ({_location(match, documents)})의 "
                "결합으로 해소됨")
    if match.missing_limitations:
        return "; ".join(match.missing_limitations[:3])
    if match.judgment in {"동일", "실질적 동일"} and not match.downgraded_from:
        return None
    return "세부 구현·조건에 차이가 있어 동일하다고 보기 어렵습니다."


def _closest_related(label: str, matrix: dict[str, dict[str, ElementMatch]],
                     mappings: list[DocumentMapping], documents: dict[str, Document]) -> str:
    """미대응 구성에 대해 원문 대조를 통과한 가장 가까운 기재 하나를 고릅니다.

    문헌 순서가 아니라 인용발명 번호 순으로 훑어, 같은 판정 자료에서는 항상 같은 문장이
    선택되게 합니다.
    """
    for mapping in mappings:
        match = matrix.get(mapping.document_id, {}).get(label)
        if match is None:
            continue
        for span in match.evidence:
            if span.verify != "verified" or not span.quote:
                continue
            shown = _clip(span.quote_translation or span.quote, EXCERPT_LIMIT)
            location = _chunk_location(mapping.document_id, span.chunk_id, documents)
            return f'{_reference_name(mapping.document_id, mappings)} "{shown}" ({location})'
    return ""


def _chunk_location(document_id: str, chunk_id: str, documents: dict[str, Document]) -> str:
    page, paragraph = _chunk_position(documents.get(document_id), chunk_id)
    if paragraph:
        return f"단락 [{paragraph}]"
    if page:
        return f"{page} 페이지"
    return chunk_id or "출처 미상"


def _usable(match: ElementMatch | None) -> bool:
    """결합 문장에 주 인용발명을 함께 세울 수 있는지. 검증된 발췌가 있어야 합니다."""
    return bool(match and match.quote and match.verify in {"verified", "partial"}
                and match.judgment != "대응 없음")


# --- 근거 위치 ----------------------------------------------------------------

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
            if (chunk_id, quote) in seen:
                continue
            seen.add((chunk_id, quote))
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


# --- 종합 분석 요약 -----------------------------------------------------------

def _summary_similarity(results: list[ClaimResult], mappings: list[DocumentMapping]) -> str:
    """출원발명과 인용발명들의 공통된 기술 내용을 한 줄로 정리합니다."""
    if any(result.status == "판정 불가" for result in results):
        return "구성대비 판정을 받지 못해 유사 내용을 요약할 수 없습니다."
    corresponded = [result for result in results if result.similarity is not None]
    if not corresponded:
        return "청구항과 인용발명 사이에 대응되는 기술 내용이 확인되지 않았습니다."
    numbers = sorted({result.adopted_reference for result in corresponded
                      if result.adopted_reference is not None})
    references = ", ".join(f"인용발명 {number}" for number in numbers) or "제시된 인용발명"
    disclosed = [result for result in corresponded if result.status == "개시됨"] or corresponded
    common = _clip(" 및 ".join(dict.fromkeys(_clip(result.claim, 60) for result in disclosed[:3])), 200)
    return f"청구항과 {references}는 {common}에 관한 기술적 목적과 핵심 메커니즘이 공통됩니다."


def _summary_difference(chain: ChainInfo, results: list[ClaimResult]) -> str:
    """구성별 차이점과 겹치지 않는 범위에서 가장 두드러진 차이를 한 줄로 정리합니다."""
    if chain.track == "analysis_incomplete":
        return "구성대비가 완료되지 않아 차이점을 특정할 수 없습니다."
    uncovered = [result.label for result in results
                 if result.similarity is None and not result.is_preamble]
    if uncovered:
        return (f"구성 {', '.join(uncovered)}은 제시된 인용발명 어디에서도 대응 기재가 확인되지 않아 "
                "추가 검색이 필요합니다.")
    residual = [label for label in chain.residual if label not in chain.uncovered]
    if residual:
        return f"구성 {', '.join(residual)}은 결합 후에도 세부 구현·하위 한정에 차이가 남아 있습니다."
    return ""


def _clip(text: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# --- 마크다운 ----------------------------------------------------------------

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
        lines += ["", "## 미커버 구성 선행기술 검색", ""]
        for hit in result.prior_art:
            lines.append(f"- ({hit.label}) {hit.document_number or hit.title or '문헌 미상'}"
                         + (f" · {hit.published}" if hit.published else "")
                         + (f" — {hit.correspondence}" if hit.correspondence else "")
                         + (f" (남은 차이: {hit.remaining_difference})" if hit.remaining_difference else "")
                         + (f" {hit.url}" if hit.url else ""))
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
    if report.preamble:
        lines += [f"> {report.preamble}", ""]
    for item in report.claims:
        heading = "전제부" if item.is_preamble else item.label
        lines += ["", f"### ({heading}) {item.claim}", ""]
        if item.similarity is not None:
            lines.append(f"유사도: {item.similarity}% {item.emoji} {item.grade}")
        lines.append(item.narrative.replace("\n", "  \n"))
        if item.difference:
            lines.append(f"→ 차이점: {item.difference}")
    lines += ["", "### 종합 분석 요약", ""]
    if report.summary_similarity:
        lines.append(f"- 유사점: {report.summary_similarity}")
    if report.summary_difference:
        lines.append(f"- 차이점: {report.summary_difference}")
    return lines


def _chain_text(chain: ChainInfo, mappings: list[DocumentMapping]) -> str:
    names = [_reference_name(document_id, mappings) for document_id in chain_documents(chain)]
    return " + ".join(names) if names else "채택된 인용발명 없음"
