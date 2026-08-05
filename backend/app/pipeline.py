"""분석 오케스트레이션.

단계는 네 가지입니다.
  1. 청구항 분해 (정규식 + 중요도 LLM 1회)
  2. 구성요소 × 문헌 전수 비교 (LLM, 캐시됨) → 발췌 검증 (문자열 대조)
  3. 인용발명 선정 (전부 코드)
  4. 보고서 조립 (전부 코드)

LLM은 구성대비 사실만 답하고, 인용발명 조합과 보고서 문장은 전부 코드가 정합니다.
같은 비교 매트릭스에서는 항상 같은 결과가 나옵니다.
"""
from . import cache
from .chain import build_chain, matrix_for
from .claims import ancestry, assign_importance, input_quality_warnings, parse_claims
from .compare import compare_claims_documents, compare_document
from .config import load_runtime_settings
from .models import AnalysisResult, ChainInfo, Claim, Document, ElementMatch
from .report import build_claim_report, build_mappings, refresh_mappings, to_markdown  # noqa: F401  (main에서 재수출)
from .verify import verify_matches


def analyze(job_id: str, claims_text: str, documents: list[Document],
            analysis_prompt: str = "", progress=None) -> AnalysisResult:
    claims = parse_claims(claims_text)
    if not claims:
        raise RuntimeError("청구항을 인식하지 못했습니다.")
    guideline = (analysis_prompt or "").strip() or load_runtime_settings().get("prompt") or ""
    validation = list(assign_importance(claims))
    validation += input_quality_warnings(claims)
    validation += _date_eligibility_warnings(documents)
    by_id = {document.id: document for document in documents}

    matches, cached_claims, compare_warnings = _compare_all(claims, documents, guideline, progress)
    validation += compare_warnings
    validation += verify_matches(matches, by_id)

    chains: dict[int, ChainInfo] = {}
    for claim in _processing_order(claims):
        claim_matrix = _claim_matrix(matches, documents, claim.number)
        chains[claim.number] = build_chain(claim, claim_matrix, chains, claims)

    ordered_chains = [chains[claim.number] for claim in claims]
    mappings = build_mappings(documents, ordered_chains)

    reports = [build_claim_report(
        claim, chains[claim.number],
        _claim_matrix(matches, documents, claim.number),
        by_id, mappings)
        for claim in claims]

    for document in documents:
        if document.ocr_required:
            validation.append(f"{document.filename}에서 텍스트를 거의 추출하지 못했습니다. OCR이 필요할 수 있습니다.")

    return AnalysisResult(
        job_id=job_id, claim_mapping=mappings, reports=reports,
        preamble=claims[0].preamble if claims else "",
        validation=validation, cached_claims=sorted(cached_claims),
    )


def extend_with_dependent_claims(existing: AnalysisResult, claims_text: str,
                                 new_claim_numbers: set[int], documents: list[Document],
                                 analysis_prompt: str = "", progress=None) -> AnalysisResult:
    """기존 보고서의 인용발명을 재사용해 새 종속항 보고서를 덧붙입니다.

    기존 청구항은 다시 판정하지 않습니다. 새 종속항 중 캐시에 없는 모든
    (청구항 × 문헌) 셀만 하나의 일괄 LLM 호출로 받아 항별 보고서로 분리합니다.
    """
    all_claims = parse_claims(claims_text)
    new_claims = [claim for claim in all_claims if claim.number in new_claim_numbers]
    if not new_claims:
        raise RuntimeError("추가할 종속항을 인식하지 못했습니다.")
    guideline = (analysis_prompt or "").strip() or load_runtime_settings().get("prompt") or ""
    validation = list(existing.validation)
    validation += assign_importance(new_claims)
    validation += input_quality_warnings(new_claims)
    matches, cached_claims, warnings = _compare_all_batch(new_claims, documents, guideline, progress)
    validation += warnings
    by_id = {document.id: document for document in documents}
    validation += verify_matches(matches, by_id)
    chains = {report.claim_number: report.chain for report in existing.reports}
    reports = list(existing.reports)
    new_matrices: dict[int, dict[str, dict[str, ElementMatch]]] = {}
    for claim in _processing_order(new_claims):
        claim_matrix = _claim_matrix(matches, documents, claim.number)
        new_matrices[claim.number] = claim_matrix
        chains[claim.number] = build_chain(claim, claim_matrix, chains, all_claims)

    existing.claim_mapping = refresh_mappings(existing.claim_mapping, list(chains.values()))
    reports += [build_claim_report(claim, chains[claim.number], new_matrices[claim.number],
                                   by_id, existing.claim_mapping)
                for claim in new_claims]
    existing.reports = sorted(reports, key=lambda report: report.claim_number)
    existing.validation = validation
    existing.cached_claims = sorted(set(existing.cached_claims) | cached_claims)
    return existing


def _claim_matrix(matches: list[ElementMatch], documents: list[Document],
                  claim_number: int) -> dict[str, dict[str, ElementMatch]]:
    """청구항 번호까지 먼저 거른 뒤 라벨 행렬을 만듭니다.

    여러 청구항이 모두 (A), (B)를 사용하므로 전체 목록을 먼저 label로 인덱싱하면
    뒤 항이 앞 항을 덮어씁니다. 항별 분리가 반드시 인덱싱보다 먼저 이루어져야 합니다.
    """
    matrix = matrix_for([match for match in matches if match.claim_number == claim_number])
    for document in documents:
        matrix.setdefault(document.id, {})
    return matrix


def _compare_all(claims: list[Claim], documents: list[Document], guideline: str,
                 progress) -> tuple[list[ElementMatch], set[int], list[str]]:
    """(청구항 × 문헌) 전수 비교. 캐시가 있으면 CLI를 부르지 않습니다."""
    matches: list[ElementMatch] = []
    cached_claims: set[int] = set()
    warnings: list[str] = []
    total = len(claims) * len(documents)
    done = 0
    for claim in claims:
        claim_cached = bool(documents)
        for document in documents:
            key = cache.cache_key(claim, document, guideline)
            cell = cache.load(key)
            if cell is None:
                claim_cached = False
                cell, cell_warnings = compare_document(claim, document, guideline)
                warnings += cell_warnings
                if not cell_warnings:
                    cache.store(key, cell)
            matches += cell
            done += 1
            if progress:
                progress(f"구성대비 {done}/{total} — 청구항 {claim.number} × {document.filename}")
        if claim_cached:
            cached_claims.add(claim.number)
    return matches, cached_claims, warnings


def _compare_all_batch(claims: list[Claim], documents: list[Document], guideline: str,
                       progress) -> tuple[list[ElementMatch], set[int], list[str]]:
    """캐시 누락 종속항을 모아 단 한 번 비교하고 결과는 기존 셀 캐시에 나눠 저장합니다."""
    matches: list[ElementMatch] = []
    cached_claims: set[int] = set()
    misses_by_claim: dict[int, list[Document]] = {}
    claims_by_number = {claim.number: claim for claim in claims}
    for claim in claims:
        claim_cached = bool(documents)
        for document in documents:
            cell = cache.load(cache.cache_key(claim, document, guideline))
            if cell is None:
                claim_cached = False
                misses_by_claim.setdefault(claim.number, []).append(document)
            else:
                matches += cell
        if claim_cached:
            cached_claims.add(claim.number)

    missing_claims = [claim for claim in claims if claim.number in misses_by_claim]
    missing_documents = [document for document in documents
                         if any(document.id == item.id for values in misses_by_claim.values() for item in values)]
    warnings: list[str] = []
    if missing_claims and missing_documents:
        if progress:
            progress(f"종속항 {len(missing_claims)}개 × 인용발명 {len(missing_documents)}건 일괄 구성대비")
        batch_matches, warnings = compare_claims_documents(missing_claims, missing_documents, guideline)
        # 이미 캐시에 있던 교차 셀은 일괄 응답에 포함되더라도 기존 확정 판정을 우선합니다.
        for claim_number, missing_documents_for_claim in misses_by_claim.items():
            claim = claims_by_number[claim_number]
            for document in missing_documents_for_claim:
                cell = [match for match in batch_matches
                        if match.claim_number == claim_number and match.document_id == document.id]
                if not cell:
                    continue
                matches += cell
                if not warnings:
                    cache.store(cache.cache_key(claim, document, guideline), cell)
    return matches, cached_claims, warnings


def _processing_order(claims: list[Claim]) -> list[Claim]:
    """독립항을 먼저, 종속항은 부모항이 확정된 뒤에 처리합니다."""
    return sorted(claims, key=lambda claim: (len(ancestry(claims, claim.number)), claim.number))


def _date_eligibility_warnings(documents: list[Document]) -> list[str]:
    dated = [f"{document.filename}={document.publication_date or document.filing_date}"
             for document in documents if document.publication_date or document.filing_date]
    detail = f" 확인된 공개·제출일: {', '.join(dated)}." if dated else ""
    return [
        "이 보고서는 기술적 구성대비를 수행합니다. 선행기술 적격성과 적용 조문을 확정하려면 "
        f"대상 청구항의 우선일과 적용 법역을 별도로 확인해야 합니다.{detail}"
    ]


def uncovered_elements(result: AnalysisResult, claims_text: str) -> list[dict]:
    """선행기술 검색 대상. 완전 미대응 구성과 결합 후 남은 하위 한정을 함께 추립니다."""
    by_number = {claim.number: claim for claim in parse_claims(claims_text)}
    targets: list[dict] = []
    for report in result.reports:
        claim = by_number.get(report.claim_number)
        coverage_by_label = {coverage.label: coverage for coverage in report.chain.element_coverage}
        labels = list(dict.fromkeys([*report.chain.uncovered, *report.chain.residual]))
        for label in labels:
            element = next((item for item in (claim.elements if claim else []) if item.label == label), None)
            # 전제부는 한정 여부가 미정이라 검색 대상에서 뺍니다. "…장치에 있어서" 같은 범주
            # 기재로 선행기술을 검색하면 결과가 의미를 갖지 못합니다.
            if element and not element.is_preamble:
                coverage = coverage_by_label.get(label)
                remaining = [value for value in (coverage.residual_difference if coverage else [])
                             if value and not _generic_residual(value)]
                targets.append({
                    "claim_number": report.claim_number,
                    "label": label,
                    # 일부 대응 구성은 이미 개시된 넓은 문언 대신 실제로 남은 제한을 검색합니다.
                    "text": "; ".join(remaining) if remaining else element.text,
                    "claim_context": element.text,
                })
    return targets


def _generic_residual(value: str) -> bool:
    return any(marker in value for marker in (
        "판정에 그쳐", "직접 개시가 아니라", "대응되는 기재가 확인되지 않았습니다"
    ))


def summarize_matrix(result: AnalysisResult) -> dict:
    """판정 추적용 내부 데이터. 별도 감사 리포트는 만들지 않고 JSON으로만 남깁니다."""
    return {
        "job_id": result.job_id,
        "documents": [mapping.model_dump() for mapping in result.claim_mapping],
        "claims": [
            {
                "claim_number": report.claim_number,
                "track": report.track,
                "rejection_basis": report.rejection_basis,
                # chain에는 구성별 전 문헌 대응(element_coverage)이 들어 있습니다.
                "chain": report.chain.model_dump(),
                "elements": [
                    {"label": item.label, "similarity": item.similarity, "grade": item.grade,
                     "status": item.status, "difference": item.difference,
                     "adopted_document": item.adopted_document,
                     "adopted_reference": item.adopted_reference,
                     "residual_difference": item.residual_difference,
                     "reference_note": item.reference_note,
                     "evidence": [evidence.model_dump() for evidence in item.evidence]}
                    for item in report.claims
                ],
            }
            for report in result.reports
        ],
        "validation": result.validation,
    }
