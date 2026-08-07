"""분석 오케스트레이션.

단계는 네 가지입니다.
  1. 청구항 분해 (정규식 + 중요도 LLM 1회, 결과는 재사용을 위해 저장)
  2. 구성요소 × 문헌 전수 비교 (LLM, 셀 단위 동시 실행·캐시됨) → 발췌 검증 (문자열 대조)
  3. 인용발명 선정 (전부 코드)
  4. 보고서 조립 (전부 코드)

LLM은 구성대비 사실만 답하고, 인용발명 조합과 보고서 문장은 전부 코드가 정합니다.
같은 비교 매트릭스에서는 항상 같은 결과가 나옵니다. 그래서 2단계를 동시에 돌리더라도
판정은 완료 순서가 아니라 (청구항, 문헌) 순서로 다시 모읍니다.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import agy, cache
from .agy import AnalysisCancelled
from .chain import build_chain, matrix_for
from .claims import ancestry, assign_importance, input_quality_warnings, parse_claims
from .compare import DEPENDENT_DOCUMENT_BUDGET_CHARS, compare_claims_documents, compare_document
from .config import COMPARE_MAX_WORKERS, load_runtime_settings
from .consistency import enforce_antecedents
from .models import AnalysisResult, ChainInfo, Claim, Document, ElementMatch
from .report import build_claim_report, build_mappings, refresh_mappings
from .verify import verify_matches


def analyze(job_id: str, claims_text: str, documents: list[Document],
            analysis_prompt: str = "", progress=None,
            decomposition: dict | None = None) -> AnalysisResult:
    claims = parse_claims(claims_text)
    if not claims:
        raise RuntimeError("청구항을 인식하지 못했습니다.")
    guideline = (analysis_prompt or "").strip() or load_runtime_settings().get("prompt") or ""
    validation = list(assign_importance(claims, decomposition))
    validation += input_quality_warnings(claims)
    validation += _date_eligibility_warnings(documents)
    by_id = {document.id: document for document in documents}

    matches, cached_claims, compare_warnings = _compare_all(claims, documents, guideline, progress)
    validation += compare_warnings
    # 발췌 검증 기록(위치 자동 복구·판정 강등)은 판정을 추적할 때만 필요한 내부 정보입니다.
    # 보고서 본문에 섞으면 구성대비 결과보다 도구의 동작 로그가 더 길어집니다.
    verify_notes = verify_matches(matches, by_id)

    chains: dict[int, ChainInfo] = {}
    for claim in _processing_order(claims):
        claim_matrix = _claim_matrix(matches, documents, claim.number)
        # 셀은 서로 독립적으로 판정되므로, 청구항 전체로 보면 성립할 수 없는 조합이 남습니다.
        # 선정에 들어가기 전에 문헌 안에서의 지시 관계 모순만 정리합니다.
        verify_notes += enforce_antecedents(claim, claim_matrix)
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
        validation=validation, verify_notes=verify_notes,
        cached_claims=sorted(cached_claims),
    )


def extend_with_dependent_claims(existing: AnalysisResult, claims_text: str,
                                 new_claim_numbers: set[int], documents: list[Document],
                                 analysis_prompt: str = "", progress=None,
                                 decomposition: dict | None = None,
                                 checkpoint=None) -> AnalysisResult:
    """기존 보고서의 인용발명을 재사용해 새 종속항 보고서를 덧붙입니다.

    기존 청구항은 다시 판정하지 않습니다. 새 종속항 중 캐시에 없는 (청구항 × 문헌) 셀만
    일괄 호출로 받고, 온전하게 돌아오지 않은 셀만 단건으로 메웁니다.

    중간에 취소되면 판정이 끝난 항까지만 보고서에 반영하고 checkpoint로 저장한 뒤 취소를
    다시 올립니다. 몇 분치 판정을 통째로 버리지 않기 위한 것입니다.
    """
    all_claims = parse_claims(claims_text)
    new_claims = [claim for claim in all_claims if claim.number in new_claim_numbers]
    if not new_claims:
        raise RuntimeError("추가할 종속항을 인식하지 못했습니다.")
    guideline = (analysis_prompt or "").strip() or load_runtime_settings().get("prompt") or ""
    validation = list(existing.validation)
    validation += assign_importance(new_claims, decomposition)
    validation += input_quality_warnings(new_claims)

    matches: list[ElementMatch] = []
    cancelled: AnalysisCancelled | None = None
    try:
        cached_claims, warnings = _compare_all_batch(
            new_claims, documents, guideline, progress, matches)
    except AnalysisCancelled as exc:
        cancelled, cached_claims, warnings = exc, set(), []
    validation += warnings

    # 모든 인용발명에 대한 판정이 모인 항만 보고서에 올립니다. 절반만 대비된 항을 넣으면
    # 아직 읽지도 않은 문헌을 그 항에 대해 "대응 없음"으로 단정하게 됩니다.
    judged = [claim for claim in new_claims if _fully_judged(matches, documents, claim.number)]
    matches = [match for match in matches
               if match.claim_number in {claim.number for claim in judged}]

    by_id = {document.id: document for document in documents}
    existing.verify_notes = list(existing.verify_notes) + verify_matches(matches, by_id)
    chains = {report.claim_number: report.chain for report in existing.reports}
    new_matrices: dict[int, dict[str, dict[str, ElementMatch]]] = {}
    added: list[Claim] = []
    for claim in _processing_order(judged):
        # 부모항의 조합이 아직 없으면 이 항은 다음 실행으로 미룹니다. 부모 없이 세우면
        # 종속항이 독립항 경로를 타서 추가 한정 하나만으로 신규성이 부정됩니다.
        if claim.depends_on is not None and claim.depends_on not in chains:
            continue
        claim_matrix = _claim_matrix(matches, documents, claim.number)
        existing.verify_notes += enforce_antecedents(claim, claim_matrix)
        new_matrices[claim.number] = claim_matrix
        chains[claim.number] = build_chain(claim, claim_matrix, chains, all_claims)
        added.append(claim)

    existing.claim_mapping = refresh_mappings(existing.claim_mapping, list(chains.values()))
    reports = list(existing.reports) + [
        build_claim_report(claim, chains[claim.number], new_matrices[claim.number],
                           by_id, existing.claim_mapping)
        for claim in added]
    existing.reports = sorted(reports, key=lambda report: report.claim_number)
    existing.validation = validation
    existing.cached_claims = sorted(set(existing.cached_claims) | cached_claims)
    if checkpoint:
        checkpoint(existing)
    if cancelled is not None:
        raise cancelled
    return existing


def _fully_judged(matches: list[ElementMatch], documents: list[Document], claim_number: int) -> bool:
    judged = {match.document_id for match in matches if match.claim_number == claim_number}
    return bool(documents) and all(document.id in judged for document in documents)


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
    cells: dict[tuple[int, str], list[ElementMatch]] = {}
    cached_claims: set[int] = set()
    tasks: list[tuple[Claim, Document]] = []
    for claim in claims:
        claim_cached = bool(documents)
        for document in documents:
            cell = cache.load(cache.cache_key(claim, document, guideline))
            if cell is None:
                claim_cached = False
                tasks.append((claim, document))
            else:
                cells[(claim.number, document.id)] = cell
        if claim_cached:
            cached_claims.add(claim.number)
    if progress and cells:
        progress(f"판정 캐시에서 {len(cells)}셀 재사용")
    warnings = _compare_cells(tasks, guideline, progress, cells)
    # 완료 순서가 아니라 (청구항, 문헌) 순서로 펼칩니다. 동시에 돌려도 매트릭스는 같습니다.
    matches = [match for claim in claims for document in documents
               for match in cells.get((claim.number, document.id), [])]
    return matches, cached_claims, warnings


def _compare_cells(tasks: list[tuple[Claim, Document]], guideline: str, progress,
                   results: dict[tuple[int, str], list[ElementMatch]],
                   budget: int | None = None) -> list[str]:
    """(청구항, 문헌) 셀을 동시에 대비하고 판정을 ``results``에 담습니다.

    셀끼리는 서로를 참조하지 않고 소요 시간의 거의 전부가 CLI 응답 대기라, 동시에 돌리면
    그만큼 줄어듭니다. 호출 횟수도 프롬프트 내용도 그대로이므로 토큰 비용은 달라지지
    않습니다. 다만 동시 요청이 provider 한도에 걸려 실패하면 재시도가 붙으므로, 동시 실행
    수는 COMPARE_MAX_WORKERS로 조절합니다.

    결과를 키로 담아 두면 호출부가 원하는 순서로 펼칠 수 있습니다. 완료 순서에 기대면
    같은 입력에서도 실행마다 매트릭스가 달라집니다.

    취소되면 이미 끝난 셀을 ``results``에 남긴 채 AnalysisCancelled를 올립니다.
    """
    if not tasks:
        return []
    job_id = agy.current_job()
    total = len(tasks)
    done = 0
    lock = threading.Lock()
    cancellations: list[AnalysisCancelled] = []
    warnings_by_cell: dict[tuple[int, str], list[str]] = {}

    def run(claim: Claim, document: Document) -> None:
        nonlocal done
        # 풀의 워커 스레드는 부모의 작업 정보를 물려받지 않습니다. 매어 두지 않으면 취소
        # 신호가 이 스레드에서 띄운 CLI 프로세스에는 닿지 않아 그대로 끝까지 돕니다.
        if job_id:
            agy.bind_job(job_id)
        cell, cell_warnings = compare_document(claim, document, guideline, budget)
        if not cell_warnings:
            cache.store(cache.cache_key(claim, document, guideline), cell)
        with lock:
            results[(claim.number, document.id)] = cell
            warnings_by_cell[(claim.number, document.id)] = cell_warnings
            done += 1
            if progress:
                try:
                    progress(f"구성대비 {done}/{total} — 청구항 {claim.number} × {document.filename}")
                except AnalysisCancelled as exc:
                    # 이 셀의 판정은 이미 끝났습니다. 진행률 보고에서 취소를 알았다고 그
                    # 결과까지 버리면, 방금 받은 판정을 다음 실행에서 또 받게 됩니다.
                    cancellations.append(exc)

    with ThreadPoolExecutor(max_workers=min(COMPARE_MAX_WORKERS, total)) as pool:
        futures = [pool.submit(run, claim, document) for claim, document in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except AnalysisCancelled as exc:
                cancellations.append(exc)

    warnings = [warning for claim, document in tasks
                for warning in warnings_by_cell.get((claim.number, document.id), [])]
    if cancellations:
        raise cancellations[0]
    return warnings


def _compare_all_batch(claims: list[Claim], documents: list[Document], guideline: str,
                       progress, matches: list[ElementMatch]) -> tuple[set[int], list[str]]:
    """캐시 누락 종속항을 한 번에 대비하고, 온전하지 않은 셀만 단건으로 메웁니다.

    판정은 ``matches``에 덧붙여 나갑니다. 반환값으로만 넘기면 중간에 취소되었을 때 이미
    끝난 셀까지 함께 사라져, 다시 눌렀을 때 처음부터 다시 돌게 됩니다.
    """
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
    if not missing_claims or not missing_documents:
        return cached_claims, []

    total = sum(len(values) for values in misses_by_claim.values())
    if progress:
        progress(f"종속항 {len(missing_claims)}개 × 인용발명 {len(missing_documents)}건"
                 f" 일괄 구성대비 (셀 {total}개)")
    batch_cells, warnings = compare_claims_documents(missing_claims, missing_documents, guideline)

    # 일괄 응답에서 온전한 셀은 그대로 채택하고, 빠진 셀만 단건으로 다시 받습니다.
    # 예전에는 셀 하나가 어긋나도 응답 전체를 버리고 전 셀을 단건으로 다시 물었습니다.
    # 셀 수십 개와 하위 제한 점검 수백 줄을 한 응답에 담다 보면 어딘가는 빠지기 마련이라,
    # 일괄 호출은 사실상 늘 헛돈이 되고 시간은 셀 수에 그대로 비례했습니다.
    still_missing: dict[int, list[Document]] = {}
    for claim_number, missing_documents_for_claim in misses_by_claim.items():
        claim = claims_by_number[claim_number]
        for document in missing_documents_for_claim:
            cell = batch_cells.get((claim_number, document.id))
            if cell:
                matches += cell
                cache.store(cache.cache_key(claim, document, guideline), cell)
            else:
                still_missing.setdefault(claim_number, []).append(document)

    remaining = sum(len(values) for values in still_missing.values())
    if progress:
        progress(f"일괄 구성대비에서 {total - remaining}/{total}셀 확보"
                 + (f", 남은 {remaining}셀은 단건으로 대비합니다" if remaining else ""))
    if still_missing:
        tasks = [(claims_by_number[claim_number], document)
                 for claim_number, values in still_missing.items() for document in values]
        cells: dict[tuple[int, str], list[ElementMatch]] = {}
        try:
            warnings += _compare_cells(tasks, guideline, progress, cells,
                                       DEPENDENT_DOCUMENT_BUDGET_CHARS)
        finally:
            # 취소로 중단되더라도 끝난 셀은 넘겨야 그때까지의 항을 보고서에 남길 수 있습니다.
            matches += [match for claim, document in tasks
                        for match in cells.get((claim.number, document.id), [])]
    return cached_claims, warnings


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
                # chain에는 구성별 전 문헌 대응(element_coverage)이 들어 있습니다.
                "chain": report.chain.model_dump(),
                "elements": [
                    {"label": item.label, "grade": item.grade, "corresponded": item.corresponded,
                     "disclosed_limitations": item.disclosed_limitations,
                     "total_limitations": item.total_limitations,
                     "evidence_locations": item.evidence_locations,
                     "status": item.status, "difference": item.difference,
                     "combination": item.combination,
                     "adopted_document": item.adopted_document,
                     "adopted_reference": item.adopted_reference,
                     "evidence": [evidence.model_dump() for evidence in item.evidence]}
                    for item in report.claims
                ],
            }
            for report in result.reports
        ],
        "validation": result.validation,
        "verify_notes": result.verify_notes,
    }
