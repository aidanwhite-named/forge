"""분석 오케스트레이션.

단계는 네 가지입니다.
  1. 청구항 분해 (정규식 + 중요도 LLM 1회, 결과는 재사용을 위해 저장)
  2. 구성요소 × 문헌 전수 비교 (LLM, 셀 단위 동시 실행·캐시됨) → 발췌 검증 (문자열 대조)
     → 원자 한정별 의미검증 (검증된 근거 묶음만 사용·캐시됨)
  3. 인용발명 선정 (전부 코드)
  4. 보고서 조립 (전부 코드)

LLM은 구성대비와 좁은 근거 의미검증만 답하고, 인용발명 조합과 보고서 문장은 전부 코드가 정합니다.
같은 비교 매트릭스에서는 항상 같은 결과가 나옵니다. 그래서 2단계를 동시에 돌리더라도
판정은 완료 순서가 아니라 (청구항, 문헌) 순서로 다시 모읍니다.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import agy, cache, eligibility
from .agy import AnalysisCancelled
from .chain import build_chain, matrix_for
from .claims import ancestry, assign_importance, input_quality_warnings, parse_claims
from .compare import (DEPENDENT_DOCUMENT_BUDGET_CHARS, DOCUMENT_BUDGET_CHARS,
                      batch_budget as compare_batch_budget,
                      compare_claims_documents, compare_document, compare_document_claims,
                      parent_context)
from .config import (COMPARE_CLAIM_BATCH, COMPARE_MAX_WORKERS, COMPARE_SAMPLES,
                     load_runtime_settings)
from .consistency import cross_document_notes, enforce_antecedents
from .entailment import validate_combination, validate_entailment
from .models import AnalysisResult, ChainInfo, Claim, Document, ElementMatch
from .report import (build_claim_report, build_mappings, pipeline_invariants,
                     refresh_mappings, report_invariants)
from .verify import verify_matches


def propose_decomposition(claims_text: str) -> tuple[dict, list[str]]:
    """청구항 분해만 받고 멈춥니다. 구성대비는 시작하지 않습니다.

    **이 함수가 따로 있는 이유는 비용이 아니라 순서입니다.** 분해는 한정 문언·core/qualifier
    배분·검색어를 정하고, 그 셋이 비교 캐시 키와 문헌에서 읽어 올 청크를 좌우합니다
    (cache.cache_key, compare._element_terms). 분해가 확정되기 전에 구성대비를 시작하면
    사람이 고칠 기회를 갖기도 전에 그 분해 위에서 판정이 끝나 있고, 고치는 순간 그 판정은
    전부 버려집니다. 그래서 확정 전에는 **한 셀도 판정하지 않습니다.**

    같은 청구항을 여러 번 분해하면 매번 다른 결과가 나옵니다(실측: 사건 4건 전부에서 한정
    수·문언·검색어가 회차마다 갈림). 그 흔들림을 표본 합의로 눌러 없애는 대신, 사람이 한 번
    보고 확정하게 합니다 — 청구항 원문만 읽으면 되는 일이라 인용문헌을 통독하는 것과는
    부담이 다릅니다.

    돌려주는 dump는 claims.dump_decomposition 형식 그대로라 pinned_decomposition으로 다시
    넣을 수 있습니다. 확정본을 그 자리에 넣으면 우선순위상 공유 캐시와 LLM을 모두 이깁니다.
    """
    claims = parse_claims(claims_text)
    if not claims:
        raise RuntimeError("청구항을 인식하지 못했습니다.")
    decomposition: dict = {}
    warnings = list(assign_importance(claims, decomposition, claims_text))
    warnings += input_quality_warnings(claims)
    return decomposition, warnings


def analyze(job_id: str, claims_text: str, documents: list[Document],
            analysis_prompt: str = "", progress=None,
            decomposition: dict | None = None,
            cache_keys: set[str] | None = None,
            priority_date: str = "",
            pinned_decomposition: dict | None = None) -> AnalysisResult:
    """cache_keys를 주면 이 분석이 사용한 판정 캐시 키를 담아 돌려줍니다.

    캐시 항목에는 문헌 원문 발췌가 들어 있어서, 분석을 지울 때 그 항목도 함께 지워야
    '삭제'가 실제 삭제가 됩니다. 어느 키가 이 분석의 것인지는 여기서만 알 수 있습니다.
    """
    claims = parse_claims(claims_text)
    if not claims:
        raise RuntimeError("청구항을 인식하지 못했습니다.")
    guideline = (analysis_prompt or "").strip() or load_runtime_settings().get("prompt") or ""
    validation = list(assign_importance(
        claims, decomposition, claims_text,
        pinned_decomposition=pinned_decomposition,
    ))
    validation += input_quality_warnings(claims)
    validation += _date_eligibility_warnings(documents, priority_date)
    by_id = {document.id: document for document in documents}

    matches, cached_claims, compare_warnings = _compare_all(claims, documents, guideline,
                                                            progress, cache_keys, claims)
    validation += compare_warnings
    # 발췌 검증 기록(위치 자동 복구·판정 강등)은 판정을 추적할 때만 필요한 내부 정보입니다.
    # 보고서 본문에 섞으면 구성대비 결과보다 도구의 동작 로그가 더 길어집니다.
    verify_notes = verify_matches(matches, by_id)
    # 문자열 대조는 인용문이 PDF에 있다는 사실만 확인합니다. 그 문장이 한정의 입력·동작·출력과
    # 인과관계를 실제로 뒷받침하는지는 좁은 독립 검증으로 다시 확인합니다. 청구항을 함께 넘겨
    # 앞 구성에서 물려받은 지시 대상을 이 구성의 요구사항으로 세지 않게 합니다.
    verify_notes += validate_entailment(matches, by_id, cache_keys, claims)
    # 의미검증은 문헌별로 따로 호출되므로 서로의 판단을 보지 못합니다. 같은 한정이
    # 문헌 간에 갈린 곳을 찾아 남깁니다(되돌리지는 않습니다).
    verify_notes += cross_document_notes(
        matches, {document.id: document.filename for document in documents})

    chains: dict[int, ChainInfo] = {}
    # 매트릭스는 한 번만 만들어 두고 선정·보고서·불변식 검사가 **같은 자료**를 봅니다.
    # 각자 다시 만들면 같은 이름의 매트릭스가 단계마다 다른 값일 수 있습니다.
    matrices = {claim.number: _claim_matrix(matches, documents, claim.number) for claim in claims}
    for claim in _processing_order(claims):
        claim_matrix = matrices[claim.number]
        # 셀은 서로 독립적으로 판정되므로, 청구항 전체로 보면 성립할 수 없는 조합이 남습니다.
        # 선정에 들어가기 전에 문헌 안에서의 지시 관계 모순만 정리합니다.
        verify_notes += enforce_antecedents(claim, claim_matrix)
        chain = build_chain(claim, claim_matrix, chains, claims)
        # 축 결손으로 유보된 구성이 있으면 **채택 조합 전체의 근거로** 한 번 더 묻고 조합을
        # 다시 세웁니다. 의미검증은 문헌별로 호출되어 다른 인용발명이 무엇을 개시했는지 보지
        # 못하는데, 진보성 판단은 결합 위에서 하는 것이라 검증도 결합 위에서 끝나야 합니다.
        # 선정 단계가 유보를 fail-open으로 통과시키는 것과 짝을 이룹니다 — 여기서 확정합니다.
        chain, resolve_notes = _resolve_pending(claim, chain, claim_matrix, chains,
                                                claims, by_id, cache_keys, progress)
        verify_notes += resolve_notes
        chains[claim.number] = chain

    ordered_chains = [chains[claim.number] for claim in claims]
    mappings = build_mappings(documents, ordered_chains)

    reports = [build_claim_report(
        claim, chains[claim.number], matrices[claim.number], by_id, mappings)
        for claim in claims]

    for document in documents:
        if document.ocr_required:
            validation.append(f"{document.filename}에서 텍스트를 거의 추출하지 못했습니다. OCR이 필요할 수 있습니다.")
    # 본문과 결론이 같은 자료를 보고 있는지 조립 직후에 맞춰 봅니다.
    verify_notes += report_invariants(reports)
    # 사건과 무관하게 항상 참이어야 하는 성질도 여기서 함께 봅니다. 회귀 하니스가 아니라
    # **실제 분석 실행**에 붙여 두는 것이 요점입니다 — 회귀는 새 청구항·새 인용발명에서 먼저
    # 나타나고, 그때 하니스는 등록된 옛 사건만 보고 통과합니다.
    verify_notes += pipeline_invariants(reports, matrices)

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
                                 checkpoint=None,
                                 cache_keys: set[str] | None = None) -> AnalysisResult:
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
    validation += assign_importance(
        new_claims, decomposition,
        # 분해 캐시 키는 **이번에 분해하는 항의 원문**이어야 합니다. 전체 청구항
        # 원문을 넣으면 종속항을 하나 더할 때마다 키가 달라져 캐시가 늘 미스입니다.
        "\n".join(claim.raw for claim in new_claims))
    validation += input_quality_warnings(new_claims)

    matches: list[ElementMatch] = []
    cancelled: AnalysisCancelled | None = None
    try:
        cached_claims, warnings = _compare_all_batch(
            new_claims, documents, guideline, progress, matches, cache_keys, all_claims)
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
    existing.verify_notes += validate_entailment(matches, by_id, cache_keys, all_claims)
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


def _resolve_pending(claim: Claim, chain: ChainInfo, matrix: dict, chains: dict[int, ChainInfo],
                     all_claims: list[Claim], by_id: dict[str, Document],
                     cache_keys: set[str] | None, progress) -> tuple[ChainInfo, list[str]]:
    """유보된 구성을 결합 근거로 확정하고, 판정이 바뀌었으면 조합을 다시 세웁니다.

    조합을 알아야 결합 문맥을 만들 수 있으므로 이 단계는 선정 뒤에 옵니다. 그래서 확정 뒤에는
    선정을 한 번 더 돌려야 합니다 — 유보가 인정으로 바뀌면 그 구성은 커버된 것이 되고, 기각으로
    확정되면 진짜 공백이 되어 어느 쪽이든 조합의 근거가 달라지기 때문입니다.

    **다시 세우는 것은 한 번뿐입니다.** 두 번째 선정에서 새로 생긴 유보를 또 확정하려 들면
    조합과 검증이 서로를 바꾸며 도는 상태가 되고, 같은 입력에서 같은 결과가 나온다는 보장이
    사라집니다. 새 유보는 유보인 채로 보고서에 남고, 그 사실은 결론 문구가 그대로 말합니다.
    """
    if chain.track == "analysis_incomplete" or not chain.combination_pending:
        return chain, []
    adopted = [document_id for document_id in [chain.primary, *chain.secondaries] if document_id]
    if not adopted:
        return chain, []
    if progress:
        progress(f"청구항 {claim.number} 결합 근거 확인 — 구성 "
                 f"{', '.join(chain.combination_pending)}")
    notes = validate_combination(claim, list(chain.combination_pending), matrix,
                                 adopted, by_id, cache_keys)
    if not notes:
        return chain, []
    return build_chain(claim, matrix, chains, all_claims), notes


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
                 progress, cache_keys: set[str] | None = None,
                 all_claims: list[Claim] | None = None
                 ) -> tuple[list[ElementMatch], set[int], list[str]]:
    """(청구항 × 문헌) 전수 비교. 캐시가 있으면 CLI를 부르지 않습니다."""
    cells: dict[tuple[int, str], list[ElementMatch]] = {}
    cached_claims: set[int] = set()
    tasks: list[tuple[Claim, Document]] = []
    groups = _claim_groups(claims)
    # 묶음은 캐시를 보기 **전에** 확정합니다. 그래야 조회 시점에 적는 cohort와 나중에 실제로
    # 나가는 프롬프트가 같아집니다.
    cohorts = {claim.number: [item.number for item in group]
               for group in groups for claim in group if len(group) > 1}
    for claim in claims:
        claim_cached = bool(documents)
        parents = parent_context(claim, all_claims)
        for document in documents:
            # 문헌 축 일괄 판정과 단건 판정은 프롬프트가 다르므로 키가 갈립니다. 더 좁게 물어본
            # 단건 쪽을 먼저 찾고, 없으면 일괄 판정을 씁니다.
            keys = [cache.cache_key(claim, document, guideline, DOCUMENT_BUDGET_CHARS, parents,
                                    mode="single", samples=COMPARE_SAMPLES)]
            if claim.number in cohorts:
                keys.append(cache.cache_key(claim, document, guideline, DOCUMENT_BUDGET_CHARS,
                                            parents, mode="document-batch",
                                            samples=COMPARE_SAMPLES,
                                            cohort=cohorts[claim.number]))
            if cache_keys is not None:
                cache_keys.update(keys)
            cell = next((found for key in keys if (found := cache.load(key)) is not None), None)
            if cell is None:
                claim_cached = False
                tasks.append((claim, document))
            else:
                cells[(claim.number, document.id)] = cell
        if claim_cached:
            cached_claims.add(claim.number)
    if progress and cells:
        progress(f"판정 캐시에서 {len(cells)}셀 재사용")
    tasks, warnings = _compare_by_document(tasks, guideline, progress, cells, cache_keys,
                                           all_claims, groups)
    warnings += _compare_cells(tasks, guideline, progress, cells, all_claims=all_claims)
    # 완료 순서가 아니라 (청구항, 문헌) 순서로 펼칩니다. 동시에 돌려도 매트릭스는 같습니다.
    matches = [match for claim in claims for document in documents
               for match in cells.get((claim.number, document.id), [])]
    return matches, cached_claims, warnings


def _claim_groups(claims: list[Claim]) -> list[list[Claim]]:
    """청구항을 **고정 규칙**으로 묶습니다. 캐시 상태를 보지 않습니다.

    '이번에 판정이 없는 청구항'만 모아 묶으면, 같은 분석을 두 번 돌릴 때 캐시가 얼마나 남아
    있느냐에 따라 묶음이 달라집니다. 묶음이 달라지면 프롬프트가 달라지고(형제 청구항이
    컨텍스트에 실립니다), 판정이 실행마다 흔들릴 뿐 아니라 키에 적은 cohort와 실제 프롬프트가
    어긋나 서로 다른 프롬프트의 판정이 한 키를 공유합니다.
    """
    if COMPARE_CLAIM_BATCH <= 1:
        return [[claim] for claim in claims]
    return [claims[index:index + COMPARE_CLAIM_BATCH]
            for index in range(0, len(claims), COMPARE_CLAIM_BATCH)]


def _compare_by_document(tasks: list[tuple[Claim, Document]], guideline: str, progress,
                         cells: dict[tuple[int, str], list[ElementMatch]],
                         cache_keys: set[str] | None,
                         all_claims: list[Claim] | None,
                         groups: list[list[Claim]]
                         ) -> tuple[list[tuple[Claim, Document]], list[str]]:
    """미판정 셀을 **문헌 축**으로 묶어 호출 수와 문헌 반복 전송을 줄입니다.

    같은 문헌이 여러 청구항의 프롬프트에 반복해 실리는 것이 이 파이프라인 토큰 비용의
    대부분입니다. 문헌 하나에 여러 청구항을 함께 실으면 그 반복이 청구항 수만큼 줄어듭니다.

    확보하지 못한 셀은 그대로 돌려주어 호출부가 단건으로 메웁니다. 일괄 응답에서 셀 하나가
    빠졌다고 전부를 버리지 않는 것은 종속항 경로에서 이미 검증된 규율입니다.

    COMPARE_CLAIM_BATCH가 1이면 이 경로는 아무것도 하지 않고 전 작업을 그대로 돌려줍니다.
    """
    if COMPARE_CLAIM_BATCH <= 1 or not tasks:
        return tasks, []
    missing = {(claim.number, document.id) for claim, document in tasks}
    documents = {document.id: document for _, document in tasks}

    warnings: list[str] = []
    resolved: set[tuple[int, str]] = set()
    for group in groups:
        if len(group) < 2:
            # 청구항 하나짜리 묶음은 단건 호출과 같은 비용인데 프롬프트만 다릅니다.
            continue
        cohort = [claim.number for claim in group]
        for document in documents.values():
            # 묶음 안에 아직 판정이 없는 셀이 하나라도 있으면 **묶음 전체**를 한 번에 묻습니다.
            # 이미 받아 둔 셀만 빼고 물으면 프롬프트가 캐시 상태를 따라 달라집니다.
            if not any((number, document.id) in missing for number in cohort):
                continue
            if progress:
                progress(f"{document.filename} × 청구항 "
                         f"{', '.join(str(number) for number in cohort)} 일괄 구성대비")
            found, call_warnings = compare_document_claims(
                group, document, guideline, DOCUMENT_BUDGET_CHARS, all_claims)
            warnings += call_warnings
            for claim in group:
                cell = found.get(claim.number)
                if not cell or (claim.number, document.id) not in missing:
                    continue
                cells[(claim.number, document.id)] = cell
                key = cache.cache_key(claim, document, guideline, DOCUMENT_BUDGET_CHARS,
                                      parent_context(claim, all_claims),
                                      mode="document-batch", samples=COMPARE_SAMPLES,
                                      cohort=cohort)
                if cache_keys is not None:
                    cache_keys.add(key)
                cache.store(key, cell)
                resolved.add((claim.number, document.id))
    leftover = [(claim, document) for claim, document in tasks
                if (claim.number, document.id) not in resolved]
    if progress and leftover:
        progress(f"일괄 구성대비에서 확보하지 못한 {len(leftover)}셀을 단건으로 대비합니다")
    return leftover, warnings


def _compare_cells(tasks: list[tuple[Claim, Document]], guideline: str, progress,
                   results: dict[tuple[int, str], list[ElementMatch]],
                   budget: int | None = None,
                   all_claims: list[Claim] | None = None) -> list[str]:
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
        cell, cell_warnings = compare_document(claim, document, guideline, budget, all_claims)
        if not cell_warnings:
            # 이 경로는 항상 단건 프롬프트 + COMPARE_SAMPLES 표본입니다. 일괄 경로가 쓰는
            # 키와 섞이지 않도록 mode를 함께 적습니다.
            cache.store(cache.cache_key(claim, document, guideline,
                                        budget or DOCUMENT_BUDGET_CHARS,
                                        parent_context(claim, all_claims),
                                        mode="single", samples=COMPARE_SAMPLES), cell)
        with lock:
            results[(claim.number, document.id)] = cell
            warnings_by_cell[(claim.number, document.id)] = cell_warnings
            done += 1
            if progress:
                try:
                    # done/total을 문자열에만 담으면 화면이 진행 바를 그릴 수 없습니다.
                    # 같은 값을 숫자로도 넘겨, 표시 방법은 호출부가 정하게 합니다.
                    progress(f"구성대비 {done}/{total} — 청구항 {claim.number} × {document.filename}",
                             done, total)
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
                       progress, matches: list[ElementMatch],
                       cache_keys: set[str] | None = None,
                       all_claims: list[Claim] | None = None) -> tuple[set[int], list[str]]:
    """캐시 누락 종속항을 한 번에 대비하고, 온전하지 않은 셀만 단건으로 메웁니다.

    판정은 ``matches``에 덧붙여 나갑니다. 반환값으로만 넘기면 중간에 취소되었을 때 이미
    끝난 셀까지 함께 사라져, 다시 눌렀을 때 처음부터 다시 돌게 됩니다.

    이 경로의 셀은 종속항 문맥 예산으로 판정되므로 캐시 키에도 그 값을 넣습니다. 최초 분석
    경로(문헌 전문에 가까운 예산)와 키를 공유하면, 좁은 문맥에서 나온 판정이 넓은 문맥으로
    다시 볼 실행에서 그대로 재사용됩니다.

    캐시는 **두 종류를 따로** 조회합니다. 같은 셀이라도 일괄 프롬프트가 낸 1표본 판정과
    단건 프롬프트가 낸 다표본 합의는 다른 산출물이라 한 키를 공유할 수 없습니다
    (cache.cache_key의 mode). 더 강한 단건 판정을 먼저 찾고, 없을 때만 일괄 판정을 씁니다.
    """
    cached_claims: set[int] = set()
    claims_by_number = {claim.number: claim for claim in claims}
    parents_by_claim = {claim.number: parent_context(claim, all_claims) for claim in claims}

    def single_key(claim: Claim, document: Document) -> str:
        return cache.cache_key(claim, document, guideline, DEPENDENT_DOCUMENT_BUDGET_CHARS,
                               parents_by_claim[claim.number], mode="single",
                               samples=COMPARE_SAMPLES)

    def batch_key(claim: Claim, document: Document, budget: int, cohort: list) -> str:
        # 일괄 호출은 CLI를 1회만 부릅니다(compare_claims_documents). 예산도 문헌 수로 나눈
        # 실제 값을 적고, 그 프롬프트에 함께 실린 청구항·문헌을 cohort로 남깁니다 — 같은 셀이라도
        # 누구와 묶였는지가 다르면 다른 프롬프트입니다.
        return cache.cache_key(claim, document, guideline, budget,
                               parents_by_claim[claim.number], mode="batch", samples=1,
                               cohort=cohort)

    misses_by_claim: dict[int, list[Document]] = {}
    for claim in claims:
        claim_cached = bool(documents)
        for document in documents:
            key = single_key(claim, document)
            if cache_keys is not None:
                cache_keys.add(key)
            cell = cache.load(key)
            if cell is None:
                claim_cached = False
                misses_by_claim.setdefault(claim.number, []).append(document)
            else:
                matches += cell
        if claim_cached:
            cached_claims.add(claim.number)

    missing_claims = [claim for claim in claims if claim.number in misses_by_claim]
    missing_ids = {item.id for values in misses_by_claim.values() for item in values}
    missing_documents = [document for document in documents if document.id in missing_ids]
    if not missing_claims or not missing_documents:
        return cached_claims, []

    # 일괄 예산과 cohort는 이번 호출에 실제로 실리는 청구항·문헌으로 정해지므로, 미판정 셀을
    # 확정한 뒤에야 키를 만들 수 있습니다.
    budget = compare_batch_budget(len(missing_documents))
    cohort = _batch_cohort(missing_claims, missing_documents)
    remaining: dict[int, list[Document]] = {}
    for claim_number, missing_documents_for_claim in misses_by_claim.items():
        claim = claims_by_number[claim_number]
        for document in missing_documents_for_claim:
            key = batch_key(claim, document, budget, cohort)
            if cache_keys is not None:
                cache_keys.add(key)
            cell = cache.load(key)
            if cell is None:
                remaining.setdefault(claim_number, []).append(document)
            else:
                matches += cell
    misses_by_claim = remaining
    if not misses_by_claim:
        return cached_claims, []
    missing_claims = [claim for claim in claims if claim.number in misses_by_claim]
    missing_ids = {item.id for values in misses_by_claim.values() for item in values}
    missing_documents = [document for document in documents if document.id in missing_ids]
    budget = compare_batch_budget(len(missing_documents))
    cohort = _batch_cohort(missing_claims, missing_documents)

    total = sum(len(values) for values in misses_by_claim.values())
    if progress:
        progress(f"종속항 {len(missing_claims)}개 × 인용발명 {len(missing_documents)}건"
                 f" 일괄 구성대비 (셀 {total}개)")
    batch_cells, warnings = compare_claims_documents(missing_claims, missing_documents, guideline,
                                                     all_claims)

    # 일괄 응답에서 온전한 셀은 그대로 채택하고, 빠진 셀만 단건으로 다시 받습니다.
    # 셀 하나가 어긋났다고 응답 전체를 버리면 안 됩니다. 셀 수십 개와 하위 제한 점검 수백
    # 줄을 한 응답에 담다 보면 어딘가는 빠지기 마련이라, 그때마다 전 셀을 다시 물으면 일괄
    # 호출이 늘 헛돈이 되고 시간은 셀 수에 그대로 비례합니다.
    still_missing: dict[int, list[Document]] = {}
    for claim_number, missing_documents_for_claim in misses_by_claim.items():
        claim = claims_by_number[claim_number]
        for document in missing_documents_for_claim:
            cell = batch_cells.get((claim_number, document.id))
            if cell:
                matches += cell
                cache.store(batch_key(claim, document, budget, cohort), cell)
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
                                       DEPENDENT_DOCUMENT_BUDGET_CHARS, all_claims)
        finally:
            # 취소로 중단되더라도 끝난 셀은 넘겨야 그때까지의 항을 보고서에 남길 수 있습니다.
            matches += [match for claim, document in tasks
                        for match in cells.get((claim.number, document.id), [])]
    return cached_claims, warnings


def _batch_cohort(claims: list[Claim], documents: list[Document]) -> list[str]:
    """한 일괄 프롬프트에 함께 실리는 청구항·문헌. 캐시 키에 적어 묶음이 다른 판정을 가릅니다."""
    return ([f"claim:{claim.number}" for claim in claims]
            + [f"doc:{document.id}" for document in documents])


def _processing_order(claims: list[Claim]) -> list[Claim]:
    """독립항을 먼저, 종속항은 부모항이 확정된 뒤에 처리합니다."""
    return sorted(claims, key=lambda claim: (len(ancestry(claims, claim.number)), claim.number))


def _date_eligibility_warnings(documents: list[Document], priority_date: str = "") -> list[str]:
    """날짜만으로 가를 수 있는 것을 갈라 보고서에 남깁니다(eligibility.py).

    추출한 날짜를 한 줄로 **나열만** 하면 후공개 선출원과 통상 선행기술이 같은 칸에 들어가고,
    대상 우선일 이후에 나온 문헌도 주 인용발명으로 뽑힙니다. 다만 자동으로 탈락시키지도
    않습니다 — 적격성은 적용 법역과 신규성·진보성 구분까지 봐야 정해지고, 날짜 추출이
    실패하는 경우도 흔하기 때문입니다.
    """
    return eligibility.warnings(documents, priority_date)


def uncovered_elements(result: AnalysisResult, claims_text: str) -> list[dict]:
    """선행기술 검색 대상. 완전 미대응 구성과 결합 후 남은 하위 한정을 함께 추립니다."""
    by_number = {claim.number: claim for claim in parse_claims(claims_text)}
    targets: list[dict] = []
    for report in result.reports:
        claim = by_number.get(report.claim_number)
        coverage_by_label = {coverage.label: coverage for coverage in report.chain.element_coverage}
        labels = list(dict.fromkeys([*report.chain.uncovered, *report.chain.residual]))
        # 결합 한도 밖 문헌에 이미 대응 기재가 있는 구성과 주지관용으로 다룬 구성은 검색
        # 대상이 아닙니다. 선행기술을 새로 찾을 이유가 없는데도 검색하면, 이미 손에 든
        # 문헌을 두고 웹에서 같은 것을 다시 찾는 일이 됩니다. 축 결손으로 유보한 구성도
        # 같습니다 — 원문 근거는 이미 업로드된 문헌에 있고, 남은 질문은 그 축을 조합의 다른
        # 인용발명이 대는가이지 새 문헌이 있는가가 아닙니다.
        skip = {*report.chain.beyond_limit, *report.chain.well_known,
                *report.chain.combination_pending}
        for label in labels:
            if label in skip:
                continue
            element = next((item for item in (claim.elements if claim else []) if item.label == label), None)
            # 전제부는 한정 여부가 미정이라 검색 대상에서 뺍니다. "…장치에 있어서" 같은 범주
            # 기재로 선행기술을 검색하면 결과가 의미를 갖지 못합니다.
            if element and not element.is_preamble:
                coverage = coverage_by_label.get(label)
                # 한도 밖 문헌이 이미 개시한 한정은 검색어에서 뺍니다. 구성 전체가 그런 경우를
                # skip이 걸러 내는 것과 같은 이유입니다 — 손에 든 문헌에 있는 기재를 웹에서
                # 다시 찾을 이유가 없습니다. 남은 한정이 그것뿐이면 이 구성은 검색하지 않습니다.
                found = report.chain.beyond_limit_residual.get(label, {})
                remaining = [value for value in (coverage.residual_difference if coverage else [])
                             if value and not _generic_residual(value) and value not in found]
                if found and not remaining:
                    continue
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
