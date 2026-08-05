from typing import Literal
from pydantic import BaseModel, Field

# 판정 어휘. 이 6개 라벨 외에는 파이프라인 어디에서도 쓰지 않습니다.
Judgment = Literal["동일", "실질적 동일", "일부 차이", "일부 유사", "차이", "대응 없음"]
Directness = Literal["direct", "inferred", "absent"]
VerifyStatus = Literal["verified", "partial", "not_found", "empty", "short"]


class JobCreate(BaseModel):
    claims: str = Field(min_length=1)
    analysis_prompt: str = ""


class DependentClaimsAdd(BaseModel):
    claims: str = Field(min_length=1)


class Chunk(BaseModel):
    """비교·검증의 최소 단위. chunk_id로만 근거 위치를 지목합니다."""
    document_id: str
    chunk_id: str                             # D1-P-0012 / D1-B-p003-02 형태
    page: int | None = None
    paragraph: str | None = None              # 특허공보 단락번호 4자리
    section: str = ""                         # 논문 섹션명(참고문헌 제외 판정에 사용)
    text: str


class Document(BaseModel):
    id: str
    filename: str
    type: Literal["patent", "paper", "technical"] = "technical"
    document_number: str = ""                 # US 10,987,654 A1 / 10-2020-0012345호
    title: str = ""
    publication_date: str = ""                # 공개일 또는 논문 최초 제출일(YYYY-MM-DD)
    filing_date: str = ""                     # 특허 출원일(확인되는 경우, YYYY-MM-DD)
    source_file: str = ""                     # 히스토리 안에 보존한 원본 PDF 상대 경로
    chunks: list[Chunk] = []
    ocr_required: bool = False


class ClaimElement(BaseModel):
    label: str                                # (A), (B) … 라벨 원문 유지
    text: str
    importance: int = 3                       # 1~5. LLM 파싱 단계에서 1회만 받습니다.
    is_sub: bool = False                      # 하위 제한(수치·조건) 여부
    is_preamble: bool = False                 # "…에 있어서" 전제부에서 세운 구성
    limitations: list[str] = []               # 독립적으로 입증해야 하는 원자적 하위 제한


class Claim(BaseModel):
    number: int
    preamble: str = ""                        # "…에 있어서" 앞 전제부
    elements: list[ClaimElement] = []
    depends_on: int | None = None             # 종속항이 참조하는 부모 청구항 번호
    raw: str = ""


class EvidenceSpan(BaseModel):
    chunk_id: str = ""
    quote: str = ""                           # 문헌 원문 그대로
    quote_translation: str = ""               # 외국어 문헌의 한국어 번역
    verify: VerifyStatus = "empty"


class LimitationCheck(BaseModel):
    """구성요소의 하위 제한 1개에 대한 직접 근거 점검."""
    index: int
    limitation: str = ""
    # 구성이 하위 제한으로 분해되지 않아 구성 원문 한 줄을 그대로 점검한 경우.
    # 이때 실패는 '누락된 하위 한정'이 아니라 구성 자체의 미개시이므로,
    # missing_limitations에 넣으면 구성 원문이 누락 한정으로 보고서에 찍히고 누락 수도 이중으로 셉니다.
    whole_element: bool = False
    disclosed: bool = False
    chunk_id: str = ""
    quote: str = ""
    quote_translation: str = ""
    verify: VerifyStatus = "empty"


class ElementMatch(BaseModel):
    """(청구항 × 구성요소 × 문헌) 한 칸의 판정. 비교 매트릭스의 셀입니다."""
    claim_number: int
    label: str
    document_id: str
    judgment: Judgment = "대응 없음"
    directness: Directness = "absent"
    reason: str = ""
    quote: str = ""
    quote_translation: str = ""
    chunk_id: str = ""
    missing_limitations: list[str] = []
    limitation_checks: list[LimitationCheck] = []
    evidence: list[EvidenceSpan] = []
    verify: VerifyStatus = "empty"
    verify_note: str = ""
    downgraded_from: str = ""                 # 발췌 검증 실패로 강등된 원 판정
    # 판정을 **받지 못한** 셀. "대응 없음"(받아본 결과 대응이 없었다)과 반드시 구분합니다.
    # 이 값이 차 있으면 그 청구항은 법적 결론을 만들지 않습니다.
    error: str = ""


class DocumentScore(BaseModel):
    document_id: str
    main_score: float = 0.0                   # 0~100 단독 적합도
    detail: dict = {}


class NoveltyScreen(BaseModel):
    selected_document: str | None = None
    complete_documents: list[str] = []
    missing_by_document: dict[str, list[str]] = {}
    result: str = "no_single_document_complete"


class SupplementCandidate(BaseModel):
    """구성 1개에 대한 문헌 1건의 보완 후보 평가. 탈락한 문헌도 사유와 함께 남깁니다."""
    document_id: str
    judgment: str = "대응 없음"
    directness: str = "absent"
    verify: str = "empty"
    has_quote: bool = False
    missing_count: int = 0
    gain: float = 0.0                         # 주 인용발명 대비 보완 이득
    better_than_primary: bool = False
    eligible: bool = False                    # 보완 근거로 쓸 자격이 있는지
    rejected_reason: str = ""                 # 자격 미달 사유
    adopted: bool = False                     # 최종 결합에서 이 구성의 근거로 채택되었는지


class ElementCoverage(BaseModel):
    """구성 1개의 문헌별 대응 전수 분석. 문헌 수 제한을 적용하기 **전**의 결과입니다."""
    label: str
    importance: int = 3
    supplement_needed: bool = False           # 다른 문헌의 더 나은 대응을 검토해야 하는지
    supplement_reason: str = ""
    primary_document: str | None = None       # 주 인용발명이 개시한 부분
    primary_judgment: str = "대응 없음"
    primary_missing: list[str] = []           # 주 인용발명의 누락 한정
    adopted_document: str | None = None       # 결합 후 실제로 채택된 문헌
    adopted_judgment: str = "대응 없음"
    adopted_role: str = "미대응"              # 주 인용발명 / 보조 인용발명 / 미대응
    residual_difference: list[str] = []       # 보완 후에도 남는 차이
    reference_document: str | None = None     # 미채택 문헌 중 더 강한 대응이 있는 경우
    reference_judgment: str = "대응 없음"
    candidates: list[SupplementCandidate] = []


class DroppedSupplement(BaseModel):
    """문헌 수 제한 때문에 결합하지 못한 보완 대응. 정보 자체는 버리지 않습니다."""
    document_id: str
    labels: list[str] = []
    reason: str = ""


class ConventionalNote(BaseModel):
    """주지관용으로 분리한 구성 1개의 근거 상태.

    분류 자체는 중요도만 보고 결정되므로, 무엇이 입증되지 않은 채 남았는지 보고서가
    그대로 드러내야 합니다. 이 기록이 없으면 요약에서 구성이 통째로 사라집니다.
    """
    label: str
    importance: int = 3
    judgment: str = "대응 없음"               # 결합 후 최종 판정
    partial_support: bool = False             # 검증된 부분 대응 근거가 있는지
    missing: list[str] = []                   # 개시가 확인되지 않은 하위 한정
    note: str = ""                            # 무엇을 더 입증해야 하는지


class ChainInfo(BaseModel):
    """한 청구항의 인용발명 조합 확정 결과. 전부 코드로 결정됩니다."""
    claim_number: int
    # analysis_incomplete는 "판정을 받지 못했다"는 뜻이고 나머지 셋과 성격이 다릅니다.
    # rejection_impossible이 "대비했더니 거절 이유가 안 선다"인 반면 이쪽은 대비 자체가 없습니다.
    # 이 둘을 한 트랙으로 묶으면 분석 실패가 출원인에게 유리한 결론으로 읽힙니다.
    track: Literal["novelty_single", "inventive_step_combination", "rejection_impossible",
                   "analysis_incomplete"] = "rejection_impossible"
    incomplete_reasons: list[str] = []        # 판정을 받지 못한 (구성, 문헌) 셀의 사유
    preamble_undisclosed: list[str] = []      # 대응이 확인되지 않은 전제부 라벨(한정 여부는 미판단)
    primary: str | None = None                # document_id
    secondaries: list[str] = []
    inherited: list[str] = []                 # 종속항이 부모항에서 상속한 문헌
    added: str | None = None                  # 종속항이 새로 추가한 문헌(최대 1개)
    uncovered: list[str] = []                 # 결합 후에도 커버 기준에 못 미친 라벨
    conventional: list[str] = []              # 주지관용 검토로 분리한 라벨(근거는 별도 입증 대상)
    conventional_notes: list[ConventionalNote] = []
    supplement_needed: list[str] = []         # 주 인용발명만으로는 불완전해 보완을 검토한 라벨
    residual: list[str] = []                  # 커버는 되었으나 결합 후에도 차이가 남는 라벨
    reference_only: list[str] = []            # 미채택 문헌에 더 강한 대응이 있는 라벨
    element_coverage: list[ElementCoverage] = []
    dropped_supplements: list[DroppedSupplement] = []
    combined_similarity: float = 0.0
    rationale: str = ""
    candidates: list[DocumentScore] = []
    novelty: NoveltyScreen = NoveltyScreen()


class Evidence(BaseModel):
    document_id: str
    filename: str
    reference_number: int | None = None       # 인용발명 N
    document_number: str | None = None
    paragraph: str | None = None
    page: int | None = None
    chunk_id: str = ""
    excerpt: str
    original_excerpt: str | None = None       # 외국어 문헌의 원문 병기
    quality: Literal["HIGH", "MEDIUM", "LOW", "UNVERIFIED"] = "MEDIUM"


class DocumentMapping(BaseModel):
    reference_number: int
    filename: str
    document_type: str = ""
    document_id: str
    document_number: str = ""
    publication_date: str = ""
    filing_date: str = ""
    source_file: str = ""                       # 히스토리에 보존된 원문 PDF 상대 경로
    role: str = ""                            # 주 인용발명 / 부 인용발명 / 미채택
    main_score: float = 0.0


class ClaimResult(BaseModel):
    label: str = ""
    is_preamble: bool = False                 # 전제부에서 세운 구성인지
    claim: str
    similarity: int | None = None             # 판정 라벨의 고정 대표값
    grade: str = ""
    emoji: str = ""
    narrative: str = ""                       # 결정론적으로 조립한 구성대비 서술
    difference: str | None = None
    combination: bool = False
    references: list[str] = []
    evidence: list[Evidence] = []
    status: str = ""
    note: str = ""
    adopted_document: str = ""                # 이 구성의 근거로 최종 채택된 문헌
    adopted_reference: int | None = None      # 그 문헌의 인용발명 번호
    primary_disclosure: str = ""              # 주 인용발명이 개시한 부분
    supplement_disclosure: str = ""           # 보완 인용발명이 개시한 부분
    residual_difference: str = ""             # 결합 후에도 남는 차이
    reference_note: str = ""                  # 미채택 문헌 중 더 강한 대응이 있을 때의 참고


class ClaimReport(BaseModel):
    claim_number: int
    depends_on: int | None = None
    preamble: str = ""
    track: str = ""
    rejection_basis: str = ""                 # 거절 이유 유형 라벨
    chain: ChainInfo
    claims: list[ClaimResult] = []
    summary: str = ""
    summary_similarity: str = ""
    summary_difference: str = ""


class PriorArtHit(BaseModel):
    claim_number: int | None = None
    label: str
    document_number: str = ""
    title: str = ""
    published: str = ""
    correspondence: str = ""
    remaining_difference: str = ""
    url: str = ""


class AnalysisResult(BaseModel):
    job_id: str
    claim_mapping: list[DocumentMapping]
    reports: list[ClaimReport] = []
    preamble: str = ""
    validation: list[str] = []
    prior_art: list[PriorArtHit] = []
    cached_claims: list[int] = []             # 판정 캐시를 재사용한 청구항 번호
