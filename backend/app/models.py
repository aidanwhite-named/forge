from typing import Literal
from pydantic import BaseModel, Field, field_validator

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


class Limitation(BaseModel):
    """구성요소를 이루는 원자적 한정 하나.

    kind가 판정 등급을 가릅니다. core는 그 구성이 실제로 무엇을 하는가(입력·처리·출력·구조)이고,
    qualifier는 그 동작을 한정하는 조건·기준·파라미터·수치입니다. 동작이 개시되어 있고 한정만
    다르면 심사 실무의 "기술 사상은 같고 세부 구현이 다름"이며, 동작 자체가 없으면 대응이
    아닙니다. 둘을 구분하지 않으면 한정 문구가 섞인 모든 조각이 함께 실패해, 대응 문단을
    정확히 찾아 놓고도 구성 전체가 "대응 없음"으로 떨어집니다.
    """
    text: str
    kind: Literal["core", "qualifier"] = "core"
    # 선택적 한정("A, B 또는 C 중 적어도 하나")의 묶음 이름. 같은 이름을 가진 항목은
    # 서로 대안이므로 **하나만 개시되면 그 묶음 전체가 충족**되고, 나머지는 차이가 아닙니다.
    # 빈 값은 단독으로 충족되어야 하는 한정입니다. 이를 구분하지 않으면 선택지를 넉넉히
    # 나열한 청구항일수록 차이점이 길어져, 실제로는 문언을 충족하는 문헌이 감점됩니다.
    alternative_group: str = ""


class ClaimElement(BaseModel):
    label: str                                # (A), (B) … 라벨 원문 유지
    text: str
    importance: int = 3                       # 1~5. LLM 파싱 단계에서 1회만 받습니다.
    is_sub: bool = False                      # 하위 제한(수치·조건) 여부
    is_preamble: bool = False                 # "…에 있어서" 전제부에서 세운 구성
    limitations: list[Limitation] = []        # 독립적으로 입증해야 하는 원자적 하위 제한
    # 이 구성의 대응 기재를 문헌에서 찾기 위한 검색어(원어·번역어). 분해 단계에서 함께 받습니다.
    # 기술분야별 동의어 사전을 코드에 심는 대신 청구항마다 새로 받으므로 분야에 매이지 않습니다.
    search_terms: list[str] = []

    @field_validator("limitations", mode="before")
    @classmethod
    def _accept_plain_limitations(cls, value):
        """문자열 목록으로 저장된 예전 기록도 그대로 읽습니다. 기본 kind는 core입니다."""
        if not isinstance(value, list):
            return value
        return [{"text": item, "kind": "core"} if isinstance(item, str) else item for item in value]


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
    kind: Literal["core", "qualifier"] = "core"
    alternative_group: str = ""               # 같은 값끼리 대안. 하나만 개시되면 묶음 충족
    # 구성이 하위 제한으로 분해되지 않아 구성 원문 한 줄을 그대로 점검한 경우.
    # 이때 실패는 '누락된 하위 한정'이 아니라 구성 자체의 미개시이므로,
    # missing_limitations에 넣으면 구성 원문이 누락 한정으로 보고서에 찍히고 누락 수도 이중으로 셉니다.
    whole_element: bool = False
    disclosed: bool = False
    chunk_id: str = ""
    quote: str = ""
    quote_translation: str = ""
    verify: VerifyStatus = "empty"


def missing_limitations(checks: list[LimitationCheck]) -> list[str]:
    """점검 결과에서 실제로 누락된 한정만 추립니다.

    대안 묶음은 하나만 개시되면 충족이므로 나머지 대안은 누락이 아닙니다. 구성 원문 한 줄을
    통째로 점검한 경우(whole_element)의 실패는 누락 '한정'이 아니라 구성 자체의 미개시라서
    목록에 올리지 않습니다.

    비교 단계와 검증 단계가 각자 이 목록을 만들면 규칙이 갈라집니다. 실제로 검증 단계가
    미개시 항목을 그대로 다시 채워 넣어, 비교 단계에서 걸러 낸 대안이 보고서의 차이점으로
    되살아났습니다. 검증은 check.disclosed를 뒤집을 수 있으므로 두 단계 모두 이 함수를
    같은 입력에 대해 다시 호출합니다.
    """
    satisfied = {check.alternative_group for check in checks
                 if check.disclosed and check.alternative_group}
    missing: list[str] = []
    for check in checks:
        if check.disclosed or check.whole_element or not check.limitation:
            continue
        if check.alternative_group in satisfied or check.limitation in missing:
            continue
        missing.append(check.limitation)
    return missing


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
    """구성 1개의 문헌별 대응 전수 분석. 결합 대상으로 고르기 **전**의 결과입니다."""
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
    candidates: list[SupplementCandidate] = []


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
    uncovered: list[str] = []                 # 결합 후에도 대응 기재를 찾지 못한 라벨
    supplement_needed: list[str] = []         # 주 인용발명만으로는 불완전해 보완을 검토한 라벨
    residual: list[str] = []                  # 커버는 되었으나 결합 후에도 차이가 남는 라벨
    element_coverage: list[ElementCoverage] = []
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
    # 이 발췌가 어느 하위 한정을 개시한 근거로 채택되었는지. 비어 있으면 구성 전체의 대표 발췌입니다.
    # 대표 발췌는 모델이 구성 하나당 한 문장만 고른 것이라 총론·고찰 문장이 뽑히는 일이 잦은데,
    # 보고서에는 그 한 문장만 찍히므로 "이 한정은 무엇으로 개시를 인정했는가"가 남지 않았습니다.
    limitation: str = ""
    kind: str = ""                            # core / qualifier


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
    # 판정 라벨이 정한 등급 밴드 안에서 근거 품질로 위치를 정한 값. 대응이 없으면 None입니다.
    similarity: int | None = None
    grade: str = ""
    emoji: str = ""
    narrative: str = ""                       # 결정론적으로 조립한 구성대비 서술 한 문장
    difference: str | None = None
    combination: bool = False                 # 두 건 이상의 인용발명을 결합해 대응시켰는지
    evidence: list[Evidence] = []
    status: str = ""
    adopted_document: str = ""                # 이 구성의 근거로 최종 채택된 문헌
    adopted_reference: int | None = None      # 그 문헌의 인용발명 번호


class ClaimReport(BaseModel):
    claim_number: int
    depends_on: int | None = None
    preamble: str = ""
    track: str = ""
    chain: ChainInfo
    claims: list[ClaimResult] = []
    # 이 청구항에 어떤 거절 이유가 서는지 한 줄. track과 rationale은 chain.py가 이미
    # 확정해 두고도 보고서에는 한 번도 나오지 않아, 읽는 사람이 구성별 유사도 표에서
    # 신규성 결론인지 진보성 결론인지를 되짚어 추정해야 했습니다.
    conclusion: str = ""
    summary_similarity: str = ""              # 종합 분석 요약의 유사점 한 줄
    summary_difference: str = ""              # 종합 분석 요약의 차이점 한 줄


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
    validation: list[str] = []                # 보고서에 함께 내보내는 사용자용 단서
    # 발췌 검증 과정 기록(인용 위치 자동 복구, 판정 강등). 판정을 되짚을 때만 쓰는
    # 내부 정보라 보고서 본문에는 넣지 않고 감사 데이터로만 남깁니다.
    verify_notes: list[str] = []
    prior_art: list[PriorArtHit] = []
    cached_claims: list[int] = []             # 판정 캐시를 재사용한 청구항 번호
