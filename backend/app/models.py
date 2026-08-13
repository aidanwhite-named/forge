from typing import Literal
from pydantic import BaseModel, Field, field_validator

# 판정 어휘. 이 6개 라벨 외에는 파이프라인 어디에서도 쓰지 않습니다.
Judgment = Literal["동일", "실질적 동일", "일부 차이", "일부 유사", "차이", "대응 없음"]
Directness = Literal["direct", "inferred", "absent"]
VerifyStatus = Literal["verified", "partial", "not_found", "empty", "short"]
EvidenceAlignment = Literal["unverified", "exact", "recovered", "not_found"]
# 의미검증 상태. "rejected"는 **문헌 단독** 심사의 결과이며 결론이 아닙니다. 축 하나가 그
# 문헌에 없을 뿐 같은 조합의 다른 인용발명이 그 축을 댈 수 있고, 그것이 진보성 결합의
# 정의입니다. 그래서 채택 조합이 확정된 뒤 결합 근거 위에서 한 번 더 묻고, 그 결과를
# "…_in_combination"으로 갈라 적습니다 — 어느 단계가 무엇을 판단했는지 섞이지 않게 합니다.
SemanticStatus = Literal["not_run", "accepted", "rejected",
                         "accepted_in_combination", "rejected_in_combination", "error"]
SemanticRelation = Literal["explicit", "necessary_implicit", "functional_equivalent", "unsupported"]


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
    # PDF 텍스트와의 정렬 품질입니다. 의미상 직접성(directness)과 분리해 기록합니다.
    alignment: EvidenceAlignment = "unverified"


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
    verify_note: str = ""                     # 위치 복구·특정 실패 기록. 개시 여부와는 무관합니다.
    alignment: EvidenceAlignment = "unverified"
    # 복합 한정이 여러 문단에 걸쳐 개시되는 경우의 근거 묶음. quote는 대표 발췌로 남깁니다.
    evidence: list[EvidenceSpan] = []
    semantic_status: SemanticStatus = "not_run"
    semantic_relation: SemanticRelation = "unsupported"
    semantic_note: str = ""
    # 결합 심사에서 이 한정의 빠진 축을 실제로 댄 인용발명. accepted_in_combination일 때만
    # 채워지며, 보고서가 "어느 문헌이 무엇을 댔는지"를 지어내지 않고 적을 수 있게 합니다.
    combination_documents: list[str] = []


def missing_limitations(checks: list[LimitationCheck]) -> list[str]:
    """점검 결과에서 실제로 누락된 한정만 추립니다.

    대안 묶음은 하나만 개시되면 충족이므로 나머지 대안은 누락이 아닙니다. 구성 원문 한 줄을
    통째로 점검한 경우(whole_element)의 실패는 누락 '한정'이 아니라 구성 자체의 미개시라서
    목록에 올리지 않습니다.

    비교 단계와 검증 단계가 각자 이 목록을 만들면 규칙이 갈라져, 비교 단계에서 걸러 낸 대안이
    검증 단계에서 되살아나 보고서의 차이점으로 나갑니다. 검증은 check.disclosed를 뒤집을 수
    있으므로 두 단계 모두 이 함수를 같은 입력에 대해 다시 호출합니다.
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
    # judgment는 **모델이 고르지 않습니다.** limitation_checks에서 coverage.derive_judgment가
    # 산출합니다. 모델에게는 아래 두 가지 좁은 질문만 묻고, 그 답과 한정별 개시 여부로
    # 등급이 정해집니다. 라벨을 자유롭게 받으면 같은 근거에서도 실행마다 등급이 흔들리는데,
    # 그 값 하나가 유사도·문헌 순위·신규성 게이트를 전부 좌우합니다.
    judgment: Judgment = "대응 없음"
    # 청구항 문언과 문헌 표기의 관계. 한정이 전부 개시된 경우에만 동일/실질적 동일을 가릅니다.
    terminology: Literal["identical", "equivalent"] = "equivalent"
    # 문헌이 그 구성을 다른 목적으로 사용하는지. 참이면 한정이 전부 개시되어도 '일부 유사'에서
    # 멈춥니다. 판정 라벨 정의의 "문헌이 그 구성을 다른 목적으로 사용함"에 해당하며, 한정별
    # 개시 여부만으로는 표현할 수 없어 따로 받습니다.
    different_purpose: bool = False
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
    alignment: EvidenceAlignment = "unverified"
    downgraded_from: str = ""                 # 발췌 검증 실패로 강등된 원 판정
    # 선행 구성이 같은 문헌에 없어 상한이 걸린 경우의 사유. 보고서의 차이점에 그대로 나갑니다.
    # 이 값이 없으면 "한정은 전부 개시(2/2)인데 등급만 낮은" 결과가 이유 없이 보이게 됩니다.
    antecedent_note: str = ""
    # 구성 간 정합성 상한을 적용하기 직전의 등급. downgraded_from은 의미검증 등 다른 단계의
    # 강등까지 함께 기록하므로, 결합 문헌이 선행 구성을 보완했을 때 정확한 직전 등급으로만
    # 복원하려면 별도 필드가 필요합니다.
    antecedent_capped_from: str = ""
    # 같은 문헌에는 없던 선행 구성을 채택 조합의 다른 문헌이 보완한 경우 그 문헌 ID.
    # 보고서가 두 문헌의 발췌를 한 문장에 함께 제시할 때 사용합니다.
    antecedent_resolved_by: list[str] = []
    # 이 셀이 빠뜨린 하위 한정을 **채택 조합 안의 다른 인용발명**이 원문으로 개시한 경우.
    # 한정 문언 → 그 한정을 댄 문헌 id.
    #
    # 진보성 결합에서 구성 하나의 커버리지는 문헌 하나에서 끝나지 않습니다. 주 인용발명이
    # 구성 전체의 골격을 대고 부 인용발명이 빠진 한정 하나를 대는 것이 결합의 기본형입니다.
    # 그런데 결합 결과를 문헌 단위로만 고르면(chain._merge의 best_match) 진 쪽 셀이 통째로
    # 버려져, 이긴 셀에 남은 누락 한정이 "결합 후에도 남는 차이"로 적힙니다 — 그 한정을
    # 원문으로 개시한 문헌을 같은 조합 안에 세워 두고도 그렇습니다.
    # antecedent_resolved_by와 같은 성격이라 채우는 자리도 같습니다: 결합 결과의 사본에만
    # 기록하고 원본 matrix 셀(감사용 단독 판정)은 건드리지 않습니다.
    combination_resolved: dict[str, str] = {}
    # --- 표본 합의 계측 -------------------------------------------------------
    # COMPARE_SAMPLES를 3으로 두는 근거는 "같은 셀이 실행마다 다른 판정을 낸다"입니다. 그
    # 불안정성을 **실행 뒤에 확인할 수 있어야** 3배 비용이 정당화됩니다. 아래 세 값은
    # judgment.json까지 실려, 표본 수를 몇으로 둘지와 조기 종료가 실제로 얼마나 먹을지를
    # 데이터로 답할 수 있게 합니다.
    sample_count: int = 0                     # 이 셀을 합친 표본 수(0=합의 경로를 타지 않음)
    sample_agreement: float = 0.0             # 전 표본이 같은 disclosed를 낸 한정의 비율(0~1)
    # 앞선 두 표본이 **모든** 한정에서 일치한 셀. 그때는 세 번째 표본이 다수결을 바꿀 수 없어
    # (2표가 이미 과반) 조기 종료 후보가 됩니다. 다만 세 번째 표본은 대표 발췌·근거 묶음을
    # 바꿀 수 있으므로 이 값이 참이라고 결과가 동일하다는 뜻은 아닙니다 — 절감 상한일 뿐입니다.
    sample_early_exit: bool = False
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
    merged_gain: float = 0.0                  # **채택 조합** 대비 증분 이득. 0이면 중복 후보
    better_than_primary: bool = False
    eligible: bool = False                    # 보완 근거로 쓸 자격이 있는지
    rejected_reason: str = ""                 # 자격 미달 사유
    # 채택되지 않은 후보가 왜 빠졌는지. 자격 미달·중복(증분 0)·이득 문턱 미달·결합 상한 중
    # 하나입니다. **비어 있으면 사유 없이 사라진 것**이고, 그때만 불변식 P2가 발화합니다.
    excluded_reason: str = ""
    adopted: bool = False                     # 최종 결합에서 이 구성의 근거로 채택되었는지
    # 표본 합의 계측치(ElementMatch에서 그대로 옮김). 이 행은 (구성 × 문헌) 전수를 담으므로
    # 셀 단위 일치율을 감사 데이터에서 집계할 수 있는 유일한 자리입니다.
    sample_count: int = 0
    sample_agreement: float = 0.0
    sample_early_exit: bool = False


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
    # 종속항이 부모 조합에 새로 더한 문헌. 부모에게서 상속한 문헌은 세지 않고, 종속항에서
    # 새로 더하는 문헌만 MAX_DEPENDENT_ADDITIONS 상한의 대상입니다.
    added: list[str] = []

    @field_validator("added", mode="before")
    @classmethod
    def _accept_single_added(cls, value):
        """문헌을 1건만 추가하던 시절의 기록(문자열·None)도 그대로 읽습니다.

        히스토리의 result.json은 종속항을 뒤에 덧붙일 때 다시 읽히므로, 형이 바뀌면
        예전 분석에 항을 추가하는 순간 검증에 실패합니다.
        """
        if value is None:
            return []
        return [value] if isinstance(value, str) else value
    # 채택하지 않았지만 구성대비 결과는 보고서에 싣는 문헌. 주 인용발명 자격을 갖춘 문헌이
    # 없어 조합 자체를 세우지 못한 경우에만 채워집니다. 역할은 '미채택'으로 남아야 하므로
    # chain_documents()에는 넣지 않습니다. 이 목록이 없으면 보고서가 "조합이 비었으니 볼
    # 것도 없다"로 읽고 본문을 통째로 비워, 원문 대조까지 통과한 대응이 "대응되는 인용발명이
    # 확인되지 않음"으로 나갑니다.
    reference_only: list[str] = []
    uncovered: list[str] = []                 # 채택 조합으로 대응 기재를 찾지 못한 라벨
    # 채택하지 **않은** 문헌에는 검증된 대응 기재가 있는 라벨.
    # uncovered에 함께 들어 있지만 성격이 다릅니다. uncovered 중 이 목록에 없는 것만
    # "어느 인용발명에도 기재가 없다"는 사실 진술이고, 여기 있는 것은 "기재는 있으나 이
    # 거절 이유에는 세우지 않았다"입니다. 구분하지 않으면 손에 든 문헌을 다시 찾게 됩니다.
    # **왜** 채택되지 않았는지는 limit_binding이 따로 답합니다 — 상한이 걸린 것과 보완 후보
    # 평가에서 떨어진 것은 다른 사실이고, 읽는 사람이 취할 후속 조치도 다릅니다.
    beyond_limit: list[str] = []
    beyond_limit_documents: dict[str, list[str]] = {}
    # 같은 문제의 **하위 한정** 판. 구성 전체는 채택 조합에 대응 기재가 있는데 그중 빠진
    # 한정 하나를 한도 밖 문헌이 개시한 경우입니다. beyond_limit이 미대응 줄을 지키는 것처럼
    # 이 값은 차이점 줄을 지킵니다 — 없으면 "길 안내 정보를 제공함"처럼, 업로드된 문헌이
    # 원문으로 개시한 한정이 그냥 남은 차이로 적히고 선행기술 검색 대상까지 됩니다.
    # label → 한정 문언 → 그 한정을 개시한 한도 밖 문헌 id.
    beyond_limit_residual: dict[str, dict[str, list[str]]] = {}
    # 주지관용기술로 다룰 수 있다고 본 라벨과 그 관용성을 실증하는 문헌.
    # 인정 자체는 심사관의 판단이므로 근거 문헌을 함께 남겨 다툴 수 있게 합니다.
    well_known: list[str] = []
    well_known_documents: dict[str, list[str]] = {}
    # 단독 문헌으로는 개시가 확인되지 않았지만, **결합 위에서 다시 물어야** 결론이 나는 라벨.
    #
    # 의미검증(entailment)은 문헌 하나만 놓고 한정을 봅니다. 그래서 "동작은 이 문헌에 있는데
    # 그 동작의 대상이 이 문헌에 없다"는 축 결손이 나오면 그 한정을 개시에서 뺍니다. 문헌
    # 단독 판단으로는 옳습니다. 그러나 진보성 결합에서 빠진 축을 다른 인용발명이 대는 것은
    # 정상이고, 그것이 결합을 세우는 이유 자체입니다. 축 결손을 uncovered로 흘려보내면
    # 보고서가 "어느 인용발명에서도 확인되지 않았다"고 적는데, 그 문헌에는 원문 근거가
    # 있습니다 — 도구가 확인하지 못한 것을 없다고 단정한 진술입니다.
    #
    # 전형적인 형태는 이렇습니다. 어느 인용발명이 청구된 동작을 원문으로 개시하는데 그 동작의
    # 대상만 그 문헌에 없고, 정작 그 대상은 같은 조합의 다른 인용발명이 개시하고 있습니다.
    # 문헌별로만 물으면 앞의 문헌은 미채택, 그 구성은 미대응으로 나갑니다.
    #
    # 그래서 여기 담기는 라벨은 uncovered가 아닙니다. 결론을 확정하지 않고 유보한다는
    # 뜻이고, 사유(어느 축이 왜 빠졌는지)를 그대로 달아 사람이 판단할 수 있게 합니다.
    combination_pending: list[str] = []
    combination_pending_reasons: dict[str, list[str]] = {}
    combination_limit: int = 0                # 이 청구항에 적용한 결합 문헌 수 상한
    # 그 상한이 **실제로 걸렸는지**. beyond_limit·beyond_limit_residual이 "채택하지 않은
    # 문헌에 그 기재가 있다"만 말하고 그 이유는 말하지 않으므로, 보고서가 이유를 지어내지
    # 않으려면 이 값이 필요합니다. 거짓이면 자리가 남아 있는데도 채택되지 않은 것이고
    # (보완 후보 평가에서 탈락), 참일 때만 "상한을 넘어 세우지 않았다"고 쓸 수 있습니다.
    limit_binding: bool = False
    # 채택이 끝난 뒤, 미채택 문헌이 **조합에 더 보탤 수 있는** 문헌 단위 이득.
    #
    # 후보 행의 gain은 주 인용발명 대비입니다. 그것만으로는 "주 인용발명보다는 낫지만 이미
    # 채택된 보조 인용발명이 같은 것을 대고 있는" 문헌과 "아무도 대지 못한 것을 대는데 빠진"
    # 문헌을 구별할 수 없습니다. 불변식 P2가 물어야 하는 것은 뒤쪽 하나뿐인데, 앞쪽까지
    # 위반으로 찍히면 동률 후보가 흔한 만큼 경고가 늘 켜져 진짜 위반이 묻힙니다.
    unadopted_gain: dict[str, float] = {}
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
    # 이 발췌만으로는 한정 문언이 그대로 읽히지 않고 **의미검증이 다리를 놓아** 개시로 인정된
    # 경우의 관계와 그 이유(entailment.validate_entailment). explicit이면 비워 둡니다.
    #
    # 없으면 보고서가 독자를 오도합니다. 의미검증은 발췌 한 문장이 아니라 그 문장이 속한 청크
    # 원문(_source_context)과 형제 한정의 인용문(element_context)까지 함께 읽고 판단하는데,
    # 보고서에 찍히는 것은 짧은 대표 발췌 하나뿐입니다. 그러면 인정의 실제 근거가 보고서에
    # 없는 채로 엉뚱한 발췌만 남아, 왜 개시로 인정됐는지 읽는 사람이 알 수 없습니다.
    # 판단을 감추지 않고 함께 내보내야 심사관이 그 다리를 다툴 수 있습니다.
    semantic_relation: str = ""
    semantic_note: str = ""


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
    corresponded: bool = False                # 대응 기재가 확인된 구성인지
    # 이 구성의 하위 한정 중 원문으로 개시가 확인된 수 / 전체 수.
    #
    # 등급 밴드(90~94 등) 안의 위치를 백분율로 찍지 않습니다. 그 값은 대응된 구성에서 거의
    # 항상 밴드 최댓값이라 등급 이름을 되풀이할 뿐이고, 무엇보다 "%"가 "청구항의 94%가
    # 개시되었다"로 읽히는데 실제 뜻은 그것이 아닙니다. 분자·분모를 그대로 내보내면 독자가
    # 아래 근거 목록과 대조해 검증할 수 있고, 값도 실제로 움직입니다.
    # 대안 묶음("A, B 또는 C 중 적어도 하나")은 하나로 셉니다.
    disclosed_limitations: int = 0
    total_limitations: int = 0
    evidence_locations: int = 0               # 근거로 인용된 서로 다른 원문 위치 수
    grade: str = ""
    emoji: str = ""
    narrative: str = ""                       # 결정론적으로 조립한 구성대비 서술 한 문장
    difference: str | None = None
    # 채택된 셀에 실제로 남은 누락 한정. difference는 이것을 문장으로 옮긴 것이므로, 둘이
    # 어긋나면 보고서가 스스로를 반박합니다(report.report_invariants가 이 값으로 확인합니다).
    # "1/5 개시"라고 적어 놓고 차이점 줄에는 빠진 한정이 한 줄도 없는 상태가 그것입니다.
    missing_limitations: list[str] = []
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
    # 이 청구항의 구성이 어떻게 갈렸는지 한 줄. 결론(신규성·진보성·거절 곤란)을 실제로
    # 정하는 것은 구성별 등급이 아니라 이 집계입니다.
    coverage_summary: str = ""
    summary_similarity: str = ""              # 종합 분석 요약의 유사점 한 줄
    summary_difference: str = ""              # 종합 분석 요약의 차이점 한 줄


# 검색 결과가 실재하는 문헌인지. 파이프라인의 나머지가 발췌를 원문 대조하는 것과 같은
# 이유로, 이 단계의 산출물도 코드가 확인합니다. 확인 실패를 삭제하지는 않고 표시만 합니다 —
# 망이 막혀 있을 수도 있고, 그때 결과를 지우면 검색이 조용히 0건이 됩니다.
PriorArtVerify = Literal["verified", "mismatch", "unreachable", "unchecked"]


class PriorArtHit(BaseModel):
    claim_number: int | None = None
    label: str
    document_number: str = ""
    title: str = ""
    published: str = ""
    correspondence: str = ""
    remaining_difference: str = ""
    url: str = ""
    # verified: URL을 열어 그 페이지에서 문헌번호를 확인함
    # mismatch: 페이지는 열렸으나 문헌번호가 없음 (지어냈을 가능성)
    # unreachable: URL이 없거나 열리지 않음
    verify: PriorArtVerify = "unchecked"
    verify_note: str = ""


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
