from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

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


class DecompositionConfirm(BaseModel):
    """사용자가 확정한 청구항 분해. claims.dump_decomposition과 같은 형식입니다.

    비워 두면 제안을 그대로 확정한 것으로 봅니다 — 대부분의 실행은 "이대로 확정" 한 번이고,
    그때마다 전체 분해를 되돌려 보내게 하면 화면이 쓸데없이 커집니다.
    """
    decomposition: dict = Field(default_factory=dict)


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


ReferenceQuality = Literal["direct", "confirmed_alias", "fuzzy", "ambiguous"]


class ReferenceAlias(BaseModel):
    """청구항이 같은 대상을 다르게 적은 자리를, **사람이** 같은 것으로 확정한 기록.

    문언이 어긋난 참조를 도구가 자동으로 이어서는 안 됩니다. 실측의 "가시 두상 영역" ↔ "가시
    두상 영상"은 동의어가 아니라 오기일 가능성이 높고, 모델에게 "같은 대상이냐"고 물으면 문맥상
    거의 언제나 "그렇다"고 답합니다. 그 답을 받아 이으면 도구가 청구항의 기재 문제를 대신
    덮어 주는 것이지 해소가 아닙니다 — 출원 중이면 고쳐야 할 기재불비이고, 등록된 권리라면
    해석의 다툼거리입니다. 어느 쪽이든 사람이 판단할 자리입니다.

    그래서 확정은 **분해 확정과 같은 관문**에서 받습니다. 별도 화면을 두면 "무엇을 확정했는지"가
    두 군데로 갈리고, 그렇게 갈린 상태가 이 파이프라인이 되풀이해 겪은 실패 형태입니다.
    """
    target: str                               # "상기 …"를 적은 구성. 이 연결을 쓰는 쪽입니다
    term: str                                 # 뒤 구성이 적은 지시 어구
    # 해소기가 찾은 도입 후보 **전부**. 하나면 문언이 어긋난 참조이고, 둘 이상이면 어느 것을
    # 가리키는지 문언만으로 정할 수 없는 자리입니다. 후보를 버리면 사용자가 고를 것이 사라져
    # 애매한 참조가 "확정하거나 버리거나" 둘 중 하나로 눌립니다.
    candidates: list[str] = []
    # 사용자가 고른 도입 구성. 반드시 candidates 안에 있어야 합니다(claims.validate_aliases).
    selected_source: str = ""
    confirmed: bool = False                   # 사람이 확정했는지. 거짓이면 판정에 쓰지 않습니다

    @property
    def settled(self) -> bool:
        """판정에 반영해도 되는 상태인지. 확정 표시와 고른 후보가 **둘 다** 있어야 합니다."""
        return self.confirmed and self.selected_source in self.candidates


class Claim(BaseModel):
    number: int
    preamble: str = ""                        # "…에 있어서" 앞 전제부
    elements: list[ClaimElement] = []
    # 문언이 어긋난 참조를 사람이 확정한 목록. 분해 확정과 같은 관문에서 받습니다.
    aliases: list[ReferenceAlias] = []
    depends_on: int | None = None             # 종속항이 참조하는 부모 청구항 번호
    raw: str = ""


class EvidenceSpan(BaseModel):
    chunk_id: str = ""
    quote: str = ""                           # 문헌 원문 그대로
    quote_translation: str = ""               # 외국어 문헌의 한국어 번역
    verify: VerifyStatus = "empty"
    # PDF 텍스트와의 정렬 품질입니다. 의미상 직접성(directness)과 분리해 기록합니다.
    alignment: EvidenceAlignment = "unverified"


class Reference(BaseModel):
    """청구항의 "상기 …" 하나가 가리키는 앞 구성과, **그 연결이 얼마나 확실한지**.

    지시 관계는 문언에서 읽어 내는데 청구항의 문언은 흔들립니다. 실측에서 (E)가 세운 "가시
    두상 영역"을 (G)가 "가시 두상 영상"으로 받아 적었고, 어구 전체를 맞춰야 하는 방식에서는
    그 참조가 없는 것이 됐습니다. 공통 부분으로 이으면 그 참조는 살아나지만, 이번에는 아래처럼
    **엉뚱한 구성에 붙는** 연결이 생깁니다.

        (A) 가시 두상 영역을 추출함
        (B) 가시 두상 색상을 산출함
        (C) 상기 가시 두상 영상을 처리함     ← "가시 두상"만으로는 A인지 B인지 모릅니다

    둘 다 지우지 않고 **갈라 둡니다.** 확인된 연결(direct)만 등급 상한의 근거가 되고, 추측
    (fuzzy·ambiguous)은 보고서에 남되 판정을 건드리지 않습니다. 참조를 놓치면 상한이 안 걸릴
    뿐이지만, 잘못 이으면 맞게 개시된 구성이 근거 없이 강등됩니다 — 방향이 더 나쁩니다.
    """
    term: str                                 # 지시 어구. 청구항 문언 그대로
    source: str                               # 그 어구를 도입한 구성 라벨
    quality: ReferenceQuality = "direct"
    # 그 어구를 도입한 구성이 여럿이면 전부. quality가 ambiguous인 연결의 후보들입니다.
    candidates: list[str] = []

    @property
    def confirmed(self) -> bool:
        """판정을 건드려도 되는 연결인지. 추측은 보고만 하고 등급은 건드리지 않습니다.

        문언이 그대로 이어진 연결(direct)과 **사람이 확정한** 별칭(confirmed_alias)만입니다.
        모델이 "같은 대상 같다"고 한 것은 여기 들어오지 않습니다(models.ReferenceAlias).
        """
        return self.quality in ("direct", "confirmed_alias")


class SemanticEvent(BaseModel):
    """검증 단계가 이 한정의 판정을 건드린 기록 한 건. **덮어쓰지 않고 쌓습니다.**

    semantic_status는 단일 필드라 마지막 단계만 남습니다. 실측에서 한정 하나가 의미검증에서
    기각된 뒤 결합검증에서 다시 기각됐는데, 기록에는 뒤엣것만 남아 앞선 기각이 사라졌습니다.
    그러면 "결합 위에서 한 번 봤다"와 "문헌 단독으로 보고 결합 위에서 또 봤다"가 구별되지
    않습니다 — 뒤엣것이 훨씬 강한 판정인데 보고서에서는 같은 무게로 읽힙니다.

    상태(현재 값)와 경위(어떻게 왔는가)는 다른 질문이므로 자리를 따로 둡니다. semantic_status는
    그대로 종단 상태로 남고, 이 목록이 그 상태에 이른 경로입니다.
    """
    stage: Literal["의미검증", "결합검증"]
    outcome: Literal["인정", "기각"]
    note: str = ""
    # 결합검증이 인정한 경우 빠진 축을 실제로 댄 인용발명.
    supplied_by: list[str] = []


# 표본 하나가 이 한정에 대해 낸 답. disclosed/missing 밖의 두 상태를 따로 둡니다 —
# **답하지 않은 것**(absent)과 **답했으나 읽을 수 없는 것**(invalid)은 서로 다르고, 둘 다
# "미개시"가 아닙니다. 셋을 한 칸에 넣으면 응답 결손이 문헌에 대한 사실 주장으로 바뀝니다.
SampleVerdict = Literal["disclosed", "missing", "absent", "invalid"]


class SampleVote(BaseModel):
    sample: int
    verdict: SampleVerdict


class SampleTally(BaseModel):
    """이 한정에 대한 **표본별 원시 답**. 집계는 저장하지 않고 여기서 유도합니다.

    _merge_votes가 다수결로 합치면서 개별 표본의 답을 버렸습니다. 남는 것은 "한정 3개 중
    1개만 만장일치"라는 구성 단위 비율 하나뿐이라, 정작 필요한 "이 한정이 2대 1로 갈렸는가,
    3대 0으로 일치했는가"를 말할 수 없었습니다. 그 차이가 **표본마다 갈린 불안정한 미대응**과
    **전 표본이 일관된 확정적 공백**을 가릅니다.

    집계를 필드로 두지 않는 이유는 원시 투표와 갈릴 수 있기 때문입니다. 유도해 쓰면 갈릴
    자리가 없습니다. total은 물어본 표본 수이고 votes는 그 전부이므로 둘은 항상 같습니다
    (report._sample_tallies_add_up).

    **레거시 기록은 지어내지 않습니다.** 원시 투표를 남기기 전에 저장된 셀은 votes가 비고
    available이 거짓입니다 — 비율에서 되짚어 만들면 있지도 않았던 표본 답이 기록에 생깁니다.
    """
    total: int = 0
    votes: list[SampleVote] = []

    @property
    def available(self) -> bool:
        """표본 기록이 남아 있는지. 거짓이면 이 셀은 원시 투표를 남기기 전에 판정된 것입니다."""
        return self.total > 0 and bool(self.votes)

    def count(self, verdict: SampleVerdict) -> int:
        return sum(1 for vote in self.votes if vote.verdict == verdict)

    @property
    def disclosed(self) -> int:
        return self.count("disclosed")

    @property
    def missing(self) -> int:
        return self.count("missing")

    @property
    def split(self) -> bool:
        """표본이 갈렸는지. 실제로 답한 것들끼리 의견이 나뉘면 참입니다."""
        return self.disclosed > 0 and self.missing > 0

    @property
    def incomplete(self) -> bool:
        """쓸 만한 답을 내지 못한 표본이 있는지(무응답·판독불가).

        표본이 **갈린 것**과 표본이 **답하지 못한 것**은 다릅니다. 앞은 같은 근거를 두고 판단이
        나뉜 것이라 사람이 원문을 봐야 하고, 뒤는 도구가 답을 받지 못한 것이라 다시 물으면
        됩니다. 한 낱말로 뭉뚱그리면 후속 조치가 갈리지 않습니다.
        """
        return self.count("absent") > 0 or self.count("invalid") > 0

    @property
    def unanimous(self) -> bool:
        """전 표본이 같은 답을 냈고 그 답이 쓸 만한지.

        보고서가 이 값으로 적을 것과 적지 않을 것을 가릅니다. **개시·미개시의 대립만 보면
        안 됩니다** — 세 표본 중 하나만 답하고 둘은 무응답·판독불가인 한정은 서로 갈린 것은
        아니지만 만장일치도 아니고, 오히려 그쪽이 더 확인이 필요한 자리입니다.
        """
        verdicts = {vote.verdict for vote in self.votes}
        return self.available and len(verdicts) == 1 and verdicts <= {"disclosed", "missing"}


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
    # 이 한정을 건드린 검증 단계의 **전체** 기록. semantic_status가 종단 상태만 남기는 것과
    # 달리 덮어쓰지 않고 쌓으므로, 한 한정이 두 단계에 걸쳐 판정된 경우가 그대로 남습니다.
    semantic_events: list[SemanticEvent] = []
    # 이 한정에 대한 표본별 원시 답. 관측 전용이며 판정에 쓰지 않습니다.
    sample_tally: SampleTally = SampleTally()
    # 결합 심사에서 이 한정의 빠진 축을 실제로 댄 인용발명. accepted_in_combination일 때만
    # 채워지며, 보고서가 "어느 문헌이 무엇을 댔는지"를 지어내지 않고 적을 수 있게 합니다.
    combination_documents: list[str] = []

    def record(self, event: SemanticEvent) -> None:
        """검증 이벤트를 쌓습니다. 같은 판정이 두 번 적용되어도 한 번만 남습니다.

        지금 파이프라인은 한 실행에서 각 단계를 한 번씩만 돌리지만, 그것은 호출부의 성질이지
        이 자료구조의 성질이 아닙니다. 쌓기만 하는 목록은 재적용에 취약하고, 중복이 들어가면
        보고서가 "두 번 기각됐다"고 적습니다 — 없는 심사를 지어내는 방향의 오류입니다.

        멱등을 여기서 보장해 두면 호출부가 늘어나도 그 성질이 유지됩니다. 단계나 사유가 다른
        이벤트는 다른 판정이므로 그대로 쌓입니다.
        """
        if self.semantic_events and self.semantic_events[-1] == event:
            return
        self.semantic_events.append(event)


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
    # 그 비율의 분자·분모. 비율만으로는 보고서에 사실대로 적을 수 없습니다 — 0.33이 "세 표본이
    # 전부 갈렸다"인지 "한정 3개 중 1개만 만장일치였다"인지가 값에서 읽히지 않아, 실제로 두
    # 사람이 같은 숫자를 정반대로 읽었습니다. 분자·분모를 그대로 실으면 그 오독이 불가능합니다.
    sample_unanimous: int = 0                 # 전 표본이 일치한 한정 수
    sample_requirements: int = 0              # 표본 일치를 물은 한정 수(비율의 분모)
    # 앞선 두 표본이 **모든** 한정에서 일치한 셀. 그때는 세 번째 표본이 다수결을 바꿀 수 없어
    # (2표가 이미 과반) 조기 종료 후보가 됩니다. 다만 세 번째 표본은 대표 발췌·근거 묶음을
    # 바꿀 수 있으므로 이 값이 참이라고 결과가 동일하다는 뜻은 아닙니다 — 절감 상한일 뿐입니다.
    sample_early_exit: bool = False
    # 판정을 **받지 못한** 셀. "대응 없음"(받아본 결과 대응이 없었다)과 반드시 구분합니다.
    # 이 값이 차 있으면 그 청구항은 법적 결론을 만들지 않습니다.
    error: str = ""

    @model_validator(mode="after")
    def _backfill_sample_counts(self) -> "ElementMatch":
        """분자·분모가 없는 옛 기록을 비율에서 되짚습니다.

        캐시 세대는 **프롬프트 문면**에서 나오므로(cache.compare_generation) 모델에 필드를
        더해도 키가 갈리지 않습니다. 옛 캐시 항목은 그대로 로드되고 새 필드만 0으로 남습니다.
        그러면 캐시 히트한 셀에서만 "표본이 갈렸다"는 표시가 조용히 사라져, 보고서가 셀마다
        다른 말을 하게 됩니다 — 같은 불안정성이 새로 판정한 셀에서는 보이고 캐시에서 온 셀에서는
        안 보이는 상태가 가장 나쁩니다.

        세대를 올려 캐시를 통째로 버리는 선택지도 있지만, 관측용 필드 하나 때문에 (구성 ×
        문헌) 판정을 전량 다시 사는 것은 값이 맞지 않습니다. 비율과 한정 수가 남아 있으므로
        분자·분모는 손실 없이 복원됩니다.

        새 기록은 _merge_votes가 두 값을 함께 쓰므로 이 경로를 타지 않습니다. 한정이 하나도
        없는 셀은 비율 자체가 정의되지 않으므로 건드리지 않습니다.
        """
        if self.sample_count > 0 and not self.sample_requirements and self.limitation_checks:
            self.sample_requirements = len(self.limitation_checks)
            self.sample_unanimous = round(self.sample_agreement * self.sample_requirements)
        return self


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
    # 의미검증을 받지 못한 한정 수. 누락 수와 따로 둡니다 — 이 행은 (구성 × 문헌) 전수라,
    # 회귀 하니스가 "문헌 단독 셀이 무엇에 근거해 그 등급을 받았는가"를 읽는 유일한 자리입니다.
    unverified_count: int = 0
    # 한정 문언 → 상태(disclosed / unverified / missing).
    #
    # 개수만으로는 **어느** 한정이 걸렸는지 알 수 없어, 등급만 고정하는 회귀는 오판을 놓칩니다.
    # 검증기가 문제의 한정은 계속 인정하면서 엉뚱한 한정을 기각해도 등급은 똑같이 내려가고,
    # 그러면 기대값이 통과합니다. 한정 단위로 기대를 걸 수 있어야 그 자리를 고정합니다.
    limitation_states: dict[str, str] = {}
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
    sample_unanimous: int = 0
    sample_requirements: int = 0
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
    # 확인된 개시가 하나도 없이 미검증만 남아 판정을 유보한 구성. residual과 **따로** 둡니다 —
    # 차이는 대비해 본 결과이고 유보는 대비 자체를 못 한 것이라, 후속 조치가 다릅니다
    # (앞은 보완 문헌 검색, 뒤는 원문 확인과 재실행).
    reserved: list[str] = []
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
    # 이 발췌를 낸 한정의 의미검증이 **수행되지 못한** 경우(응답 결손). 개시 판정은 그대로
    # 두고 사실만 표시합니다.
    #
    # 표시가 없으면 보고서의 신호가 뒤집힙니다. 의미검증을 통과한 근거에는 "발췌 문언
    # 그대로는 아니며…" 단서가 붙으므로, 검증을 **받지 못한** 근거는 아무 표시가 없어
    # 원문 그대로의 가장 튼튼한 근거로 읽힙니다. 실측에서 🟢을 받은 구성의 근거가 전부
    # 그 상태였습니다.
    verification_incomplete: bool = False


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


class TrailStep(BaseModel):
    """한정 하나의 개시 판정이 **초기 비교 이후에 움직인** 기록 한 줄."""
    index: int
    limitation: str = ""
    stage: Literal["의미검증", "결합검증"] = "의미검증"
    outcome: Literal["기각", "인정"] = "기각"
    note: str = ""
    # 결합검증이 인정한 경우 빠진 축을 실제로 댄 인용발명. 지어내지 않고 그대로 적습니다.
    combination_documents: list[str] = []


class VerificationTrail(BaseModel):
    """구성 하나가 **어떤 단계를 거쳐** 지금 판정이 되었는지. 문헌별로 한 건.

    이 값이 없으면 보고서는 결과만 남기고 경위를 버립니다. 실제로 그랬습니다 — 어떤 구성의
    한정 하나가 초기 비교에서 개시로 나왔다가 의미검증에서 근거 불일치로 기각되고 결합검증에서
    다시 기각된 사건이, 보고서에는 "대응되는 인용발명이 확인되지 않음" 한 줄로만 남았습니다.
    그 한 줄만 읽은 사람은 도구가 그 구성을 **검토하지 않았다**고 읽습니다. 실제로는 세 번
    검토했고 두 번은 근거를 들어 기각한 것인데, 판단의 강도가 보고서에서 사라진 것입니다.

    경위는 이미 LimitationCheck.semantic_status/semantic_note에 구조화되어 있었습니다.
    보고서가 accepted만 렌더링하고 rejected·*_in_combination을 읽지 않았을 뿐입니다.
    """
    document_id: str = ""
    reference_number: int | None = None
    # 표본 합의 계측치. 갈린 셀에서만 보고서에 나갑니다(만장일치 셀은 적을 것이 없습니다).
    sample_count: int = 0
    sample_unanimous: int = 0
    sample_requirements: int = 0
    steps: list[TrailStep] = []
    # 표가 갈린 한정의 (번호, 문언, 표본별 답). 구성 단위 비율만으로는 어느 한정이 몇 대 몇으로
    # 갈렸는지 말할 수 없어, 갈린 자리를 짚어 재실행하거나 사람이 확인할 수가 없습니다.
    tallies: list[tuple[int, str, SampleTally]] = []

    @property
    def split(self) -> bool:
        """표본이 갈린 셀인지. 물어본 한정이 있고 그중 만장일치가 아닌 것이 있을 때."""
        return self.sample_requirements > 0 and self.sample_unanimous < self.sample_requirements


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
    # 의미검증을 수행하지 못한 한정. missing_limitations와 **반드시 따로** 둡니다 — 앞의
    # 것은 "이 문헌에 없다"는 문헌에 대한 사실 주장이고, 이것은 "확인하지 못했다"는 도구의
    # 상태입니다. 한 목록에 섞으면 검증기 결손이 문헌의 결손으로 보고서에 적힙니다.
    unverified_limitations: list[str] = []
    # 미완료 한정 **수**. 목록과 따로 두는 이유는 하위 한정으로 분해되지 않은 점검
    # (whole_element)이 목록에서 빠지기 때문입니다. 지표 줄의 산술("개시 확인 = 개시 - 미완료")이
    # 목록 길이에 기대면 그 경우에 숫자가 어긋납니다.
    unverified_limitation_count: int = 0
    combination: bool = False                 # 두 건 이상의 인용발명을 결합해 대응시켰는지
    evidence: list[Evidence] = []
    status: str = ""
    adopted_document: str = ""                # 이 구성의 근거로 최종 채택된 문헌
    adopted_reference: int | None = None      # 그 문헌의 인용발명 번호
    # 이 구성의 판정 경위. **미대응 구성에도 채웁니다** — 오히려 그쪽이 더 필요합니다.
    # 채택된 문헌이 없으면 evidence도 근거 목록도 비므로, 경위가 없으면 그 구성에 대해
    # 보고서가 말하는 것은 "확인되지 않았다" 한 줄뿐이 됩니다.
    trail: list[VerificationTrail] = []


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
