"""커버리지 계산. LLM 없이 판정 라벨과 근거만으로 문헌 적합도와 표출 유사도를 산출합니다.

문헌 선정에 쓰는 내부 지표(score_document)와 보고서에 찍히는 정량 지표
(limitation_counts)는 서로 다른 값입니다. 후자는 하위 한정의 개시 수를 그대로 셉니다.
"""
from .models import Claim, ClaimElement, ElementMatch

# 판정 라벨 → 내부 강도. 평균이 필수 구성의 결손을 덮지 않도록 보수적으로 벌립니다.
JUDGMENT_SIMILARITY = {
    "동일": 1.00,
    "실질적 동일": 0.85,
    "일부 차이": 0.55,
    "일부 유사": 0.35,
    "차이": 0.15,
    "대응 없음": 0.00,
}
# 보고서 표출 등급. "차이"·"대응 없음"은 대응으로 세지 않으므로 여기에 없습니다.
# 등급 뒤에 붙던 백분율 구간은 없앴습니다 — 그 값은 등급 이름을 숫자로 다시 쓴 것에 가까웠고,
# "%" 기호 때문에 "청구항의 몇 %가 개시되었다"로 잘못 읽혔습니다. 정량 지표는
# limitation_counts(개시 한정 수 / 전체 한정 수)가 대신합니다.
REPORT_GRADES = {
    "동일": ("동일", "🔵"),
    "실질적 동일": ("실질적 동일", "🟢"),
    "일부 차이": ("기술 사상 동일, 세부 구현 방식의 단순 변경", "🟠"),
    "일부 유사": ("핵심 기능 유사하나 목적/효과에 일부 차이", "🟡"),
}
UNCORRESPONDED_GRADE = ("대응 안됨", "⚪")

CORE_IMPORTANCE_THRESHOLD = 4     # 이 이상이면 차별적 핵심 구성
# 주 인용발명 자격 게이트가 후보를 한 건도 남기지 못했을 때 되돌아가는 하한입니다.
# 분해 프롬프트가 1~2를 "범용 부품"·"통상의 인터페이스"로 정의하므로 그 아래로는 넓히지
# 않습니다. 중요도는 매 실행 LLM이 새로 매기는 값이라 4 하나만 하드 게이트로 두면 한 칸
# 흔들릴 때 전 문헌이 미채택으로 떨어집니다(chain._eligible_primaries).
SUBSTANTIVE_IMPORTANCE = 3
DIRECT_STRONG = 0.55              # 직접 근거가 이 이상이면 유효한 직접 개시
CRITICAL_GAP = 0.35               # 이 미만이면 핵심 공백
# 주 인용발명 후보로 남기려면 최고 문헌의 핵심 직접 개시량 대비 이 **비율** 이상이어야 합니다.
#
# 절대 차로 잡으면 안 됩니다. core_direct는 최고 문헌도 0.5 안팎이라 0.20을 빼는 순간 임계가
# 최고점의 60% 수준으로 내려앉고, 사실상 0점 문헌만 걸러져 게이트가 있는 척만 하게 됩니다.
# 비율로 두면 점수 분포가 좁든 넓든 같은 뜻("최고 문헌에 크게 못 미침")을 유지합니다.
PRIMARY_CANDIDATE_RATIO = 0.75

# --- 대응의 우열 -------------------------------------------------------------
# 판정 라벨·직접성·검증 상태의 서열. 점수 하나로 뭉개지 않고 차원별로 비교합니다.
JUDGMENT_RANK = {"대응 없음": 0, "차이": 1, "일부 유사": 2, "일부 차이": 3, "실질적 동일": 4, "동일": 5}
_JUDGMENT_BY_RANK = {rank: judgment for judgment, rank in JUDGMENT_RANK.items()}
DIRECTNESS_RANK = {"absent": 0, "inferred": 1, "direct": 2}
VERIFY_RANK = {"not_found": 0, "empty": 0, "short": 1, "partial": 2, "verified": 3}
FULL_JUDGMENTS = {"동일", "실질적 동일"}      # 하위 한정까지 개시된 것으로 볼 수 있는 판정
VERIFY_OK = {"verified", "partial"}          # 발췌가 원문에 실재한다고 확인된 상태
# 이 판정들은 대응 기재가 없는 것으로 봅니다. '차이'는 라벨 정의부터 "개시한다고 보기 어렵다"입니다.
NO_CORRESPONDENCE_JUDGMENTS = {"대응 없음", "차이"}
# 보완 이득 가중치. 판정 단계 상승을 가장 크게, 나머지는 근거 품질 개선으로 봅니다.
GAIN_WEIGHTS = {"judgment": 1.00, "directness": 0.35, "evidence": 0.25, "missing": 0.20}
# 한정 단위 보완의 가중치(limitation_gain). 누락 한정을 **전부** 메웠을 때의 값이며,
# 판정 1등급 상승(GAIN_WEIGHTS["judgment"] = 1.00)보다 낮게 둡니다.
LIMITATION_GAIN_WEIGHT = 0.60


def judgment_at_rank(rank: int) -> str:
    """서열값을 판정 라벨로 되돌립니다.

    상한을 씌우는 단계가 네 곳(verify·consistency·entailment·chain)입니다. 각자 역인덱스를
    따로 만들면 라벨을 하나 바꿀 때마다 다섯 곳을 함께 고쳐야 하므로 여기 한 벌만 둡니다.
    """
    return _JUDGMENT_BY_RANK[rank]


def counted_checks(checks: list) -> list:
    """대안 묶음을 **한 항목으로 접은** 점검 목록. 커버율·근거 품질의 분모입니다.

    "A, B 또는 C 중 적어도 하나"는 요구사항 하나이지 셋이 아닙니다. 개별 대안을 각각 세면
    선택지를 넉넉히 나열한 청구항일수록 분모만 커져, 문언을 충족한 문헌이 오히려 낮게
    나옵니다. 그래서 한 묶음은 언제나 정확히 한 자리를 차지하고, 개시 여부는 그 묶음의
    아무 대안이나 개시되었는지로 정합니다.

    **충족된 묶음에서 미개시 대안만** 빼는 방식은 쓰지 않습니다. 그러면 같은 묶음에서 모델이
    대안을 몇 개 개시로 표시했느냐에 따라 분모가 흔들려 (1,1)·(2,2)·(3,3)이 제각각 찍힙니다.
    비율은 셋 다 1.0이라 점수와 선정은 같지만, 같은 청구항을 충족한 두 문헌이 서로 다른
    개시 수를 달고 나갑니다.
    """
    counted: list = []
    seen: set[str] = set()
    for check in checks:
        group = check.alternative_group
        if not group:
            counted.append(check)
            continue
        if group in seen:
            continue
        seen.add(group)
        # 묶음의 대표는 개시된 대안이 있으면 그것, 없으면 첫 대안입니다. 대표의 disclosed가
        # 곧 묶음의 충족 여부가 되므로 아래 disclosed_count가 따로 묶음을 다시 볼 필요가 없습니다.
        counted.append(next((item for item in checks
                             if item.alternative_group == group and item.disclosed), check))
    return counted


def disclosed_count(checks: list, resolved: set[str] | None = None) -> tuple[int, int]:
    """(개시가 확인된 하위 한정 수, 분모). 대안 묶음은 양쪽에서 한 자리만 차지합니다.

    커버율(atomic_coverage)과 보고서 정량 지표(limitation_counts)가 **같은 셈법**을 쓰도록
    한 곳에 둡니다. 부르는 쪽마다 대안 묶음 처리를 다시 적으면 같은 뜻의 계산이 여러 벌
    생기고, 그중 하나만 고쳐지는 순간 두 지표가 어긋납니다.

    resolved는 채택 조합의 다른 문헌이 개시한 한정입니다. 원본 문헌 셀에서는 항상 비어
    있고, chain._merge가 만든 결합 사본에서만 들어옵니다. 대안 묶음 중 하나가 다른 문헌으로
    해소된 경우에도 묶음 전체를 한 자리로 세어야 하므로 그룹 단위로 함께 확인합니다.
    """
    resolved = resolved or set()
    resolved_groups = {check.alternative_group for check in checks
                       if check.limitation in resolved and check.alternative_group}
    counted = counted_checks(checks)
    disclosed = sum(1 for check in counted
                    if check.disclosed
                    or check.limitation in resolved
                    or (check.alternative_group and check.alternative_group in resolved_groups))
    return disclosed, len(counted)


def atomic_coverage(match: ElementMatch) -> float | None:
    """개시가 확인된 하위 제한의 비율. 점검 결과가 없으면 None으로 두어 라벨 강도를 그대로 씁니다.

    반드시 limitation_checks만 셉니다. evidence는 compare.py의 근거 규칙상 **누락 한정에
    가장 가까운 보조 발췌**라서, 그것을 커버된 항목으로 세면 "이 한정은 문헌에 없다"는
    증거를 많이 모을수록 커버율이 올라가는 역전이 생깁니다. quote 1건을 커버 1건으로 세는
    것도 같은 문제입니다 — 하위 제한이 10개든 1개든 분자가 1이 됩니다.

    점검 결과가 없을 때 None을 쓰는 근거: 누락 한정 수만으로는 분모(전체 하위 제한 수)를
    알 수 없습니다. 누락이 있으면 _build_matches가 이미 판정을 '일부 차이' 이하로 강등하고
    quality_key도 누락 수를 세므로, 여기서 추정값을 지어내지 않아도 벌점은 반영됩니다.

    **분자는 확인된 개시만 셉니다.** 보고서 정량 지표(limitation_counts)와 분자가 다른 것은
    이 모듈 첫 줄에 적힌 그대로입니다 — 문헌 선정에 쓰는 값과 보고서에 찍히는 값은 서로 다른
    질문에 답합니다. 보고서는 "이 문헌에 무엇이 있다고 판정했나"를 적어야 하므로 미완료를
    빼면 없는 누락을 지어내게 되고, 선정은 "무엇이 확인되었나"를 물어야 하므로 확인되지 않은
    한정 위에서 문헌 순위가 뒤집히면 안 됩니다.
    """
    states = limitation_states(match.limitation_checks, set(match.combination_resolved))
    if not states:
        return None
    return sum(1 for _, state in states if state == DISCLOSED) / len(states)


# --- 한정 하나의 상태: 개시 · 미완료 · 누락 --------------------------------------
# 의미검증이 응답에서 항목을 빠뜨리면 그 한정은 검증을 **받지 못한** 상태로 남습니다
# (entailment._mark_unchecked). 이것을 개시나 누락 어느 한쪽에 접으면 반대 방향의 오류가
# 하나씩 생깁니다.
#
#   개시로 접으면 — 검증을 건너뛴 한정이 보고서에서 가장 튼튼한 근거로 보입니다. 의미검증을
#   통과한 근거에만 "발췌 문언 그대로는 아니며…" 단서가 붙으므로, 검증을 받지 못한 근거는
#   무표시, 즉 원문 그대로로 읽힙니다. 실측에서 🟢을 받은 구성 D·E의 한정이 정확히 그것들이었습니다.
#
#   누락으로 접으면 — "검증기가 응답을 빠뜨렸다"가 "이 문헌에 그 한정이 없다"라는 **문헌에
#   대한 사실 주장**으로 바뀝니다. 그 문장은 그대로 보고서의 차이점 줄로 나갑니다.
#
# 그래서 disclosed를 뒤집지 않고 상태를 하나 더 둡니다. 등급·신규성·문헌 선정·근거 출력이
# 전부 이 함수 하나를 보고, 각자 semantic_status를 다시 해석하지 않습니다.
DISCLOSED = "disclosed"
UNVERIFIED = "unverified"
MISSING = "missing"
_STATE_RANK = {MISSING: 0, UNVERIFIED: 1, DISCLOSED: 2}
# 의미검증을 **시도했으나** 판단을 받지 못한 상태만 미완료로 봅니다. "not_run"은 원문 대조를
# 통과한 발췌 묶음이 없어 대상에 아예 오르지 않은 한정이라 성격이 다릅니다 — 그쪽은
# quote·verify 게이트(ineligible_reason·chain._directly_disclosed)가 이미 따로 막고 있고,
# 여기까지 미완료로 묶으면 정상 경로의 사건 대부분이 미완료로 찍혀 상한이 무의미해집니다.
UNVERIFIED_SEMANTIC_STATUS = "error"


def is_unverified_check(check) -> bool:
    """이 점검 하나가 의미검증을 받지 못한 상태인지. 대안 묶음은 보지 않습니다.

    근거 한 줄에 단서를 붙일지 정하는 자리(report._limitation_evidence_lines)가 필요로 하는
    것은 묶음 단위 상태가 아니라 **그 발췌를 낸 점검**의 상태입니다. 상태 문자열을 소비자가
    직접 비교하지 않도록 여기에 둡니다.
    """
    return bool(check is not None and check.disclosed
                and check.semantic_status == UNVERIFIED_SEMANTIC_STATUS)


def _raw_state(check, resolved: set[str], resolved_groups: set[str]) -> str:
    if check.limitation in resolved or (check.alternative_group
                                        and check.alternative_group in resolved_groups):
        return DISCLOSED
    if not check.disclosed:
        return MISSING
    return UNVERIFIED if check.semantic_status == UNVERIFIED_SEMANTIC_STATUS else DISCLOSED


def limitation_states(checks: list, resolved: set[str] | None = None) -> list[tuple]:
    """접힌 한정마다 (점검, 상태). counted_checks와 같은 순서·같은 자릿수입니다.

    대안 묶음은 묶음 전체가 한 상태를 갖습니다. 확인된 대안이 하나라도 있으면 개시이고,
    확인되지 않은 대안만 남았으면 미완료입니다 — "A 또는 B 중 하나"에서 A가 검증을 통과했다면
    B의 검증이 빠졌다는 사실은 그 요구사항의 충족 여부를 바꾸지 못합니다.

    resolved는 채택 조합의 다른 인용발명이 댄 한정입니다(disclosed_count와 같은 입력).
    결합 심사를 거쳐 들어온 값이므로 개시로 봅니다.
    """
    resolved = resolved or set()
    resolved_groups = {check.alternative_group for check in checks
                       if check.limitation in resolved and check.alternative_group}
    group_state: dict[str, str] = {}
    for check in checks:
        if not check.alternative_group:
            continue
        group_state[check.alternative_group] = max(
            group_state.get(check.alternative_group, MISSING),
            _raw_state(check, resolved, resolved_groups),
            key=_STATE_RANK.get)
    return [(check, group_state[check.alternative_group] if check.alternative_group
             else _raw_state(check, resolved, resolved_groups))
            for check in counted_checks(checks)]


def is_reserved(match: ElementMatch | None) -> bool:
    """확인된 개시가 하나도 없이 미검증만 남은 대응. **차이가 아니라 판단 없음**입니다.

    등급·서술·차이점·결론·내부 지표가 전부 이 하나를 봅니다. 자리마다 조건을 다시 적으면
    한 보고서가 서로 다른 말을 합니다 — 지표는 '판정 유보'인데 결론은 '차이가 남습니다'로
    나간 것이 실측 형태입니다.

    누락뿐인 대응(미검증 0건)은 여기에 해당하지 않습니다. 그쪽은 개시가 없다고 **판정된**
    것이고, 차이로 적는 것이 맞습니다.
    """
    states = limitation_states(match.limitation_checks,
                               set(match.combination_resolved)) if match else []
    return bool(states
                and any(state == UNVERIFIED for _, state in states)
                and not any(state == DISCLOSED for _, state in states))


def limitation_state_map(match: ElementMatch | None) -> dict[str, str]:
    """한정 문언 → 상태. 감사 데이터와 회귀 기대값이 한정 단위로 걸릴 수 있게 합니다.

    대안 묶음은 counted_checks가 접은 대표 하나만 실립니다. 묶음 전체가 한 요구사항이므로
    대안마다 상태를 적으면 같은 요구사항이 여러 줄로 세어집니다.
    """
    if match is None:
        return {}
    return {check.limitation: state
            for check, state in limitation_states(match.limitation_checks,
                                                  set(match.combination_resolved))
            if check.limitation}


def reserved_labels(claim: Claim, matches: dict[str, ElementMatch]) -> list[str]:
    """판정을 유보한 구성. residual(차이가 남는 구성)과 반드시 갈라 셉니다."""
    return [element.label for element in claim.elements
            if is_reserved(matches.get(element.label))]


def unverified_count(match: ElementMatch | None) -> int:
    """의미검증을 받지 못한 한정 수. 구성 원문 한 줄 점검(whole_element)도 함께 셉니다."""
    if match is None:
        return 0
    states = limitation_states(match.limitation_checks, set(match.combination_resolved))
    return sum(1 for _, state in states if state == UNVERIFIED)


def unverified_limitations(match: ElementMatch | None) -> list[str]:
    """보고서에 적을 미완료 한정 문언. whole_element는 뺍니다.

    구성 원문 한 줄을 통째로 점검한 경우의 limitation은 구성 문언 전체라, 차이점 줄에 그대로
    실으면 구성을 한 번 더 읽어 주는 문장이 됩니다(models.missing_limitations와 같은 규율).
    개수는 unverified_count가 whole_element까지 세므로 상한 판단에서는 빠지지 않습니다.
    """
    if match is None:
        return []
    texts: list[str] = []
    for check, state in limitation_states(match.limitation_checks,
                                          set(match.combination_resolved)):
        if state != UNVERIFIED or not check.limitation or check.whole_element:
            continue
        if check.limitation not in texts:
            texts.append(check.limitation)
    return texts


# --- 판정 등급 산출 -----------------------------------------------------------
# 등급은 **코드가 계산합니다.** 모델이 6개 라벨 중 하나를 직접 고르게 하면, 그 값 하나가
# JUDGMENT_SIMILARITY·JUDGMENT_RANK·has_correspondence·chain._directly_disclosed를 전부
# 좌우하면서도 근거에는 묶이지 않습니다. 같은 문헌·같은 발췌에서도 실행마다 '실질적 동일'과
# '대응 없음' 사이를 오가고, 그 흔들림이 그대로 결론까지 갑니다.
#
# 재료는 이미 전부 있습니다. limitation_checks의 core/qualifier별 개시 여부는 항목마다 발췌를
# 요구하고 독립 의미검증(entailment.py)까지 거치므로 라벨보다 훨씬 단단히 묶여 있습니다.
# 남는 두 가지 — 용어가 같은가(동일/실질적 동일), 문헌이 그 구성을 다른 목적으로 쓰는가 —
# 만 모델에게 좁게 묻고, 나머지는 이 함수가 정합니다.
#
# 이 함수는 compare(최초 판정)와 entailment(재심 후 재산출) 양쪽이 함께 씁니다. 두 곳이 각자
# 사다리를 들고 있으면 같은 조건에서 다른 등급이 나옵니다(실제로 그랬습니다 — entailment만
# directness를 absent로 내려 문헌이 보조 자격을 잃었습니다).
TERMINOLOGY_VALUES = {"identical", "equivalent"}


def derive_judgment(checks: list, *, has_evidence: bool, terminology: str = "equivalent",
                    different_purpose: bool = False) -> str:
    """한정별 개시 여부에서 판정 등급을 산출합니다.

    core는 그 구성이 실제로 무엇을 하는가이고 qualifier는 그 동작을 한정하는 조건이므로,
    core가 대응 여부를 가르고 qualifier가 등급을 가릅니다(compare.py [요구사항의 두 종류]).

      core 0개 개시  → 관련 원문이 있으면 "차이", 원문도 없으면 "대응 없음"
      core 일부 개시 → "일부 유사"
      core 전부 개시 → different_purpose면 "일부 유사"
                       qualifier 누락이 있으면 "일부 차이"
                       의미검증 미완료가 있으면 "일부 차이" (상한)
                       전부 개시면 terminology에 따라 "동일" 또는 "실질적 동일"

    has_evidence는 원문 대조에 걸 수 있는 발췌가 하나라도 있는지입니다. "차이"와 "대응 없음"의
    차이가 정확히 이것이라(compare.py: "관련 원문도 제시할 수 없다면 '차이'가 아니라 '대응 없음'"),
    이 구분을 등급 산출에서 잃으면 보고서가 "가장 가까운 기재"를 붙일 근거를 잃습니다.

    **미완료는 개시 쪽에 세고 상한만 씌웁니다.** 누락 쪽에 세면 core가 하나 빠진 것이 되어
    "일부 유사"로 내려가는데, 그 등급의 뜻은 "이 문헌에 그 동작이 없다"입니다. 확인하지 못한
    것을 없는 것으로 적는 셈이라 방향이 반대인 오류가 됩니다. 반대로 개시로만 세고 상한을
    두지 않으면 검증을 건너뛴 한정이 동일급 판정을 받습니다 — 실측 과대판정이 그것입니다.
    """
    states = limitation_states(checks)
    if not states:
        # 점검 자체가 없으면 등급을 세울 근거가 없습니다. compare._requirements가 구성 원문
        # 한 줄이라도 core로 만들어 주므로 정상 경로에서는 오지 않습니다.
        return "대응 없음"
    # core를 선언하지 않은 분해 결과에서는 전체 한정이 그 역할을 합니다.
    gate = [pair for pair in states if pair[0].kind == "core"] or states
    held_gate = sum(1 for _, state in gate if state != MISSING)
    if held_gate == 0:
        return "차이" if has_evidence else "대응 없음"
    if held_gate < len(gate):
        return "일부 유사"
    if different_purpose:
        return "일부 유사"
    if any(state == MISSING for _, state in states):
        return "일부 차이"
    if any(state == UNVERIFIED for _, state in states):
        return "일부 차이"
    return "동일" if terminology == "identical" else "실질적 동일"


# 직접성 계수. absent는 "근거 원문이 없음"이므로 direct_similarity가 이미 0으로 봅니다.
# 종전 0.8은 그 판단과 정면으로 어긋나, 발췌가 아예 없는 '동일'(0.80)이 원문으로 검증된
# '일부 차이'(0.55)보다 높은 유사도를 받았습니다. 이 값은 보고서의 종합 유사도
# (combined_similarity)와 문헌 적합도 양쪽에 그대로 들어갑니다.
DIRECTNESS_FACTOR = {"direct": 1.0, "inferred": 0.85, "absent": 0.55}
# 발췌가 없거나 원문 대조를 통과하지 못한 대응. 판정 라벨은 verify가 이미 상한을 씌우지만,
# 유사도는 그와 별개로 **근거의 실재 여부**를 반영해야 합니다. 그러지 않으면 지어낸 발췌 위에
# 세운 대응과 원문으로 확인된 대응이 보고서에서 같은 숫자로 나갑니다.
UNVERIFIED_EVIDENCE_FACTOR = 0.70
# 의미검증을 받지 못한 한정의 비중만큼 유사도를 낮춥니다. 위 계수와 같은 성격입니다 —
# 그쪽은 "발췌가 원문에 실재하는가", 이쪽은 "그 발췌가 한정을 뒷받침하는가"이고, 둘 다
# 판정 라벨과 **별개로** 근거의 실재를 반영해야 합니다.
#
# 이 계수가 없으면 라벨이 전부를 지배합니다. 등급 상한이 미검증 셀을 '일부 차이'(0.55)에
# 묶어 두어도, 실제로 한정을 확인한 '일부 유사'(0.35) 셀보다 여전히 높습니다. atomic_coverage는
# 0.70~1.00 사이에서만 움직여 그 차이를 뒤집지 못합니다. 실측 형태로 재현하면 주 인용발명
# 점수가 32.42 대 4.57로, 아무것도 확인하지 못한 문헌이 주 인용발명이 됐습니다.
#
# 값은 UNVERIFIED_EVIDENCE_FACTOR와 같게 둡니다. 두 계수가 재는 것이 같은 종류의 결손이라
# 무게를 다르게 줄 근거가 없고, 무엇보다 이 계수는 atomic_coverage(이미 확정 개시만 셈)와
# **곱해집니다**. 한 등급 강등에 해당하는 0.64를 쓰면 두 항이 같은 사실을 두 번 깎아 사실상
# 두 등급이 내려가고, 실제로 그 값에서는 문헌이 주 인용발명 자격(CRITICAL_GAP)에서 떨어져
# 구성이 "어느 인용발명에서도 확인되지 않음"으로 적혔습니다 — 원문 발췌가 있는데도 그렇습니다.
UNVERIFIED_SEMANTIC_FACTOR = UNVERIFIED_EVIDENCE_FACTOR


def item_similarity(match: ElementMatch | None) -> float:
    """판정 라벨을 대체하지 않고, 같은 라벨 안에서 하위 제한 누락·직접성·근거 실재만 반영합니다."""
    if match is None:
        return 0.0
    base = JUDGMENT_SIMILARITY.get(match.judgment, 0.0)
    atomic = atomic_coverage(match)
    factor = DIRECTNESS_FACTOR.get(match.directness, 1.0)
    if not match.quote or match.verify not in VERIFY_OK:
        factor *= UNVERIFIED_EVIDENCE_FACTOR
    factor *= _semantic_factor(match)
    if atomic is None:
        return base * factor
    return base * (0.70 + 0.30 * atomic) * factor


def _semantic_factor(match: ElementMatch) -> float:
    """의미검증을 받지 못한 한정의 비중에 따른 감쇠. **확정 개시가 0이면 0입니다.**

    비중으로 두는 이유: 다섯 중 하나가 미검증인 셀과 전부 미검증인 셀은 같은 것이 아닙니다.
    앞의 것을 뒤의 것과 똑같이 깎으면, 대부분을 확인한 문헌이 아무것도 확인하지 못한 문헌과
    같은 자리에 놓입니다.

    **확정 개시가 하나도 없으면 계수가 아니라 0입니다.** 감쇠 계수를 아무리 낮게 잡아도
    "확인이 0인 문헌이 확인이 1인 문헌을 밀어내지 않는다"는 보장이 서지 않습니다 — 한정
    2개짜리에 맞춘 값이 5개짜리에서 다시 뒤집혔습니다(4.14 대 4.09). 이것은 크기의 문제가
    아니라 순서의 문제라 계수로 표현할 수 없고, 규칙으로 두어야 합니다. 내부 지표가 묻는
    것은 "무엇이 확인되었나"이므로, 확인된 것이 없으면 0이 정확한 답이기도 합니다.

    누락뿐인 셀(미검증 0건)에는 적용하지 않습니다. 그쪽은 개시가 없다고 **판정된** 것이고
    atomic_coverage와 판정 라벨이 이미 그 사실을 반영합니다.
    """
    states = limitation_states(match.limitation_checks, set(match.combination_resolved))
    if not states:
        return 1.0
    unverified = sum(1 for _, state in states if state == UNVERIFIED)
    if not unverified:
        return 1.0
    if is_reserved(match):
        return 0.0
    return 1.0 - (1.0 - UNVERIFIED_SEMANTIC_FACTOR) * (unverified / len(states))


def direct_similarity(match: ElementMatch | None) -> float:
    """원문 발췌로 확인되는 강도. 넓은 기능적 유사성이 직접 개시를 밀어내지 못하게 합니다."""
    if match is None:
        return 0.0
    similarity = item_similarity(match)
    if similarity < CRITICAL_GAP:
        return 0.0
    if not match.quote or match.verify in {"not_found", "empty", "short"}:
        return similarity * 0.35
    if match.directness == "absent":
        return 0.0
    if match.directness == "inferred":
        return similarity * 0.65
    return similarity


def core_elements(claim: Claim) -> list[ClaimElement]:
    """차별적 핵심 구성. 하나도 없으면 전 구성을 핵심으로 봅니다."""
    return [element for element in claim.elements if is_core(element)] or list(claim.elements)


def core_direct_score(claim: Claim, matches: dict[str, ElementMatch],
                      elements: list[ClaimElement] | None = None) -> float:
    """핵심 구성의 **직접** 개시량(0~1). 중요도 가중 평균입니다.

    문헌 순위(score_document)와 주 인용발명 자격 게이트가 같은 정의를 쓰도록 한 곳에 둡니다.
    한쪽만 단순 평균을 쓰면 같은 '핵심 직접 개시량'이라는 이름으로 두 곳이 다른 값을 쓰게 되고,
    임계(PRIMARY_CANDIDATE_RATIO)도 서로 다른 척도 위에서 비교됩니다.

    elements를 주면 그 구성들로 잽니다. 중요도 분류가 이 사건에서 신호를 갖지 못할 때
    전 구성으로 다시 재기 위한 것입니다(chain._eligible_primaries).
    """
    core = elements or core_elements(claim)
    weight = sum(element.importance for element in core) or 1
    return sum(element.importance * direct_similarity(matches.get(element.label))
               for element in core) / weight


def rows_for(claim: Claim, matches: dict[str, ElementMatch]) -> list[dict]:
    return [
        {
            "label": element.label,
            "importance": float(element.importance),
            "similarity": item_similarity(matches.get(element.label)),
            "direct": direct_similarity(matches.get(element.label)),
            "match": matches.get(element.label),
            "element": element,
        }
        for element in claim.elements
    ]


def weighted(rows: list[dict], key: str = "similarity") -> float:
    numerator = sum(row["importance"] * row[key] for row in rows)
    denominator = sum(row["importance"] for row in rows)
    return numerator / denominator if denominator else 0.0


def score_document(claim: Claim, matches: dict[str, ElementMatch]) -> tuple[float, dict]:
    """문헌 1건의 단독 적합도(0~100). 기술분야 어휘 사전에 의존하지 않습니다.

    차별적 핵심 구성의 **직접** 개시량을 주점수로 삼고, 범용 구성의 폭은 동률 해소에만 씁니다.
    핵심 구성을 통째로 놓친 문헌은 평균이 높아도 감점됩니다.
    """
    rows = rows_for(claim, matches)
    if not rows:
        return 0.0, {}
    core = [row for row in rows if row["importance"] >= CORE_IMPORTANCE_THRESHOLD] or rows
    core_weight = sum(row["importance"] for row in core) or 1.0

    element_coverage = weighted(rows)
    core_coverage = sum(row["importance"] * (row["similarity"] if row["similarity"] >= CRITICAL_GAP else 0.0)
                        for row in core) / core_weight
    # 주 인용발명 자격 게이트와 같은 정의를 씁니다(coverage.core_direct_score).
    core_direct = core_direct_score(claim, matches)
    core_breadth = sum(row["importance"] for row in core if row["direct"] >= DIRECT_STRONG) / core_weight
    critical_gap = sum(row["importance"] for row in core if row["similarity"] < CRITICAL_GAP) / core_weight
    evidence_adjusted = sum(
        row["importance"] * row["similarity"] * (
            0.55
            + 0.25 * bool(row["match"] and row["match"].quote)
            + 0.10 * bool(row["match"] and row["match"].chunk_id)
            + 0.10 * bool(row["match"] and row["match"].reason)
        )
        for row in rows
    ) / (sum(row["importance"] for row in rows) or 1.0)

    # 핵심 공백을 가산식으로 빼면, 공백이 많은 두 문헌의 점수가 모두 0으로 잘린다. 그러면
    # 실제로는 핵심 구성 두 개를 부분 개시한 문헌과 한 개만 부분 개시한 문헌이 동률이 되고,
    # 주 인용발명이 기술적 근접성이 아니라 document_id 순서로 정해진다. 공백은 감점하되
    # 양의 대응 신호를 지우지 않도록 승산식으로 반영한다.
    positive = (0.40 * core_direct
                + 0.25 * core_coverage
                + 0.15 * core_breadth
                + 0.12 * element_coverage
                + 0.08 * evidence_adjusted)
    main_score = positive * (1.0 - 0.20 * critical_gap)
    detail = {
        "core_labels": [row["label"] for row in core],
        "core_direct": round(core_direct, 4),
        "core_coverage": round(core_coverage, 4),
        "core_breadth": round(core_breadth, 4),
        "element_coverage": round(element_coverage, 4),
        "evidence_adjusted": round(evidence_adjusted, 4),
        "critical_gap": round(critical_gap, 4),
        "formula": "(0.40*핵심직접 + 0.25*핵심커버 + 0.15*핵심폭 + 0.12*전체 + 0.08*근거품질) * (1 - 0.20*핵심공백)",
    }
    return round(main_score * 100, 2), detail


def limitation_counts(match: ElementMatch | None) -> tuple[int, int]:
    """이 구성의 하위 한정 중 **개시가 확인된 수 / 전체 수**.

    보고서에 찍는 정량 지표입니다. 종전의 백분율 유사도는 판정 라벨이 정한 등급 밴드 안의
    위치라서, 대응된 구성에서는 거의 항상 밴드 최댓값이었고 등급 이름을 되풀이할 뿐이었습니다.
    무엇보다 "94%"가 "청구항의 94%가 개시되었다"로 읽히는데 실제 뜻은 그것이 아니었습니다.

    이 값은 분자·분모가 그대로 보고서에 나가므로 독자가 근거 목록과 대조해 검증할 수 있고,
    한정이 빠질 때마다 실제로 움직입니다. 대안 묶음은 하나로 셉니다 — 선택지를 넉넉히 나열한
    청구항일수록 분모만 커지면, 문언을 충족하는 문헌이 오히려 낮게 나옵니다.
    """
    if match is None:
        return 0, 0
    return disclosed_count(match.limitation_checks, set(match.combination_resolved))


def evidence_locations(match: ElementMatch | None) -> int:
    """개시 근거로 인용된 서로 다른 원문 위치 수.

    한 문단을 모든 한정의 근거로 되풀이 인용한 대응은, 한정마다 다른 문단을 짚은 대응보다
    약합니다. 전자는 그 문단 하나의 해석이 무너지면 구성 전체가 무너집니다.
    """
    if match is None:
        return 0
    locations = {check.chunk_id or check.quote for check in match.limitation_checks
                 if check.disclosed and check.quote}
    locations.update(span.chunk_id or span.quote
                     for check in match.limitation_checks if check.disclosed
                     for span in check.evidence if span.quote)
    if not locations and match.quote:
        return 1
    return len(locations)


def report_grade(match: ElementMatch | None) -> tuple[str, str]:
    grade = REPORT_GRADES.get(match.judgment) if match else None
    return grade or UNCORRESPONDED_GRADE


def has_correspondence(match: ElementMatch | None) -> bool:
    """대응 기재가 **존재**하는지. 차이가 없는지(커버)와는 다른 질문입니다.

    거절 이유는 대응 기재가 있는 구성 위에서만 세울 수 있고, 남은 차이는 보고서에 별도로
    표시해야 합니다. 둘을 한 임계값으로 뭉치면 세부 구현이 다르다는 이유만으로 대응 기재가
    없다고 보고하게 됩니다.

    '차이'는 라벨 정의 자체가 "관련 기재는 있으나 청구항 구성을 개시한다고 보기 어렵다"이므로
    대응으로 세지 않습니다. 검증에 실패한 근거도 여기서 걸러집니다.
    """
    return bool(
        match
        and match.judgment not in NO_CORRESPONDENCE_JUDGMENTS
        and match.quote
        and match.directness != "absent"
        and match.verify in VERIFY_OK
        # 모델 판정이 잘못 들어오더라도 하위 제한 점검이 0/N이면 대응 기재로 세지 않습니다.
        #
        # **커버율(atomic_coverage)로 묻지 않습니다.** 그 값은 확정 개시만 세므로, 한정이
        # 전부 미검증인 셀도 0.0이 되어 여기서 걸립니다. 그러면 원문 발췌가 그대로 있는
        # 구성이 uncovered로 떨어지고 결론이 "어느 인용발명에서도 확인되지 않았습니다"라고
        # 적습니다 — 손에 든 문헌을 다시 찾아 나서게 만드는 거짓 진술이고, 불변식 P1이
        # 막으려는 바로 그것입니다. 이 게이트가 물어야 하는 것은 "확인되었는가"가 아니라
        # "없다고 판정되었는가"이므로, 전부 누락일 때만 걸러 냅니다.
        and _anything_not_missing(match)
    )


def _anything_not_missing(match: ElementMatch) -> bool:
    states = limitation_states(match.limitation_checks, set(match.combination_resolved))
    return not states or any(state != MISSING for _, state in states)


def no_correspondence_labels(claim: Claim, matches: dict[str, ElementMatch]) -> list[str]:
    """어느 문헌에서도 검증된 대응 기재를 찾지 못한 구성. 진짜 공백입니다."""
    return [element.label for element in claim.elements
            if not has_correspondence(matches.get(element.label))]


def difference_labels(claim: Claim, matches: dict[str, ElementMatch]) -> list[str]:
    """채택 조합으로 대응은 되었지만 **실제로 남은** 차이가 있는 구성.

    단독 셀의 판정 라벨은 결합 동기까지 평가하지 않으므로, 다른 채택 문헌이 누락 한정을
    모두 메워도 보수적으로 '일부 차이'에 머물 수 있습니다. 그 라벨만 보고 residual에 다시
    올리면 보고서는 같은 한정을 앞에서는 "결합으로 해소"하고 결론에서는 "결합 후에도 남음"으로
    적습니다. 결합으로 해소한 한정 외에 남은 누락·지시관계·다른 목적이 없으면 커버리지 기준의
    잔여 차이는 없습니다. 진보성의 결합 동기·용이성은 별도 결론 문구가 계속 유보합니다.
    """
    return [element.label for element in claim.elements
            if has_correspondence(matches.get(element.label))
            and _has_residual_difference(matches.get(element.label))]


def _has_residual_difference(match: ElementMatch | None) -> bool:
    if match is None or is_complete(match):
        return False
    if (match.combination_resolved and not match.missing_limitations
            and not match.antecedent_note and not match.different_purpose):
        return False
    return True


def disclosed_limitations(match: ElementMatch | None) -> set[str]:
    """이 문헌이 **원문 대조를 통과한 근거로** 개시한 하위 한정의 문언.

    missing_limitations의 반대편입니다. 어떤 문헌이 남은 차이를 실제로 메우는지 물으려면
    구성 단위 대응(has_correspondence)으로는 부족합니다 — 그 구성에 대응은 있어도 정작
    빠진 그 한정은 없을 수 있고, 그때 "이 문헌에 기재가 있다"고 적으면 없는 개시를
    단언하게 됩니다. 검증(verify)까지 요구하는 것은 has_correspondence와 같은 이유입니다.

    **의미검증을 받지 못한 한정은 여기서 빠집니다.** 이 집합이 곧 filled_limitations를 거쳐
    "주 인용발명이 빠뜨린 그 한정을 이 문헌이 댔다"는 결합의 확정 근거가 됩니다. 확인하지
    못한 한정으로 그 문장을 세우면, 등급 상한으로 막아 둔 미완료가 문헌 선정 쪽으로 우회해
    들어옵니다 — 상한은 표시를 누를 뿐 결합을 막지 못하기 때문입니다.

    아래 함수들과 마찬가지로 판정 라벨은 보지 않습니다. 1차 사실만 봅니다.
    """
    if match is None or match.error:
        return set()
    return {check.limitation for check in match.limitation_checks
            if check.disclosed and check.limitation and check.verify in VERIFY_OK
            and not is_unverified_check(check)}


# --- 한정 단위 사실 계층 -------------------------------------------------------
# 이 아래 함수들은 **판정 라벨을 보지 않습니다.** 라벨(judgment)은 한정별 개시 여부에서
# 유도한 표시값인데(derive_judgment), 그 유도값이 다시 결합·선정의 하드 게이트로 올라가면서
# 1차 사실이 두 번 압축됐습니다. 압축은 비가역이라, 새 사건마다 다른 곳에서 절벽이 납니다.
#
# 그래서 결합 판단은 라벨 대신 여기를 봅니다. 라벨은 보고서에 찍는 데만 씁니다.


def combination_supported(match: ElementMatch | None) -> dict[str, list[str]]:
    """결합 심사가 인정한 한정과 빠진 축을 실제로 댄 인용발명.

    entailment.validate_combination이 채택 조합 전체의 근거 위에서 다시 물어 인정한 것만
    담깁니다. 문헌 단독으로는 여전히 개시가 아니므로 그 셀의 disclosed는 건드리지 않고,
    결합 결과에서만 메워진 것으로 셉니다(chain._absorb_limitations).
    """
    if match is None or match.error:
        return {}
    return {check.limitation: list(check.combination_documents)
            for check in match.limitation_checks
            if check.semantic_status == "accepted_in_combination" and check.limitation}


def rejected_limitations(match: ElementMatch | None) -> set[str]:
    """1차 판정은 개시였는데 **의미검증이 축 결손으로 기각한** 한정.

    아직 결합 심사를 거치지 않은 것만 셉니다. 결합 위에서 확인이 끝난 것은 인정이든 기각이든
    더 이상 유보가 아니므로(accepted_in_combination / rejected_in_combination) 여기 오지
    않습니다. 그러지 않으면 확정된 공백이 영영 유보로 남습니다.

    entailment.validate_entailment는 기각할 때 check.disclosed를 False로 덮어씁니다.
    그러면 "이 문헌에 근거가 아예 없다"와 "근거는 있는데 축 하나가 이 문헌에 없다"가
    같은 값이 되어, 결합 단계에서는 둘을 구별할 수 없습니다. 다행히 판단 자체는
    semantic_status에 남으므로 여기서 되살려 읽습니다.

    두 상태를 가르는 것이 왜 중요한가: 전자는 다른 문헌을 찾아야 하고, 후자는 **이미 손에
    든 다른 인용발명이 그 축을 대고 있는지**를 물어야 합니다. 진보성 결합이 정확히 그
    작업입니다. 뭉뚱그리면 결합으로 세울 수 있는 거절 이유가 "구성 곤란"으로 나갑니다.
    """
    if match is None or match.error:
        return set()
    return {check.limitation for check in match.limitation_checks
            if check.semantic_status == "rejected" and check.limitation
            and check.verify in VERIFY_OK}


def evidenced_limitations(match: ElementMatch | None) -> set[str]:
    """이 문헌이 그 한정에 대해 **원문 대조를 통과한 근거를 실제로 낸** 것 전부.

    개시로 확정된 것과 축 결손으로 기각된 것을 함께 봅니다. 결합 후보를 고를 때 물어야 할
    질문은 "이 문헌이 이 구성을 개시했는가"가 아니라 "이 문헌이 여기에 보탤 원문이 있는가"
    이기 때문입니다. 부 인용발명은 원래 구성 전체로는 주 인용발명보다 약합니다.

    **의미검증을 받지 못한 것도 함께 봅니다.** 그 한정에도 원문 대조를 통과한 발췌가 그대로
    있습니다(그것이 의미검증 대상이 되는 전제조건입니다). 여기서 빠뜨리면 불변식 P1이
    "공백으로 적었으나 문헌에 근거가 있다"를 볼 수 없게 되는데, 미검증 구성이 uncovered로
    떨어지는 경우가 정확히 P1이 잡아야 할 상황입니다 — 감시자가 감시할 자리에서 눈을 감습니다.
    """
    return (disclosed_limitations(match) | rejected_limitations(match)
            | unchecked_limitations(match))


def unchecked_limitations(match: ElementMatch | None) -> set[str]:
    """의미검증을 받지 못했으나 **원문 대조는 통과한** 한정의 문언.

    rejected_limitations와 짝입니다. 그쪽은 "판단해 봤더니 뒷받침하지 못한다", 이쪽은
    "판단을 받지 못했다"이고, 둘 다 그 문헌에 원문이 있다는 사실은 같습니다.
    """
    if match is None or match.error:
        return set()
    return {check.limitation for check in match.limitation_checks
            if is_unverified_check(check) and check.limitation and check.verify in VERIFY_OK}


def has_evidence(match: ElementMatch | None) -> bool:
    """결합 탐색 전용의 대응 유무. has_correspondence와 달리 라벨을 보지 않습니다.

    has_correspondence는 보고서에 "이 구성은 대응된다"고 적어도 되는지를 답하므로 보수적인
    채로 두어야 합니다. 반면 "다른 문헌을 더 볼 가치가 있는가"에 그 기준을 쓰면, 아직
    대응이 서지 않았다는 이유로 대응을 세울 문헌을 탈락시키는 순환이 됩니다.

    한정 점검이 없는 셀에는 잴 재료가 없으므로 여기서는 거짓입니다. 그 경우의 처리는 부르는
    쪽이 정합니다(chain._fills_a_gap) — 재료가 없다는 사실과 근거가 없다는 사실을 같은 값으로
    돌려주면, 부르는 쪽이 둘을 구별할 방법이 없습니다.
    """
    return bool(evidenced_limitations(match))


def pending_labels(matrix: dict[str, dict[str, ElementMatch]],
                   labels: list[str]) -> dict[str, list[str]]:
    """labels 중 **축 결손 기각이 걸려 있어** 결합 위에서 다시 물어야 하는 것과 그 사유.

    공백으로 넘어온 라벨 중 이 목록에 든 것은 "어느 인용발명에서도 확인되지 않았다"고 적을
    수 없습니다. 원문 근거는 있고, 이 도구가 결합 위에서 그것을 확인하는 단계를 아직 거치지
    않았을 뿐입니다.

    **병합 셀이 아니라 매트릭스를 봅니다.** 기각은 특정 문헌의 판정에 붙는 사실인데, 병합은
    구성 하나를 문헌 하나에 통째로 넘기므로(best_match) 진 쪽 셀의 기각 기록은 병합 결과에
    남지 않습니다. 그러면 축 결손을 안고 있는 문헌이 우연히 병합에서 지는 것만으로 유보가
    공백으로 바뀝니다. beyond_limit·well_known이 같은 이유로 매트릭스를 보는 것과 같습니다.
    """
    found: dict[str, list[str]] = {}
    for label in labels:
        reasons = [f"인용발명 {document_id}: {reason}"
                   for document_id in sorted(matrix)
                   for reason in pending_reasons(matrix[document_id].get(label))]
        if reasons:
            found[label] = reasons
    return found


def pending_reasons(match: ElementMatch | None) -> list[str]:
    """축 결손으로 기각된 한정과 그 사유. 보고서가 유보를 사실대로 적기 위한 재료입니다."""
    if match is None or match.error:
        return []
    return [f"{check.limitation} — {check.semantic_note}".strip(" —")
            for check in match.limitation_checks
            if check.semantic_status == "rejected" and check.limitation
            and check.verify in VERIFY_OK]


def combined_similarity(claim: Claim, chain_matches: dict[str, ElementMatch]) -> float:
    """결합 후 청구항 전체 유사도(0~100). 구성별로 가장 좋은 판정을 채택한 결과입니다."""
    return round(weighted(rows_for(claim, chain_matches)) * 100, 2)


def best_match(candidates: list[ElementMatch | None]) -> ElementMatch | None:
    """구성 1개에 대한 여러 문헌의 대응 중 가장 강한 것. 동률이면 입력 순서를 따릅니다.

    자격(발췌·검증)을 갖춘 대응을 먼저 봅니다. 판정 라벨만 높고 근거가 없는 대응이
    검증된 대응을 밀어내면, 보고서가 근거 없는 발췌 위에 서게 됩니다.
    """
    present = [match for match in candidates if match is not None]
    if not present:
        return None
    eligible = [match for match in present if is_eligible_supplement(match)]
    if eligible:
        return max(eligible, key=quality_key)
    # 모두 근거 자격이 없으면 라벨만 더 높은 셀로 바꾸지 않습니다. _merge는 현재 채택 셀을
    # 첫 번째로 넘기므로, 이렇게 해야 발췌 없는 "차이"가 주 문헌의 "대응 없음"을 밀어내고
    # 보조 인용발명의 채택 근거인 것처럼 보고서에 찍히지 않습니다.
    return present[0]


# --- 보완 검토 판정 -----------------------------------------------------------

def quality_key(match: ElementMatch | None) -> tuple:
    """대응의 우열. 점수 하나로 줄이지 않고 요구 우선순위 그대로 사전식 비교합니다.

    판정 강도 → 직접성 → 원문 발췌 → 검증 상태 → 누락 한정 수 → 의미검증 미완료 수 →
    내부 유사도 순입니다.

    미완료 수가 여기 들어가는 이유: 등급이 같고 근거 품질도 같다면 **검증을 받은 셀**이
    결합의 근거가 되어야 합니다. 등급 상한(derive_judgment)만으로는 두 셀이 같은 칸에
    나란히 놓이는 경우를 가르지 못하고, 그때 입력 순서가 채택 문헌을 정하게 됩니다.
    """
    if match is None:
        return (0, 0, 0, 0, 0, 0, 0.0)
    return (
        JUDGMENT_RANK.get(match.judgment, 0),
        DIRECTNESS_RANK.get(match.directness, 0),
        1 if match.quote else 0,
        VERIFY_RANK.get(match.verify, 0),
        -len(match.missing_limitations),
        -unverified_count(match),
        round(item_similarity(match), 6),
    )


def decisive_key(match: ElementMatch | None) -> tuple:
    """동률 해소용 마지막 점수를 뺀 실질 비교 키. '명확히 우수'의 기준입니다.

    미완료 수까지 포함합니다(내부 유사도만 뺍니다). 검증을 받은 셀과 받지 못한 셀의 차이는
    미세한 점수 차가 아니라 실질적인 근거 품질 차이라, 보완으로 인정해야 합니다.
    """
    return quality_key(match)[:6]


def is_complete(match: ElementMatch | None) -> bool:
    """더 나은 대응을 찾을 이유가 없는 상태. 이 조건을 모두 만족해야 보완 검토를 끝냅니다."""
    return bool(
        match
        and match.judgment in FULL_JUDGMENTS
        and match.directness == "direct"
        and match.quote
        and match.verify in VERIFY_OK
        and not match.missing_limitations
        and not unverified_count(match)
    )


def supplement_reason(match: ElementMatch | None) -> str:
    """왜 다른 문헌을 더 봐야 하는지. 빈 문자열이면 보완 검토가 필요 없습니다."""
    if match is None or match.judgment == "대응 없음":
        return "대응 기재 없음"
    reasons: list[str] = []
    if match.judgment not in FULL_JUDGMENTS:
        reasons.append(f"{match.judgment} 판정")
    if match.missing_limitations:
        reasons.append(f"누락 한정 {len(match.missing_limitations)}건")
    # 누락과 **따로** 적습니다. 둘을 합쳐 세면 "이 문헌에 없는 한정"과 "확인하지 못한 한정"이
    # 한 숫자가 되어, 보완 문헌을 찾아야 하는지 검증을 다시 돌려야 하는지 구별되지 않습니다.
    if unverified_count(match):
        reasons.append(f"의미검증 미완료 {unverified_count(match)}건")
    if match.directness != "direct":
        reasons.append(f"직접 개시 아님({match.directness})")
    if not match.quote:
        reasons.append("원문 발췌 없음")
    elif match.verify not in VERIFY_OK:
        reasons.append(f"발췌 검증 미통과({match.verify})")
    return "; ".join(reasons)


def supplement_needed_labels(claim: Claim, matches: dict[str, ElementMatch]) -> list[str]:
    """다른 문헌의 더 나은 대응을 검토해야 하는 구성. uncovered보다 넓습니다.

    '일부 차이'는 최종적으로 일부 커버된 상태일 수 있으나, 여기에는 반드시 포함됩니다.
    """
    return [element.label for element in claim.elements
            if not is_complete(matches.get(element.label))]


def ineligible_reason(match: ElementMatch | None) -> str:
    """보완 문헌 자격 미달 사유. 빈 문자열이면 보완 근거로 쓸 수 있습니다."""
    if match is None or match.judgment == "대응 없음":
        return "대응 기재 없음"
    if not match.quote:
        return "원문 발췌 없음"
    if match.directness == "absent":
        return "직접 근거 없음"
    if match.verify not in VERIFY_OK:
        return f"발췌 검증 실패({match.verify})"
    return ""


def is_eligible_supplement(match: ElementMatch | None) -> bool:
    return not ineligible_reason(match)


def is_better_match(candidate: ElementMatch | None, current: ElementMatch | None) -> bool:
    """내부 점수가 아니라 판정·직접성·근거·누락 한정 중 하나 이상이 실제로 개선될 때만 참입니다.

    점수만 미세하게 높은 대응(같은 판정·같은 근거 품질)은 보완으로 보지 않습니다.
    """
    if candidate is None:
        return False
    return decisive_key(candidate) > decisive_key(current)


def filled_limitations(candidate: ElementMatch | None,
                       current: ElementMatch | None) -> set[str]:
    """후보가 **현재 채택 셀에 빠진** 하위 한정 중 원문으로 개시한 것.

    진보성 결합에서 부 인용발명을 데려오는 이유가 바로 이것입니다. 부 인용발명은 그 구성
    전체를 주 인용발명보다 잘 개시하는 문헌이 아니라, 주 인용발명이 빠뜨린 한정 하나를
    대는 문헌입니다. 그래서 구성 단위 우열(is_better_match)만 물으면 이 기여는 보이지
    않습니다 — 오히려 부 인용발명 쪽 등급이 낮은 것이 정상입니다.

    주 인용발명이 '일부 차이'(rank 3)로 구성 전체를 덮고 한정 하나만 빠뜨린 상태에서, 그
    한정을 원문으로 개시한 문헌이 구성 전체로는 '차이'(rank 1)인 경우가 전형입니다. 구성
    단위 우열만 물으면 그 문헌이 전부 탈락해 결합이 1건에 머뭅니다.
    """
    if candidate is None or current is None:
        return set()
    return disclosed_limitations(candidate) & set(current.missing_limitations)


def limitation_gain(candidate: ElementMatch | None, current: ElementMatch | None) -> float:
    """한정 단위 보완 이득. 채택 셀의 누락 한정 중 몇 할을 메웠는지입니다.

    supplement_gain과 같은 척도 위에 올리기 위해 비율에 가중치를 곱합니다. 누락을 전부
    메워도 supplement_gain의 판정 1등급 상승(1.0)보다 작게 두었습니다. 구성 전체의 등급이
    올라가는 것과 빠진 한정 하나가 메워지는 것은 같은 무게가 아니고, 둘이 경합하면 등급이
    올라가는 쪽을 먼저 집는 것이 맞습니다.
    """
    filled = filled_limitations(candidate, current)
    missing = len(current.missing_limitations) if current else 0
    if not filled or not missing:
        return 0.0
    return round(LIMITATION_GAIN_WEIGHT * len(filled) / missing, 6)


def supplement_gain(candidate: ElementMatch | None, current: ElementMatch | None) -> float:
    """보완 이득. 점수 차가 아니라 개선된 '차원'을 합산합니다."""
    if candidate is None:
        return 0.0
    judgment_step = (JUDGMENT_RANK.get(candidate.judgment, 0)
                     - (JUDGMENT_RANK.get(current.judgment, 0) if current else 0)) / 5
    directness_step = (DIRECTNESS_RANK.get(candidate.directness, 0)
                       - (DIRECTNESS_RANK.get(current.directness, 0) if current else 0)) / 2
    evidence_step = _evidence_strength(candidate) - _evidence_strength(current)
    now, following = _missing_count(current), _missing_count(candidate)
    missing_step = (now - following) / max(now, following, 1)
    return round(GAIN_WEIGHTS["judgment"] * judgment_step
                 + GAIN_WEIGHTS["directness"] * directness_step
                 + GAIN_WEIGHTS["evidence"] * evidence_step
                 + GAIN_WEIGHTS["missing"] * missing_step, 6)


def residual_difference(match: ElementMatch | None) -> list[str]:
    """결합 후에도 남는 차이. 누락 한정을 먼저 쓰고, 없으면 판정 자체의 한계를 씁니다."""
    if match is None:
        return ["대응되는 기재가 확인되지 않았습니다"]
    if match.judgment == "대응 없음":
        if match.missing_limitations:
            return list(match.missing_limitations)
        if any(span.verify == "verified" and span.quote for span in match.evidence):
            return ["관련 기재는 있으나 청구항 한정 전체의 개시는 확인되지 않았습니다"]
        return ["대응되는 기재가 확인되지 않았습니다"]
    residual = list(match.missing_limitations)
    # 미완료를 누락 목록에 섞지 않고 별도 문장으로 적습니다. 누락으로 적으면 "이 문헌에 그
    # 한정이 없다"가 되고, 아예 적지 않으면 등급이 왜 상한에 걸렸는지가 본문에서 사라져
    # 결론과 본문이 어긋납니다(report.report_invariants가 실측에서 잡은 형태입니다).
    unverified = unverified_limitations(match)
    if unverified:
        residual.append(f"{'; '.join(unverified)} 한정은 의미검증을 수행하지 못해 "
                        "개시가 확인되지 않았습니다")
    # 확인된 개시가 하나도 없으면 여기서 끝냅니다. "'일부 차이' 판정에 그쳐 하위 한정까지
    # 동일하다고 보기 어렵습니다"를 덧붙이면, 상한으로 눌러 둔 라벨을 대비해 본 결과인 것처럼
    # 되읽게 됩니다 — 그 라벨 자체가 판단을 받지 못해서 나온 값입니다.
    if is_reserved(match):
        return residual
    if match.judgment not in FULL_JUDGMENTS:
        residual.append(f"{match.judgment} 판정에 그쳐 하위 한정까지 동일하다고 보기 어렵습니다")
    if match.directness != "direct":
        residual.append("직접 개시가 아니라 추론에 의한 대응입니다")
    return residual


def _evidence_strength(match: ElementMatch | None) -> float:
    if match is None or not match.quote:
        return 0.0
    return VERIFY_RANK.get(match.verify, 0) / 3


def _missing_count(match: ElementMatch | None) -> int:
    return len(match.missing_limitations) if match else 0


def is_core(element: ClaimElement) -> bool:
    return element.importance >= CORE_IMPORTANCE_THRESHOLD


# --- 주지관용기술 -------------------------------------------------------------
# 주지관용으로 다룰 수 있는 구성의 중요도 상한. 분해 단계의 기준으로 2는 "통상의 인터페이스·
# 입출력 구성", 1은 "어느 발명에나 나타나는 범용 부품(메모리, 프로세서, 전원부)"입니다.
WELL_KNOWN_IMPORTANCE_MAX = 2
# 관용성을 실증할 최소 문헌 수. 중요도 하나만으로 가르지 않기 위한 조건입니다.
WELL_KNOWN_MIN_DOCUMENTS = 2


def well_known_labels(claim: Claim, matrix: dict[str, dict[str, ElementMatch]],
                      labels: list[str]) -> dict[str, list[str]]:
    """공백 구성 중 주지관용기술로 다룰 수 있는 것과, 그 관용성을 실증하는 문헌.

    **중요도만으로 가르지 않습니다.** "중요도 2 이하"라는 기준 하나로 별도 절을 만들면 무엇도
    실제로 인정되지 않으면서 미개시 사실만 흐려집니다. 업로드된 문헌 **여러 건이 같은 구성을
    실제로 언급한다**는 실증을 함께 요구합니다.
    한 문헌에만 있으면 그것은 주지관용이 아니라 그냥 인용발명이고, 어느 문헌에도 없으면 이
    도구가 주지관용이라고 말할 근거를 가지고 있지 않습니다.

    판정 라벨은 보지 않고 **원문 대조를 통과한 발췌가 있는지**만 봅니다. 대응으로 세지 않은
    '차이'·'일부 유사'도 그 문헌이 그 구성을 다루기는 한다는 증거이기 때문입니다.

    이 함수는 주지관용이라고 **단정하지 않습니다**. 최종 인정은 심사관의 판단이므로, 결론에
    근거 문헌과 함께 드러내 다툴 수 있게 하는 것이 목적입니다.
    """
    by_label = {element.label: element for element in claim.elements}
    found: dict[str, list[str]] = {}
    for label in labels:
        element = by_label.get(label)
        if element is None or element.is_preamble or element.importance > WELL_KNOWN_IMPORTANCE_MAX:
            continue
        mentions = sorted(
            document_id for document_id, matches in matrix.items()
            if (match := matches.get(label)) is not None and not match.error
            and match.quote and match.verify == "verified" and match.judgment != "대응 없음")
        if len(mentions) >= WELL_KNOWN_MIN_DOCUMENTS:
            found[label] = mentions
    return found
