"""커버리지 계산. LLM 없이 판정 라벨과 근거만으로 문헌 적합도를 산출합니다.

여기서 나오는 값은 보고서에 찍히는 유사도 퍼센트가 아니라 문헌 선정용 내부 지표입니다.
보고서 퍼센트는 판정 라벨의 고정 대표값(REPORT_PERCENT)을 씁니다.
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
# 보고서 표출용 고정 대표값. 연속 계산값이 아닙니다.
REPORT_PERCENT = {
    "동일": 97,
    "실질적 동일": 92,
    "일부 차이": 87,
    "일부 유사": 82,
    "차이": 70,
    "대응 없음": None,
}
REPORT_GRADE = {
    "동일": ("동일", "🔵"),
    "실질적 동일": ("실질적 동일", "🟢"),
    # 구성대비 등급은 개시 범위만 말합니다. 여기서 "단순 변경"이라고 쓰면 뒤의
    # 진보성 판단을 선취하므로, 용이 도출 여부와 무관한 중립 표현을 사용합니다.
    "일부 차이": ("기술 사상 일부 대응, 하위 한정 차이", "🟠"),
    "일부 유사": ("핵심 기능 유사하나 목적/효과에 일부 차이", "🟡"),
    "차이": ("차이", "⚪"),
    "대응 없음": ("대응 안됨", "⚪"),
}

CORE_IMPORTANCE_THRESHOLD = 4     # 이 이상이면 차별적 핵심 구성
COVER_THRESHOLD = 0.55            # "일부 차이" 이상이면 (최종) 커버된 것으로 봅니다
DIRECT_STRONG = 0.55              # 직접 근거가 이 이상이면 유효한 직접 개시
CRITICAL_GAP = 0.35               # 이 미만이면 핵심 공백
SINGLE_SUFFICIENT = 95.0          # 주 인용발명 단독으로 충분하다고 보는 결합 유사도
PRIMARY_CANDIDATE_MARGIN = 0.20   # 최고 문헌 대비 이만큼 못 미치면 주 인용발명 후보 제외

# --- 대응의 우열 -------------------------------------------------------------
# 판정 라벨·직접성·검증 상태의 서열. 점수 하나로 뭉개지 않고 차원별로 비교합니다.
JUDGMENT_RANK = {"대응 없음": 0, "차이": 1, "일부 유사": 2, "일부 차이": 3, "실질적 동일": 4, "동일": 5}
DIRECTNESS_RANK = {"absent": 0, "inferred": 1, "direct": 2}
VERIFY_RANK = {"not_found": 0, "empty": 0, "short": 1, "partial": 2, "verified": 3}
FULL_JUDGMENTS = {"동일", "실질적 동일"}      # 하위 한정까지 개시된 것으로 볼 수 있는 판정
VERIFY_OK = {"verified", "partial"}          # 발췌가 원문에 실재한다고 확인된 상태
# 이 판정들은 대응 기재가 없는 것으로 봅니다. '차이'는 라벨 정의부터 "개시한다고 보기 어렵다"입니다.
NO_CORRESPONDENCE_JUDGMENTS = {"대응 없음", "차이"}
# 보완 이득 가중치. 판정 단계 상승을 가장 크게, 나머지는 근거 품질 개선으로 봅니다.
GAIN_WEIGHTS = {"judgment": 1.00, "directness": 0.35, "evidence": 0.25, "missing": 0.20}


def atomic_coverage(match: ElementMatch) -> float | None:
    """개시가 확인된 하위 제한의 비율. 점검 결과가 없으면 None으로 두어 라벨 강도를 그대로 씁니다.

    반드시 limitation_checks만 셉니다. evidence는 compare.py의 근거 규칙상 **누락 한정에
    가장 가까운 보조 발췌**라서, 그것을 커버된 항목으로 세면 "이 한정은 문헌에 없다"는
    증거를 많이 모을수록 커버율이 올라가는 역전이 생깁니다. quote 1건을 커버 1건으로
    세던 것도 같은 문제였습니다. 하위 제한이 10개든 1개든 분자가 1이었습니다.

    점검 결과가 없을 때 None을 쓰는 근거: 누락 한정 수만으로는 분모(전체 하위 제한 수)를
    알 수 없습니다. 누락이 있으면 _build_matches가 이미 판정을 '일부 차이' 이하로 강등하고
    quality_key도 누락 수를 세므로, 여기서 추정값을 지어내지 않아도 벌점은 반영됩니다.
    """
    if not match.limitation_checks:
        return None
    disclosed = sum(1 for check in match.limitation_checks if check.disclosed)
    return disclosed / len(match.limitation_checks)


def item_similarity(match: ElementMatch | None) -> float:
    """판정 라벨을 대체하지 않고, 같은 라벨 안에서 하위 제한 누락과 직접성만 반영합니다."""
    if match is None:
        return 0.0
    base = JUDGMENT_SIMILARITY.get(match.judgment, 0.0)
    atomic = atomic_coverage(match)
    factor = {"direct": 1.0, "inferred": 0.9, "absent": 0.8}.get(match.directness, 1.0)
    if atomic is None:
        return base * factor
    return base * (0.70 + 0.30 * atomic) * factor


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
    core_direct = sum(row["importance"] * row["direct"] for row in core) / core_weight
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


def uncovered_labels(claim: Claim, matches: dict[str, ElementMatch]) -> list[str]:
    """내부 강도가 커버 기준에 못 미친 구성.

    **거절 가부를 여기서 정하지 마십시오.** item_similarity는 판정 라벨에 직접성과 누락
    한정을 곱하므로 '일부 차이'는 direct·누락 0인 한 점에서만 기준을 넘습니다. 즉 이 목록은
    "차이가 남아 있다"는 뜻이지 "대응 기재가 없다"는 뜻이 아닙니다. 대응 자체의 유무는
    no_correspondence_labels가 판단하며, 거절 이유 구성 가부는 그쪽을 기준으로 합니다.
    """
    return [element.label for element in claim.elements
            if item_similarity(matches.get(element.label)) < COVER_THRESHOLD]


def has_correspondence(match: ElementMatch | None) -> bool:
    """대응 기재가 **존재**하는지. 차이가 없는지(커버)와는 다른 질문입니다.

    거절 이유는 대응 기재가 있는 구성 위에서만 세울 수 있고, 남은 차이는 보고서에 별도로
    표시해야 합니다. 둘을 한 임계값으로 뭉치면 세부 구현이 다르다는 이유만으로 대응 기재가
    없다고 보고하게 됩니다.

    '차이'는 라벨 정의 자체가 "관련 기재는 있으나 청구항 구성을 개시한다고 보기 어렵다"이므로
    대응으로 세지 않습니다. 검증에 실패한 근거도 여기서 걸러집니다.
    """
    atomic = atomic_coverage(match) if match is not None else None
    return bool(
        match
        and match.judgment not in NO_CORRESPONDENCE_JUDGMENTS
        and match.quote
        and match.directness != "absent"
        and match.verify in VERIFY_OK
        # 모델 판정이 잘못 들어오더라도 하위 제한 점검이 0/N이면 대응 기재로 세지 않습니다.
        and (atomic is None or atomic > 0.0)
    )


def no_correspondence_labels(claim: Claim, matches: dict[str, ElementMatch]) -> list[str]:
    """어느 문헌에서도 검증된 대응 기재를 찾지 못한 구성. 진짜 공백입니다."""
    return [element.label for element in claim.elements
            if not has_correspondence(matches.get(element.label))]


def difference_labels(claim: Claim, matches: dict[str, ElementMatch]) -> list[str]:
    """대응 기재는 있으나 청구항과 완전히 같다고는 볼 수 없는 구성."""
    return [element.label for element in claim.elements
            if has_correspondence(matches.get(element.label)) and not is_complete(matches.get(element.label))]


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

    판정 강도 → 직접성 → 원문 발췌 → 검증 상태 → 누락 한정 수 → 내부 유사도 순입니다.
    """
    if match is None:
        return (0, 0, 0, 0, 0, 0.0)
    return (
        JUDGMENT_RANK.get(match.judgment, 0),
        DIRECTNESS_RANK.get(match.directness, 0),
        1 if match.quote else 0,
        VERIFY_RANK.get(match.verify, 0),
        -len(match.missing_limitations),
        round(item_similarity(match), 6),
    )


def decisive_key(match: ElementMatch | None) -> tuple:
    """동률 해소용 마지막 점수를 뺀 실질 비교 키. '명확히 우수'의 기준입니다."""
    return quality_key(match)[:5]


def is_complete(match: ElementMatch | None) -> bool:
    """더 나은 대응을 찾을 이유가 없는 상태. 이 조건을 모두 만족해야 보완 검토를 끝냅니다."""
    return bool(
        match
        and match.judgment in FULL_JUDGMENTS
        and match.directness == "direct"
        and match.quote
        and match.verify in VERIFY_OK
        and not match.missing_limitations
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
    if match.judgment not in FULL_JUDGMENTS:
        residual.append(f"{match.judgment} 판정에 그쳐 하위 한정까지 동일하다고 보기 어렵습니다")
    elif match.directness != "direct":
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
