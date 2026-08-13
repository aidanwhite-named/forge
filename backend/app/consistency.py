"""구성 간 정합성. 같은 문헌 안에서 서로 모순되는 셀 판정을 바로잡습니다.

비교 단계는 (구성 × 문헌) 셀을 서로 독립적으로 판정합니다. 셀 하나만 보면 그럴듯한 판정이
청구항 전체로 보면 성립할 수 없는 경우가 생깁니다. 대표적인 것이 **선행 구성을 참조하는
구성**입니다.

  (A) … 제1반사부재 및 제2반사부재            ← 그 문헌에 대응 기재 없음
  (B) 상기 제1 반사부재 및 상기 제2 반사부재 사이에 배치되는 광원   ← '일부 유사'로 개시 인정

문헌에 제1·제2 반사부재가 없으면 "그 둘 **사이에** 배치되는 광원"도 그 문헌에 있을 수
없습니다. 그런데 (B) 셀만 떼어 놓고 보면 "광원(패널 디스플레이)"이라는 문장이 있으므로
모델은 부분 대응을 줍니다. 그 결과 남는 것은 "광원을 포함한다"뿐이고, 이는 그 분야의
어떤 장치나 만족하는 문장입니다.

여기서는 LLM을 다시 부르지 않고 청구항 문언의 **지시 관계**만으로 상한을 씌웁니다.
지시 관계는 "상기 X" 표현에서 그대로 읽어 낼 수 있으므로 기술분야 사전이 필요 없습니다.
"""
import re

from .coverage import JUDGMENT_RANK, has_correspondence, judgment_at_rank
from .models import Claim, ElementMatch

# 지시 대상 어구가 끝나는 자리. 조사·연결어미를 만나면 거기까지가 대상입니다.
# 위치 관계를 나타내는 명사(사이·중·간·내·외)는 뒤에 조사가 바로 붙어 공백이 없으므로
# 따로 끊습니다. 끊지 않으면 "제2 반사부재 사이"가 통째로 지시 대상이 되어, 정작 앞
# 구성에 있는 "제2반사부재"와 문자열이 어긋납니다.
#
# **조사는 겹쳐 붙습니다(`+`).** 하나만 끊으면 "상기 결함탐지부로부터"에서 '로' 뒤가 공백이
# 아니라 '부터'라 그 자리를 넘기고, 다음 경계인 '부터 '에서 끊어 지시 대상이 "결함탐지부로"가
# 됩니다. 앞 구성의 문언은 "…결함탐지부"이므로 이 어구는 어디에도 걸리지 않고, 그 구성은
# **지시 관계가 아예 없는 것**으로 처리됩니다. 실측에서 로부터/으로부터 형태의 참조
# (결함탐지부·수평구조물검출부·3차원모델생성부·문자인식부)가 이렇게 통째로 유실됐습니다.
_BOUNDARY = re.compile(
    r"(?:사이|중|간|내부|외부|내|외)(?=[에의를은는]|\s|$)"
    r"|(?:은|는|이|가|을|를|에|의|와|과|로|으로|및|또는|에서|부터|까지)+(?:\s|$)")
# 상한의 하한선. 원문 대조를 통과한 발췌가 있으면 **부분 대응**까지는 남깁니다.
#
# '차이'까지 내리면 그 구성은 미대응이 되고, 그 문헌은 조합에서 빠져 근거 목록에서도
# 사라집니다. 지시 대상이 다를 뿐 구성의 나머지 substance는 실제로 개시한 문헌이 그렇게
# 통째로 지워지면, 심사관은 이미 확인된 문단을 다시 찾게 됩니다. 상한의 목적은 "완전 개시로
# 세지 않는 것"이지 "기재가 없다고 단정하는 것"이 아니므로 부분 대응에서 멈춥니다.
_PARTIAL = JUDGMENT_RANK["일부 유사"]


def _heads(text: str) -> list[str]:
    """"상기 …" 뒤에서 지시 대상 어구만 끊어 냅니다. 표기는 청구항 문언 그대로 둡니다."""
    heads: list[str] = []
    for fragment in re.split(r"상기", str(text or ""))[1:]:
        head = _BOUNDARY.split(fragment.strip(), 1)[0].strip(" ,.;·")
        if len(re.sub(r"\s+", "", head)) >= 2:
            heads.append(head)
    return heads


def anaphora(text: str) -> list[str]:
    """"상기 …"가 가리키는 대상 어구를 공백을 지운 형태로 뽑습니다.

    공보와 청구항은 같은 용어를 띄어쓰기만 달리 적는 일이 흔해서("제1반사부재" ↔
    "제1 반사부재") 공백을 지운 뒤 비교합니다.
    """
    targets: list[str] = []
    for head in _heads(text):
        collapsed = re.sub(r"\s+", "", head)
        if collapsed not in targets:
            targets.append(collapsed)
    return targets


def antecedent_terms(claim: Claim) -> dict[str, list[str]]:
    """구성마다 그것이 "상기 …"로 가리키는 **앞선 구성**의 지시 어구를 문언 그대로 모읍니다.

    antecedents()는 상한을 씌울 **라벨**을 주지만, 의미검증(entailment.py)은 라벨이 아니라
    한정 문장 안의 낱말을 보고 판단합니다. 어느 낱말이 이 구성이 새로 도입한 수단이 아니라
    앞 구성에서 이미 세워 둔 대상인지 알려면 어구 자체가 필요합니다.

    분해된 한정 문언은 지시어를 풀어 적으므로("상기 플라이휠의 회전 운동을 …" → "크랭크-슬라이드
    기구부가 플라이휠의 회전 운동을 …으로 변환함") 한정만 봐서는 그 낱말이 이 구성의 요구사항인지
    앞 구성에서 온 지시 대상인지 구분할 수 없습니다. 구분은 구성 원문에서만 읽어 낼 수 있습니다.
    """
    collapsed = [(element.label, re.sub(r"\s+", "", element.text)) for element in claim.elements]
    terms: dict[str, list[str]] = {}
    for index, element in enumerate(claim.elements):
        for head in _heads(element.text):
            target = re.sub(r"\s+", "", head)
            if not any(target in text for _, text in collapsed[:index]):
                continue
            bucket = terms.setdefault(element.label, [])
            if head not in bucket:
                bucket.append(head)
    return terms


def antecedents(claim: Claim) -> dict[str, list[str]]:
    """구성마다 그것이 "상기 …"로 참조하는 **앞선 구성**의 라벨을 찾습니다.

    지시 대상은 그 어구가 **처음 등장한 구성 하나**입니다. 부모 청구항에서 온 용어는 이
    청구항의 행렬에 없으므로 자연히 걸리지 않습니다.

    **어구를 담은 앞 구성을 모두 잇지 않습니다.** 그렇게 하면 그 어구를 자기도 "상기 …"로
    참조하고 있을 뿐인 구성까지 지시 대상이 됩니다. 청구항은 같은 대상을 여러 구성이 반복해
    참조하므로 이것은 예외가 아니라 기본값입니다.

        (A) … 데이터수집부                      ← 여기서 도입
        (B) 상기 데이터수집부에 수집된 … 결함탐지부   ← A를 참조할 뿐
        (D) 상기 데이터수집부에 수집된 … 수평구조물검출부

    D의 지시 대상은 A 하나인데 모두 이으면 [A, B, C]가 되고, enforce_antecedents는
    min(선행 구성 판정)으로 상한을 잡으므로 **D와 아무 관계 없는 B의 미대응이 D의 등급을
    끌어내립니다.** 실측에서 한정이 2/2 전부 개시된 구성이 "실질적 동일 → 일부 유사"로
    강등된 채 보고서 결론의 '차이가 남는 구성'에 실렸습니다 — 본문의 집계와 결론이 서로
    모순하는 상태입니다.

    이 방향의 오류는 P3 불변식이 잡지 못합니다(report._grades_never_exceed_their_own_evidence는
    등급이 유도값보다 **높은** 쪽만 봅니다). 여기서 틀리면 어디서도 걸리지 않습니다.
    """
    collapsed = [(element.label, re.sub(r"\s+", "", element.text), element.is_preamble)
                 for element in claim.elements]
    links: dict[str, list[str]] = {}
    for index, element in enumerate(claim.elements):
        for target in anaphora(element.text):
            source = _introducer(target, collapsed[:index])
            if source is None:
                continue
            bucket = links.setdefault(element.label, [])
            if source not in bucket:
                bucket.append(source)
    return links


def _introducer(target: str, preceding: list[tuple[str, str, bool]]) -> str | None:
    """지시 어구를 처음 도입한 구성. 전제부는 그것밖에 없을 때만 씁니다.

    전제부는 발명의 명칭을 그대로 옮겨 적으므로 뒤 구성이 쓰는 용어의 **부분 문자열**을
    거의 언제나 품습니다. "결함 인지형 건축물 외벽 3차원 모델링 시스템에 있어서"가 "3차원
    모델"을 품는 식입니다. 그러면 "상기 3차원 모델"의 지시 대상이 그 모델을 실제로 생성하는
    구성이 아니라 전제부로 잡히고, 전제부는 "컴퓨터로 실행되는 …시스템"이라 어느 문헌에서나
    완전 개시로 나오므로 상한이 사실상 풀립니다.

    전제부가 한정적 의미를 갖는지는 사건마다 다른 법적 판단이라 이 파이프라인은 전제부를
    결론 게이트에서도 빼냅니다(chain.blocking_labels). 지시 관계에서도 같은 자리에 둡니다 —
    실제 구성이 그 어구를 도입했다면 그쪽이 지시 대상입니다.
    """
    for label, text, is_preamble in preceding:
        if target in text and not is_preamble:
            return label
    return next((label for label, text, _ in preceding if target in text), None)


# --- 교차문헌 일관성 ----------------------------------------------------------
# 같은 한정을 놓고 문헌마다 어긋난 판정이 나오는 것을 찾아 남깁니다. 보고할 최대 건수만
# 제한합니다 — 이 노트가 구성대비 결과보다 길어지면 읽히지 않습니다.
MAX_DIVERGENCE_NOTES = 12


def cross_document_notes(matches: list[ElementMatch],
                         filenames: dict[str, str] | None = None) -> list[str]:
    """같은 한정이 한 문헌에서는 개시로, 다른 문헌에서는 미개시로 갈린 경우를 남깁니다.

    의미검증(entailment.validate_entailment)은 **문헌별로 따로 호출**됩니다. 어떤 문헌을
    심사하는 호출은 다른 문헌에 대해 무엇을 인정했는지 볼 수 없으므로, 같은 성격의 기재가
    한쪽에서는 인정되고 다른 쪽에서는 기각되는 일이 구조적으로 생깁니다. 프롬프트에
    "일관되게 판단하라"고 적어도 호출이 갈려 있으면 닿지 않습니다.

    **자동으로 되돌리지 않습니다.** 어느 쪽이 옳은지는 원문을 읽어야 정해지고, 코드가 한쪽으로
    맞추면 지금까지 되풀이된 과잉 교정·과소 교정을 한 번 더 하게 됩니다. 갈린 사실과 양쪽
    근거를 나란히 남겨 읽는 사람이 판단하게 하는 것이 이 함수의 전부입니다.
    """
    names = filenames or {}
    grouped: dict[tuple[int, str, int, str], dict[str, tuple[str, str]]] = {}
    for match in matches:
        if match.error:
            continue
        for check in match.limitation_checks:
            if check.semantic_status not in {"accepted", "rejected"}:
                continue
            key = (match.claim_number, match.label, check.index, check.limitation)
            grouped.setdefault(key, {})[match.document_id] = (
                check.semantic_status, check.quote or check.semantic_note)

    notes: list[str] = []
    for (claim_number, label, _, limitation), verdicts in sorted(grouped.items()):
        accepted = sorted(document_id for document_id, (status, _) in verdicts.items()
                          if status == "accepted")
        rejected = sorted(document_id for document_id, (status, _) in verdicts.items()
                          if status == "rejected")
        if not accepted or not rejected:
            continue
        if len(notes) >= MAX_DIVERGENCE_NOTES:
            notes.append(f"이 밖에도 문헌 간 판정이 갈린 한정이 더 있습니다. "
                         f"(표시 상한 {MAX_DIVERGENCE_NOTES}건)")
            break
        shown = []
        for document_id in (*accepted, *rejected):
            status, detail = verdicts[document_id]
            mark = "인정" if status == "accepted" else "기각"
            shown.append(f"{names.get(document_id, f'문헌 {document_id}')}({mark}: "
                         f"{(detail or '').strip()[:110]})")
        notes.append(
            f"청구항 {claim_number} ({label}) 한정 '{limitation[:60]}'의 판정이 문헌 간에 "
            f"갈렸습니다 — {' / '.join(shown)}. 의미검증은 문헌별로 따로 수행되므로 서로의 "
            "판단을 보지 못합니다. 같은 성격의 기재인지 원문으로 확인하십시오.")
    return notes


def enforce_antecedents(claim: Claim, matrix: dict[str, dict[str, ElementMatch]]) -> list[str]:
    """선행 구성이 대응되지 않은 문헌에서는 그것을 참조하는 구성도 개시로 세지 않습니다.

    상한은 **같은 문헌 안에서만** 걸립니다. 문헌 A가 앞 구성을, 문헌 B가 뒤 구성을 개시한
    경우는 정상적인 결합이므로 건드리지 않습니다 — 결합은 이후 선정 단계가 판단합니다.
    """
    links = antecedents(claim)
    if not links:
        return []
    notes: list[str] = []
    for document_id in sorted(matrix):
        matches = matrix[document_id]
        for label, sources in links.items():
            match = matches.get(label)
            if match is None or match.error:
                continue
            limit = min((JUDGMENT_RANK.get(matches[source].judgment, 0)
                         for source in sources if source in matches), default=None)
            if limit is None or JUDGMENT_RANK.get(match.judgment, 0) <= limit:
                continue
            unresolved = [source for source in sources
                          if source in matches and not has_correspondence(matches[source])]
            if not unresolved:
                continue
            # 발췌가 원문 대조를 통과했다면 부분 대응까지는 남깁니다.
            floor = _PARTIAL if match.quote and match.verify in {"verified", "partial"} else 0
            capped = max(min(JUDGMENT_RANK.get(match.judgment, 0), limit), floor)
            if capped >= JUDGMENT_RANK.get(match.judgment, 0):
                continue
            match.antecedent_capped_from = match.judgment
            match.downgraded_from = match.downgraded_from or match.judgment
            notes.append(f"청구항 {claim.number} ({label}) / 문헌 {document_id}: "
                         f"{match.judgment} → {judgment_at_rank(capped)} "
                         f"(참조 구성 {', '.join(unresolved)}이 같은 문헌에서 대응되지 않음)")
            # 보고서에도 남깁니다. 적지 않으면 "한정은 전부 개시인데 등급만 낮은" 결과가
            # 이유 없이 보이고, 읽는 사람은 집계와 등급 중 어느 쪽이 맞는지 알 수 없습니다.
            match.antecedent_note = (
                f"같은 인용발명에서 구성 {', '.join(unresolved)}의 대응이 확인되지 않아, "
                "이를 참조하는 이 구성을 완전 개시로 보지 않았습니다")
            match.judgment = judgment_at_rank(capped)
    return notes
