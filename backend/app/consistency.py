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
from .models import Claim, ElementMatch, Reference, ReferenceAlias

# 위치 관계를 나타내는 명사(사이·중·간·내·외)에서 후보 넓히기를 멈춥니다. 넘어가면 "제2
# 반사부재 사이"까지 후보가 되는데, 앞 구성의 문언은 "제2반사부재"라 더 긴 쪽이 걸릴 일은
# 없습니다. 다만 후보를 일찍 잘라 두면 무관한 낱말이 섞이지 않습니다.
_POSITIONAL = re.compile(r"(?:사이|중|간|내부|외부|내|외)(?=[에의를은는]|\s|$)")
# 상한의 하한선. 원문 대조를 통과한 발췌가 있으면 **부분 대응**까지는 남깁니다.
#
# '차이'까지 내리면 그 구성은 미대응이 되고, 그 문헌은 조합에서 빠져 근거 목록에서도
# 사라집니다. 지시 대상이 다를 뿐 구성의 나머지 substance는 실제로 개시한 문헌이 그렇게
# 통째로 지워지면, 심사관은 이미 확인된 문단을 다시 찾게 됩니다. 상한의 목적은 "완전 개시로
# 세지 않는 것"이지 "기재가 없다고 단정하는 것"이 아니므로 부분 대응에서 멈춥니다.
_PARTIAL = JUDGMENT_RANK["일부 유사"]


# 지시 어구 후보를 넓혀 보는 최대 어절 수. 청구항의 지시 대상은 길어야 대여섯 어절이고,
# 그보다 길게 잡으면 문장 뒤쪽의 무관한 낱말까지 후보에 들어옵니다.
_MAX_HEAD_WORDS = 6
# 후보 꼬리에서 떼어 낼 조사. **어구의 끝을 찍는 데 쓰지 않습니다** — 이미 어절 경계로 자른
# 후보의 꼬리만 다듬습니다. 어디서 끝나는지는 앞 구성의 문언이 정합니다(_heads).
#
# 겹쳐 붙는 조사를 한 번에 뗍니다(`+`). "…결함탐지부로부터"에서 '로'만 떼면 "결함탐지부로"가
# 남아 앞 구성의 "…결함탐지부"와 어긋나고, 그 참조는 통째로 유실됩니다.
_TAIL = re.compile(r"(?:은|는|이|가|을|를|에|의|와|과|로|으로|및|또는|에서|부터|까지)+$")


def _candidates(fragment: str) -> list[tuple[str, bool]]:
    """"상기 …" 뒤의 지시 어구 후보를 **어절 경계로 넓혀 가며** 만듭니다. (어구, 닫혔는지).

    조사 하나로 어구의 끝을 찍는 방식은 조사가 어구 **안에** 있으면 그 자리에서 잘립니다.
    실측: "상기 복수의 3차원 기준점들"이 '복수의'의 '의'에서 잘려 지시 대상이 **"복수"**가
    됐습니다. 그런 낱말을 도입한 구성은 없으므로 그 참조는 통째로 유실됩니다.

    어디서 끝나는지는 문법만으로 정할 수 없습니다("상기 검출부의 출력"의 대상은 검출부이고,
    "상기 복수의 기준점"의 대상은 복수의 기준점입니다). 그래서 여기서는 **정하지 않고**
    후보만 늘어놓고, 실제로 앞 구성이 도입한 어구인지로 고르는 일은 호출부가 합니다.

    **닫혔는지**는 그 자리에서 어구가 문법적으로 끝났는지입니다. 조사가 붙었거나(…기준점들**의**,
    …수집부**에**) 문장·경계가 왔으면 닫힌 것이고, 조사 없이 다음 낱말이 이어지면(가시 두상 →
    **영상**) 어구 한가운데를 자른 것입니다. 뒤엣것으로 앞 구성에 걸리면 그 연결은 **공통
    접두어 추측**이지 지시 관계의 확인이 아닙니다 — 호출부가 그 둘을 갈라 씁니다.
    """
    stopped = _POSITIONAL.split(fragment.strip(), 1)[0]
    words = stopped.split()
    limit = min(len(words), _MAX_HEAD_WORDS)
    candidates: dict[str, bool] = {}
    for size in range(1, limit + 1):
        joined = " ".join(words[:size])
        head = _TAIL.sub("", joined).strip(" ,.;·")
        if len(re.sub(r"\s+", "", head)) < 2:
            continue
        # 꼬리 조사를 실제로 떼어 냈거나, 후보를 더 넓힐 자리가 없으면(위치 명사·어절 상한·
        # 문장 끝) 그 자리에서 어구가 닫힌 것입니다.
        #
        # 같은 어구가 두 크기에서 나오면 **닫힌 쪽으로 셉니다.** "…점군 및"은 4어절에서
        # 열린 채로, 5어절에서 '및'을 떼며 닫힌 채로 같은 문자열이 나오는데, 먼저 본 것을
        # 남기면 닫힌 어구가 열린 것으로 기록되어 확인된 연결이 추측으로 강등됩니다.
        candidates[head] = candidates.get(head, False) or head != joined or size == limit
    return list(candidates.items())


def _heads(text: str, preceding: list[tuple[str, str, bool]]) -> list[Reference]:
    """"상기 …"가 가리키는 지시 관계. 어구·도입 구성과 **그 연결의 확실성**.

    후보 중 **앞 구성에 실제로 있는 것**을 고릅니다. 어구의 끝을 문법으로 찍지 않고 앞 구성의
    문언으로 확인하므로, 조사가 어구 안에 있어도 유실되지 않습니다.

    **가장 긴 후보를 고르되, 그 후보가 어구를 다 덮었는지로 확실성을 가릅니다.** 짧은 후보를
    닫혔다는 이유로 먼저 고르면 안 됩니다 — "복수의 3차원 기준점들"에서 '의'를 뗀 **"복수"**가
    닫힌 두 글자 후보로 이기고, 그것이 바로 이 모듈이 고쳐 온 오류입니다.

    어구 한가운데를 자른 접두어가 앞 구성에 우연히 걸리는 일이 잦습니다. 실측한 오연결입니다.

        (A) 가시 두상 **영역**을 추출함
        (B) 가시 두상 **색상**을 산출함
        (C) 상기 가시 두상 **영상**을 처리함     ← "가시 두상"만으로는 A인지 B인지 모릅니다

    조사 없이 다음 낱말로 이어지는 자리에서 끊긴 연결(fuzzy)과, 그 어구를 도입한 구성이 둘
    이상인 연결(ambiguous)은 **추측**입니다. 지우지는 않습니다 — (E)의 "가시 두상 영역"을
    (G)가 "가시 두상 영상"으로 받아 적은 실측처럼, 청구항의 표기 흔들림을 잡아내는 것도 이
    추측이기 때문입니다. 다만 등급 상한의 근거로는 쓰지 않습니다(enforce_antecedents).
    """
    found: list[Reference] = []
    for fragment in re.split(r"상기", str(text or ""))[1:]:
        resolved = [(len(re.sub(r"\s+", "", head)), closed, head, hit)
                    for head, closed in _candidates(fragment)
                    if (hit := _introducers(re.sub(r"\s+", "", head), preceding))]
        if not resolved:
            continue
        _, closed, head, sources = max(resolved)
        quality = "direct" if closed and len(sources) == 1 else (
            "ambiguous" if len(sources) > 1 else "fuzzy")
        found.append(Reference(term=head, source=sources[0], quality=quality,
                               candidates=list(sources)))
    return found


def references(claim: Claim) -> dict[str, list[Reference]]:
    """구성마다 (지시 어구 원문, 그 어구를 도입한 구성 라벨).

    antecedents()와 antecedent_terms()가 **같은 자료**를 씁니다. 전에는 둘이 각자 청구항을
    훑으면서 어구를 조금씩 다르게 끊었고, 그래서 같은 참조가 한쪽에는 잡히고 다른 쪽에는
    안 잡히는 일이 생겼습니다. 지시 관계는 하나이므로 읽는 자리도 하나여야 합니다.
    """
    collapsed = [(element.label, re.sub(r"\s+", "", element.text), element.is_preamble)
                 for element in claim.elements]
    # 사람이 확정한 별칭만 승격시킵니다. 확정되지 않은 항목은 목록에 있어도 추측 그대로입니다 —
    # 확정 화면에 올랐다는 사실과 사람이 그렇다고 답했다는 사실은 다릅니다.
    settled = {(alias.target, alias.term): alias
               for alias in claim.aliases if alias.settled}
    links: dict[str, list[Reference]] = {}
    for index, element in enumerate(claim.elements):
        # 같은 어구를 한 구성이 두 번 받아 쓰면(H가 "상기 복수의 3차원 기준점들의 …"과 "상기
        # 복수의 3차원 기준점들 각각에 …"을 함께 쓰는 식) 같은 연결이 확실성만 달리 두 번
        # 나옵니다. 강한 쪽으로 모읍니다 — 한 번이라도 확인된 연결을 추측으로 적으면, 걸려야
        # 할 상한이 걸리지 않습니다.
        best: dict[tuple[str, str], Reference] = {}
        for found in _heads(element.text, collapsed[:index]):
            # 확정된 별칭은 **사용자가 고른 후보로** 갈아 끼웁니다. 애매한 참조에서 해소기가
            # 기본으로 집은 후보와 사람이 고른 후보가 다를 수 있고, 그 자리가 바로 사람에게
            # 물은 이유입니다. 해소기의 기본값을 남겨 두면 물어본 의미가 없습니다.
            # **지금 해소기가 찾은 후보**에 대조합니다. 별칭이 지고 있는 후보 목록은 확정 당시의
            # 것이라, 청구항이 바뀌어 후보에서 빠진 선택도 자기 목록 안에서는 여전히 성립합니다.
            # 그것을 통과시키면 사용자가 보지 않은 연결이 확정된 채로 판정에 들어갑니다.
            alias = settled.get((element.label, found.term))
            if alias is not None and alias.selected_source not in found.candidates:
                alias = None
            if alias is not None and not found.confirmed:
                found = found.model_copy(update={"quality": "confirmed_alias",
                                                 "source": alias.selected_source})
            key = (found.term, found.source)
            if key not in best or (found.confirmed and not best[key].confirmed):
                best[key] = found
        if best:
            links[element.label] = list(best.values())
    return links


def antecedent_terms(claim: Claim) -> dict[str, list[str]]:
    """구성마다 그것이 "상기 …"로 가리키는 **앞선 구성**의 지시 어구를 문언 그대로 모읍니다.

    antecedents()는 상한을 씌울 **라벨**을 주지만, 의미검증(entailment.py)은 라벨이 아니라
    한정 문장 안의 낱말을 보고 판단합니다. 어느 낱말이 이 구성이 새로 도입한 수단이 아니라
    앞 구성에서 이미 세워 둔 대상인지 알려면 어구 자체가 필요합니다.

    분해된 한정 문언은 지시어를 풀어 적으므로("상기 플라이휠의 회전 운동을 …" → "크랭크-슬라이드
    기구부가 플라이휠의 회전 운동을 …으로 변환함") 한정만 봐서는 그 낱말이 이 구성의 요구사항인지
    앞 구성에서 온 지시 대상인지 구분할 수 없습니다. 구분은 구성 원문에서만 읽어 낼 수 있습니다.

    **antecedents()와 같은 이유로 확인된 연결만 돌려줍니다.** 이 값은 관측용이 아니라 의미검증의
    입력입니다(entailment._references). 프롬프트는 여기 실린 낱말을 "이미 앞 구성이 세워 둔
    대상"으로 놓고 **그 낱말이 근거에 없다는 이유로 기각하지 말라**고 지시하므로, 추측으로 이은
    어구를 실으면 등급 상한은 안 걸어도 개시 판정의 요구사항이 느슨해집니다. 오연결이 만드는
    오류가 근거 없는 **강등**에서 근거 없는 **인정**으로 방향만 바뀔 뿐입니다.
    """
    return {label: terms
            for label, found in references(claim).items()
            if (terms := list(dict.fromkeys(item.term for item in found if item.confirmed)))}


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

    **확인된 연결만 돌려줍니다.** 이 값을 쓰는 곳은 전부 판정을 바꾸는 자리입니다 — 등급 상한
    (enforce_antecedents), 결합 상한 복원(chain), 차이점 서술(report). 추측을 섞어 두면 그
    셋이 각자 "여기서는 걸러야 한다"를 기억해야 하고, 한 곳이라도 잊으면 근거 없는 강등이
    조용히 나갑니다. 추측까지 필요한 곳은 references()를 직접 봅니다.
    """
    return {label: sources
            for label, found in references(claim).items()
            if (sources := list(dict.fromkeys(item.source for item in found if item.confirmed)))}


def pending_aliases(claim: Claim) -> list[ReferenceAlias]:
    """확정을 기다리는 별칭 후보. 결정론적 해소가 **확정하지 못한 것만** 올립니다.

    **모델에게 묻지 않습니다.** 여기 오르는 후보는 이미 문언에서 읽어 낸 것이고, 모델이 더할
    수 있는 것은 "영역과 영상이 같은 대상인가" 같은 의미 판단뿐인데 그것이 바로 사람이 해야
    한다고 정한 판단입니다. 추측 위에 추측을 얹지 않습니다.

    이미 확정된 항목은 상태를 그대로 지고 남습니다 — 확정 화면을 다시 열었을 때 사용자가
    앞서 무엇을 확정했는지 보여야 하고, 그 값이 곧 다음 실행의 입력입니다.
    """
    stored = {(alias.target, alias.term): alias for alias in claim.aliases}
    pending: dict[tuple[str, str], ReferenceAlias] = {}
    for target, found in references(claim).items():
        for item in found:
            if item.quality == "direct":
                continue
            key = (target, item.term)
            kept = stored.get(key)
            # 후보가 하나면 고를 것이 없습니다. 비워 두면 "선택되지 않음"이 되어 화면에서
            # 확정 자체를 할 수 없고(체크박스가 선택을 기다립니다), 실제 사건의 후보는 대부분
            # 이 형태입니다. 남은 결정이 확정 여부 하나뿐인 자리이므로 미리 채웁니다 —
            # 채우는 것은 **선택**이지 확정이 아니라, confirmed는 그대로 거짓입니다.
            default = item.candidates[0] if len(item.candidates) == 1 else ""
            pending[key] = ReferenceAlias(
                target=target, term=item.term, candidates=list(item.candidates),
                # 앞서 고른 값은 **여전히 후보 안에 있을 때만** 지고 남습니다. 청구항이나
                # 해소 결과가 바뀌어 후보에서 빠진 선택을 되살리면, 사용자가 보지 않은 연결이
                # 확정된 채로 다음 실행에 들어갑니다.
                selected_source=(kept.selected_source
                                 if kept and kept.selected_source in item.candidates
                                 else default),
                confirmed=bool(kept and kept.confirmed
                               and kept.selected_source in item.candidates))
    return list(pending.values())


def reference_warnings(claims: list[Claim]) -> list[str]:
    """확정하지 못한 지시 관계. **보고서 본문에 나갑니다**(verify_notes가 아닙니다).

    enforce_antecedents가 이런 연결로는 등급을 건드리지 않으므로, 적어 두지 않으면 그 판단은
    어디에도 남지 않습니다. 그러면 두 가지가 함께 사라집니다 — 도구가 지시 관계를 확정하지
    못했다는 사실과, **청구항 문언 자체가 어긋나 있을 수 있다는 신호**입니다. 실측에서 한
    구성이 세운 "가시 두상 영역"을 다음 구성이 "가시 두상 영상"으로 받아 적었는데, 그것은
    도구가 고칠 문제가 아니라 사람이 청구항을 손봐야 할 문제였습니다.
    """
    warnings: list[str] = []
    for claim in claims:
        for label, found in references(claim).items():
            for item in (entry for entry in found if not entry.confirmed):
                where = (f"구성 {', '.join(item.candidates)} 중 하나로 좁히지 못했습니다"
                         if item.quality == "ambiguous"
                         else f"구성 {item.source}으로 추정했습니다(어구가 온전히 일치하지 않음)")
                warnings.append(
                    f"청구항 {claim.number} ({label})의 \"{item.term}\"은 지시 대상을 {where}. "
                    "등급 상한의 근거로는 쓰지 않았으니 청구항 문언을 확인하십시오.")
    return warnings


def _introduces(target: str, text: str) -> bool:
    """이 구성이 그 어구를 **도입**하는지. 되받아 쓰기만 하는 것은 도입이 아닙니다.

    청구항에서 도입과 참조는 문언으로 갈립니다 — 도입한 자리에는 "상기"가 없고, 되받는
    자리에는 있습니다. 단순 포함으로 보면 그 둘이 구별되지 않아, 같은 대상을 여러 구성이
    이어 참조하는 흔한 청구항에서 지시 대상이 도입자가 아니라 **바로 앞의 참조자**로 잡힙니다.
    그러면 enforce_antecedents가 min(선행 구성 판정)으로 상한을 잡으므로, 아무 관계 없는
    구성의 미대응이 등급을 끌어내립니다.

        (A) … 데이터수집부                          ← 도입
        (B) 상기 데이터수집부에 수집된 … 결함탐지부      ← 되받을 뿐
        (D) 상기 데이터수집부에 수집된 … 수평구조물검출부  ← 지시 대상은 A다

    B의 문언은 D가 쓴 어구를 통째로 품으므로, 긴 어구를 우선하면 D의 지시 대상이 B가 됩니다.
    "상기"가 앞에 붙었는지만 보면 그 자리가 도입인지 참조인지 바로 갈립니다.
    """
    start = text.find(target)
    while start != -1:
        if not text[:start].endswith("상기"):
            return True
        start = text.find(target, start + 1)
    return False


def _introducers(target: str, preceding: list[tuple[str, str, bool]]) -> list[str]:
    """이 어구를 도입한 구성 **전부**. 앞선 것이 먼저 옵니다.

    하나만 돌려주면 "이 어구를 도입한 구성이 여럿이라 어느 쪽인지 모른다"는 사실이 사라집니다.
    그 상태에서 첫 번째를 골라 등급 상한의 근거로 쓰면, 맞게 개시된 구성이 근거 없이 강등될
    수 있습니다 — 참조 누락(상한이 안 걸림)보다 나쁜 방향입니다.
    """
    named = [label for label, text, is_preamble in preceding
             if _introduces(target, text) and not is_preamble]
    if named:
        return named
    return [label for label, text, _ in preceding if _introduces(target, text)]


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
        if _introduces(target, text) and not is_preamble:
            return label
    return next((label for label, text, _ in preceding if _introduces(target, text)), None)


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

    **확인된 지시 관계만 상한의 근거입니다.** 어구 한가운데서 끊긴 연결이나 도입 구성이 둘
    이상인 연결은 청구항 문언에서 확인된 것이 아니라 공통 부분에서 미루어 짐작한 것이고, 그런
    짐작으로 등급을 내리면 **맞게 개시된 구성이 근거 없이 강등됩니다.** 참조를 놓치는 쪽은
    상한이 안 걸릴 뿐이지만 이쪽은 없는 결격을 만들어 내므로, 방향이 더 나쁩니다.

    짐작을 지우지는 않습니다. 청구항 문언 자체가 어긋나 있을 수도 있으므로(실측: 한 구성이
    세운 "가시 두상 영역"을 다음 구성이 "가시 두상 영상"으로 받아 적음) 그 사실을 노트로
    남겨 사람이 확인하게 합니다.
    """
    links = antecedents(claim)          # 확인된 연결만 담깁니다
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
