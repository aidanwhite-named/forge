"""청구항 분해. 라벨·번호·종속 관계는 전부 정규식으로 확정하고,
구성요소 중요도만 LLM에 1회 물어봅니다.

글자 수나 특정 어휘로 구성의 성격을 추정하는 휴리스틱은 두지 않습니다.
그런 판정은 사건마다 달라져 코드로 고정하면 설명할 수 없는 결과가 나옵니다.
"""
import json
import re
from pathlib import Path

from . import cache
from .agy import run_cli
from .config import DECOMPOSITION_FILE, FORCE_REDECOMPOSE
from .models import Claim, ClaimElement, Limitation

_CLAIM_HEADER = re.compile(
    r"(?:^|\n)\s*(?:(?:\d+)\s*[.)]\s*)?"
    r"(?:【\s*청구항\s*(\d+)\s*】|\[\s*(?:CLAIM|청구항)\s*(\d+)\s*\]|청구항\s*(\d+)\s*[.:]?|제\s*(\d+)\s*항\s*[.:])",
    re.IGNORECASE,
)
_DEPENDENCY = re.compile(
    r"(?:제\s*(\d+)\s*항|청구항\s*(\d+))\s*(?:내지\s*(?:제\s*\d+\s*항|\d+)\s*)?(?:중\s*(?:어느\s*)?한\s*항\s*)?에\s*있어서",
)
_DEPENDENCY_EN = re.compile(r"\bof\s+claim\s+(\d+)|\baccording\s+to\s+claim\s+(\d+)", re.IGNORECASE)
_PREAMBLE_SPLIT = re.compile(r"에\s*있어서\s*[,.]?")
_LABEL_SPLIT = re.compile(r"(?=\(\s*[A-Za-z]{1,2}\s*\))")
_LABEL_HEAD = re.compile(r"^\(\s*([A-Za-z]{1,2})\s*\)\s*")

IMPORTANCE_PROMPT = """[역할]
특허 심사관으로서 청구항 구성요소의 상대적 중요도만 매깁니다. 인용발명 비교는 하지 않습니다.

[중요도 기준]
5 = 이 발명을 다른 발명과 구별짓는 차별적 핵심 구성(고유한 구조 관계, 제어 조건, 수치 한정, 상호작용)
4 = 발명의 목적 달성에 직접 기여하는 주요 구성
3 = 일반적인 처리·연산 구성
2 = 통상의 인터페이스·입출력 구성
1 = 어느 발명에나 나타나는 범용 부품(메모리, 프로세서, 전원부 등)

나열 순서는 중요도의 근거가 아닙니다. 첫 번째로 기재된 입출력 구성이라도 범용이면 1~2를 주십시오.
is_sub는 앞선 구성에 붙는 하위 제한(수치·조건·재질 한정)이면 true입니다.

limitations에는 해당 구성에서 독립적으로 입증해야 하는 구조·기능·조건·입력·처리·출력·결합관계를
빠짐없이 원자적 문장으로 나누어 적으십시오. 단순한 문법 조각으로 쪼개지 말고, 각 항목만 읽어도
무엇을 입증해야 하는지 알 수 있게 적으십시오.

각 항목의 kind는 다음 둘 중 하나입니다. **이 구분이 이후 판정 등급을 가릅니다.**
- core: 그 구성이 실제로 무엇을 하는가. 동작·구조·데이터 흐름 자체(무엇을 입력받아 무엇을
  처리하고 무엇을 내보내는가, 어떤 구성요소와 어떻게 연결되는가).
- qualifier: 그 동작을 한정하는 기준·조건·파라미터·수치·명칭(무엇을 기준으로, 얼마 이상일 때,
  어떤 이름의 처리부가).

**core 하나에 독립적으로 확인할 수 있는 두 동작을 합치지 마십시오.** "A를 획득하고 B를 획득함"
처럼 문헌에서 각각 따로 확인할 수 있는 동작은 core 두 개로 나눕니다. 합쳐 두면 한쪽만 개시한
문헌이 그 구성 전체를 미개시로 받아, 절반을 실제로 개시한 문헌과 아무것도 개시하지 않은 문헌이
같은 판정이 됩니다. 반대로 한 동작을 조건·시점·목적별로 쪼개 여러 core로 만들지도 마십시오 —
그것은 qualifier입니다.

**core 항목에 qualifier 문구를 섞어 쓰지 마십시오.** 예를 들어 "수요 지표가 임계값 미만이면
콘텐츠를 외부 저장소로 이전한다"는 구성은 core "콘텐츠를 내부 저장소에서 외부 저장소로
이전하여 보관함"과 qualifier "이전 여부를 수요 지표와 임계값의 비교로 결정함"으로 나눕니다.
한 항목에 둘을 섞으면 조건 하나가 달라졌을 뿐인데 동작까지 미개시로 처리됩니다.
"~에 따라", "~을 이용하여", "~와 결합하여", "동적으로" 같은 조건은 생략하지 말고 qualifier로
적으십시오.

**같은 조건을 core와 qualifier에 중복해서 적지 마십시오.** 특히 시간·시점·트리거·판단 기준을
나타내는 "~하는 도중", "~에 근거하여", "~에 따라", "업데이트된 ~을 기준으로"는 core에서
제외하고 qualifier에 한 번만 적으십시오. 예를 들어 "편집 중 처리시간 정보를 업데이트함"은
core "처리시간 정보를 업데이트함"과 qualifier "업데이트 시점을 편집 수행 중으로 한정함"으로,
"업데이트된 처리시간에 근거하여 버퍼 개수를 조정함"은 core "버퍼 개수를 조정함"과 qualifier
"조정 기준을 업데이트된 처리시간으로 한정함"으로 나눕니다. core에 조건을 포함한 문장 전체를
적고 qualifier에 같은 조건을 다시 적으면, 조건 하나의 미개시가 동작과 조건 두 항목을 동시에
실패시켜 실제보다 낮은 판정이 됩니다.

core에는 기본 동작과 그 동작의 필수 대상·입력·출력의 **정체성**을 남기십시오. 반면 그 동작을
언제·무엇을 기준으로·어떤 용도로 수행하는지는 발명의 변별점이어도 qualifier입니다. qualifier가
빠지면 "일부 차이" 이하로 내려가므로 변별력이 사라지는 것이 아닙니다. 예를 들어
"조회수·체류 시간·완주율 중 적어도 하나를 포함하는 지표를 산출함"에서는 산출 대상인 지표의
종류가 동작의 정체성이므로 core 대안으로 남기지만, "처리시간 정보에 근거하여 편집용 버퍼
개수를 결정함"에서는 core "버퍼 개수를 결정함", qualifier "결정 기준을 처리시간 정보로
한정함", qualifier "결정된 버퍼의 용도를 편집용으로 한정함"으로 나눕니다.

다만 **대상 자체가 다른 구성요소와의 관계로 정의되는 경우** 그 관계는 단순 목적 qualifier가
아니라 core의 정체성입니다. "학습된 모델에 입력되어 그 모델로부터 목표 영상을 출력시키는
보정 영상을 획득함"을 core "보정 영상을 획득함"과 qualifier "학습된 모델에 입력함"으로
떼지 마십시오. 그렇게 쪼개면 아무 보정행렬이나 얻는 문헌이 핵심 동작을 개시한 것으로
올라갑니다. 이 경우에는 "학습된 모델에 입력되어 그 모델로부터 목표 영상을 출력시키는 보정
영상을 획득함"을 하나의 core로 남기고, 학습 완료 **시점**이나 목표 영상의 구체적 균일도 값만
qualifier로 분리하십시오.

같은 원칙은 모델·프로세서·제어부의 정체성에도 적용합니다. "광학계를 모델링하는 뉴럴
네트워크를 학습함"에서 학습되는 객체가 무엇을 모델링하는지는 그 객체의 정체성을 정합니다.
문헌의 한 모델은 광학계를 모사하고 다른 뉴럴 네트워크는 영상을 생성하는 경우, 둘을 합쳐
청구된 하나의 뉴럴 네트워크로 읽어서는 안 됩니다. 독립 동작을 원자화하되 **같은 주체에
동시에 귀속되어야 하는 역할과 입력→출력 방향**은 core에서 끊지 마십시오.

여러 대안에 공통인 문구는 각 대안에 되풀이하지 마십시오. "디코딩·비디오·인코딩·디스플레이·
전송 처리시간 중 적어도 하나에 근거하여 편집용 버퍼 개수를 결정함"은 처리시간 종류만 대안
묶음으로 만들고, "이에 근거함"과 "편집용"은 각각 공통 qualifier 한 항목으로 한 번만 둡니다.
공통 목적 하나가 없다는 이유로 대안 다섯 개가 동시에 실패하면 같은 차이를 중복 계상한 것입니다.

**선택적 한정은 alternative_group으로 묶으십시오.** "A, B 또는 C 중 적어도 하나", "~중 어느
하나", "또는"으로 열거된 항목은 서로 대안이므로 **하나만 개시되면 그 묶음 전체가 충족**됩니다.
각 항목을 따로 적되 같은 묶음 이름(예: "지표종류", "토큰속성")을 부여하십시오. 묶지 않으면
선택지를 넉넉히 나열한 청구항일수록 차이점이 길어져, 문언을 충족하는 문헌이 오히려 감점됩니다.
반대로 "A와 B를 모두" 또는 "A하고 B하는"처럼 병렬로 요구되는 항목은 묶지 말고 빈 값으로
두십시오. 확실하지 않으면 묶지 마십시오.

search_terms에는 이 구성의 대응 기재를 문헌 본문에서 찾기 위한 검색어를 5~12개 적으십시오.
청구항이 한국어라도 인용발명은 영어·일본어·중국어 공보일 수 있으므로, 그 기술 개념을 해당
분야에서 실제로 쓰는 **영어 표현**을 반드시 함께 넣으십시오. 청구항에 쓰인 표현뿐 아니라
같은 개념의 통용 표현(상위어·업계 관용어)도 넣어야 표현이 다른 문헌을 놓치지 않습니다.

[출력]
JSON 객체 하나만 출력하십시오.
{"elements": [{"claim_number": 1, "label": "A", "importance": 5, "is_sub": false,
  "limitations": [{"text": "입력 데이터를 공통 좌표계로 변환함", "kind": "core"},
                  {"text": "변환된 데이터를 전역 장면과 결합함", "kind": "core"},
                  {"text": "깊이 센서로 입력을 취득함", "kind": "core", "alternative_group": "입력수단"},
                  {"text": "스테레오 카메라로 입력을 취득함", "kind": "core", "alternative_group": "입력수단"},
                  {"text": "변환 대상을 프레임별 신뢰도로 선별함", "kind": "qualifier"}],
  "search_terms": ["공통 좌표계", "common coordinate system", "world coordinate", "registration"]}]}

[구성요소]
"""


def parse_claims(claims_text: str) -> list[Claim]:
    """입력 전문을 청구항 단위로 나눕니다. 헤더가 없으면 전체를 청구항 1로 봅니다."""
    text = claims_text.replace("\r\n", "\n").strip()
    if not text:
        return []
    matches = list(_CLAIM_HEADER.finditer(text))
    blocks: list[tuple[int, str]] = []
    if matches:
        for index, found in enumerate(matches):
            number = int(next(group for group in found.groups() if group))
            start = found.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            blocks.append((number, text[start:end].strip()))
    else:
        blocks.append((1, text))
    claims = [_build_claim(number, body) for number, body in blocks if body]
    return _resolve_chains(claims)


# (A)~(ZZ) 자동 라벨은 영문자만 쓰므로 숫자가 섞인 이 라벨과는 절대 충돌하지 않습니다.
PREAMBLE_LABEL = "P0"


def _build_claim(number: int, body: str) -> Claim:
    preamble, remainder = _split_preamble(body)
    element = _preamble_element(preamble)
    return Claim(number=number, preamble=preamble,
                 elements=([element] if element else []) + _split_elements(remainder),
                 depends_on=_dependency(body), raw=body)


def _preamble_element(preamble: str) -> ClaimElement | None:
    """전제부에 기술적 한정이 남아 있으면 판정 대상 구성으로 세웁니다.

    전제부를 판정에서 빼면, 제한적 전제부가 인용발명에 없어도 나머지 구성만 같으면
    신규성이 부정됩니다. 전제부가 한정적인지 여부는 사건마다 다르므로 코드가 추정하지
    않고, 다른 구성과 똑같이 대비시킨 뒤 결과를 보고서에 그대로 노출합니다.

    다만 종속항의 "제1항에 있어서"는 의존 관계 표시일 뿐 기술 내용이 아닙니다. 그것까지
    구성으로 세우면 모든 종속항이 대응 없는 구성을 하나씩 달고 시작합니다.
    """
    text = _DEPENDENCY_EN.sub("", _DEPENDENCY.sub("", preamble))
    text = re.sub(r"\s+", " ", text).strip().strip(",.;· ")
    if len(text) < 2:
        return None
    return ClaimElement(label=PREAMBLE_LABEL, text=text, is_preamble=True)


def _split_preamble(body: str) -> tuple[str, str]:
    """"…에 있어서" 앞부분을 전제부로 떼어냅니다. 없으면 첫 라벨 앞을 씁니다."""
    found = _PREAMBLE_SPLIT.search(body)
    if found:
        return re.sub(r"\s+", " ", body[:found.end()]).strip(), body[found.end():]
    parts = [part for part in _LABEL_SPLIT.split(body) if part.strip()]
    if parts and not _LABEL_HEAD.match(parts[0].strip()) and len(parts) > 1:
        return re.sub(r"\s+", " ", parts[0]).strip(), body[len(parts[0]):]
    return "", body


def _split_elements(body: str) -> list[ClaimElement]:
    """(A)~(Z) 라벨이 있으면 라벨로만 나눕니다.

    줄바꿈으로 쪼개면 여러 줄에 걸친 하나의 구성이 둘로 갈려 이후 라벨이 전부 밀립니다.
    라벨이 없을 때만 문단 단위로 나누고 A부터 순서대로 부여합니다.
    """
    parts = [part for part in _LABEL_SPLIT.split(body) if part.strip()]
    labeled = [part.strip() for part in parts if _LABEL_HEAD.match(part.strip())]
    if labeled:
        elements = []
        for part in labeled:
            label = _LABEL_HEAD.match(part).group(1).upper()
            text = re.sub(r"\s+", " ", _LABEL_HEAD.sub("", part)).strip().rstrip(";,")
            if text:
                elements.append(ClaimElement(label=label, text=text))
        return elements
    blocks = [block for block in re.split(r"\n\s*\n|;\s*\n|\n(?=\s*(?:상기|여기서|이때))", body) if block.strip()]
    if len(blocks) == 1:
        blocks = [line for line in body.split("\n") if len(line.strip()) > 10]
    return [ClaimElement(label=auto_label(index), text=re.sub(r"\s+", " ", block).strip().rstrip(";,"))
            for index, block in enumerate(blocks) if block.strip()]


def auto_label(index: int) -> str:
    """A…Z를 넘어가면 AA, AB…로 이어 붙입니다."""
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _dependency(body: str) -> int | None:
    found = _DEPENDENCY.search(body) or _DEPENDENCY_EN.search(body)
    if not found:
        return None
    number = next((group for group in found.groups() if group), None)
    return int(number) if number else None


def _resolve_chains(claims: list[Claim]) -> list[Claim]:
    """존재하지 않는 부모항을 가리키는 종속 관계는 끊습니다.

    부모항이 입력되지 않은 종속항은 "에 있어서" 뒤의 직접 기재 구성만 대비하게 됩니다.
    """
    numbers = {claim.number for claim in claims}
    for claim in claims:
        if claim.depends_on == claim.number or claim.depends_on not in numbers:
            claim.depends_on = None
    return claims


def ancestry(claims: list[Claim], number: int) -> list[int]:
    """종속항이 상속하는 부모항 번호를 위쪽부터 반환합니다."""
    by_number = {claim.number: claim for claim in claims}
    chain: list[int] = []
    current = by_number.get(number)
    seen = {number}
    while current and current.depends_on and current.depends_on not in seen:
        chain.insert(0, current.depends_on)
        seen.add(current.depends_on)
        current = by_number.get(current.depends_on)
    return chain


# 저장된 분해 결과의 형식 버전. 필드 구성뿐 아니라 원자 한정 분해 규칙이 바뀌어도 올립니다.
# v5: 같은 모델·구성요소의 정체성을 이루는 역할과 입력→출력 관계를 core에 보존합니다.
#     v4는 "보정 이미지를 획득함"만 core로 떼고 "학습된 NN에 입력되어 목표 영상을 출력시킴"을
#     qualifier로 보냈습니다. 그 결과 NN이 전혀 없는 보정행렬 문헌이 핵심 동작을 개시한 것으로
#     올라가, 관계 전체가 없는 구성도 '일부 유사'가 되었습니다.
# v4: core 하나에 독립적으로 확인 가능한 두 동작을 합치지 않도록 원자성을 요구합니다. 같은
#     청구항이 실행마다 다르게 분해되어(입력 획득 + 출력 획득 → core 1개로 병합) 판정과
#     문헌 선정까지 흔들린 사례가 있었습니다.
# v3: 공통 목적·용도 문구를 선택지마다 반복하지 않고, 동작·기준·용도·대안을 분리합니다.
# v2: 시점·트리거·판단 기준을 core에 섞고 qualifier에 다시 적던 중복 분해를 금지합니다.
#     같은 조건 하나가 빠졌다는 이유로 핵심 동작까지 함께 미개시가 되는 판정을 바로잡습니다.
DECOMPOSITION_VERSION = 5


def _pinned_decomposition() -> dict | None:
    """FORGE_DECOMPOSITION_FILE로 고정한 분해. 없거나 읽을 수 없으면 None입니다."""
    if not DECOMPOSITION_FILE:
        return None
    try:
        value = json.loads(Path(DECOMPOSITION_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("claims") else None


def assign_importance(claims: list[Claim], decomposition: dict | None = None,
                      claims_text: str = "", *,
                      pinned_decomposition: dict | None = None) -> list[str]:
    """구성요소 중요도·하위 한정·검색어를 받습니다. 실패해도 기본값 3으로 진행합니다.

    decomposition은 이 분해 결과를 읽고 쓰는 저장소입니다. 이미 분해된 청구항은 그대로
    되살리고 남은 것만 LLM에 물어봅니다. pinned_decomposition은 회귀 실행처럼 환경변수와
    무관하게 사건별 분해를 고정해야 할 때만 사용합니다.

    저장하는 이유는 호출 한 번을 아끼는 데 있지 않습니다. 이 분해 결과(limitations,
    search_terms)가 비교 캐시 키에 그대로 들어가는데(cache.cache_key), 같은 청구항을 다시
    분해하면 같은 뜻의 문장이 한두 글자 다르게 나오고 그것만으로 이전 판정 캐시가 전부
    무효가 됩니다. 그러면 취소 후 재시도가 처음부터 다시 돌고, "같은 입력이면 같은 결과"도
    성립하지 않습니다. 분해를 고정해야 캐시가 실제로 동작합니다.
    """
    notes: list[str] = []
    # 우선순위: 고정 파일 > 이 작업이 이미 쓰던 분해 > 입력 해시 캐시 > LLM.
    # 고정 파일이 가장 앞인 이유는 그것이 실험 통제 장치이기 때문입니다 — 켜 두었으면 다른
    # 어떤 경로도 그것을 밀어내서는 안 됩니다.
    # 인자로 받은 것이 환경변수보다 앞섭니다. 회귀 하니스는 한 프로세스에서 여러 사건을
    # 돌리는데, 환경변수는 import 시점에 한 번만 읽히므로 두 번째 사건부터 듣지 않습니다.
    explicitly_pinned = pinned_decomposition is not None
    source = ("회귀 사건의 고정 분해" if explicitly_pinned
              else f"고정 파일({DECOMPOSITION_FILE})")
    pinned = pinned_decomposition if explicitly_pinned else _pinned_decomposition()
    restored = {claim.number for claim in claims
                if _restore_elements(claim, pinned, strict_version=False)}
    if restored:
        notes.append(f"청구항 {', '.join(str(number) for number in sorted(restored))}의 구성 분해를 "
                     f"{source}에서 읽었습니다. 이 보고서의 분해는 자동 생성된 것이 아닙니다.")
    restored |= {claim.number for claim in claims
                 if claim.number not in restored and _restore_elements(claim, decomposition)}

    shared_key = ""
    if claims_text and not FORCE_REDECOMPOSE:
        shared_key = cache.decomposition_key(claims_text, DECOMPOSITION_VERSION)
        shared = cache.load_decomposition(shared_key)
        restored |= {claim.number for claim in claims
                     if claim.number not in restored and _restore_elements(claim, shared)}

    pending = [claim for claim in claims if claim.number not in restored]
    warnings = _request_importance(pending) if pending else []
    # 분해를 받지 못한 청구항이 있으면 그 실행의 분해는 온전하지 않으므로 공유 캐시에 넣지
    # 않습니다. 빈 분해가 고착되면 이후 모든 실행이 그것을 재사용합니다.
    if shared_key and not warnings and claims_text:
        cache.store_decomposition(shared_key, dump_decomposition(claims))
    warnings = notes + warnings
    if decomposition is not None:
        # 분해를 받지 못한 청구항(기본값으로 진행)은 저장하지 않습니다. 저장하면 그 빈
        # 분해가 고정되어 다음 실행에서도 계속 재사용됩니다.
        keep = claims if not warnings else [claim for claim in claims if claim.number in restored]
        if keep:
            decomposition.update(dump_decomposition(keep, decomposition))
    return warnings


def dump_decomposition(claims: list[Claim], existing: dict | None = None) -> dict:
    """분해 결과를 저장 가능한 형태로 옮깁니다. 기존 기록은 유지하고 덮어씁니다."""
    stored = dict((existing or {}).get("claims") or {})
    for claim in claims:
        stored[str(claim.number)] = [
            {"label": element.label, "text": element.text, "importance": element.importance,
             "is_sub": element.is_sub, "search_terms": list(element.search_terms),
             "limitations": [limitation.model_dump() for limitation in element.limitations]}
            for element in claim.elements
        ]
    return {"version": DECOMPOSITION_VERSION, "claims": stored}


def _restore_elements(claim: Claim, decomposition: dict | None,
                      strict_version: bool = True) -> bool:
    """저장된 분해를 청구항에 되씌웁니다. 하나라도 어긋나면 아무것도 바꾸지 않습니다.

    구성 원문까지 대조합니다. 청구항 문언이 바뀌었는데 라벨만 보고 예전 분해를 씌우면,
    보고서에는 새 문언이 실리고 판정은 옛 한정을 기준으로 내려집니다.

    strict_version=False는 **고정 분해 파일 전용**입니다. 실험은 대개 분해 버전을 올린 뒤에
    하는데, 버전으로 막으면 정작 비교 대상인 이전 분해를 쓸 수 없습니다. 구성 원문 대조는
    이때도 그대로 하므로 다른 청구항의 분해가 잘못 씌워지지는 않습니다.
    """
    if not decomposition:
        return False
    if strict_version and decomposition.get("version") != DECOMPOSITION_VERSION:
        return False
    stored = (decomposition.get("claims") or {}).get(str(claim.number))
    if not isinstance(stored, list) or len(stored) != len(claim.elements) or not claim.elements:
        return False
    by_label = {str(item.get("label", "")).strip().upper(): item
                for item in stored if isinstance(item, dict)}
    payloads = []
    for element in claim.elements:
        item = by_label.get(element.label.upper())
        if item is None or str(item.get("text", "")) != element.text:
            return False
        payloads.append(item)
    try:
        restored = [[Limitation.model_validate(value) for value in item.get("limitations") or []]
                    for item in payloads]
    except (TypeError, ValueError):
        return False
    for element, item, limitations in zip(claim.elements, payloads, restored):
        try:
            element.importance = max(1, min(5, int(item.get("importance", 3))))
        except (TypeError, ValueError):
            element.importance = 3
        element.is_sub = bool(item.get("is_sub"))
        element.search_terms = _unique_strings(item.get("search_terms"))[:16]
        element.limitations = limitations[:12]
    return True


def _request_importance(claims: list[Claim]) -> list[str]:
    """구성요소 중요도를 LLM에 1회만 물어봅니다."""
    payload = [
        {"claim_number": claim.number, "label": element.label, "text": element.text}
        for claim in claims for element in claim.elements
    ]
    if not payload:
        return []
    try:
        raw = run_cli(IMPORTANCE_PROMPT + json.dumps(payload, ensure_ascii=False), expect="elements")
    except RuntimeError as exc:
        for claim in claims:
            for element in claim.elements:
                element.limitations = []
                element.search_terms = []
        return [f"구성요소 중요도를 받지 못해 전부 기본값(3)으로 진행했습니다: {exc}"]
    by_key = {}
    for item in raw.get("elements") or []:
        if isinstance(item, dict):
            by_key[(str(item.get("claim_number", "")), str(item.get("label", "")).strip().upper())] = item
    for claim in claims:
        for element in claim.elements:
            item = by_key.get((str(claim.number), element.label.upper())) or {}
            try:
                element.importance = max(1, min(5, int(item.get("importance", 3))))
            except (TypeError, ValueError):
                element.importance = 3
            element.is_sub = bool(item.get("is_sub"))
            element.search_terms = _unique_strings(item.get("search_terms"))[:16]
            limitations = _build_limitations(item.get("limitations"))
            # 분해 응답이 누락되면 비워 둡니다. 검증을 건너뛰는 것이 아니라, 구성 원문 한 줄을
            # 점검하는 폴백을 compare에서 한 번만 적용하기 위해서입니다. 여기서 원문을 채워
            # 넣으면 그것이 실제 분해 결과와 구분되지 않아, 구성 원문이 '누락된 하위 한정'으로
            # 보고서에 찍히고 누락 수도 이중으로 세어집니다.
            element.limitations = limitations[:12]
    return []


def _build_limitations(values) -> list[Limitation]:
    """분해 응답을 core/qualifier가 붙은 한정 목록으로 만듭니다.

    kind가 없거나 알 수 없는 값이면 core로 둡니다. qualifier로 잘못 넣으면 실제 동작이
    빠졌는데도 "조건만 다르다"로 완화되므로, 불확실할 때는 엄격한 쪽을 택합니다.
    """
    limitations: list[Limitation] = []
    seen: set[str] = set()
    for value in values or []:
        raw = value if isinstance(value, dict) else {"text": value}
        text = re.sub(r"\s+", " ", str(raw.get("text") or "")).strip().rstrip(";,")
        if not text or text in seen:
            continue
        seen.add(text)
        kind = str(raw.get("kind") or "").strip().lower()
        limitations.append(Limitation(
            text=text, kind="qualifier" if kind == "qualifier" else "core",
            alternative_group=re.sub(r"\s+", " ", str(raw.get("alternative_group") or "")).strip()))
    return _drop_lone_groups(limitations)


def _drop_lone_groups(limitations: list[Limitation]) -> list[Limitation]:
    """혼자뿐인 대안 묶음은 묶음 표시를 지웁니다.

    항목이 하나면 그것은 대안이 아니라 단독 필수 한정입니다. 표시를 남겨 두면 그 한정이
    미개시일 때도 "묶음이 충족되지 않았을 뿐"으로 읽혀 누락 판정이 흐려집니다.
    """
    counts: dict[str, int] = {}
    for limitation in limitations:
        if limitation.alternative_group:
            counts[limitation.alternative_group] = counts.get(limitation.alternative_group, 0) + 1
    for limitation in limitations:
        if counts.get(limitation.alternative_group, 0) < 2:
            limitation.alternative_group = ""
    return limitations


def _unique_strings(values) -> list[str]:
    cleaned: list[str] = []
    for value in values or []:
        text = re.sub(r"\s+", " ", str(value or "")).strip().rstrip(";,")
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def input_quality_warnings(claims: list[Claim]) -> list[str]:
    """판정을 바꾸지 않고, 구성 분해를 어긋나게 할 수 있는 입력 이상만 경고합니다.

    특정 오탈자 목록을 코드에 심지 않습니다. 사건마다 달라 유지될 수 없고, 심사관이
    이미 읽고 있는 원문을 대신 판단하는 일이 됩니다.
    """
    return [f"청구항 {claim.number}의 괄호 짝이 맞지 않습니다. 구성 분해 결과를 확인해 주세요."
            for claim in claims if claim.raw.count("(") != claim.raw.count(")")]
