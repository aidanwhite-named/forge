"""청구항 분해. 라벨·번호·종속 관계는 전부 정규식으로 확정하고,
구성요소 중요도만 LLM에 1회 물어봅니다.

글자 수나 한국어 키워드로 "이건 주지관용 구성"을 추정하는 휴리스틱은 두지 않습니다.
그런 판정은 사건마다 달라져 코드로 고정하면 설명할 수 없는 결과가 나옵니다.
"""
import re

from .agy import run_cli
from .models import Claim, ClaimElement

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
무엇을 입증해야 하는지 알 수 있게 적으십시오. 특히 "~에 따라", "~을 이용하여", "~와 결합하여",
"동적으로" 같은 조건은 생략하지 마십시오.

[출력]
JSON 객체 하나만 출력하십시오.
{"elements": [{"claim_number": 1, "label": "A", "importance": 5, "is_sub": false,
  "limitations": ["입력 데이터를 공통 좌표계로 변환함", "변환된 데이터를 전역 장면과 결합함"]}]}

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


def assign_importance(claims: list[Claim]) -> list[str]:
    """구성요소 중요도를 LLM에 1회만 물어봅니다. 실패해도 기본값 3으로 진행합니다."""
    payload = [
        {"claim_number": claim.number, "label": element.label, "text": element.text}
        for claim in claims for element in claim.elements
    ]
    if not payload:
        return []
    import json
    try:
        raw = run_cli(IMPORTANCE_PROMPT + json.dumps(payload, ensure_ascii=False), expect="elements")
    except RuntimeError as exc:
        for claim in claims:
            for element in claim.elements:
                element.limitations = []
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
            limitations = []
            for value in item.get("limitations") or []:
                limitation = re.sub(r"\s+", " ", str(value or "")).strip().rstrip(";,")
                if limitation and limitation not in limitations:
                    limitations.append(limitation)
            # 분해 응답이 누락되면 비워 둡니다. 검증을 건너뛰는 것이 아니라, 구성 원문 한 줄을
            # 점검하는 폴백을 compare에서 한 번만 적용하기 위해서입니다. 여기서 원문을 채워
            # 넣으면 그것이 실제 분해 결과와 구분되지 않아, 구성 원문이 '누락된 하위 한정'으로
            # 보고서에 찍히고 누락 수도 이중으로 세어집니다.
            element.limitations = limitations[:12]
    return []


def input_quality_warnings(claims: list[Claim]) -> list[str]:
    """판정을 바꾸지 않고, 법적 의미를 흔들 수 있는 명백한 입력 이상만 경고합니다."""
    warnings: list[str] = []
    for claim in claims:
        for element in claim.elements:
            if re.search(r"각\s*영사(?:의|가)\s*포즈", element.text):
                warnings.append(
                    f"청구항 {claim.number} ({element.label})의 '각 영사의 포즈'는 "
                    "'각 영상의 포즈' 오탈자일 수 있습니다. 원문을 확인해 주세요."
                )
        if claim.raw.count("(") != claim.raw.count(")"):
            warnings.append(f"청구항 {claim.number}의 괄호 짝이 맞지 않습니다. 구성 분해 결과를 확인해 주세요.")
    return warnings
