"""비교 판정 캐시.

파이프라인에서 유일하게 비싼 단계가 (청구항 × 문헌) 비교이므로 그 결과만 캐시합니다.
키는 청구항 구성 원문·문헌 청크·프롬프트 버전·모델의 해시라서, 히스토리를 지워도
같은 입력이면 같은 판정이 재사용됩니다. 캐시 디렉터리는 히스토리 밖에 둡니다.
"""
import hashlib
import json

from .config import DATA_DIR, load_runtime_settings
from .models import Claim, Document, ElementMatch

CACHE_DIR = DATA_DIR / "comparison_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# 프롬프트나 정규화 규칙을 바꾸면 이 값을 올려 과거 캐시를 무효화합니다.
# v13: "동일"·"실질적 동일"에 대상 대응 조건을 걸었습니다. 동작을 가리키는 낱말이 같아도
#      그 동작이 걸리는 대상이 다르면 등가가 아닙니다. 종전 프롬프트는 "용어만 다르며
#      기술적 의미와 작동 관계가 같음"이라고만 해서, 같은 분야에서 비슷한 목적을 가진
#      문장이면 대상이 달라도 실질적 동일로 올라갔습니다.
# v12: 선택적 한정("중 적어도 하나")을 대안 묶음으로 묶어 하나만 개시되면 충족으로 봅니다.
# v11: 하위 한정을 core/qualifier로 나누고, 열거 항목의 상위 개념 인정을 막았습니다.
# v10: 구성요소별 검색어를 프롬프트에 넣고, 판단 이유를 연결어미로 받도록 바꿨습니다.
# v9: "차이" 판정에도 가장 가까운 실제 원문을 evidence로 의무화했습니다.
# v8: 하위 제한이 하나도 개시되지 않은 응답을 대응 기재로 인정하지 않는 규칙을 추가했습니다.
# v7: 단건·일괄 경로가 같은 스키마(requirements + limitation_checks)를 요구하도록 통합했습니다.
# 두 경로는 이 키를 공유하므로, 버전을 올리지 않으면 느슨한 스키마로 만든 셀이 엄격한
# 경로의 재실행에서 그대로 재사용되어 불일치가 캐시에 영구 고착됩니다.
PROMPT_VERSION = "compare-v13-operand-correspondence"


def cache_key(claim: Claim, document: Document, guideline: str = "") -> str:
    settings = load_runtime_settings()
    payload = {
        "version": PROMPT_VERSION,
        "provider": settings["provider"],
        "model": settings["model"],
        # 판단 지침을 바꾸면 판정이 달라지므로 키에 포함합니다.
        "guideline": (guideline or "").strip(),
        # search_terms는 프롬프트에 그대로 들어가 판정을 바꾸므로 키에 포함합니다.
        "claim": [{"label": element.label, "text": element.text,
                   "limitations": [limitation.model_dump() for limitation in element.limitations],
                   "search_terms": element.search_terms} for element in claim.elements],
        "preamble": claim.preamble,
        "document": [{"chunk_id": chunk.chunk_id, "text": chunk.text} for chunk in document.chunks],
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def load(key: str) -> list[ElementMatch] | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [ElementMatch.model_validate(item) for item in raw]
    except (OSError, json.JSONDecodeError, ValueError):
        path.unlink(missing_ok=True)
        return None


def store(key: str, matches: list[ElementMatch]) -> None:
    try:
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps([match.model_dump() for match in matches], ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # 캐시는 최적화일 뿐이라 저장 실패로 분석을 멈추지 않습니다.


def clear() -> int:
    removed = 0
    for path in CACHE_DIR.glob("*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def discard(keys) -> int:
    """지정한 판정만 캐시에서 지웁니다.

    분석을 삭제할 때 씁니다. 판정 캐시에는 업로드한 문헌의 **원문 발췌**가 그대로 들어
    있으므로, 히스토리만 지우고 이것을 남기면 사용자가 지웠다고 생각한 문헌의 문장이
    디스크에 계속 남습니다. 키는 그 분석이 실제로 사용한 것만 받습니다 — 다른 분석과
    공유되는 키를 지워도 판정이 틀려지지는 않고 다음 실행에서 다시 받을 뿐입니다.
    """
    removed = 0
    for key in set(keys or ()):
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            path.unlink(missing_ok=True)
            removed += 1
    return removed
