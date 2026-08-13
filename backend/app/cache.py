"""비교 판정과 근거 의미검증 캐시.

파이프라인의 비싼 LLM 단계인 (청구항 × 문헌) 비교와 근거 의미검증 결과를 캐시합니다.
키는 청구항 구성 원문·문헌 청크·프롬프트 버전·모델의 해시라서, 히스토리를 지워도
같은 입력이면 같은 판정이 재사용됩니다. 캐시 디렉터리는 히스토리 밖에 둡니다.
"""
import hashlib
import json

from .config import COMPARE_SAMPLES, DATA_DIR, load_runtime_settings
from .models import Claim, Document, ElementMatch

CACHE_DIR = DATA_DIR / "comparison_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ENTAILMENT_CACHE_DIR = DATA_DIR / "entailment_cache"
ENTAILMENT_KEY_PREFIX = "entailment:"


def fingerprint(*parts: str) -> str:
    """프롬프트 문면에서 캐시 세대를 직접 뽑습니다. 손으로 올리는 버전 번호를 대신합니다.

    프롬프트를 고칠 때마다 사람이 버전 상수를 올리는 방식에는 두 가지 문제가 있습니다.
    하나는 **올리는 것을 잊으면 조용히 틀린다**는 것입니다 — 새 프롬프트로 받아야 할 판정
    자리에 옛 프롬프트의 판정이 그대로 재사용되고, 캐시에 들어간 뒤로는 무엇이 어느
    프롬프트에서 나왔는지 구별할 방법이 없습니다. 다른 하나는 번호가 늘어날수록 코드에
    변경 이력이 쌓인다는 것입니다.

    문면을 해시하면 둘 다 사라집니다. 프롬프트가 한 글자라도 달라지면 키가 저절로 갈리고,
    같으면 저절로 재사용됩니다. 사람이 관리할 값이 없으므로 잊을 것도, 남을 자국도 없습니다.

    대신 공백 한 칸을 고쳐도 캐시가 갈립니다. 그것이 맞는 동작입니다 — 프롬프트가 달라졌는데
    같은 판정을 재사용해도 되는지는 사람이 눈으로 가릴 수 있는 문제가 아닙니다.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]


def compare_generation() -> str:
    """구성대비 프롬프트의 세대. 의미검증 프롬프트는 별도로 셉니다(entailment).

    프롬프트를 소유한 모듈에서 읽어야 하는데 compare가 claims를 거쳐 이 모듈을 다시
    참조하므로, 순환 import를 피해 호출 시점에 읽고 결과를 재사용합니다.
    """
    global _COMPARE_GENERATION
    if _COMPARE_GENERATION is None:
        from . import compare
        _COMPARE_GENERATION = fingerprint(
            compare.COMPARE_PROMPT, compare.DOCUMENT_COMPARE_PROMPT,
            compare.BATCH_COMPARE_PROMPT, compare._JUDGMENT_LABELS,
            compare._UNTRUSTED_NOTICE)
    return _COMPARE_GENERATION


_COMPARE_GENERATION: str | None = None


def cache_key(claim: Claim, document: Document, guideline: str = "",
              context_budget: int = 0, parents: list[dict] | None = None,
              mode: str = "single", samples: int | None = None,
              cohort: list | None = None) -> str:
    """이 판정을 만들어 낸 프롬프트를 식별합니다.

    context_budget은 문헌 한 건을 프롬프트에 실을 문자 예산입니다. **반드시 키에 넣어야
    합니다.** 같은 (청구항 × 문헌)이라도 최초 분석은 문헌 전문에 가까운 예산으로, 종속항
    추가는 그보다 훨씬 좁은 예산으로 판정합니다. 이 값을 빼면 좁은 문맥에서 "대응 없음"으로
    떨어진 셀이 넓은 문맥으로 다시 볼 경로에서 그대로 재사용되어, 문헌에 기재가 있는데도
    없다는 판정이 캐시에 고착됩니다.

    parents는 종속항 프롬프트에 실리는 부모항 문언입니다. 같은 문언의 종속항이라도 부모항이
    다르면 "상기 …"의 대상이 달라져 판정이 달라지므로 함께 넣습니다.

    mode와 samples는 **호출부가 실제로 한 일**을 적습니다. 어긋나기 쉬운 자리가 셋입니다.
      - 일괄 경로는 CLI를 1회만 부르는데(compare.compare_claims_documents) 키에는 전역
        COMPARE_SAMPLES(=3)가 들어갔습니다.
      - 일괄 경로가 실제로 쓰는 문헌 예산은 DOCUMENT_BUDGET_CHARS // 문헌수인데 키에는
        DEPENDENT_DOCUMENT_BUDGET_CHARS가 들어갔습니다(문헌이 정확히 3건일 때만 우연히 일치).
      - 두 프롬프트는 [역할]·[출력] 스키마가 다르고 일괄 쪽은 형제 청구항·타 문헌까지 컨텍스트에
        싣습니다. 예산과 표본 수가 같아도 **같은 프롬프트가 아닙니다.**
    그 결과 "1표본 일괄 판정"과 "3표본 합의 단건 판정"이 같은 파일 이름을 놓고 서로를
    덮어썼습니다. mode를 필수 필드로 넣으면 digest가 전부 바뀌므로 그렇게 오염된 옛 키는
    도달 불가가 됩니다. 숫자 두 개만 고치면 단건 경로가 옛 오염분을 계속 읽습니다.
    """
    settings = load_runtime_settings()
    payload = {
        "version": compare_generation(),
        "provider": settings["provider"],
        "model": settings["model"],
        # 판단 지침을 바꾸면 판정이 달라지므로 키에 포함합니다.
        "guideline": (guideline or "").strip(),
        "context_budget": int(context_budget),
        "mode": str(mode),
        # 함께 묶여 한 프롬프트에 실린 항목들. 일괄 경로의 프롬프트에는 형제 청구항과 다른
        # 문헌이 함께 들어가므로, 같은 셀이라도 **누구와 묶였는지가 달라지면 다른 프롬프트**입니다.
        # 이것을 키에서 빼면 배치 크기를 바꾸거나 청구항을 더한 실행이 이전 묶음의 판정을 그대로
        # 재사용합니다. 단건 경로는 묶임이 없으므로 빈 값입니다.
        "cohort": sorted(str(item) for item in cohort or []),
        # 같은 셀이라도 1회 물어본 판정과 3회 다수결로 낸 판정은 다른 값입니다. 키에 넣지
        # 않으면 샘플링을 켠 실행이 1회짜리 캐시를 그대로 재사용해, 켠 효과가 사라집니다.
        "samples": int(samples if samples is not None else COMPARE_SAMPLES),
        "parents": parents or [],
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


# --- 청구항 분해 캐시 ---------------------------------------------------------
# 분해는 판정보다 앞선 단계인데도 캐시가 없어서, 같은 청구항을 다시 분석할 때마다 LLM이 새로
# 쪼갰습니다. 그 결과 두 가지가 함께 무너졌습니다.
#   - 재현성: 같은 입력에서 core 개수와 한정 수가 달라져 판정과 인용발명 선정까지 바뀝니다.
#   - 비용: 분해 결과(limitations·search_terms)가 비교 캐시 키에 그대로 들어가므로,
#     분해가 한 글자만 달라져도 (청구항 × 문헌) 판정 캐시가 **전량 미스**가 됩니다.
# 히스토리를 뒤져 가장 최근 것을 쓰는 방식은 쓰지 않습니다. 어느 것을 고를지가 정의되지 않고,
# 히스토리를 지우면 결과가 달라지기 때문입니다. 입력 자체를 키로 삼습니다.
DECOMPOSITION_CACHE_DIR = DATA_DIR / "decomposition_cache"


def decomposition_key(claims_text: str, version: str) -> str:
    """청구항 원문 + 분해 프롬프트 세대 + provider/model. 이 넷이 같으면 같은 분해를 씁니다."""
    settings = load_runtime_settings()
    payload = {"claims": (claims_text or "").strip(), "version": str(version),
               "provider": settings["provider"], "model": settings["model"]}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_decomposition(key: str) -> dict | None:
    path = DECOMPOSITION_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) and value.get("claims") else None
    except (OSError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None


def store_decomposition(key: str, value: dict) -> None:
    if not value or not value.get("claims"):
        return
    try:
        DECOMPOSITION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (DECOMPOSITION_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # 캐시는 최적화일 뿐이라 저장 실패로 분석을 멈추지 않습니다.


def clear() -> int:
    removed = 0
    for directory in (CACHE_DIR, ENTAILMENT_CACHE_DIR, DECOMPOSITION_CACHE_DIR):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
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
        if key.startswith(ENTAILMENT_KEY_PREFIX):
            path = ENTAILMENT_CACHE_DIR / f"{key.removeprefix(ENTAILMENT_KEY_PREFIX)}.json"
        else:
            path = CACHE_DIR / f"{key}.json"
        if path.exists():
            path.unlink(missing_ok=True)
            removed += 1
    return removed
