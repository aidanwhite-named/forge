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
# 프롬프트나 정규화 규칙을 바꾸면 이 값을 올려 과거 캐시를 무효화합니다.
# 이 값은 **구성대비 프롬프트만** 가릅니다. 근거 의미검증 프롬프트는 별도 버전을
# 가지므로(entailment.PROMPT_VERSION) 양쪽을 함께 고쳤으면 둘 다 올려야 합니다.
# 올려도 옛 캐시 파일이 지워지지는 않고 도달만 불가해집니다. 디스크를 비우려면
# DELETE /api/cache(cache.clear)를 쓰십시오.
# v21: 네 축(동작·대상·집합성·인과관계)을 **독립으로** 판정하게 했습니다. v20은 '생성·전달 ≠
#      획득'을 목록으로 줬는데, 문헌이 획득을 문언으로 명시한 셀까지 '동작 축 결손'으로
#      기각해 그 문헌이 조합에서 통째로 빠졌습니다. 축을 나눠 물으면 대상 차이는 대상
#      사유로, 동작 부재는 동작 사유로 갈립니다.
# v20: 역할 환원이 명칭을 넘어 동작·대상·집합성·인과관계까지 지우는 것을 막습니다.
#      실측에서 '광원이 이미지 광을 도파관으로 출력한다'가 '입력 광학 이미지들의 세트를
#      획득함'의 근거로 통과해 그 구성이 '실질적 동일 4/4'로 보고되었습니다.
# v19: 청크 페이로드에서 page·paragraph를 뺐습니다(chunk_id가 같은 값을 담고 있습니다).
#      section은 값이 있을 때만 싣습니다. 프롬프트 문자 수가 실측 12.6% 줄어듭니다.
#      함께, 키에 mode를 넣어 일괄(1표본)과 단건(다표본) 판정을 갈랐습니다. v18 이하의 셀은
#      두 경로가 한 키를 공유해 서로를 덮어쓴 상태라 그대로 재사용할 수 없습니다.
# v18: 판정 등급을 코드가 산출합니다(coverage.derive_judgment). 모델은 라벨 대신 terminology와
#      different_purpose만 답합니다. v17 이하의 캐시 셀은 **모델이 고른 라벨**을 담고 있어
#      그대로 재사용하면 산출된 등급과 섞입니다.
# v17: 등급을 고르기 전에 한정을 '역할'로 바꿔 문헌에서 그 역할을 하는 구성을 먼저 찾도록
#      대응 관계 탐색 단계를 넣었습니다. 종전에는 청구항 전용 명칭(스위치 박스·리미트 스위치)이
#      문헌 용어(제1 몸체·스위치 버튼)와 다르다는 이유로 탐색이 멈춰, 같은 역할을 원문으로
#      개시한 문헌이 "차이"·발췌 없음으로 떨어지고 보조 인용발명 자격까지 잃었습니다.
#      함께, 한 인과 사슬을 형제 한정끼리 나눠 담지 말고 공통 문장은 양쪽에 모두 넣도록
#      요구합니다. 한정별로만 읽는 의미검증이 그 쪼갬 때문에 개시된 한정을 기각했습니다.
# v16: 관계형 한정에서 수단·동작·인과 연결을 완성하는 문장을 evidence에 모두 요구합니다.
# v15: 한정마다 복수 문단의 근거 묶음을 요구하고 문헌 유형을 의미판정 입력에서 제거했습니다.
#      v14 캐시에는 이 묶음이 없어 독립 entailment 검증을 온전히 수행할 수 없습니다.
# v14: 문헌을 끝까지 훑고 가장 직접적인 실시 기재를 고르도록 발췌 선택 규칙을 넣었고,
#      종속항 프롬프트에 부모항 문언(parent_claims)을 실었습니다. 둘 다 판정을 바꿉니다.
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
PROMPT_VERSION = "compare-v21-independent-axes"


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

    mode와 samples는 **호출부가 실제로 한 일**을 적습니다. 종전에는 셋 다 어긋나 있었습니다.
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
        "version": PROMPT_VERSION,
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


def decomposition_key(claims_text: str, version: int) -> str:
    """청구항 원문 + 분해 버전 + provider/model. 이 넷이 같으면 같은 분해를 씁니다."""
    settings = load_runtime_settings()
    payload = {"claims": (claims_text or "").strip(), "version": int(version),
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
