"""미커버 구성 선행기술 웹 검색.

분석 파이프라인에 자동으로 끼워 넣지 않고 별도 요청으로만 실행합니다.
CLI가 외부 웹에 나가는 단계이므로, 사용자가 명시적으로 눌렀을 때만 동작해야 합니다.
"""
import json
import re
import ssl
from concurrent.futures import ThreadPoolExecutor

import httpx

from .agy import AnalysisCancelled, run_cli
from .config import PRIOR_ART_VERIFY_TIMEOUT
from .models import PriorArtHit

PRIOR_ART_PROMPT = """[역할]
심사 중 인용발명으로 커버되지 않은 청구항 구성에 대해서만 공개된 선행기술을 웹에서 찾습니다.

[규칙]
- 각 결과의 claim_number와 label은 대응하는 입력값을 그대로 반환하십시오.
- 특허공보(공개번호가 확인되는 것)와 학술논문(정식 제목이 확인되는 것)만 제시하십시오.
- 최초 공개일을 확인할 수 없는 문헌은 제외하십시오.
- 구성과의 대응 내용과 남은 차이점을 각각 한 문장으로 적으십시오.
- 확인되지 않는 문헌번호나 URL을 지어내지 마십시오. 찾지 못하면 해당 구성은 결과에서 빼십시오.

[출력] JSON 객체 하나만 출력하십시오.
{"hits": [{"claim_number": 1, "label": "B", "document_number": "US 2019/0123456 A1", "title": "문헌 제목",
  "published": "2019-04-25", "correspondence": "대응 내용", "remaining_difference": "남은 차이",
  "url": "https://..."}]}

[미커버 구성]
"""


class SearchFailed(RuntimeError):
    """CLI가 답을 내지 못했습니다.

    "찾지 못했습니다(0건)"와 반드시 구분해야 합니다. 실패를 빈 결과로 돌려주면
    호출부가 그것을 검색 결과로 받아들여, 지난 검색에서 찾아 둔 선행기술을 빈 목록으로
    덮어쓰고 보고서에 저장해 버립니다.
    """


def search(uncovered: list[dict]) -> list[PriorArtHit]:
    """uncovered는 [{"label": "B", "text": "구성 원문"}] 형태입니다."""
    if not uncovered:
        return []
    try:
        raw = run_cli(PRIOR_ART_PROMPT + json.dumps(uncovered, ensure_ascii=False), expect="hits")
    except AnalysisCancelled:
        # AnalysisCancelled도 RuntimeError를 상속합니다. 아래에서 함께 잡으면 취소가
        # 검색 실패로 둔갑하므로, 먼저 걸러 그대로 올립니다.
        raise
    except RuntimeError as exc:
        raise SearchFailed(f"선행기술 검색에 실패했습니다: {exc}") from exc
    hits: list[PriorArtHit] = []
    for item in raw.get("hits") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().strip("()").upper()
        document_number = _clean(item.get("document_number"))
        title = _clean(item.get("title"))
        if not label or not (document_number or title):
            continue
        try:
            claim_number = int(item.get("claim_number"))
        except (TypeError, ValueError):
            claim_number = None
        hits.append(PriorArtHit(
            claim_number=claim_number,
            label=label,
            document_number=document_number,
            title=title,
            published=_clean(item.get("published")),
            correspondence=_clean(item.get("correspondence")),
            remaining_difference=_clean(item.get("remaining_difference")),
            url=_url(item.get("url")),
        ))
    return hits


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _url(value) -> str:
    url = _clean(value)
    return url if url.startswith(("http://", "https://")) else ""


# --- 결과 검증 ----------------------------------------------------------------
# 이 파이프라인의 원칙은 "LLM은 사실만 답하고 확인은 코드가 한다"입니다. 구성대비 발췌는
# verify.py가 원문과 대조하는데, 선행기술 검색 결과만 그 원칙 밖에 있었습니다. 문헌번호도
# URL도 공개일도 모델이 적어 준 그대로 보고서에 실려, 지어낸 문헌인지 구별할 수 없었습니다.
# 여기서는 URL을 실제로 열어 그 페이지에 문헌번호가 있는지만 확인합니다.

_MAX_VERIFY_WORKERS = 4
_VERIFY_BODY_LIMIT = 400_000
_USER_AGENT = "Mozilla/5.0 (compatible; EvidenceForge/1.0; +patent-analysis)"


def _collapse(value: str) -> str:
    """구분자를 지운 대조용 형태. 'WO 2022/019489 A1' ↔ 'WO2022019489A1'."""
    return re.sub(r"[^0-9a-z]", "", str(value or "").lower())


def verify_hits(hits: list[PriorArtHit]) -> None:
    """각 결과의 URL을 열어 문헌번호를 대조하고 verify를 채웁니다(제자리 수정).

    확인에 실패해도 결과를 버리지 않습니다. 사내망이 외부를 막아 두었을 수도 있고, 그때
    결과를 지우면 검색이 조용히 0건이 됩니다. 판단은 보고서를 읽는 사람이 하도록 표시만 합니다.
    """
    if not hits or PRIOR_ART_VERIFY_TIMEOUT <= 0:
        return
    # certifi 번들만 쓰면 TLS를 가로채는 사내망에서 전부 실패합니다. OS 신뢰 저장소를
    # 쓰면 그런 환경의 사설 CA도 그대로 통합니다.
    context = ssl.create_default_context()
    with httpx.Client(verify=context, timeout=PRIOR_ART_VERIFY_TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": _USER_AGENT}) as client:
        with ThreadPoolExecutor(max_workers=min(_MAX_VERIFY_WORKERS, len(hits))) as pool:
            list(pool.map(lambda hit: _verify_hit(client, hit), hits))


def _verify_hit(client: httpx.Client, hit: PriorArtHit) -> None:
    if not hit.url:
        hit.verify, hit.verify_note = "unreachable", "URL이 제시되지 않아 실재 여부를 확인하지 못했습니다."
        return
    try:
        response = client.get(hit.url)
    except (httpx.HTTPError, ssl.SSLError, OSError) as exc:
        hit.verify = "unreachable"
        hit.verify_note = f"URL을 열지 못했습니다({type(exc).__name__}). 문헌을 직접 확인하십시오."
        return
    if response.status_code >= 400:
        hit.verify = "unreachable"
        hit.verify_note = f"URL이 HTTP {response.status_code}를 반환했습니다."
        return
    number = _collapse(hit.document_number)
    if not number:
        hit.verify, hit.verify_note = "unreachable", "문헌번호가 없어 대조할 수 없습니다."
        return
    if number in _collapse(response.text[:_VERIFY_BODY_LIMIT]):
        hit.verify, hit.verify_note = "verified", ""
        return
    hit.verify = "mismatch"
    hit.verify_note = "URL은 열렸으나 그 페이지에서 문헌번호를 찾지 못했습니다."
