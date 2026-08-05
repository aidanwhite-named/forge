"""미커버 구성 선행기술 웹 검색.

분석 파이프라인에 자동으로 끼워 넣지 않고 별도 요청으로만 실행합니다.
CLI가 외부 웹에 나가는 단계이므로, 사용자가 명시적으로 눌렀을 때만 동작해야 합니다.
"""
import json
import re

from .agy import run_cli
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


def search(uncovered: list[dict]) -> tuple[list[PriorArtHit], list[str]]:
    """uncovered는 [{"label": "B", "text": "구성 원문"}] 형태입니다."""
    if not uncovered:
        return [], []
    try:
        raw = run_cli(PRIOR_ART_PROMPT + json.dumps(uncovered, ensure_ascii=False), expect="hits")
    except RuntimeError as exc:
        return [], [f"선행기술 검색에 실패했습니다: {exc}"]
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
    return hits, []


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _url(value) -> str:
    url = _clean(value)
    return url if url.startswith(("http://", "https://")) else ""
