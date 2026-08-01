import json
import re
from .agy import run_agy
from .config import load_runtime_settings
from .models import AnalysisResult, ClaimResult, DocumentMapping, Evidence
from .prompts import DEFAULT_ANALYSIS_PROMPT

# 사용자의 분석 지시를 그대로 수행시키되, 결과만 기계가 읽을 수 있는 필드에 담게 하는 규약입니다.
# 지시문 자체를 덮어쓰지 않도록 형식 요구만 담습니다.
OUTPUT_CONTRACT = """
[출력 규약]
위 [분석 지시]를 그대로 수행하되, 결과는 아래 스키마를 따르는 JSON 객체 **하나만** 출력하십시오. 코드펜스, 머리말, 설명 문장을 붙이지 마십시오.
{
  "document_mapping": [
    {
      "reference_number": 1,
      "document_id": "CONTEXT documents의 id",
      "document_number": "문헌 고유번호(예: US 10,987,654 A1 / 10-2020-0012345호). 문헌에서 확인되지 않으면 빈 문자열",
      "role": "주 인용발명 | 부 인용발명"
    }
  ],
  "claims": [
    {
      "label": "입력 구성요소의 라벨(A, B, C…)을 그대로",
      "similarity": 0-100 정수, 대응 인용발명이 없으면 null,
      "grade": "동일 | 실질적 동일 | 기술 사상 동일, 세부 구현 방식의 단순 변경 | 핵심 기능 유사하나 목적/효과에 일부 차이 | 대응 안됨",
      "emoji": "🔵 | 🟢 | 🟠 | 🟡 | ⚪",
      "narrative": "[분석 지시]가 요구한 구성대비 서술을 라벨 없이 하나의 자연스러운 문장으로. 인용발명 명칭·문헌번호·발췌문·단락번호·판단 이유를 문장 안에 녹여 작성",
      "difference": "차이점 한 줄. 없으면 null",
      "combination": true 또는 false (두 개 이상의 인용발명을 결합했는지),
      "references": ["근거가 된 document id"],
      "evidence": [
        {
          "document_id": "문서 id",
          "paragraph": "단락번호 4자리 또는 null",
          "page": 페이지 번호 정수 또는 null,
          "excerpt": "문서 원문 인용(외국어 문헌은 한국어 번역문)",
          "original_excerpt": "외국어 문헌의 원문 1줄, 국문 문헌은 null",
          "quality": "HIGH | MEDIUM | LOW | UNVERIFIED"
        }
      ],
      "status": "개시됨 | 부분 개시 | 미개시",
      "note": "판단 근거 한두 문장"
    }
  ],
  "summary": "전체 분석 요약",
  "summary_similarity": "종합 분석 요약의 유사점 한 줄",
  "summary_difference": "종합 분석 요약의 차이점 한 줄. 없으면 빈 문자열",
  "validation": ["검증 시 주의할 점"]
}

형식 규칙:
- claims 배열의 길이·순서·label은 입력 claims와 정확히 같아야 합니다. 구성요소를 임의로 나누거나 합치지 마십시오.
- narrative는 반드시 한 문장으로 작성하고 줄바꿈을 넣지 마십시오. "유사도:" 표기와 "→ 차이점:" 줄은 별도 필드로 나가므로 narrative 안에 다시 쓰지 마십시오.
- document_mapping은 대응도가 높은 순서대로 reference_number 1, 2, 3…을 매기며 업로드 순서와 무관합니다. 모든 문서를 빠짐없이 한 번씩만 포함하십시오.
- 문헌을 지칭할 때는 "인용발명 N (문헌번호)" 형태를 쓰고, N은 document_mapping에서 정한 번호를 전체 분석 내내 동일하게 유지하십시오.
- excerpt는 CONTEXT documents에 실제로 존재하는 문장을 그대로 인용하고 지어내지 마십시오.
- 어느 인용발명에서도 대응 구성을 찾지 못하면 similarity는 null, emoji는 "⚪", grade는 "대응 안됨", evidence는 빈 배열, status는 "미개시"로 두십시오.

CONTEXT:
"""

_LABEL_SPLIT = re.compile(r"(?=\(\s*[A-Z]\s*\))")
_LABEL_HEAD = re.compile(r"^\(\s*([A-Z])\s*\)\s*")
_GRADES = ((95, "동일", "🔵"), (90, "실질적 동일", "🟢"), (85, "기술 사상 동일, 세부 구현 방식의 단순 변경", "🟠"),
           (80, "핵심 기능 유사하나 목적/효과에 일부 차이", "🟡"), (0, "대응 안됨", "⚪"))


def _auto_label(index: int) -> str:
    """A…Z를 넘어가면 AA, AB…로 이어 붙입니다."""
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def split_claims(claims_text: str) -> tuple[str, list[dict]]:
    """(A)~(Z) 라벨을 기준으로만 구성요소를 나누고 라벨을 보존합니다.

    줄바꿈으로 쪼개면 여러 줄에 걸친 하나의 구성이 둘로 갈려 이후 라벨이 전부 밀리므로,
    라벨이 있으면 라벨만으로 나누고 구성 내부의 줄바꿈은 공백으로 합칩니다.
    라벨이 없는 청구항은 문단(빈 줄) 단위로 나눈 뒤 A부터 순서대로 부여합니다.
    """
    text = claims_text.replace("\r\n", "\n").strip()
    parts = [part for part in _LABEL_SPLIT.split(text) if part.strip()]
    labeled = [part for part in parts if _LABEL_HEAD.match(part.strip())]
    preamble = ""
    claims: list[dict] = []
    if labeled:
        if parts and not _LABEL_HEAD.match(parts[0].strip()):
            preamble = re.sub(r"\s+", " ", parts[0]).strip()
        for part in labeled:
            part = part.strip()
            label = _LABEL_HEAD.match(part).group(1)
            body = re.sub(r"\s+", " ", _LABEL_HEAD.sub("", part)).strip()
            claims.append({"label": label, "text": body})
        return preamble, claims
    blocks = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) == 1:
        blocks = [line for line in text.split("\n") if line.strip()]
    for index, block in enumerate(blocks):
        claims.append({"label": _auto_label(index), "text": re.sub(r"\s+", " ", block).strip()})
    return preamble, claims


def build_context(preamble: str, claims: list[dict], documents: list[dict]) -> str:
    """LLM에 넘길 최소 컨텍스트. 원문 text/pages는 chunks와 중복이라 제외합니다."""
    payload = {
        "claim_preamble": preamble,
        "claims": claims,
        "documents": [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "type": doc["type"],
                "chunks": [
                    {"page": chunk["page"], "paragraph": chunk["paragraph"], "text": chunk["text"]}
                    for chunk in doc["chunks"]
                ],
            }
            for doc in documents
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_prompt(guideline: str, preamble: str, claims: list[dict], documents: list[dict]) -> str:
    return f"[분석 지시]\n{guideline.strip()}\n{OUTPUT_CONTRACT}{build_context(preamble, claims, documents)}"


def resolve_mapping(raw_mapping, documents: list[dict]) -> tuple[list[DocumentMapping], list[str]]:
    """인용발명 번호는 모델이 정한 대응도 순서를 따르고, 빠진 문헌만 업로드 순서로 뒤에 붙입니다."""
    by_id = {doc["id"]: doc for doc in documents}
    by_name = {doc["filename"]: doc for doc in documents}
    ordered: list[tuple[dict, str]] = []
    seen: set[str] = set()
    warnings: list[str] = []
    for item in raw_mapping or []:
        if not isinstance(item, dict):
            continue
        doc = by_id.get(str(item.get("document_id", "")).strip()) or by_name.get(str(item.get("filename", "")).strip())
        if doc is None or doc["id"] in seen:
            continue
        seen.add(doc["id"])
        ordered.append((doc, str(item.get("document_number") or "").strip()))
    if not ordered:
        warnings.append("모델이 인용발명 순위를 지정하지 않아 업로드 순서로 번호를 매겼습니다.")
    elif len(ordered) < len(documents):
        warnings.append("일부 문헌이 매핑 테이블에서 누락되어 업로드 순서로 뒤에 배치했습니다.")
    for doc in documents:
        if doc["id"] not in seen:
            seen.add(doc["id"])
            ordered.append((doc, ""))
    mappings = [
        DocumentMapping(reference_number=index, filename=doc["filename"], document_type=doc["type"],
                        document_id=doc["id"], document_number=number,
                        role="주 인용발명" if index == 1 else "부 인용발명")
        for index, (doc, number) in enumerate(ordered, 1)
    ]
    return mappings, warnings


def grade_for(similarity: int | None) -> tuple[str, str]:
    if similarity is None:
        return "대응 안됨", "⚪"
    for threshold, name, emoji in _GRADES:
        if similarity >= threshold:
            return name, emoji
    return "대응 안됨", "⚪"


def _as_similarity(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return None


def _build_evidence(raw_evidence, mappings: list[DocumentMapping], documents: list[dict]) -> list[Evidence]:
    by_id = {mapping.document_id: mapping for mapping in mappings}
    names = {doc["id"]: doc["filename"] for doc in documents}
    evidence: list[Evidence] = []
    for item in raw_evidence or []:
        if not isinstance(item, dict) or not str(item.get("excerpt", "")).strip():
            continue
        document_id = str(item.get("document_id", "")).strip()
        mapping = by_id.get(document_id)
        paragraph = item.get("paragraph")
        page = item.get("page")
        evidence.append(Evidence(
            document_id=document_id,
            filename=names.get(document_id) or str(item.get("filename", "")),
            # 인용발명 번호는 매핑 테이블에서 다시 계산해 전 구성요소에 걸쳐 일관성을 강제합니다.
            reference_number=mapping.reference_number if mapping else None,
            document_number=(mapping.document_number if mapping else None) or None,
            paragraph=str(paragraph) if paragraph not in (None, "") else None,
            page=page if isinstance(page, int) else None,
            excerpt=str(item["excerpt"]).strip(),
            original_excerpt=str(item["original_excerpt"]).strip() if str(item.get("original_excerpt") or "").strip() else None,
            quality=item.get("quality") if item.get("quality") in {"HIGH", "MEDIUM", "LOW", "UNVERIFIED"} else "MEDIUM",
        ))
    return evidence


def build_claims(raw_claims, claims: list[dict], mappings: list[DocumentMapping],
                 documents: list[dict]) -> tuple[list[ClaimResult], list[str]]:
    """입력 구성요소를 기준으로 결과를 정렬합니다. 라벨이 맞지 않으면 순서로 보정합니다."""
    raw_list = [item for item in (raw_claims or []) if isinstance(item, dict)]
    by_label = {}
    for item in raw_list:
        label = str(item.get("label", "")).strip().strip("()").upper()
        if label and label not in by_label:
            by_label[label] = item
    warnings: list[str] = []
    results: list[ClaimResult] = []
    # 라벨이 하나라도 오면 라벨로만 맞춥니다. 위치로 폴백하면 (A)가 (B)의 분석을 가져가
    # 이후 구성요소가 통째로 밀리기 때문입니다. 라벨이 전혀 없을 때만 순서로 대응시킵니다.
    positional = not by_label
    if positional and raw_list:
        warnings.append("응답에 구성요소 라벨이 없어 입력 순서로 대응시켰습니다.")
    for index, claim in enumerate(claims):
        item = by_label.get(claim["label"])
        if item is None:
            item = raw_list[index] if positional and index < len(raw_list) else {}
            if not positional:
                warnings.append(f"({claim['label']}) 구성에 대한 응답이 없어 미개시로 처리했습니다.")
        similarity = _as_similarity(item.get("similarity"))
        grade, emoji = grade_for(similarity)
        difference = str(item.get("difference") or "").strip()
        narrative = re.sub(r"\s+", " ", str(item.get("narrative") or "")).strip()
        if not narrative and similarity is None:
            narrative = f"({claim['label']}) 구성에 대응되는 인용발명이 확인되지 않음 — 추가 검색 필요"
        results.append(ClaimResult(
            label=claim["label"],
            claim=claim["text"],  # 구성 원문은 입력을 그대로 씁니다. 모델이 고쳐 쓴 문장을 신뢰하지 않습니다.
            similarity=similarity,
            grade=str(item.get("grade") or grade).strip() if similarity is not None else grade,
            emoji=str(item.get("emoji") or emoji).strip() or emoji,
            narrative=narrative,
            difference=difference or None,
            combination=bool(item.get("combination")),
            references=[str(ref) for ref in item.get("references") or [] if str(ref).strip()],
            evidence=_build_evidence(item.get("evidence"), mappings, documents),
            status=str(item.get("status") or ("미개시" if similarity is None else "")).strip(),
            note=str(item.get("note") or "").strip(),
        ))
    if len(raw_list) != len(claims):
        warnings.append(f"입력 구성요소는 {len(claims)}개인데 응답은 {len(raw_list)}개라 입력 기준으로 정렬했습니다.")
    return results, warnings


def analyze(job_id: str, claims_text: str, documents: list[dict], effort: str,
            analysis_prompt: str | None = None) -> AnalysisResult:
    preamble, claims = split_claims(claims_text)
    guideline = (analysis_prompt or "").strip() or load_runtime_settings().get("prompt") or DEFAULT_ANALYSIS_PROMPT
    raw = run_agy(build_prompt(guideline, preamble, claims, documents), effort)
    mappings, warnings = resolve_mapping(raw.get("document_mapping"), documents)
    claim_results, claim_warnings = build_claims(raw.get("claims"), claims, mappings, documents)
    validation = [str(item) for item in raw.get("validation") or []] + warnings + claim_warnings
    return AnalysisResult(job_id=job_id, claim_mapping=mappings, claims=claim_results, preamble=preamble,
                          summary=str(raw.get("summary", "")), summary_similarity=str(raw.get("summary_similarity", "")),
                          summary_difference=str(raw.get("summary_difference", "")), validation=validation)


def to_markdown(result: AnalysisResult) -> str:
    lines = ["# 구성대비 분석", "", "## 문헌 매핑 테이블", "",
             "| 인용발명 | 문헌번호 | 파일명 | 문서 유형 |", "|---|---|---|---|"]
    lines += [f"| 인용발명 {m.reference_number}{' (주 인용발명)' if m.reference_number == 1 else ''} "
              f"| {m.document_number or '-'} | {m.filename} | {m.document_type} |" for m in result.claim_mapping]
    if result.preamble:
        lines += ["", f"> {result.preamble}"]
    for claim in result.claims:
        lines += ["", f"## ({claim.label}) {claim.claim}", ""]
        if claim.similarity is None:
            lines.append(claim.narrative or f"({claim.label}) 구성에 대응되는 인용발명이 확인되지 않음 — 추가 검색 필요")
            continue
        lines += [f"유사도: {claim.similarity}% {claim.emoji} {claim.grade}", "", claim.narrative]
        if claim.difference:
            lines.append(f"→ 차이점: {claim.difference}")
    lines += ["", "## 종합 분석 요약", ""]
    if result.summary_similarity:
        lines.append(f"- 유사점: {result.summary_similarity}")
    if result.summary_difference:
        lines.append(f"- 차이점: {result.summary_difference}")
    if result.summary:
        lines += ["", result.summary]
    if result.validation:
        lines += ["", "## 검증 참고", ""] + [f"- {item}" for item in result.validation]
    return "\n".join(lines) + "\n"
