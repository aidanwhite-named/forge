"""발췌 검증. LLM 호출 없이 문자열 대조만으로 수행합니다.

청구항 문언과 문헌의 낱말을 섞어 만든 그럴듯한 문장은 직접 인용이 아닙니다.
검증을 통과하지 못한 근거는 직접 개시(direct)로 인정하지 않고 판정 상한을 씌웁니다.
이 규칙이 없으면 이후의 모든 결정론적 계산이 지어낸 발췌 위에서 돌아갑니다.
"""
import re
from functools import lru_cache

from .models import Document, ElementMatch, missing_limitations
from .pdf import chunk_text, document_corpus

MIN_QUOTE_LEN = 15
MIN_SEGMENT_LEN = 12
# 검증에 실패한 근거로는 이 등급을 넘는 판정을 인정하지 않습니다.
UNVERIFIED_JUDGMENT_CAP = "일부 차이"
_RANK = {"대응 없음": 0, "차이": 1, "일부 유사": 2, "일부 차이": 3, "실질적 동일": 4, "동일": 5}
_BY_RANK = {rank: judgment for judgment, rank in _RANK.items()}
# 낱말을 잇는 구분자. 추출 과정에서 생겼는지 원문에 있었는지 구분할 수 없으므로 함께 지웁니다.
_SEPARATOR_RE = re.compile(r"[^0-9a-z가-힣]+")


def normalize(value: str) -> str:
    """추출 과정에서만 생기는 공백·하이픈 줄바꿈을 지웁니다. 낱말 자체는 바꾸지 않습니다."""
    text = str(value or "").replace(" ", " ")
    text = re.sub(r"(?<=\w)-\s*\r?\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def collapse(value: str) -> str:
    """공백·하이픈·구두점을 모두 지운 비교용 형태.

    공보 PDF는 줄바꿈 자리에서 낱말을 쪼갭니다("commu nication", "pro vide"). 모델이 그 낱말을
    붙여 쓰면 정확 대조는 실패하지만 인용 자체는 진짜입니다. 구분자를 지운 뒤 연속 부분열로
    다시 확인해 이 경우만 구제합니다. 여러 문장을 조합해 지어낸 발췌는 여기서도 붙지 않습니다.
    """
    return _SEPARATOR_RE.sub("", normalize(value))


@lru_cache(maxsize=8)
def _collapsed_corpus(normalized_corpus: str) -> str:
    """말뭉치는 문헌당 한 번만 접습니다. 구성요소마다 다시 접으면 비교가 눈에 띄게 느려집니다."""
    return _SEPARATOR_RE.sub("", normalized_corpus)


def is_verbatim(quote: str, corpus: str, min_segment_len: int = MIN_SEGMENT_LEN) -> bool:
    """'…'로 축약된 각 구간이 모두 원문에 그대로 존재할 때만 참입니다."""
    normalized_corpus = normalize(corpus)
    if not normalized_corpus:
        return False
    segments = [
        normalize(re.sub(r"^\s*\[[^\]]+\]\s*", "", segment))
        for segment in re.split(r"\s*(?:…|\.{3,})\s*", str(quote or ""))
    ]
    segments = [segment for segment in segments if len(segment) >= min_segment_len]
    if not segments:
        return False
    if all(segment in normalized_corpus for segment in segments):
        return True
    collapsed_corpus = _collapsed_corpus(normalized_corpus)
    return all(collapse(segment) in collapsed_corpus for segment in segments)


def verify_matches(matches: list[ElementMatch], documents: dict[str, Document]) -> list[str]:
    """비교 매트릭스를 제자리에서 검증하고, 강등된 항목을 경고로 돌려줍니다."""
    try:
        return _verify_matches(matches, documents)
    finally:
        # _collapsed_corpus의 키와 값이 각각 문헌 말뭉치 **전문**입니다. 한 번의 검증
        # 안에서는 같은 말뭉치를 수백 번 접으므로 캐시가 필요하지만, 검증이 끝난 뒤에도
        # 프로세스가 살아 있는 내내 문헌 8건 분량을 붙들고 있을 이유는 없습니다.
        _collapsed_corpus.cache_clear()


def _verify_matches(matches: list[ElementMatch], documents: dict[str, Document]) -> list[str]:
    corpus_cache: dict[str, str] = {}
    notes: list[str] = []
    for match in matches:
        document = documents.get(match.document_id)
        if document is None:
            match.verify, match.verify_note = "not_found", "인용발명 문서를 찾을 수 없습니다."
            match.alignment = "not_found"
            _cap(match, notes, "문서 없음", f"문서 ID {match.document_id}")
            continue
        if match.document_id not in corpus_cache:
            corpus_cache[match.document_id] = document_corpus(document)
        corpus = corpus_cache[match.document_id]

        failed_limitations: list[str] = []
        for check in match.limitation_checks:
            if not check.disclosed:
                continue
            for span in check.evidence:
                _verify_span(span, document, corpus)
            check.verify = _status(check.quote, corpus)
            cited_check = chunk_text(document, check.chunk_id) if check.chunk_id else ""
            if check.verify == "verified" and not (
                    cited_check and is_verbatim(check.quote, cited_check)):
                check.chunk_id = _find_chunk_id(document, check.quote)
                cited_check = chunk_text(document, check.chunk_id) if check.chunk_id else ""
            if check.verify == "verified":
                check.alignment = "exact"
                # 원문 대조를 통과한 근거는 청크 위치를 특정하지 못하더라도 개시로 인정합니다.
                # 대표 발췌에 이미 적용하는 규칙(아래 locator_verified 분기)과 같은 규칙입니다.
                #
                # 공보 PDF는 단락과 무관한 자리에서 잘리므로 한 문장이 두 청크에 나뉘는 일이
                # 흔합니다. 그때마다 disclosed를 뒤집으면 문헌에 그대로 있는 한정이 '누락 한정'이
                # 되고 구성 판정까지 강등되는데, 강등의 근거는 문헌의 내용이 아니라 청크를 어디서
                # 잘랐는가입니다. 실제로 이 경로 하나가 주 인용발명이 단독으로 개시한 구성을
                # 미개시로 만들어, 신규성 결론이 불필요한 문헌 결합으로 바뀌었습니다.
                if not cited_check:
                    check.verify_note = "원문은 확인되었으나 단일 청크 위치를 특정하지 못했습니다."
                    notes.append(f"청구항 {match.claim_number} ({match.label}) / {document.filename}: "
                                 f"하위 한정 근거의 청크 위치를 특정하지 못했습니다(원문 대조는 통과).")
                continue
            recovered = _recover(check.quote, cited_check)
            if recovered:
                check.quote, check.verify = recovered, "verified"
                check.alignment = "recovered"
                check.verify_note = "PDF 추출문에서 가장 가까운 실제 문장으로 정렬했습니다."
                continue
            # 대표 발췌가 깨졌더라도 같은 한정을 위한 근거 묶음에 검증된 문장이 있으면 개시
            # 여부를 뒤집지 않습니다. 의미상 충족 여부는 독립 entailment 검증이 전체 묶음으로 봅니다.
            supporting = next((span for span in check.evidence if span.verify == "verified"), None)
            if supporting is not None:
                check.quote = supporting.quote
                check.quote_translation = supporting.quote_translation
                check.chunk_id = supporting.chunk_id
                check.verify = "verified"
                check.alignment = supporting.alignment
                check.verify_note = "대표 발췌 대신 검증된 근거 묶음의 문장을 사용했습니다."
                continue
            check.disclosed = False
            check.alignment = "not_found"
            failed_limitations.append(check.limitation)
        # 검증이 check.disclosed를 뒤집으므로 누락 목록은 여기서 다시 계산합니다. 점검 결과가
        # 없는 셀은 모델이 적어 준 목록뿐이라 그대로 둡니다.
        if match.limitation_checks:
            match.missing_limitations = missing_limitations(match.limitation_checks)
        if failed_limitations:
            _cap(match, notes, f"하위 제한 근거 미검증 {len(failed_limitations)}건", document.filename)

        for span in match.evidence:
            _verify_span(span, document, corpus)

        # 복합 한정의 근거 묶음만 있고 구성 대표 발췌가 비어 있으면, 검증된 첫 한정 근거를
        # 보고서·점수 계산용 대표 발췌로 올립니다. 의미 판정은 아래 별도 단계가 담당합니다.
        if not match.quote:
            representative = next((check for check in match.limitation_checks
                                   if check.disclosed and check.verify == "verified" and check.quote), None)
            if representative is not None:
                match.quote = representative.quote
                match.quote_translation = representative.quote_translation
                match.chunk_id = representative.chunk_id
                match.verify = "verified"
                match.alignment = representative.alignment

        if not match.quote:
            match.verify = "empty"
            match.alignment = "not_found"
            if match.directness == "direct":
                match.directness = "inferred"
                match.verify_note = "직접 개시로 판정되었으나 발췌가 없어 추론 근거로 낮췄습니다."
                _cap(match, notes, "발췌 없음", document.filename)
            continue
        if len(match.quote) < MIN_QUOTE_LEN:
            match.verify, match.verify_note = "short", "발췌가 너무 짧아 근거로 확인할 수 없습니다."
            match.alignment = "not_found"
            _cap(match, notes, "발췌 과소", document.filename)
            continue

        match.verify = _status(match.quote, corpus)
        cited = chunk_text(document, match.chunk_id) if match.chunk_id else ""
        locator_verified = bool(cited and is_verbatim(match.quote, cited))
        if match.verify == "verified" and not locator_verified:
            invalid = match.chunk_id
            recovered_id = _find_chunk_id(document, match.quote)
            match.chunk_id = recovered_id
            if recovered_id:
                cited = chunk_text(document, recovered_id)
                match.verify_note = (f"원문은 확인되었으나 인용 위치({invalid or '미지정'})가 잘못되어 "
                                     f"{recovered_id}(으)로 자동 복구했습니다.")
                notes.append(f"청구항 {match.claim_number} ({match.label}) / {document.filename}: "
                             f"인용 위치를 {recovered_id}(으)로 자동 복구했습니다.")
            else:
                match.verify_note = ("발췌 원문은 문헌 전체에서 확인되었으나 단일 청크 위치를 "
                                     "자동으로 특정하지 못했습니다.")
                match.alignment = "exact"
                # 인용문 자체가 원문 대조를 통과했다면 위치 메타데이터 오류만으로 판정과
                # 직접성을 강등하지 않습니다. 위치 점수만 빠져 문헌 적합도에는 소폭 반영됩니다.
                continue
        elif match.chunk_id and not cited:
            invalid = match.chunk_id
            match.verify_note = f"인용한 chunk_id({invalid})가 문헌에 존재하지 않습니다."
            match.chunk_id = ""
            match.alignment = "not_found"
            _cap(match, notes, "chunk_id 불일치 및 발췌 미검증", document.filename)
            match.directness = "inferred" if match.directness == "direct" else match.directness
            continue
        if match.verify != "verified":
            recovered = _recover(match.quote, cited)
            if recovered:
                match.quote, match.verify = recovered, "verified"
                match.verify_note = "대표 발췌를 인용한 청크의 정확한 원문으로 복구했습니다."
                match.alignment = "recovered"
                continue
            match.verify_note = "발췌가 원문과 일치하지 않아 직접 개시로 인정하지 않았습니다."
            match.alignment = "not_found"
            match.directness = "inferred" if match.verify == "partial" else "absent"
            _cap(match, notes, "발췌 불일치", document.filename)
        else:
            match.alignment = "exact"
    return notes


def _verify_span(span, document: Document, corpus: str) -> bool:
    """근거 묶음의 한 문장을 검증합니다. 추출 복구 여부는 의미상 직접성과 분리합니다."""
    span.verify = _status(span.quote, corpus)
    cited = chunk_text(document, span.chunk_id) if span.chunk_id else ""
    if span.verify == "verified":
        if not (cited and is_verbatim(span.quote, cited)):
            span.chunk_id = _find_chunk_id(document, span.quote)
        span.alignment = "exact"
        return True
    recovered = _recover(span.quote, cited)
    if recovered:
        span.quote = recovered
        span.verify = "verified"
        span.alignment = "recovered"
        return True
    span.alignment = "not_found"
    return False


def _status(quote: str, corpus: str) -> str:
    if not quote:
        return "empty"
    if len(quote) < MIN_QUOTE_LEN:
        return "short"
    if is_verbatim(quote, corpus):
        return "verified"
    segments = [segment for segment in re.split(r"\s*(?:…|\.{3,})\s*", quote) if len(normalize(segment)) >= MIN_SEGMENT_LEN]
    if any(is_verbatim(segment, corpus) for segment in segments):
        return "partial"
    return "not_found"


def _cap(match: ElementMatch, notes: list[str], reason: str, document_label: str = "") -> None:
    """검증되지 않은 근거의 판정에 상한을 씌웁니다."""
    limit = _RANK[UNVERIFIED_JUDGMENT_CAP]
    if match.directness == "absent":
        limit = min(limit, _RANK["차이"])
    if _RANK.get(match.judgment, 0) <= limit:
        return
    match.downgraded_from = match.judgment
    match.judgment = _BY_RANK[limit]
    source = document_label or f"문서 ID {match.document_id}"
    notes.append(f"청구항 {match.claim_number} ({match.label}) / {source}: "
                 f"{match.downgraded_from} → {match.judgment} ({reason})")


def _find_chunk_id(document: Document, quote: str) -> str:
    """문헌 전체에서 검증된 인용문을 실제로 포함하는 청크 위치로 되돌립니다.

    특허 PDF의 한 청크에 [0013]~[0028]이 함께 들어가도 모델이 D1-P-0013을 만들 수 있습니다.
    이때 인용문이 D1-P-0008 안에 그대로 있다면 가짜 근거가 아니라 위치 세분화 오류이므로,
    실제 저장된 청크로 고쳐 판정은 유지합니다.
    """
    if not quote:
        return ""
    candidates = [chunk for chunk in document.chunks if is_verbatim(quote, chunk.text)]
    if not candidates:
        return _find_starting_chunk_id(document, quote)
    return min(candidates, key=lambda chunk: (len(chunk.text), chunk.page or 0, chunk.chunk_id)).chunk_id


def _find_starting_chunk_id(document: Document, quote: str) -> str:
    """청크 경계에 걸친 발췌는 그 문장이 **시작되는** 청크를 위치로 씁니다.

    공보 PDF는 단락 경계와 무관한 자리에서 잘리므로, 문헌 전체에서는 그대로 확인되는
    문장이 어느 단일 청크에도 온전히 들어 있지 않을 수 있습니다. 그때 위치를 비워 두면
    근거는 확인되었는데 어디를 보라고 적을 수 없는 보고서가 됩니다. 앞부분이 가장 길게
    일치하는 청크가 그 문장이 시작된 자리이므로 그것을 인용 위치로 삼습니다.
    """
    collapsed = collapse(quote)
    if len(collapsed) < MIN_SEGMENT_LEN:
        return ""
    scored = [(_leading_overlap(collapsed, collapse(chunk.text)), chunk) for chunk in document.chunks]
    length, chunk = max(scored, key=lambda item: (item[0], -len(item[1].text), item[1].chunk_id),
                        default=(0, None))
    return chunk.chunk_id if chunk is not None and length >= MIN_SEGMENT_LEN else ""


def _leading_overlap(collapsed_quote: str, collapsed_chunk: str) -> int:
    """청크에 들어 있는 발췌 앞부분의 최대 길이. 이분 탐색이라 청크 수에 선형입니다."""
    if not collapsed_chunk or collapsed_quote[:MIN_SEGMENT_LEN] not in collapsed_chunk:
        return 0
    low, high, best = MIN_SEGMENT_LEN, len(collapsed_quote), 0
    while low <= high:
        middle = (low + high) // 2
        if collapsed_quote[:middle] in collapsed_chunk:
            best, low = middle, middle + 1
        else:
            high = middle - 1
    return best


def _recover(model_quote: str, cited_chunk: str) -> str:
    """인용한 청크 안에서만 가장 가까운 실제 문장을 찾습니다. 다른 문헌은 뒤지지 않습니다."""
    if not cited_chunk:
        return ""
    wanted = _tokens(model_quote)
    if not wanted:
        return ""
    best, best_score = "", 0.0
    for sentence in re.split(r"(?<=[.。])\s+|(?<=다\.)\s+", cited_chunk):
        sentence = sentence.strip()
        if len(sentence) < MIN_QUOTE_LEN:
            continue
        overlap = len(wanted & _tokens(sentence)) / len(wanted)
        if overlap > best_score:
            best, best_score = sentence, overlap
    return best if best_score >= 0.5 else ""


_RECOVERY_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with",
                       "is", "are", "be", "by", "that", "this", "상기", "및", "또는", "하는", "한다", "포함"}


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", str(value or "").lower())
            if token not in _RECOVERY_STOPWORDS}
