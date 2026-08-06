"""PDF 추출. 비교·검증이 chunk_id 하나로 원문을 되찾을 수 있게 만드는 것이 목표입니다.

특허문헌은 단락번호를, 그 외 문헌은 페이지 내 블록 순번을 chunk_id에 담습니다.
논문은 섹션 표제만 인식해 참고문헌·감사의 글을 비교 대상에서 제외하며,
그 외에는 특허문헌과 같은 경로로 처리합니다.

공보는 대부분 2단 조판입니다. 추출기 기본 읽기 순서는 좌우 단을 줄 단위로 번갈아 섞어
한 문장을 두 동강 내므로, 그대로 두면 발췌 검증이 문헌 전체에서 실패합니다. 다만 줄을
직접 조립하지는 않습니다. 줄 묶기는 추출기가 이미 정확히 해 주고, 우리가 낱말 좌표로
다시 묶으면 어센더 없는 짧은 낱말이 별도 줄로 갈라져 문장이 흐트러집니다. 여기서는
추출기가 만든 줄을 단위로 좌우 단만 갈라 읽기 순서를 복원합니다.
"""
import re
from datetime import datetime
from pathlib import Path
from statistics import median

import fitz

from .models import Chunk, Document

# 공보 서지사항에서 문헌 고유번호를 뽑는 패턴. 앞에 오는 것이 대표 번호입니다.
# 추출기가 구분자 둘레에 공백을 끼워 넣고("US 2019 / 0236835") 종류 코드를 다른 줄로
# 밀어내는 일이 잦으므로, 공백을 허용하고 종류 코드는 선택으로 둡니다.
# 종류 코드는 반드시 문자+숫자(A1·B1·B2)여야 합니다. 문자 하나만 허용하면 뒤따르는
# 제목의 첫 글자를 종류 코드로 삼켜버립니다("US 11,456,887 VIRTUAL…" → "US 11,456,887 V").
_DOCUMENT_NUMBER_PATTERNS = (
    re.compile(r"\b(US\s*(?:20)?\d{2}\s*/?\s*\d{6,7}(?:\s+[A-Z]\d)?)", re.IGNORECASE),
    re.compile(r"\b(US\s*\d{1,2}\s*,?\s*\d{3}\s*,?\s*\d{3}(?:\s+[A-Z]\d)?)", re.IGNORECASE),
    re.compile(r"\b(WO\s*\d{4}\s*/?\s*\d{6}(?:\s+[A-Z]\d)?)", re.IGNORECASE),
    re.compile(r"\b(EP\s*\d{7}(?:\s+[A-Z]\d)?)", re.IGNORECASE),
    re.compile(r"\b(CN\s*\d{8,10}\s*[A-Z]?\d?)", re.IGNORECASE),
    re.compile(r"\b((?:JP|CN)\s*\d{4}\s*-?\s*\d{6,7}(?:\s+[A-Z]?\d)?)", re.IGNORECASE),
    re.compile(r"(10\s*-\s*\d{4}\s*-\s*\d{7})"),
    re.compile(r"(10\s*-\s*\d{7})"),
)
# 다른 나라 공보의 첫 페이지에는 우선권·PCT·패밀리 번호가 함께 나옵니다. 일반 US 패턴을
# 먼저 적용하면 한국 공개번호 대신 PCT/US 번호를 대표 문헌번호로 잘못 고릅니다. 공보가
# 자기 번호라고 명시한 (11) 공개번호·등록번호는 국가별 일반 패턴보다 우선합니다.
_LABELED_DOCUMENT_NUMBER_PATTERNS = (
    re.compile(
        r"(?:\(\s*11\s*\)\s*)?(?:공개번호|등록번호|공고번호)\s*[:：]?\s*"
        r"(10\s*-\s*\d{4}\s*-\s*\d{7})"
    ),
    re.compile(
        r"(?:Publication\s+(?:No\.?|Number)|Patent\s+No\.?)\s*[:：]?\s*"
        r"((?:US|WO|EP|CN|JP)\s*[0-9][0-9\s/,.-]*(?:\s+[A-Z]\d?)?)",
        re.IGNORECASE,
    ),
)
# 공보 단락번호 표기. 최근 US 공보는 [0039], 2010년 전후 공보는 0039. / (0039) / 0039␣
# 형태로 추출됩니다. 대괄호만 인식하면 구형 공보의 인용 위치가 전부 페이지 번호로 떨어지고,
# 청크도 페이지 블록 단위로 뭉쳐 근거 위치가 뭉툭해집니다.
#
# 네 자리 숫자 사이에 공백이나 마침표가 하나 끼어드는 경우(`0.617`, `0 617`)까지 받습니다.
# 구형 공보에서 추출기가 자주 만들어 내는 형태인데, 이를 빼면 한 문헌의 단락 절반가량이
# 인식되지 않고 그 단락들이 앞 청크에 흡수되어 **다른 단락번호로 인용**됩니다.
_MARKER = r"(\d[ .]?\d{3})"
# 번호 뒤 마침표·괄호는 같은 문헌 안에서도 들쭉날쭉하므로 선택으로 둡니다. 이를 별개
# 표기로 나누면 한 문헌의 단락이 두 패턴에 쪼개져 어느 쪽도 전체를 잡지 못합니다.
_PARAGRAPH_STYLES = (
    re.compile(rf"(?m)^\s*\[\s*{_MARKER}\s*\]"),
    re.compile(rf"(?m)^\s*\(\s*{_MARKER}\s*\)"),
    re.compile(rf"(?m)^\s*{_MARKER}\s*[.)]?(?=\s)"),
)
# 표기 하나를 문헌의 단락번호로 인정하기 위한 최소 출현 수.
_MIN_PARAGRAPH_MARKERS = 8
_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?("
    r"abstract|introduction|related\s+work|background|method(?:s|ology)?|approach|"
    r"experiment(?:s|al\s+results)?|result(?:s)?|discussion|conclusion(?:s)?|"
    r"references|bibliography|acknowledg(?:e)?ments?|appendix|"
    r"초록|요약|서론|관련\s*연구|배경|방법|실험|결과|고찰|결론|참고\s*문헌|감사의\s*글|부록"
    r")\s*$",
    re.IGNORECASE,
)
# 비교 대상에서 제외할 섹션. 근거로 인용되어도 기술적 개시로 볼 수 없는 부분입니다.
_EXCLUDED_SECTIONS = {"references", "bibliography", "acknowledgment", "acknowledgement",
                      "acknowledgments", "acknowledgements", "참고문헌", "감사의글", "부록", "appendix"}
_PATENT_HINTS = ("claims", "what is claimed", "특허청구범위", "청구범위", "발명의 상세한 설명",
                 "detailed description", "prior art", "int. cl", "patent application publication")
_PAPER_HINTS = ("abstract", "introduction", "references", "doi", "arxiv", "et al.")

# 2단 조판 판별·복원 상수. 전부 페이지 좌표(pt) 기준이며 단위는 추출기가 만든 '줄'입니다.
_GUTTER_MARGIN = 6.0        # 중앙 경계선의 좌우 여유 폭
_MIN_COLUMN_LINES = 6       # 한 단으로 인정할 최소 줄 수
_STRADDLE_RATIO = 0.10      # 중앙을 가로지르는 줄이 이 비율을 넘으면 단일 단으로 봅니다
_PARAGRAPH_GAP = 1.5        # 줄 간격이 중앙값의 이 배를 넘으면 문단 경계로 봅니다
_MIN_CHUNK_CHARS = 20       # 이보다 짧은 조각은 근거로 쓸 수 없어 청크로 만들지 않습니다


def extract_pdf(path: Path, document_id: str) -> Document:
    """PDF 한 건을 chunk 단위로 펼칩니다. 페이지 텍스트 원본은 보관하지 않습니다."""
    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page_text(page))
    full_text = "\n".join(pages)
    paragraph_pattern = detect_paragraph_pattern(full_text)
    doc_type = classify(full_text, paragraph_pattern)
    chunks = _chunk_pages(pages, document_id, doc_type, paragraph_pattern)
    publication_date, filing_date = extract_dates(full_text, doc_type)
    return Document(
        id=document_id,
        filename=path.name,
        type=doc_type,
        document_number=extract_document_number(full_text),
        title=_extract_title(pages),
        publication_date=publication_date,
        filing_date=filing_date,
        chunks=chunks,
        ocr_required=len(full_text.strip()) < 40,
    )


def page_text(page) -> str:
    """페이지 1장의 본문. 2단 조판이면 단을 갈라 읽기 순서를 복원합니다.

    단을 나눌 만큼 줄이 없는 페이지(표지·도면 등)는 추출기 기본 순서를 그대로 씁니다.
    """
    lines = _page_lines(page)
    if len(lines) < _MIN_COLUMN_LINES * 2:
        return page.get_text("text").strip()
    columns = [_column_text(column) for column in _split_columns(lines, page.rect)]
    return "\n\n".join(column for column in columns if column).strip()


def _page_lines(page) -> list[tuple[float, float, float, str]]:
    """추출기가 조립한 줄을 (좌, 우, 상, 본문)으로 펼칩니다.

    낱말 좌표로 줄을 다시 묶지 않는 것이 핵심입니다. 어센더나 대문자가 없는 낱말
    ("can", "or", "so")은 같은 줄에 있어도 글리프 상단이 2~3pt 낮게 잡혀, 허용 오차로
    묶으면 별도 줄로 갈라진 뒤 뒤로 밀립니다. 그 결과 "the software program code
    decrypt and re-encrypt the / can"처럼 문장이 흐트러져 원문 대조가 통째로 실패합니다.
    """
    lines: list[tuple[float, float, float, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = re.sub(r"\s+", " ", "".join(span.get("text", "") for span in line.get("spans", []))).strip()
            if not text:
                continue
            x0, top, x1, _ = line["bbox"]
            lines.append((x0, x1, top, text))
    return lines


def _split_columns(lines: list, rect) -> list[list]:
    """중앙선을 가로지르는 줄이 거의 없을 때만 2단으로 봅니다.

    단 통합 제목·표가 많은 페이지는 가로지르는 줄이 늘어나므로 단일 단으로 남습니다.
    """
    middle = (rect.x0 + rect.x1) / 2
    straddling = sum(1 for x0, x1, _, _ in lines
                     if x0 < middle - _GUTTER_MARGIN and x1 > middle + _GUTTER_MARGIN)
    if straddling > len(lines) * _STRADDLE_RATIO:
        return [lines]
    left = [line for line in lines if (line[0] + line[1]) / 2 <= middle]
    right = [line for line in lines if (line[0] + line[1]) / 2 > middle]
    if min(len(left), len(right)) < _MIN_COLUMN_LINES:
        return [lines]
    return [left, right]


def _column_text(lines: list) -> str:
    """단 하나를 위에서 아래로 세우고, 줄 간격이 벌어진 곳을 문단 경계로 되살립니다.

    _chunk_pages가 빈 줄로 청크를 나누므로 여기서 문단 경계를 복원해야 청크 단위가 유지됩니다.
    """
    ordered = sorted(lines, key=lambda line: (round(line[2], 1), line[0]))
    if not ordered:
        return ""
    gaps = [later[2] - earlier[2] for earlier, later in zip(ordered, ordered[1:]) if later[2] > earlier[2]]
    typical = median(gaps) if gaps else 0.0
    text = [ordered[0][3]]
    for previous, current in zip(ordered, ordered[1:]):
        spaced = bool(typical) and current[2] - previous[2] > typical * _PARAGRAPH_GAP
        text.append(("\n\n" if spaced else "\n") + current[3])
    return "".join(text)


def detect_paragraph_pattern(text: str) -> re.Pattern | None:
    """이 문헌이 실제로 쓰는 단락번호 표기 하나만 고릅니다.

    네 표기를 동시에 허용하면 표 안의 네 자리 수치나 참고문헌 번호가 단락번호로 잡혀
    엉뚱한 위치가 인용됩니다. 단락번호는 문헌을 따라 대체로 증가한다는 성질로 그런
    오검출을 걸러내고, 가장 많이 나타난 표기 하나만 채택합니다.
    """
    best, best_count = None, 0
    for pattern in _PARAGRAPH_STYLES:
        numbers = [int(paragraph_number(value)) for value in pattern.findall(text)]
        if len(numbers) < _MIN_PARAGRAPH_MARKERS or not _mostly_increasing(numbers):
            continue
        if len(numbers) > best_count:
            best, best_count = pattern, len(numbers)
    return best


def paragraph_number(marker: str) -> str:
    """추출기가 숫자 사이에 끼워 넣은 공백·마침표를 지운 네 자리 단락번호."""
    return re.sub(r"\D", "", marker)


def _mostly_increasing(numbers: list[int]) -> bool:
    """단락번호처럼 대체로 오름차순인지. 표의 수치 나열과 구분하는 기준입니다."""
    steps = list(zip(numbers, numbers[1:]))
    if not steps:
        return False
    return sum(1 for earlier, later in steps if later > earlier) >= len(steps) * 0.8


def classify(text: str, paragraph_pattern: re.Pattern | None = None) -> str:
    """단락번호와 공보 서지 어휘를 우선 보고, 그 다음 논문 어휘를 봅니다."""
    normalized = text.lower()
    if paragraph_pattern is not None:
        return "patent"
    if any(hint in normalized for hint in _PATENT_HINTS):
        return "patent"
    if sum(hint in normalized for hint in _PAPER_HINTS) >= 2:
        return "paper"
    return "technical"


def extract_document_number(text: str) -> str:
    head = text[:4000]
    for pattern in _LABELED_DOCUMENT_NUMBER_PATTERNS:
        found = pattern.search(head)
        if found:
            return _normalize_document_number(found.group(1))
    for pattern in _DOCUMENT_NUMBER_PATTERNS:
        found = pattern.search(head)
        if found:
            return _normalize_document_number(found.group(1))
    return ""


def _normalize_document_number(value: str) -> str:
    """PDF 추출기가 구분자 둘레에 넣은 공백만 제거합니다."""
    number = re.sub(r"\s*([/,-])\s*", r"\1", re.sub(r"\s+", " ", value))
    return number.strip()


def extract_dates(text: str, doc_type: str = "technical") -> tuple[str, str]:
    """공보의 공개·출원일과 논문의 최초 제출일을 감사용 메타데이터로 추출합니다.

    이 값만으로 선행기술 적격성을 판단하지 않습니다. 대상 청구항의 우선일과 적용 법역이
    별도로 필요하므로, 보고서에는 확인된 문헌 날짜와 그 한계를 함께 표시합니다.
    """
    head = text[:6000]
    publication = _labeled_date(
        head,
        (r"(?:申请公布日|公开日|Publication\s+Date|(?<!국제)공개일자|(?<!국제)공개일)\s*[:：]?\s*",
         r"(?:Published|Publication)\s*[:：]?\s*"),
    )
    filing = _labeled_date(
        head,
        (r"(?:申请日|Filing\s+Date|출원일자|출원일)(?:\s*\(국제\))?\s*[:：]?\s*",),
    )
    if doc_type == "patent" and not publication:
        publication = _us_header_publication_date(head)
    if doc_type == "paper" and not publication:
        arxiv = re.search(
            r"arXiv:\s*\d{4}\.\d+(?:v\d+)?\s*\[[^\]]+\]\s*"
            r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})",
            head,
            re.IGNORECASE,
        )
        if arxiv:
            try:
                publication = datetime.strptime(" ".join(arxiv.groups()), "%d %b %Y").date().isoformat()
            except ValueError:
                pass
    return publication, filing


def _us_header_publication_date(text: str) -> str:
    """US 공개공보 머리말의 ``Nov. 9, 2023`` 형식 공개일을 찾습니다.

    같은 면에 우선권·계속출원 날짜가 여럿 있으므로 아무 영문 날짜나 고르지 않습니다.
    공개번호 연도와 같은 날짜만 채택하면 US 2023/… 공보의 2021년 계속출원일을 공개일로
    잘못 쓰는 일을 피할 수 있습니다.
    """
    number = extract_document_number(text)
    year_match = re.match(r"US\s*(20\d{2})\s*/", number, re.IGNORECASE)
    if not year_match:
        return ""
    publication_year = int(year_match.group(1))
    month = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
             r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?")
    for found in re.finditer(
            rf"\b({month})\.?\s+(\d{{1,2}}),\s+(20\d{{2}})\b", text, re.IGNORECASE):
        if int(found.group(3)) != publication_year:
            continue
        try:
            normalized = f"{found.group(1)[:3]} {found.group(2)} {found.group(3)}"
            return datetime.strptime(normalized, "%b %d %Y").date().isoformat()
        except ValueError:
            continue
    return ""


def _labeled_date(text: str, labels: tuple[str, ...]) -> str:
    date = r"(20\d{2})\s*[.\-/年년]\s*(\d{1,2})\s*[.\-/月월]\s*(\d{1,2})\s*(?:日|일)?"
    for label in labels:
        found = re.search(label + date, text, re.IGNORECASE)
        if found:
            try:
                return datetime(int(found.group(1)), int(found.group(2)), int(found.group(3))).date().isoformat()
            except ValueError:
                continue
    return ""


def _extract_title(pages: list[str]) -> str:
    for line in (pages[0] if pages else "").splitlines():
        line = line.strip()
        if 10 <= len(line) <= 200 and not line.isdigit():
            return line
    return ""


def _chunk_pages(pages: list[str], document_id: str, doc_type: str,
                 paragraph_pattern: re.Pattern | None) -> list[Chunk]:
    """근거 위치가 단락 하나를 가리키도록, 단락번호가 있으면 그 경계로 청크를 나눕니다.

    조판 블록만으로 나누면 한 청크에 여러 단락이 들어가 인용 위치가 뭉툭해지고, 모델이
    고른 chunk_id와 실제 발췌 위치가 어긋나 검증 단계에서 자동 복구를 거치게 됩니다.
    """
    chunks: list[Chunk] = []
    section = ""
    for page_number, text in enumerate(pages, 1):
        block_index = 0
        for block in re.split(r"\n\s*\n", text):
            heading = _section_of(block)
            if heading:
                section = heading
                continue
            if doc_type == "paper" and _is_excluded_section(section):
                continue
            for paragraph, body in _split_paragraphs(block, paragraph_pattern):
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) < _MIN_CHUNK_CHARS:
                    continue
                if paragraph:
                    chunk_id = f"D{document_id}-P-{paragraph}"
                else:
                    block_index += 1
                    chunk_id = f"D{document_id}-B-p{page_number:03d}-{block_index:02d}"
                chunks.append(Chunk(
                    document_id=document_id, chunk_id=chunk_id, page=page_number,
                    paragraph=paragraph, section=section, text=body,
                ))
    return chunks


def _split_paragraphs(block: str, pattern: re.Pattern | None) -> list[tuple[str | None, str]]:
    """조판 블록을 단락번호 경계로 자릅니다. 번호가 없으면 블록 하나를 그대로 돌려줍니다."""
    if pattern is None:
        return [(None, block)]
    marks = list(pattern.finditer(block))
    if not marks:
        return [(None, block)]
    segments: list[tuple[str | None, str]] = []
    if marks[0].start() > 0:
        segments.append((None, block[:marks[0].start()]))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(block)
        # 번호 표기를 본문에 남깁니다. 모델이 "[0052] If the stream…"처럼 번호까지 포함해
        # 인용하는 일이 잦은데, 잘라내면 그 발췌가 원문 대조에서 실패합니다.
        segments.append((paragraph_number(mark.group(1)), block[mark.start():end]))
    return segments


def _section_of(block: str) -> str:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) > 60:
        return ""
    found = _SECTION_RE.match(lines[0])
    return found.group(1).strip() if found else ""


def _is_excluded_section(section: str) -> bool:
    return re.sub(r"[\s.]", "", section).lower() in _EXCLUDED_SECTIONS


def document_corpus(document: Document) -> str:
    """발췌 검증에 쓰는 문서 전체 말뭉치. 비교에 넘긴 chunk와 정확히 같은 범위입니다."""
    return "\n".join(chunk.text for chunk in document.chunks)


def chunk_text(document: Document, chunk_id: str) -> str:
    target = (chunk_id or "").strip()
    if not target:
        return ""
    for chunk in document.chunks:
        if chunk.chunk_id == target:
            return chunk.text
    return ""
