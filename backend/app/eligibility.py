"""선행기술 적격성 **분류**. 적격/부적격을 단정하지 않고 확인할 것을 갈라 놓습니다.

이 도구는 "이 문헌은 선행기술이다/아니다"를 결론짓지 않습니다. 적격성은 적용 법역,
신규성 판단인지 진보성 판단인지, 동일 출원인·발명자 예외에 해당하는지까지 확인해야
정해지고, 그중 어느 것도 업로드된 PDF만으로는 알 수 없습니다.

여기서 하는 일은 **날짜만으로 가를 수 있는 것**을 갈라, 각 문헌에 대해 심사관이 무엇을
더 확인해야 하는지 드러내는 것까지입니다. 특히 후공개 선출원(공개는 대상 출원 뒤지만
출원은 앞선 문헌)은 주요 법역에서 신규성 근거로만 쓸 수 있고 진보성 근거로는 쓸 수
없으므로, 통상의 선행기술과 같은 칸에 넣으면 안 됩니다.

날짜를 확인하지 못한 문헌은 적격도 부적격도 아닌 '날짜 불명'입니다. 추출 실패를 적격으로
흘리면 없는 근거 위에 거절 이유가 서고, 부적격으로 흘리면 멀쩡한 인용발명이 조용히
사라집니다. 공개일이 비어 있는 문헌은 드물지 않으므로 이 칸은 예외가 아니라 흔한 경우입니다.
"""
import re

from .models import Document

ORDINARY = "통상 선행기술"
SECRET_PRIOR_APPLICATION = "선출원 후보(후공개)"
LATER = "후행 문헌"
UNKNOWN = "날짜 불명"
NO_PRIORITY_DATE = "대상 우선일 미입력"

_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_priority_date(value: str) -> str:
    """YYYY-MM-DD만 받습니다. 형식이 어긋나면 빈 문자열이라 분류가 통째로 보류됩니다."""
    text = re.sub(r"[./]", "-", str(value or "").strip())
    return text if _ISO.match(text) else ""


def classify_document(document: Document, priority_date: str) -> tuple[str, str]:
    """(분류, 사유). 문헌 1건을 대상 우선일에 비추어 가릅니다.

    ISO 문자열은 사전식 비교가 곧 시간순 비교라 날짜 파싱을 따로 하지 않습니다.
    """
    if not priority_date:
        return NO_PRIORITY_DATE, "대상 청구항의 출원일·우선일이 입력되지 않아 적격성을 가리지 않았습니다."
    published, filed = document.publication_date, document.filing_date
    if not published and not filed:
        return UNKNOWN, "공개일과 출원일을 모두 추출하지 못해 적격성을 가릴 수 없습니다."
    if published and published < priority_date:
        return ORDINARY, f"공개일 {published}이 대상 우선일 {priority_date}보다 앞섭니다."
    if filed and filed < priority_date:
        # 공개는 뒤지만 출원이 앞선 문헌. 한국 특허법 제29조제3항, EPC Art. 54(3),
        # 35 U.S.C. §102(a)(2)가 각각 다루며 요건도 예외도 법역마다 다릅니다.
        detail = f"공개일 {published}" if published else "공개일 미확인"
        return SECRET_PRIOR_APPLICATION, (
            f"출원일 {filed}은 대상 우선일 {priority_date}보다 앞서지만 {detail}입니다. "
            "후공개 선출원은 주요 법역에서 **신규성 근거로만** 쓸 수 있고 진보성 근거로는 "
            "쓸 수 없으며, 동일 출원인·발명자 예외에 해당하는지 별도로 확인해야 합니다.")
    if published and published >= priority_date and not filed:
        return UNKNOWN, (f"공개일 {published}이 대상 우선일 {priority_date} 이후이나 출원일을 "
                         "추출하지 못해 선출원 여부를 가릴 수 없습니다.")
    return LATER, (f"출원일 {filed or '미확인'}·공개일 {published or '미확인'} 모두 대상 우선일 "
                   f"{priority_date} 이후입니다. 선행기술로 쓸 수 없습니다.")


def classify(documents: list[Document], priority_date: str = "") -> list[dict]:
    normalized = normalize_priority_date(priority_date)
    return [{"document_id": document.id, "filename": document.filename,
             "publication_date": document.publication_date, "filing_date": document.filing_date,
             "category": category, "detail": detail}
            for document in documents
            for category, detail in [classify_document(document, normalized)]]


def warnings(documents: list[Document], priority_date: str = "") -> list[str]:
    """보고서 validation에 실을 줄. 분류별로 묶어 후속 조치가 다른 것을 섞지 않습니다."""
    if not documents:
        return []
    rows = classify(documents, priority_date)
    normalized = normalize_priority_date(priority_date)
    if not normalized:
        raw = str(priority_date or "").strip()
        head = ("입력한 우선일 형식을 인식하지 못했습니다(YYYY-MM-DD로 입력하십시오). "
                if raw else "")
        dated = ", ".join(f"{row['filename']}={row['publication_date'] or row['filing_date'] or '미확인'}"
                          for row in rows)
        return [f"{head}대상 청구항의 출원일·우선일이 없어 선행기술 적격성을 가리지 않았습니다. "
                f"이 보고서는 기술적 구성대비만 수행합니다. 확인된 문헌 날짜: {dated}."]

    notes = [f"대상 우선일 {normalized} 기준으로 분류했습니다. 적격성 최종 판단에는 적용 법역과 "
             "신규성·진보성 구분, 동일 출원인·발명자 예외 확인이 함께 필요합니다."]
    for category, lead in (
        (LATER, "다음 문헌은 대상 우선일 이후의 문헌이라 선행기술로 쓸 수 없습니다"),
        (SECRET_PRIOR_APPLICATION, "다음 문헌은 후공개 선출원 후보입니다"),
        (UNKNOWN, "다음 문헌은 날짜를 확인하지 못해 적격성을 가리지 못했습니다"),
    ):
        matched = [row for row in rows if row["category"] == category]
        if matched:
            detail = " ".join(f"{row['filename']}: {row['detail']}" for row in matched)
            notes.append(f"{lead} — {detail}")
    ordinary = [row["filename"] for row in rows if row["category"] == ORDINARY]
    if ordinary:
        notes.append(f"통상 선행기술로 볼 수 있는 문헌: {', '.join(ordinary)}.")
    return notes
