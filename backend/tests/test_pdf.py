"""PDF 추출. 공보의 2단 조판을 읽기 순서대로 되살리지 못하면 이후 발췌 검증이 전부 실패한다."""
import fitz
import pytest

from app.pdf import classify, extract_dates, extract_pdf, document_corpus, page_text
from app.verify import is_verbatim

# 단 판별에는 최소 낱말 수가 필요하므로, 대비할 문장 뒤에 채움 줄을 붙여 실제 공보 밀도를 흉내낸다.
LEFT = (["a communication server in", "communication with the server", "and a plurality of devices"]
        + [f"left filler line number {index} of the specification" for index in range(12)])
RIGHT = (["bus 1012 may include an", "Accelerated Graphics Port", "or another suitable bus"]
        + [f"right filler line number {index} of the specification" for index in range(12)])


def test_a_paper_disclaimer_containing_jurisdictional_claims_is_not_a_patent():
    text = ("Abstract Introduction DOI 10.3390/example References. "
            "The publisher stays neutral with regard to jurisdictional claims in published maps.")
    assert classify(text) == "paper"


def two_column_pdf(path, left=LEFT, right=RIGHT) -> str:
    """좌우 단의 같은 줄이 y좌표를 공유하는, 공보와 같은 조판을 만든다."""
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    for index, (left_line, right_line) in enumerate(zip(left, right)):
        top = 100 + index * 14
        page.insert_text((60, top), left_line, fontsize=9)
        page.insert_text((330, top), right_line, fontsize=9)
    document.save(path)
    document.close()
    return str(path)


def test_columns_are_read_down_one_side_before_the_other(tmp_path):
    path = two_column_pdf(tmp_path / "two-column.pdf")
    with fitz.open(path) as document:
        text = page_text(document[0])
    assert "a communication server in communication with the server" in " ".join(text.split())
    assert text.index("or another suitable bus") > text.index("and a plurality of devices")


def test_interleaved_columns_would_break_quotation_checks(tmp_path):
    """단을 가르지 않으면 한 문장이 반대쪽 단의 줄에 끊긴다. 이 회귀를 막는 것이 목적이다."""
    path = two_column_pdf(tmp_path / "two-column.pdf")
    with fitz.open(path) as document:
        naive = document[0].get_text("text")
    quote = "a communication server in communication with the server"
    assert not is_verbatim(quote, naive)                       # 기본 읽기 순서로는 실패하고
    assert is_verbatim(quote, document_corpus(extract_pdf(tmp_path / "two-column.pdf", "1")))   # 복원 후에는 통과한다


def test_a_single_column_page_is_left_alone(tmp_path):
    """표·도면처럼 단을 가로지르는 페이지까지 억지로 쪼개면 안 된다."""
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    for index in range(20):
        page.insert_text((60, 100 + index * 14),
                         f"line {index} runs across the entire width of this page without any gutter",
                         fontsize=9)
    path = tmp_path / "one-column.pdf"
    document.save(path)
    document.close()
    with fitz.open(path) as opened:
        text = page_text(opened[0])
    assert text.count("runs across the entire width") == 20
    assert "line 0 runs across the entire width of this page without any gutter" in text


@pytest.mark.parametrize("header, expected", [
    ("US 2019 / 0236835 A1 EXTENDED REALITY VIRTUAL ASSISTANT", "US 2019/0236835 A1"),
    ("US 11,456,887 VIRTUAL MEETING FACILITATOR", "US 11,456,887"),
])
def test_document_numbers_survive_the_spacing_the_extractor_adds(header, expected):
    from app.pdf import extract_document_number
    assert extract_document_number(header) == expected


def test_a_chinese_publication_number_and_dates_are_extracted():
    from app.pdf import extract_document_number
    text = "CN 119359955 A 申请公布日 2025.01.24 申请日 2024年09月18日"
    assert extract_document_number(text) == "CN 119359955 A"
    assert extract_dates(text, "patent") == ("2025-01-24", "2024-09-18")


def test_labeled_korean_publication_number_wins_over_pct_us_number():
    """한국 공보 첫 장의 PCT/US 번호를 대표 문헌번호로 오인하면 안 된다."""
    from app.pdf import extract_document_number
    text = (
        "(86) 국제출원번호 PCT/US2022/015581 (87) 국제공개번호 WO 2022/173721\n"
        "(11) 공개번호 10-2023-0140574 (43) 공개일자 2023년10월06일"
    )
    assert extract_document_number(text) == "10-2023-0140574"
    assert extract_dates(text, "patent")[0] == "2023-10-06"


def test_local_korean_publication_date_wins_over_international_publication_date():
    text = (
        "(22) 출원일자(국제) 2022년02월08일 "
        "국제공개일자 2022년08월18일 "
        "(43) 공개일자 2023년10월06일"
    )
    assert extract_dates(text, "patent") == ("2023-10-06", "2022-02-08")


@pytest.mark.parametrize("text, expected", [
    ("US 2023/0362144 A1 filed Jul. 28, 2021 Nov. 9, 2023", "2023-11-09"),
    ("US 2011/0173345 A1 claims priority Aug. 17, 2009 Jul. 14, 2011", "2011-07-14"),
])
def test_us_header_publication_date_uses_the_publication_number_year(text, expected):
    assert extract_dates(text, "patent")[0] == expected


def test_an_arxiv_submission_date_is_extracted_for_a_paper():
    text = "arXiv:2412.01234v2 [cs.CV] 2 Dec 2024\nAbstract"
    assert extract_dates(text, "paper") == ("2024-12-02", "")
