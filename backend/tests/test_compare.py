from app.compare import (BATCH_COMPARE_PROMPT, COMPARE_PROMPT, _build_matches,
                         select_chunks_for_claims)
from app.coverage import atomic_coverage, has_correspondence
from app.models import Chunk, Claim, ClaimElement, Document


def test_prompts_do_not_request_unused_per_element_similarity_text():
    assert "similarity_point" not in COMPARE_PROMPT
    assert "similarity_point" not in BATCH_COMPARE_PROMPT


def test_korean_claim_finds_english_body_through_element_search_terms():
    """기술분야 동의어 사전을 코드에 두지 않고, 구성요소마다 받은 검색어로 찾는다."""
    target = Claim(number=3, elements=[ClaimElement(
        label="A", text="회의 발언을 분석하여 질문을 자동 생성하는 기능",
        search_terms=["meeting", "utterance", "question", "generating"])])
    chunks = [Chunk(document_id="1", chunk_id=f"D1-{index}", page=index + 1,
                    text=("unrelated apparatus details " * 30))
              for index in range(60)]
    chunks[35].text = ("The system analyzes a meeting utterance and uses a transcript for "
                       "automatically generating questions relating to the presentation session.")
    document = Document(id="1", filename="prior-art.pdf", type="patent", chunks=chunks)

    selected = select_chunks_for_claims([target], document, document_count=6)

    assert any("automatically generating questions" in chunk.text for chunk in selected)
    # 제목 한 두 줄만 선택되는 회귀도 막습니다.
    assert sum(len(chunk.text) for chunk in selected) >= 10000


def test_every_element_gets_its_own_share_of_the_context_budget():
    """청구항 전체 키워드 하나로 고르면 흔한 구성이 예산을 독점하고 희소 구성이 밀려난다."""
    common = Claim(number=1, elements=[
        ClaimElement(label="A", text="요청을 수신하는 단계", search_terms=["request", "receive"]),
        ClaimElement(label="B", text="콜드 스토리지로 이전하는 단계",
                     search_terms=["cold storage", "archive", "tier"]),
    ])
    chunks = [Chunk(document_id="1", chunk_id=f"D1-{index}", page=index + 1,
                    text=("the server receives a request from the client device " * 20))
              for index in range(80)]
    chunks[70].text = "Aged fragments are archived to the cold storage tier of the subsystem."
    document = Document(id="1", filename="prior-art.pdf", type="patent", chunks=chunks)

    selected = select_chunks_for_claims([common], document, budget=6000)

    assert any("archived to the cold storage" in chunk.text for chunk in selected)


def _element_with(core: str, qualifier: str) -> ClaimElement:
    return ClaimElement(label="C", text=f"{qualifier} {core}", limitations=[
        {"text": core, "kind": "core"}, {"text": qualifier, "kind": "qualifier"}])


def _judge(element: ClaimElement, quote: str, core_disclosed: bool) -> str:
    target = Claim(number=1, elements=[element])
    document = Document(id="1", filename="prior.pdf", type="patent", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0039", page=4, paragraph="0039", text=quote)])
    raw = [{
        "label": "C", "judgment": "일부 차이", "directness": "direct",
        "quote": quote, "chunk_id": "D1-P-0039",
        "limitation_checks": [
            {"index": 0, "disclosed": core_disclosed, "quote": quote, "chunk_id": "D1-P-0039"},
            {"index": 1, "disclosed": False, "quote": "", "chunk_id": ""},
        ],
    }]
    return _build_matches(raw, target, document)[0][0].judgment


def test_a_missing_qualifier_alone_does_not_erase_the_correspondence():
    """같은 동작을 다른 기준으로 하고 있을 뿐이면 '차이'가 아니라 '일부 차이'다.

    한정 문구가 섞인 조각을 전부 같은 무게로 세면, 대응 문단을 정확히 찾아 놓고도
    조건 하나가 다르다는 이유로 구성 전체가 "대응 없음"으로 보고된다.
    """
    quote = "As the fragments age out, they are deleted from the Archiver and archived to Storage."
    element = _element_with(core="콘텐츠를 내부 저장소에서 외부 저장소로 이전하여 보관함",
                            qualifier="이전 여부를 수요 지표와 임계값의 비교로 결정함")

    assert _judge(element, quote, core_disclosed=True) == "일부 차이"
    # 반대로 동작 자체가 확인되지 않으면 관련 문장이 있어도 대응으로 세지 않는다.
    assert _judge(element, quote, core_disclosed=False) == "차이"


def test_one_disclosed_alternative_satisfies_the_whole_set():
    """"A, B 또는 C 중 적어도 하나"는 하나만 개시되면 문언이 충족된다.

    나머지 대안을 누락으로 세면 선택지를 넉넉히 나열한 청구항일수록 차이점이 길어져,
    실제로는 청구항을 읽어내는 문헌이 오히려 감점된다.
    """
    quote = "The one-way hash value contains a time-to-live value."
    element = ClaimElement(label="A", text="세션 토큰은 시간 제한, 범위 제한 또는 회수 기능 중 적어도 하나", limitations=[
        {"text": "세션 토큰이 시간 제한을 가짐", "kind": "core", "alternative_group": "토큰속성"},
        {"text": "세션 토큰이 범위 제한을 가짐", "kind": "core", "alternative_group": "토큰속성"},
        {"text": "세션 토큰이 회수 기능을 가짐", "kind": "core", "alternative_group": "토큰속성"}])
    target = Claim(number=4, elements=[element])
    document = Document(id="1", filename="prior.pdf", type="patent", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0612", page=24, paragraph="0612", text=quote)])
    raw = [{"label": "A", "judgment": "실질적 동일", "directness": "direct",
            "quote": quote, "chunk_id": "D1-P-0612",
            "limitation_checks": [
                {"index": 0, "disclosed": True, "quote": quote, "chunk_id": "D1-P-0612"},
                {"index": 1, "disclosed": False, "quote": "", "chunk_id": ""},
                {"index": 2, "disclosed": False, "quote": "", "chunk_id": ""}]}]

    match = _build_matches(raw, target, document)[0][0]

    assert match.missing_limitations == []          # 충족된 묶음의 미개시 대안은 차이가 아니다
    assert match.judgment == "실질적 동일"           # 누락이 없으므로 강등되지 않는다
    assert atomic_coverage(match) == 1.0            # 분모에서도 빠진다


def test_an_unsatisfied_alternative_set_is_still_missing():
    """묶음의 어느 대안도 개시되지 않았다면 그 묶음은 여전히 누락이다."""
    element = ClaimElement(label="A", text="세션 토큰은 시간 제한 또는 회수 기능 중 적어도 하나", limitations=[
        {"text": "세션 토큰이 시간 제한을 가짐", "kind": "core", "alternative_group": "토큰속성"},
        {"text": "세션 토큰이 회수 기능을 가짐", "kind": "core", "alternative_group": "토큰속성"}])
    target = Claim(number=4, elements=[element])
    document = Document(id="1", filename="prior.pdf", type="patent", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0612", page=24, paragraph="0612", text="무관한 기재")])
    raw = [{"label": "A", "judgment": "일부 차이", "directness": "inferred", "quote": "", "chunk_id": "",
            "limitation_checks": [{"index": 0, "disclosed": False, "quote": "", "chunk_id": ""},
                                  {"index": 1, "disclosed": False, "quote": "", "chunk_id": ""}]}]

    match = _build_matches(raw, target, document)[0][0]

    assert len(match.missing_limitations) == 2
    assert match.judgment == "차이"


def test_batch_match_requires_every_atomic_limitation_check():
    quote = "The renderer adopts a level-of-detail strategy for real-time rendering."
    target = Claim(number=13, elements=[ClaimElement(
        label="A", text="LOD를 반영하여 동적으로 로딩 및 렌더링하는 것",
        limitations=["LOD를 반영함", "가우시안 스플랫을 동적으로 로딩함"])
    ])
    document = Document(id="2", filename="paper.pdf", type="paper", chunks=[
        Chunk(document_id="2", chunk_id="D2-B-p004-07", page=4, text=quote)
    ])
    raw = [{
        "label": "A", "judgment": "동일", "directness": "direct",
        "quote": quote, "chunk_id": "D2-B-p004-07", "missing_limitations": [],
        "limitation_checks": [{
            "index": 0, "disclosed": True, "quote": quote, "chunk_id": "D2-B-p004-07"
        }],
    }]

    matches, warnings = _build_matches(raw, target, document)

    assert matches[0].judgment == "일부 차이"
    assert matches[0].missing_limitations == ["가우시안 스플랫을 동적으로 로딩함"]
    assert len(matches[0].limitation_checks) == 2
    assert warnings


def test_zero_disclosed_limitations_cannot_be_reported_as_partial_disclosure():
    """관련 문장만 있고 청구항 하위 제한은 0/N이면 87% 대응으로 흘려보내지 않는다."""
    quote = "The application receives a general request to play selected content."
    target = Claim(number=1, elements=[ClaimElement(
        label="D", text="가상 세션을 생성하고 세션 토큰을 발급함",
        limitations=["가상 세션을 생성함", "세션 토큰을 발급함"])
    ])
    document = Document(id="1", filename="playback.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", text=quote)
    ])
    raw = [{
        "label": "D", "judgment": "일부 차이", "directness": "inferred",
        "quote": quote, "chunk_id": "D1-P-0001",
        "limitation_checks": [
            {"index": 0, "disclosed": False, "quote": "", "chunk_id": ""},
            {"index": 1, "disclosed": False, "quote": "", "chunk_id": ""},
        ],
    }]

    matches, _ = _build_matches(raw, target, document)
    match = matches[0]
    match.verify = "verified"

    assert match.judgment == "차이"
    assert match.downgraded_from == "일부 차이"
    assert has_correspondence(match) is False


# --- 단건·일괄 경로의 스키마 동일성 -------------------------------------------

def test_both_paths_demand_the_same_evidence_schema():
    """같은 청구항이 최초 분석에 있었는지 나중에 추가됐는지에 따라 기준이 달라지면 안 된다."""
    for prompt in (COMPARE_PROMPT, BATCH_COMPARE_PROMPT):
        assert "requirements" in prompt
        assert "limitation_checks" in prompt
        assert "하위 제한 점검" in prompt
        assert '"차이"는 관련 기술 기재가 실제로 있는 경우' in prompt


def test_single_path_sends_requirements_and_requires_the_checks(monkeypatch):
    from app import compare

    captured: dict = {}

    def fake(prompt: str, expect: str = "claims"):
        captured["prompt"] = prompt
        # limitation_checks를 돌려주지 않는 응답
        return {"matches": [{"label": "A", "judgment": "동일", "directness": "direct",
                             "quote": "원문 발췌 문장입니다", "chunk_id": "D1-P-0001"}]}

    monkeypatch.setattr(compare, "run_cli", fake)
    target = Claim(number=1, elements=[ClaimElement(
        label="A", text="구성 A", limitations=["첫째 조건", "둘째 조건"])])
    document = Document(id="1", filename="d1.pdf",
                        chunks=[Chunk(document_id="1", chunk_id="D1-P-0001", text="본문")])

    matches, warnings = compare.compare_document(target, document)

    assert '"requirements"' in captured["prompt"]
    assert "첫째 조건" in captured["prompt"]
    # 점검 결과가 없으면 미개시로 처리되고 판정도 강등된다(일괄 경로와 동일).
    assert matches[0].judgment == "차이" and matches[0].downgraded_from == "동일"
    assert matches[0].missing_limitations == ["첫째 조건", "둘째 조건"]
    assert warnings and "하위 제한" in warnings[0]


def test_the_whole_element_fallback_is_not_reported_as_a_missing_sub_limitation():
    """분해 결과가 없어 구성 원문 한 줄을 점검한 경우, 그 실패는 '누락 한정'이 아니다."""
    element = ClaimElement(label="A", text="구성 A 원문", limitations=[])
    target = Claim(number=1, elements=[element])
    document = Document(id="1", filename="d1.pdf")
    matches, _ = _build_matches(
        [{"label": "A", "judgment": "대응 없음", "directness": "absent",
          "limitation_checks": [{"index": 0, "disclosed": False}]}],
        target, document)

    assert matches[0].missing_limitations == []
    assert matches[0].limitation_checks[0].whole_element is True


def test_document_text_is_fenced_as_untrusted_and_the_guideline_is_subordinate(monkeypatch):
    """CLI에는 진짜 system/developer 경계가 없다. 최소한 순서와 표시로 구분한다."""
    from app import compare

    captured: dict = {}

    def fake(prompt: str, expect: str = "claims"):
        captured["p"] = prompt
        return {"matches": []}

    monkeypatch.setattr(compare, "run_cli", fake)
    target = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A")])
    document = Document(id="1", filename="d1.pdf",
                        chunks=[Chunk(document_id="1", chunk_id="D1-P-0001", text="본문")])

    compare.compare_document(target, document, guideline="사용자가 넣은 지침")
    prompt = captured["p"]

    assert "비신뢰 데이터" in prompt
    # 불변 규칙이 사용자 지침보다 앞에 오고, 비신뢰 문헌은 맨 뒤에 온다.
    assert prompt.index("[판정 라벨]") < prompt.index("사용자가 넣은 지침") < prompt.index("CONTEXT:")
