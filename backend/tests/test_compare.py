from app import compare
from app.compare import (BATCH_COMPARE_PROMPT, COMPARE_PROMPT, _build_matches, compare_document,
                         select_chunks_for_claims)
from app.coverage import atomic_coverage, derive_judgment, has_correspondence
from app.models import Chunk, Claim, ClaimElement, Document, LimitationCheck


def test_prompts_do_not_request_unused_per_element_similarity_text():
    assert "similarity_point" not in COMPARE_PROMPT
    assert "similarity_point" not in BATCH_COMPARE_PROMPT


def test_document_type_is_not_sent_as_a_semantic_judgment_criterion(monkeypatch):
    """paper/patent 유형은 섹션 정리용 메타데이터이지 구성 개시 기준이 아니다."""
    captured = {}
    quote = "The controller stores the request in a processing queue."
    target = Claim(number=1, elements=[ClaimElement(label="A", text="요청을 큐에 저장함")])
    document = Document(id="1", filename="prior.pdf", type="patent", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", page=1, text=quote)])

    def fake(prompt, expect="matches"):
        captured["prompt"] = prompt
        return {"matches": [{"label": "A", "judgment": "동일", "directness": "direct",
                             "quote": quote, "chunk_id": "D1-P-0001",
                             "limitation_checks": [{"index": 0, "disclosed": True,
                                                    "quote": quote, "chunk_id": "D1-P-0001"}]}]}

    monkeypatch.setattr(compare, "run_cli", fake)
    compare_document(target, document)

    assert '"filename": "prior.pdf"' in captured["prompt"]
    assert '"type": "patent"' not in captured["prompt"]


def test_limitation_check_preserves_multiple_evidence_spans():
    first = "GNSS neighbors are selected using a distance threshold."
    second = "Matching feature counts form the graph edge weights."
    target = Claim(number=1, elements=[ClaimElement(
        label="A", text="GNSS와 상대정합 정보를 이용해 그룹화함",
        limitations=[{"text": "GNSS와 상대정합 정보가 그룹화 기준에 함께 기여함", "kind": "core"}])])
    document = Document(id="1", filename="sensors.pdf", type="paper", chunks=[
        Chunk(document_id="1", chunk_id="D1-B-p005-01", page=5, text=first),
        Chunk(document_id="1", chunk_id="D1-B-p006-01", page=6, text=second)])
    raw = [{"label": "A", "judgment": "실질적 동일", "directness": "direct",
            "quote": first, "chunk_id": "D1-B-p005-01",
            "limitation_checks": [{"index": 0, "disclosed": True, "quote": first,
                                   "chunk_id": "D1-B-p005-01",
                                   "evidence": [{"quote": second, "chunk_id": "D1-B-p006-01"}]}]}]

    match = _build_matches(raw, target, document)[0][0]

    assert match.limitation_checks[0].disclosed is True
    assert [span.quote for span in match.limitation_checks[0].evidence] == [second]


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
    # 발췌도 evidence도 없으므로 "차이"가 아니라 "대응 없음"이다. 프롬프트의 근거 규칙이
    # "관련 원문도 제시할 수 없다면 '차이'가 아니라 '대응 없음'"이라고 정한 것과 같다.
    # 종전에는 모델이 낸 라벨을 강등하는 방식이라 근거가 하나도 없어도 "차이"에서 멈췄다.
    assert match.judgment == "대응 없음"


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

    # 두 한정이 모두 core인데 하나만 개시되었으므로 "core가 일부만 개시됨" = 일부 유사이다.
    # 종전 경로는 "core가 하나라도 개시되었는가"만 봐서 부분 개시 core를 전부 개시 core와
    # 같이 취급했고, 그래서 qualifier만 빠진 경우와 구별되지 않았다.
    assert matches[0].judgment == "일부 유사"
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
    # downgraded_from은 비어 있다. 등급을 산출하게 된 뒤로 compare 단계에는 "강등할 원 판정"이
    # 존재하지 않는다 — 모델이 라벨을 내지 않기 때문이다. 이후 단계(의미검증)의 강등만 기록된다.
    assert match.downgraded_from == ""
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
    # 점검 결과가 없으면 미개시로 처리되고, 등급도 그 결과로 산출된다(일괄 경로와 동일).
    # 발췌는 있으므로 "대응 없음"이 아니라 "차이"에서 멈춘다.
    assert matches[0].judgment == "차이" and matches[0].downgraded_from == ""
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
    # 규칙 블록의 첫 머리를 기준으로 잡는다. 개별 절 제목은 규칙이 바뀌면 함께 바뀐다.
    assert prompt.index("[요구사항의 두 종류]") < prompt.index("사용자가 넣은 지침") < prompt.index("CONTEXT:")


# --- 발췌를 첫 적중에서 끝내지 않기 -------------------------------------------

def test_the_strongest_response_wins_when_a_label_is_answered_twice():
    """같은 라벨이 두 번 오면 뒤에 온 더 강한 판정을 채택한다.

    모델이 문헌 앞쪽 총론으로 한 번 답한 뒤 뒤쪽 실시예를 찾아 다시 답하는 일이 있다.
    먼저 온 것을 집으면 그것은 "앞에서 비슷한 문장을 보고 멈춘" 결과와 똑같아진다.
    """
    target = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A", limitations=[])])
    document = Document(id="1", filename="d1.pdf")
    weak = {"label": "A", "judgment": "일부 유사", "directness": "inferred",
            "quote": "컴퓨팅 장치를 사용할 수 있다는 총론", "chunk_id": "D1-P-0003",
            "limitation_checks": [{"index": 0, "disclosed": False}]}
    strong = {"label": "A", "judgment": "실질적 동일", "directness": "direct",
              "quote": "제어부는 고유 식별값에 매칭된 영상을 호출한다", "chunk_id": "D1-P-0088",
              "limitation_checks": [{"index": 0, "disclosed": True,
                                     "quote": "제어부는 고유 식별값에 매칭된 영상을 호출한다",
                                     "chunk_id": "D1-P-0088"}]}

    matches, _ = _build_matches([weak, strong], target, document)
    reversed_matches, _ = _build_matches([strong, weak], target, document)

    assert matches[0].judgment == "실질적 동일" and matches[0].chunk_id == "D1-P-0088"
    assert reversed_matches[0].chunk_id == "D1-P-0088"      # 순서와 무관하게 같은 결과


def test_a_later_limitation_check_with_evidence_beats_an_earlier_empty_one():
    """하위 한정도 마찬가지다. 뒤에서 실제 기재를 찾아 답한 것을 버리지 않는다."""
    target = Claim(number=1, elements=[ClaimElement(
        label="A", text="구성 A", limitations=[{"text": "임계값 비교로 이전 여부를 정함", "kind": "core"}])])
    document = Document(id="1", filename="d1.pdf")

    matches, warnings = _build_matches([{
        "label": "A", "judgment": "실질적 동일", "directness": "direct",
        "quote": "원문 발췌 문장입니다", "chunk_id": "D1-P-0010",
        "limitation_checks": [
            {"index": 0, "disclosed": False},
            {"index": 0, "disclosed": True, "quote": "임계값 미만이면 외부 저장소로 이전한다",
             "chunk_id": "D1-P-0091"},
        ]}], target, document)

    assert matches[0].limitation_checks[0].disclosed is True
    assert matches[0].limitation_checks[0].chunk_id == "D1-P-0091"
    assert matches[0].missing_limitations == [] and not warnings


def test_the_prompt_requires_reading_the_document_to_the_end():
    """앞쪽 총론에서 멈추지 말라는 규칙이 두 호출 경로 모두에 있어야 한다."""
    for prompt in (COMPARE_PROMPT, BATCH_COMPARE_PROMPT):
        assert "첫 번째로 비슷해 보이는 문장에서 멈추지 마십시오" in prompt
        assert "실시예" in prompt


def test_the_middle_of_a_document_is_not_starved_by_front_loading():
    """검색이 적중하지 못해도 본문 중반·후반이 통째로 잘리지 않는다.

    공보의 앞부분은 서지사항·배경기술·요약이고, 그 구성을 실제로 어떻게 하는지는 상세한
    설명에 있다. 예산의 절반을 앞에서부터 채우면 대응 기재가 있는 구간이 먼저 잘린다.
    """
    target = Claim(number=1, elements=[ClaimElement(
        label="A", text="아무 것도 적중하지 않는 구성", search_terms=["zzzznomatch"])])
    chunks = [Chunk(document_id="1", chunk_id=f"D1-{index:03d}", page=index + 1,
                    text=f"{index:03d} " + ("본문 문장입니다. " * 40))
              for index in range(100)]
    document = Document(id="1", filename="prior-art.pdf", type="patent", chunks=chunks)

    selected = select_chunks_for_claims([target], document, budget=8000)
    positions = [int(chunk.chunk_id.split("-")[1]) for chunk in selected]

    assert positions, "안전망이 아무것도 고르지 못했습니다"
    assert max(positions) > 60, "문헌 후반이 통째로 빠졌습니다"
    assert any(30 <= position <= 60 for position in positions), "문헌 중반이 통째로 빠졌습니다"


def test_a_multiword_search_term_outranks_an_incidental_word_hit():
    """구문이 통째로 있는 문단이, 흔한 낱말 하나만 겹친 긴 문단보다 앞선다."""
    target = Claim(number=1, elements=[ClaimElement(
        label="A", text="공통 좌표계로 변환하는 단계",
        search_terms=["common coordinate system"])])
    chunks = [Chunk(document_id="1", chunk_id=f"D1-{index:03d}", page=index + 1,
                    text="the system stores data in the system memory of the system. " * 12)
              for index in range(40)]
    chunks[33].text = "Each frame is transformed into a common coordinate system before merging."
    document = Document(id="1", filename="prior-art.pdf", type="patent", chunks=chunks)

    selected = select_chunks_for_claims([target], document, budget=3000)

    assert any("common coordinate system" in chunk.text for chunk in selected)


# --- 종속항 프롬프트의 부모항 문언 --------------------------------------------

def test_a_dependent_claim_prompt_carries_the_parent_claim_text(monkeypatch):
    """"상기 이미지"가 무엇인지 부모항 문언 없이는 알 수 없다.

    종속항 행렬에는 "…에 있어서" 뒤의 추가 한정만 들어 있다. 대상을 모른 채 낱말만 맞추면
    문헌 어디에 있는 아무 "이미지" 언급이나 대응으로 잡힌다.
    """
    from app import compare

    captured: dict = {}

    def fake(prompt: str, expect: str = "claims"):
        captured["prompt"] = prompt
        return {"matches": []}

    monkeypatch.setattr(compare, "run_cli", fake)
    parent = Claim(number=1, preamble="영상 처리 장치에 있어서,",
                   elements=[ClaimElement(label="A", text="이미지를 수신하는 입력부")])
    child = Claim(number=2, depends_on=1, preamble="제1항에 있어서,",
                  elements=[ClaimElement(label="A", text="상기 이미지는 정적 이미지인 것")])
    document = Document(id="1", filename="d1.pdf",
                        chunks=[Chunk(document_id="1", chunk_id="D1-P-0001", text="본문")])

    compare.compare_document(child, document, all_claims=[parent, child])

    assert "parent_claims" in captured["prompt"]
    assert "이미지를 수신하는 입력부" in captured["prompt"]
    assert "지시어 해석 전용" in captured["prompt"]


def test_an_independent_claim_prompt_has_no_parent_block(monkeypatch):
    """독립항에는 부모항이 없다. 빈 블록을 넣어 프롬프트를 늘리지 않는다."""
    from app import compare

    captured: dict = {}

    def fake(prompt: str, expect: str = "claims"):
        captured["p"] = prompt
        return {"matches": []}

    monkeypatch.setattr(compare, "run_cli", fake)
    target = Claim(number=1, elements=[ClaimElement(label="A", text="구성 A")])
    document = Document(id="1", filename="d1.pdf",
                        chunks=[Chunk(document_id="1", chunk_id="D1-P-0001", text="본문")])

    compare.compare_document(target, document, all_claims=[target])

    assert '"parent_claims"' not in captured["p"]


# --- 판정 등급 산출 ------------------------------------------------------------

def _checks(*specs) -> list[LimitationCheck]:
    """(kind, disclosed) 쌍으로 점검 목록을 만든다. quote는 개시된 항목에만 붙인다."""
    return [LimitationCheck(index=index, kind=kind, limitation=f"한정 {index}", disclosed=disclosed,
                            quote="원문 발췌" if disclosed else "")
            for index, (kind, disclosed) in enumerate(specs)]


def test_the_grade_ladder_is_derived_from_core_and_qualifier_disclosure():
    """등급은 모델이 고르지 않고 한정별 개시 여부에서 산출된다."""
    full = _checks(("core", True), ("qualifier", True))
    assert derive_judgment(full, has_evidence=True) == "실질적 동일"
    assert derive_judgment(full, has_evidence=True, terminology="identical") == "동일"
    # 용어 관계는 **전부 개시일 때만** 등급을 가른다.
    partial = _checks(("core", True), ("qualifier", False))
    assert derive_judgment(partial, has_evidence=True, terminology="identical") == "일부 차이"
    # core가 일부만 개시되면 qualifier 상태와 무관하게 '일부 유사'다.
    assert derive_judgment(_checks(("core", True), ("core", False), ("qualifier", True)),
                           has_evidence=True) == "일부 유사"


def test_a_different_purpose_caps_a_fully_disclosed_element_at_partial_similarity():
    """"문헌이 그 구성을 다른 목적으로 사용함"은 한정별 boolean으로 표현할 수 없어 따로 받는다."""
    full = _checks(("core", True), ("qualifier", True))
    assert derive_judgment(full, has_evidence=True, different_purpose=True) == "일부 유사"
    assert derive_judgment(full, has_evidence=True, terminology="identical",
                           different_purpose=True) == "일부 유사"


def test_evidence_presence_separates_a_difference_from_no_correspondence():
    """core 미개시일 때 '차이'와 '대응 없음'을 가르는 것은 제시된 원문의 유무뿐이다."""
    none_disclosed = _checks(("core", False), ("qualifier", False))
    assert derive_judgment(none_disclosed, has_evidence=True) == "차이"
    assert derive_judgment(none_disclosed, has_evidence=False) == "대응 없음"


def test_a_satisfied_alternative_group_does_not_drag_the_grade_down():
    """대안 묶음은 하나만 개시되면 충족이므로, 나머지 대안 때문에 등급이 내려가면 안 된다."""
    checks = [
        LimitationCheck(index=0, kind="core", limitation="가", disclosed=True,
                        quote="원문 발췌", alternative_group="묶음"),
        LimitationCheck(index=1, kind="core", limitation="나", disclosed=False,
                        alternative_group="묶음"),
    ]
    assert derive_judgment(checks, has_evidence=True) == "실질적 동일"


def test_the_model_cannot_set_the_grade_directly():
    """응답에 judgment를 넣어도 무시하고 한정별 개시 여부로 산출한다."""
    target = Claim(number=1, elements=[ClaimElement(
        label="A", text="구성 A",
        limitations=[{"text": "동작을 수행함", "kind": "core"},
                     {"text": "조건으로 한정함", "kind": "qualifier"}])])
    document = Document(id="1", filename="d.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", text="원문 발췌 문장입니다")])
    raw = [{"label": "A", "judgment": "동일", "directness": "direct",
            "quote": "원문 발췌 문장입니다", "chunk_id": "D1-P-0001",
            "limitation_checks": [
                {"index": 0, "disclosed": True, "quote": "원문 발췌 문장입니다", "chunk_id": "D1-P-0001"},
                {"index": 1, "disclosed": False, "quote": "", "chunk_id": ""}]}]

    match = _build_matches(raw, target, document)[0][0]

    # 모델은 "동일"이라고 했지만 qualifier가 미개시이므로 "일부 차이"가 산출된다.
    assert match.judgment == "일부 차이"
    assert match.terminology == "equivalent" and match.different_purpose is False


# --- 자기일관성 샘플링 ----------------------------------------------------------

def _sampling_claim() -> Claim:
    return Claim(number=1, elements=[ClaimElement(
        label="A", text="구성 A",
        limitations=[{"text": "동작을 수행함", "kind": "core"},
                     {"text": "조건으로 한정함", "kind": "qualifier"}])])


def _sample(core: bool, qualifier: bool, quote: str = "원문 발췌 문장입니다") -> dict:
    return {"label": "A", "directness": "direct", "quote": quote, "chunk_id": "D1-P-0001",
            "limitation_checks": [
                {"index": 0, "disclosed": core, "quote": quote if core else "",
                 "chunk_id": "D1-P-0001" if core else ""},
                {"index": 1, "disclosed": qualifier, "quote": quote if qualifier else "",
                 "chunk_id": "D1-P-0001" if qualifier else ""}]}


def _response(core: bool, qualifier: bool) -> list[dict]:
    """표본 하나. consensus는 표본들의 리스트를 받고, 표본 하나는 match 배열이다."""
    return [_sample(core, qualifier)]


def test_a_limitation_is_disclosed_when_the_majority_of_samples_say_so():
    """한정별 개시 여부에만 투표한다. 등급은 그 결과의 함수이므로 따로 투표하지 않는다."""
    claim = _sampling_claim()
    merged = compare.consensus([_response(True, True), _response(True, False),
                                _response(True, True)], claim)

    checks = {check["index"]: check["disclosed"] for check in merged[0]["limitation_checks"]}
    assert checks == {0: True, 1: True}      # core 3/3, qualifier 2/3


def test_a_tie_counts_as_undisclosed():
    """근거가 반반이면 개시로 인정하지 않는다. 인정하는 쪽이 청구항을 죽이는 방향이다."""
    claim = _sampling_claim()
    merged = compare.consensus([_response(True, True), _response(True, False)], claim)

    checks = {check["index"]: check["disclosed"] for check in merged[0]["limitation_checks"]}
    assert checks == {0: True, 1: False}     # qualifier 1/2는 동률이므로 미개시


def test_sampling_collapses_a_split_vote_into_one_grade():
    """실행마다 다른 답을 내던 셀이 하나의 등급으로 모인다."""
    claim = _sampling_claim()
    document = Document(id="1", filename="d.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", text="원문 발췌 문장입니다")])
    # 한 표본은 전부 개시(→실질적 동일), 두 표본은 core만 개시(→일부 차이).
    split = [_response(True, True), _response(True, False), _response(True, False)]

    merged = compare.consensus(split, claim)
    match = _build_matches(merged, claim, document)[0][0]

    assert match.judgment == "일부 차이"
    assert match.limitation_checks[0].disclosed is True
    assert match.limitation_checks[1].disclosed is False


def test_a_single_sample_passes_through_untouched():
    """샘플링을 끈 설정에서 이 경로는 아무것도 바꾸지 않아야 한다."""
    claim = _sampling_claim()
    only = _response(True, False)
    assert compare.consensus([only], claim) == only


def test_compare_document_asks_repeatedly_and_survives_a_failed_sample(monkeypatch):
    """표본 하나가 실패해도 나머지로 다수결을 낸다. 전부 실패했을 때만 미판정이다."""
    claim = _sampling_claim()
    document = Document(id="1", filename="d.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", text="원문 발췌 문장입니다")])
    calls: list[int] = []

    def flaky(prompt, expect="matches"):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("agy CLI가 JSON을 반환하지 않았습니다")
        return {"matches": [_sample(True, len(calls) == 1)]}

    monkeypatch.setattr(compare, "run_cli", flaky)
    matches, warnings = compare.compare_document(claim, document, samples=3)

    assert len(calls) == 3 and not warnings
    # 성공한 두 표본 중 qualifier는 1/2 동률이므로 미개시 → 일부 차이.
    assert matches[0].judgment == "일부 차이"


def test_every_sample_failing_is_reported_as_unjudged(monkeypatch):
    """미판정은 '대응 없음'과 다르다. 전부 실패하면 error를 세워 결론을 만들지 않게 한다."""
    claim = _sampling_claim()
    document = Document(id="1", filename="d.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", text="본문")])

    def always_fails(prompt, expect="matches"):
        raise RuntimeError("CLI 실행 실패")

    monkeypatch.setattr(compare, "run_cli", always_fails)
    matches, warnings = compare.compare_document(claim, document, samples=3)

    assert warnings and matches[0].error


# --- 표본 합의 계측 -----------------------------------------------------------
# 이 네 테스트는 계기 자체를 지킵니다. 종전 sample_agreement는 `vote is winner` 항등 비교라
# 완전 일치든 완전 불일치든 언제나 "1/3"을 냈고, _build_matches가 읽지도 않아 버려졌습니다.
# 값이 늘 상수여도 아무 테스트가 실패하지 않았던 것이 이 필드가 오래 죽어 있던 이유입니다.

def test_sample_agreement_separates_unanimous_cells_from_split_cells():
    claim = _sampling_claim()
    unanimous = compare.consensus([_response(True, True)] * 3, claim)[0]
    split = compare.consensus([_response(True, True), _response(False, False),
                               _response(True, False)], claim)[0]
    assert unanimous["sample_agreement"] == 1.0
    assert split["sample_agreement"] == 0.0          # 두 한정 모두 표본이 갈림
    assert unanimous["sample_count"] == split["sample_count"] == 3


def test_early_exit_flag_marks_only_cells_whose_first_two_samples_agree():
    claim = _sampling_claim()
    # 앞의 두 표본이 모든 한정에서 일치 → 세 번째는 다수결을 바꿀 수 없다.
    agreed = compare.consensus([_response(True, False), _response(True, False),
                                _response(False, True)], claim)[0]
    # 첫 한정에서 앞의 두 표본이 갈림 → 세 번째가 결과를 정한다.
    contested = compare.consensus([_response(True, True), _response(False, True),
                                   _response(True, True)], claim)[0]
    assert agreed["sample_early_exit"] is True
    assert contested["sample_early_exit"] is False
    assert agreed["limitation_checks"][0]["disclosed"] is True
    assert agreed["limitation_checks"][1]["disclosed"] is False


def test_sample_metrics_reach_the_element_match():
    """계측치가 ElementMatch까지 실려야 judgment.json에서 집계할 수 있다."""
    claim = _sampling_claim()
    document = Document(id="1", filename="d.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", text="원문 발췌 문장입니다")])
    merged = compare.consensus([_response(True, True)] * 3, claim)
    match = _build_matches(merged, claim, document)[0][0]
    assert match.sample_count == 3
    assert match.sample_agreement == 1.0
    assert match.sample_early_exit is True


def test_batch_and_single_judgments_do_not_share_a_cache_key():
    """같은 셀이라도 일괄 1표본과 단건 다표본은 다른 산출물이므로 키가 달라야 한다."""
    from app import cache
    claim = _sampling_claim()
    document = Document(id="1", filename="d.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0001", text="원문 발췌 문장입니다")])
    single = cache.cache_key(claim, document, "", 30000, [], mode="single", samples=3)
    batch = cache.cache_key(claim, document, "", 30000, [], mode="batch", samples=1)
    same_numbers = cache.cache_key(claim, document, "", 30000, [], mode="batch", samples=3)
    assert single != batch                 # 종전에는 이 둘이 같은 파일을 놓고 서로를 덮어썼다
    assert single != same_numbers          # 예산·표본이 같아도 프롬프트가 다르면 다른 키
