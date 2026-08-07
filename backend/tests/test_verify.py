from app.models import Chunk, Document, ElementMatch, LimitationCheck
from app.verify import is_verbatim, verify_matches

CORPUS = "메모리 컨트롤러는 데이터 쓰기 요청을 큐에 저장한 후 순차적으로 처리한다."
DOCUMENT = Document(id="1", filename="a.pdf", type="patent",
                    chunks=[Chunk(document_id="1", chunk_id="D1-P-0021", page=2, paragraph="0021", text=CORPUS)])


def match(**overrides) -> ElementMatch:
    base = dict(claim_number=1, label="A", document_id="1", judgment="동일", directness="direct",
                quote="데이터 쓰기 요청을 큐에 저장한 후 순차적으로 처리한다", chunk_id="D1-P-0021")
    return ElementMatch(**{**base, **overrides})


def test_verbatim_requires_every_segment_to_exist():
    assert is_verbatim("데이터 쓰기 요청을 큐에 저장한 후", CORPUS)
    assert is_verbatim("메모리 컨트롤러는 데이터 … 순차적으로 처리한다", CORPUS)
    assert not is_verbatim("데이터 쓰기 요청을 스택에 저장한 후", CORPUS)


def test_a_word_split_by_the_extractor_still_verifies():
    """공보 PDF는 줄바꿈 자리에서 낱말을 쪼갠다. 모델이 붙여 쓴 인용을 가짜로 보면 안 된다."""
    corpus = "a commu nication server in communication with the server , and a plurality of devices"
    assert is_verbatim("a communication server in communication with the server", corpus)
    assert not is_verbatim("a communication server directly attached to the storage", corpus)


def test_a_sentence_assembled_from_scattered_terms_is_not_a_quotation():
    """청구항 문언과 문헌 낱말을 섞어 만든 문장은 직접 인용으로 인정하지 않는다."""
    fabricated = match(quote="메모리 컨트롤러가 요청을 병렬로 분배하여 지연 없이 처리하는 구성을 개시한다")
    notes = verify_matches([fabricated], {"1": DOCUMENT})
    assert fabricated.verify == "not_found"
    # 근거가 아예 확인되지 않으므로 directness가 absent가 되고, 상한도 한 단계 더 내려간다.
    assert fabricated.judgment == "차이"
    assert fabricated.downgraded_from == "동일"
    assert fabricated.directness == "absent"
    assert notes


def test_verified_quote_keeps_its_judgment():
    verified = match()
    assert verify_matches([verified], {"1": DOCUMENT}) == []
    assert verified.verify == "verified"
    assert verified.judgment == "동일" and verified.directness == "direct"


def test_a_verified_quote_with_an_unknown_chunk_id_is_relocated_without_downgrade():
    """CN 공보처럼 한 청크가 여러 문단을 담아도, 실제 원문이면 위치만 복구한다."""
    wrong = match(chunk_id="D1-P-9999")
    notes = verify_matches([wrong], {"1": DOCUMENT})
    assert wrong.chunk_id == "D1-P-0021"
    assert wrong.directness == "direct"
    assert wrong.judgment == "동일"
    assert wrong.verify == "verified"
    assert "자동 복구" in wrong.verify_note and notes


def test_an_unknown_chunk_id_with_a_fabricated_quote_is_still_downgraded():
    wrong = match(chunk_id="D1-P-9999", quote="문헌에는 없는 병렬 분배 장치의 구체적인 동작 문장입니다")
    notes = verify_matches([wrong], {"1": DOCUMENT})
    assert wrong.chunk_id == ""
    assert wrong.directness == "inferred"
    assert wrong.judgment == "일부 차이"
    assert wrong.verify == "not_found" and notes


def test_an_existing_but_wrong_chunk_location_is_also_repaired():
    document = DOCUMENT.model_copy(deep=True)
    document.chunks.append(Chunk(document_id="1", chunk_id="D1-P-0099", page=9,
                                 paragraph="0099", text="전혀 다른 실시예를 설명하는 충분히 긴 문장입니다."))
    wrong = match(chunk_id="D1-P-0099")
    notes = verify_matches([wrong], {"1": document})
    assert wrong.chunk_id == "D1-P-0021"
    assert wrong.judgment == "동일" and wrong.directness == "direct"
    assert "자동 복구" in wrong.verify_note and notes


def test_paraphrase_is_recovered_from_the_cited_chunk_but_not_promoted():
    """복구한 문장은 구성 일부만 뒷받침할 수 있으므로 직접 개시로 올리지 않는다."""
    paraphrased = match(quote="메모리 컨트롤러는 쓰기 요청을 큐에 넣고 차례대로 처리하는 구성이다")
    verify_matches([paraphrased], {"1": DOCUMENT})
    assert paraphrased.verify == "verified"
    assert paraphrased.quote == CORPUS
    assert paraphrased.directness == "inferred"


def test_direct_judgment_without_a_quote_is_downgraded():
    empty = match(quote="", chunk_id="")
    verify_matches([empty], {"1": DOCUMENT})
    assert empty.verify == "empty"
    assert empty.directness == "inferred"
    assert empty.judgment == "일부 차이"


def test_unverified_atomic_limitation_evidence_blocks_a_full_match():
    checked = match(limitation_checks=[LimitationCheck(
        index=0, limitation="우선순위에 따라 처리함", disclosed=True,
        quote="문헌에 존재하지 않는 우선순위 처리 문장입니다", chunk_id="D1-P-0021")])

    notes = verify_matches([checked], {"1": DOCUMENT})

    assert checked.limitation_checks[0].disclosed is False
    assert checked.limitation_checks[0].verify == "not_found"
    assert checked.missing_limitations == ["우선순위에 따라 처리함"]
    assert checked.judgment == "일부 차이"
    assert notes


def test_a_paraphrased_atomic_quote_is_recovered_from_its_cited_chunk():
    checked = match(limitation_checks=[LimitationCheck(
        index=0, limitation="쓰기 요청을 큐에 저장하여 순차 처리함", disclosed=True,
        quote="메모리 컨트롤러는 쓰기 요청을 큐에 넣고 차례대로 처리한다", chunk_id="D1-P-0021")])

    notes = verify_matches([checked], {"1": DOCUMENT})

    assert checked.limitation_checks[0].disclosed is True
    assert checked.limitation_checks[0].verify == "verified"
    assert checked.limitation_checks[0].quote == CORPUS
    assert checked.directness == "inferred"
    assert notes == []


def test_verification_does_not_resurrect_a_satisfied_alternative():
    """검증 단계가 누락 목록을 다시 채울 때도 대안 묶음 규칙을 지켜야 한다.

    비교 단계와 검증 단계가 각자 목록을 만들면 규칙이 갈라진다. 실제로 검증 단계가
    미개시 대안을 그대로 다시 넣어, 비교 단계에서 걸러 낸 항목이 보고서의 차이점으로
    되살아났다.
    """
    quote = "The one-way hash value contains a time-to-live value."
    document = Document(id="1", filename="prior.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0612", page=24, paragraph="0612", text=quote)])
    match = ElementMatch(
        claim_number=4, label="A", document_id="1", judgment="동일", directness="direct",
        quote=quote, chunk_id="D1-P-0612", limitation_checks=[
            LimitationCheck(index=0, limitation="토큰이 시간 제한을 가짐", alternative_group="토큰속성",
                            disclosed=True, quote=quote, chunk_id="D1-P-0612"),
            LimitationCheck(index=1, limitation="토큰이 회수 기능을 가짐", alternative_group="토큰속성"),
        ])

    verify_matches([match], {"1": document})

    assert match.missing_limitations == []
    assert match.judgment == "동일"


def test_a_limitation_quote_split_across_chunks_stays_disclosed():
    """청크 경계에 걸친 근거를 미개시로 뒤집으면, 강등의 근거가 문헌 내용이 아니라 자르는 위치가 된다.

    공보 PDF는 단락과 무관한 자리에서 잘리므로 한 문장이 두 청크에 나뉘는 일이 흔하다.
    문헌 전체에서는 그대로 확인되는 문장인데도 단일 청크에 없다는 이유로 disclosed를
    뒤집으면, 실제로 개시된 한정이 '누락 한정'이 되고 구성 판정까지 강등된다. 대표 발췌는
    이미 같은 상황에서 판정을 유지하므로 두 경로가 서로 다른 답을 내고 있었다.
    """
    head = "In response to determining that there is unused programmer ad inventory ,"
    tail = "the ad router server may send an ad request message to the MSO ADS ."
    document = Document(id="1", filename="prior.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0064", page=7, paragraph="0064", text=head),
        Chunk(document_id="1", chunk_id="D1-P-0065", page=7, paragraph="0065", text=tail)])
    straddling = ElementMatch(
        claim_number=1, label="C", document_id="1", judgment="실질적 동일", directness="direct",
        quote=head, chunk_id="D1-P-0064", limitation_checks=[
            LimitationCheck(index=0, limitation="판매 결과에 따라 정보를 전송함", disclosed=True,
                            quote=f"{head} {tail}", chunk_id="D1-P-0064")])

    notes = verify_matches([straddling], {"1": document})

    assert straddling.limitation_checks[0].disclosed is True
    assert straddling.limitation_checks[0].verify == "verified"
    assert straddling.missing_limitations == []
    assert straddling.judgment == "실질적 동일" and straddling.downgraded_from == ""
    # 위치는 문장이 시작된 청크로 되돌린다. 비워 두면 근거는 확인되었는데 어디를 보라고
    # 적을 수 없는 보고서가 된다.
    assert straddling.limitation_checks[0].chunk_id == "D1-P-0064"
    assert notes == []


def test_a_fabricated_quote_is_still_rejected_when_it_spans_nothing():
    """청크 경계 구제가 '원문에 없는 문장'까지 통과시켜서는 안 된다."""
    document = Document(id="1", filename="prior.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0064", page=7, paragraph="0064",
              text="The ad router server may hold any ad request to the MSO ADS .")])
    fabricated = ElementMatch(
        claim_number=1, label="C", document_id="1", judgment="실질적 동일", directness="direct",
        quote="The ad router server may hold any ad request to the MSO ADS .",
        chunk_id="D1-P-0064", limitation_checks=[
            LimitationCheck(index=0, limitation="판매 결과에 따라 정보를 전송함", disclosed=True,
                            quote="The server predicts future demand and reserves the slot .",
                            chunk_id="D1-P-0064")])

    verify_matches([fabricated], {"1": document})

    assert fabricated.limitation_checks[0].disclosed is False
    assert fabricated.missing_limitations == ["판매 결과에 따라 정보를 전송함"]


def test_verification_restores_a_limitation_whose_quote_fails():
    """반대로 개시로 적힌 근거가 원문 대조에 실패하면 그 한정은 누락으로 되돌아간다."""
    real = "The one-way hash value contains a time-to-live value."
    document = Document(id="1", filename="prior.pdf", chunks=[
        Chunk(document_id="1", chunk_id="D1-P-0612", page=24, paragraph="0612", text=real)])
    match = ElementMatch(
        claim_number=4, label="A", document_id="1", judgment="동일", directness="direct",
        quote=real, chunk_id="D1-P-0612", limitation_checks=[
            LimitationCheck(index=0, limitation="토큰이 시간 제한을 가짐", alternative_group="토큰속성",
                            disclosed=True, quote="The token is revoked on playback end.",
                            chunk_id="D1-P-0612"),
            LimitationCheck(index=1, limitation="토큰이 회수 기능을 가짐", alternative_group="토큰속성"),
        ])

    verify_matches([match], {"1": document})

    assert match.missing_limitations == ["토큰이 시간 제한을 가짐", "토큰이 회수 기능을 가짐"]
