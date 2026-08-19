"""판정 **경위**를 보고서가 버리지 않는다.

실측(두상 모델 사건) 구성 E에서 일어난 일이다. 한정 하나가 초기 비교에서 개시로 나왔다가
의미검증이 근거를 대조해 기각했고, 채택 조합 전체 위에서 다시 심사해 또 기각했다. 표본 3회는
한정 3개 중 1개에서만 만장일치였다. 그런데 보고서에 남은 것은 다음 한 줄뿐이었다.

    (E) 구성에 대응되는 인용발명이 확인되지 않음 — 추가 검색 필요

이 한 줄만 읽은 사람은 도구가 그 구성을 **검토하지 않았다**고 읽는다. 실제로는 세 번 검토했고
두 번은 근거를 들어 기각한 것이다. 판단의 강도가 사라지면 읽는 사람은 그 자리를 스스로 메우고,
같은 실행 기록을 두고 정반대 결론이 나온다 — 실제로 그렇게 됐다.

경위는 이미 LimitationCheck.semantic_status/semantic_note에 구조화되어 있었다. 보고서가
accepted만 읽고 rejected·*_in_combination을 읽지 않았을 뿐이다. 그래서 이 파일이 고정하는 것은
"경위를 새로 만든다"가 아니라 **이미 가진 것을 버리지 않는다**이다.
"""
from app.chain import build_chain, matrix_for
from app.models import (AnalysisResult, Chunk, Claim, ClaimElement, Document, ElementMatch,
                        LimitationCheck, SemanticEvent, missing_limitations)
from app.report import build_claim_report, build_mappings, to_markdown


def _check(index: int, limitation: str, *, disclosed: bool = False, status: str = "not_run",
           note: str = "", supplied: list[str] | None = None,
           events: list[SemanticEvent] | None = None) -> LimitationCheck:
    return LimitationCheck(index=index, limitation=limitation, kind="core", disclosed=disclosed,
                           semantic_status=status, semantic_note=note,
                           semantic_events=events or [],
                           combination_documents=supplied or [],
                           quote="원문 발췌 문장입니다", chunk_id="D1-B-p005-01", verify="verified")


def _match(checks: list[LimitationCheck], *, document_id: str = "1", judgment: str = "차이",
           unanimous: int = 0, requirements: int = 0, samples: int = 0) -> ElementMatch:
    return ElementMatch(claim_number=1, label="E", document_id=document_id, judgment=judgment,
                        directness="inferred", quote="원문 발췌 문장입니다",
                        chunk_id="D1-B-p005-01", verify="verified", limitation_checks=checks,
                        missing_limitations=missing_limitations(checks),
                        sample_count=samples, sample_unanimous=unanimous,
                        sample_requirements=requirements,
                        sample_agreement=round(unanimous / requirements, 4) if requirements else 0.0)


def _rendered(*matches: ElementMatch) -> str:
    claim = Claim(number=1, elements=[ClaimElement(
        label="E", importance=4,
        text="상기 융합 3차원 점군에서 복수의 3차원 기준점들을 포함하는 가시 두상 영역을 추출하는 단계")])
    documents = {
        match.document_id: Document(
            id=match.document_id, filename=f"D{match.document_id}.pdf", type="paper",
            chunks=[Chunk(document_id=match.document_id, chunk_id=match.chunk_id, page=5,
                          text="원문 발췌 문장입니다")])
        for match in matches}
    matrix = matrix_for(list(matches))
    chain = build_chain(claim, matrix, {}, [claim])
    mappings = build_mappings(list(documents.values()), [chain])
    report = build_claim_report(claim, chain, matrix, documents, mappings)
    return to_markdown(AnalysisResult(job_id="job", claim_mapping=mappings, reports=[report]))


# --- 실측 사건 재현 -----------------------------------------------------------

_REJECTED_ALONE = ("대상 및 동작 축 결손. 제시된 근거(D1-B-p005-01)는 전처리 단계에서 3D "
                   "랜드마크가 있는 최적 프레임 4개를 선별·출력하는 내용에 불과하며, 융합 "
                   "점군으로부터 가시 두상 영역을 추출하는 대상 및 동작을 뒷받침하지 못함")
_REJECTED_COMBINED = ("인용발명 2는 ToF와 스테레오 깊이의 신뢰도·노이즈 기반 융합에 관한 "
                      "것일 뿐, 양 문헌의 결합 문맥 어디에서도 융합 3차원 점군으로부터 가시 "
                      "두상 영역을 추출하는 처리 동작이 확인되지 않음")


def _the_head_model_case() -> ElementMatch:
    """구성 E × 인용발명 1을 실행 기록 그대로 옮긴다.

    의미검증은 **disclosed 한정만** 대상으로 하므로(entailment.validate_entailment), 초기
    비교의 다수결에서 이미 미개시로 떨어진 한정 0·2는 검증을 받은 적이 없다 — not_run이다.
    개시로 나온 한정 1 하나만 의미검증에 올라가 기각됐고, 채택 조합이 정해진 뒤 결합 근거
    위에서 한 번 더 심사받아 또 기각됐다. **한 한정에 이벤트가 둘**인 형태가 이 사건의 핵심이다.
    """
    return _match([
        _check(0, "상기 융합 3차원 점군에서 가시 두상 영역을 추출함"),
        _check(2, "상기 기준점들의 용도를 후속 스마트폰 영상과의 2차원-3차원 대응으로 한정함"),
        _check(1, "추출 대상을 복수의 3차원 기준점들을 포함하는 영역으로 한정함",
               status="rejected_in_combination", note=_REJECTED_COMBINED,
               events=[SemanticEvent(stage="의미검증", outcome="기각", note=_REJECTED_ALONE),
                       SemanticEvent(stage="결합검증", outcome="기각", note=_REJECTED_COMBINED)]),
    ], unanimous=1, requirements=3, samples=3)


def test_a_rejected_limitation_says_which_stage_rejected_it_and_why():
    markdown = _rendered(_the_head_model_case())

    assert "판정 경위:" in markdown
    assert "결합검증 기각" in markdown
    # 기각 사유가 함께 나가야 그 판단을 다툴 수 있다. 결과만 적으면 다툴 대상이 없다.
    assert "프레임 4개" in markdown


def test_both_events_of_one_limitation_survive():
    """한정 하나가 두 단계에서 기각되면 **두 줄이** 남아야 한다.

    semantic_status는 단일 필드라 결합검증 결과가 의미검증 결과를 덮는다. 종단 상태만 읽으면
    "결합 위에서 한 번 봤다"로 보이는데, 실제로는 문헌 단독으로 보고 결합 위에서 또 본 것이다.
    뒤엣것이 훨씬 강한 판정이므로 그 차이가 보고서에서 사라지면 안 된다.
    """
    markdown = _rendered(_the_head_model_case())
    trail = markdown[markdown.index("판정 경위:"):]

    assert trail.count("의미검증 기각") == 1
    assert trail.count("결합검증 기각") == 1
    # 두 줄 모두 같은 한정을 가리키고, 사유는 단계마다 다르다.
    assert trail.count("한정 #1") == 2
    assert "전처리 단계" in trail and "결합 문맥" in trail


def test_limitations_that_never_reached_verification_add_no_events():
    """의미검증은 disclosed 한정만 대상으로 한다. 받은 적 없는 판정을 지어내지 않는다."""
    markdown = _rendered(_the_head_model_case())
    trail = markdown[markdown.index("판정 경위:"):]

    assert "한정 #0" not in trail
    assert "한정 #2" not in trail


def test_a_split_cell_reports_the_numerator_and_denominator_not_the_ratio():
    """0.33은 '표본이 전부 갈렸다'로도 '한정 3개 중 1개만 만장일치'로도 읽힌다.

    실제로 그 오독이 결론까지 갔다. 분자·분모를 적으면 한 가지로만 읽힌다.
    """
    markdown = _rendered(_the_head_model_case())

    assert "한정 3개 중 1개만 표본 3회 만장일치" in markdown
    assert "0.33" not in markdown


def test_the_trail_survives_when_no_document_was_adopted():
    """경위가 가장 필요한 것은 미대응 구성이다.

    채택 문헌이 없으면 근거 목록도 비므로, 채택된 셀만 훑으면 그 구성에서 정확히 아무것도
    남지 않는다 — 보고서가 "확인되지 않았습니다" 한 줄로 끝나는 상태가 그것이다.
    """
    markdown = _rendered(_the_head_model_case())

    assert "확인되지 않음" in markdown          # 판정 자체는 그대로 남고
    assert "판정 경위:" in markdown             # 그 판정에 이른 경위도 함께 남는다


def test_a_clean_cell_adds_no_trail_noise():
    """만장일치로 통과한 셀까지 한 줄씩 나가면 갈린 셀이 그 소음에 묻힌다."""
    clean = _match([_check(0, "가시 두상 영역을 추출함", disclosed=True, status="accepted")],
                   judgment="실질적 동일", unanimous=1, requirements=1, samples=3)
    markdown = _rendered(clean)

    assert "판정 경위:" not in markdown


def test_the_invariant_fires_when_a_moved_judgment_is_dropped_from_the_report():
    """경위를 다시 버리는 회귀는 조립 직후에 기계적으로 잡혀야 한다.

    _evidence_is_never_erased가 **인정된** 근거를 지키는 것과 같은 이유다. 인정만 지키고
    기각을 버리면 보고서는 한쪽으로만 검증 가능해진다.
    """
    from app.report import build_claim_report, pipeline_invariants

    match = _the_head_model_case()
    claim = Claim(number=1, elements=[ClaimElement(label="E", importance=4, text="구성 E 원문")])
    document = Document(id=match.document_id, filename="D1.pdf", type="paper",
                        chunks=[Chunk(document_id=match.document_id, chunk_id=match.chunk_id,
                                      page=5, text="원문 발췌 문장입니다")])
    matrix = matrix_for([match])
    chain = build_chain(claim, matrix, {}, [claim])
    mappings = build_mappings([document], [chain])
    report = build_claim_report(claim, chain, matrix, {match.document_id: document}, mappings)

    assert not pipeline_invariants([report], {1: matrix})       # 정상 조립은 조용하다

    for item in report.claims:                                  # 경위를 버리면
        item.trail = []
    notes = pipeline_invariants([report], {1: matrix})
    assert notes and "경위에서 빠졌습니다" in notes[0]           # 즉시 발화한다


# --- 옛 캐시 항목 ---------------------------------------------------------------
# 캐시 세대는 **프롬프트 문면**에서 나온다(cache.compare_generation). 모델에 필드를 더해도
# 키가 갈리지 않으므로 옛 항목이 그대로 로드되고 새 필드만 0으로 남는다. 그러면 캐시 히트한
# 셀에서만 "표본이 갈렸다"가 조용히 사라져, 같은 보고서가 셀마다 다른 말을 한다.

def _legacy_cache_entry() -> list[dict]:
    """새 필드가 없던 시절의 캐시 JSON. 비율은 있고 분자·분모가 없다."""
    match = _the_head_model_case()
    raw = match.model_dump()
    raw.pop("sample_unanimous")
    raw.pop("sample_requirements")
    for check in raw["limitation_checks"]:
        check.pop("semantic_events")
    return [raw]


def test_a_legacy_cache_entry_recovers_its_sample_counts(tmp_path, monkeypatch):
    import json
    from app import cache

    (tmp_path / "legacy.json").write_text(
        json.dumps(_legacy_cache_entry(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    loaded = cache.load("legacy")

    assert loaded is not None
    # 비율 0.3333과 한정 3개에서 "3개 중 1개"가 손실 없이 복원된다.
    assert (loaded[0].sample_requirements, loaded[0].sample_unanimous) == (3, 1)


def test_a_legacy_cache_entry_still_reports_that_the_samples_split(tmp_path, monkeypatch):
    """복원의 목적은 숫자가 아니라 **보고서가 같은 말을 하는 것**이다."""
    import json
    from app import cache

    (tmp_path / "legacy.json").write_text(
        json.dumps(_legacy_cache_entry(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    markdown = _rendered(cache.load("legacy")[0])

    assert "한정 3개 중 1개만 표본 3회 만장일치" in markdown


def test_a_legacy_entry_without_an_event_log_still_shows_its_last_judgment():
    """옛 기록은 경로를 잃었지만 종단 상태는 남아 있다. 아무것도 안 적는 것보다 낫다."""
    legacy = ElementMatch.model_validate(_legacy_cache_entry()[0])
    markdown = _rendered(legacy)

    assert "결합검증 기각" in markdown        # 종단 상태에서 되짚은 한 줄
    assert "의미검증 기각" not in markdown    # 없는 기록을 지어내지는 않는다


def test_a_cell_that_never_voted_is_left_alone():
    """표본 경로를 타지 않은 셀에 없는 분모를 만들어 주면 안 된다."""
    single = _match([_check(0, "가시 두상 영역을 추출함", disclosed=True, status="accepted")],
                    judgment="실질적 동일", samples=0)

    assert (single.sample_requirements, single.sample_unanimous) == (0, 0)
    assert "판정 경위:" not in _rendered(single)


def test_a_fresh_cell_keeps_the_counts_the_vote_actually_produced():
    """복원 경로가 새 기록을 덮어쓰면 계측치가 조용히 틀어진다."""
    fresh = _match([_check(0, "a"), _check(1, "b"), _check(2, "c")],
                   unanimous=2, requirements=3, samples=3)
    revalidated = ElementMatch.model_validate(fresh.model_dump())

    assert (revalidated.sample_requirements, revalidated.sample_unanimous) == (3, 2)


def test_re_applying_the_same_judgment_does_not_invent_a_second_rejection():
    """쌓기만 하는 목록은 재적용에 취약하다. 중복이 들어가면 없는 심사를 지어낸다."""
    check = _check(0, "가시 두상 영역을 추출함")
    rejection = SemanticEvent(stage="의미검증", outcome="기각", note=_REJECTED_ALONE)
    check.record(rejection)
    check.record(rejection)

    assert len(check.semantic_events) == 1
    # 단계나 사유가 다르면 다른 판정이므로 그대로 쌓인다.
    check.record(SemanticEvent(stage="결합검증", outcome="기각", note=_REJECTED_COMBINED))
    assert len(check.semantic_events) == 2


def test_a_combination_acceptance_names_the_document_that_supplied_the_axis():
    """결합으로 인정했으면 어느 문헌이 빠진 축을 댔는지 적는다. 지어내지 않고 기록에서 읽는다."""
    accepted = _match([_check(0, "가시 두상 영역을 추출함", disclosed=True,
                              status="accepted_in_combination", supplied=["2"],
                              note="융합 점군은 인용발명 2가 개시함")],
                      judgment="실질적 동일")
    other = _match([_check(0, "융합 3차원 점군을 생성함", disclosed=True, status="accepted")],
                   document_id="2", judgment="실질적 동일")
    markdown = _rendered(accepted, other)

    assert "결합검증 인정" in markdown
    assert "빠진 축은 인용발명" in markdown


# --- 표본별 원시 투표이 보고서까지 닿는가 -------------------------------------------

def _voted(index: int, limitation: str, *verdicts: str) -> LimitationCheck:
    from app.models import SampleTally, SampleVote
    check = _check(index, limitation)
    check.sample_tally = SampleTally(
        total=len(verdicts),
        votes=[SampleVote(sample=sample, verdict=verdict)
               for sample, verdict in enumerate(verdicts)])
    return check


def test_the_report_says_how_the_votes_actually_split():
    """"2대 1로 갈린 미개시"와 "3대 0으로 일치한 미개시"는 다음 조치가 다르다."""
    markdown = _rendered(_match(
        [_voted(0, "가시 두상 영역을 추출함", "disclosed", "missing", "missing"),
         _voted(1, "기준점들을 포함하는 영역으로 한정함", "missing", "missing", "missing")],
        unanimous=1, requirements=2, samples=3))

    assert ("한정 #0 「가시 두상 영역을 추출함」 판정 불일치: 개시 1표 · 미개시 2표 (표본 3회)"
            in markdown)
    assert "한정 #1" not in markdown          # 만장일치는 위 한 줄이 이미 말한다


def test_the_report_separates_no_answer_from_a_vote_for_undisclosed():
    markdown = _rendered(_match(
        [_voted(0, "가시 두상 영역을 추출함", "disclosed", "absent", "missing")],
        unanimous=0, requirements=1, samples=3))

    assert "판정 불일치 · 응답 결손: 개시 1표 · 미개시 1표 · 무응답 1표 (표본 3회)" in markdown


def test_a_legacy_cell_adds_no_vote_lines():
    markdown = _rendered(_the_head_model_case())

    assert "표본 3회 만장일치" in markdown      # 구성 단위 지표는 그대로 나오고
    assert "개시 0표" not in markdown           # 없는 투표는 지어내지 않는다


def test_the_invariant_catches_a_tally_that_lost_a_vote():
    """"3표 중 2표"라고 적힌 줄이 실제로 두 표만 받은 것이면 없는 표본을 셈에 넣게 된다."""
    from app.models import SampleTally, SampleVote
    from app.report import build_claim_report, pipeline_invariants

    match = _match([_voted(0, "가시 두상 영역을 추출함", "disclosed", "missing", "missing")])
    claim = Claim(number=1, elements=[ClaimElement(label="E", importance=4, text="구성 E")])
    document = Document(id="1", filename="D1.pdf", type="paper",
                        chunks=[Chunk(document_id="1", chunk_id=match.chunk_id, page=5,
                                      text="원문 발췌 문장입니다")])
    matrix = matrix_for([match])
    chain = build_chain(claim, matrix, {}, [claim])
    report = build_claim_report(claim, chain, matrix, {"1": document},
                               build_mappings([document], [chain]))

    assert not pipeline_invariants([report], {1: matrix})
    match.limitation_checks[0].sample_tally = SampleTally(
        total=3, votes=[SampleVote(sample=0, verdict="disclosed")])
    notes = pipeline_invariants([report], {1: matrix})
    assert notes and "[불변식 S1]" in notes[0]


def test_a_limitation_most_samples_could_not_answer_is_surfaced_too():
    """개시·미개시의 대립만 보면, 셋 중 하나만 답한 한정이 조용히 지나간다.

    갈린 것은 아니지만 만장일치도 아니고, 오히려 그쪽이 더 확인이 필요한 자리다.
    """
    markdown = _rendered(_match(
        [_voted(0, "기준점들을 포함하는 영역으로 한정함", "disclosed", "absent", "invalid")],
        unanimous=0, requirements=1, samples=3))

    assert "응답 결손: 개시 1표 · 미개시 0표 · 무응답 1표 · 판독불가 1표 (표본 3회)" in markdown


def test_the_headline_does_not_claim_the_samples_disagreed():
    """만장일치가 아닌 이유는 판단이 나뉜 것일 수도, 답을 받지 못한 것일 수도 있다."""
    markdown = _rendered(_match(
        [_voted(0, "기준점들을 포함하는 영역으로 한정함", "disclosed", "absent", "invalid")],
        unanimous=0, requirements=1, samples=3))

    assert "초기 비교 결과가 불안정합니다" in markdown
    assert "표본이 갈렸습니다" not in markdown
