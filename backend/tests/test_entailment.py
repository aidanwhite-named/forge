"""이번 GNSS/SfM 사례에서 드러난 의미 과대판정과 단계적 근거 묶음의 회귀 테스트."""
from app import cache, entailment
from app.entailment import validate_entailment
from app.models import (Chunk, Claim, ClaimElement, Document, ElementMatch, EvidenceSpan,
                        Limitation, LimitationCheck)
from app.verify import verify_matches


def _document(document_id: str, filename: str, quotes: list[str]) -> Document:
    return Document(id=document_id, filename=filename, type="paper", chunks=[
        Chunk(document_id=document_id, chunk_id=f"D{document_id}-B-p{index + 1:03d}-01",
              page=index + 1, text=quote)
        for index, quote in enumerate(quotes)])


def test_remote_sensing_outcome_quotes_do_not_entail_local_model_generation_or_merging(monkeypatch):
    """문장이 PDF에 존재해도 청구된 수단·관계를 말하지 않으면 B·D의 개시 근거가 아니다."""
    b_quote = ("Second, a GNSS-constrained incremental structureless BA is carried out in the "
               "selected sub-scene to obtain reliable initial camera orientations and absolute scale.")
    d_quote = ("These results indicate that the proposed method can successfully recover complete camera "
               "trajectories and continuous sparse scene structures for long-corridor UAV datasets.")
    document = _document("1", "remotesensing-18-02321-v2.pdf", [b_quote, d_quote])
    matches = [
        ElementMatch(
            claim_number=1, label="B", document_id="1", judgment="실질적 동일", directness="direct",
            quote=b_quote, chunk_id="D1-B-p001-01", limitation_checks=[LimitationCheck(
                index=0, kind="core", limitation=("클러스터 영상의 특징점 대응관계에 기초하여 "
                                                  "클러스터별 로컬 3D 모델을 생성함"),
                disclosed=True, quote=b_quote, chunk_id="D1-B-p001-01")]),
        ElementMatch(
            claim_number=1, label="D", document_id="1", judgment="실질적 동일", directness="direct",
            quote=d_quote, chunk_id="D1-B-p002-01", limitation_checks=[LimitationCheck(
                index=0, kind="core", limitation=("국소 3D 모델 사이의 공통 대응관계에 기초하여 "
                                                  "복수의 국소 3D 모델을 전역 모델로 병합함"),
                disclosed=True, quote=d_quote, chunk_id="D1-B-p002-01")]),
    ]
    verify_matches(matches, {"1": document})

    def fake(prompt, expect="entailments"):
        assert expect == "entailments"
        assert b_quote in prompt and d_quote in prompt
        return {"entailments": [
            {"item_id": "1:B:0", "supported": False, "relation": "unsupported",
             "directness": "inferred", "reason": "카메라 자세와 스케일만 기재하고 특징점 기반 로컬 모델 생성을 기재하지 않는다."},
            {"item_id": "1:D:0", "supported": False, "relation": "unsupported",
             "directness": "inferred", "reason": "연속 장면 회복이라는 결과만 있고 공통 대응관계에 의한 모델 병합이 없다."},
        ]}

    monkeypatch.setattr(entailment, "run_cli", fake)
    notes = validate_entailment(matches, {"1": document})

    assert [match.judgment for match in matches] == ["차이", "차이"]
    # direct는 유지될 수 없지만 absent로도 내리지 않습니다. absent는 "근거 원문이 없음"인데
    # 이 셀에는 원문 대조를 통과한 발췌가 그대로 남아 있고, absent로 내리면 그 문헌이 보조
    # 인용발명으로 기여할 자격까지 잃습니다(coverage.ineligible_reason). compare._build_matches가
    # 같은 조건에서 쓰는 강등과 맞춥니다.
    assert [match.directness for match in matches] == ["inferred", "inferred"]
    assert all(not match.limitation_checks[0].disclosed for match in matches)
    assert all(match.limitation_checks[0].semantic_status == "rejected" for match in matches)
    assert len(notes) == 2


def test_sensors_staged_evidence_bundle_can_entail_the_grouping_limitation(monkeypatch):
    """GNSS 필터·상대정합 가중치·그래프 분할이 같은 흐름이면 여러 문단을 함께 심사한다."""
    gnss = ("After obtaining each image and all GNSS neighbors that meet the distance threshold, "
            "we generated a pre-matched camera graph structure based on the GNSS neighborhood.")
    weight = ("The weight of the edge represents the number of matching feature points between the two images.")
    grouping = ("We use a normalized-cut algorithm to divide the camera graph into multiple subgraphs.")
    document = _document("3", "sensors-21-03939.pdf", [gnss, weight, grouping])
    match = ElementMatch(
        claim_number=1, label="A", document_id="3", judgment="실질적 동일", directness="direct",
        quote=gnss, chunk_id="D3-B-p001-01", limitation_checks=[LimitationCheck(
            index=0, kind="core",
            limitation=("영상의 GNSS 정보와 영상 간 상대정합 정보를 이용한 기준에 기초하여 "
                        "복수의 영상을 클러스터로 그룹화함"),
            disclosed=True, quote=gnss, chunk_id="D3-B-p001-01",
            evidence=[EvidenceSpan(chunk_id="D3-B-p002-01", quote=weight),
                      EvidenceSpan(chunk_id="D3-B-p003-01", quote=grouping)])])
    verify_matches([match], {"3": document})

    def fake(prompt, expect="entailments"):
        assert gnss in prompt and weight in prompt and grouping in prompt
        return {"entailments": [{
            "item_id": "1:A:0", "supported": True,
            "relation": "functional_equivalent", "directness": "direct",
            "reason": "GNSS가 그래프 후보를 정하고 상대정합 수가 가중치를 정하며 그 그래프를 분할한다.",
        }]}

    monkeypatch.setattr(entailment, "run_cli", fake)
    cache_keys: set[str] = set()
    validate_entailment([match], {"3": document}, cache_keys)

    check = match.limitation_checks[0]
    assert check.disclosed is True
    assert check.semantic_status == "accepted"
    assert check.semantic_relation == "functional_equivalent"
    assert match.judgment == "실질적 동일" and match.directness == "direct"
    semantic_keys = [key for key in cache_keys if key.startswith(cache.ENTAILMENT_KEY_PREFIX)]
    assert len(semantic_keys) == 1
    digest = semantic_keys[0].removeprefix(cache.ENTAILMENT_KEY_PREFIX)
    assert (cache.ENTAILMENT_CACHE_DIR / f"{digest}.json").exists()
    assert cache.discard(cache_keys) == 1
    assert not (cache.ENTAILMENT_CACHE_DIR / f"{digest}.json").exists()


def test_a_verified_quote_carries_its_local_paragraph_context_into_entailment(monkeypatch):
    """US2010 [0033] 회귀: 짧은 프리뷰 발췌 때문에 같은 단락의 편집-버퍼 연결을 버리지 않는다."""
    quote = ("The real time engine 166 provides a preview of the output video sequence by "
             "rendering the output video sequence in substantially real time.")
    paragraph = (
        "0033. " + quote + " To render frames, the real time engine 166 decodes frames, applies "
        "effects, and composites frames. The real time engine 166 can allocate buffers dynamically. "
        "For example, a decode buffer, an effects buffer, and a composite buffer can be allocated "
        "a different capacity on-the-fly for each video segment."
    )
    document = _document("2", "US20100178024A1.pdf", [paragraph])
    match = ElementMatch(
        claim_number=1, label="D", document_id="2", judgment="실질적 동일", directness="direct",
        quote=quote, chunk_id="D2-B-p001-01", limitation_checks=[LimitationCheck(
            index=0, kind="core",
            limitation="제1 프레임 버퍼들을 이용하여 멀티미디어 데이터의 비디오 편집을 수행함",
            disclosed=True, quote=quote, chunk_id="D2-B-p001-01")])
    verify_matches([match], {"2": document})

    def fake(prompt, expect="entailments"):
        assert expect == "entailments"
        assert '"source_context"' in prompt
        assert "applies effects, and composites frames" in prompt
        assert "an effects buffer, and a composite buffer" in prompt
        return {"entailments": [{
            "item_id": "1:D:0", "supported": True,
            "relation": "functional_equivalent", "directness": "direct",
            "reason": "같은 단락이 효과 적용·합성과 이에 사용되는 효과·합성 버퍼를 함께 개시한다.",
        }]}

    monkeypatch.setattr(entailment, "run_cli", fake)
    notes = validate_entailment([match], {"2": document})

    assert notes == []
    assert match.judgment == "실질적 동일" and match.directness == "direct"
    assert match.limitation_checks[0].semantic_status == "accepted"


def test_source_context_is_limited_around_the_verified_quote():
    """비정상적으로 큰 청크도 검증 프롬프트를 무제한 키우지 않는다."""
    prefix = "p" * 8000
    quote = "verified relation between the editing operation and buffers"
    suffix = "s" * 8000
    document = _document("9", "large.pdf", [prefix + quote + suffix])

    context = entailment._source_context(document, "D9-B-p001-01", quote)

    assert len(context) == entailment.MAX_SOURCE_CONTEXT_CHARS
    assert quote in context


_WHOLE = ("We use a normalized-cut algorithm to divide the camera graph into multiple subgraphs "
          "so that each subgraph can be reconstructed independently.")


def _whole_element_match(quote: str) -> ElementMatch:
    """청구항 분해가 한정을 뽑지 못해 구성 원문 한 줄을 통째로 점검한 셀."""
    return ElementMatch(
        claim_number=1, label="A", document_id="1", judgment="실질적 동일", directness="direct",
        quote=quote, chunk_id="D1-B-p001-01", limitation_checks=[LimitationCheck(
            index=0, kind="core", whole_element=True,
            limitation="영상의 GNSS 정보와 영상 간 상대정합 정보를 이용하여 복수의 영상을 클러스터로 그룹화하는 단계",
            disclosed=True, quote=quote, chunk_id="D1-B-p001-01")])


def test_a_rejected_whole_element_check_is_reflected_in_the_judgment(monkeypatch):
    """분해 결과가 없는 구성이라고 해서 의미검증 기각이 노트로만 남아서는 안 된다."""
    document = _document("1", "sensors-21-03939.pdf", [_WHOLE])
    match = _whole_element_match(_WHOLE)
    verify_matches([match], {"1": document})

    monkeypatch.setattr(entailment, "run_cli", lambda prompt, expect="entailments": {"entailments": [
        {"item_id": "1:A:0", "supported": False, "relation": "unsupported",
         "directness": "inferred", "reason": "그래프를 나눈다는 기재만 있고 GNSS·상대정합 기준이 없다."}]})
    notes = validate_entailment([match], {"1": document})

    assert match.judgment == "차이" and match.directness == "inferred"
    assert match.downgraded_from == "실질적 동일"
    assert match.limitation_checks[0].semantic_status == "rejected"
    # whole_element의 실패는 누락 '한정'이 아니라 구성 자체의 미개시라 누락 목록에는 올리지 않는다.
    assert match.missing_limitations == []
    assert "이 구성의 기재" in match.reason
    assert len(notes) == 1


def test_an_incomplete_response_is_not_cached_and_leaves_the_claim_analyzable(monkeypatch):
    """항목이 빠진 응답이 청구항을 죽이거나 캐시에 남아 매 실행마다 되살아나서는 안 된다."""
    document = _document("1", "sensors-21-03939.pdf", [_WHOLE])
    calls: list[str] = []

    def truncated(prompt, expect="entailments"):
        calls.append(prompt)
        return {"entailments": []}          # 스키마는 맞고 항목만 빠진 응답

    monkeypatch.setattr(entailment, "run_cli", truncated)
    first = _whole_element_match(_WHOLE)
    verify_matches([first], {"1": document})
    cache_keys: set[str] = set()
    notes = validate_entailment([first], {"1": document}, cache_keys)

    # 보조 검증의 결손은 chain.py의 analysis_incomplete로 번지지 않아야 한다.
    assert first.error == ""
    assert first.judgment == "실질적 동일" and first.directness == "direct"
    assert first.limitation_checks[0].semantic_status == "error"
    assert len(notes) == 1
    digest = next(iter(cache_keys)).removeprefix(cache.ENTAILMENT_KEY_PREFIX)
    assert not (cache.ENTAILMENT_CACHE_DIR / f"{digest}.json").exists()

    # 캐시가 비어 있으므로 같은 입력을 다시 돌리면 CLI에 다시 묻고, 그때 판정이 반영된다.
    monkeypatch.setattr(entailment, "run_cli", lambda prompt, expect="entailments": {"entailments": [
        {"item_id": "1:A:0", "supported": False, "relation": "unsupported",
         "directness": "inferred", "reason": "GNSS·상대정합 기준이 없다."}]})
    second = _whole_element_match(_WHOLE)
    verify_matches([second], {"1": document})
    validate_entailment([second], {"1": document})

    assert len(calls) == 1
    assert second.judgment == "차이"
    assert second.limitation_checks[0].semantic_status == "rejected"


_CRANK = ("A motor, 305, turns a crank, 306, which in turns pushes a piston, 303, up and down "
          "and is guided by guide rails, 302A & 302B. The piston in turn pushes the screen, 22, "
          "up and down using the 3 or more support struts, 301.")


def _volumetric_claim() -> Claim:
    """체적 디스플레이 청구항. (F)는 (E)가 세운 "상기 플라이휠"을 가리킨다."""
    return Claim(number=1, elements=[
        ClaimElement(label="E", importance=4,
                     text="상기 구동 모터의 회전축에 결합되어 회전 관성을 제공하는 플라이휠",
                     limitations=[Limitation(text="플라이휠이 회전 관성을 제공함", kind="core")]),
        ClaimElement(label="F", importance=5,
                     text="상기 플라이휠의 회전 운동을 상기 투사 스크린의 직선 왕복 운동으로 "
                          "변환하기 위해, 상기 플라이휠에 편심 결합된 힌지 핀과 링크 부재를 "
                          "포함하는 크랭크 - 슬라이드 기구부",
                     limitations=[Limitation(
                         text="크랭크-슬라이드 기구부가 플라이휠의 회전 운동을 투사 스크린의 "
                              "직선 왕복 운동으로 변환함", kind="core")]),
    ])


def test_an_antecedent_from_another_element_is_not_charged_to_this_one(monkeypatch):
    """앞 구성이 세운 지시 대상("상기 플라이휠")이 이 구성의 개시 요구사항이 되어서는 안 된다.

    실측: 모터→크랭크→피스톤→스크린 왕복을 원문 그대로 개시한 문헌이 "'플라이휠'에 대한 개시가
    전혀 없다"는 이유로 (F) 전체를 기각당했다. 플라이휠은 (E)가 세운 대상이고 (E)는 이미 그
    자리에서 미개시로 판정되어 있었으므로, 같은 사실을 두 구성에 두 번 계상한 것이다. 그 결과
    실제로 개시된 크랭크-슬라이드 기구부가 보고서에서 "대응 기재 없음"으로 나갔다.
    """
    document = _document("2", "WO2014165863A2.pdf", [_CRANK])
    match = ElementMatch(
        claim_number=1, label="F", document_id="2", judgment="일부 차이", directness="direct",
        quote=_CRANK, chunk_id="D2-B-p001-01",
        missing_limitations=["힌지 핀이 플라이휠에 편심 결합됨"],
        limitation_checks=[LimitationCheck(
            index=0, kind="core",
            limitation="크랭크-슬라이드 기구부가 플라이휠의 회전 운동을 투사 스크린의 직선 왕복 운동으로 변환함",
            disclosed=True, quote=_CRANK, chunk_id="D2-B-p001-01")])
    verify_matches([match], {"2": document})

    seen: list[str] = []

    def fake(prompt, expect="entailments"):
        seen.append(prompt)
        return {"entailments": [{
            "item_id": "1:F:0", "supported": True, "relation": "functional_equivalent",
            "directness": "direct",
            "reason": "모터의 회전을 크랭크가 받아 피스톤을 상하로 밀고 그 피스톤이 스크린을 왕복시킨다.",
        }]}

    monkeypatch.setattr(entailment, "run_cli", fake)
    notes = validate_entailment([match], {"2": document}, None, [_volumetric_claim()])

    # 지시 대상은 심사자에게 "앞 구성이 세운 대상"으로 전달되어야 한다.
    assert '"reference_terms"' in seen[0] and "플라이휠" in seen[0]
    assert "투사 스크린" in seen[0]
    # 개시된 크랭크-슬라이드 기구부가 살아 남아야 근거가 보고서 본문에 실린다.
    assert match.judgment == "일부 차이" and match.directness == "direct"
    assert match.limitation_checks[0].semantic_status == "accepted"
    assert match.downgraded_from == "" and notes == []


def test_reference_terms_are_omitted_for_elements_that_introduce_their_own_subject(monkeypatch):
    """앞 구성을 참조하지 않는 구성에는 지시 어구를 붙이지 않는다(빈 목록도 보내지 않는다)."""
    document = _document("2", "WO2014165863A2.pdf", [_CRANK])
    match = ElementMatch(
        claim_number=1, label="E", document_id="2", judgment="일부 차이", directness="direct",
        quote=_CRANK, chunk_id="D2-B-p001-01", limitation_checks=[LimitationCheck(
            index=0, kind="core", limitation="플라이휠이 회전 관성을 제공함",
            disclosed=True, quote=_CRANK, chunk_id="D2-B-p001-01")])
    verify_matches([match], {"2": document})

    seen: list[str] = []

    def fake(prompt, expect="entailments"):
        seen.append(prompt)
        return {"entailments": [{"item_id": "1:E:0", "supported": False,
                                 "relation": "unsupported", "directness": "inferred",
                                 "reason": "회전 관성을 제공하는 부재의 기재가 없다."}]}

    monkeypatch.setattr(entailment, "run_cli", fake)
    validate_entailment([match], {"2": document}, None, [_volumetric_claim()])

    # (E)는 "상기 구동 모터"를 참조하지만 그 구성은 이 청구항 행렬 앞자리에 없으므로 비어야 한다.
    # 프롬프트 본문에는 규칙 설명으로 낱말이 나오므로 CONTEXT의 JSON 키만 확인합니다.
    assert '"reference_terms"' not in seen[0]
    assert match.judgment == "차이" and match.directness == "inferred"


def test_an_unchecked_limitation_restores_the_recovery_downgrade(monkeypatch):
    """verify.py가 이 단계에 넘긴 alignment=recovered 강등은 검증이 못 돌면 되살려야 한다."""
    document = _document("1", "sensors-21-03939.pdf", [_WHOLE])
    paraphrase = "We use a normalized-cut algorithm to divide the camera graph into subgraphs."
    match = _whole_element_match(paraphrase)
    verify_matches([match], {"1": document})
    assert match.limitation_checks[0].alignment == "recovered"

    monkeypatch.setattr(entailment, "run_cli",
                        lambda prompt, expect="entailments": {"entailments": []})
    validate_entailment([match], {"1": document})

    assert match.limitation_checks[0].semantic_status == "error"
    assert match.directness == "inferred"     # 복구된 근거는 검증 없이 직접 개시로 두지 않는다
    assert match.judgment == "실질적 동일"      # 다만 비교 판정 자체를 없애지는 않는다


def test_a_document_keeps_supplement_eligibility_for_limitations_it_verifiably_discloses(monkeypatch):
    """core가 기각돼도, 원문으로 검증해 개시한 한정까지 보조 인용발명 자격을 잃어서는 안 된다.

    실측(스마트 윈도우 청구항 × KR101276566): 인용발명 2의 (B)는 core "외부에서 인가되는 전압을
    컨트롤 박스로 인가하거나 차단하는 부재를 포함함"이 기각됐지만, qualifier 두 건은 원문 대조를
    통과해 개시가 인정됐고 그 두 건이 바로 주 인용발명이 놓친 한정이었다. 종전에는 이 단계가
    directness를 absent로 내려 coverage.ineligible_reason이 "직접 근거 없음"으로 걸러 냈고,
    문헌은 자기가 실제로 개시한 한정조차 보완하지 못했다. 더 나쁜 것은 chain._residual_overflow가
    미채택 문헌의 개시를 "결합 문헌 수 상한을 넘어 세우지 않았다"고 적어, 자격 탈락을 상한 탓으로
    잘못 설명한 점이다.
    """
    core_quote = "제1 몸체(101)는 원격조종 자동차의 전원부를 제어하기 위한 제어부와 전기적으로 연결된다."
    switch_quote = ("제1 스위치 버튼(104)이 제1 몸체(101)의 내측으로 눌리는 경우 제1 몸체(101)의 "
                    "능동형 스위치 회로가 개방되고, 원위치로 복귀되면 단락된다.")
    document = _document("2", "KR101276566B1.pdf", [core_quote, switch_quote])
    match = ElementMatch(
        claim_number=1, label="B", document_id="2", judgment="실질적 동일", directness="direct",
        quote=core_quote, chunk_id="D2-B-p001-01", limitation_checks=[
            LimitationCheck(index=0, kind="core",
                            limitation="외부에서 인가되는 전압을 컨트롤 박스로 인가하거나 차단하는 부재를 포함함",
                            disclosed=True, quote=core_quote, chunk_id="D2-B-p001-01"),
            LimitationCheck(index=1, kind="qualifier",
                            limitation="전압 인가 및 차단 여부를 스위치의 작동 상태에 따라 결정함",
                            disclosed=True, quote=switch_quote, chunk_id="D2-B-p002-01"),
        ])
    verify_matches([match], {"2": document})

    monkeypatch.setattr(entailment, "run_cli", lambda prompt, expect="entailments": {"entailments": [
        {"item_id": "1:B:0", "supported": False, "relation": "unsupported", "directness": "inferred",
         "reason": "상대 부재로 전압을 넘기는 전력 경로가 아니라 자기 기기의 전원부를 제어한다."},
        {"item_id": "1:B:1", "supported": True, "relation": "functional_equivalent",
         "directness": "direct", "reason": "버튼의 눌림 상태로 회로가 개폐된다."}]})
    validate_entailment([match], {"2": document})

    from app.coverage import disclosed_limitations, ineligible_reason, is_eligible_supplement

    assert match.judgment == "차이"                       # core가 기각됐으므로 대응으로는 세지 않는다
    assert match.directness == "inferred"                 # 그러나 근거 원문은 남아 있다
    assert ineligible_reason(match) == ""
    assert is_eligible_supplement(match)                  # 개시한 한정으로는 보완할 수 있어야 한다
    assert disclosed_limitations(match) == {"전압 인가 및 차단 여부를 스위치의 작동 상태에 따라 결정함"}
