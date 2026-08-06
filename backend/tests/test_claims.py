from app import claims as claims_module
from app.claims import (ancestry, assign_importance, input_quality_warnings,
                        parse_claims)


def test_multiline_element_stays_one_component_and_keeps_its_label():
    """줄바꿈으로 쪼개면 (B)가 둘로 갈려 이후 라벨이 전부 밀린다."""
    claims = parse_claims("청구항 1.\r\n(A) 네트워크 모듈;\r\n(B) 영상, 음성을\r\n수집하는 입력 모듈;\r\n(C) 저장 모듈")
    assert [element.label for element in claims[0].elements] == ["A", "B", "C"]
    assert claims[0].elements[1].text == "영상, 음성을 수집하는 입력 모듈"


def test_multiple_claims_are_split_by_header():
    claims = parse_claims(
        "【청구항 1】\n(A) 네트워크 모듈; (B) 저장 모듈\n"
        "【청구항 2】\n제1항에 있어서, (A) 상기 저장 모듈이 SSD인 장치"
    )
    assert [claim.number for claim in claims] == [1, 2]
    assert claims[1].depends_on == 1
    assert claims[0].depends_on is None


def test_numbered_list_prefix_is_not_compared_as_part_of_the_preamble():
    """문서에서 복사한 '1. 청구항 1.'의 앞 목록 번호는 발명의 기술 구성이 아니다."""
    claims = parse_claims(
        "1. 청구항 1.\n콘텐츠 플랫폼의 제어 방법에 있어서,\n"
        "(A) 요청을 수신하는 단계\n(B) 콘텐츠를 저장하는 단계"
    )

    assert len(claims) == 1 and claims[0].number == 1
    assert claims[0].preamble == "콘텐츠 플랫폼의 제어 방법에 있어서,"
    assert claims[0].elements[0].text == "콘텐츠 플랫폼의 제어 방법에 있어서"
    assert [element.label for element in claims[0].elements] == ["P0", "A", "B"]


def test_dependency_on_a_missing_parent_is_dropped():
    """부모항이 입력되지 않으면 '에 있어서' 뒤의 직접 기재만 대비하게 된다."""
    claims = parse_claims("청구항 5.\n제3항에 있어서, (A) 냉각팬을 더 포함하는 장치")
    assert claims[0].depends_on is None


def test_ancestry_walks_the_dependency_chain():
    claims = parse_claims(
        "청구항 1.\n(A) 장치\n청구항 2.\n제1항에 있어서, (A) 추가 한정\n청구항 3.\n제2항에 있어서, (A) 또 다른 한정"
    )
    assert ancestry(claims, 3) == [1, 2]
    assert ancestry(claims, 1) == []


def test_preamble_is_separated_at_the_transition_phrase():
    claims = parse_claims("전자장치에 있어서, (A) 메모리; (B) 프로세서")
    assert claims[0].preamble == "전자장치에 있어서,"
    # 전제부도 판정 대상 구성으로 세운다(P0). 대비에서 통째로 빠지면 제한적 전제부가
    # 인용발명에 없어도 나머지 구성만 같으면 신규성이 부정된다.
    assert [element.label for element in claims[0].elements] == ["P0", "A", "B"]
    assert claims[0].elements[0].is_preamble is True


def test_dependency_only_preamble_is_not_turned_into_an_element():
    """"제1항에 있어서"는 의존 관계 표시일 뿐 기술 내용이 아니다.

    이것까지 구성으로 세우면 모든 종속항이 대응 없는 구성을 하나씩 달고 시작한다.
    """
    claims = parse_claims("【청구항 1】(A) 메모리\n【청구항 2】제1항에 있어서, (A) 상기 메모리는 휘발성인 장치")
    assert [element.label for element in claims[1].elements] == ["A"]


def test_unlabeled_claims_fall_back_to_sequential_labels():
    claims = parse_claims("첫 번째 구성이 되는 문장\n\n두 번째 구성이 되는 문장")
    assert [element.label for element in claims[0].elements] == ["A", "B"]


def test_unbalanced_parentheses_are_reported_without_touching_the_text():
    """구성 분해를 어긋나게 할 수 있는 입력 이상만 알린다.

    특정 오탈자 목록을 코드에 심지 않는다. 사건마다 달라 유지될 수 없고, 심사관이 이미
    읽고 있는 원문을 도구가 대신 판단하는 일이 된다.
    """
    claims = parse_claims("(A) 포즈를 추정(하는 프로세서")
    assert any("괄호" in warning for warning in input_quality_warnings(claims))
    assert "추정(하는" in claims[0].elements[0].text


# --- 구성분해 결과의 저장·재사용 ------------------------------------------------

def test_a_stored_decomposition_is_reused_without_calling_the_cli(monkeypatch):
    """같은 청구항을 다시 분해하면 판정 캐시가 통째로 무효가 된다.

    분해 결과(limitations, search_terms)가 비교 캐시 키에 그대로 들어가므로, 같은 뜻의
    문장이 한두 글자 다르게 나오는 것만으로 이전 판정이 전부 미스가 된다. 실제로 같은
    청구항 1·같은 문헌·같은 지침으로 두 번 돌린 결과가 "…를 기반으로 이미지를 생성함"과
    "…를 입력으로 받아서 이미지를 생성함"으로 갈렸고, 셀 판정을 처음부터 다시 받았다.
    """
    calls: list[str] = []

    def fake(prompt, expect="claims"):
        calls.append(expect)
        return {"elements": [{"claim_number": 1, "label": "A", "importance": 5,
                              "search_terms": ["큐", "queue"],
                              "limitations": [{"text": "쓰기 요청을 큐에 저장함", "kind": "core"}]}]}

    monkeypatch.setattr(claims_module, "run_cli", fake)
    store: dict = {}
    first = parse_claims("(A) 쓰기 요청을 큐에 저장하는 메모리 컨트롤러")
    assert assign_importance(first, store) == []
    assert calls == ["elements"]

    second = parse_claims("(A) 쓰기 요청을 큐에 저장하는 메모리 컨트롤러")
    assert assign_importance(second, store) == []
    assert calls == ["elements"]                                  # 두 번째는 CLI를 부르지 않는다
    assert [item.model_dump() for item in second[0].elements[0].limitations] == \
           [item.model_dump() for item in first[0].elements[0].limitations]
    assert second[0].elements[0].search_terms == first[0].elements[0].search_terms
    assert second[0].elements[0].importance == 5


def test_an_edited_claim_is_decomposed_again(monkeypatch):
    """라벨만 보고 예전 분해를 씌우면 보고서에는 새 문언이, 판정에는 옛 한정이 실린다."""
    calls: list[str] = []

    def fake(prompt, expect="claims"):
        calls.append(expect)
        return {"elements": [{"claim_number": 1, "label": "A", "importance": 3,
                              "limitations": [{"text": "저장함", "kind": "core"}]}]}

    monkeypatch.setattr(claims_module, "run_cli", fake)
    store: dict = {}
    assign_importance(parse_claims("(A) 쓰기 요청을 큐에 저장하는 메모리 컨트롤러"), store)
    assign_importance(parse_claims("(A) 읽기 요청을 스택에 저장하는 메모리 컨트롤러"), store)
    assert calls == ["elements", "elements"]


def test_a_failed_decomposition_is_not_stored(monkeypatch):
    """분해를 못 받아 기본값으로 진행한 결과를 저장하면 그 빈 분해가 계속 재사용된다."""
    def failing(prompt, expect="claims"):
        raise RuntimeError("CLI 실행에 실패했습니다")

    monkeypatch.setattr(claims_module, "run_cli", failing)
    store: dict = {}
    warnings = assign_importance(parse_claims("(A) 쓰기 요청을 큐에 저장하는 것"), store)
    assert warnings and "기본값" in warnings[0]
    assert store == {}
