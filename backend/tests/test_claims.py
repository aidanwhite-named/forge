import json

import pytest

from app import claims
from app import claims as claims_module
from app.claims import (ancestry, assign_importance, decomposition_generation,
                        input_quality_warnings, parse_claims,
                        validate_confirmed_decomposition)


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


def test_a_version_one_decomposition_is_recomputed_under_the_atomic_split_rules(monkeypatch):
    """구형 분해의 core/qualifier 중복을 새 비교 캐시에 그대로 고착시키지 않는다."""
    calls: list[str] = []

    def fake(prompt, expect="claims"):
        calls.append(prompt)
        return {"elements": [{
            "claim_number": 1, "label": "E", "importance": 5,
            "limitations": [
                {"text": "프로세싱 시간 정보를 업데이트함", "kind": "core"},
                {"text": "업데이트 시점을 비디오 편집 수행 중으로 한정함", "kind": "qualifier"},
            ],
        }]}

    monkeypatch.setattr(claims_module, "run_cli", fake)
    legacy = {"version": 1, "claims": {"1": [{
        "label": "E", "text": "비디오 편집 중 프로세싱 시간 정보를 업데이트함",
        "importance": 5,
        "limitations": [{
            "text": "비디오 편집 중 프로세싱 시간 정보를 업데이트함", "kind": "core",
        }],
    }]}}
    parsed = parse_claims("(E) 비디오 편집 중 프로세싱 시간 정보를 업데이트함")

    assert assign_importance(parsed, legacy) == []

    assert len(calls) == 1
    assert "같은 조건을 core와 qualifier에 중복" in calls[0]
    assert "여러 대안에 공통인 문구는 각 대안에 되풀이하지" in calls[0]
    assert legacy["version"] == claims_module.decomposition_generation()
    assert [item.text for item in parsed[0].elements[0].limitations] == [
        "프로세싱 시간 정보를 업데이트함", "업데이트 시점을 비디오 편집 수행 중으로 한정함",
    ]


def test_a_failed_decomposition_is_not_stored(monkeypatch):
    """분해를 못 받아 기본값으로 진행한 결과를 저장하면 그 빈 분해가 계속 재사용된다."""
    def failing(prompt, expect="claims"):
        raise RuntimeError("CLI 실행에 실패했습니다")

    monkeypatch.setattr(claims_module, "run_cli", failing)
    store: dict = {}
    warnings = assign_importance(parse_claims("(A) 쓰기 요청을 큐에 저장하는 것"), store)
    assert warnings and "기본값" in warnings[0]
    assert store == {}


# --- 고정 분해 (실험 통제) --------------------------------------------------------
# 분해는 매 실행 LLM이 새로 만들기 때문에 같은 청구항이 실행마다 다르게 쪼개집니다. 프롬프트
# 한 곳만 바꾼 효과를 재려면 분해를 붙들어 둘 수 있어야 합니다.

def _pinned_file(tmp_path, version: int, text: str = "큐에 저장함"):
    payload = {"version": version, "claims": {"1": [
        {"label": "A", "text": "쓰기 요청을 큐에 저장하는 저장부", "importance": 5,
         "is_sub": False, "search_terms": ["queue"],
         "limitations": [{"text": text, "kind": "core", "alternative_group": ""}]}]}}
    path = tmp_path / "pinned.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_pinned_decomposition_is_used_even_across_a_version_bump(tmp_path, monkeypatch):
    """실험은 대개 버전을 올린 뒤에 한다. 버전으로 막으면 비교 대상 분해를 쓸 수 없다."""
    path = _pinned_file(tmp_path, version="옛 분해 세대")
    monkeypatch.setattr(claims, "DECOMPOSITION_FILE", str(path))
    monkeypatch.setattr(claims, "run_cli",
                        lambda *a, **k: pytest.fail("고정 분해가 있는데 LLM을 불렀습니다."))

    parsed = claims.parse_claims("쓰기 요청을 큐에 저장하는 저장부")
    notes = claims.assign_importance(parsed)

    assert [item.text for item in parsed[0].elements[0].limitations] == ["큐에 저장함"]
    assert any("고정 파일" in note for note in notes)      # 보고서에 드러나야 한다


def test_a_pinned_decomposition_is_ignored_when_the_claim_text_differs(tmp_path, monkeypatch):
    """버전 검사는 건너뛰어도 구성 원문 대조는 남는다. 다른 청구항에 씌우면 안 된다."""
    path = _pinned_file(tmp_path, version=decomposition_generation())
    monkeypatch.setattr(claims, "DECOMPOSITION_FILE", str(path))
    called: list[str] = []

    def fake(prompt, expect="claims"):
        called.append(expect)
        return {"elements": []}

    monkeypatch.setattr(claims, "run_cli", fake)
    parsed = claims.parse_claims("전혀 다른 청구항 문언을 가진 제어부")
    claims.assign_importance(parsed)
    assert called == ["elements"]                        # 고정 분해를 쓰지 않고 새로 물었다


def test_an_explicit_pinned_decomposition_takes_priority_over_the_environment(tmp_path,
                                                                              monkeypatch):
    """회귀 하니스의 사건별 분해가 프로세스 전역 고정 파일과 섞이면 안 된다."""
    path = _pinned_file(tmp_path, version="옛 분해 세대", text="사건별 분해")
    pinned = json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        claims, "_pinned_decomposition",
        lambda: pytest.fail("명시적으로 전달한 분해 대신 환경 고정 파일을 읽었습니다."),
    )
    monkeypatch.setattr(claims, "run_cli",
                        lambda *a, **k: pytest.fail("고정 분해가 있는데 LLM을 불렀습니다."))

    parsed = claims.parse_claims("쓰기 요청을 큐에 저장하는 저장부")
    notes = claims.assign_importance(parsed, pinned_decomposition=pinned)

    assert [item.text for item in parsed[0].elements[0].limitations] == ["사건별 분해"]
    assert any("회귀 사건의 고정 분해" in note for note in notes)


# --- 확정 분해 검증 --------------------------------------------------------------
# _restore_elements는 청구항 번호·라벨·구성 원문이 하나라도 어긋나면 조용히 False를 돌려주고,
# 파이프라인은 확정본을 버린 채 LLM 재분해로 넘어간다. 사용자가 확인한 것과 다른 분해로 판정이
# 도는데 아무도 알아채지 못한다 — 조용한 실패를 400으로 바꾸는 것이 이 검증의 목적이다.

def _proposal(**overrides) -> dict:
    element = {"label": "A", "text": "쓰기 요청을 큐에 저장하는 것", "importance": 4,
               "is_sub": False, "search_terms": ["큐"],
               "limitations": [{"text": "쓰기 요청을 큐에 저장함", "kind": "core",
                                "alternative_group": ""}]}
    element.update(overrides)
    return {"version": "test", "claims": {"1": [element]}}


def test_an_unchanged_confirmation_passes():
    assert validate_confirmed_decomposition(_proposal(), _proposal()) == []


def test_the_user_may_edit_limitations_kind_importance_and_search_terms():
    """사용자가 정할 자리다. 여기까지 막으면 확정 단계가 '확인' 버튼 하나로 줄어든다."""
    edited = _proposal(importance=2, search_terms=["우선순위 큐", "priority queue"],
                       limitations=[{"text": "쓰기 요청을 받음", "kind": "core",
                                     "alternative_group": ""},
                                    {"text": "저장 대상을 큐로 한정함", "kind": "qualifier",
                                     "alternative_group": "g1"}])
    assert validate_confirmed_decomposition(_proposal(), edited) == []


def test_the_element_text_cannot_be_edited():
    """구성 원문이 어긋나면 _restore_elements가 확정본을 통째로 버린다."""
    problems = validate_confirmed_decomposition(_proposal(), _proposal(text="다른 문언"))
    assert problems and "구성 원문은 고칠 수 없습니다" in problems[0]


def test_a_missing_or_extra_claim_is_rejected():
    assert "빠졌습니다" in " ".join(
        validate_confirmed_decomposition(_proposal(), {"version": "test", "claims": {}}))
    extra = _proposal()
    extra["claims"]["2"] = list(extra["claims"]["1"])
    assert "제안에 없는 청구항" in " ".join(
        validate_confirmed_decomposition(_proposal(), extra))


def test_duplicate_labels_are_rejected():
    """_restore_elements는 라벨로 dict를 만들어 중복을 조용히 덮어쓴다."""
    doubled = _proposal()
    doubled["claims"]["1"] = doubled["claims"]["1"] * 2
    assert "중복" in " ".join(validate_confirmed_decomposition(_proposal(), doubled))


def test_a_reordered_or_missing_label_is_rejected():
    dropped = {"version": "test", "claims": {"1": []}}
    assert "라벨과 순서" in " ".join(validate_confirmed_decomposition(_proposal(), dropped))


def test_an_element_without_limitations_is_rejected():
    """한정이 없으면 구성 원문 한 줄을 통째로 점검하게 되어, 확정한 것이 무엇인지 알 수 없다."""
    assert "한정이 최소 하나는" in " ".join(
        validate_confirmed_decomposition(_proposal(), _proposal(limitations=[])))


def test_an_empty_limitation_text_is_rejected():
    blank = _proposal(limitations=[{"text": "   ", "kind": "core", "alternative_group": ""}])
    assert "문언이 비어 있습니다" in " ".join(
        validate_confirmed_decomposition(_proposal(), blank))


def test_an_unknown_kind_is_rejected():
    """kind가 판정 등급을 가른다. 모르는 값이 들어오면 Limitation 검증에서 늦게 터진다."""
    wrong = _proposal(limitations=[{"text": "무엇을 함", "kind": "essential",
                                    "alternative_group": ""}])
    assert "core 또는 qualifier" in " ".join(
        validate_confirmed_decomposition(_proposal(), wrong))


def test_values_that_would_be_silently_truncated_are_rejected():
    """_restore_elements는 넘치는 만큼을 말없이 잘라 낸다. 확정 단계에서 그러면 사용자가 적어
    넣은 한정이 사라진 채 분석이 돈다."""
    many = _proposal(limitations=[{"text": f"한정 {index}", "kind": "core",
                                   "alternative_group": ""} for index in range(13)])
    assert "한정은 12개까지" in " ".join(validate_confirmed_decomposition(_proposal(), many))

    terms = _proposal(search_terms=[f"검색어{index}" for index in range(17)])
    assert "검색어는 16개까지" in " ".join(validate_confirmed_decomposition(_proposal(), terms))

    assert "중요도는 1~5" in " ".join(
        validate_confirmed_decomposition(_proposal(), _proposal(importance=9)))
