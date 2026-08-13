"""분해 전용 반복 측정의 결정론적 부분. LLM을 부르지 않는 축 비교만 검사한다.

이 도구는 아직 런타임 게이트가 아니라 측정 도구다. 그래서 검사할 것은 "무엇을 실질 차이로
보는가"가 아니라 "축을 뭉치거나 흘리지 않는가" 하나다. 축이 서로 새면 라벨이 한 번 흔들릴
때 전 축이 함께 빨개져, 어느 단계를 고쳐야 하는지 알 수 없게 된다.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import decompose_probe as probe  # noqa: E402


def element(**overrides) -> dict:
    base = {"label": "A", "text": "구성 A", "importance": 4, "is_sub": False,
            "is_preamble": False, "search_terms": ["공통 좌표계", "common coordinate system"],
            "limitations": [{"text": "무엇을 함", "kind": "core", "alternative_group": ""},
                            {"text": "조건으로 한정함", "kind": "qualifier",
                             "alternative_group": "g1"},
                            {"text": "다른 조건으로 한정함", "kind": "qualifier",
                             "alternative_group": "g1"}]}
    return {**base, **overrides}


def observation(elements: list[dict], chunks: dict | None = None) -> dict:
    return {"run": 1, "claims": {"1": elements},
            "selected_chunks": chunks or {"1": ["D1-P-0001", "D1-P-0002"]}}


def broken_axes(first: dict, second: dict) -> list[str]:
    comparison = probe.compare([first, second])
    return [axis for axis in probe.AXES if not comparison["1"]["axes"][axis]["stable"]]


def test_identical_decompositions_break_no_axis():
    base = observation([element()])
    assert broken_axes(base, copy.deepcopy(base)) == []


def test_each_axis_is_counted_on_its_own():
    """한 축이 갈렸다고 다른 축까지 갈린 것으로 세면 어느 단계를 고칠지 알 수 없다."""
    base = observation([element()])

    reworded = copy.deepcopy(base)
    reworded["claims"]["1"][0]["limitations"][0]["text"] = "무엇을 수행함"
    assert broken_axes(base, reworded) == ["limitation_text"]

    regraded = copy.deepcopy(base)
    regraded["claims"]["1"][0]["importance"] = 3
    assert broken_axes(base, regraded) == ["importance"]

    researched = copy.deepcopy(base)
    researched["claims"]["1"][0]["search_terms"] = ["world coordinate"]
    assert broken_axes(base, researched) == ["search_terms"]

    rekinded = copy.deepcopy(base)
    rekinded["claims"]["1"][0]["limitations"][0]["kind"] = "qualifier"
    assert broken_axes(base, rekinded) == ["kinds"]


def test_a_changed_label_set_is_its_own_axis():
    base = observation([element()])
    added = observation([element(), element(label="B", text="구성 B")])
    assert "labels" in broken_axes(base, added)
    assert "limitation_count" not in broken_axes(base, added)   # A의 한정 수는 그대로다


def test_selected_chunks_are_measured_even_when_the_decomposition_looks_equal():
    """검색어가 청크 순위를 정하므로, 분해가 갈리면 모델이 **읽는 근거**가 달라진다.

    이 축이 없으면 문언 차이를 표기 흔들림으로 넘길지 판단할 근거가 없다.
    """
    base = observation([element()])
    moved = observation([element()], chunks={"1": ["D1-P-0007", "D1-P-0009"]})
    assert broken_axes(base, moved) == ["selected_chunks"]


def test_alternative_groups_compare_by_shape_not_by_name():
    """같은 두 한정을 묶었는데 그룹 이름만 다른 것을 차이로 세면 실질 변화가 묻힌다."""
    base = observation([element()])
    renamed = copy.deepcopy(base)
    for limitation in renamed["claims"]["1"][0]["limitations"]:
        if limitation["alternative_group"]:
            limitation["alternative_group"] = "대안1"
    assert broken_axes(base, renamed) == []

    regrouped = copy.deepcopy(base)
    regrouped["claims"]["1"][0]["limitations"][2]["alternative_group"] = ""
    assert broken_axes(base, regrouped) == ["alternative_groups"]


def test_normalization_only_strips_spacing_and_case():
    """더 걷어 내면 서로 다른 요구사항이 같은 문자열로 접혀, 도구가 '안 갈렸다'는 근거를
    스스로 만든다. 판단은 사람이 할 일이므로 표기만 맞춘다."""
    assert probe._normalize("  Common  Coordinate System ") == "common coordinate system"
    assert probe._normalize('"층수"를 인식함') == "층수를 인식함"
    # 조사·어미는 건드리지 않는다 — 대상이 다른 두 한정이다.
    assert probe._normalize("층수를 인식함") != probe._normalize("층수가 인식됨")


def test_unstable_labels_are_reported_per_axis():
    """청구항 단위 참/거짓만으로는 한 구성 때문인지 전면적인지 구별되지 않는다."""
    base = observation([element(), element(label="B", text="구성 B")])
    shifted = copy.deepcopy(base)
    shifted["claims"]["1"][1]["importance"] = 2

    comparison = probe.compare([base, shifted])
    assert comparison["1"]["unstable_labels"]["importance"] == ["B"]


def test_cases_with_the_same_claim_text_are_deduplicated_across_whitespace():
    """실측의 두 neareye 사건은 439자와 444자로 띄어쓰기만 다르다.

    원문 그대로 해싱하면 중복이 걸리지 않아, 한 사건의 성질이 통계를 두 배로 끌고 간다.
    """
    live = {"claims_text": "청구항 1 (A) 근안 디스플레이 장치의 세트를 획득하는 단계"}
    waveguide = {"claims_text": "청구항 1  (A) 근안 디스플레이 장치의 세트를\n획득하는 단계"}
    other = {"claims_text": "청구항 1 (A) 전혀 다른 청구항"}

    assert probe._fingerprint(live) == probe._fingerprint(waveguide)
    assert probe._fingerprint(live) != probe._fingerprint(other)
