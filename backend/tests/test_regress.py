"""회귀 하니스의 채점·대조 로직. LLM을 부르지 않는 순수 부분만 검사한다.

하니스가 조용히 틀리면 그때부터 모든 프롬프트 수정이 근거 없이 진행된다. 특히
"채점하지 않았는데 통과로 세는" 실수는 겉보기에 초록이라 오래 살아남는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import regress  # noqa: E402


def _observation(judgment: str, corresponded: bool = True, disclosed: int = 2,
                 total: int = 2, track: str = "inventive_step_combination",
                 primary: str = "1") -> dict:
    return {"kind": "forge", "at": "2026-01-01T00:00:00+00:00", "claims": {"1": {
        "track": track, "primary": primary, "secondaries": [], "residual": [], "uncovered": [],
        "elements": {"A": {"judgment": judgment, "corresponded": corresponded,
                           "disclosed": disclosed, "total": total, "document": "1"}}}}}


def test_a_grade_inside_the_expected_range_passes():
    expected = {"adjudicated": True,
                "claims": {"1": {"A": {"corresponded": True, "min_grade": "일부 차이"}}}}
    findings = regress.score(_observation("실질적 동일"), expected)
    assert all(item["ok"] for item in findings)


def test_a_grade_below_the_minimum_fails():
    expected = {"adjudicated": True,
                "claims": {"1": {"A": {"corresponded": True, "min_grade": "실질적 동일"}}}}
    findings = regress.score(_observation("일부 유사"), expected)
    assert [item["ok"] for item in findings] == [False]
    assert "최소" in findings[0]["reason"]


def test_a_grade_above_the_maximum_fails():
    """max_grade는 '이보다 높게 보면 안 된다'는 기대다. 과대 개시를 잡는 쪽이다."""
    expected = {"adjudicated": True,
                "claims": {"1": {"A": {"corresponded": False, "max_grade": "차이"}}}}
    findings = regress.score(_observation("실질적 동일"), expected)
    reasons = " ".join(item["reason"] for item in findings if not item["ok"])
    assert "corresponded" in reasons and "최대" in reasons


def test_an_element_missing_from_the_observation_is_a_failure():
    """판정을 못 받은 것을 조용히 넘기면 하니스가 회귀를 통과시킨다."""
    expected = {"adjudicated": True, "claims": {"1": {"B": {"corresponded": True}}}}
    findings = regress.score(_observation("동일"), expected)
    assert [item["ok"] for item in findings] == [False]
    assert "관측에 없습니다" in findings[0]["reason"]


def test_claim_level_expectations_catch_a_changed_citation_combination():
    """인용발명 조합이 바뀌는 회귀가 실제로 났으므로 걸 수 있어야 한다."""
    expected = {"adjudicated": True, "claims": {"1": {"_primary": "2"}}}
    findings = regress.score(_observation("동일", primary="1"), expected)
    assert [item["ok"] for item in findings] == [False]
    assert "primary" in findings[0]["reason"]


def test_diff_reports_only_what_actually_changed():
    before = _observation("일부 차이", disclosed=2, total=3)
    after = _observation("실질적 동일", disclosed=3, total=3)
    lines = regress.diff(before, after)
    assert len(lines) == 1 and "일부 차이 2/3" in lines[0] and "실질적 동일 3/3" in lines[0]
    assert regress.diff(before, before) == []


def test_diff_reports_a_changed_citation_combination():
    before = _observation("동일", primary="1")
    after = _observation("동일", primary="2")
    assert any("primary: 1 → 2" in line for line in regress.diff(before, after))


def test_observations_with_different_schemas_are_compared_on_shared_fields_only():
    """저장된 관측에는 forge/sampled/llm 세 스키마가 섞여 있다. 스키마 차이를 판정 변화로
    세면 전 셀이 '변경'으로 찍혀 실제 변화가 묻힌다."""
    sampled = {"claims": {"1": {"elements": {"A": {
        "judgment": "동일", "corresponded": True,
        "disclosed_median": 2, "total": 2, "runs": 3, "stability": 1.0}}}}}
    forge = {"claims": {"1": {"elements": {"A": {
        "judgment": "동일", "corresponded": True,
        "disclosed": 2, "total": 2, "document": "1"}}}}}
    assert regress.diff(sampled, forge) == []          # 같은 판정이므로 변화 없음

    # 판정이 실제로 다르면 스키마가 달라도 잡는다.
    forge["claims"]["1"]["elements"]["A"]["judgment"] = "일부 차이"
    assert len(regress.diff(sampled, forge)) == 1


def test_an_llm_observation_without_counts_still_compares_on_judgment():
    """kind=llm 관측에는 개시 수가 없다. 있는 축으로만 비교해야 한다."""
    llm = {"claims": {"1": {"elements": {"A": {
        "judgment": "일부 유사", "corresponded": True, "note": "모델 자체 판단"}}}}}
    forge = {"claims": {"1": {"elements": {"A": {
        "judgment": "일부 유사", "corresponded": True,
        "disclosed": 1, "total": 3, "document": "2"}}}}}
    assert regress.diff(llm, forge) == []


# --- 안정성 측정 -------------------------------------------------------------------
# 1회 관측만으로는 등급이 한 칸 달라졌을 때 회귀인지 그날의 운인지 가릴 수 없다.

def test_aggregate_records_the_spread_not_just_the_winner():
    """대표값만 남기면 3:0으로 안정된 셀과 2:1로 갈린 셀이 같아 보인다."""
    runs = [_observation("일부 차이"), _observation("일부 차이"), _observation("일부 유사")]
    merged = regress.aggregate(runs)
    element = merged["claims"]["1"]["elements"]["A"]
    assert merged["kind"] == "sampled"
    assert element["judgment"] == "일부 차이" and element["stability"] == "2/3"
    assert element["spread"] == {"일부 차이": 2, "일부 유사": 1}


def test_a_tie_between_grades_resolves_to_the_lower_one():
    """표본 동률을 미개시로 보는 파이프라인과 같은 방향으로 맞춘다."""
    merged = regress.aggregate([_observation("실질적 동일"), _observation("일부 유사")])
    assert merged["claims"]["1"]["elements"]["A"]["judgment"] == "일부 유사"


def test_aggregate_tracks_a_citation_combination_that_moves_between_runs():
    """인용발명 조합이 회차마다 달라지는 것은 그 자체로 알아야 할 사실이다."""
    merged = regress.aggregate([_observation("동일", primary="1"),
                                _observation("동일", primary="2"),
                                _observation("동일", primary="2")])
    assert merged["claims"]["1"]["primary"] == "2"
    assert merged["claims"]["1"]["primary_spread"] == {"1": 1, "2": 2}


def test_a_sampled_observation_can_still_be_scored_and_diffed():
    """반복 측정 결과도 1회 관측과 같은 기준으로 채점·대조되어야 한다."""
    merged = regress.aggregate([_observation("실질적 동일", disclosed=2, total=2)] * 3)
    expected = {"adjudicated": True,
                "claims": {"1": {"A": {"corresponded": True, "min_grade": "실질적 동일"}}}}
    assert all(item["ok"] for item in regress.score(merged, expected))
    assert regress.diff(_observation("실질적 동일", disclosed=2, total=2), merged) == []
