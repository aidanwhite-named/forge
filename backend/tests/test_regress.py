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
    return {"kind": "forge", "at": "2026-01-01T00:00:00+00:00",
            "versions": {"decomposition": 5}, "claims": {"1": {
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


def test_element_document_and_limitation_counts_are_scored():
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "document": "4", "disclosed": 4, "total": 4}}}}
    observed = _observation("실질적 동일", disclosed=3, total=4)
    findings = regress.score(observed, expected)
    reasons = " ".join(item["reason"] for item in findings if not item["ok"])
    assert "document='1'" in reasons and "disclosed=3" in reasons


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


def test_an_empty_combination_regression_is_catchable():
    """부 인용발명이 한 건도 서지 않는 회귀는 셀 채점으로는 드러나지 않는다.

    실측에서 그런 일이 났다. 주 인용발명만으로도 구성별 등급은 그대로였기 때문에 A·B·C
    어느 셀도 변하지 않았고, 달라진 것은 조합뿐이었다. 조합 자체를 걸 수 있어야 잡힌다.
    """
    expected = {"adjudicated": True, "claims": {"1": {"_secondaries": ["4"]}}}
    findings = regress.score(_observation("일부 차이"), expected)     # secondaries=[]
    assert [item["ok"] for item in findings] == [False]
    assert "secondaries" in findings[0]["reason"]


def test_an_expected_empty_combination_is_still_scored():
    """"결합이 서면 안 된다"도 기대값이다. 빈 값이라고 채점에서 빠지면 적을 수 없다."""
    expected = {"adjudicated": True, "claims": {"1": {"_secondaries": []}}}
    assert all(item["ok"] for item in regress.score(_observation("일부 차이"), expected))

    observed = _observation("일부 차이")
    observed["claims"]["1"]["secondaries"] = ["2"]
    assert [item["ok"] for item in regress.score(observed, expected)] == [False]


def test_residual_and_uncovered_expectations_are_scored():
    """조합 문헌만 맞고 결합 후 공백이 틀린 보고서를 통과시키면 안 된다."""
    observed = _observation("일부 차이")
    observed["claims"]["1"]["residual"] = ["B"]
    observed["claims"]["1"]["uncovered"] = ["C"]
    expected = {"adjudicated": True, "claims": {"1": {
        "_residual": ["B"], "_uncovered": ["C"]}}}
    assert all(item["ok"] for item in regress.score(observed, expected))

    expected["claims"]["1"]["_residual"] = ["A", "B"]
    assert any(not item["ok"] and "residual" in item["reason"]
               for item in regress.score(observed, expected))


def test_a_stale_frozen_decomposition_is_a_failure():
    """분해 프롬프트가 바뀌면 그 위에서 확정한 기대값도 다시 봐야 한다.

    세대는 프롬프트 문면의 해시라(cache.fingerprint) 사람이 올리는 번호가 아니다. 기대값에
    적어 둔 세대와 실행의 세대가 다르면, 사람이 확정할 때 본 분해가 지금의 분해가 아니다.
    """
    expected = {"adjudicated": True, "decomposition_generation": "옛 세대", "claims": {}}
    findings = regress.score(_observation("동일"), expected)
    assert [item["ok"] for item in findings] == [False]
    assert "decomposition" in findings[0]["reason"]


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


def test_aggregate_preserves_claim_level_list_expectations():
    first = _observation("일부 차이")
    second = _observation("일부 차이")
    third = _observation("일부 차이")
    for observation in (first, second):
        observation["claims"]["1"]["secondaries"] = ["4"]
        observation["claims"]["1"]["residual"] = ["B"]
        observation["claims"]["1"]["uncovered"] = ["C"]
    merged = regress.aggregate([first, second, third])
    assert merged["claims"]["1"]["secondaries"] == ["4"]
    assert merged["claims"]["1"]["residual"] == ["B"]
    assert merged["claims"]["1"]["uncovered"] == ["C"]


def test_a_sampled_observation_can_still_be_scored_and_diffed():
    """반복 측정 결과도 1회 관측과 같은 기준으로 채점·대조되어야 한다."""
    merged = regress.aggregate([_observation("실질적 동일", disclosed=2, total=2)] * 3)
    expected = {"adjudicated": True,
                "claims": {"1": {"A": {"corresponded": True, "min_grade": "실질적 동일"}}}}
    assert all(item["ok"] for item in regress.score(merged, expected))
    assert regress.diff(_observation("실질적 동일", disclosed=2, total=2), merged) == []


def test_repeated_runs_do_not_switch_off_invariant_scoring():
    """--runs를 붙이면 불변식 채점이 꺼지던 결손.

    _report는 'invariants' **키의 존재**를 보고 "기록 없음"을 판단한다. aggregate가 그 키를
    버리면 반복 측정 결과는 언제나 "불변식 도입 이전 관측"으로 읽혀 위반이 있어도 통과한다.
    안정성을 재려고 쓰는 모드가, 기대값 없이도 도는 유일한 채점을 끄는 셈이었다.
    """
    clean, broken = _observation("동일"), _observation("동일")
    clean["invariants"] = []
    broken["invariants"] = ["[불변식 P1] 청구항 1 (B): 공백으로 적었으나 근거가 있습니다"]

    merged = regress.aggregate([clean, clean, clean])
    assert merged["invariants"] == []

    # 한 회차라도 깨졌으면 위반이다. 간헐적으로만 깨지는 쪽이 오히려 오래 살아남는다.
    merged = regress.aggregate([clean, clean, broken])
    assert len(merged["invariants"]) == 1
    assert "회차 1/3에서만" in merged["invariants"][0]


def test_a_run_without_invariant_records_is_not_reported_as_clean():
    """없는 것을 '위반 없음'으로 적으면 옛 관측이 전부 초록으로 보인다."""
    merged = regress.aggregate([_observation("동일"), _observation("동일")])
    assert "invariants" not in merged


def test_the_pin_flag_and_decomposition_survive_aggregation():
    """--runs --pin 결과에서 핀 여부가 사라지면 "앱 경로가 아니다"라는 경고가 사라진다."""
    runs = []
    for shape in ("2한정", "2한정", "5한정"):
        observation = _observation("동일")
        observation["pinned_decomposition"] = True
        observation["decomposition"] = {"1": {"A": shape}}
        runs.append(observation)

    merged = regress.aggregate(runs)

    assert merged["pinned_decomposition"] is True
    # 분해가 갈린 회차가 있으면 등급 변화를 비교 단계 탓으로 읽으면 안 된다.
    assert merged["decomposition_spread"]["stable"] is False
    assert merged["decomposition_spread"]["unstable"]["청구항 1 (A)"] == ["2한정", "5한정"]


def test_a_label_seen_in_only_one_run_is_counted_as_unstable():
    """분모를 '그 구성이 나타난 회차'로 잡으면 1/1 즉 완전 안정으로 집계된다.

    _report의 불안정 목록은 runs > 1을 요구하므로 그 구성은 경고에서 통째로 빠진다. 분해가
    흔들리면 불안정이 정확히 이 형태로 나타나므로, 이 셀을 지우면 반복 측정이 자기가 재려던
    것을 못 본다.
    """
    with_label, without_label = _observation("동일"), _observation("동일")
    del without_label["claims"]["1"]["elements"]["A"]

    element = regress.aggregate([with_label, without_label, without_label])["claims"]["1"]["elements"]["A"]

    assert element["runs"] == 3 and element["observed_runs"] == 1
    assert element["stability"] == "1/3"
    assert element["hits"] != element["runs"]          # _report의 불안정 판정 조건


def test_a_limitation_count_that_moves_between_runs_is_recorded():
    """한정 수가 갈리면 개시 수 중앙값과 짝이 맞지 않는 분모가 된다."""
    merged = regress.aggregate([_observation("동일", total=2), _observation("동일", total=2),
                                _observation("동일", total=8)])
    element = merged["claims"]["1"]["elements"]["A"]
    assert element["total"] == 2 and element["total_spread"] == {2: 2, 8: 1}


def test_a_claim_missing_from_one_run_does_not_stop_aggregation():
    """분해가 실패한 회차가 섞여도 집계는 계속되어야 한다."""
    complete, empty = _observation("동일"), _observation("동일")
    empty["claims"] = {}

    merged = regress.aggregate([complete, empty, complete])

    assert merged["claims"]["1"]["runs"] == 3
    assert merged["claims"]["1"]["observed_runs"] == 2
