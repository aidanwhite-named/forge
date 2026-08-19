"""회귀 하니스의 채점·대조 로직. LLM을 부르지 않는 순수 부분만 검사한다.

하니스가 조용히 틀리면 그때부터 모든 프롬프트 수정이 근거 없이 진행된다. 특히
"채점하지 않았는데 통과로 세는" 실수는 겉보기에 초록이라 오래 살아남는다.
"""
import json
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


def test_repeated_runs_bypass_the_decomposition_cache(monkeypatch, tmp_path):
    """분해 캐시를 그대로 두면 일반 반복 실행이 회차마다 같은 분해를 재사용한다.

    claims.assign_importance는 청구항 원문 해시로 공유 캐시를 먼저 읽는다. 구성대비·의미검증
    캐시만 비우면 분해는 고정된 채로 남고, --pin 실행도 등록된 분해를 쓰므로 양쪽 다 분해가
    고정된다. 두 벌을 비교해 분해 단계의 기여를 분리하려던 실험이 아무것도 재지 못한다.
    """
    from app import cache as cache_module

    seen: list = []

    def fake_run_case(case, progress=None, save=True, pin=False, slots=("latest",)):
        seen.append(cache_module.DECOMPOSITION_CACHE_DIR)
        return _observation("동일") | {"invariants": [], "pinned_decomposition": pin}

    monkeypatch.setattr(regress, "run_case", fake_run_case)
    original = cache_module.DECOMPOSITION_CACHE_DIR

    regress.run_case_repeatedly({"id": "x", "dir": tmp_path}, 3)

    assert len(set(seen)) == 3                    # 회차마다 다른 임시 디렉터리
    assert original not in seen                   # 평소 캐시는 읽지도 쓰지도 않는다
    assert cache_module.DECOMPOSITION_CACHE_DIR == original      # finally에서 복구


def test_every_run_keeps_its_own_artifacts(monkeypatch, tmp_path):
    """마지막 회차만 남기면 불안정을 발견하고도 회차별 진단에 판정을 다시 받아야 한다.

    핀 실행의 슬롯도 갈라야 한다. 관측 파일만 가르고 산출물을 같은 자리에 쓰면 핀 실행이
    직전 일반 실행의 result/judgment/report를 덮어써 비교 대상 한쪽이 사라진다.
    """
    slots_seen: list = []

    def fake_run_case(case, progress=None, save=True, pin=False, slots=None):
        slots_seen.append(slots)
        return _observation("동일") | {"invariants": []}

    monkeypatch.setattr(regress, "run_case", fake_run_case)
    regress.run_case_repeatedly({"id": "x", "dir": tmp_path}, 3)
    assert slots_seen == [("run-1",), ("run-2",), ("run-3", "latest")]

    slots_seen.clear()
    regress.run_case_repeatedly({"id": "x", "dir": tmp_path}, 2, pin=True)
    assert slots_seen == [("run-1-pinned",), ("run-2-pinned", "latest-pinned")]


def test_a_decomposition_that_keeps_its_count_but_changes_content_is_unstable():
    """같은 3한정이라도 한정 문언이나 core/qualifier 배분이 바뀌면 같은 분해가 아니다.

    개수만 비교하면 그 변동이 '안정'으로 집계되고, 등급이 흔들린 원인을 비교 단계에서만
    찾게 된다. 원문을 관측에 남기지 않으려면 구조 해시가 유일한 방법이다.
    """
    same_count = [{"claims": {"1": {"A": "3한정 #aaaaaaaaaaaa"}}},
                  {"claims": {"1": {"A": "3한정 #bbbbbbbbbbbb"}}}]
    spread = regress._decomposition_spread(
        [{"decomposition": item["claims"]} for item in same_count])

    assert spread["stable"] is False
    assert spread["unstable"]["청구항 1 (A)"] == ["3한정 #aaaaaaaaaaaa", "3한정 #bbbbbbbbbbbb"]


def test_the_structure_digest_reacts_to_more_than_the_limitation_count():
    """kind 하나만 바뀌어도 그 구성의 근거 요구가 달라지므로 같은 분해로 볼 수 없다."""
    base = {"label": "A", "text": "…부", "importance": 4, "is_sub": False,
            "search_terms": ["x"],
            "limitations": [{"text": "무엇을 함", "kind": "core", "alternative_group": ""}]}
    moved_kind = {**base, "limitations": [
        {"text": "무엇을 함", "kind": "qualifier", "alternative_group": ""}]}
    reworded = {**base, "limitations": [
        {"text": "무엇을 수행함", "kind": "core", "alternative_group": ""}]}

    assert regress._structure_digest(base) == regress._structure_digest(dict(base))
    assert regress._structure_digest(base) != regress._structure_digest(moved_kind)
    assert regress._structure_digest(base) != regress._structure_digest(reworded)
    # 검색어 순서는 판정을 바꾸지 않으므로 흔들림으로 세지 않는다.
    assert regress._structure_digest({**base, "search_terms": ["x"]}) == \
        regress._structure_digest({**base, "search_terms": ["x"]})


def test_pinned_and_normal_measurements_are_kept_in_separate_files(tmp_path):
    """--pin 실행이 직전 일반 실행의 -latest 별칭을 덮어쓰면 두 벌 비교가 성립하지 않는다."""
    case = {"dir": tmp_path}
    normal = regress.aggregate([_observation("동일") | {"pinned_decomposition": False}] * 2)
    pinned = regress.aggregate([_observation("일부 유사") | {"pinned_decomposition": True}] * 2)

    assert normal["kind"] == "sampled" and pinned["kind"] == "sampled-pinned"
    regress.save_observation(case, normal)
    regress.save_observation(case, pinned)

    observations = tmp_path / "observations"
    assert (observations / "sampled-latest.json").exists()
    assert (observations / "sampled-pinned-latest.json").exists()
    # 일반 실행 결과가 핀 실행에 덮이지 않았는지 확인한다.
    kept = json.loads((observations / "sampled-latest.json").read_text(encoding="utf-8"))
    assert kept["claims"]["1"]["elements"]["A"]["judgment"] == "동일"


def test_a_claim_missing_from_one_run_does_not_stop_aggregation():
    """분해가 실패한 회차가 섞여도 집계는 계속되어야 한다."""
    complete, empty = _observation("동일"), _observation("동일")
    empty["claims"] = {}

    merged = regress.aggregate([complete, empty, complete])

    assert merged["claims"]["1"]["runs"] == 3
    assert merged["claims"]["1"]["observed_runs"] == 2


# --- 문헌별 단독 셀 ------------------------------------------------------------
# 결합 후 보고서 행에만 기대값을 걸면, "주 인용발명 단독으로 개시"와 "보조 인용발명이
# 메워 준 것"이 같은 값으로 관측된다. 실측 과대판정(구성 E)이 정확히 그 형태였다.

def _with_cells(observation: dict, cells: dict) -> dict:
    observation["claims"]["1"]["elements"]["A"]["cells"] = cells
    return observation


def test_a_standalone_cell_can_be_scored_apart_from_the_combined_row():
    """결합 후 행은 통과하는데 문헌 단독 셀이 과대한 경우를 잡는다."""
    observation = _with_cells(_observation("실질적 동일"), {
        "2": {"judgment": "실질적 동일", "directness": "direct", "verify": "verified",
              "missing": 0, "adopted": True},
        "1": {"judgment": "일부 유사", "directness": "direct", "verify": "verified",
              "missing": 0, "adopted": False}})
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "min_grade": "일부 차이",                       # 결합 후 행은 이대로 좋다
        "cells": {"2": {"max_grade": "일부 차이"}}}}}}   # 문헌 2 단독으로는 여기까지

    findings = regress.score(observation, expected)
    failed = [item for item in findings if not item["ok"]]

    assert len(failed) == 1
    assert "문헌 2 단독" in failed[0]["where"] and "최대" in failed[0]["reason"]


def test_a_standalone_cell_expectation_that_holds_passes():
    observation = _with_cells(_observation("실질적 동일"), {
        "2": {"judgment": "일부 차이", "missing": 1, "adopted": True}})
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "cells": {"2": {"max_grade": "일부 차이", "missing": 1, "adopted": True}}}}}}

    assert all(item["ok"] for item in regress.score(observation, expected))


def test_a_cell_expectation_against_an_old_observation_fails_loudly():
    """기대값을 적어 두었는데 관측에 셀이 없으면 조용히 통과시키지 않는다.

    채점되지 않는 기대값이 가장 위험하다 — 하니스가 켜져 있다고 믿게 된다.
    """
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "cells": {"2": {"max_grade": "일부 차이"}}}}}}

    findings = regress.score(_observation("실질적 동일"), expected)
    failed = [item for item in findings if not item["ok"]]

    assert len(failed) == 1 and "문헌별 셀이 없습니다" in failed[0]["reason"]


def test_diff_reports_a_standalone_cell_that_flipped_under_an_unchanged_row():
    """보고서 행이 그대로여도 결합의 근거가 바뀐 것은 회귀다."""
    before = _with_cells(_observation("실질적 동일"), {
        "1": {"judgment": "실질적 동일", "missing": 0, "adopted": True},
        "2": {"judgment": "차이", "missing": 2, "adopted": False}})
    after = _with_cells(_observation("실질적 동일"), {
        "1": {"judgment": "차이", "missing": 2, "adopted": False},
        "2": {"judgment": "실질적 동일", "missing": 0, "adopted": True}})

    lines = regress.diff(before, after)

    assert len(lines) == 2
    assert any("문헌 1 단독: 실질적 동일 채택 → 차이 누락2" in line for line in lines)
    assert regress.diff(before, before) == []


def test_diff_ignores_cells_that_the_old_observation_never_recorded():
    before = _observation("실질적 동일")
    after = _with_cells(_observation("실질적 동일"),
                        {"1": {"judgment": "실질적 동일", "missing": 0, "adopted": True}})

    assert regress.diff(before, after) == []


def test_aggregate_keeps_a_standalone_cell_that_moves_between_runs():
    """셀 하나가 갈리면 채택 문헌이 통째로 바뀐다. 동률은 낮은 등급으로 대표한다."""
    high = _with_cells(_observation("실질적 동일"),
                       {"2": {"judgment": "실질적 동일", "missing": 0, "adopted": True}})
    low = _with_cells(_observation("실질적 동일"),
                      {"2": {"judgment": "일부 유사", "missing": 2, "adopted": False}})

    merged = regress.aggregate([high, low])
    cell = merged["claims"]["1"]["elements"]["A"]["cells"]["2"]

    assert cell["judgment"] == "일부 유사"          # 동률 → 낮은 등급
    assert cell["stability"] == "1/2"
    assert cell["spread"] == {"실질적 동일": 1, "일부 유사": 1}
    assert cell["adopted"] is False                 # 과반이 아니면 채택으로 적지 않는다


def test_aggregate_keeps_every_cell_field_the_expectations_can_score():
    """집계가 필드를 버리면 --runs 2 이상에서 그 기대값이 조용히 채점되지 않는다.

    채점되지 않는 기대값은 없는 기대값보다 나쁘다 — 하니스가 켜져 있다고 믿게 만든다.
    """
    runs = [_with_cells(_observation("일부 차이"), {"2": {
        "judgment": "일부 차이", "directness": "direct", "verify": "verified",
        "missing": 0, "unverified": 3, "adopted": True}}) for _ in range(2)]

    cell = regress.aggregate(runs)["claims"]["1"]["elements"]["A"]["cells"]["2"]

    assert cell["unverified"] == 3
    assert cell["directness"] == "direct" and cell["verify"] == "verified"
    # 집계본도 그대로 채점되어야 한다.
    expected = {"adjudicated": True, "claims": {"1": {"A": {"cells": {"2": {
        "unverified": 3, "directness": "direct", "verify": "verified"}}}}}}
    assert all(item["ok"] for item in regress.score(regress.aggregate(runs), expected))


def test_aggregate_resolves_a_split_directness_conservatively():
    """동률을 좋은 쪽으로 대표하면 불안정이 안정으로 보인다. 등급 집계와 같은 철학이다."""
    direct = _with_cells(_observation("일부 차이"),
                         {"2": {"judgment": "일부 차이", "directness": "direct",
                                "verify": "verified"}})
    inferred = _with_cells(_observation("일부 차이"),
                           {"2": {"judgment": "일부 차이", "directness": "inferred",
                                  "verify": "partial"}})

    forward = regress.aggregate([direct, inferred])["claims"]["1"]["elements"]["A"]["cells"]["2"]
    backward = regress.aggregate([inferred, direct])["claims"]["1"]["elements"]["A"]["cells"]["2"]

    assert forward == backward                        # 회차 순서가 대표값을 바꾸지 않는다
    assert forward["directness"] == "inferred"        # 사전순이면 낙관적인 direct가 뽑힌다
    assert forward["verify"] == "partial"


def test_a_limitation_level_expectation_catches_the_wrong_limitation_being_rejected():
    """등급만 걸면 검증기가 **문제의 한정은 계속 인정한 채** 다른 한정을 기각해도 통과한다."""
    observation = _with_cells(_observation("일부 유사"), {"2": {
        "judgment": "일부 유사", "missing": 1, "unverified": 0, "adopted": True,
        "limitations": {"절대좌표계에 대응하는 최종 3D 모델을 생성함": "disclosed",
                        "전역 좌표 정합을 최적화함": "missing"}}})
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "max_grade": "일부 차이",                                  # 등급 기대는 통과하지만
        "cells": {"2": {"must_reject_contains": ["절대좌표계"]}}}}}}  # 한정 기대가 잡는다

    failed = [item for item in regress.score(observation, expected) if not item["ok"]]

    assert len(failed) == 1
    assert "must_reject_contains" in failed[0]["reason"] and "disclosed" in failed[0]["reason"]


def test_a_rejected_limitation_satisfies_the_expectation():
    observation = _with_cells(_observation("일부 유사"), {"2": {
        "judgment": "일부 유사",
        "limitations": {"절대좌표계에 대응하는 최종 3D 모델을 생성함": "missing"}}})
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "cells": {"2": {"must_reject_contains": ["절대좌표계"]}}}}}}

    assert all(item["ok"] for item in regress.score(observation, expected))


def test_an_unverified_limitation_does_not_count_as_rejected():
    """미완료는 기각이 아니다. 확인하지 못한 것으로 기대값을 만족시키면 안 된다."""
    observation = _with_cells(_observation("일부 차이"), {"2": {
        "judgment": "일부 차이",
        "limitations": {"절대좌표계에 대응하는 최종 3D 모델을 생성함": "unverified"}}})
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "cells": {"2": {"must_reject_contains": ["절대좌표계"]}}}}}}

    failed = [item for item in regress.score(observation, expected) if not item["ok"]]
    assert len(failed) == 1 and "unverified" in failed[0]["reason"]


def test_a_phrase_that_no_longer_appears_in_the_decomposition_fails_loudly():
    """분해가 그 어구를 잃으면 기대값은 아무것도 지키지 못한다. 조용히 통과시키지 않는다."""
    observation = _with_cells(_observation("일부 유사"), {"2": {
        "judgment": "일부 유사", "limitations": {"전역 좌표 정합을 최적화함": "missing"}}})
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "cells": {"2": {"must_reject_contains": ["절대좌표계"]}}}}}}

    failed = [item for item in regress.score(observation, expected) if not item["ok"]]
    assert len(failed) == 1 and "분해에 없습니다" in failed[0]["reason"]


def test_must_disclose_is_not_satisfied_by_an_unverified_limitation():
    observation = _with_cells(_observation("일부 차이"), {"1": {
        "judgment": "일부 차이",
        "limitations": {"절대좌표계에 대응하는 최종 3D 모델을 생성함": "unverified"}}})
    expected = {"adjudicated": True, "claims": {"1": {"A": {
        "cells": {"1": {"must_disclose_contains": ["절대좌표계"]}}}}}}

    assert [item["ok"] for item in regress.score(observation, expected)] == [False]


def test_a_novelty_expectation_checks_containment_not_equality():
    observation = _observation("실질적 동일")
    observation["claims"]["1"]["novelty_missing"] = {"2": ["A", "E"], "1": ["B"]}
    expected = {"adjudicated": True, "claims": {"1": {"_novelty_missing": {"2": ["E"]}}}}

    assert all(item["ok"] for item in regress.score(observation, expected))

    expected["claims"]["1"]["_novelty_missing"] = {"1": ["E"]}
    failed = [item for item in regress.score(observation, expected) if not item["ok"]]
    assert len(failed) == 1 and "E이 없습니다" in failed[0]["reason"]


def test_a_limitation_that_moves_between_runs_satisfies_neither_rule():
    """불안정을 어느 방향으로도 통과로 읽으면 안 된다 — 규칙마다 안전한 방향이 반대다."""
    high = _with_cells(_observation("실질적 동일"),
                       {"2": {"judgment": "실질적 동일", "limitations": {"절대좌표계 한정": "disclosed"}}})
    low = _with_cells(_observation("일부 유사"),
                      {"2": {"judgment": "일부 유사", "limitations": {"절대좌표계 한정": "missing"}}})

    merged = regress.aggregate([high, low])
    assert merged["claims"]["1"]["elements"]["A"]["cells"]["2"]["limitations"] == {
        "절대좌표계 한정": regress.UNSTABLE_STATE}

    for rule in ("must_reject_contains", "must_disclose_contains"):
        expected = {"adjudicated": True, "claims": {"1": {"A": {
            "cells": {"2": {rule: ["절대좌표계"]}}}}}}
        assert not all(item["ok"] for item in regress.score(merged, expected)), rule


def test_aggregate_keeps_cells_absent_when_a_run_never_recorded_them():
    with_cells = _with_cells(_observation("동일"), {"1": {"judgment": "동일"}})
    without = _observation("동일")

    merged = regress.aggregate([with_cells, without])

    assert merged["claims"]["1"]["elements"]["A"]["cells"] is None
