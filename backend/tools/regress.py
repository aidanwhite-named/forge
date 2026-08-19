"""회귀 하니스. 동결된 사건들에 대해 판정을 다시 받고 기대값과 대조합니다.

**왜 필요한가.** 이 파이프라인에서 판정 품질을 바꾸는 것은 대부분 프롬프트 문구입니다.
그런데 프롬프트를 고친 효과는 셀 하나를 눈으로 확인하는 것으로는 알 수 없습니다. 관측한
셀 하나를 고치면 다른 셀이 깨집니다. 한 번 고칠 때마다 **전 사건 × 전 구성**을 다시
채점해야 합니다.

**사건 데이터는 저장소에 없습니다.** cases/ 에는 분석 대상 청구항 원문과 인용문헌 추출
본문이 들어 있어 공개 저장소에 올리지 않습니다(.gitignore). 하니스 코드만 커밋하고 데이터는
각자 로컬에 둡니다. 사건이 없으면 이 스크립트는 조용히 아무것도 하지 않습니다.

**사건 한 건의 구성**
    case.json            제목·원본 job·문헌 목록·그때의 analysis_prompt
    claims.txt           청구항 원문
    claim_elements.json  등록 당시의 분해. 기본 실행에서는 쓰지 않고 --pin일 때만 씁니다.
    documents.json       PDF 추출본(동결). 재추출하면 추출 로직 변경이 섞입니다.
    expected.json        사람이 판정한 기대값. adjudicated=false면 채점에서 뺍니다.
    observations/        실행 스냅샷. -latest 별칭과 타임스탬프본을 함께 남깁니다.

**사용**
    python backend/tools/regress.py                     전 사건 실행·채점(앱과 같은 경로)
    python backend/tools/regress.py --case smart-window 한 사건만
    python backend/tools/regress.py --score-only        저장된 최신 관측만 채점(LLM 미호출)
    python backend/tools/regress.py --diff A.json B.json 두 관측 대조
    python backend/tools/regress.py --adopt history/<job_id>  실패한 실행을 사건으로 등록
    python backend/tools/regress.py --pin               분해를 고정(프롬프트 실험 전용)

**채점의 축은 사건별 정답이 아니라 성질입니다.** expected.json은 사람이 인용문헌을 통독해야
쓸 수 있어서 좀처럼 늘지 않고, 늘지 않으면 다음 사건은 언제나 처음 보는 사건입니다. 그래서
어떤 청구항·어떤 문헌에서도 참이어야 하는 불변식(report.pipeline_invariants)을 기대값과
무관하게 채점합니다. --adopt로 등록만 해 두면 그날부터 그 사건이 회귀를 잡습니다.
"""
import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

DEFAULT_CASES_DIR = ROOT / "cases"
# 등급 서열은 앱과 같은 표를 씁니다. 하니스가 자기 사다리를 들면 같은 등급을 두 곳이
# 다르게 읽습니다.
from app.coverage import DIRECTNESS_RANK, JUDGMENT_RANK, VERIFY_RANK   # noqa: E402
from app.report import to_markdown                # noqa: E402


# --- 관측 ---------------------------------------------------------------------

def _kind(base: str, pinned: bool) -> str:
    """핀 실행과 일반 실행의 관측을 다른 파일 계열로 가릅니다.

    save_observation이 kind로 파일 이름을 짓기 때문에, 같은 이름을 쓰면 --pin 실행이 직전
    일반 실행의 -latest 별칭을 덮어씁니다. 일반 N회와 핀 N회를 나란히 놓고 분해 단계의
    기여를 분리하려는 것이 이 모드의 목적인데, 두 벌 중 하나가 남지 않으면 비교가 성립하지
    않습니다.
    """
    return f"{base}-pinned" if pinned else base


def observe(result, kind: str = "forge", note: str = "", pinned: bool = False,
            decomposition: dict | None = None) -> dict:
    """AnalysisResult를 사건 간 비교가 가능한 최소 형태로 줄입니다.

    보고서 문장이나 발췌는 담지 않습니다. 그것까지 넣으면 표현이 조금 바뀔 때마다 관측이
    달라져 무엇이 실제로 변했는지 묻히고, 문헌 원문이 관측 파일로 새어 나갑니다.

    **불변식 위반과 분해 지문을 함께 담습니다.** 앞의 것은 기대값 없이도 채점할 수 있는
    유일한 신호이고, 뒤의 것은 판정이 흔들렸을 때 그 원인이 분해인지 비교인지를 가릅니다 —
    분해는 앱에서 매 실행 새로 만들어지므로 같은 사건에서도 달라질 수 있고, 그 차이 하나가
    인용발명 조합까지 뒤집습니다.

    **구성 하나에 대해 결합 후 행과 문헌별 단독 셀을 함께 담습니다.** 보고서 행은 이미
    결합이 끝난 결과라, 거기에만 기대값을 걸면 "주 인용발명 단독으로는 개시가 아닌데 보조
    인용발명이 메워 준 것"과 "주 인용발명 단독으로 개시"가 같은 값으로 관측됩니다. 실측에서
    정확히 그 형태의 과대판정이 나왔습니다 — 구성 E가 문헌 2 단독으로 실질적 동일을 받았고,
    절대좌표계 한정을 실제로 댄 것은 문헌 1이었는데 보고서 행만 보면 구별되지 않습니다.
    문헌별 셀을 함께 남기면 그 두 가지에 서로 다른 기대값을 걸 수 있습니다.
    """
    claims: dict[str, dict] = {}
    for report in result.reports:
        coverage = {entry.label: entry for entry in report.chain.element_coverage}
        elements = {}
        for item in report.claims:
            elements[item.label] = {
                "judgment": item.grade if item.grade in JUDGMENT_RANK else _grade_of(item),
                "corresponded": item.corresponded,
                "disclosed": item.disclosed_limitations,
                "total": item.total_limitations,
                "document": item.adopted_document or "",
                "cells": _cells(coverage.get(item.label)),
            }
        claims[str(report.claim_number)] = {
            "track": report.track,
            "primary": report.chain.primary or "",
            "secondaries": list(report.chain.secondaries),
            "residual": list(report.chain.residual),
            "reserved": list(report.chain.reserved),
            "uncovered": list(report.chain.uncovered),
            "pending": list(report.chain.combination_pending),
            # 신규성 게이트가 문헌별로 무엇을 빠뜨렸다고 보았는지. 셀 등급과 별개입니다 —
            # 게이트는 등급 외에 직접성·검증·누락까지 함께 요구하므로, 등급이 같아도 이쪽이
            # 갈릴 수 있고 그 차이가 곧 "이 문헌 하나로 신규성이 부정되는가"입니다.
            "novelty_missing": {document_id: list(labels) for document_id, labels
                                in (report.chain.novelty.missing_by_document or {}).items()},
            "elements": elements,
        }
    return {"kind": _kind(kind, pinned), "at": datetime.now(timezone.utc).isoformat(),
            "note": note, "versions": _versions(), "pinned_decomposition": pinned,
            "invariants": [item for item in result.verify_notes if item.startswith("[불변식")],
            "decomposition": _decomposition_fingerprint(result, decomposition),
            "claims": claims}


def _cells(coverage) -> dict[str, dict]:
    """구성 1개에 대한 **문헌별 단독 셀** 판정. 결합 전의 값입니다.

    ElementCoverage.candidates는 (구성 × 문헌) 전수를 담는 유일한 자리입니다. 채택되지
    않은 문헌도 사유와 함께 남아 있어, 결합 결과 한 줄로는 볼 수 없는 두 가지를 여기서만
    구별할 수 있습니다 — 어느 문헌이 그 구성을 실제로 개시했는가, 그리고 채택된 문헌이
    빠뜨린 한정을 어느 문헌이 댔는가.

    발췌나 사유 문언은 담지 않습니다. 표현이 조금 달라질 때마다 관측이 통째로 '변경'으로
    찍히고, 문헌 원문이 관측 파일로 새어 나갑니다(observe와 같은 규율).
    """
    if coverage is None:
        return {}
    return {candidate.document_id: {
        "judgment": candidate.judgment,
        "directness": candidate.directness,
        "verify": candidate.verify,
        "missing": candidate.missing_count,
        # 누락과 따로 남깁니다. 이 셀이 어떤 등급을 받았을 때 그것이 **검증된 개시** 위에
        # 선 것인지 확인하지 못한 한정 위에 선 것인지가 여기서만 구별됩니다.
        "unverified": candidate.unverified_count,
        "adopted": candidate.adopted,
        # 한정 문언 → 상태. 등급만 고정하면 검증기가 **문제의 그 한정은 계속 인정한 채**
        # 다른 한정을 잘못 기각해도 등급이 내려가 회귀가 통과합니다. 오판을 자리째 고정하려면
        # 한정 단위로 걸 수 있어야 합니다.
        #
        # 청구항 원문이 관측 파일에 들어가지만, cases/는 저장소 밖이고(.gitignore) 여기 실리는
        # 것은 인용문헌 발췌가 아니라 **사용자 자신의 청구항 분해**입니다. 문헌 원문이 새어
        # 나가지 않는다는 observe의 규율은 그대로입니다.
        "limitations": dict(candidate.limitation_states),
    } for candidate in coverage.candidates}


def _decomposition_fingerprint(result, decomposition: dict | None = None) -> dict:
    """구성별 한정 수와 분해 구조 해시. 분해가 달라졌는지 한 줄로 보기 위한 것입니다.

    한정 문언 전체를 남기면 관측 파일에 청구항 원문이 그대로 들어가고, 표현이 한 글자만
    달라져도 전 구성이 '변경'으로 찍혀 무엇이 실제로 변했는지 묻힙니다. 그래서 사람이 읽는
    자리에는 개수만 씁니다.

    **개수만으로는 분해가 같은지 알 수 없습니다.** 같은 3한정이라도 한정 문언이 다르거나
    core/qualifier 배분이 바뀌면 비교 캐시 키·판정 근거·검색어가 전부 달라집니다. 개수가
    같다는 이유로 '안정'으로 집계하면, 분해가 회차마다 흔들리는데도 그 사실이 측정에서
    사라집니다 — 일반 실행과 --pin 실행을 비교하는 목적이 정확히 그것을 재는 것입니다.
    그래서 개수 옆에 구조 해시를 함께 답니다. 해시라서 원문은 새어 나가지 않습니다.
    """
    stored = (decomposition or {}).get("claims") or {}
    fingerprint: dict[str, dict[str, str]] = {}
    for report in result.reports:
        number = str(report.claim_number)
        digests = {entry.get("label"): _structure_digest(entry)
                   for entry in stored.get(number) or []}
        fingerprint[number] = {
            item.label: f"{item.total_limitations}한정"
                        + (f" #{digests[item.label]}" if item.label in digests else "")
            for item in report.claims}
    return fingerprint


def _structure_digest(entry: dict) -> str:
    """구성 하나의 분해 구조 해시.

    판정을 좌우하는 것만 넣습니다. 구성 문언·중요도·종속 여부·검색어와 한정별
    문언·kind·대체군입니다. 이 중 하나만 달라져도 그 구성의 비교 캐시 키와 근거 요구가
    달라지므로, 같은 분해로 볼 수 없습니다.
    """
    payload = {
        "text": entry.get("text", ""),
        "importance": entry.get("importance"),
        "is_sub": entry.get("is_sub"),
        "search_terms": sorted(entry.get("search_terms") or []),
        "limitations": [{"text": item.get("text", ""), "kind": item.get("kind", ""),
                         "alternative_group": item.get("alternative_group", "")}
                        for item in entry.get("limitations") or []],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _grade_of(item) -> str:
    """보고서 표시 등급을 판정 라벨로 되돌립니다.

    ClaimResult.grade는 사람이 읽는 문구('기술 사상 동일, 세부 구현 방식의 단순 변경')라
    서열표에 없습니다. 관측에는 서열을 매길 수 있는 라벨을 남겨야 채점이 됩니다.
    """
    from app.coverage import REPORT_GRADES
    for judgment, (label, _) in REPORT_GRADES.items():
        if label == item.grade:
            return judgment
    return "대응 없음"


def _versions() -> dict:
    from app.cache import compare_generation
    from app.claims import decomposition_generation
    from app.entailment import entailment_generation
    from app.config import COMPARE_SAMPLES, load_runtime_settings
    settings = load_runtime_settings()
    return {"compare": compare_generation(), "entailment": entailment_generation(),
            "decomposition": decomposition_generation(), "samples": COMPARE_SAMPLES,
            "provider": settings["provider"], "model": settings["model"]}


# --- 채점 ---------------------------------------------------------------------

def score(observation: dict, expected: dict) -> list[dict]:
    """기대값과 대조한 결과. expected에 적히지 않은 구성은 채점하지 않습니다.

    min_grade/max_grade는 **범위**입니다. 정확한 등급을 요구하면 사람이 확정하지 못한
    경계(예: '일부 유사'인지 '일부 차이'인지)까지 하니스가 강제하게 되고, 그러면 기대값을
    실제 출력에 맞춰 고치는 일이 반복됩니다.
    """
    findings: list[dict] = []
    if "decomposition_generation" in expected:
        actual_version = (observation.get("versions") or {}).get("decomposition")
        wanted_version = expected["decomposition_generation"]
        findings.append({
            "where": "분해 세대",
            "ok": actual_version == wanted_version,
            "reason": (f"decomposition={actual_version!r}"
                       if actual_version == wanted_version
                       else f"decomposition={actual_version!r}, 기대 {wanted_version!r}"),
        })
    for number, elements in (expected.get("claims") or {}).items():
        observed_claim = (observation.get("claims") or {}).get(number) or {}
        observed_elements = observed_claim.get("elements") or {}
        for label, spec in elements.items():
            if label.startswith("_"):               # 청구항 수준 기대값. 아래에서 따로 봅니다.
                continue
            actual = normalize(observed_elements.get(label))
            where = f"청구항 {number} ({label})"
            if actual is None:
                findings.append({"where": where, "ok": False, "reason": "판정이 관측에 없습니다"})
                continue
            for entry in _check_element(where, spec, actual):
                findings.append(entry)
        # 청구항 수준 기대값은 선택입니다. 구성 라벨과 섞이지 않도록 "_track"·"_primary"처럼
        # 밑줄을 붙여 적습니다. 인용발명 조합이 통째로 바뀌는 회귀를 걸 수 있게 둡니다.
        #
        # _secondaries가 필요한 이유: 부 인용발명이 **한 건도** 채택되지 않아도 구성별 등급은
        # 주 인용발명만으로 그대로라, 셀 채점으로는 전혀 드러나지 않습니다. 조합 자체를 걸 수
        # 있어야 잡힙니다.
        #
        # 키가 있으면 값이 비어 있어도 채점합니다. `if wanted`로 걸러 내면 "_secondaries": []
        # 즉 "결합이 서면 안 된다"는 기대를 아예 적을 수 없습니다.
        for key in ("track", "primary", "secondaries", "residual", "reserved", "uncovered"):
            marker = f"_{key}"
            if marker not in elements:
                continue
            wanted = elements[marker]
            actual_value = observed_claim.get(key)
            if actual_value != wanted:
                findings.append({"where": f"청구항 {number}", "ok": False,
                                 "reason": f"{key}={actual_value!r}, 기대 {wanted!r}"})
            else:
                findings.append({"where": f"청구항 {number}", "ok": True,
                                 "reason": f"{key}={actual_value!r}"})
        findings += _check_novelty(number, elements.get("_novelty_missing"),
                                   observed_claim.get("novelty_missing"))
    return findings


def _check_novelty(number: str, wanted: dict | None, actual: dict | None) -> list[dict]:
    """신규성 게이트의 문헌별 누락. **포함**을 봅니다.

    등가로 걸면 무관한 구성 하나가 움직일 때마다 기대값을 고쳐야 하고, 그러면 정작 지키려던
    한 줄("이 문헌 단독으로는 그 구성이 개시되지 않는다")이 잦은 수정에 묻힙니다.
    """
    if not wanted:
        return []
    where = f"청구항 {number} 신규성"
    if actual is None:
        return [{"where": where, "ok": False,
                 "reason": "관측에 신규성 누락 기록이 없습니다 (관측을 다시 받으십시오)"}]
    findings: list[dict] = []
    for document_id, labels in sorted(wanted.items()):
        observed = actual.get(document_id)
        if observed is None:
            findings.append({"where": f"{where} 문헌 {document_id}", "ok": False,
                             "reason": "이 문헌의 기록이 없습니다"})
            continue
        absent = [label for label in labels if label not in observed]
        findings.append({"where": f"{where} 문헌 {document_id}", "ok": not absent,
                         "reason": (f"누락 목록에 {', '.join(absent)}이 없습니다 (관측 {observed})"
                                    if absent else f"{', '.join(labels)} 포함")})
    return findings


def _check_element(where: str, spec: dict, actual: dict) -> list[dict]:
    findings: list[dict] = []
    if "corresponded" in spec and actual["corresponded"] != spec["corresponded"]:
        findings.append({"where": where, "ok": False,
                         "reason": f"corresponded={actual['corresponded']}, "
                                   f"기대 {spec['corresponded']}"})
    rank = JUDGMENT_RANK.get(actual["judgment"], 0)
    if "min_grade" in spec and rank < JUDGMENT_RANK.get(spec["min_grade"], 0):
        findings.append({"where": where, "ok": False,
                         "reason": f"{actual['judgment']} < 최소 {spec['min_grade']}"})
    if "max_grade" in spec and rank > JUDGMENT_RANK.get(spec["max_grade"], 5):
        findings.append({"where": where, "ok": False,
                         "reason": f"{actual['judgment']} > 최대 {spec['max_grade']}"})
    if "document" in spec and actual["document"] != spec["document"]:
        findings.append({"where": where, "ok": False,
                         "reason": f"document={actual['document']!r}, 기대 {spec['document']!r}"})
    for key in ("disclosed", "total"):
        if key in spec and actual[key] != spec[key]:
            findings.append({"where": where, "ok": False,
                             "reason": f"{key}={actual[key]!r}, 기대 {spec[key]!r}"})
    findings += _check_cells(where, spec.get("cells") or {}, actual.get("cells"))
    if not findings:
        findings.append({"where": where, "ok": True, "reason": _brief(actual)})
    return findings


def _check_cells(where: str, wanted: dict, actual: dict | None) -> list[dict]:
    """문헌별 단독 셀 기대값. 결합 후 행과 **따로** 채점합니다.

    구성 하나에 두 종류의 기대를 걸 수 있어야 합니다. "이 문헌 단독으로는 여기까지"와
    "결합까지 마친 보고서 행은 여기까지"는 서로 다른 주장이고, 과대판정은 대개 앞의 것이
    틀렸는데 뒤의 것이 맞아서 가려집니다.

    관측에 cells가 없으면(옛 관측) 조용히 통과시키지 않고 불합격으로 적습니다. 기대값을
    적어 두었는데 채점되지 않는 상태가 가장 위험합니다 — 하니스가 켜져 있다고 믿게 됩니다.
    """
    findings: list[dict] = []
    for document_id, spec in sorted(wanted.items()):
        label = f"{where} 문헌 {document_id} 단독"
        if actual is None:
            findings.append({"where": label, "ok": False,
                             "reason": "관측에 문헌별 셀이 없습니다 (관측을 다시 받으십시오)"})
            continue
        cell = actual.get(document_id)
        if cell is None:
            findings.append({"where": label, "ok": False, "reason": "이 문헌의 셀이 관측에 없습니다"})
            continue
        rank = JUDGMENT_RANK.get(cell.get("judgment"), 0)
        failed = False
        if "min_grade" in spec and rank < JUDGMENT_RANK.get(spec["min_grade"], 0):
            findings.append({"where": label, "ok": False,
                             "reason": f"{cell.get('judgment')} < 최소 {spec['min_grade']}"})
            failed = True
        if "max_grade" in spec and rank > JUDGMENT_RANK.get(spec["max_grade"], 5):
            findings.append({"where": label, "ok": False,
                             "reason": f"{cell.get('judgment')} > 최대 {spec['max_grade']}"})
            failed = True
        for key in ("missing", "unverified", "adopted", "directness", "verify"):
            if key in spec and cell.get(key) != spec[key]:
                findings.append({"where": label, "ok": False,
                                 "reason": f"{key}={cell.get(key)!r}, 기대 {spec[key]!r}"})
                failed = True
        for entry in _check_limitations(label, spec, cell):
            failed = failed or not entry["ok"]
            findings.append(entry)
        if not failed:
            findings.append({"where": label, "ok": True, "reason": _brief_cell(cell)})
    return findings


# 한정 단위 기대값. 값은 상태 이름이 아니라 **문언의 일부**입니다.
#
# 인덱스로 걸면 분해가 한 번 흔들릴 때 전부 어긋나고, 문언 전체로 걸면 표현이 한 글자만
# 달라져도 어긋납니다. 판정을 가르는 것은 대개 한정 안의 특정 어구 하나(여기서는
# '절대좌표계')이므로, 그 어구가 살아 있는 한 안정적으로 걸립니다.
_LIMITATION_RULES = {
    # 이 어구를 담은 한정은 그 문헌에서 **기각**되어야 합니다(개시도 미완료도 아님).
    "must_reject_contains": ("missing",),
    # 이 어구를 담은 한정은 **확정 개시**여야 합니다. 미완료는 통과가 아닙니다.
    "must_disclose_contains": ("disclosed",),
}


def _check_limitations(where: str, spec: dict, cell: dict) -> list[dict]:
    """한정 단위 기대값 채점.

    어구가 **어느 한정에도 없으면 불합격**입니다. 분해가 그 어구를 잃었다면 이 기대값은
    더 이상 아무것도 지키지 못하는데, 조용히 통과시키면 하니스가 켜져 있다고 믿게 됩니다.
    """
    findings: list[dict] = []
    states = cell.get("limitations")
    for key, allowed in _LIMITATION_RULES.items():
        phrases = spec.get(key) or []
        if not phrases:
            continue
        if states is None:
            findings.append({"where": where, "ok": False,
                             "reason": f"{key}: 관측에 한정별 상태가 없습니다 (관측을 다시 받으십시오)"})
            continue
        for phrase in phrases:
            matched = {text: state for text, state in states.items() if phrase in text}
            if not matched:
                findings.append({"where": where, "ok": False,
                                 "reason": f"{key}: '{phrase}'를 담은 한정이 분해에 없습니다"})
                continue
            offending = {text: state for text, state in matched.items() if state not in allowed}
            if offending:
                detail = "; ".join(f"{_clip_text(text)}={state}"
                                   for text, state in sorted(offending.items()))
                findings.append({"where": where, "ok": False,
                                 "reason": f"{key} '{phrase}': {detail} (기대 {'/'.join(allowed)})"})
            else:
                findings.append({"where": where, "ok": True,
                                 "reason": f"{key} '{phrase}': {len(matched)}건 모두 {'/'.join(allowed)}"})
    return findings


def _clip_text(text: str, limit: int = 40) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def normalize(element: dict | None) -> dict | None:
    """관측 스키마가 여러 벌이라 공통 형태로 맞춥니다.

    저장된 관측에는 최소 세 종류가 섞여 있습니다.
      forge    judgment·corresponded·disclosed·total·document
      sampled  disclosed_median·hits·runs·spread·stability (여러 번 돌린 통계)
      llm      judgment·corresponded·note (Forge가 아니라 모델이 직접 답한 대조군)
    없는 항목은 None으로 두고, **양쪽 다 값이 있을 때만** 비교합니다. 스키마가 다르다는
    이유로 전 셀이 '변경'으로 찍히면 무엇이 실제로 달라졌는지 묻힙니다.
    """
    if not element:
        return None
    disclosed = element.get("disclosed")
    if disclosed is None:
        disclosed = element.get("disclosed_median")
    # cells는 문헌별 단독 셀입니다. 옛 관측에는 아예 없으므로 None으로 두어 "기록이 없다"와
    # "셀이 하나도 없다"를 구별합니다 — 빈 dict로 채우면 옛 관측이 전부 "문헌별 판정이
    # 사라졌다"로 대조되고, 기대값 채점도 조용히 통과합니다.
    return {"judgment": element.get("judgment"),
            "corresponded": element.get("corresponded"),
            "disclosed": disclosed,
            "total": element.get("total"),
            "document": element.get("document"),
            "cells": element.get("cells")}


_COMPARED_FIELDS = ("judgment", "corresponded", "disclosed", "total", "document")


def _element_changed(old: dict | None, new: dict | None) -> bool:
    if (old is None) != (new is None):
        return True
    if old is None:
        return False
    return any(old[key] is not None and new[key] is not None and old[key] != new[key]
               for key in _COMPARED_FIELDS)


def diff(before: dict, after: dict) -> list[str]:
    """두 관측의 셀 단위 차이. 프롬프트를 고친 효과를 보는 주 도구입니다."""
    lines: list[str] = []
    numbers = sorted(set(before.get("claims", {})) | set(after.get("claims", {})))
    for number in numbers:
        old_claim = before.get("claims", {}).get(number, {})
        new_claim = after.get("claims", {}).get(number, {})
        for key in ("track", "primary", "secondaries", "residual", "reserved", "uncovered"):
            old_value, new_value = old_claim.get(key), new_claim.get(key)
            # 옛 관측에는 residual·uncovered가 없습니다. 없는 것을 변화로 세지 않습니다.
            if old_value is not None and new_value is not None and old_value != new_value:
                lines.append(f"  청구항 {number} {key}: {old_value} → {new_value}")
        labels = sorted(set(old_claim.get("elements", {})) | set(new_claim.get("elements", {})))
        for label in labels:
            old = normalize(old_claim.get("elements", {}).get(label))
            new = normalize(new_claim.get("elements", {}).get(label))
            if _element_changed(old, new):
                lines.append(f"  청구항 {number} ({label}): {_brief(old)} → {_brief(new)}")
            lines += _cell_diff(f"청구항 {number} ({label})", old, new)
    return lines


def _cell_diff(where: str, old: dict | None, new: dict | None) -> list[str]:
    """문헌별 단독 셀의 변화. 결합 후 행이 그대로여도 그 아래가 뒤집힐 수 있습니다.

    보조 인용발명이 메워 주던 한정을 주 인용발명이 직접 개시하게 되면(또는 그 반대로 바뀌면)
    보고서 행은 한 글자도 달라지지 않습니다. 결합의 근거가 통째로 바뀐 것이므로 회귀로
    읽어야 하는데, 행만 대조하면 보이지 않습니다.
    """
    old_cells = (old or {}).get("cells")
    new_cells = (new or {}).get("cells")
    # 옛 관측에는 cells가 없습니다. 없는 것을 변화로 세지 않습니다(normalize 참조).
    if old_cells is None or new_cells is None:
        return []
    lines: list[str] = []
    for document_id in sorted(set(old_cells) | set(new_cells)):
        before, after = old_cells.get(document_id), new_cells.get(document_id)
        if before != after:
            lines.append(f"  {where} 문헌 {document_id} 단독: "
                         f"{_brief_cell(before)} → {_brief_cell(after)}")
    return lines


def _brief(element: dict | None) -> str:
    if not element:
        return "(없음)"
    counts = (f" {element['disclosed']}/{element['total']}"
              if element["disclosed"] is not None and element["total"] is not None else "")
    document = f" doc{element['document']}" if element["document"] else ""
    return f"{element['judgment'] or '?'}{counts}{document}"


def _brief_cell(cell: dict | None) -> str:
    if not cell:
        return "(없음)"
    parts = [str(cell.get("judgment") or "?")]
    if cell.get("missing"):
        parts.append(f"누락{cell['missing']}")
    if cell.get("directness") and cell["directness"] != "direct":
        parts.append(str(cell["directness"]))
    if cell.get("adopted"):
        parts.append("채택")
    return " ".join(parts)


# --- 실행 ---------------------------------------------------------------------

def load_case(directory: pathlib.Path) -> dict:
    def read(name):
        path = directory / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def read_claims() -> str:
        """줄바꿈을 **번역하지 않고** 읽습니다.

        Path.read_text는 universal newline이라 CRLF를 LF로 바꿔 놓습니다. 분해 캐시 키가
        청구항 원문 해시라(cache.decomposition_key), 그것만으로 하니스는 앱이 저장해 둔
        분해를 영원히 못 찾습니다. 매 실행 분해를 새로 받으니 비용이 들고, 무엇보다 앱과
        다른 분해로 판정하게 되어 하니스가 재현하려던 대상 자체가 달라집니다.
        """
        path = directory / "claims.txt"
        if not path.exists():
            return ""
        with open(path, encoding="utf-8", newline="") as handle:
            return handle.read()

    return {
        "id": directory.name,
        "dir": directory,
        "case": read("case.json") or {},
        "claims_text": read_claims(),
        "decomposition": read("claim_elements.json"),
        "documents": read("documents.json") or [],
        "expected": read("expected.json") or {},
    }


def run_case(case: dict, progress=None, save: bool = True, pin: bool = False,
             slots: tuple[str, ...] | None = None) -> dict:
    """사건 하나를 동결된 문헌으로 다시 판정합니다. PDF 재추출은 하지 않습니다.

    **분해는 기본적으로 동결하지 않습니다.** pinned_decomposition을 항상 넘기면 하니스가 도는
    것은 앱이 실제로 도는 경로가 아닙니다. 청구항 분해는 매 실행 LLM이 새로 만들고, 그 결과가
    한정 문언·core/qualifier 배분·비교 캐시 키를 전부 좌우합니다. 동결해 두면 하니스는 그
    단계를 건너뛴 채 만점을 찍고 앱만 실패합니다 — 기대값과 전 항목 일치하는 사건에서 같은
    입력의 앱 실행이 인용발명을 한 건도 채택하지 못할 수 있습니다.

    pin=True는 **프롬프트 실험 전용**입니다. 분해를 고정해야 비교 프롬프트만의 효과를 볼 수
    있는 경우가 있고, 그때는 이 실행이 앱 경로가 아니라는 사실이 관측에 함께 남습니다.
    """
    from app.models import Document
    from app.pipeline import analyze, summarize_matrix

    documents = [Document.model_validate(item) for item in case["documents"]]
    # analyze가 이 dict를 제자리에서 채웁니다(claims.assign_importance). **이 회차에 실제로
    # 쓰인 분해가 남는 곳은 여기뿐입니다** — 보고서 행에는 한정 개수밖에 없어서, 이것 없이는
    # 회차 간 분해가 갈렸는지를 개수로만 짐작해야 합니다.
    decomposition: dict = {}
    result = analyze(f"regress-{case['id']}", case["claims_text"], documents,
                     case["case"].get("analysis_prompt", ""), progress,
                     decomposition=decomposition,
                     pinned_decomposition=case["decomposition"] if pin else None)
    observation = observe(result, note=f"regress {case['id']}", pinned=pin,
                          decomposition=decomposition)
    if not save:
        return observation
    # 관측은 셀 판정만 담습니다. 어느 한정이 왜 빠졌는지는 거기 없어서, 등급이 한 칸
    # 달라진 이유를 물으려면 판정을 다시 받아야 했습니다. 전체 산출물을 함께 남겨 두면
    # 진단이 공짜가 됩니다. cases/는 저장소 밖이라 원문 발췌가 새어 나가지 않습니다.
    # 슬롯 이름도 실행 종류를 따라갑니다. 관측 파일만 가르고 산출물을 같은 자리에 쓰면,
    # 핀 실행이 직전 일반 실행의 result/judgment/report를 덮어써 두 벌 중 하나가 사라집니다.
    _save_artifacts(case, result, summarize_matrix(result), slots or (_kind("latest", pin),))
    return observation


def run_case_repeatedly(case: dict, runs: int, progress=None, pin: bool = False) -> dict:
    """같은 사건을 여러 번 판정해 **판정이 얼마나 흔들리는지**를 함께 기록합니다.

    1회 관측만으로는 등급이 한 칸 달라졌을 때 그것이 회귀인지 그날의 운인지 가릴 수
    없습니다. 실제로 한 구성이 core 5개짜리라 개시 수가 하나만 달라져도 '일부 차이'와
    '일부 유사' 사이를 오갑니다. 옛 관측(sampled-*.json)이 runs·stability·spread를
    남긴 이유가 이것입니다.

    **판정 캐시를 우회합니다.** 우회하지 않으면 2회차부터는 1회차 판정을 그대로 읽어
    변동이 0으로 측정됩니다 — 측정하려는 대상을 측정 도구가 지워 버립니다. 회차마다
    빈 임시 디렉터리를 캐시로 물려, 이 모드의 결과가 평소 캐시를 오염시키지도 않습니다.

    **분해 캐시도 함께 우회합니다.** 구성대비·의미검증만 비우면 청구항 분해는
    claims.assign_importance가 공유 캐시(cache.decomposition_key)에서 그대로 읽어 옵니다.
    그러면 일반 실행 N회가 **같은 분해를 N번 재사용**하고, --pin 실행도 등록된 분해를 쓰므로
    양쪽 다 분해가 고정된 상태가 됩니다. 두 벌을 비교해 분해 단계의 기여를 분리하려던
    실험이 아무것도 재지 못하게 됩니다 — 분해는 한정 문언·core/qualifier 배분·비교 캐시
    키를 전부 좌우하므로 이 파이프라인에서 가장 큰 변동원입니다(run_case 참조).
    """
    import tempfile
    from app import cache as cache_module
    from app import entailment as entailment_module

    saved = (cache_module.CACHE_DIR, cache_module.ENTAILMENT_CACHE_DIR,
             cache_module.DECOMPOSITION_CACHE_DIR, entailment_module.CACHE_DIR)
    observations: list[dict] = []
    try:
        for index in range(runs):
            if progress:
                progress(f"안정성 측정 {index + 1}/{runs}회차")
            with tempfile.TemporaryDirectory(prefix="forge-regress-") as scratch:
                throwaway = pathlib.Path(scratch)
                cache_module.CACHE_DIR = throwaway / "compare"
                cache_module.ENTAILMENT_CACHE_DIR = throwaway / "entailment"
                cache_module.DECOMPOSITION_CACHE_DIR = throwaway / "decomposition"
                entailment_module.CACHE_DIR = throwaway / "entailment"
                for directory in (cache_module.CACHE_DIR, entailment_module.CACHE_DIR,
                                  cache_module.DECOMPOSITION_CACHE_DIR):
                    directory.mkdir(parents=True, exist_ok=True)
                # 회차마다 전체 산출물을 남깁니다. 마지막 회차만 남기면 불안정을 발견하고도
                # 어느 회차가 어떻게 달랐는지 물으려면 판정을 다시 받아야 합니다.
                slots = (_kind(f"run-{index + 1}", pin),) + (
                    (_kind("latest", pin),) if index == runs - 1 else ())
                observations.append(run_case(case, progress, pin=pin, slots=slots))
    finally:
        (cache_module.CACHE_DIR, cache_module.ENTAILMENT_CACHE_DIR,
         cache_module.DECOMPOSITION_CACHE_DIR, entailment_module.CACHE_DIR) = saved
    return aggregate(observations)


def aggregate(observations: list[dict]) -> dict:
    """여러 회차를 하나의 sampled 관측으로 합칩니다. 옛 스키마를 그대로 씁니다.

    최빈 판정을 대표로 삼되, **동률이면 낮은 등급을 씁니다.** 파이프라인이 표본 동률을
    미개시로 보는 것과 같은 방향입니다(compare.consensus). 대표값 하나만 보고 넘어가지
    않도록 spread에 분포를 그대로 남깁니다.

    **분모는 관측된 회차가 아니라 전체 회차 수입니다.** 셋 중 하나에만 나타난 구성을 그
    구성이 나타난 회차로만 나누면 1/1, 즉 완전 안정으로 집계되고 _report의 불안정 목록
    (runs > 1을 요구)에서도 통째로 빠집니다. 분해가 흔들리면 불안정이 정확히 그 형태로
    나타나므로, 그것을 지우면 이 모드는 자기가 재려던 것을 못 봅니다. 라벨과 청구항 번호도
    마지막 회차가 아니라 **전 회차의 합집합**으로 모읍니다.

    **불변식·분해 지문·핀 여부를 반드시 이월합니다.** 이월하지 않으면 _report가 "불변식
    기록 없음"으로 읽어 **--runs를 붙이는 순간 불변식 채점이 꺼집니다** — 기대값이 없어도
    도는 유일한 채점입니다. 핀 여부가 사라지면 run_case가 보장한 "이 실행이 앱 경로가
    아니라는 사실이 관측에 남는다"도 함께 사라집니다.
    """
    from collections import Counter
    from statistics import median

    if not observations:
        return {}
    base = observations[-1]
    runs = len(observations)
    claims: dict[str, dict] = {}
    for number in sorted({number for obs in observations for number in obs.get("claims", {})}):
        # 어떤 회차에 이 청구항이 통째로 없을 수 있습니다(분해 실패·판정 누락). 없는 회차를
        # 그냥 인덱싱하면 집계가 멈추므로 있는 것만 모으고, 분모는 전체 회차로 둡니다.
        present = [obs["claims"][number] for obs in observations if number in obs.get("claims", {})]
        elements: dict[str, dict] = {}
        for label in sorted({label for claim in present for label in claim.get("elements", {})}):
            seen = [claim["elements"][label] for claim in present
                    if label in claim.get("elements", {})]
            spread = Counter(item["judgment"] for item in seen)
            top = max(spread.values())
            judgment = min((name for name, count in spread.items() if count == top),
                           key=lambda name: JUDGMENT_RANK.get(name, 0))
            corresponded = sum(1 for item in seen if item["corresponded"]) * 2 > len(seen)
            # 한정 수도 회차마다 갈릴 수 있습니다. 첫 회차 값을 고정으로 쓰면 개시 수 중앙값과
            # 짝이 맞지 않는 분모가 되어 "2/8 개시"가 무엇을 뜻하는지 알 수 없게 됩니다.
            totals = Counter(item["total"] for item in seen)
            elements[label] = {
                "judgment": judgment, "hits": spread[judgment], "runs": runs,
                "observed_runs": len(seen),
                "stability": f"{spread[judgment]}/{runs}",
                "corresponded": corresponded, "spread": dict(spread),
                "disclosed_median": median(item["disclosed"] for item in seen),
                "total": totals.most_common(1)[0][0], "total_spread": dict(totals),
                "cells": _aggregate_cells(seen, runs),
            }
        tracks = Counter(claim["track"] for claim in present)
        primaries = Counter(claim["primary"] for claim in present)
        list_votes = {key: Counter(tuple(claim.get(key) or []) for claim in present)
                      for key in ("secondaries", "residual", "reserved", "uncovered")}
        claims[number] = {
            "runs": runs, "observed_runs": len(present),
            "track": tracks.most_common(1)[0][0], "track_spread": dict(tracks),
            "primary": primaries.most_common(1)[0][0], "primary_spread": dict(primaries),
            **{key: list(votes.most_common(1)[0][0]) for key, votes in list_votes.items()},
            **{f"{key}_spread": {str(list(value)): count for value, count in votes.items()}
               for key, votes in list_votes.items()},
            "elements": elements,
        }
    pinned = bool(base.get("pinned_decomposition"))
    merged = {"kind": _kind("sampled", pinned), "at": base["at"], "note": f"{runs}회 반복",
              "versions": base.get("versions"),
              "pinned_decomposition": pinned,
              "decomposition": base.get("decomposition", {}),
              "decomposition_spread": _decomposition_spread(observations),
              "claims": claims}
    violations = _merged_invariants(observations)
    if violations is not None:
        merged["invariants"] = violations
    return merged


def _aggregate_cells(seen: list[dict], runs: int) -> dict | None:
    """회차별 문헌별 셀을 하나로 합칩니다. 대표값 규칙은 구성 등급과 같습니다.

    최빈 판정을 대표로 삼되 **동률이면 낮은 등급**을 씁니다(aggregate 참조). 문헌별 셀에서
    특히 중요한 이유가 있습니다 — 결합은 셀 하나가 한 등급 흔들리면 채택 문헌 자체가
    바뀌므로, 여기서 높은 쪽을 대표로 잡으면 실제로는 회차마다 갈리는 조합이 안정된
    것처럼 보입니다.

    한 회차라도 기록이 없으면 None입니다. 없는 것을 빈 목록으로 적으면 옛 관측과 새 관측이
    구별되지 않습니다(_merged_invariants와 같은 규율).
    """
    from collections import Counter
    from statistics import median

    if any(item.get("cells") is None for item in seen):
        return None
    merged: dict[str, dict] = {}
    for document_id in sorted({key for item in seen for key in item["cells"]}):
        cells = [item["cells"][document_id] for item in seen if document_id in item["cells"]]
        spread = Counter(cell.get("judgment") for cell in cells)
        top = max(spread.values())
        judgment = min((name for name, count in spread.items() if count == top),
                       key=lambda name: JUDGMENT_RANK.get(name, 0))
        merged[document_id] = {
            "judgment": judgment,
            "hits": spread[judgment], "runs": runs,
            "stability": f"{spread[judgment]}/{runs}",
            "spread": dict(spread),
            "missing": median(cell.get("missing") or 0 for cell in cells),
            # 미완료 수는 반드시 이월합니다. 버리면 --runs 2 이상에서 unverified 기대값이
            # 채점되지 않는데, 채점되지 않는 기대값은 하니스가 켜져 있다는 착각만 만듭니다.
            "unverified": median(cell.get("unverified") or 0 for cell in cells),
            # 직접성·검증 상태도 최빈값으로 남깁니다. 등급이 같아도 이 둘이 흔들리면 보조
            # 인용발명 자격(coverage.ineligible_reason)이 회차마다 갈립니다.
            "directness": _majority((cell.get("directness") for cell in cells),
                                    DIRECTNESS_RANK),
            "verify": _majority((cell.get("verify") for cell in cells), VERIFY_RANK),
            # 채택은 과반일 때만 참으로 봅니다. 한 회차만 채택된 문헌을 채택으로 적으면
            # 결합이 흔들렸다는 사실이 대표값에서 사라집니다.
            "adopted": sum(1 for cell in cells if cell.get("adopted")) * 2 > len(cells),
            "limitations": _aggregate_limitations(cells),
        }
    return merged


# 회차마다 상태가 갈린 한정. must_reject_contains와 must_disclose_contains **어느 쪽도**
# 만족하지 않는 값이라, 불안정이 어느 방향으로도 통과로 읽히지 않습니다.
#
# 다수결이나 보수적 대표값을 쓸 수 없습니다. 낮은 상태를 대표로 잡으면 must_reject가 거짓
# 통과하고, 높은 상태를 잡으면 must_disclose가 거짓 통과합니다 — 규칙마다 안전한 방향이
# 반대라 하나의 대표값으로는 둘 다 지킬 수 없습니다.
UNSTABLE_STATE = "unstable"


def _aggregate_limitations(cells: list[dict]) -> dict | None:
    if any(cell.get("limitations") is None for cell in cells):
        return None
    merged: dict[str, str] = {}
    for text in {key for cell in cells for key in cell["limitations"]}:
        seen = {cell["limitations"].get(text) for cell in cells}
        merged[text] = seen.pop() if len(seen) == 1 else UNSTABLE_STATE
    return merged


def _majority(values, rank: dict[str, int]) -> str | None:
    """최빈값. **동률이면 보수적인 쪽**을 씁니다.

    사전순으로 정하면 direct/inferred가 1:1일 때 낙관적인 direct가 뽑힙니다. 등급 집계가
    동률에서 낮은 등급을 쓰는 것과 같은 이유로(aggregate), 회차마다 갈리는 값을 좋은 쪽으로
    대표하면 불안정이 안정으로 보입니다. 순위표를 받아 낮은 값을 씁니다.
    """
    from collections import Counter

    counted = Counter(value for value in values if value is not None)
    if not counted:
        return None
    top = max(counted.values())
    return min((name for name, count in counted.items() if count == top),
               key=lambda name: (rank.get(name, 0), name))


def _merged_invariants(observations: list[dict]) -> list[str] | None:
    """한 회차라도 깨졌으면 위반입니다. None은 "기록이 없다"는 뜻입니다.

    회차 다수결로 지우지 않습니다. 불변식은 어떤 실행에서도 참이어야 하는 성질이라 3회 중
    1회만 깨져도 깨진 것이고, 오히려 간헐적으로만 깨지는 쪽이 찾기 어려워 더 오래 남습니다.
    몇 회차에서 나왔는지는 함께 적어 재현 빈도를 알 수 있게 합니다.

    기록이 **하나라도 없는 회차**가 있으면 None을 돌려 키 자체를 만들지 않습니다. 없는 것을
    빈 목록(=위반 없음)으로 적으면 옛 관측이 전부 초록으로 보입니다 — _report가 값이 아니라
    키의 존재를 보는 것과 같은 이유입니다.
    """
    from collections import Counter

    if any("invariants" not in observation for observation in observations):
        return None
    counted: Counter[str] = Counter()
    for observation in observations:
        counted.update(observation.get("invariants") or [])
    runs = len(observations)
    return [line if count == runs else f"{line} (회차 {count}/{runs}에서만)"
            for line, count in sorted(counted.items())]


def _decomposition_spread(observations: list[dict]) -> dict:
    """회차 간 분해가 갈렸는지. 일반 실행과 --pin 실행을 비교할 때 읽어야 하는 값입니다.

    같은 사건을 일반으로 N회, 분해를 고정해 N회 돌려 두 변동 폭을 비교하면 분해 단계가 만든
    기여를 분리할 수 있습니다. 그러려면 회차마다 분해가 **실제로** 갈렸는지가 산출물에
    남아 있어야 하는데, 등급 분포만으로는 그것이 분해 차이인지 비교 차이인지 알 수 없습니다.
    """
    shapes: dict[str, list[str]] = {}
    for observation in observations:
        for number, labels in (observation.get("decomposition") or {}).items():
            for label, shape in labels.items():
                shapes.setdefault(f"청구항 {number} ({label})", []).append(shape)
    runs = len(observations)
    unstable: dict[str, list[str]] = {}
    for key, values in sorted(shapes.items()):
        if len(set(values)) > 1:
            unstable[key] = sorted(set(values))
        elif len(values) != runs:
            # 어떤 회차에는 그 구성이 아예 없었다는 뜻입니다. 분해가 라벨 자체를 다르게
            # 잘랐다는 신호라 등급이 갈린 것보다 큰 변동입니다.
            unstable[key] = [f"{len(values)}/{runs}회차에만 존재"]
    return {"stable": not unstable, "unstable": unstable}


def _save_artifacts(case: dict, result, judgment: dict,
                    slots: tuple[str, ...] = ("latest",)) -> None:
    """산출물을 지정한 슬롯마다 보존합니다.

    반복 측정에서 마지막 회차만 남기면, 불안정하다는 사실을 발견해도 **어느 회차에서 어떤
    한정·근거·오류가 달랐는지** 물으려면 판정을 다시 받아야 합니다. 회차별 산출물은 이미
    만들어져 있으므로 그것을 버리지 않는 것만으로 진단이 공짜가 됩니다.
    """
    try:
        payload = json.dumps(result.model_dump(), ensure_ascii=False)
        summary = json.dumps(judgment, ensure_ascii=False)
        # 회귀 판정만 맞고 실제 사용자 보고서가 어긋나는 퇴행도 사람이 바로 확인할 수 있게
        # 앱과 같은 렌더러로 최신 보고서를 함께 보존합니다.
        markdown = to_markdown(result)
        for slot in slots:
            directory = case["dir"] / "runs" / slot
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "result.json").write_text(payload, encoding="utf-8")
            (directory / "judgment.json").write_text(summary, encoding="utf-8")
            (directory / "report.md").write_text(markdown, encoding="utf-8")
            (directory / "report.txt").write_text(markdown, encoding="utf-8")
    except OSError:
        pass  # 진단용 부산물입니다. 저장 실패로 회귀 실행을 멈추지 않습니다.


def save_observation(case: dict, observation: dict) -> pathlib.Path:
    directory = case["dir"] / "observations"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = observation["at"].replace(":", "").replace("-", "").split(".")[0] + "Z"
    # 1회 관측(forge)과 반복 측정(sampled)은 담는 것이 달라 파일 계열을 나눕니다.
    kind = observation.get("kind", "forge")
    path = directory / f"{kind}-{stamp}.json"
    payload = json.dumps(observation, ensure_ascii=False, indent=1)
    path.write_text(payload, encoding="utf-8")
    (directory / f"{kind}-latest.json").write_text(payload, encoding="utf-8")
    return path


def adopt(job_directory: pathlib.Path, cases_dir: pathlib.Path, name: str = "") -> pathlib.Path:
    """실제 분석 결과(history/<job_id>)를 회귀 사건으로 등록합니다.

    이 파이프라인의 회귀는 언제나 "새 청구항·새 인용발명을 넣으니 또 안 된다"로 나타납니다.
    그 사건이 코퍼스에 들어오는 경로가 없으면 하니스는 계속 옛 사건만 보게 되는데,
    case.json·claims.txt·documents.json을 손으로 옮겨 적어야 한다면 아무도 등록하지 않습니다.
    등록이 한 줄이면 실패한 실행이 그대로 회귀 시험이 됩니다 — 기대값을 확정하지 않아도
    불변식 채점은 그날부터 걸립니다.

    **claims.txt는 줄바꿈을 그대로 보존합니다.** 손으로 옮기다 CRLF가 한 번 더 변환되면
    분해 캐시 키가 갈립니다 — 키가 청구항 원문 해시라서 같은 청구항이 캐시에 두 벌로
    들어가고, 서로 다른 분해가 나란히 저장된 채 하니스와 앱이 각각 다른 쪽을 씁니다.
    """
    meta = json.loads((job_directory / "meta.json").read_text(encoding="utf-8"))
    target = cases_dir / (name or job_directory.name)
    target.mkdir(parents=True, exist_ok=True)

    with open(target / "claims.txt", "w", encoding="utf-8", newline="") as handle:
        handle.write(meta.get("claims", ""))
    for filename in ("documents.json", "claim_elements.json"):
        source = job_directory / filename
        if source.exists():
            (target / filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (target / "case.json").write_text(json.dumps({
        "title": f"{job_directory.name} 실행에서 등록",
        "source_job": job_directory.name,
        "adopted_at": datetime.now(timezone.utc).isoformat(),
        "analysis_prompt": meta.get("analysis_prompt", ""),
        "documents": meta.get("documents", []),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    if not (target / "expected.json").exists():
        # 기대값은 비워 둡니다. 사람이 문헌을 읽고 확정하기 전까지 셀 채점은 하지 않지만,
        # 불변식 채점은 이 상태에서도 돕니다.
        (target / "expected.json").write_text(json.dumps({
            "adjudicated": False,
            "drafted_from": f"history/{job_directory.name}",
            "note": "실행에서 자동 등록했습니다. 문헌을 확인해 기대값을 채운 뒤 adjudicated를 "
                    "true로 바꾸십시오. 그 전에도 불변식 채점은 이 사건에 걸립니다.",
            "claims": {},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def _report(case: dict, observation: dict, previous: dict | None) -> bool:
    expected = case["expected"]
    print(f"\n=== {case['id']} — {case['case'].get('title', '')} ===")
    # 하니스 이전에 저장된 관측에는 versions가 없습니다. 그것 때문에 채점이 멈추면 예전
    # 기록을 기준선으로 쓸 수 없습니다.
    versions = observation.get("versions")
    if versions:
        print(f"    버전 {versions.get('compare')} / {versions.get('entailment')} / "
              f"분해 v{versions.get('decomposition')} / 표본 {versions.get('samples')}")
    else:
        print(f"    버전 기록 없음 (하니스 이전 관측, at={observation.get('at', '?')[:19]})")

    unstable = [(number, label, element)
                for number, claim in observation.get("claims", {}).items()
                for label, element in claim.get("elements", {}).items()
                if element.get("runs", 1) > 1 and element.get("hits") != element.get("runs")]
    # 구성 등급만 보면 **인용발명 조합이 통째로 뒤집힌 회차를 놓칩니다.** 결론을 가르는 것은
    # 셀 등급이 아니라 이쪽이고, 실측에서 셀 하나가 갈리자 주 인용발명·보조 문헌·공백 구성이
    # 회차마다 전부 달라졌습니다. spread는 이미 저장하고 있었는데 화면에 내지 않았습니다.
    swings = [(number, key, claim[f"{key}_spread"])
              for number, claim in sorted(observation.get("claims", {}).items())
              for key in ("track", "primary", "secondaries", "residual", "uncovered")
              if len(claim.get(f"{key}_spread") or {}) > 1]
    drift = (observation.get("decomposition_spread") or {}).get("unstable") or {}
    # 결합 후 행이 안정적이어도 그 아래 문헌별 셀은 갈릴 수 있습니다. 채택 문헌이 회차마다
    # 바뀌는데 등급만 같은 경우가 그렇고, 결합 근거가 통째로 달라진 것이므로 함께 냅니다.
    unstable_cells = [(number, label, document_id, cell)
                      for number, claim in observation.get("claims", {}).items()
                      for label, element in claim.get("elements", {}).items()
                      for document_id, cell in (element.get("cells") or {}).items()
                      if isinstance(cell, dict) and cell.get("runs", 1) > 1
                      and cell.get("hits") != cell.get("runs")]
    if unstable or swings or drift or unstable_cells:
        print("  [안정성] 회차마다 갈린 항목 — 이 값들의 변화를 회귀로 읽으면 안 됩니다")
        for number, label, element in unstable:
            print(f"    청구항 {number} ({label}): {element['stability']} {element['spread']}")
        for number, label, document_id, cell in unstable_cells:
            print(f"    청구항 {number} ({label}) 문헌 {document_id} 단독: "
                  f"{cell['stability']} {cell['spread']}")
        for number, key, spread in swings:
            print(f"    청구항 {number} {key}: {spread}")
        if drift:
            print("    ※ 청구항 분해가 회차마다 달라졌습니다 — 위 변화는 분해 차이일 수 있습니다")
            for key, shapes in drift.items():
                print(f"      {key}: {shapes}")
    elif str(observation.get("kind") or "").startswith("sampled"):
        print("  [안정성] 전 구성·조합·분해가 회차 간 일치")

    if observation.get("pinned_decomposition"):
        print("  ※ 분해를 고정하고 돌렸습니다. 이 실행은 앱이 실제로 도는 경로가 아닙니다 "
              "— 분해 단계의 회귀는 잡히지 않습니다.")

    if previous:
        changes = diff(previous, observation)
        print("  [직전 관측 대비]" + ("" if changes else " 변화 없음"))
        for line in changes:
            print(line)

    # 불변식은 **기대값과 무관하게** 채점합니다. 사건별 정답은 사람이 문헌을 통독해야 쓸 수
    # 있어서 사건이 좀처럼 늘지 않고, 늘지 않으면 다음 사건은 여전히 처음 보는 사건입니다.
    # 반면 아래 성질들은 어떤 청구항·어떤 문헌에서도 참이어야 하므로, 사건을 등록해 두기만
    # 하면 그날부터 채점됩니다.
    violations = observation.get("invariants") or []
    if "invariants" not in observation:
        # 불변식 기록 이전에 저장된 관측입니다. 없는 것을 "위반 없음"으로 읽으면 옛 기록이
        # 전부 초록으로 보입니다.
        print("  [불변식] 기록 없음 (불변식 도입 이전 관측). 다시 돌려야 확인됩니다.")
    elif violations:
        print("  [불변식] 사건과 무관하게 성립해야 하는 성질이 깨졌습니다")
        for line in violations:
            print(f"    !! {line}")
    else:
        print("  [불변식] 위반 없음")

    if not expected.get("adjudicated"):
        if violations:
            # 기대값이 없다고 불변식 위반까지 넘기면, 초안 상태의 사건은 무엇을 넣어도
            # 통과합니다. 사건 대부분이 초안이므로 그때 하니스는 사실상 꺼져 있습니다.
            print("  [채점] expected.adjudicated=false지만 불변식 위반이 있어 불합격입니다.")
            return False
        print("  [채점] expected.adjudicated=false — 기대값이 확정되지 않아 셀 채점은 하지 않습니다.")
        return None
    findings = score(observation, expected)
    failed = [item for item in findings if not item["ok"]]
    for item in findings:
        print(f"  {'OK ' if item['ok'] else '!! '}{item['where']}: {item['reason']}")
    print(f"  [채점] {len(findings) - len(failed)}/{len(findings)} 통과")
    return not failed and not violations


def main() -> int:
    # Windows에서 stdout이 파이프로 연결되면 기본 CP949가 긴 대시·화살표를 인코딩하지
    # 못해 결과를 한 줄도 내기 전에 종료될 수 있습니다. 회귀 결과는 UTF-8로 고정합니다.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Forge 회귀 하니스")
    parser.add_argument("--cases-dir", default=str(DEFAULT_CASES_DIR))
    parser.add_argument("--case", action="append", help="사건 id (여러 번 지정 가능)")
    parser.add_argument("--score-only", action="store_true",
                        help="LLM을 부르지 않고 저장된 최신 관측만 채점합니다 "
                             "(--pin과 함께 쓰면 핀 실행의 관측을 봅니다)")
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="관측 파일 두 개를 직접 대조합니다")
    parser.add_argument("--runs", type=int, default=1,
                        help="같은 사건을 N회 판정해 안정성을 함께 측정합니다. "
                             "2 이상이면 판정 캐시를 우회하므로 회차마다 비용이 듭니다.")
    parser.add_argument("--pin", action="store_true",
                        help="청구항 분해를 사건에 동결된 값으로 고정합니다. 비교 프롬프트만의 "
                             "효과를 볼 때만 쓰십시오 — 앱이 실제로 도는 경로가 아닙니다.")
    parser.add_argument("--adopt", metavar="JOB_DIR",
                        help="history/<job_id> 실행을 회귀 사건으로 등록하고 끝냅니다.")
    parser.add_argument("--name", default="", help="--adopt로 만들 사건 id (기본: job_id)")
    args = parser.parse_args()

    if args.adopt:
        job_directory = pathlib.Path(args.adopt)
        if not (job_directory / "meta.json").exists():
            print(f"실행 기록을 찾지 못했습니다: {job_directory}")
            return 1
        target = adopt(job_directory, pathlib.Path(args.cases_dir), args.name)
        print(f"사건을 등록했습니다: {target}")
        print("이제 `python backend/tools/regress.py --case "
              f"{target.name}`으로 돌아갑니다. 기대값을 확정하기 전에도 불변식은 채점됩니다.")
        return 0

    if args.diff:
        before, after = (json.loads(pathlib.Path(p).read_text(encoding="utf-8")) for p in args.diff)
        lines = diff(before, after)
        print("\n".join(lines) if lines else "변화 없음")
        return 0

    root = pathlib.Path(args.cases_dir)
    if not root.is_dir():
        print(f"사건 디렉터리가 없습니다: {root}")
        return 0
    directories = sorted(item for item in root.iterdir()
                         if item.is_dir() and (item / "case.json").exists())
    if args.case:
        directories = [item for item in directories if item.name in set(args.case)]
    if not directories:
        print("실행할 사건이 없습니다.")
        return 0

    # 채점한 사건과 건너뛴 사건을 반드시 갈라 셉니다. 기대값이 확정되지 않은 사건을 통과로
    # 세면, 아무것도 채점하지 않고 "전 사건 통과"를 찍는 거짓 초록이 됩니다.
    outcomes: list[bool | None] = []
    for directory in directories:
        case = load_case(directory)
        # 기준선도 실행 종류를 따라갑니다. 핀 실행을 직전 일반 실행과 대조하면 분해를
        # 고정한 효과가 회귀로 찍힙니다 — 두 실행은 애초에 다른 것을 재고 있습니다.
        latest = directory / "observations" / f"{_kind('forge', args.pin)}-latest.json"
        previous = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None
        if args.score_only:
            if previous is None:
                print(f"\n=== {case['id']} === 저장된 관측이 없습니다.")
                continue
            observation, previous = previous, None
        else:
            def progress(stage, done=None, total=None):
                print(f"    [{done or '-'}/{total or '-'}] {stage}", flush=True)

            if args.runs > 1:
                observation = run_case_repeatedly(case, args.runs, progress, pin=args.pin)
            else:
                observation = run_case(case, progress, pin=args.pin)
            save_observation(case, observation)
        outcomes.append(_report(case, observation, previous))

    scored = [item for item in outcomes if item is not None]
    skipped = len(outcomes) - len(scored)
    failed = [item for item in scored if not item]
    summary = f"\n사건 {len(outcomes)}건 — 채점 {len(scored)}건"
    if scored:
        summary += f" (통과 {len(scored) - len(failed)}, 불합격 {len(failed)})"
    if skipped:
        summary += f", 기대값 미확정으로 건너뜀 {skipped}건"
    if not scored:
        summary += ("\n※ 셀 채점된 사건이 없습니다. 불변식은 전 사건에서 돌았고 위반이 없었습니다 —"
                    " 사건과 무관한 성질만 확인된 상태입니다. 구성별 등급·인용발명 조합까지 걸려면"
                    " expected.json의 adjudicated를 true로 바꾸십시오.")
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
