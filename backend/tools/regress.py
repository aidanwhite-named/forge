"""회귀 하니스. 동결된 사건들에 대해 판정을 다시 받고 기대값과 대조합니다.

**왜 필요한가.** 이 파이프라인에서 판정 품질을 바꾸는 것은 대부분 프롬프트 문구입니다.
그런데 프롬프트를 고친 효과는 셀 하나를 눈으로 확인하는 것으로는 알 수 없습니다. 실제로
세 번 연속으로 "관측한 셀 하나를 고치자 다른 셀이 깨지는" 일이 났습니다. 고친 셀만 보고
넘어갔기 때문입니다. 한 번 고칠 때마다 **전 사건 × 전 구성**을 다시 채점해야 합니다.

**사건 데이터는 저장소에 없습니다.** cases/ 에는 분석 대상 청구항 원문과 인용문헌 추출
본문이 들어 있어 공개 저장소에 올리지 않습니다(.gitignore). 하니스 코드만 커밋하고 데이터는
각자 로컬에 둡니다. 사건이 없으면 이 스크립트는 조용히 아무것도 하지 않습니다.

**사건 한 건의 구성**
    case.json            제목·원본 job·문헌 목록·그때의 analysis_prompt
    claims.txt           청구항 원문
    claim_elements.json  분해 결과(동결). 이것을 고정해야 프롬프트 효과만 남습니다.
    documents.json       PDF 추출본(동결). 재추출하면 추출 로직 변경이 섞입니다.
    expected.json        사람이 판정한 기대값. adjudicated=false면 채점에서 뺍니다.
    observations/        실행 스냅샷. -latest 별칭과 타임스탬프본을 함께 남깁니다.

**사용**
    python backend/tools/regress.py                     전 사건 실행·채점
    python backend/tools/regress.py --case smart-window 한 사건만
    python backend/tools/regress.py --score-only        저장된 최신 관측만 채점(LLM 미호출)
    python backend/tools/regress.py --diff A.json B.json 두 관측 대조
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

DEFAULT_CASES_DIR = ROOT / "cases"
# 등급 서열은 앱과 같은 표를 씁니다. 하니스가 자기 사다리를 들면 같은 등급을 두 곳이
# 다르게 읽습니다.
from app.coverage import JUDGMENT_RANK           # noqa: E402
from app.report import to_markdown                # noqa: E402


# --- 관측 ---------------------------------------------------------------------

def observe(result, kind: str = "forge", note: str = "") -> dict:
    """AnalysisResult를 사건 간 비교가 가능한 최소 형태로 줄입니다.

    보고서 문장이나 발췌는 담지 않습니다. 그것까지 넣으면 표현이 조금 바뀔 때마다 관측이
    달라져 무엇이 실제로 변했는지 묻히고, 문헌 원문이 관측 파일로 새어 나갑니다.
    """
    claims: dict[str, dict] = {}
    for report in result.reports:
        elements = {}
        for item in report.claims:
            elements[item.label] = {
                "judgment": item.grade if item.grade in JUDGMENT_RANK else _grade_of(item),
                "corresponded": item.corresponded,
                "disclosed": item.disclosed_limitations,
                "total": item.total_limitations,
                "document": item.adopted_document or "",
            }
        claims[str(report.claim_number)] = {
            "track": report.track,
            "primary": report.chain.primary or "",
            "secondaries": list(report.chain.secondaries),
            "residual": list(report.chain.residual),
            "uncovered": list(report.chain.uncovered),
            "elements": elements,
        }
    return {"kind": kind, "at": datetime.now(timezone.utc).isoformat(), "note": note,
            "versions": _versions(), "claims": claims}


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
    from app.cache import PROMPT_VERSION as compare_version
    from app.claims import DECOMPOSITION_VERSION
    from app.entailment import PROMPT_VERSION as entailment_version
    from app.config import COMPARE_SAMPLES, load_runtime_settings
    settings = load_runtime_settings()
    return {"compare": compare_version, "entailment": entailment_version,
            "decomposition": DECOMPOSITION_VERSION, "samples": COMPARE_SAMPLES,
            "provider": settings["provider"], "model": settings["model"]}


# --- 채점 ---------------------------------------------------------------------

def score(observation: dict, expected: dict) -> list[dict]:
    """기대값과 대조한 결과. expected에 적히지 않은 구성은 채점하지 않습니다.

    min_grade/max_grade는 **범위**입니다. 정확한 등급을 요구하면 사람이 확정하지 못한
    경계(예: '일부 유사'인지 '일부 차이'인지)까지 하니스가 강제하게 되고, 그러면 기대값을
    실제 출력에 맞춰 고치는 일이 반복됩니다.
    """
    findings: list[dict] = []
    if "decomposition_version" in expected:
        actual_version = (observation.get("versions") or {}).get("decomposition")
        wanted_version = expected["decomposition_version"]
        findings.append({
            "where": "분해 버전",
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
        # 밑줄을 붙여 적습니다. 인용발명 조합이 바뀌는 회귀가 실제로 났으므로 걸 수 있게 둡니다.
        #
        # _secondaries가 필요한 이유: 실측에서 부 인용발명이 **한 건도** 채택되지 않는 회귀가
        # 났는데, 구성별 등급은 주 인용발명만으로도 그대로라 셀 채점으로는 전혀 드러나지
        # 않았습니다. 조합 자체를 걸 수 있어야 잡힙니다.
        #
        # 키가 있으면 값이 비어 있어도 채점합니다. `if wanted`로 걸러 내면 "_secondaries": []
        # 즉 "결합이 서면 안 된다"는 기대를 아예 적을 수 없습니다.
        for key in ("track", "primary", "secondaries", "residual", "uncovered"):
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
    if not findings:
        findings.append({"where": where, "ok": True, "reason": _brief(actual)})
    return findings


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
    return {"judgment": element.get("judgment"),
            "corresponded": element.get("corresponded"),
            "disclosed": disclosed,
            "total": element.get("total"),
            "document": element.get("document")}


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
        for key in ("track", "primary", "secondaries", "residual", "uncovered"):
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
    return lines


def _brief(element: dict | None) -> str:
    if not element:
        return "(없음)"
    counts = (f" {element['disclosed']}/{element['total']}"
              if element["disclosed"] is not None and element["total"] is not None else "")
    document = f" doc{element['document']}" if element["document"] else ""
    return f"{element['judgment'] or '?'}{counts}{document}"


# --- 실행 ---------------------------------------------------------------------

def load_case(directory: pathlib.Path) -> dict:
    def read(name):
        path = directory / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return {
        "id": directory.name,
        "dir": directory,
        "case": read("case.json") or {},
        "claims_text": (directory / "claims.txt").read_text(encoding="utf-8")
        if (directory / "claims.txt").exists() else "",
        "decomposition": read("claim_elements.json"),
        "documents": read("documents.json") or [],
        "expected": read("expected.json") or {},
    }


def run_case(case: dict, progress=None, save: bool = True) -> dict:
    """사건 하나를 동결된 문헌·분해로 다시 판정합니다. PDF 재추출은 하지 않습니다."""
    from app.models import Document
    from app.pipeline import analyze, summarize_matrix

    documents = [Document.model_validate(item) for item in case["documents"]]
    result = analyze(f"regress-{case['id']}", case["claims_text"], documents,
                     case["case"].get("analysis_prompt", ""), progress,
                     decomposition={}, pinned_decomposition=case["decomposition"])
    observation = observe(result, note=f"regress {case['id']}")
    if not save:
        return observation
    # 관측은 셀 판정만 담습니다. 어느 한정이 왜 빠졌는지는 거기 없어서, 등급이 한 칸
    # 달라진 이유를 물으려면 판정을 다시 받아야 했습니다. 전체 산출물을 함께 남겨 두면
    # 진단이 공짜가 됩니다. cases/는 저장소 밖이라 원문 발췌가 새어 나가지 않습니다.
    _save_artifacts(case, result, summarize_matrix(result))
    return observation


def run_case_repeatedly(case: dict, runs: int, progress=None) -> dict:
    """같은 사건을 여러 번 판정해 **판정이 얼마나 흔들리는지**를 함께 기록합니다.

    1회 관측만으로는 등급이 한 칸 달라졌을 때 그것이 회귀인지 그날의 운인지 가릴 수
    없습니다. 실제로 한 구성이 core 5개짜리라 개시 수가 하나만 달라져도 '일부 차이'와
    '일부 유사' 사이를 오갑니다. 옛 관측(sampled-*.json)이 runs·stability·spread를
    남긴 이유가 이것입니다.

    **판정 캐시를 우회합니다.** 우회하지 않으면 2회차부터는 1회차 판정을 그대로 읽어
    변동이 0으로 측정됩니다 — 측정하려는 대상을 측정 도구가 지워 버립니다. 회차마다
    빈 임시 디렉터리를 캐시로 물려, 이 모드의 결과가 평소 캐시를 오염시키지도 않습니다.
    """
    import tempfile
    from app import cache as cache_module
    from app import entailment as entailment_module

    saved = (cache_module.CACHE_DIR, cache_module.ENTAILMENT_CACHE_DIR,
             entailment_module.CACHE_DIR)
    observations: list[dict] = []
    try:
        for index in range(runs):
            if progress:
                progress(f"안정성 측정 {index + 1}/{runs}회차")
            with tempfile.TemporaryDirectory(prefix="forge-regress-") as scratch:
                throwaway = pathlib.Path(scratch)
                cache_module.CACHE_DIR = throwaway / "compare"
                cache_module.ENTAILMENT_CACHE_DIR = throwaway / "entailment"
                entailment_module.CACHE_DIR = throwaway / "entailment"
                for directory in (cache_module.CACHE_DIR, entailment_module.CACHE_DIR):
                    directory.mkdir(parents=True, exist_ok=True)
                observations.append(run_case(case, progress, save=index == runs - 1))
    finally:
        (cache_module.CACHE_DIR, cache_module.ENTAILMENT_CACHE_DIR,
         entailment_module.CACHE_DIR) = saved
    return aggregate(observations)


def aggregate(observations: list[dict]) -> dict:
    """여러 회차를 하나의 sampled 관측으로 합칩니다. 옛 스키마를 그대로 씁니다.

    최빈 판정을 대표로 삼되, **동률이면 낮은 등급을 씁니다.** 파이프라인이 표본 동률을
    미개시로 보는 것과 같은 방향입니다(compare.consensus). 대표값 하나만 보고 넘어가지
    않도록 spread에 분포를 그대로 남깁니다.
    """
    from collections import Counter
    from statistics import median

    if not observations:
        return {}
    base = observations[-1]
    claims: dict[str, dict] = {}
    for number, claim in base.get("claims", {}).items():
        elements: dict[str, dict] = {}
        for label in claim.get("elements", {}):
            seen = [obs["claims"][number]["elements"][label]
                    for obs in observations
                    if label in obs.get("claims", {}).get(number, {}).get("elements", {})]
            spread = Counter(item["judgment"] for item in seen)
            top = max(spread.values())
            judgment = min((name for name, count in spread.items() if count == top),
                           key=lambda name: JUDGMENT_RANK.get(name, 0))
            corresponded = sum(1 for item in seen if item["corresponded"]) * 2 > len(seen)
            elements[label] = {
                "judgment": judgment, "hits": spread[judgment], "runs": len(seen),
                "stability": f"{spread[judgment]}/{len(seen)}",
                "corresponded": corresponded, "spread": dict(spread),
                "disclosed_median": median(item["disclosed"] for item in seen),
                "total": seen[0]["total"],
            }
        tracks = Counter(obs["claims"][number]["track"] for obs in observations)
        primaries = Counter(obs["claims"][number]["primary"] for obs in observations)
        list_votes = {
            key: Counter(tuple(obs["claims"][number].get(key) or []) for obs in observations)
            for key in ("secondaries", "residual", "uncovered")
        }
        claims[number] = {
            "track": tracks.most_common(1)[0][0], "track_spread": dict(tracks),
            "primary": primaries.most_common(1)[0][0], "primary_spread": dict(primaries),
            **{key: list(votes.most_common(1)[0][0]) for key, votes in list_votes.items()},
            **{f"{key}_spread": {str(list(value)): count for value, count in votes.items()}
               for key, votes in list_votes.items()},
            "elements": elements,
        }
    return {"kind": "sampled", "at": base["at"], "note": f"{len(observations)}회 반복",
            "versions": base.get("versions"), "claims": claims}


def _save_artifacts(case: dict, result, judgment: dict) -> None:
    directory = case["dir"] / "runs" / "latest"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        (directory / "result.json").write_text(
            json.dumps(result.model_dump(), ensure_ascii=False), encoding="utf-8")
        (directory / "judgment.json").write_text(
            json.dumps(judgment, ensure_ascii=False), encoding="utf-8")
        # 회귀 판정만 맞고 실제 사용자 보고서가 어긋나는 퇴행도 사람이 바로 확인할 수 있게
        # 앱과 같은 렌더러로 최신 보고서를 함께 보존합니다.
        markdown = to_markdown(result)
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
    if unstable:
        print("  [안정성] 회차마다 판정이 갈린 구성 — 이 셀들은 등급 변화를 회귀로 읽으면 안 됩니다")
        for number, label, element in unstable:
            print(f"    청구항 {number} ({label}): {element['stability']} {element['spread']}")
    elif observation.get("kind") == "sampled":
        print("  [안정성] 전 구성이 회차 간 일치")

    if previous:
        changes = diff(previous, observation)
        print("  [직전 관측 대비]" + ("" if changes else " 변화 없음"))
        for line in changes:
            print(line)

    if not expected.get("adjudicated"):
        print("  [채점] expected.adjudicated=false — 기대값이 확정되지 않아 채점하지 않습니다.")
        return None
    findings = score(observation, expected)
    failed = [item for item in findings if not item["ok"]]
    for item in findings:
        print(f"  {'OK ' if item['ok'] else '!! '}{item['where']}: {item['reason']}")
    print(f"  [채점] {len(findings) - len(failed)}/{len(findings)} 통과")
    return not failed


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
                        help="LLM을 부르지 않고 저장된 forge-latest.json만 채점합니다")
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="관측 파일 두 개를 직접 대조합니다")
    parser.add_argument("--runs", type=int, default=1,
                        help="같은 사건을 N회 판정해 안정성을 함께 측정합니다. "
                             "2 이상이면 판정 캐시를 우회하므로 회차마다 비용이 듭니다.")
    args = parser.parse_args()

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
        latest = directory / "observations" / "forge-latest.json"
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
                observation = run_case_repeatedly(case, args.runs, progress)
            else:
                observation = run_case(case, progress)
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
        summary += "\n※ 채점된 사건이 없습니다. expected.json의 adjudicated를 true로 바꾸기 전에는 "
        summary += "이 실행이 회귀를 잡아 주지 않습니다."
    print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
