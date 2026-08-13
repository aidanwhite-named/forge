"""분해 전용 반복 측정. 구성대비도 의미검증도 부르지 않습니다.

**왜 따로 만드는가.** 답하려는 질문은 "청구항 분해가 회차마다 실질적으로 얼마나 갈리는가"
하나입니다. 그 질문에 답하는 데 문헌 본문을 실은 구성대비 3표본과 의미검증까지 돌릴 이유가
없습니다. 분해는 청구항 원문만 싣는 호출 하나라, 같은 회차 수를 훨씬 싸게 잽니다.
회귀 하니스(regress.py --runs)는 전 단계를 돌리므로 이 질문에는 과합니다.

**이것은 측정 도구입니다. 런타임 게이트가 아닙니다.** 여기서 '갈렸다'로 세는 것은 결정론적
축의 불일치일 뿐, 두 분해가 의미상 다르다는 판정이 아닙니다. 문언이 한 글자 달라도 여기서는
불일치로 찍힙니다. 무엇을 실질 차이로 볼지는 사람이 이 산출물에서 분해 쌍을 얼마간 판독한
뒤에 정할 일이고, 의미축 판정자는 그 다음에 만들 것입니다. 이 도구의 출력을 그대로 사용자
확인 게이트로 쓰면, 재 보기도 전에 임계를 정하는 셈이 됩니다.

**축을 뭉쳐 세지 않습니다.** "몇 %가 갈렸다" 하나로는 게이트를 설계할 수 없습니다. 라벨이
갈린 것과 검색어 표기만 갈린 것은 하류에 닿는 방식이 다릅니다 — 앞은 구성 자체가 달라지고,
뒤는 문헌에서 읽어 오는 청크가 달라집니다(compare.select_chunks_for_claims). 그래서 축마다
따로 세고, **선택된 chunk_id 집합까지 함께 잽니다.** 검색어 차이가 실제로 읽는 근거를
바꾸는지는 그것으로만 확인됩니다.

**사용**
    python backend/tools/decompose_probe.py                 전 사건 3회
    python backend/tools/decompose_probe.py --runs 5        회차 지정
    python backend/tools/decompose_probe.py --case smart-window
    python backend/tools/decompose_probe.py --report-only   저장된 분해만 다시 집계(LLM 미호출)
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

DEFAULT_CASES_DIR = ROOT / "cases"

# 축마다 따로 셉니다. 순서는 하류에 닿는 크기순 — 라벨이 갈리면 구성 자체가 달라지고,
# 검색어가 갈리면 읽는 청크가 달라집니다.
AXES = ("labels", "limitation_count", "kinds", "importance", "alternative_groups",
        "limitation_text", "search_terms", "selected_chunks")
# 구성 라벨로 키가 잡히는 축. 라벨 집합이 달라졌을 때 이것들까지 함께 갈린 것으로 세지
# 않으려면 비교 범위를 공통 라벨로 좁혀야 합니다(compare 참조).
PER_LABEL_AXES = frozenset({"limitation_count", "kinds", "importance", "alternative_groups",
                            "limitation_text", "search_terms"})

AXIS_LABELS = {
    "labels": "라벨 집합",
    "limitation_count": "한정 수",
    "kinds": "core/qualifier 구분",
    "importance": "중요도",
    "alternative_groups": "대안 구조",
    "limitation_text": "정규화 문언 집합",
    "search_terms": "검색어 집합",
    "selected_chunks": "선택 chunk_id 집합",
}


# --- 분해 실행 -----------------------------------------------------------------

def probe(case: dict, runs: int, progress=None) -> list[dict]:
    """분해만 N회 받습니다. 회차마다 분해 캐시를 격리합니다.

    격리하지 않으면 2회차부터 1회차 분해를 그대로 읽어 변동이 0으로 측정됩니다
    (claims.assign_importance가 청구항 원문 해시로 공유 캐시를 먼저 봅니다). 회차마다 빈
    임시 디렉터리를 물리므로 **운영 캐시는 읽지도 쓰지도 않습니다** — 이 측정 때문에 앱이
    쓰던 분해가 바뀌면 안 됩니다.
    """
    from app import cache as cache_module
    from app.claims import assign_importance, decomposition_generation, parse_claims

    saved = cache_module.DECOMPOSITION_CACHE_DIR
    observations: list[dict] = []
    try:
        for index in range(runs):
            if progress:
                progress(f"    분해 {index + 1}/{runs}회차")
            with tempfile.TemporaryDirectory(prefix="forge-decompose-") as scratch:
                fresh = pathlib.Path(scratch) / "decomposition"
                fresh.mkdir(parents=True, exist_ok=True)
                cache_module.DECOMPOSITION_CACHE_DIR = fresh
                claims = parse_claims(case["claims_text"])
                started = time.monotonic()
                # pinned_decomposition=None이지만 환경변수 고정 파일(FORGE_DECOMPOSITION_FILE)이
                # 켜져 있으면 그쪽이 이깁니다. 그 상태의 측정은 무의미하므로 아래에서 경고합니다.
                warnings = assign_importance(claims, {}, case["claims_text"])
                observations.append({
                    "run": index + 1,
                    "seconds": round(time.monotonic() - started, 1),
                    "warnings": warnings,
                    "version": decomposition_generation(),
                    "claims": _dump(claims),
                    "selected_chunks": _selected_chunks(claims, case["documents"]),
                })
    finally:
        cache_module.DECOMPOSITION_CACHE_DIR = saved
    return observations


def _dump(claims) -> dict:
    """원시 분해를 그대로 보존합니다. 축 비교는 이 위에서 다시 계산합니다.

    요약만 남기면 나중에 축을 하나 더 보고 싶을 때 판정을 다시 받아야 합니다. 사람이 분해
    쌍을 판독하려면 애초에 원문이 있어야 하고, cases/는 저장소 밖이라 새어 나가지 않습니다.
    """
    return {str(claim.number): [
        {"label": element.label, "text": element.text, "importance": element.importance,
         "is_sub": element.is_sub, "is_preamble": element.is_preamble,
         "search_terms": list(element.search_terms),
         "limitations": [limitation.model_dump() for limitation in element.limitations]}
        for element in claim.elements] for claim in claims}


def _selected_chunks(claims, documents) -> dict:
    """이 분해로 문헌에서 실제로 골라 오는 청크. 결정론적이라 LLM을 부르지 않습니다.

    검색어와 한정 문언이 청크 순위를 정하므로(compare._element_terms), 분해가 갈리면 모델이
    **읽는 근거 자체**가 달라집니다. 문언 차이를 표기 흔들림으로 넘길지 판단하려면 그것이
    실제로 청크를 바꾸는지가 있어야 합니다.

    파이프라인의 일괄 비교 경로와 같은 인자를 씁니다(compare.py의 documents 조립부).
    다른 예산으로 재면 여기서 같다고 나와도 실제 실행에서는 다를 수 있습니다.
    """
    from app.compare import select_chunks_for_claims
    from app.models import Document

    parsed = [Document.model_validate(item) for item in documents]
    return {document.id: [chunk.chunk_id for chunk in
                          select_chunks_for_claims(claims, document, len(parsed))]
            for document in parsed}


# --- 축별 비교 -----------------------------------------------------------------

def _normalize(text: str) -> str:
    """문언 비교용 정규화. 공백·대소문자·따옴표류만 걷어 냅니다.

    여기서 더 걷어 내면(조사 제거·어간 추출) 서로 다른 요구사항이 같은 문자열로 접히고,
    그러면 이 측정이 "갈리지 않았다"고 보고하는 근거를 스스로 만듭니다. 판단은 사람이
    할 일이므로 도구는 **표기만** 맞춥니다.
    """
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    return re.sub(r"[\"'“”‘’()（）]", "", collapsed)


def _axis_values(elements: list[dict], chunks: dict) -> dict:
    """구성 목록 하나에서 축별 비교값을 뽑습니다. 값이 같으면 그 축은 회차 간 일치입니다."""
    by_label = {element["label"]: element for element in elements}
    return {
        "labels": sorted(by_label),
        "limitation_count": {label: len(element.get("limitations") or [])
                             for label, element in sorted(by_label.items())},
        "kinds": {label: [item.get("kind", "") for item in element.get("limitations") or []]
                  for label, element in sorted(by_label.items())},
        "importance": {label: element.get("importance")
                       for label, element in sorted(by_label.items())},
        # 대안 묶음은 이름이 아니라 **묶임의 형태**로 비교합니다. 같은 두 한정을 묶었는데
        # 그룹 이름만 다른 것을 차이로 세면 실질 변화가 그 잡음에 묻힙니다.
        "alternative_groups": {label: _group_shape(element.get("limitations") or [])
                               for label, element in sorted(by_label.items())},
        "limitation_text": {label: sorted(_normalize(item.get("text", ""))
                                          for item in element.get("limitations") or [])
                            for label, element in sorted(by_label.items())},
        "search_terms": {label: sorted({_normalize(term)
                                        for term in element.get("search_terms") or []})
                         for label, element in sorted(by_label.items())},
        "selected_chunks": {document_id: list(ids) for document_id, ids in sorted(chunks.items())},
    }


def _group_shape(limitations: list[dict]) -> list[list[int]]:
    """대안 묶음을 '어느 한정끼리 묶였는가'로만 나타냅니다(그룹 이름은 버립니다)."""
    groups: dict[str, list[int]] = {}
    for index, limitation in enumerate(limitations):
        name = str(limitation.get("alternative_group") or "")
        if name:
            groups.setdefault(name, []).append(index)
    return sorted(groups.values())


def compare(observations: list[dict]) -> dict:
    """회차 간 축별 일치 여부. 청구항 단위로 셉니다.

    **불일치는 축마다 독립으로 셉니다.** 한 축이 갈렸다고 다른 축까지 갈린 것으로 세면,
    라벨이 한 번 흔들릴 때 전 축이 함께 빨개져 어느 단계를 고쳐야 하는지 알 수 없습니다.
    """
    numbers = sorted({number for observation in observations
                      for number in observation.get("claims", {})})
    result: dict[str, dict] = {}
    for number in numbers:
        per_run = []
        for observation in observations:
            elements = observation.get("claims", {}).get(number)
            if elements is None:
                continue
            per_run.append(_axis_values(elements, observation.get("selected_chunks") or {}))
        # 라벨별 축은 **전 회차에 공통으로 있는 라벨**에서만 비교합니다. 라벨이 하나
        # 늘거나 줄면 라벨별 dict의 키 집합이 달라져 한정 수·중요도·검색어가 전부 '갈림'으로
        # 찍히는데, 실제로 달라진 것은 라벨 집합 하나입니다. 그 사실은 labels 축이 이미
        # 말하고 있고, 여기서 또 세면 축을 나눈 의미가 사라집니다.
        common = set(per_run[0]["labels"]) if per_run else set()
        for values in per_run[1:]:
            common &= set(values["labels"])
        axes: dict[str, dict] = {}
        for axis in AXES:
            seen = [json.dumps(_restrict(axis, values[axis], common),
                               ensure_ascii=False, sort_keys=True) for values in per_run]
            variants = sorted(set(seen))
            axes[axis] = {
                "stable": len(variants) <= 1,
                "variants": len(variants),
                "runs": len(observations),
                "observed_runs": len(per_run),
                # 공통 라벨이 줄면 라벨별 축의 분모도 줄어듭니다. 그 사실을 적지 않으면
                # "한정 수는 안 갈렸다"가 비교한 구성이 둘뿐이어서 나온 결과일 수 있습니다.
                "compared_labels": len(common) if axis in PER_LABEL_AXES else None,
            }
        # 어느 구성에서 갈렸는지까지 남깁니다. 청구항 단위 참/거짓만으로는 한 구성 때문인지
        # 전면적인지 구별되지 않고, 그 둘은 대응이 다릅니다.
        axes_by_label = {}
        for axis in PER_LABEL_AXES:
            unstable = []
            for key in sorted(common):
                seen = [json.dumps(values[axis].get(key), ensure_ascii=False, sort_keys=True)
                        for values in per_run]
                if len(set(seen)) > 1:
                    unstable.append(key)
            if unstable:
                axes_by_label[axis] = unstable
        result[number] = {"axes": axes, "unstable_labels": axes_by_label,
                          "common_labels": sorted(common)}
    return result


def _restrict(axis: str, values, labels: set[str]):
    """라벨별 축은 공통 라벨로 좁히고, 나머지는 그대로 둡니다."""
    return {key: value for key, value in values.items()
            if key in labels} if axis in PER_LABEL_AXES else values


# --- 실행 ---------------------------------------------------------------------

def _fingerprint(case: dict) -> str:
    """청구항 원문 해시. 같은 청구항을 두 사건으로 등록해 둔 것을 하나로 셉니다.

    실측에서 neareye-live와 neareye-waveguide가 같은 청구항·같은 문헌으로 등록돼 있어,
    "채점 2건 통과"가 사실은 한 사건을 두 번 센 결과였습니다. 불일치율을 그렇게 세면
    중복된 사건의 성질이 전체 통계를 두 배로 끌고 갑니다.

    **공백을 정규화한 뒤 해싱합니다.** 그 두 사건의 원문은 439자와 444자로, 띄어쓰기와
    줄바꿈만 다릅니다. 원문 그대로 해싱하면 중복이 걸리지 않아 이 함수가 막으려던 이중
    계상이 그대로 일어납니다.

    다만 **파이프라인은 정규화하지 않습니다**(cache.decomposition_key는 원문을 그대로
    해싱합니다). 즉 이 둘은 같은 청구항이면서 분해 캐시에는 두 벌로 들어갑니다 — 사람이
    보기에 같은 사건이 도구에게는 다른 입력인 상태입니다. 통계에서 하나로 세는 것과
    캐시에서 하나로 합치는 것은 다른 문제이므로 여기서는 통계만 바로잡습니다.
    """
    collapsed = re.sub(r"\s+", " ", case["claims_text"] or "").strip()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()[:12]


def _save(case: dict, observations: list[dict], comparison: dict) -> pathlib.Path:
    directory = case["dir"] / "decompositions"
    directory.mkdir(parents=True, exist_ok=True)
    for observation in observations:
        (directory / f"run-{observation['run']}.json").write_text(
            json.dumps(observation, ensure_ascii=False, indent=1), encoding="utf-8")
    payload = {"at": datetime.now(timezone.utc).isoformat(),
               "runs": len(observations), "comparison": comparison}
    (directory / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return directory


def _load_saved(case: dict) -> list[dict]:
    directory = case["dir"] / "decompositions"
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("run-*.json"))] if directory.is_dir() else []


def _report(case: dict, observations: list[dict], comparison: dict) -> None:
    print(f"\n=== {case['id']} — {case['case'].get('title', '')} ===")
    versions = {observation.get("version") for observation in observations}
    seconds = [observation.get("seconds") for observation in observations
               if observation.get("seconds") is not None]
    print(f"    분해 세대 {'/'.join(sorted(str(v) for v in versions))}"
          + (f" · 회차당 {min(seconds)}~{max(seconds)}초" if seconds else ""))
    for observation in observations:
        for warning in observation.get("warnings") or []:
            print(f"    ! {observation['run']}회차: {warning[:120]}")
    for number, entry in sorted(comparison.items()):
        print(f"  청구항 {number}")
        for axis in AXES:
            stat = entry["axes"][axis]
            mark = "일치" if stat["stable"] else f"갈림({stat['variants']}종)"
            labels = entry["unstable_labels"].get(axis)
            detail = f" — {', '.join(labels)}" if labels else ""
            print(f"    {'OK ' if stat['stable'] else '!! '}{AXIS_LABELS[axis]:18} {mark}{detail}")


def _summary(rows: list[tuple[str, dict]]) -> None:
    print("\n--- 축별 불일치 (중복 제거 후 청구항 단위) ---")
    totals: dict[str, list[int]] = {axis: [0, 0] for axis in AXES}
    for _, comparison in rows:
        for entry in comparison.values():
            for axis in AXES:
                totals[axis][1] += 1
                if not entry["axes"][axis]["stable"]:
                    totals[axis][0] += 1
    for axis in AXES:
        broken, total = totals[axis]
        share = f"{broken / total * 100:.0f}%" if total else "-"
        print(f"  {AXIS_LABELS[axis]:18} {broken}/{total} 갈림 ({share})")
    print("\n※ 이 수치는 결정론적 축의 불일치입니다. 두 분해가 **의미상** 다르다는 판정이"
          " 아닙니다 — 문언이 한 글자만 달라도 갈림으로 찍힙니다. 실질 차이의 기준은 사람이"
          " decompositions/run-*.json에서 분해 쌍을 판독한 뒤에 정할 일이고, 의미축 판정자는"
          " 그 다음입니다. 이 출력을 그대로 사용자 확인 게이트로 쓰지 마십시오.")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="분해 전용 반복 측정")
    parser.add_argument("--cases-dir", default=str(DEFAULT_CASES_DIR))
    parser.add_argument("--case", action="append", help="사건 id (여러 번 지정 가능)")
    parser.add_argument("--runs", type=int, default=3, help="사건당 분해 횟수 (기본 3)")
    parser.add_argument("--report-only", action="store_true",
                        help="LLM을 부르지 않고 저장된 분해만 다시 집계합니다")
    args = parser.parse_args()

    from app.claims import DECOMPOSITION_FILE
    if DECOMPOSITION_FILE and not args.report_only:
        print(f"FORGE_DECOMPOSITION_FILE이 설정되어 있습니다({DECOMPOSITION_FILE}). "
              "고정 분해가 LLM 호출을 밀어내 변동이 0으로 측정됩니다. 해제하고 다시 돌리십시오.")
        return 1

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from regress import load_case                                        # noqa: E402

    root = pathlib.Path(args.cases_dir)
    if not root.is_dir():
        print(f"사건 디렉터리가 없습니다: {root}")
        return 0
    directories = sorted(item for item in root.iterdir()
                         if item.is_dir() and (item / "case.json").exists())
    if args.case:
        directories = [item for item in directories if item.name in set(args.case)]

    rows: list[tuple[str, dict]] = []
    seen: dict[str, str] = {}
    for directory in directories:
        case = load_case(directory)
        digest = _fingerprint(case)
        if digest in seen:
            print(f"\n=== {case['id']} === 청구항 원문이 {seen[digest]}와 같아 건너뜁니다"
                  " (불일치율을 두 번 세지 않습니다).")
            continue
        seen[digest] = case["id"]

        observations = _load_saved(case) if args.report_only else probe(
            case, args.runs, progress=lambda line: print(line, flush=True))
        if not observations:
            print(f"\n=== {case['id']} === 저장된 분해가 없습니다.")
            continue
        comparison = compare(observations)
        if not args.report_only:
            _save(case, observations, comparison)
        _report(case, observations, comparison)
        rows.append((case["id"], comparison))

    if rows:
        _summary(rows)
    print(f"\n사건 {len(rows)}건 측정 (중복 제외).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
