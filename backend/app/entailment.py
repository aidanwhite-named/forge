"""검증된 발췌가 청구항의 원자 한정을 실제로 뒷받침하는지 독립 재심합니다.

compare 단계의 LLM은 문헌을 탐색하면서 판정까지 한 번에 수행합니다. 그 응답이 넓은 기능적
추론을 한 경우, verify.py의 문자열 대조는 인용문이 PDF에 있다는 사실만 확인할 수 있을 뿐
그 문장이 한정을 entail하는지는 확인하지 못합니다. 이 모듈은 문헌 전문을 다시 주지 않고
검증된 근거 묶음과 한정만 좁게 대조하여 최초 판정의 자기확증을 줄입니다.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from .agy import AnalysisCancelled, run_cli
from .cache import ENTAILMENT_CACHE_DIR, ENTAILMENT_KEY_PREFIX
from .config import load_runtime_settings
from .consistency import antecedent_terms
from .coverage import JUDGMENT_RANK
from .models import Claim, Document, ElementMatch, LimitationCheck, missing_limitations

PROMPT_VERSION = "entailment-v2-antecedent-terms"
CACHE_DIR = ENTAILMENT_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RELATIONS = {"explicit", "necessary_implicit", "functional_equivalent", "unsupported"}
_DIRECTNESS = {"direct", "inferred"}
_BY_RANK = {rank: judgment for judgment, rank in JUDGMENT_RANK.items()}

ENTAILMENT_PROMPT = """[역할]
당신은 최초 구성대비와 독립된 근거 심사자입니다. 최초 판정은 제공하지 않으며, 아래 원자 한정과
PDF 원문 대조를 통과한 근거 묶음만 보고 그 묶음이 한정 전체를 실제로 뒷받침하는지 판단합니다.

[판정 원칙]
- supported=true는 근거 묶음이 한정의 입력·대상·동작·출력 및 청구된 인과관계를 모두 뒷받침할
  때만 허용됩니다. 결과가 비슷하다는 문장만으로 특정 수단·조건·관계를 인정하지 마십시오.
- 한 문장에 전부 적혀 있을 필요는 없습니다. 여러 문장이 같은 알고리즘·실시 흐름의 연속 단계임이
  원문에서 드러나고, 합치면 요구된 인과 경로가 완성되는 경우 evidence를 묶어 판단할 수 있습니다.
- "A와 B를 이용하여 X"라는 한정은 A와 B가 단계적으로 서로 다른 중간 상태를 만들더라도 둘 다
  최종 X 또는 X가 사용하는 기준에 실제로 영향을 주면 충족될 수 있습니다. 다만 가능성만 있거나
  A가 X 이후의 별도 단계에만 사용되면 충족되지 않습니다.
- 청구항의 추상 명칭과 원문의 명칭이 달라도 명시된 데이터 흐름과 기능이 같으면
  functional_equivalent가 될 수 있습니다. 일반 기술상식으로 빠진 연결을 보충하지 마십시오.
- item의 reference_terms는 이 구성이 아니라 **앞선 다른 구성요소**가 도입한 지시 대상입니다
  (청구항 문언의 "상기 …"). 그 대상 자체가 이 근거 묶음에 개시되어 있는지는 여기서 묻지
  않습니다 — 별도 구성으로 이미 판정되었고, 같은 문헌 안에서 앞 구성이 대응되지 않을 때
  이 구성을 완전 개시로 세지 않는 일은 뒤의 별도 단계가 처리합니다. 이 항목에서는 지시
  대상을 **주어진 것으로 놓고**, 이 구성이 새로 더하는 수단·구조·동작과 그 인과관계를
  근거가 뒷받침하는지만 판단하십시오. 지시 대상의 명칭이 근거에 없다는 이유만으로
  supported=false를 주지 마십시오.
- relation은 explicit, necessary_implicit, functional_equivalent, unsupported 중 하나입니다.
- directness=direct는 근거 묶음 자체에 필요한 데이터 흐름이 명시된 경우입니다. 필연적 추론이
  더 필요하면 inferred입니다. 최초 판정을 올리기 위한 검토가 아니라 잘못된 개시 인정을 거르는
  검토이므로 근거가 애매하면 supported=false로 두십시오.

[출력]
JSON 객체 하나만 반환하십시오.
{"entailments": [{"item_id": "1:A:0", "supported": true,
  "relation": "functional_equivalent", "directness": "direct",
  "reason": "근거 묶음이 한정 전체를 뒷받침하거나 못하는 이유"}]}
items의 item_id를 하나도 빠짐없이 정확히 한 번씩 반환하십시오.
"""


def validate_entailment(matches: list[ElementMatch], documents: dict[str, Document],
                        cache_keys: set[str] | None = None,
                        claims: list[Claim] | None = None) -> list[str]:
    """원문 검증을 통과한 disclosed 한정만 재심하고 결과를 제자리에서 반영합니다."""
    references = _references(claims)
    grouped: dict[str, list[tuple[ElementMatch, LimitationCheck]]] = defaultdict(list)
    for match in matches:
        if match.error:
            continue
        for check in match.limitation_checks:
            if check.disclosed and _verified_bundle(check):
                grouped[match.document_id].append((match, check))

    notes: list[str] = []
    for document_id, targets in grouped.items():
        document = documents.get(document_id)
        if document is None:
            _mark_all_unchecked(targets, "의미검증 대상 문서를 찾을 수 없습니다.",
                                f"문서 ID {document_id}", notes)
            continue
        items = [_item(match, check, references) for match, check in targets]
        payload = {"document_id": document_id, "filename": document.filename, "items": items}
        try:
            raw = _cached_run(payload, cache_keys)
        except AnalysisCancelled:
            raise
        except RuntimeError as exc:
            _mark_all_unchecked(targets, f"근거 의미검증에 실패했습니다: {exc}",
                                document.filename, notes)
            continue

        returned = _returned(raw)
        touched: set[int] = set()
        for match, check in targets:
            item_id = _item_id(match, check)
            verdict = returned.get(item_id)
            if verdict is None or not isinstance(verdict.get("supported"), bool):
                _mark_unchecked(match, check, "의미검증 응답에 이 한정의 판단이 없습니다.")
                notes.append(f"청구항 {match.claim_number} ({match.label}) / {document.filename}: "
                             f"한정 {check.index}의 의미검증을 수행하지 못했습니다.")
                continue
            relation = str(verdict.get("relation") or "unsupported").strip()
            if relation not in _RELATIONS:
                relation = "unsupported"
            reason = " ".join(str(verdict.get("reason") or "").split())
            if verdict["supported"]:
                check.semantic_status = "accepted"
                check.semantic_relation = relation if relation != "unsupported" else "explicit"
                check.semantic_note = reason
                directness = str(verdict.get("directness") or "").strip()
                if directness not in _DIRECTNESS:
                    directness = "inferred" if relation == "necessary_implicit" else "direct"
                # 독립 검증은 최초 판정을 올리지 않고 상한만 씌웁니다.
                if directness == "inferred" and match.directness == "direct":
                    match.directness = "inferred"
            else:
                check.semantic_status = "rejected"
                check.semantic_relation = "unsupported"
                check.semantic_note = reason or "근거 묶음이 원자 한정 전체를 뒷받침하지 않습니다."
                check.disclosed = False
                notes.append(
                    f"청구항 {match.claim_number} ({match.label}) / {document.filename}: "
                    f"한정 {check.index} 개시를 의미검증에서 제외했습니다 ({check.semantic_note})")
            touched.add(id(match))

        for match, _ in targets:
            if id(match) in touched:
                _reconcile_match(match)
                touched.discard(id(match))
    return notes


def _verified_bundle(check: LimitationCheck) -> list:
    spans = []
    if check.quote and check.verify == "verified":
        spans.append((check.chunk_id, check.quote, check.quote_translation, check.alignment))
    spans.extend((span.chunk_id, span.quote, span.quote_translation, span.alignment)
                 for span in check.evidence if span.quote and span.verify == "verified")
    return spans


def _references(claims: list[Claim] | None) -> dict[tuple[int, str], list[str]]:
    """(청구항, 구성) → 그 구성이 앞선 구성에서 물려받은 지시 어구.

    이것을 넘기지 않으면 심사자는 지시 대상까지 이 근거 묶음이 개시해야 한다고 읽습니다.
    실측에서 "크랭크-슬라이드 기구부가 **플라이휠의** 회전 운동을 투사 스크린의 직선 왕복
    운동으로 변환함" 한정이, 모터→크랭크→피스톤→스크린 왕복을 원문 그대로 개시한 문헌에서
    "'플라이휠'에 대한 개시가 전혀 없다"는 이유로 기각됐습니다. 플라이휠은 앞 구성이 세운
    대상이고 그 구성은 이미 그 자리에서 미개시로 판정되어 있었으므로, 같은 사실이 두 구성에
    두 번 계상되면서 실제로 개시된 크랭크-슬라이드 기구부가 "대응 기재 없음"이 됐습니다.
    """
    return {(claim.number, label): terms
            for claim in claims or []
            for label, terms in antecedent_terms(claim).items()}


def _item(match: ElementMatch, check: LimitationCheck,
          references: dict[tuple[int, str], list[str]] | None = None) -> dict:
    evidence = []
    seen: set[tuple[str, str]] = set()
    for chunk_id, quote, translation, alignment in _verified_bundle(check):
        key = (chunk_id, quote)
        if key in seen:
            continue
        seen.add(key)
        evidence.append({"chunk_id": chunk_id, "original": quote,
                         "translation": translation, "alignment": alignment})
    item = {"item_id": _item_id(match, check), "element_label": match.label,
            "limitation": check.limitation, "kind": check.kind, "evidence": evidence}
    terms = (references or {}).get((match.claim_number, match.label))
    if terms:
        item["reference_terms"] = terms
    return item


def _item_id(match: ElementMatch, check: LimitationCheck) -> str:
    return f"{match.claim_number}:{match.label}:{check.index}"


def _reconcile_match(match: ElementMatch) -> None:
    """의미검증에서 빠진 한정을 누락 목록·판정 상한에 일관되게 반영합니다.

    whole_element 점검(분해 결과가 없어 구성 원문 한 줄을 통째로 본 경우)도 여기서는 그대로
    셉니다. 그 실패는 누락 '한정'이 아니라 구성 자체의 미개시라서 missing_limitations 목록에는
    오르지 않지만(models.missing_limitations), 상한 계산에서까지 빼면 "의미검증에서 제외했다"는
    노트와 '실질적 동일' 판정이 같은 셀에 함께 남습니다. 이 단계가 걸러 내려는 과대 개시가
    분해 실패한 구성에서만 그대로 통과하게 됩니다.
    """
    match.missing_limitations = missing_limitations(match.limitation_checks)
    checks = match.limitation_checks
    core = [check for check in checks if check.kind == "core"]
    rejected = [check for check in checks if check.semantic_status == "rejected"]
    if not rejected:
        return
    disclosed_core = sum(1 for check in core if check.disclosed)
    if core and disclosed_core == 0:
        cap = "차이"
        match.directness = "absent"
    elif core and disclosed_core < len(core):
        cap = "일부 유사"
        if match.directness == "direct":
            match.directness = "inferred"
    else:
        cap = "일부 차이"
    if JUDGMENT_RANK.get(match.judgment, 0) > JUDGMENT_RANK[cap]:
        match.downgraded_from = match.downgraded_from or match.judgment
        match.judgment = _BY_RANK[JUDGMENT_RANK[cap]]
    # whole_element 점검의 limitation은 구성 원문 한 줄 전체라, 차이점 줄에 그대로 실으면
    # 구성 문언을 한 번 더 읽어 주는 문장이 됩니다.
    limitations = "; ".join(check.limitation for check in rejected
                            if check.limitation and not check.whole_element)
    if limitations:
        match.reason = f"검증된 인용문만으로는 {limitations} 한정을 뒷받침하지 못하므로"
    else:
        match.reason = "검증된 인용문만으로는 이 구성의 기재를 뒷받침하지 못하므로"


def _mark_unchecked(match: ElementMatch, check: LimitationCheck, message: str) -> None:
    """의미검증을 수행하지 못한 한정. 비교 판정은 남기고 그 사실만 기록합니다.

    여기서 match.error를 세우면 chain.py가 그 청구항 전체를 analysis_incomplete로 돌려,
    같은 청구항의 다른 문헌 판정까지 함께 결론에서 빠집니다. 이 단계는 과대 개시를 걸러 내는
    보조 검증이므로 응답 결손 한 건으로 청구항의 결론을 없애지 않습니다.

    다만 verify.py는 추출 복구(alignment=recovered)에 걸던 직접성 강등을 이 단계에 넘겼습니다.
    검증이 실제로 돌지 못했다면 그 보상이 없으므로, 복구된 근거에 한해 종전의 보수적 처리를
    여기서 되살립니다.
    """
    check.semantic_status = "error"
    check.semantic_note = message
    if match.directness == "direct" and "recovered" in (check.alignment, match.alignment):
        match.directness = "inferred"


def _mark_all_unchecked(targets, message: str, where: str, notes: list[str]) -> None:
    """문헌 단위로 의미검증이 통째로 실패한 경우. 노트는 한 줄만 남깁니다."""
    for match, check in targets:
        _mark_unchecked(match, check, message)
    notes.append(f"{where}: {message} (한정 {len(targets)}건의 의미검증을 건너뛰었습니다)")


def _cached_run(payload: dict, cache_keys: set[str] | None = None) -> dict:
    settings = load_runtime_settings()
    key_payload = {"version": PROMPT_VERSION, "provider": settings["provider"],
                   "model": settings["model"], "payload": payload}
    digest = hashlib.sha256(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if cache_keys is not None:
        cache_keys.add(f"{ENTAILMENT_KEY_PREFIX}{digest}")
    path = CACHE_DIR / f"{digest}.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and _covers_all_items(value, payload):
                return value
        except (OSError, json.JSONDecodeError):
            pass
        # 항목이 빠진 응답은 다시 물어야 합니다. 남겨 두면 그 결손이 매 실행마다 재생됩니다.
        path.unlink(missing_ok=True)
    prompt = f"{ENTAILMENT_PROMPT}\nCONTEXT:\n{json.dumps(payload, ensure_ascii=False)}"
    value = run_cli(prompt, expect="entailments")
    # 온전한 응답만 캐시합니다. 비교 셀이 경고가 없을 때만 저장되는 것과 같은 규율입니다
    # (pipeline._compare_cells). 스키마만 맞고 item_id가 빠진 응답은 run_cli의 재시도에
    # 걸리지 않으므로, 여기서 막지 않으면 한 번의 결손이 캐시에 고착됩니다.
    if not _covers_all_items(value, payload):
        return value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    return value


def _returned(raw: dict) -> dict[str, dict]:
    return {str(item.get("item_id") or "").strip(): item
            for item in raw.get("entailments") or [] if isinstance(item, dict)}


def _covers_all_items(raw: dict, payload: dict) -> bool:
    """요청한 item_id마다 supported 불리언이 하나씩 돌아왔는지 확인합니다."""
    returned = _returned(raw)
    return all(isinstance((returned.get(item["item_id"]) or {}).get("supported"), bool)
               for item in payload["items"])
