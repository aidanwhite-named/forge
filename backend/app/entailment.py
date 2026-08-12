"""검증된 발췌가 청구항의 원자 한정을 실제로 뒷받침하는지 독립 재심합니다.

compare 단계의 LLM은 문헌을 탐색하면서 판정까지 한 번에 수행합니다. 그 응답이 넓은 기능적
추론을 한 경우, verify.py의 문자열 대조는 인용문이 PDF에 있다는 사실만 확인할 수 있을 뿐
그 문장이 한정을 entail하는지는 확인하지 못합니다. 이 모듈은 문헌 전문을 다시 주지 않고
검증된 근거 묶음과 한정만 좁게 대조하여 최초 판정의 자기확증을 줄입니다.
"""
import hashlib
import json
from collections import defaultdict
from .agy import AnalysisCancelled, run_cli
from .cache import ENTAILMENT_CACHE_DIR, ENTAILMENT_KEY_PREFIX
from .config import load_runtime_settings
from .consistency import antecedent_terms
from .coverage import JUDGMENT_RANK, derive_judgment, judgment_at_rank
from .models import Claim, Document, ElementMatch, LimitationCheck, missing_limitations
from .pdf import chunk_text

# v9: 실제 입력 제시와 특정 시스템을 통한 촬영/출력이 같은 실시 흐름으로 명시되면 인과
#     경로를 인정하되, 장치 유형 총론 하나만으로는 계속 인정하지 않습니다.
# v8: 주체 정체성 검사를 해당 원자 한정의 범위에만 적용합니다. 단순 NN 학습 한정을 별도
#     광학 모델링 한정의 실패로 함께 기각하지 않고, 제시한 입력측 영상과 대응 촬영 결과의
#     세트는 영상쌍 획득으로 인정하되 특정 도파관 인과 한정과는 분리합니다.
# v7: 같은 청구 주체의 속성을 서로 다른 모델·부품에서 합치는 것을 막고 입력→출력 방향을
#     별도 축으로 확인합니다. 광학 프록시의 모델링 역할과 별도 HoloNet의 뉴럴 네트워크
#     정체성이 합쳐지고, target→phase 흐름이 correction→target으로 뒤집힌 실측을 막습니다.
# v6: 네 축을 독립 판정하도록 바꾸고, 배경기술·요약·연구 동기·향후 적용 가능성을 개시 근거에서
#     뺐습니다. v5는 두 방향으로 어긋났습니다 — 문헌이 '획득'을 명시한 한정을 동작 축 결손으로
#     기각하는 한편, 서론의 과제 서술을 다른 한정의 근거로 통과시켰습니다.
# v5: 역할 대조가 명칭을 넘어 **동작·대상·집합성·인과관계**까지 지우는 것을 막습니다. 실측에서
#     "광원이 이미지 광을 생성해 도파관으로 출력한다"는 장치 정상 동작 기재가 "입력 광학
#     이미지들의 세트를 획득함" 한정의 근거로 통과해, 그 구성이 '실질적 동일 4/4'로 나갔습니다.
#     같은 유형을 이 단계가 다른 셀에서는 세 건 기각했으므로 규칙의 부재가 아니라 경계의 부재였습니다.
# v4: 같은 구성요소의 다른 한정이 제출한 검증된 인용문을 element_context로 함께 넘깁니다.
#     최초 대비가 한 인과 사슬을 형제 한정끼리 쪼개 담으면, 한정별로만 읽는 이 단계가 실제로
#     개시된 한정을 근거 없음으로 기각했습니다.
#     함께, 명칭이 아니라 역할로 대조하도록 못박았습니다. 종전 심사자는 청구항 전용 명칭
#     ("스위치 박스"·"제1 리미트 스위치")이 원문에 없다는 것만으로 기각해, compare 단계가
#     역할로 옳게 찾아낸 대응을 이 단계가 그대로 거부했습니다.
PROMPT_VERSION = "entailment-v9-causal-path-evidence"
CACHE_DIR = ENTAILMENT_CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 인용문이 속한 단락의 국소 문맥만 의미검증에 제공합니다. 특허 단락은 보통 이보다 훨씬
# 짧지만, 비정상적으로 큰 청크 하나가 검증 프롬프트 대부분을 차지하지 않도록 상한을 둡니다.
MAX_SOURCE_CONTEXT_CHARS = 12000

_RELATIONS = {"explicit", "necessary_implicit", "functional_equivalent", "unsupported"}
_DIRECTNESS = {"direct", "inferred"}

ENTAILMENT_PROMPT = """[역할]
당신은 최초 구성대비와 독립된 근거 심사자입니다. 최초 판정은 제공하지 않으며, 아래 원자 한정과
PDF 원문 대조를 통과한 근거 묶음만 보고 그 묶음이 한정 전체를 실제로 뒷받침하는지 판단합니다.

[판정 원칙]
- supported=true는 근거 묶음이 한정의 입력·대상·동작·출력 및 청구된 인과관계를 모두 뒷받침할
  때만 허용됩니다. 결과가 비슷하다는 문장만으로 특정 수단·조건·관계를 인정하지 마십시오.
- 한 문장에 전부 적혀 있을 필요는 없습니다. 여러 문장이 같은 알고리즘·실시 흐름의 연속 단계임이
  원문에서 드러나고, 합치면 요구된 인과 경로가 완성되는 경우 evidence를 묶어 판단할 수 있습니다.
- evidence의 source_context는 원문 대조를 통과한 인용문이 속한 **동일 단락 또는 동일 국소 청크의
  PDF 추출문**입니다. 대표 인용문이 짧더라도 source_context 안에서 대상·동작·수단과 그 연결이
  명시되면 함께 근거로 사용할 수 있습니다. 다만 그 국소 문맥에 없는 연결을 일반 기술상식으로
  보충하거나, 서로 다른 실시예인 문장을 임의로 결합해서는 안 됩니다.
- element_context는 **같은 구성요소의 다른 한정**에 대해 제출되어 원문 대조를 통과한 인용문입니다.
  최초 대비가 하나의 인과 사슬을 형제 한정끼리 나눠 담는 일이 있어, 그 때문에 실제로 개시된
  한정이 근거 없는 것으로 처리되는 것을 막기 위해 함께 제공합니다. evidence와 **같은 실시 흐름**
  임이 원문에서 드러날 때만 인과 경로를 잇는 데 사용하고, 그렇게 사용했다면 어느 문장을 썼는지
  reason에 밝히십시오. 다른 실시예이거나 연결이 원문에 없으면 사용하지 마십시오.
  **element_context만으로는 supported=true를 줄 수 없습니다.** 그 한정 자신의 evidence가 대상 또는
  동작 중 적어도 하나를 직접 뒷받침해야 합니다.
- "A와 B를 이용하여 X"라는 한정은 A와 B가 단계적으로 서로 다른 중간 상태를 만들더라도 둘 다
  최종 X 또는 X가 사용하는 기준에 실제로 영향을 주면 충족될 수 있습니다. 다만 가능성만 있거나
  A가 X 이후의 별도 단계에만 사용되면 충족되지 않습니다.
- **명칭이 아니라 역할로 대조하십시오.** 한정에 등장하는 명칭은 그 출원인이 자기 구성에 붙인
  이름일 뿐이고, 인용문헌은 같은 것을 저마다 다른 이름(부·부재·모듈·엔진·유닛·회로)으로
  부릅니다. 판단 전에 한정을 "무엇이 어떤 조건에서 무엇에 대해 무엇을 하는가"로 바꿔 적고,
  근거가 **그 역할**을 뒷받침하는지 보십시오. 데이터·신호·물질·동력의 흐름과 기능이 같으면
  명칭이 달라도 functional_equivalent입니다.
- **그 명칭이 근거에 없다는 것만으로 supported=false를 주지 마십시오.** 그것은 용어 차이이지
  근거의 결손이 아닙니다. 기각하려면 근거에 없는 것이 명칭이 아니라 **역할 자체**(대상·동작
  또는 둘을 잇는 인과관계)여야 하고, reason에는 어느 역할이 비어 있는지 적으십시오.
  "청구항이 말하는 '○○부'가 개시되어 있지 않다"처럼 명칭의 부재만 사유로 적었다면 잘못
  기각한 것입니다. 그 역할을 하는 구성이 다른 이름으로 있는지 다시 확인하십시오.
- 다만 역할이 실제로 다르면 기각하십시오. 대상이 다르거나, 동작을 촉발하는 조건이 다르거나,
  청구된 인과 경로가 원문에 없으면 명칭이 같더라도 supported=false입니다.
  일반 기술상식으로 빠진 연결을 보충하지 마십시오.
- **역할로 대조하라는 위 지시는 명칭을 무시하라는 뜻이지, 동작·대상·집합성·인과관계까지
  무시하라는 뜻이 아닙니다.** 네 축을 **각각 독립으로** 판정하십시오.
  - **동작**: 근거가 그 동작을 명시하는가. 근거가 획득·수집·측정을 **문언으로 적고 있다면
    동작 축을 결손으로 적지 마십시오.** 생성·전달·표시·통과만 있을 때에만 이 축이 빈 것입니다.
  - **대상**: 그 동작이 걸리는 것이 한정의 대상과 같거나 기능적으로 등가인가. 동작이 같아도
    대상이 다르면 supported=false이며, 그때 사유는 "동작이 없다"가 아니라 **"대상이 다르다"**입니다.
  - **집합성**: 한정이 복수의 세트를 요구하는데 근거가 한 사례·한 경로만 보이면 결손입니다.
  - **인과관계**: 두 가지가 각각 있는 것과, 하나가 다른 하나를 낳는 것은 다릅니다.
  reason에는 **비어 있는 축의 이름**과 그 축이 실제로 요구하는 것을 함께 적으십시오. 축을
  지목하지 못하면서 기각했다면 명칭 차이를 근거 결손으로 잘못 읽은 것입니다.
  같은 문헌 안에서 어떤 한정에는 개시를 인정하고 다른 한정에는 같은 성격의 기재를 부정하는
  일이 없도록, 판단 기준을 항목 사이에서 일관되게 유지하십시오.
- **주체 정체성과 방향성을 별도로 확인하십시오.** 청구항이 같은 모델·네트워크·부재에 여러
  역할을 귀속시키면 근거도 같은 실체에 그 역할을 귀속시켜야 합니다. 광학계를 모델링하는
  프록시와 별개의 영상 생성 뉴럴 네트워크를 합쳐 하나의 "광학계를 모델링하는 뉴럴 네트워크"로
  읽지 마십시오. 이 경우 supported=false이고 reason에 **주체 정체성 축 결손**이라고 적으십시오.
- 학습에 사용된 프록시·교사 모델·손실 함수의 역할을 학습 대상 네트워크 자체의 역할로 옮기지
  마십시오. 원문이 두 실체가 동일하거나 학습 후 그 역할을 승계한다고 명시할 때만 연결합니다.
- 입력→출력 방향이 반대이면 supported=false입니다. 타겟 영상을 입력받아 위상 패턴을 출력하는
  모델은 보정 영상을 입력받아 타겟 영상을 출력하는 모델을 뒷받침하지 않습니다. 네트워크 밖에서
  보정행렬을 영상에 적용해 디스플레이로 보내는 절차도, 그 보정 영상이 학습된 네트워크의 입력이고
  목표 영상이 그 네트워크의 출력이라는 원문 연결이 없으면 뒷받침하지 않습니다. reason에는
  **방향성 축 결손** 또는 **네트워크 경계 결손**을 명시하십시오.
- 주체 정체성 검사는 **현재 item의 limitation이 함께 요구하는 역할**에만 적용하십시오. 현재
  item이 단순히 "뉴럴 네트워크를 학습함"이라면 실제로 학습되는 뉴럴 네트워크의 근거가 있는지
  판단하고, 형제 한정인 "그 네트워크가 광학계를 모델링함"의 실패를 가져와 함께 기각하지
  마십시오. 다만 "신경망 학습과 유사하다"는 비유는 실제 뉴럴 네트워크 학습의 근거가 아닙니다.
- 입력/출력 이미지 **세트 획득** 한정에서는 같은 캘리브레이션 흐름의 제시·공급된
  stimulus/target 이미지와 대응하는 촬영·관찰 결과가 복수로 존재하면 입력측·출력측 이미지쌍을
  뒷받침할 수 있습니다. 이때 정확한 명칭보다 제시→촬영 대응과 복수성을 봅니다. 그러나 특정
  도파관에 입력되어 그 도파관으로부터 출력된다는 별도 한정은 이 일반 영상쌍으로 추론하지 말고,
  도파관 및 인과 경로를 직접 확인하십시오.
- 인과 한정의 evidence 묶음에 (1) 시스템에 입력측 영상을 실제로 제시하는 문장과 (2) 그
  결과 이미지가 **해당 시스템을 통해** 촬영·출력된다는 문장이 같은 장치·실시 흐름으로 들어
  있으면, 두 문장을 연결하여 인과 경로를 뒷받침할 수 있습니다. 특히 "images were taken through
  a diffractive waveguide eyepiece" 같은 실제 통과·촬영 문장은 단순히 "display may be a
  waveguide display"라는 장치 유형 총론과 다릅니다. 전자는 입력 제시 문장과 같은 흐름이면
  supported=true가 가능하고, 후자만 있으면 인과관계 축 결손입니다.
- 근거가 **배경기술·해결 과제·연구 동기·요약·향후 적용 가능성**을 서술한 문장이면 그 자체로는
  개시가 아닙니다. "…가 되기 어렵다", "…에 적용될 수 있을 것이다", "본 연구의 목표는 …"
  같은 문장은 그 구성을 실제로 수행하는 실시 기재가 따로 확인될 때만 supported=true입니다.
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
        items = [_item(match, check, references, document) for match, check in targets]
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
          references: dict[tuple[int, str], list[str]] | None = None,
          document: Document | None = None) -> dict:
    evidence = []
    seen: set[tuple[str, str]] = set()
    for chunk_id, quote, translation, alignment in _verified_bundle(check):
        key = (chunk_id, quote)
        if key in seen:
            continue
        seen.add(key)
        row = {"chunk_id": chunk_id, "original": quote,
               "translation": translation, "alignment": alignment}
        context = _source_context(document, chunk_id, quote)
        if context:
            row["source_context"] = context
        evidence.append(row)
    item = {"item_id": _item_id(match, check), "element_label": match.label,
            "limitation": check.limitation, "kind": check.kind, "evidence": evidence}
    siblings = _sibling_context(match, check, seen, document)
    if siblings:
        item["element_context"] = siblings
    terms = (references or {}).get((match.claim_number, match.label))
    if terms:
        item["reference_terms"] = terms
    return item


def _sibling_context(match: ElementMatch, check: LimitationCheck,
                     seen: set[tuple[str, str]], document: Document | None) -> list[dict]:
    """같은 구성요소의 **다른** 한정이 제출한, 원문 검증을 통과한 인용문.

    최초 대비는 한정마다 근거 묶음을 따로 만드는데, 하나의 인과 사슬이 형제 한정으로 쪼개져
    담기는 일이 있습니다. 실측에서 core "상대 부재의 결합 상태에 따라 전원 경로를 개폐하는
    부재를 포함함"의 묶음에는 부재의 존재만 말하는 문장이 들어가고, 개폐 동작을 명시한 같은
    실시예의 문장은 qualifier 묶음에 들어갔습니다. 이 단계는 한정마다 그 묶음만 읽으므로
    core는 근거 없는 것으로 기각되었고, 문헌이 그 구성을 원문으로 개시했는데도 구성 전체가
    "대응 없음"이 되었습니다.

    같은 구성요소·같은 문헌의 검증된 인용문만 넘기므로 새로운 개시 경로를 만들지 않습니다.
    실제로 인과 경로를 잇는 데 쓸 수 있는지는 프롬프트의 '같은 실시 흐름' 조건이 정하고,
    이것만으로 supported를 줄 수 없다는 제약도 함께 겁니다.
    """
    rows: list[dict] = []
    for sibling in match.limitation_checks:
        if sibling.index == check.index:
            continue
        for chunk_id, quote, translation, alignment in _verified_bundle(sibling):
            key = (chunk_id, quote)
            if key in seen:
                continue
            seen.add(key)
            row = {"chunk_id": chunk_id, "original": quote,
                   "translation": translation, "alignment": alignment,
                   "from_limitation": sibling.limitation}
            context = _source_context(document, chunk_id, quote)
            if context:
                row["source_context"] = context
            rows.append(row)
    return rows


def _source_context(document: Document | None, chunk_id: str, quote: str) -> str:
    """검증된 인용문 주변의 동일 청크 원문을 의미검증용으로 되돌립니다.

    compare 단계의 대표 발췌는 보고서 가독성과 문자열 검증을 위해 한 문장으로 짧게 잡습니다.
    그러나 관계형 한정은 같은 특허 단락의 뒤 문장이 수단이나 인과 연결을 완성하는 경우가 많습니다.
    인용 위치가 원문 검증을 통과한 경우에만 이 문맥을 붙이므로, 다른 단락을 검색해 빠진 내용을
    임의로 보충하는 경로는 만들지 않습니다.
    """
    if document is None or not chunk_id:
        return ""
    text = " ".join(chunk_text(document, chunk_id).split())
    if not text:
        return ""
    if len(text) <= MAX_SOURCE_CONTEXT_CHARS:
        return text

    needle = " ".join(str(quote or "").split())
    position = text.casefold().find(needle.casefold()) if needle else -1
    if position < 0:
        return text[:MAX_SOURCE_CONTEXT_CHARS]
    start = max(0, position - MAX_SOURCE_CONTEXT_CHARS // 2)
    end = min(len(text), start + MAX_SOURCE_CONTEXT_CHARS)
    start = max(0, end - MAX_SOURCE_CONTEXT_CHARS)
    return text[start:end]


def _item_id(match: ElementMatch, check: LimitationCheck) -> str:
    return f"{match.claim_number}:{match.label}:{check.index}"


def _reconcile_match(match: ElementMatch) -> None:
    """의미검증에서 빠진 한정을 누락 목록·판정 상한에 일관되게 반영합니다.

    whole_element 점검(분해 결과가 없어 구성 원문 한 줄을 통째로 본 경우)도 여기서는 그대로
    셉니다. 그 실패는 누락 '한정'이 아니라 구성 자체의 미개시라서 missing_limitations 목록에는
    오르지 않지만(models.missing_limitations), 상한 계산에서까지 빼면 "의미검증에서 제외했다"는
    노트와 '실질적 동일' 판정이 같은 셀에 함께 남습니다. 이 단계가 걸러 내려는 과대 개시가
    분해 실패한 구성에서만 그대로 통과하게 됩니다.

    directness는 **compare._build_matches와 같은 규칙**으로 내립니다. core가 하나도 남지 않으면
    direct → inferred이며, "absent"로는 내리지 않습니다. absent의 정의는 "근거 원문이 없음"인데
    (compare.py [직접성 directness]), 이 시점의 셀에는 원문 대조를 통과한 발췌가 남아 있고
    개시가 인정된 하위 한정도 있을 수 있습니다. 종전에는 여기서만 absent로 내려서, 같은 조건을
    두 모듈이 다르게 처리했고 그 차이가 결론까지 갔습니다 — coverage.ineligible_reason이
    absent를 "직접 근거 없음"으로 걸러 내므로, 그 문헌은 **원문으로 검증해 개시한 한정마저**
    보조 인용발명으로 기여할 자격을 잃었습니다. 게다가 chain._residual_overflow는 미채택 문헌의
    개시를 "결합 문헌 수 상한을 넘어 세우지 않았다"고 적으므로, 실제로는 자격에서 탈락한 것을
    보고서가 상한 탓으로 잘못 설명했습니다.
    """
    match.missing_limitations = missing_limitations(match.limitation_checks)
    checks = match.limitation_checks
    rejected = [check for check in checks if check.semantic_status == "rejected"]
    if not rejected:
        return
    # 최초 판정과 **같은 사다리**로 다시 산출합니다(coverage.derive_judgment). 이 단계가 바꾼
    # 것은 한정별 disclosed뿐이므로, 등급은 그 결과로 따라와야 합니다. 종전에는 여기에 별도
    # 상한 사다리가 있어서 같은 조건을 두 모듈이 다르게 처리했습니다.
    cap = derive_judgment(
        checks,
        has_evidence=bool(match.quote or match.evidence or any(check.quote for check in checks)),
        terminology=match.terminology,
        different_purpose=match.different_purpose)
    if cap != "일부 차이" and match.directness == "direct":
        match.directness = "inferred"
    # 독립 검증은 최초 판정을 **올리지 않고 상한만** 씌웁니다. 재산출이 더 높게 나오더라도
    # (예: 모델이 처음에 스스로 낮춰 놓은 경우) 이 단계에서 끌어올리지 않습니다.
    if JUDGMENT_RANK.get(match.judgment, 0) > JUDGMENT_RANK[cap]:
        match.downgraded_from = match.downgraded_from or match.judgment
        match.judgment = judgment_at_rank(JUDGMENT_RANK[cap])
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
