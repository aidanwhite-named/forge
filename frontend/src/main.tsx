import {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './styles.css';

const API = 'http://localhost:8330/api';

type Result = {
  job_id: string;
  claim_mapping: any[];
  reports: any[];
  preamble: string;
  validation: string[];
  prior_art: any[];
  cached_claims: number[];
};

type Tab = 'analysis' | 'decompose' | 'result' | 'history' | 'logs' | 'settings';
type Settings = {provider: string; model: string; prompt: string};
type LogItem = {job_id: string; size: number; updated_at?: string};
type Progress = {done: number | null; total: number};
// detail은 응답이 실패했을 때 FastAPI가 돌려주는 오류 메시지 자리입니다.
type Job = {status: string; stage?: string; error?: string; progress?: Progress; detail?: string};

// 분해 확정 관문에서 주고받는 형식. backend/app/claims.py의 dump_decomposition이 쓰는 모양
// 그대로입니다. **라벨과 구성 원문은 화면에서 고칠 수 없습니다** — 청구항 파서가 결정론적으로
// 만든 뼈대라 validate_confirmed_decomposition이 400으로 돌려보냅니다. 고쳐야 한다면 청구항
// 원문을 고쳐 처음부터 다시 실행할 일입니다.
type Limitation = {text: string; kind: 'core' | 'qualifier'; alternative_group: string};
type ElementDraft = {
  label: string;
  text: string;
  importance: number;
  is_sub: boolean;
  search_terms: string[];
  limitations: Limitation[];
};
// 청구항이 같은 대상을 다르게 적은 자리. 도구가 자동으로 잇지 않고 **여기서 확정을 받습니다** —
// "가시 두상 영역" ↔ "가시 두상 영상"은 동의어가 아니라 오기일 수 있고, 그 판단은 사람 몫입니다.
// 확정 관문을 분해와 같은 화면에 두는 이유는 "무엇을 확정했는지"가 한 군데 남아야 하기 때문입니다.
type ReferenceAlias = {
  target: string;          // "상기 …"를 적은 구성
  term: string;
  candidates: string[];    // 해소기가 찾은 도입 후보 전부
  selected_source: string; // 사용자가 고른 것. 서버가 candidates 안인지 대조합니다
  confirmed: boolean;
};
type Decomposition = {
  version: string;
  claims: Record<string, ElementDraft[]>;
  aliases?: Record<string, ReferenceAlias[]>;
};
type Review = {
  job_id: string;
  claims_text: string;
  version: string;
  decomposition: Decomposition;
  warnings: string[];
};
type ViewTransition = {ready?: Promise<void>; finished: Promise<void>};
type WithViewTransition = Document & {
  startViewTransition?: (callback: () => void) => ViewTransition;
};

const sleep = (milliseconds: number) => new Promise(resolve => setTimeout(resolve, milliseconds));
const scrollToTop = () => window.scrollTo({top: 0, behavior: 'smooth'});

/** 셀 진행률 바. 서버가 done/total을 줄 때만 그립니다. */
function ProgressBar({progress}: {progress: Progress | null}) {
  if (!progress?.total) return null;
  const done = progress.done || 0;
  const percent = Math.min(100, Math.round((done / progress.total) * 100));
  return (
    <div className="run-progress">
      <div className="track" role="progressbar" aria-valuenow={done} aria-valuemin={0}
           aria-valuemax={progress.total}>
        <i style={{width: `${percent}%`}} />
      </div>
      <span>{done}/{progress.total}</span>
    </div>
  );
}
// 청구항이 같은 대상을 다르게 적은 자리를 사람이 확정하는 자리. 도구는 이 연결을 스스로
// 잇지 않습니다 — 문언이 어긋난 참조를 자동으로 이으면 청구항의 기재 문제를 도구가 대신
// 덮어 주게 되고, 출원 중이면 고쳐야 할 기재불비가 조용히 지나갑니다.
//
// 확정한 것만 판정 경로 셋(등급 상한·의미검증 입력·차이점 서술)에 들어갑니다. 확정하지 않으면
// 보고서에 "추정했다"는 경고로만 남고 판정은 그대로입니다.
export function AliasGate({draft, setDraft}: {
  draft: Decomposition | null;
  setDraft: (next: Decomposition) => void;
}) {
  const entries = Object.entries(draft?.aliases || {});
  const total = entries.reduce((count, [, items]) => count + items.length, 0);
  if (!draft || !total) return null;

  const edit = (number: string, index: number, patch: Partial<ReferenceAlias>) => {
    const aliases = {...(draft.aliases || {})};
    aliases[number] = aliases[number].map((item, position) =>
      position === index ? {...item, ...patch} : item);
    setDraft({...draft, aliases});
  };

  return (
    <section className="alias-gate" aria-label="지시 관계 확정">
      <strong>지시 대상을 확정해 주십시오 ({total}건)</strong>
      <p>
        청구항이 같은 대상을 다르게 적었을 수 있는 자리입니다. 도구가 문언만으로는 확정하지
        못했습니다. <b>확정한 것만</b> 판정에 반영되고, 두면 보고서에 추정으로 표시됩니다.
        문언 자체가 잘못된 것이라면 청구항을 고치는 쪽이 맞습니다.
      </p>
      {entries.map(([number, items]) => items.map((item, index) => {
        // 후보가 둘 이상이면 문언만으로는 어느 것인지 정할 수 없는 자리입니다. 첫 후보를
        // 기본으로 집어 두면 사람이 고르지 않은 값이 확정되므로, 고르기 전에는 비워 둡니다.
        const choosing = item.candidates.length > 1;
        // 후보가 하나면 고를 것이 없으므로 체크만으로 확정됩니다. 서버가 내주는 값에 이미
        // 채워져 있지만, 옛 기록이나 직접 만든 요청으로 비어 올 수 있으므로 여기서도 채웁니다 —
        // 잠긴 채로 남으면 사용자는 확정할 방법이 없고 왜 안 되는지도 알 수 없습니다.
        const only = item.candidates.length === 1 ? item.candidates[0] : '';
        const ready = item.candidates.includes(item.selected_source) || Boolean(only);
        return (
          <div key={`${number}-${index}`} className={item.confirmed ? 'alias settled' : 'alias'}>
            <label>
              <input type="checkbox" checked={item.confirmed}
                     disabled={!ready}
                     onChange={() => edit(number, index, {
                       confirmed: !item.confirmed,
                       selected_source: item.selected_source || only,
                     })} />
              <span>
                청구항 {number} 구성 <b>{item.target}</b>의 <b>"{item.term}"</b>을{' '}
                {choosing ? '아래에서 고른 구성' : <b>구성 {item.candidates[0]}</b>}이 세운
                대상과 같은 것으로 봅니다
              </span>
            </label>
            {choosing && (
              <select value={item.selected_source}
                      onChange={event => edit(number, index, {
                        selected_source: event.target.value,
                        // 고른 값을 바꾸면 확정도 풉니다. 앞서 확정한 것은 **다른 후보**에
                        // 대한 판단이라 그대로 이어 붙이면 사람이 하지 않은 확정이 됩니다.
                        confirmed: false,
                      })}>
                <option value="">어느 구성인지 고르십시오</option>
                {item.candidates.map(label => (
                  <option key={label} value={label}>구성 {label}</option>
                ))}
              </select>
            )}
          </div>
        );
      }))}
    </section>
  );
}

// 구성 하나의 **판정 경위**. 결과만 보여 주면 근거를 들어 기각한 판정과 아예 검토하지 못한
// 판정이 화면에서 똑같이 "대응 없음"으로 보입니다. 실측에서 그 화면을 두고 두 사람이 정반대
// 결론을 냈습니다 — 한쪽은 도구가 검토하지 않았다고 읽었고, 실제로는 근거를 들어 두 번
// 기각한 것이었습니다. 보고서(report._trail_lines)와 같은 자료를 같은 순서로 보여 줍니다.
export function VerificationTrail({trail}: {trail: any[] | undefined}) {
  const rows = (trail || []).filter(item =>
    item.steps?.length || _split(item) || item.tallies?.length);
  if (!rows.length) return null;
  return (
    <section className="trail" aria-label="판정 경위">
      <b>판정 경위</b>
      {rows.map((item, index) => (
        <ul key={item.document_id || index}>
          {_split(item) && (
            // 비율(0.33)이 아니라 분자·분모를 적습니다. 비율은 "표본이 전부 갈렸다"로도
            // "한정 3개 중 1개만 만장일치"로도 읽히고, 실제로 그 오독이 결론까지 갔습니다.
            //
            // **"갈렸다"고 단정하지 않습니다.** 만장일치가 아닌 이유는 판단이 나뉜 것일 수도,
            // 표본이 답하지 못한 것일 수도 있습니다. 어느 쪽인지는 아래 한정별 줄이 말합니다.
            <li className="split">⚠️ 초기 비교 결과가 불안정합니다 — 한정 {item.sample_requirements}개 중{' '}
              {item.sample_unanimous}개만 표본 {item.sample_count}회 만장일치</li>
          )}
          {/* 어느 한정이 몇 대 몇이었는지. 구성 단위 비율만으로는 "2대 1로 갈린 미개시"와
              "3대 0으로 일치한 미개시"가 같은 값이 되는데, 그 둘은 다음 조치가 다릅니다. */}
          {(item.tallies || []).map(([index, text, tally]: [number, string, Tally]) => {
            // 집계는 **원시 투표에서 셉니다.** 백엔드도 같은 규율이라(models.SampleTally),
            // 따로 받은 숫자를 믿으면 두 자리가 갈릴 수 있습니다.
            const at = (verdict: string) =>
              (tally.votes || []).filter(vote => vote.verdict === verdict).length;
            const parts = [`개시 ${at('disclosed')}표`, `미개시 ${at('missing')}표`];
            if (at('absent')) parts.push(`무응답 ${at('absent')}표`);
            if (at('invalid')) parts.push(`판독불가 ${at('invalid')}표`);
            return (
              <li key={`tally-${index}`} className="tally">
                <span className="kind">{_tallyKind(tally)}</span>
                <span className="limitation">한정 #{index} 「{text}」</span>
                <span className="why">{parts.join(' · ')} (표본 {tally.total}회)</span>
              </li>
            );
          })}
          {/* 단계 순서로 묶습니다. 한정 번호 순으로 늘어놓으면 두 단계가 번갈아 나와,
              어느 단계가 무엇을 걸렀는지가 줄을 세어야 보입니다. */}
          {['의미검증', '결합검증'].flatMap(stage =>
            (item.steps || []).filter((step: any) => step.stage === stage).map((step: any) => (
              <li key={`${stage}-${step.index}`}>
                <span className={step.outcome === '기각' ? 'stage-out' : 'stage-in'}>
                  {step.stage} {step.outcome}</span>
                <span className="limitation">한정 #{step.index} 「{step.limitation}」</span>
                {step.note && <span className="why">{step.note}</span>}
              </li>
            )))}
        </ul>
      ))}
    </section>
  );
}

// 표본이 갈린 셀인지. 물어본 한정이 있고 그중 만장일치가 아닌 것이 있을 때.
// backend/app/models.py VerificationTrail.split과 같은 정의입니다.
const _split = (item: any) => (item.sample_requirements || 0) > 0
  && (item.sample_unanimous || 0) < item.sample_requirements;

type Tally = {total: number; votes: {sample: number; verdict: string}[]};

/** 이 한정이 왜 만장일치가 아닌지. backend/app/report.py의 _tally_kind와 같은 규칙입니다.
 *
 * 판정 불일치는 같은 근거를 두고 판단이 나뉜 것이라 사람이 원문을 봐야 하고, 응답 결손은
 * 도구가 답을 받지 못한 것이라 다시 물으면 됩니다. 후속 조치가 다르므로 뭉치지 않습니다.
 */
export function _tallyKind(tally: Tally): string {
  const at = (verdict: string) =>
    (tally.votes || []).filter(vote => vote.verdict === verdict).length;
  const kinds: string[] = [];
  if (at('disclosed') > 0 && at('missing') > 0) kinds.push('판정 불일치');
  if (at('absent') > 0 || at('invalid') > 0) kinds.push('응답 결손');
  return kinds.join(' · ') || '표본 불일치';
}

// 선행기술 결과의 실재 확인 표시 → [CSS 클래스, 라벨]. 확인하지 못한 것도 감추지 않습니다.
const PRIOR_ART_VERIFY: Record<string, [string, string]> = {
  verified: ['verified', '✅ 확인됨'],
  mismatch: ['mismatch', '⚠️ 번호 불일치'],
  unreachable: ['unchecked', '❔ 확인 불가'],
  unchecked: ['unchecked', '❔ 미확인'],
};

// 등급 이모지 → 배지 색 톤. 구성 라벨 (A)~(Z) 동그라미를 등급 이모지와 같은 색으로 칠해,
// 이모지를 따로 읽지 않아도 구성 목록을 훑는 것만으로 등급이 보이게 합니다.
// 색은 backend/app/coverage.py의 REPORT_GRADES가 붙이는 이모지를 그대로 따릅니다.
const GRADE_TONE: Record<string, string> = {
  '🔵': 'tone-identical',    // 동일
  '🟢': 'tone-substantial',  // 실질적 동일
  '🟠': 'tone-variation',    // 기술 사상 동일, 세부 구현 방식의 단순 변경
  '🟡': 'tone-partial',      // 핵심 기능 유사하나 목적/효과에 일부 차이
  '⚪': 'tone-none',         // 대응 안됨
  '⚠️': 'tone-unjudged',     // 판정 불가
};

const formatLogDate = (value?: string) => {
  if (!value) return '수정 시각 없음';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '수정 시각 없음' : date.toLocaleString('ko-KR');
};

// backend/app/claims.py의 같은 이름 상수와 맞춥니다. 넘기면 서버가 확정을 거절합니다 —
// 조용히 잘라 내면 사용자가 적어 넣은 한정이 말없이 사라진 채 분석이 돕니다.
const MAX_LIMITATIONS = 12;
const MAX_SEARCH_TERMS = 16;

/** backend/app/claims.py의 _unique_strings와 같은 규칙. 중복 판정 기준을 서버와 맞춥니다. */
function uniqueStrings(values: string[]): string[] {
  const cleaned: string[] = [];
  for (const value of values) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim().replace(/[;,]+$/, '');
    if (text && !cleaned.includes(text)) cleaned.push(text);
  }
  return cleaned;
}

/** 서버가 보낸 제안을 편집용 초안으로 옮깁니다.
 *
 * 값의 자료형을 여기서 못박습니다. importance는 정수, is_sub는 진짜 boolean이어야 하는데
 * (validate_confirmed_decomposition), 폼 입력을 거치면 문자열이 되기 쉽습니다. 문자열
 * "false"는 서버의 bool()에서 참이 되어 조용히 뒤집히므로 입력 시점마다 막는 대신
 * 초안 형식을 처음부터 고정합니다.
 */
function cloneDecomposition(source: Decomposition): Decomposition {
  const claims: Record<string, ElementDraft[]> = {};
  for (const [number, elements] of Object.entries(source?.claims || {})) {
    claims[number] = (elements || []).map(element => ({
      label: String(element.label ?? ''),
      text: String(element.text ?? ''),
      importance: Number(element.importance ?? 3),
      is_sub: Boolean(element.is_sub),
      search_terms: (element.search_terms || []).map(term => String(term ?? '')),
      limitations: (element.limitations || []).map(limitation => ({
        text: String(limitation.text ?? ''),
        kind: limitation.kind === 'qualifier' ? 'qualifier' : 'core',
        alternative_group: String(limitation.alternative_group ?? ''),
      })),
    }));
  }
  const aliases: Record<string, ReferenceAlias[]> = {};
  for (const [number, items] of Object.entries(source?.aliases || {})) {
    aliases[number] = (items || []).map(item => ({
      target: String(item.target ?? ''),
      term: String(item.term ?? ''),
      candidates: (item.candidates || []).map(label => String(label ?? '')),
      selected_source: String(item.selected_source ?? ''),
      // 기본값은 언제나 미확정입니다. 읽지 못한 값이 확정으로 살아나면 사람이 승인하지 않은
      // 연결이 판정을 바꿉니다(backend/app/claims.py의 _restore_aliases와 같은 규칙).
      confirmed: item.confirmed === true,
    }));
  }
  const cloned: Decomposition = {version: source?.version || '', claims};
  if (Object.keys(aliases).length) cloned.aliases = aliases;
  return cloned;
}

/** 확정 전에 화면에서 먼저 거르는 문제들.
 *
 * backend/app/claims.py의 _element_problems를 그대로 옮겼습니다. 서버가 여전히 최종
 * 판단자이지만(그쪽이 거절하면 확정 대기 상태가 유지됩니다), 왕복 한 번을 기다린 뒤에야
 * "대안군에 항목이 하나뿐"이라는 말을 듣게 하면 고치는 자리를 다시 찾아야 합니다.
 */
function decompositionProblems(draft: Decomposition | null): string[] {
  const problems: string[] = [];
  for (const [number, elements] of Object.entries(draft?.claims || {})) {
    for (const element of elements) {
      const where = `청구항 ${number} ${element.label}`;
      if (!Number.isInteger(element.importance) || element.importance < 1 || element.importance > 5) {
        problems.push(`${where}: 중요도는 1~5의 정수여야 합니다`);
      }
      if (element.search_terms.length > MAX_SEARCH_TERMS) {
        problems.push(`${where}: 검색어는 ${MAX_SEARCH_TERMS}개까지입니다`);
      }
      if (uniqueStrings(element.search_terms).length !== element.search_terms.length) {
        problems.push(`${where}: 검색어가 중복되었거나 비어 있습니다`);
      }
      if (!element.limitations.length) {
        // 한정이 없으면 구성 원문 한 줄을 통째로 점검하게 되어(whole_element), 사용자가
        // 확정한 것이 무엇인지 알 수 없는 상태로 판정이 돕니다.
        problems.push(`${where}: 한정이 최소 하나는 있어야 합니다`);
        continue;
      }
      if (element.limitations.length > MAX_LIMITATIONS) {
        problems.push(`${where}: 한정은 ${MAX_LIMITATIONS}개까지입니다`);
      }
      const texts: string[] = [];
      const groups = new Map<string, number>();
      element.limitations.forEach((limitation, index) => {
        const text = limitation.text.trim();
        if (!text) problems.push(`${where}: 한정 ${index + 1}의 문언이 비어 있습니다`);
        else texts.push(text);
        const group = limitation.alternative_group.trim();
        if (group) groups.set(group, (groups.get(group) || 0) + 1);
      });
      // 확정 경로에는 _build_limitations가 돌지 않아 중복이 걸러지지 않습니다. 같은 문언이
      // 둘 남으면 total_limitations가 부풀어 개시율이 실제보다 낮게 집계됩니다.
      if (uniqueStrings(texts).length !== texts.length) {
        problems.push(`${where}: 같은 문언의 한정이 중복되었습니다`);
      }
      // 항목이 하나뿐인 대안군은 그 한정이 미개시일 때 '묶음이 충족되지 않았을 뿐'으로 읽혀
      // 누락 판정이 흐려집니다. 짝을 채우거나 표시를 빼야 합니다.
      const lonely = [...groups.entries()].filter(([, count]) => count < 2)
        .map(([name]) => name).sort();
      if (lonely.length) problems.push(`${where}: 대안군 ${lonely.join(', ')}에 항목이 하나뿐입니다`);
    }
  }
  return problems;
}

/** 구성별 검색어 편집기. 검색어는 문헌 청크 순위를 정하므로(compare._element_terms)
 *  자유 입력으로 두지 않고 한 건씩 확정해 담습니다. */
function TermEditor({terms, disabled, onChange}: {
  terms: string[];
  disabled: boolean;
  onChange: (next: string[]) => void;
}) {
  const [entry, setEntry] = useState('');
  const full = terms.length >= MAX_SEARCH_TERMS;

  function commit() {
    const value = entry.replace(/\s+/g, ' ').trim().replace(/[;,]+$/, '');
    setEntry('');
    if (!value || full || terms.includes(value)) return;
    onChange([...terms, value]);
  }

  return (
    <div className="term-editor">
      <div className="term-list">
        {terms.map((term, index) => (
          <span key={`${term}-${index}`} className="term">
            {term}
            <button
              type="button"
              disabled={disabled}
              aria-label={`검색어 ${term} 제거`}
              onClick={() => onChange(terms.filter((_, spot) => spot !== index))}
            >×</button>
          </span>
        ))}
        {!terms.length && <small className="term-empty">검색어 없음</small>}
      </div>
      <input
        aria-label="검색어 추가"
        value={entry}
        disabled={disabled || full}
        placeholder={full ? `최대 ${MAX_SEARCH_TERMS}개까지입니다` : '검색어를 입력하고 Enter'}
        onChange={event => setEntry(event.target.value)}
        onKeyDown={event => {
          if (event.key !== 'Enter') return;
          event.preventDefault();
          commit();
        }}
        onBlur={commit}
      />
    </div>
  );
}

function App() {
  const [tab, setTab] = useState<Tab>('analysis');
  const [claims, setClaims] = useState('');
  // 대상 청구항의 출원일·우선일. 비워 두면 백엔드가 선행기술 적격성 분류를 보류하고
  // 구성대비만 수행합니다(eligibility.py). 추정해서 채우지 않습니다.
  const [priorityDate, setPriorityDate] = useState('');
  const [dependentClaims, setDependentClaims] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [selectedLog, setSelectedLog] = useState('');
  const [logContent, setLogContent] = useState('');
  const [logsLoading, setLogsLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  // 선행기술 검색은 종속항 대비와 다른 작업입니다. actionBusy를 같이 쓰면 검색 중에
  // 종속항 카드의 취소 버튼까지 떠서, 누르면 엉뚱한 취소 경로가 돕니다.
  const [priorArtBusy, setPriorArtBusy] = useState(false);
  const [stage, setStage] = useState('입력 대기');
  const [progress, setProgress] = useState<Progress | null>(null);
  // 서버가 재시작되면 분석이 중단됩니다. 404 대신 사유를 받아 그대로 보여 줍니다.
  const [interrupted, setInterrupted] = useState('');
  const [message, setMessage] = useState('');
  // 분해 확정 관문. review는 서버가 보낸 제안(원문 대조용 청구항 포함)이고 draft는 사용자가
  // 고치는 사본입니다. 둘을 나눠 두어야 "제안과 무엇이 달라졌는지"를 화면에서 셀 수 있습니다.
  const [review, setReview] = useState<Review | null>(null);
  const [draft, setDraft] = useState<Decomposition | null>(null);
  const [reviewProblems, setReviewProblems] = useState<string[]>([]);
  const [confirming, setConfirming] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    provider: 'agy',
    model: 'gemini-3.6-flash-medium',
    prompt: '',
  });
  const [models, setModels] = useState<string[]>([]);
  const activeJob = useRef<string | null>(null);
  const uploadController = useRef<AbortController | null>(null);
  const cancelRequested = useRef(false);
  const priorArtController = useRef<AbortController | null>(null);
  const priorArtCancelled = useRef(false);

  const busy = generating || actionBusy || priorArtBusy;

  useEffect(() => {
    refreshHistory();
    fetch(API + '/settings')
      .then(response => response.json())
      .then(next => {
        setSettings(next);
        loadModels(next);
      })
      .catch(() => undefined);
  }, []);

  async function refreshHistory() {
    try {
      const response = await fetch(API + '/history');
      setHistory(response.ok ? await response.json() : []);
    } catch {
      setHistory([]);
    }
  }

  async function loadLog(jobId: string) {
    setSelectedLog(jobId);
    setLogsLoading(true);
    try {
      const response = await fetch(`${API}/logs/${jobId}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || '로그를 불러오지 못했습니다.');
      setLogContent(data.content || '');
    } catch (error: any) {
      setLogContent(error.message || '로그를 불러오지 못했습니다.');
    } finally {
      setLogsLoading(false);
    }
  }

  async function refreshLogs(preferredId = selectedLog) {
    setLogsLoading(true);
    try {
      const response = await fetch(API + '/logs');
      const items: LogItem[] = response.ok ? await response.json() : [];
      setLogs(items);
      const nextId = items.some(item => item.job_id === preferredId)
        ? preferredId
        : items[0]?.job_id || '';
      if (nextId) {
        await loadLog(nextId);
      } else {
        setSelectedLog('');
        setLogContent('');
      }
    } catch {
      setLogs([]);
      setSelectedLog('');
      setLogContent('');
    } finally {
      setLogsLoading(false);
    }
  }

  // 탭 전환은 한 곳에서만 합니다. navigate와 openResult가 각자 뷰 전환 분기를 들고 있으면
  // 한쪽만 고쳐져 같은 앱 안에서 전환 동작이 갈립니다.
  function switchTab(next: Tab) {
    const change = () => setTab(next);
    const {startViewTransition} = document as WithViewTransition;
    if (!startViewTransition) {
      change();
      requestAnimationFrame(scrollToTop);
      return;
    }
    const transition = startViewTransition.call(document, change);
    // 전환이 건너뛰어질 수 있습니다(탭이 백그라운드이거나 모션 축소 설정). 그때 ready는
    // InvalidStateError로 거부되지만 change()는 그대로 실행되어 화면은 바뀝니다. 아무도
    // 받지 않는 거부라 콘솔에 처리되지 않은 예외로 쌓여 진짜 오류를 덮으므로 삼킵니다.
    transition.ready?.catch(() => undefined);
    transition.finished.then(scrollToTop, scrollToTop);
  }

  function navigate(next: Tab) {
    if (next === 'result' && !result) return;
    if (next === 'decompose' && !review) return;
    if (next === 'logs') void refreshLogs();
    switchTab(next);
  }

  function openResult(next: Result) {
    setResult(next);
    // setResult가 반영된 뒤에 전환해야 구성대비 탭이 한 프레임 비어 보이지 않습니다.
    requestAnimationFrame(() => switchTab('result'));
  }

  async function loadModels(next: Settings, refresh = false) {
    try {
      const query = new URLSearchParams({
        provider: next.provider,
        ...(refresh ? {refresh: 'true'} : {}),
      });
      const response = await fetch(API + '/settings/models?' + query);
      const data = await response.json();
      const nextModels = response.ok && Array.isArray(data.models) ? data.models : [];
      setModels(nextModels);
      if (nextModels.length && !nextModels.includes(next.model)) {
        setSettings(current => current.provider === next.provider
          ? {...current, model: nextModels[0]}
          : current);
      }
      if (refresh) {
        setMessage(nextModels.length
          ? `모델 ${nextModels.length}개를 불러왔습니다.`
          : '모델 목록을 불러오지 못했습니다. 현재 설정된 모델을 유지합니다.');
      }
    } catch {
      setModels([]);
    }
  }

  function changeProvider(provider: string) {
    const next = {...settings, provider};
    setSettings(next);
    loadModels(next);
  }

  async function saveSettings(next = settings) {
    const response = await fetch(API + '/settings', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(next),
    });
    const data = await response.json();
    if (response.ok) setSettings(data);
    setMessage(response.ok ? '설정을 저장했습니다.' : `저장 실패: ${data.detail || ''}`);
  }

  async function resetPrompt() {
    await saveSettings({...settings, prompt: ''});
    setMessage('분석 지침 프롬프트를 기본값으로 되돌렸습니다.');
  }

  async function testSettings() {
    const response = await fetch(API + '/settings/test', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(settings),
    });
    setMessage((await response.json()).message);
  }

  async function run() {
    if (!claims.trim() || !files.length) {
      setMessage('청구항과 PDF를 모두 입력해 주세요.');
      return;
    }

    setGenerating(true);
    setMessage('');
    setInterrupted('');
    setProgress(null);
    setStage('작업 준비 중');
    cancelRequested.current = false;

    try {
      const preparedResponse = await fetch(API + '/jobs/prepare', {method: 'POST'});
      const prepared = await preparedResponse.json();
      if (!preparedResponse.ok) throw new Error(prepared.detail || '작업을 준비하지 못했습니다.');
      activeJob.current = prepared.job_id;

      if (cancelRequested.current) {
        await fetch(`${API}/jobs/${prepared.job_id}`, {method: 'DELETE'});
        return;
      }

      setStage('문서 업로드 중');
      const form = new FormData();
      form.append('claims', claims);
      form.append('analysis_prompt', settings.prompt || '');
      form.append('priority_date', priorityDate.trim());
      files.forEach(file => form.append('pdf_files', file));
      uploadController.current = new AbortController();
      const startResponse = await fetch(`${API}/jobs/${prepared.job_id}/start`, {
        method: 'POST',
        body: form,
        signal: uploadController.current.signal,
      });
      const started = await startResponse.json();
      if (!startResponse.ok) throw new Error(started.detail || '보고서 생성을 시작하지 못했습니다.');

      await trackJob(prepared.job_id);
    } catch (error: any) {
      if (error?.name !== 'AbortError' && !cancelRequested.current) {
        setMessage(error?.message || '분석에 실패했습니다.');
      }
    } finally {
      uploadController.current = null;
      activeJob.current = null;
      setGenerating(false);
    }
  }

  /** 작업 상태를 끝까지 따라갑니다.
   *
   * 분해 확정 관문(awaiting_decomposition)에서는 폴링을 멈추고 확인 화면으로 넘깁니다.
   * 이 상태는 서버가 사용자를 기다리는 자리라, 계속 폴링하면 화면은 영원히 "생성 중"으로
   * 남습니다. 확정 뒤에는 confirmDecomposition이 이 함수를 다시 부릅니다.
   */
  async function trackJob(jobId: string) {
    while (!cancelRequested.current) {
      await sleep(650);
      const statusResponse = await fetch(`${API}/jobs/${jobId}`);
      const job: Job = await statusResponse.json();
      if (!statusResponse.ok) throw new Error(job.detail || '작업 상태를 확인하지 못했습니다.');
      setStage(job.stage || '분석 중');
      setProgress(job.progress || null);
      if (job.status === 'interrupted') {
        setInterrupted(job.error || '분석이 중단되었습니다.');
        return;
      }
      if (job.status === 'failed') throw new Error(job.error || '분석에 실패했습니다.');
      if (job.status === 'cancelled') return;
      if (job.status === 'awaiting_decomposition') {
        await openDecompositionReview(jobId);
        return;
      }
      if (job.status !== 'completed') continue;

      const resultResponse = await fetch(`${API}/jobs/${jobId}/result`);
      const nextResult = await resultResponse.json();
      if (!resultResponse.ok) throw new Error(nextResult.detail || '결과를 불러오지 못했습니다.');
      setStage('완료');
      setMessage('보고서가 생성되었습니다.');
      await refreshHistory();
      openResult(nextResult);
      return;
    }
  }

  /** 확정 대기 중인 제안을 불러와 확인 화면을 엽니다. */
  async function openDecompositionReview(jobId: string) {
    const response = await fetch(`${API}/jobs/${jobId}/decomposition`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '분해 제안을 불러오지 못했습니다.');
    // 제안도 초안과 같은 정규화를 거쳐 담습니다. 둘의 형식이 다르면 "고친 구성" 표시가
    // 자료형 차이만으로 켜져, 손대지 않은 구성이 편집된 것처럼 보입니다.
    const next: Review = {
      job_id: jobId,
      claims_text: payload.claims_text || '',
      version: payload.version || '',
      decomposition: cloneDecomposition(payload.decomposition || {version: '', claims: {}}),
      warnings: payload.warnings || [],
    };
    setReview(next);
    setDraft(cloneDecomposition(next.decomposition));
    setReviewProblems([]);
    setProgress(null);
    setStage('청구항 분해 확인 대기');
    // openResult와 달리 rAF로 미루지 않습니다. 분해는 20초 안팎이 걸려 그동안 창을 옮겨 두기
    // 쉬운데, 배경 탭에서는 rAF 콜백이 밀려 화면이 '분석'에 머뭅니다. 관문에 왔다는 사실이
    // 바로 보이지 않으면 아까와 같은 무한 대기로 보입니다. 위 setState들이 먼저 반영되므로
    // 곧바로 전환해도 빈 화면이 스치지 않습니다.
    switchTab('decompose');
  }

  /** 확정한 분해로 구성대비를 시작합니다.
   *
   * 서버가 400을 주면 **확정 대기 상태가 그대로 유지됩니다.** 초안을 지우지 않고 사유만
   * 띄우는 이유입니다 — 고쳐서 다시 보내는 것이 정상 경로이고, 여기서 화면을 닫으면
   * 사용자는 오타 하나에 업로드부터 다시 해야 합니다.
   */
  async function confirmDecomposition() {
    if (!review || !draft || confirming) return;
    const problems = decompositionProblems(draft);
    if (problems.length) {
      setReviewProblems(problems);
      return;
    }
    const jobId = review.job_id;
    setConfirming(true);
    setReviewProblems([]);
    try {
      const response = await fetch(`${API}/jobs/${jobId}/decomposition/confirm`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({decomposition: draft}),
      });
      const confirmed = await response.json();
      if (!response.ok) {
        setReviewProblems([confirmed.detail || '분해를 확정하지 못했습니다.']);
        return;
      }
    } catch (error: any) {
      setReviewProblems([error?.message || '분해를 확정하지 못했습니다.']);
      return;
    } finally {
      setConfirming(false);
    }
    // 여기부터는 구성대비가 서버에서 이미 돌고 있습니다. 관문을 닫고 진행 표시로 돌아갑니다.
    setReview(null);
    setDraft(null);
    await resumeTracking(jobId);
  }

  /** 확정 이후의 진행을 분석 화면에서 이어 봅니다. run()의 뒷부분과 같은 자리입니다. */
  async function resumeTracking(jobId: string) {
    setGenerating(true);
    setMessage('');
    setInterrupted('');
    setProgress(null);
    setStage('구성대비 준비 중');
    cancelRequested.current = false;
    activeJob.current = jobId;
    switchTab('analysis');
    try {
      await trackJob(jobId);
    } catch (error: any) {
      if (error?.name !== 'AbortError' && !cancelRequested.current) {
        setMessage(error?.message || '분석에 실패했습니다.');
      }
    } finally {
      activeJob.current = null;
      setGenerating(false);
    }
  }

  /** 확정 대기 중에 취소합니다. 서버는 이때 업로드 임시 폴더도 함께 정리합니다. */
  async function cancelDecomposition() {
    if (!review) return;
    cancelRequested.current = true;
    try {
      await fetch(`${API}/jobs/${review.job_id}`, {method: 'DELETE'});
    } catch {
      // 서버가 취소를 받지 못했어도 화면은 닫습니다. 확정하지 않은 작업은 분석으로 넘어가지
      // 않으므로, 남더라도 결과를 만들지는 않습니다.
    }
    setReview(null);
    setDraft(null);
    setReviewProblems([]);
    setStage('취소됨');
    setMessage('분해 확인 단계에서 보고서 생성을 취소했습니다.');
    switchTab('analysis');
  }

  /** 구성 하나를 고칩니다. 라벨·구성 원문은 patch에 담지 않습니다(서버가 거절합니다). */
  function updateElement(number: string, index: number, patch: Partial<ElementDraft>) {
    setDraft(current => {
      if (!current) return current;
      const elements = (current.claims[number] || []).map((element, position) =>
        position === index ? {...element, ...patch} : element);
      return {...current, claims: {...current.claims, [number]: elements}};
    });
  }

  function updateLimitation(number: string, index: number, spot: number, patch: Partial<Limitation>) {
    setDraft(current => {
      if (!current) return current;
      const elements = (current.claims[number] || []).map((element, position) => {
        if (position !== index) return element;
        const limitations = element.limitations.map((limitation, place) =>
          place === spot ? {...limitation, ...patch} : limitation);
        return {...element, limitations};
      });
      return {...current, claims: {...current.claims, [number]: elements}};
    });
  }

  async function cancelGeneration() {
    // 종속항 대비 취소는 cancelDependentClaims가 따로 처리합니다(저장된 항을 되불러와야 함).
    if (!generating) return;
    cancelRequested.current = true;
    setStage('취소 중');
    uploadController.current?.abort();
    const jobId = activeJob.current;
    if (jobId) {
      try {
        await fetch(`${API}/jobs/${jobId}`, {method: 'DELETE'});
      } catch {
        // The local abort still prevents result retrieval; backend cancellation can be retried.
      }
    }
    setStage('취소됨');
    setMessage('보고서 생성을 취소했습니다. 실행 중인 분석 프로세스도 종료했습니다.');
    setGenerating(false);
  }

  async function cancelDependentClaims() {
    if (!actionBusy || !activeJob.current) return;
    const jobId = activeJob.current;
    cancelRequested.current = true;
    setStage('취소 중');
    try {
      await fetch(`${API}/jobs/${jobId}`, {method: 'DELETE'});
    } catch {
      // 서버가 취소를 받지 못했어도 폴링은 멈춥니다. 취소는 다시 누를 수 있습니다.
    }
    // 판정이 끝난 항은 서버가 저장해 두므로, 취소가 확정될 때까지 기다렸다 결과를 받습니다.
    for (let attempt = 0; attempt < 40; attempt += 1) {
      await sleep(650);
      const statusResponse = await fetch(`${API}/jobs/${jobId}`);
      if (!statusResponse.ok) break;
      const job: Job = await statusResponse.json();
      setStage(job.stage || '취소 중');
      if (job.status === 'running' || job.status === 'cancelling') continue;
      const resultResponse = await fetch(`${API}/jobs/${jobId}/result`);
      if (resultResponse.ok) setResult(await resultResponse.json());
      break;
    }
    await refreshHistory();
    setStage('취소됨');
    setMessage('종속항 대비를 취소했습니다. 판정이 끝난 항은 보고서에 남아 있습니다.');
    setActionBusy(false);
    activeJob.current = null;
  }

  async function removeHistory(id: string) {
    if (!confirm('이 분석 히스토리를 삭제할까요?')) return;
    const response = await fetch(API + '/history/' + id, {method: 'DELETE'});
    if (!response.ok) {
      setMessage('삭제에 실패했습니다.');
      return;
    }
    setHistory(current => current.filter(item => item.job_id !== id));
    setMessage('분석 히스토리를 삭제했습니다.');
  }

  async function clearHistory() {
    if (!history.length || !confirm(`저장된 분석 히스토리 ${history.length}건을 모두 삭제할까요?`)) return;
    const response = await fetch(API + '/history', {method: 'DELETE'});
    const data = await response.json();
    if (!response.ok) {
      setMessage('삭제에 실패했습니다.');
      return;
    }
    setHistory([]);
    setMessage(`분석 히스토리 ${data.removed}건을 삭제했습니다.`);
  }

  async function clearLogs() {
    if (!logs.length || !confirm(`저장된 로그 ${logs.length}건을 모두 삭제할까요?`)) return;
    const response = await fetch(API + '/logs', {method: 'DELETE'});
    const data = await response.json();
    if (!response.ok) {
      setMessage('로그 삭제에 실패했습니다.');
      return;
    }
    setLogs([]);
    setSelectedLog('');
    setLogContent('');
    setMessage(`로그 ${data.removed}건을 삭제했습니다.`);
  }

  async function load(id: string) {
    const response = await fetch(API + '/history/' + id);
    const data = await response.json();
    if (!response.ok || !data.result) {
      setMessage('화면에 표시할 결과가 없습니다.');
      return;
    }
    setMessage('');
    openResult(data.result);
  }

  const refName = (id: string, response: Result) => {
    const mapping = response.claim_mapping.find(item => item.document_id === id);
    return mapping
      ? `인용발명 ${mapping.reference_number}${mapping.document_number ? ` (${mapping.document_number})` : ''}`
      : `문헌 ${id}`;
  };

  const chainText = (report: any, response: Result) => {
    const ids = [report.chain.primary, ...report.chain.secondaries].filter(Boolean);
    const names = ids.map((id: string) => refName(id, response));
    // 주지관용기술을 함께 세운 거절 이유는 인용발명 단독과 다른 거절 이유입니다.
    // 빼고 적으면 화면에서 둘이 구별되지 않습니다.
    const wellKnown: string[] = report.chain.well_known || [];
    if (wellKnown.length) names.push(`주지관용기술 (구성 ${wellKnown.join(', ')})`);
    if (names.length) return names.join(' + ');
    // 주 인용발명이 서지 않아도 구성대비 결과는 아래에 그대로 표시됩니다. 어느 문헌의
    // 대비인지 밝히지 않으면 근거 발췌의 출처가 채택된 인용발명인 것처럼 읽힙니다.
    const referenced: string[] = report.chain.reference_only || [];
    if (referenced.length) {
      const shown = referenced.map((id: string) => refName(id, response)).join(', ');
      return `채택 인용발명 없음 — 아래 구성대비는 ${shown}에 대한 대비 결과이며 모두 미채택입니다`;
    }
    return '채택 인용발명 없음';
  };

  async function searchPriorArt() {
    if (!result || busy) return;
    const jobId = result.job_id;
    const controller = new AbortController();
    priorArtController.current = controller;
    priorArtCancelled.current = false;
    setPriorArtBusy(true);
    setMessage('미커버 구성의 선행기술을 검색 중입니다…');
    try {
      const response = await fetch(`${API}/jobs/${jobId}/prior-art`,
        {method: 'POST', signal: controller.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail);
      const next = await (await fetch(`${API}/jobs/${jobId}/result`)).json();
      setResult(next);
      setMessage(data.message || `선행기술 ${data.hits.length}건을 찾았습니다.`);
    } catch (error: any) {
      // 취소 안내는 cancelPriorArt가 이미 띄웠습니다. 중단된 요청을 실패로 덮어쓰지 않습니다.
      if (priorArtCancelled.current || error?.name === 'AbortError') return;
      setMessage(error.message || '선행기술 검색에 실패했습니다.');
    } finally {
      priorArtController.current = null;
      setPriorArtBusy(false);
    }
  }

  async function cancelPriorArt() {
    if (!priorArtBusy || !result) return;
    priorArtCancelled.current = true;
    setMessage('선행기술 검색을 취소하는 중입니다…');
    try {
      // 로컬 abort만 하면 서버는 그대로 CLI를 물고 있습니다. 먼저 서버에 알려 프로세스를
      // 정리해야 곧바로 다시 검색할 수 있습니다(실행 중이면 서버가 409로 막습니다).
      await fetch(`${API}/jobs/${result.job_id}/prior-art`, {method: 'DELETE'});
    } catch {
      // 서버가 취소를 받지 못해도 아래 abort로 대기는 끝납니다. 취소는 다시 누를 수 있습니다.
    }
    priorArtController.current?.abort();
    setMessage('선행기술 검색을 취소했습니다. 기존 보고서는 그대로입니다.');
    setPriorArtBusy(false);
  }

  async function addDependentClaims() {
    if (!result || !dependentClaims.trim()) return;
    const jobId = result.job_id;
    setActionBusy(true);
    setMessage('');
    setInterrupted('');
    setProgress(null);
    setStage('종속항 구성대비 준비 중');
    cancelRequested.current = false;
    activeJob.current = jobId;
    try {
      const response = await fetch(`${API}/jobs/${jobId}/dependent-claims`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({claims: dependentClaims}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail);
      setDependentClaims('');

      // 종속항 대비도 수 분이 걸립니다. 초기 분석과 같은 폴링으로 진행률을 보여 주고,
      // 취소되더라도 그때까지 확정된 항은 서버가 저장해 두므로 결과를 그대로 불러옵니다.
      while (!cancelRequested.current) {
        await sleep(650);
        const statusResponse = await fetch(`${API}/jobs/${jobId}`);
        const job: Job = await statusResponse.json();
        if (!statusResponse.ok) throw new Error(job.detail || '작업 상태를 확인하지 못했습니다.');
        setStage(job.stage || '구성대비 중');
        setProgress(job.progress || null);
        if (job.status === 'interrupted') {
          setInterrupted(job.error || '종속항 대비가 중단되었습니다.');
          return;
        }
        if (job.status === 'failed') throw new Error(job.error || '종속항 분석에 실패했습니다.');
        if (job.status !== 'completed' && job.status !== 'cancelled') continue;

        const resultResponse = await fetch(`${API}/jobs/${jobId}/result`);
        const nextResult = await resultResponse.json();
        if (!resultResponse.ok) throw new Error(nextResult.detail || '결과를 불러오지 못했습니다.');
        const added = (nextResult.reports || [])
          .map((report: any) => report.claim_number)
          .filter((number: number) => data.added_claims.includes(number));
        setResult(nextResult);
        await refreshHistory();
        if (job.status === 'cancelled') {
          setMessage(added.length
            ? `취소했습니다. 판정이 끝난 청구항 ${added.join(', ')}은 보고서에 남겼습니다.`
            : '취소했습니다. 판정이 끝난 종속항이 없어 보고서는 그대로입니다.');
        } else {
          setMessage(`청구항 ${added.join(', ')}을 보고서에 추가했습니다.`);
        }
        setStage(job.status === 'cancelled' ? '취소됨' : '완료');
        return;
      }
    } catch (error: any) {
      if (!cancelRequested.current) setMessage(error.message || '종속항 분석에 실패했습니다.');
    } finally {
      activeJob.current = null;
      setActionBusy(false);
    }
  }

  // 확정 버튼을 막을지는 매 입력마다 다시 셉니다. 누른 뒤에 알려 주면 고칠 자리를 다시
  // 찾아야 하고, 이 화면은 스크롤이 깁니다.
  const liveProblems = tab === 'decompose' ? decompositionProblems(draft) : [];
  const elementCount = Object.values(draft?.claims || {})
    .reduce((sum, elements) => sum + elements.length, 0);
  const editedCount = Object.entries(draft?.claims || {}).reduce((sum, [number, elements]) => (
    sum + elements.filter((element, index) => JSON.stringify(element)
      !== JSON.stringify(review?.decomposition.claims[number]?.[index])).length
  ), 0);

  // 분해 확인은 확정 대기 중에만 나타납니다. 상시 메뉴로 두면 들어갈 것이 없는 자리가 되고,
  // 대기 중에 빼 두면 다른 탭에 다녀온 사용자가 확정 화면으로 돌아올 길을 잃습니다.
  const navItems: Array<[Tab, string]> = [
    ['analysis', '분석'],
    ...(review ? ([['decompose', '분해 확인']] as Array<[Tab, string]>) : []),
    ['result', '구성대비'],
    ['history', '히스토리'],
    ['logs', '로그'],
    ['settings', '설정'],
  ];

  return (
    <div className="app">
      <header>
        <button className="brand" onClick={() => navigate('analysis')} aria-label="Evidence Forge 홈">
          <span className="mark"><img src="/forge-emblem.svg" alt="" /></span>
          <b>Evidence Forge</b>
        </button>
        <nav aria-label="주요 메뉴">
          {navItems.map(([id, label]) => (
            <button
              key={id}
              className={tab === id ? 'active' : ''}
              disabled={id === 'result' && !result}
              onClick={() => navigate(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <span className={`status ${generating ? 'working' : ''}`}>
          <i /> {generating ? stage : 'READY'}
        </span>
      </header>

      {tab === 'analysis' && (
        <main className="panel analysis-main">
          <section className="page-heading compact-heading">
            <div>
              <p className="eyebrow">NEW REPORT</p>
              <h1>분석</h1>
            </div>
            <span className="input-count">{files.length}/7 PDF</span>
          </section>

          <section className="grid" aria-label="분석 자료 입력">
            <div className="card input-card">
              <div className="card-heading">
                <label htmlFor="claim-input">청구항</label>
                <span>{claims.length}자</span>
              </div>
              <textarea
                id="claim-input"
                value={claims}
                disabled={generating}
                onChange={event => setClaims(event.target.value)}
                placeholder="(A), (B), (C)로 구성요소를 구분해 입력하세요."
              />
              <div className="field-row">
                <label htmlFor="priority-date">대상 출원일·우선일</label>
                <input
                  id="priority-date"
                  type="date"
                  value={priorityDate}
                  disabled={generating}
                  onChange={event => setPriorityDate(event.target.value)}
                />
                <small>
                  선택 입력. 넣으면 각 인용발명을 통상 선행기술·후공개 선출원·후행 문헌·날짜
                  불명으로 갈라 보고서에 표시합니다. 비워 두면 적격성을 가리지 않고 구성대비만
                  수행합니다.
                </small>
              </div>
            </div>

            <div className="card upload-card">
              <div className="card-heading">
                <label>인용발명·참조 문헌</label>
                <span>PDF · 최대 7개</span>
              </div>
              <label className="drop">
                <input
                  aria-label="참조 문헌 PDF 선택"
                  type="file"
                  accept="application/pdf"
                  multiple
                  disabled={generating}
                  onChange={event => setFiles(Array.from(event.target.files || []).slice(0, 7))}
                />
                <span className="drop-icon" aria-hidden="true">+</span>
                <strong>PDF 선택</strong>
                <small>파일당 최대 25MB</small>
                </label>
              <div className="file-list">
                {files.map((file, index) => (
                  <div key={file.name + file.size + file.lastModified} className="file">
                    <span>PDF</span>
                    <span className="file-name">{file.name}</span>
                    <button
                      type="button"
                      aria-label={`${file.name} 제거`}
                      disabled={generating}
                      onClick={() => setFiles(files.filter((_, itemIndex) => itemIndex !== index))}
                    >×</button>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className={`run-dock ${generating ? 'is-running' : ''}`} aria-live="polite">
            <div className="run-status">
              <span className="run-dot" />
              <div>
                <strong>{generating ? stage : '보고서 준비'}</strong>
                <small>{generating ? '분석 프로세스가 실행 중입니다.' : '입력한 청구항과 문헌으로 구성대비합니다.'}</small>
                {generating && <ProgressBar progress={progress} />}
              </div>
            </div>
            <div className="run-actions">
              {generating && (
                <button type="button" className="cancel" onClick={cancelGeneration}>취소</button>
              )}
              <button
                type="button"
                className="primary generate"
                disabled={generating || !claims.trim() || !files.length}
                onClick={run}
              >
                {generating ? '생성 중' : '보고서 생성'} <span aria-hidden="true">→</span>
              </button>
            </div>
          </section>
          {review && (
            <div className="notice gate-pending" role="status">
              <span>청구항 분해 제안이 준비되었습니다. 확인하고 확정해야 구성대비가 시작됩니다.</span>
              <button type="button" className="ghost" onClick={() => navigate('decompose')}>
                분해 확인하기
              </button>
            </div>
          )}
          {interrupted && <div className="job-interrupted" role="status">{interrupted}</div>}
          {message && <div className="notice" role="status">{message}</div>}
        </main>
      )}

      {tab === 'decompose' && review && draft && (
        <main className="panel decompose-main">
          <section className="page-heading compact-heading">
            <div>
              <p className="eyebrow">STEP 1 · DECOMPOSITION</p>
              <h1>청구항 분해 확인</h1>
            </div>
            <span className="input-count">
              구성 {elementCount}개{editedCount > 0 && ` · ${editedCount}개 수정됨`}
            </span>
          </section>

          <div className="gate-brief">
            <strong>구성대비는 아직 시작하지 않았습니다.</strong>
            <span>
              여기서 확정한 분해가 이후 모든 단계의 기준이 됩니다. 한정 문언과 검색어가 문헌에서
              읽어 올 문단과 비교 캐시 키를 정하므로, 확정한 뒤에 고치면 판정을 처음부터 다시
              받아야 합니다. 청구항 원문을 옆에 두고 <b>한정이 청구항에 실제로 적힌 것인지</b>를
              보아 주십시오.
            </span>
          </div>

          {review.warnings.map((warning, index) => (
            <div key={index} className="gate-warning" role="status">{warning}</div>
          ))}

          <AliasGate draft={draft} setDraft={setDraft} />

          {/* 고칠 곳은 입력하는 동안 계속 띄웁니다. 확정 버튼만 잠가 두면 왜 눌리지 않는지
              알 수 없고, 이 화면은 스크롤이 길어 짐작으로 찾기 어렵습니다. 서버가 거절한
              사유(reviewProblems)는 화면 검증을 통과한 뒤에만 남으므로 겹치지 않습니다. */}
          {(liveProblems.length > 0 || reviewProblems.length > 0) && (
            <div className="gate-problems" role="alert">
              <strong>
                {liveProblems.length ? '확정하기 전에 고쳐야 합니다' : '서버가 확정을 거절했습니다'}
              </strong>
              <ul>
                {(liveProblems.length ? liveProblems : reviewProblems)
                  .map((problem, index) => <li key={index}>{problem}</li>)}
              </ul>
            </div>
          )}

          <section className="decompose-grid">
            <aside className="card claims-source">
              <div className="card-heading">
                <label>청구항 원문</label>
                <span>대조용</span>
              </div>
              <pre>{review.claims_text}</pre>
            </aside>

            <div className="decompose-claims">
              {Object.entries(draft.claims)
                .sort((left, right) => Number(left[0]) - Number(right[0]))
                .map(([number, elements]) => (
                  <section key={number} className="card claim-decompose">
                    <div className="card-heading">
                      <label>청구항 {number}</label>
                      <span>구성 {elements.length}개</span>
                    </div>

                    {elements.map((element, index) => {
                      const proposed = review.decomposition.claims[number]?.[index];
                      const edited = JSON.stringify(element) !== JSON.stringify(proposed);
                      return (
                        <article key={element.label + index} className={`element-edit ${edited ? 'is-edited' : ''}`}>
                          <div className="element-head">
                            <b>{element.label}</b>
                            {edited && <span className="edited-chip">수정됨</span>}
                            <label className="element-field">
                              중요도
                              <select
                                value={element.importance}
                                disabled={confirming}
                                onChange={event =>
                                  updateElement(number, index, {importance: Number(event.target.value)})}
                              >
                                {[1, 2, 3, 4, 5].map(value => (
                                  <option key={value} value={value}>{value}</option>
                                ))}
                              </select>
                            </label>
                            <label className="element-field checkbox">
                              <input
                                type="checkbox"
                                checked={element.is_sub}
                                disabled={confirming}
                                onChange={event =>
                                  updateElement(number, index, {is_sub: event.target.checked})}
                              />
                              하위 제한
                            </label>
                          </div>

                          {/* 구성 원문은 청구항 파서가 만든 뼈대라 고칠 수 없습니다. 읽기 전용인
                              이유를 함께 적어 두지 않으면 편집란을 찾다가 확정을 미루게 됩니다. */}
                          <p className="element-source">{element.text}</p>

                          <div className="limit-block">
                            <div className="limit-heading">
                              <span>한정 {element.limitations.length}/{MAX_LIMITATIONS}</span>
                              <button
                                type="button"
                                className="ghost tiny"
                                disabled={confirming || element.limitations.length >= MAX_LIMITATIONS}
                                onClick={() => updateElement(number, index, {
                                  limitations: [...element.limitations,
                                                {text: '', kind: 'core', alternative_group: ''}],
                                })}
                              >한정 추가</button>
                            </div>
                            {element.limitations.map((limitation, spot) => (
                              <div key={spot} className="limit-row">
                                <textarea
                                  aria-label={`${element.label} 한정 ${spot + 1} 문언`}
                                  value={limitation.text}
                                  disabled={confirming}
                                  rows={2}
                                  onChange={event =>
                                    updateLimitation(number, index, spot, {text: event.target.value})}
                                />
                                <div className="limit-meta">
                                  <select
                                    aria-label={`${element.label} 한정 ${spot + 1} 종류`}
                                    value={limitation.kind}
                                    disabled={confirming}
                                    onChange={event => updateLimitation(number, index, spot,
                                      {kind: event.target.value as Limitation['kind']})}
                                  >
                                    <option value="core">core · 동작·구조</option>
                                    <option value="qualifier">qualifier · 조건·수치</option>
                                  </select>
                                  <input
                                    aria-label={`${element.label} 한정 ${spot + 1} 대안군`}
                                    value={limitation.alternative_group}
                                    disabled={confirming}
                                    placeholder="대안군(선택)"
                                    onChange={event => updateLimitation(number, index, spot,
                                      {alternative_group: event.target.value})}
                                  />
                                  <button
                                    type="button"
                                    className="danger tiny"
                                    disabled={confirming}
                                    aria-label={`${element.label} 한정 ${spot + 1} 삭제`}
                                    onClick={() => updateElement(number, index, {
                                      limitations: element.limitations
                                        .filter((_, place) => place !== spot),
                                    })}
                                  >삭제</button>
                                </div>
                              </div>
                            ))}
                            {!element.limitations.length && (
                              <p className="limit-empty">
                                한정이 없으면 구성 원문 한 줄을 통째로 점검하게 됩니다. 최소 하나가 필요합니다.
                              </p>
                            )}
                          </div>

                          <div className="term-block">
                            <span className="term-heading">검색어 {element.search_terms.length}/{MAX_SEARCH_TERMS}</span>
                            <TermEditor
                              terms={element.search_terms}
                              disabled={confirming}
                              onChange={next => updateElement(number, index, {search_terms: next})}
                            />
                          </div>
                        </article>
                      );
                    })}
                  </section>
                ))}
            </div>
          </section>

          <section className="run-dock gate-dock" aria-live="polite">
            <div className="run-status">
              <span className="run-dot" />
              <div>
                <strong>{confirming ? '구성대비 시작 중' : '분해 확정 대기'}</strong>
                <small>
                  {liveProblems.length
                    ? `고칠 곳 ${liveProblems.length}군데가 남았습니다.`
                    : '확정하면 곧바로 문헌 대비가 시작됩니다.'}
                </small>
              </div>
            </div>
            <div className="run-actions">
              <button type="button" className="cancel" disabled={confirming} onClick={cancelDecomposition}>
                취소
              </button>
              <button
                type="button"
                className="ghost"
                disabled={confirming || !editedCount}
                onClick={() => {
                  setDraft(cloneDecomposition(review.decomposition));
                  setReviewProblems([]);
                }}
              >제안으로 되돌리기</button>
              <button
                type="button"
                className="primary generate"
                disabled={confirming || liveProblems.length > 0}
                onClick={confirmDecomposition}
              >
                {confirming ? '시작 중' : '확정하고 구성대비'} <span aria-hidden="true">→</span>
              </button>
            </div>
          </section>
        </main>
      )}

      {tab === 'result' && (
        <main className="panel result-main">
          <section className="page-heading result-heading">
            <div>
              <p className="eyebrow">COMPARISON</p>
              <h1>구성대비</h1>
            </div>
            {result && (
              <div className="result-actions">
                {priorArtBusy && (
                  <button type="button" className="cancel" onClick={cancelPriorArt}>취소</button>
                )}
                <button type="button" className="ghost" disabled={busy} onClick={searchPriorArt}>
                  {priorArtBusy ? '검색 중…' : '부족한 구성 검색'}
                </button>
                <a className="download" href={`${API}/jobs/${result.job_id}/download?format=md`}>내려받기</a>
              </div>
            )}
          </section>

          {result && (
            <>
              <section className="result-overview" aria-label="보고서 요약">
                <div><strong>{result.reports?.length || 0}</strong><span>청구항</span></div>
                <div><strong>{result.claim_mapping?.length || 0}</strong><span>인용발명</span></div>
                <div><strong>{result.cached_claims?.length || 0}</strong><span>캐시 재사용</span></div>
              </section>

              <section className="mapping card">
                <h2>문헌</h2>
                {result.claim_mapping.map(mapping => (
                  <div key={mapping.document_id} className="mapping-row">
                    <b>인용발명 {mapping.reference_number}</b>
                    <span>{mapping.filename}</span>
                    <small>{[
                      mapping.document_number,
                      (mapping.publication_date || mapping.filing_date) && `공개·제출일 ${mapping.publication_date || mapping.filing_date}`,
                      mapping.role || mapping.document_type,
                    ].filter(Boolean).join(' · ')}</small>
                    {mapping.source_file && (
                      <a href={`${API}/jobs/${result.job_id}/sources/${mapping.document_id}`} target="_blank" rel="noreferrer">원문</a>
                    )}
                  </div>
                ))}
              </section>

              {!result.reports && (
                <div className="notice">이전 형식의 히스토리입니다. 내려받기로 확인하세요.</div>
              )}

              {(result.reports || []).map(report => (
                <section key={report.claim_number} className="claim-report">
                  <div className="claim-title">
                    <div>
                      <p className="claim-kicker">
                        <span className="claim-label">청구항 {report.claim_number}</span>
                        {report.depends_on && <span className="claim-dependency">{report.depends_on}항 종속</span>}
                      </p>
                      <h2>{report.track === 'analysis_incomplete'
                        ? '구성대비 미완료'
                        : chainText(report, result)}</h2>
                      <small>{report.track === 'analysis_incomplete'
                        ? '판정을 받지 못해 결론을 만들지 않았습니다'
                        : report.summary_similarity}</small>
                    </div>
                  </div>

                  {report.conclusion && <p className="claim-conclusion">{report.conclusion}</p>}
                  {report.coverage_summary && <p className="claim-coverage">{report.coverage_summary}</p>}

                  <div className="results">
                    {report.claims.map((claim: any, index: number) => (
                      <article key={claim.label || index} className="card claim">
                        <div className="claim-head">
                          <span className={`badge ${GRADE_TONE[claim.emoji] || 'tone-none'}`}
                                title={claim.grade || claim.status || ''}>
                            {claim.is_preamble ? '전제부'
                              : (claim.label || String.fromCharCode(65 + index))}</span>
                          {/* 백분율 대신 셀 수 있는 값을 보여 준다. 분자·분모가 그대로 보여야
                              아래 근거와 대조해 검증할 수 있다. */}
                          <strong>{claim.total_limitations
                            ? `한정 ${claim.disclosed_limitations}/${claim.total_limitations}`
                            : '—'}</strong>
                          {/* 등급 색은 왼쪽 라벨 배지가 지고 있으므로 이모지는 붙이지 않습니다.
                              같은 정보를 색과 이모지로 두 번 말하면 읽는 눈만 늘어납니다. */}
                          <span className="quality">{claim.grade || claim.status}</span>
                          {claim.evidence_locations > 0 && (
                            <span className="reference-chip">근거 {claim.evidence_locations}곳</span>
                          )}
                          {/* 개시 수에서 빼지 않고 옆에 붙인다. 미완료는 미개시가 아니라
                              "확인하지 못했다"이고, 분자에서 빼면 없는 누락을 지어내게 된다.
                              그렇다고 적지 않으면 "3/3 개시"만 남아 그 셋이 무엇으로
                              확인됐는지 알 수 없다. */}
                          {claim.unverified_limitations?.length > 0 && (
                            <span className="reference-chip chip-warn"
                                  title={claim.unverified_limitations.join('\n')}>
                              ⚠️ 의미검증 미완료 {claim.unverified_limitations.length}건</span>
                          )}
                          {claim.adopted_reference && <span className="reference-chip">인용발명 {claim.adopted_reference}</span>}
                          {claim.combination && <span className="reference-chip">결합</span>}
                        </div>
                        <p>{claim.claim}</p>
                        {claim.narrative && (
                          <section className="reasoning" aria-label="구성대비">
                            <div className="narrative">{claim.narrative}</div>
                          </section>
                        )}
                        {claim.difference && <div className="diff"><b>차이점</b> {claim.difference}</div>}
                        <VerificationTrail trail={claim.trail} />
                      </article>
                    ))}
                  </div>

                  <div className="summary">
                    <h3>종합 분석 요약</h3>
                    {report.summary_similarity && <p><b>유사점</b> {report.summary_similarity}</p>}
                    {report.summary_difference && <p><b>차이점</b> {report.summary_difference}</p>}
                  </div>
                </section>
              ))}

              <section className="card dependent-add">
                <h2>종속항 추가</h2>
                <textarea
                  value={dependentClaims}
                  onChange={event => setDependentClaims(event.target.value)}
                  placeholder={'【청구항 2】\n제1항에 있어서, (A) 추가 한정…'}
                />
                <div className="action">
                  <span className="hint">
                    {actionBusy ? stage : '기존 문헌을 그대로 사용합니다.'}
                    {actionBusy && <ProgressBar progress={progress} />}
                  </span>
                  {actionBusy && (
                    <button type="button" className="cancel" onClick={cancelDependentClaims}>취소</button>
                  )}
                  <button className="primary" disabled={busy || !dependentClaims.trim()} onClick={addDependentClaims}>
                    {actionBusy ? '비교 중' : '일괄 추가'}
                  </button>
                </div>
              </section>

              {!!result.prior_art?.length && (
                <section className="card prior-art">
                  <h2>추가 선행기술</h2>
                  <p className="hint">
                    웹 검색 결과입니다. 아래 표시는 제시된 URL을 열어 문헌번호가 그 페이지에
                    있는지만 확인한 것이며, 선행기술 적격성 판단이 아닙니다.
                  </p>
                  {result.prior_art.map((hit: any, index: number) => (
                    <div key={index} className="mapping-row">
                      <b>{hit.claim_number ? `청구항 ${hit.claim_number} ` : ''}({hit.label})</b>
                      <span>{hit.document_number || hit.title}</span>
                      <span className={`verify-chip ${PRIOR_ART_VERIFY[hit.verify]?.[0] || 'unchecked'}`}
                            title={hit.verify_note || ''}>
                        {PRIOR_ART_VERIFY[hit.verify]?.[1] || '미확인'}
                      </span>
                      <small>{hit.correspondence}</small>
                      {hit.url && <a href={hit.url} target="_blank" rel="noreferrer">열기</a>}
                    </div>
                  ))}
                </section>
              )}

              {!!result.validation?.length && (
                <details className="validation">
                  <summary>검증 참고 {result.validation.length}건</summary>
                  {result.validation.map((item, index) => <p key={index}>{item}</p>)}
                </details>
              )}
            </>
          )}
          {message && <div className="notice" role="status">{message}</div>}
        </main>
      )}

      {tab === 'history' && (
        <main className="panel">
          <section className="page-heading">
            <div><p className="eyebrow">ARCHIVE</p><h1>히스토리</h1></div>
            <button className="danger" disabled={!history.length} onClick={clearHistory}>전체 삭제</button>
          </section>
          <div className="card list">
            {history.length ? history.map(item => (
              <div key={item.job_id} className="list-row">
                <button onClick={() => load(item.job_id)}>
                  <b>{item.created_at.slice(0, 16).replace('T', ' ')}</b>
                  <span>{item.documents?.join(', ')}</span>
                  <small>보기 →</small>
                </button>
                <button className="row-del" title="히스토리 삭제" onClick={() => removeHistory(item.job_id)}>×</button>
              </div>
            )) : <p className="empty">저장된 분석이 없습니다.</p>}
          </div>
          {message && <div className="notice">{message}</div>}
        </main>
      )}

      {tab === 'settings' && (
        <main className="panel">
          <section className="page-heading"><div><p className="eyebrow">CONFIGURATION</p><h1>설정</h1></div></section>
          <div className="card settings">
            <label>연동 CLI
              <select value={settings.provider} onChange={event => changeProvider(event.target.value)}>
                <option value="agy">agy</option>
                <option value="claude">Claude</option>
                <option value="gpt">Codex</option>
              </select>
            </label>
            <label>모델
              <select
                value={settings.model}
                onChange={event => setSettings({...settings, model: event.target.value})}
              >
                {!models.includes(settings.model) && settings.model && (
                  <option value={settings.model}>{settings.model}</option>
                )}
                {models.map(model => <option key={model} value={model}>{model}</option>)}
              </select>
              <button className="ghost" onClick={() => loadModels(settings, true)}>새로고침</button>
            </label>
            <label>분석 지침
              <textarea className="prompt-input" value={settings.prompt} onChange={event => setSettings({...settings, prompt: event.target.value})} />
            </label>
            <div className="action">
              <button onClick={testSettings}>연결 테스트</button>
              <button onClick={resetPrompt}>기본값</button>
              <button className="primary" onClick={() => saveSettings()}>저장</button>
            </div>
            {message && <div className="notice">{message}</div>}
          </div>
        </main>
      )}

      {tab === 'logs' && (
        <main className="panel">
          <section className="page-heading">
            <div><p className="eyebrow">OBSERVABILITY</p><h1>로그</h1></div>
            <button className="danger" disabled={!logs.length} onClick={clearLogs}>전체 삭제</button>
          </section>
          <section className="card log-viewer">
            <aside className="log-list" aria-label="저장된 로그">
              {logs.length ? logs.map(item => (
                <button
                  key={item.job_id}
                  className={selectedLog === item.job_id ? 'active' : ''}
                  onClick={() => loadLog(item.job_id)}
                >
                  <b>{item.job_id.slice(0, 8)}</b>
                  <span>{formatLogDate(item.updated_at)}</span>
                  <small>{item.size.toLocaleString()} bytes</small>
                </button>
              )) : <p className="empty">저장된 로그가 없습니다.</p>}
            </aside>
            <div className="log-content">
              <div className="log-toolbar">
                <b>{selectedLog || '로그를 선택하세요'}</b>
                <button className="ghost" disabled={logsLoading} onClick={() => refreshLogs()}>
                  {logsLoading ? '불러오는 중' : '새로고침'}
                </button>
              </div>
              <pre>{logContent || (logsLoading ? '로그를 불러오는 중입니다…' : '표시할 로그가 없습니다.')}</pre>
            </div>
          </section>
          {message && <div className="notice">{message}</div>}
        </main>
      )}

      <footer>All rights reserved by Aidan</footer>
    </div>
  );
}

// 테스트가 이 모듈을 불러와도 앱 전체를 띄우지 않게 합니다. 컴포넌트 하나를 확인하려고
// 브라우저 전체를 흉내 낼 이유가 없습니다.
const root = document.getElementById('root');
if (root) createRoot(root).render(<App />);
