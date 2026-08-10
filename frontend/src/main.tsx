import React, {useEffect, useRef, useState} from 'react';
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

type Tab = 'analysis' | 'result' | 'history' | 'logs' | 'settings';
type Settings = {provider: string; model: string; prompt: string};
type LogItem = {job_id: string; size: number; updated_at?: string};
type Progress = {done: number | null; total: number};
// detail은 응답이 실패했을 때 FastAPI가 돌려주는 오류 메시지 자리입니다.
type Job = {status: string; stage?: string; error?: string; progress?: Progress; detail?: string};
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
// 선행기술 결과의 실재 확인 표시 → [CSS 클래스, 라벨]. 확인하지 못한 것도 감추지 않습니다.
const PRIOR_ART_VERIFY: Record<string, [string, string]> = {
  verified: ['verified', '✅ 확인됨'],
  mismatch: ['mismatch', '⚠️ 번호 불일치'],
  unreachable: ['unchecked', '❔ 확인 불가'],
  unchecked: ['unchecked', '❔ 미확인'],
};

const formatLogDate = (value?: string) => {
  if (!value) return '수정 시각 없음';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '수정 시각 없음' : date.toLocaleString('ko-KR');
};

function App() {
  const [tab, setTab] = useState<Tab>('analysis');
  const [claims, setClaims] = useState('');
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
      setModels(response.ok && Array.isArray(data.models) ? data.models : []);
      if (refresh) {
        setMessage(data.models?.length
          ? `모델 ${data.models.length}개를 불러왔습니다.`
          : '모델 목록을 불러오지 못했습니다. 모델명을 직접 입력하세요.');
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

  async function clearCache() {
    if (!confirm('저장된 판정 캐시를 모두 지울까요? 다음 분석은 전부 다시 판정합니다.')) return;
    const response = await fetch(API + '/cache', {method: 'DELETE'});
    const data = await response.json();
    setMessage(response.ok
      ? `판정 캐시 ${data.removed}건을 지웠습니다.`
      : '판정 캐시를 지우지 못했습니다.');
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
      files.forEach(file => form.append('pdf_files', file));
      uploadController.current = new AbortController();
      const startResponse = await fetch(`${API}/jobs/${prepared.job_id}/start`, {
        method: 'POST',
        body: form,
        signal: uploadController.current.signal,
      });
      const started = await startResponse.json();
      if (!startResponse.ok) throw new Error(started.detail || '보고서 생성을 시작하지 못했습니다.');

      while (!cancelRequested.current) {
        await sleep(650);
        const statusResponse = await fetch(`${API}/jobs/${prepared.job_id}`);
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
        if (job.status !== 'completed') continue;

        const resultResponse = await fetch(`${API}/jobs/${prepared.job_id}/result`);
        const nextResult = await resultResponse.json();
        if (!resultResponse.ok) throw new Error(nextResult.detail || '결과를 불러오지 못했습니다.');
        setStage('완료');
        setMessage('보고서가 생성되었습니다.');
        await refreshHistory();
        openResult(nextResult);
        return;
      }
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

  const navItems: Array<[Tab, string]> = [
    ['analysis', '분석'],
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
          {interrupted && <div className="job-interrupted" role="status">{interrupted}</div>}
          {message && <div className="notice" role="status">{message}</div>}
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
                          <span className="badge">{claim.is_preamble ? '전제부'
                            : (claim.label || String.fromCharCode(65 + index))}</span>
                          {/* 백분율 대신 셀 수 있는 값을 보여 준다. 분자·분모가 그대로 보여야
                              아래 근거와 대조해 검증할 수 있다. */}
                          <strong>{claim.total_limitations
                            ? `한정 ${claim.disclosed_limitations}/${claim.total_limitations}`
                            : '—'}</strong>
                          <span className="quality">{claim.emoji} {claim.grade || claim.status}</span>
                          {claim.evidence_locations > 0 && (
                            <span className="reference-chip">근거 {claim.evidence_locations}곳</span>
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
                value={models.includes(settings.model) ? settings.model : '__custom__'}
                onChange={event => setSettings({...settings, model: event.target.value === '__custom__' ? '' : event.target.value})}
              >
                {models.map(model => <option key={model} value={model}>{model}</option>)}
                <option value="__custom__">직접 입력</option>
              </select>
              <button className="ghost" onClick={() => loadModels(settings, true)}>새로고침</button>
            </label>
            {!models.includes(settings.model) && (
              <label>모델명
                <input value={settings.model} onChange={event => setSettings({...settings, model: event.target.value})} />
              </label>
            )}
            <label>분석 지침
              <textarea className="prompt-input" value={settings.prompt} onChange={event => setSettings({...settings, prompt: event.target.value})} />
            </label>
            <div className="action">
              <span className="hint">
                판정 캐시에는 업로드한 문헌의 원문 발췌가 남습니다. 저장물을 완전히 비우려면
                여기서 지우세요.
              </span>
              <button onClick={testSettings}>연결 테스트</button>
              <button onClick={resetPrompt}>기본값</button>
              <button className="danger" onClick={clearCache}>판정 캐시 비우기</button>
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

createRoot(document.getElementById('root')!).render(<App />);
