/** 판정 경위의 한정별 투표 렌더링.
 *
 * 백엔드가 trail.tallies를 실어 보내도 화면이 읽지 않으면 사용자에게는 없는 것과 같다.
 * 실제로 그랬다 — Markdown에는 한정별 투표가 나가는데 화면에는 구성 단위 비율 한 줄만
 * 남아, "2대 1로 갈린 미개시"와 "3대 0으로 일치한 미개시"가 화면에서 같아 보였다.
 */
import {render, screen} from '@testing-library/react';
import {describe, expect, test} from 'vitest';
import {VerificationTrail, _tallyKind} from './main';

const votes = (...verdicts: string[]) => ({
  total: verdicts.length,
  votes: verdicts.map((verdict, sample) => ({sample, verdict})),
});

const trail = (tallies: any[], extra: object = {}) => [{
  document_id: '1', reference_number: 1, sample_count: 3,
  sample_unanimous: 0, sample_requirements: tallies.length, steps: [], tallies, ...extra,
}];

describe('한정별 투표', () => {
  test('갈린 표를 몇 대 몇으로 적는다', () => {
    render(<VerificationTrail trail={trail([
      [0, '가시 두상 영역을 추출함', votes('disclosed', 'missing', 'missing')]])} />);

    expect(screen.getByText(/한정 #0 「가시 두상 영역을 추출함」/)).toBeDefined();
    expect(screen.getByText(/개시 1표 · 미개시 2표 \(표본 3회\)/)).toBeDefined();
  });

  test('무응답·판독불가를 미개시로 세지 않는다', () => {
    render(<VerificationTrail trail={trail([
      [0, '기준점 영역으로 한정함', votes('disclosed', 'absent', 'invalid')]])} />);

    expect(screen.getByText(/개시 1표 · 미개시 0표 · 무응답 1표 · 판독불가 1표/)).toBeDefined();
  });

  test('구성 단위 지표가 없어도 투표만으로 경위를 띄운다', () => {
    // 이 픽스처는 sample_requirements가 0이라 셀 단위 불안정 표시(_split)가 서지 않는다.
    // 투표를 rows 필터에 넣지 않으면 화면에서 통째로 사라지는 자리다.
    render(<VerificationTrail trail={trail([
      [0, '가시 두상 영역을 추출함', votes('disclosed', 'missing', 'missing')]],
      {sample_requirements: 0, sample_unanimous: 0})} />);

    expect(screen.getByLabelText('판정 경위')).toBeDefined();
    expect(screen.getByText(/개시 1표 · 미개시 2표/)).toBeDefined();
  });

  test('투표도 단계도 없는 셀은 아무것도 띄우지 않는다', () => {
    const {container} = render(<VerificationTrail trail={trail([], {
      sample_requirements: 3, sample_unanimous: 3})} />);

    expect(container.querySelector('.trail')).toBeNull();
  });
});

describe('왜 만장일치가 아닌지', () => {
  test('판단이 나뉜 것과 답을 받지 못한 것을 가른다', () => {
    // 앞은 사람이 원문을 봐야 하고 뒤는 다시 물으면 된다. 후속 조치가 다르다.
    expect(_tallyKind(votes('disclosed', 'missing', 'missing'))).toBe('판정 불일치');
    expect(_tallyKind(votes('disclosed', 'absent', 'invalid'))).toBe('응답 결손');
    expect(_tallyKind(votes('disclosed', 'missing', 'absent'))).toBe('판정 불일치 · 응답 결손');
  });

  test('셀 머리줄은 "갈렸다"고 단정하지 않는다', () => {
    render(<VerificationTrail trail={trail([
      [0, '기준점 영역으로 한정함', votes('disclosed', 'absent', 'invalid')]])} />);

    expect(screen.getByText(/초기 비교 결과가 불안정합니다/)).toBeDefined();
    expect(screen.queryByText(/표본이 갈렸습니다/)).toBeNull();
  });
});
