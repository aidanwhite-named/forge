/** 지시 관계 확정 관문의 상호작용.
 *
 * 이 자리의 결함은 백엔드 테스트로 잡히지 않는다. 실제로 그랬다 — API 테스트가
 * selected_source를 코드로 직접 넣어 확정 경로를 통과시켰는데, 화면에서는 후보가 하나인
 * 관계의 체크박스가 잠긴 채로 남아 **확정할 방법이 없었다.** 실제 사건의 후보는 대부분
 * 그 형태다.
 */
import {useState} from 'react';
import {render, screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {describe, expect, test} from 'vitest';
import {AliasGate} from './main';

type Alias = {target: string; term: string; candidates: string[];
              selected_source: string; confirmed: boolean};

function Harness({alias}: {alias: Alias}) {
  const [draft, setDraft] = useState<Parameters<typeof AliasGate>[0]['draft']>(
    {version: 't', claims: {}, aliases: {'1': [alias]}});
  return (
    <>
      <AliasGate draft={draft} setDraft={setDraft} />
      <output>{JSON.stringify(draft?.aliases?.['1'][0])}</output>
    </>
  );
}

const state = () => JSON.parse(screen.getByRole('status').textContent || '{}');

describe('후보가 하나인 관계', () => {
  const single: Alias = {target: 'C', term: '가시 두상', candidates: ['A'],
                         selected_source: 'A', confirmed: false};

  test('고를 것이 없으므로 선택창을 띄우지 않는다', () => {
    render(<Harness alias={single} />);
    expect(screen.queryByRole('combobox')).toBeNull();
  });

  test('체크만으로 확정된다', async () => {
    render(<Harness alias={single} />);
    await userEvent.click(screen.getByRole('checkbox'));
    expect(state()).toMatchObject({confirmed: true, selected_source: 'A'});
  });

  test('selected_source가 비어 와도 체크가 잠기지 않는다', async () => {
    render(<Harness alias={{...single, selected_source: ''}} />);
    const box = screen.getByRole('checkbox') as HTMLInputElement;

    expect(box.disabled).toBe(false);
    await userEvent.click(box);
    expect(state()).toMatchObject({confirmed: true, selected_source: 'A'});
  });
});

describe('후보가 둘 이상인 관계', () => {
  const many: Alias = {target: 'C', term: '가시 두상', candidates: ['A', 'B'],
                       selected_source: '', confirmed: false};

  test('고르기 전에는 확정할 수 없다', () => {
    render(<Harness alias={many} />);
    expect((screen.getByRole('checkbox') as HTMLInputElement).disabled).toBe(true);
  });

  test('고른 뒤에야 확정할 수 있다', async () => {
    render(<Harness alias={many} />);
    await userEvent.selectOptions(screen.getByRole('combobox'), 'B');
    await userEvent.click(screen.getByRole('checkbox'));

    expect(state()).toMatchObject({selected_source: 'B', confirmed: true});
  });

  test('고른 값을 바꾸면 확정이 풀린다', async () => {
    render(<Harness alias={many} />);
    await userEvent.selectOptions(screen.getByRole('combobox'), 'A');
    await userEvent.click(screen.getByRole('checkbox'));
    expect(state().confirmed).toBe(true);

    // 앞선 확정은 **다른 후보**에 대한 판단이다. 그대로 이어 붙이면 하지 않은 확정이 된다.
    await userEvent.selectOptions(screen.getByRole('combobox'), 'B');
    expect(state()).toMatchObject({selected_source: 'B', confirmed: false});
  });
});
