import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture(autouse=True)
def isolate_app_data(tmp_path, monkeypatch):
    """테스트가 Forge 루트의 실제 로그·히스토리와 판정 캐시를 건드리지 않도록 격리한다."""
    for name in ("LOG_DIR", "HISTORY_DIR", "JOBS_DIR", "DATA_DIR"):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(f"app.main.{name}", directory)
    cache_dir = tmp_path / "comparison_cache"
    cache_dir.mkdir()
    monkeypatch.setattr("app.cache.CACHE_DIR", cache_dir)
    entailment_cache = tmp_path / "entailment_cache"
    entailment_cache.mkdir()
    monkeypatch.setattr("app.entailment.CACHE_DIR", entailment_cache)
    monkeypatch.setattr("app.cache.ENTAILMENT_CACHE_DIR", entailment_cache)
    # 분해 캐시도 격리합니다. 빠뜨리면 테스트가 실제 저장소에 분해를 남기고, 다음 테스트가
    # 그것을 재사용해 "LLM을 몇 번 불렀는가"를 검사하는 테스트들이 서로 간섭합니다.
    decomposition_cache = tmp_path / "decomposition_cache"
    decomposition_cache.mkdir()
    monkeypatch.setattr("app.cache.DECOMPOSITION_CACHE_DIR", decomposition_cache)
    # 선행기술 결과 검증은 외부 웹을 호출합니다. 테스트에서 나가지 않도록 막습니다.
    monkeypatch.setattr("app.priorart.verify_hits", lambda hits: None)


@pytest.fixture(autouse=True)
def single_sample(monkeypatch):
    """자기일관성 샘플링은 런타임 노브다. 테스트는 1회로 고정한다.

    기본값(3회)을 그대로 두면 "셀당 CLI 1회"를 검사하는 테스트들이 호출 수만 3배로 보고
    실패한다. 그 테스트들이 지키려는 것은 중복 호출이 없다는 사실이지 절대 호출 수가 아니다.
    샘플링 자체는 samples를 명시하는 전용 테스트에서 검증한다.
    """
    monkeypatch.setattr("app.compare.COMPARE_SAMPLES", 1)
    monkeypatch.setattr("app.cache.COMPARE_SAMPLES", 1)


@pytest.fixture(autouse=True)
def block_the_real_cli(monkeypatch):
    """어떤 테스트도 실제 CLI를 실행하지 않도록 막는다.

    스텁을 깜빡한 단계가 생기면 조용히 느려지는 대신 즉시 실패하게 한다.
    각 테스트는 필요한 단계만 자기 스텁으로 덮어쓴다.
    """
    def forbidden(prompt, expect="claims"):
        raise AssertionError(f"테스트에서 실제 CLI를 호출했습니다 (expect={expect}).")

    for module in ("compare", "claims", "priorart", "entailment"):
        monkeypatch.setattr(f"app.{module}.run_cli", forbidden)
    # 기존 파이프라인 단위 테스트는 비교 이후의 결정론적 조립을 검사합니다. 의미검증 자체는
    # 전용 테스트에서 실제 함수를 좁은 응답 스텁으로 검증하고, 나머지 테스트에서는 통과시킵니다.
    monkeypatch.setattr("app.pipeline.validate_entailment",
                        lambda matches, documents, cache_keys=None, claims=None: [])
    # 결합 단위 의미검증도 같은 규율입니다. 유보가 생기는 사건에서만 도는 단계라, 켜 두면
    # 무관한 테스트가 조용히 CLI를 부르게 됩니다. 전용 테스트에서 실제 함수를 스텁 응답으로
    # 검증합니다.
    monkeypatch.setattr(
        "app.pipeline.validate_combination",
        lambda claim, labels, matrix, adopted, documents, cache_keys=None: [])
