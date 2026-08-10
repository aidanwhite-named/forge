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
    # 선행기술 결과 검증은 외부 웹을 호출합니다. 테스트에서 나가지 않도록 막습니다.
    monkeypatch.setattr("app.priorart.verify_hits", lambda hits: None)


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
