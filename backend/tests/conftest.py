import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture(autouse=True)
def isolate_app_data(tmp_path, monkeypatch):
    """테스트가 backend/data의 실제 로그·히스토리를 남기지 않도록 격리한다."""
    for name in ("LOG_DIR", "HISTORY_DIR"):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(f"app.main.{name}", directory)
