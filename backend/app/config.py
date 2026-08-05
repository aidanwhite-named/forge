from pathlib import Path
import os
import json
import shutil

from .prompts import DEFAULT_ANALYSIS_PROMPT

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = PROJECT_DIR / "history"
LOG_DIR = PROJECT_DIR / "logs"
for directory in (HISTORY_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_storage() -> None:
    """기존 backend/data 저장물을 Forge 루트 저장소로 한 번 복사합니다.

    원본을 지우지 않아 이전 버전으로 되돌려도 기록이 사라지지 않습니다. 같은 이름이 이미
    새 저장소에 있으면 새 저장물을 우선하고 건너뜁니다.
    """
    for legacy, target in ((DATA_DIR / "history", HISTORY_DIR), (DATA_DIR / "logs", LOG_DIR)):
        if not legacy.is_dir() or legacy.resolve() == target.resolve():
            continue
        for source in legacy.iterdir():
            destination = target / source.name
            if destination.exists():
                continue
            try:
                if source.is_dir():
                    shutil.copytree(source, destination)
                elif source.is_file():
                    shutil.copy2(source, destination)
            except OSError:
                # 저장소 마이그레이션 실패가 앱 기동 자체를 막아서는 안 됩니다.
                continue


_migrate_legacy_storage()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "agy").lower()
AGY_MODEL = os.getenv("AGY_MODEL", "gemini-3.6-flash-medium")
AGY_TIMEOUT_SECONDS = int(os.getenv("AGY_TIMEOUT_SECONDS", "300"))  # agy --print-timeout 기본값 5분에 맞춘다
AGY_MAX_RETRIES = int(os.getenv("AGY_MAX_RETRIES", "2"))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "25"))
MAX_TOTAL_UPLOAD_SIZE_MB = int(os.getenv("MAX_TOTAL_UPLOAD_SIZE_MB", "100"))
TEMP_FILE_TTL_MINUTES = int(os.getenv("TEMP_FILE_TTL_MINUTES", "60"))
SETTINGS_FILE = DATA_DIR / "settings.json"

def load_runtime_settings() -> dict:
    defaults = {"provider": LLM_PROVIDER, "model": AGY_MODEL, "prompt": DEFAULT_ANALYSIS_PROMPT}
    if not SETTINGS_FILE.exists(): return defaults
    try:
        value = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return {**defaults, **{k: value[k] for k in defaults if k in value and value[k]}}
    except (OSError, json.JSONDecodeError): return defaults

def save_runtime_settings(value: dict) -> dict:
    settings = {"provider": value["provider"], "model": value["model"],
                "prompt": value.get("prompt") or DEFAULT_ANALYSIS_PROMPT}
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return settings
