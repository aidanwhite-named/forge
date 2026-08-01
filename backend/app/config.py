from pathlib import Path
import os
import json

from .prompts import DEFAULT_ANALYSIS_PROMPT

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = DATA_DIR / "history"
LOG_DIR = DATA_DIR / "logs"
for directory in (HISTORY_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "agy").lower()
LLM_COMMAND = os.getenv("LLM_COMMAND", LLM_PROVIDER)
AGY_COMMAND = LLM_COMMAND
AGY_MODEL = os.getenv("AGY_MODEL", "gemini-3.6-flash-medium")
AGY_TIMEOUT_SECONDS = int(os.getenv("AGY_TIMEOUT_SECONDS", "300"))  # agy --print-timeout 기본값 5분에 맞춘다
AGY_MAX_RETRIES = int(os.getenv("AGY_MAX_RETRIES", "2"))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "25"))
MAX_TOTAL_UPLOAD_SIZE_MB = int(os.getenv("MAX_TOTAL_UPLOAD_SIZE_MB", "100"))
TEMP_FILE_TTL_MINUTES = int(os.getenv("TEMP_FILE_TTL_MINUTES", "60"))
SETTINGS_FILE = DATA_DIR / "settings.json"

def load_runtime_settings() -> dict:
    defaults = {"provider": LLM_PROVIDER, "command": LLM_COMMAND, "model": AGY_MODEL, "prompt": DEFAULT_ANALYSIS_PROMPT}
    if not SETTINGS_FILE.exists(): return defaults
    try:
        value = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return {**defaults, **{k: value[k] for k in defaults if k in value and value[k]}}
    except (OSError, json.JSONDecodeError): return defaults

def save_runtime_settings(value: dict) -> dict:
    settings = {"provider": value["provider"], "command": value["command"], "model": value["model"],
                "prompt": value.get("prompt") or DEFAULT_ANALYSIS_PROMPT}
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return settings
