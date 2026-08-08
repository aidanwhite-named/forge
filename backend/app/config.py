from pathlib import Path
import os
import json
import shutil

from .prompts import DEFAULT_ANALYSIS_PROMPT

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
# 기본값은 Forge 루트입니다. 컨테이너에서는 PROJECT_DIR이 파일시스템 루트("/")가 되어
# 보고서와 로그가 마운트된 볼륨 **바깥**에 쓰이고 재시작마다 사라집니다. 배포 환경이
# 저장 위치를 볼륨 안으로 옮길 수 있도록 환경변수로 열어 둡니다.
HISTORY_DIR = Path(os.getenv("FORGE_HISTORY_DIR") or PROJECT_DIR / "history")
LOG_DIR = Path(os.getenv("FORGE_LOG_DIR") or PROJECT_DIR / "logs")
# 진행 중인 작업의 상태. 서버가 재시작되어도 프런트가 "작업을 찾을 수 없습니다" 대신
# 중단 사실을 볼 수 있어야 합니다.
JOBS_DIR = DATA_DIR / "jobs"
for directory in (DATA_DIR, JOBS_DIR, HISTORY_DIR, LOG_DIR):
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
# 셀 단위 구성대비를 동시에 몇 개까지 돌릴지. 셀은 서로 독립이고 대기 시간이 거의 전부라
# 병렬로 돌리면 그만큼 줄어듭니다. 다만 하나가 CLI 프로세스 하나라서 무한정 올리면
# 메모리와 provider 쪽 동시 요청 한도에 걸립니다.
COMPARE_MAX_WORKERS = max(1, int(os.getenv("FORGE_COMPARE_WORKERS", "4")))
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "25"))
MAX_TOTAL_UPLOAD_SIZE_MB = int(os.getenv("MAX_TOTAL_UPLOAD_SIZE_MB", "100"))
# 끝난 작업의 메모리 레코드를 얼마나 두고 걷어낼지. 결과는 전부 히스토리에 있으므로
# 걷어내도 조회는 그대로 됩니다. prepare만 하고 버려진 작업도 이 값으로 정리됩니다.
JOB_RECORD_TTL_MINUTES = int(os.getenv("FORGE_JOB_RECORD_TTL_MINUTES", "60"))
# 작업 로그 한 건의 크기 상한. 넘으면 앞부분을 잘라 뒤쪽을 남깁니다. 진행 로그는 셀 수에
# 비례해 늘어나므로 상한이 없으면 한 작업이 디스크를 계속 먹습니다.
LOG_MAX_BYTES = int(os.getenv("FORGE_LOG_MAX_BYTES", str(2 * 1024 * 1024)))
# 선행기술 검색 결과의 URL을 실제로 열어 문헌번호를 대조할 때의 건당 제한 시간(초).
# 0으로 두면 대조하지 않습니다.
PRIOR_ART_VERIFY_TIMEOUT = float(os.getenv("FORGE_PRIOR_ART_VERIFY_TIMEOUT", "15"))
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
