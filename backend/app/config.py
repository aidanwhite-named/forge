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
# 구성대비 셀 하나를 몇 번 물어 다수결로 합칠지. 회귀 하니스로 재 보니 같은 셀이 3회 실행에서
# '실질적 동일'·'차이'·'대응 없음'을 모두 냈습니다(안정성 1/3). 그 상태에서는 사용자가 보는
# 보고서를 그날의 운이 정하고, 코드를 고쳐도 효과를 1회 실행으로 확인할 수 없습니다.
# 비용이 그대로 배수로 늘어나므로 값을 열어 둡니다. 1이면 샘플링을 끕니다.
COMPARE_SAMPLES = max(1, int(os.getenv("FORGE_COMPARE_SAMPLES", "3")))
# 인용발명 1건을 대비할 때 한 호출에 몇 개 청구항까지 함께 실을지.
#
# 1이면 종전과 같은 (청구항 × 문헌) 축이라 문헌 본문이 청구항 수 × 표본 수만큼 다시 실립니다.
# 2 이상이면 문헌 축으로 묶어 문헌 본문을 그만큼 덜 보냅니다. 문헌이 한 건뿐인 호출이라
# 문맥 예산을 나눌 필요가 없고, 각 셀이 보는 원문 분량은 단건 경로와 같습니다.
#
# **기본값을 1로 둡니다.** 이 값을 올리면 응답 하나에 담기는 판정 수가 청구항 수만큼 늘어
# 출력 잘림·라벨 누락 위험이 커지는데, 그 임계는 모델·청구항 길이마다 달라 실행해 봐야
# 알 수 있습니다. 빠진 셀은 단건으로 자동 폴백하므로 판정이 사라지지는 않지만, 폴백이 잦으면
# 절감이 사라지고 시간만 늘어납니다. 회귀 코퍼스로 확인한 뒤 올리십시오.
COMPARE_CLAIM_BATCH = max(1, int(os.getenv("FORGE_COMPARE_CLAIM_BATCH", "1")))
# 청구항 분해를 이 파일로 **고정**합니다(claim_elements.json 형식). 실험용 통제 장치입니다.
#
# 분해는 매 실행 LLM이 새로 만들기 때문에 같은 청구항이 실행마다 다르게 쪼개집니다. 실측에서
# 한 구성의 core가 1개↔2개로, 다른 구성의 한정이 4개↔5개로 갈렸고 그 차이가 판정과 인용발명
# 선정까지 흔들었습니다. 프롬프트 한 곳만 바꾼 효과를 재려면 분해를 붙들어 두어야 합니다.
#
# 지정하면 **분해 버전 검사를 건너뜁니다.** 실험은 대개 버전을 올린 뒤에 하므로, 버전으로
# 막으면 정작 비교하려던 이전 분해를 쓸 수 없습니다. 다만 구성 원문 대조는 그대로 하므로
# 다른 청구항의 분해가 잘못 씌워지지는 않습니다.
DECOMPOSITION_FILE = os.getenv("FORGE_DECOMPOSITION_FILE", "").strip()
# 분해 캐시를 무시하고 항상 다시 분해합니다. 분해 프롬프트를 손볼 때 씁니다.
FORCE_REDECOMPOSE = os.getenv("FORGE_FORCE_REDECOMPOSE", "").lower() in {"1", "true", "yes"}
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
