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
# 분해 확정을 기다리는 동안 업로드한 PDF가 머무는 자리. tempfile.mkdtemp로 만들면 서버가
# 죽어도 폴더가 디스크에 그대로 남는데, 경로는 메모리에만 있어서 앱이 다시 찾지 못합니다 —
# 아무도 지우지 않는 고아 폴더가 재시작마다 쌓입니다. 앱이 아는 자리에 두어야 기동할 때
# 걷어낼 수 있습니다(main._sweep_orphan_staging).
STAGING_DIR = DATA_DIR / "staging"
for directory in (DATA_DIR, JOBS_DIR, STAGING_DIR, HISTORY_DIR, LOG_DIR):
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
# **전체** 동시 CLI 프로세스 상한. 단계별 노브를 곱한 값이 그대로 동시 실행 수가 되지 않도록
# 마지막에 한 번 더 조입니다(구성대비 셀 4 × 표본 3 = 12).
#
# CLI 하나가 프로세스 하나이고 호출마다 8천 토큰짜리 에이전트 프리앰블이 얹히므로, 무한정
# 올리면 메모리와 provider 동시 요청 한도에 걸립니다. 한도에 걸린 호출은 실패해 재시도가
# 붙으므로 **더 느려집니다** — 이 상한은 성능 제한이 아니라 성능 보호입니다.
MAX_CONCURRENT_CLI = max(1, int(os.getenv("FORGE_MAX_CONCURRENT_CLI", "8")))
# 의미검증 배치를 동시에 몇 개까지 돌릴지. 배치는 서로를 참조하지 않고 소요 시간의 거의
# 전부가 CLI 응답 대기라, 구성대비 셀과 같은 이유로 병렬화됩니다. 실측(1청구항 × 문헌 3건)에서
# 이 단계가 CLI를 18회 부르는 동안 구성대비는 3회였는데, 직렬로 돌아 전체 시간의 절반 이상을
# 혼자 썼습니다 — 호출 수가 가장 많은 단계가 유일하게 직렬이었습니다.
#
# 기본값이 전역 상한과 **같습니다.** 이 단계는 구성대비가 끝난 뒤 혼자 돌기 때문에 예산을
# 나눠 쓸 상대가 없습니다. 더 작게 잡으면 가장 긴 단계에서 남은 자리를 놀리게 됩니다.
ENTAILMENT_MAX_WORKERS = max(1, int(os.getenv("FORGE_ENTAILMENT_WORKERS",
                                              str(MAX_CONCURRENT_CLI))))
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
# 분해는 매 실행 LLM이 새로 만들기 때문에 같은 청구항이 실행마다 다르게 쪼개집니다. 한 구성의
# core가 1개↔2개로, 다른 구성의 한정이 4개↔5개로 갈리고 그 차이가 판정과 인용발명 선정까지
# 흔듭니다. 프롬프트 한 곳만 바꾼 효과를 재려면 분해를 붙들어 두어야 합니다.
#
# 지정하면 **분해 세대 검사를 건너뜁니다.** 실험은 대개 프롬프트를 고친 뒤에 하므로, 세대로
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

def model_id(value: str) -> str:
    """모델 이름에서 표시용 꼬리를 떼어 냅니다.

    `agy models`는 "id<TAB>표시 이름"을 출력하는데 목록을 줄째로 실어 나르던 동안 저장된
    설정 파일에는 탭이 붙은 값이 그대로 남아 있습니다(agy._agy_models). 그 값을 --model에
    실으면 CLI가 통째로 거부해 모든 호출이 실패하므로, 읽고 쓸 때 첫 탭 앞까지만 씁니다.
    목록 파싱을 고쳐도 **이미 저장된 설정은 낫지 않기 때문에** 읽는 쪽에도 둡니다.
    """
    return str(value or "").split("\t", 1)[0].strip()

def load_runtime_settings() -> dict:
    defaults = {"provider": LLM_PROVIDER, "model": AGY_MODEL, "prompt": DEFAULT_ANALYSIS_PROMPT}
    if not SETTINGS_FILE.exists(): return defaults
    try:
        value = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        settings = {**defaults, **{k: value[k] for k in defaults if k in value and value[k]}}
        return {**settings, "model": model_id(settings["model"]) or defaults["model"]}
    except (OSError, json.JSONDecodeError): return defaults

def save_runtime_settings(value: dict) -> dict:
    settings = {"provider": value["provider"], "model": model_id(value["model"]),
                "prompt": value.get("prompt") or DEFAULT_ANALYSIS_PROMPT}
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    return settings
