import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

from .config import (AGY_TIMEOUT_SECONDS, AGY_MAX_RETRIES, MAX_CONCURRENT_CLI,
                     load_runtime_settings)

# Windows CreateProcess는 명령줄 전체를 32767자로 제한하고, 초과하면
# WinError 206(FileNotFoundError)으로 실패합니다. 긴 프롬프트는 인자 대신
# 작업 디렉터리에 파일로 넘겨 CLI가 직접 읽게 합니다.
ARGV_PROMPT_LIMIT = 24000

# 프롬프트에는 사용자가 업로드한 PDF 본문이 그대로 들어갑니다. 비신뢰 텍스트를 코딩
# 에이전트에 넘기는 것이므로, 도구 사용을 provider별로 최소화하고 항상 빈 임시
# 디렉터리에서 실행합니다. 이 파이프라인은 텍스트 in / JSON out만 필요합니다.
#
# 그래도 도구가 필요한 환경을 위한 탈출구. 기본은 꺼져 있습니다.
ALLOW_TOOL_BYPASS = os.getenv("FORGE_CLI_ALLOW_TOOL_BYPASS", "").lower() in {"1", "true", "yes"}


class AnalysisCancelled(RuntimeError):
    """Raised inside a worker as soon as its cancellation is requested."""


_job_local = threading.local()
_job_lock = threading.RLock()
_cancel_events: dict[str, threading.Event] = {}
# 작업 하나가 여러 CLI를 **동시에** 띄웁니다(구성대비 셀 병렬 처리). 프로세스를 하나만
# 붙들고 있으면 나중에 뜬 것이 앞의 것을 덮어써서, 취소했을 때 살아남은 프로세스가
# 계속 돌고 사용자는 멈춘 줄 압니다. 작업당 전부 들고 있다가 함께 정리합니다.
_active_processes: dict[str, set[subprocess.Popen]] = {}
# 동시에 살아 있을 수 있는 CLI 프로세스 수. 단계마다 제 나름의 병렬도를 갖는데(구성대비 셀 ×
# 표본, 의미검증 배치) 그 값들이 곱해진 만큼 프로세스가 뜨면 provider 한도에 걸립니다. 한도에
# 걸린 호출은 실패해 재시도가 붙으므로 병렬화의 이득이 그대로 사라집니다. 마지막 관문을 여기
# 한 곳에 두면 호출부는 자기 단계의 병렬도만 정하면 됩니다.
_cli_slots = threading.BoundedSemaphore(MAX_CONCURRENT_CLI)


def register_job(job_id: str) -> None:
    """Create the cancellation token before upload/analysis work starts."""
    with _job_lock:
        _cancel_events[job_id] = threading.Event()


def bind_job(job_id: str) -> None:
    """Associate CLI calls from the current worker thread with a job."""
    _job_local.job_id = job_id


def current_job() -> str | None:
    """현재 스레드가 매인 작업. 병렬 워커가 부모 스레드의 작업을 이어받을 때 씁니다."""
    return getattr(_job_local, "job_id", None)


def finish_job(job_id: str) -> None:
    with _job_lock:
        _active_processes.pop(job_id, None)
        _cancel_events.pop(job_id, None)
    if current_job() == job_id:
        del _job_local.job_id


def is_cancelled(job_id: str | None = None) -> bool:
    job_id = job_id or getattr(_job_local, "job_id", None)
    if not job_id:
        return False
    with _job_lock:
        event = _cancel_events.get(job_id)
    return bool(event and event.is_set())


def raise_if_cancelled(job_id: str | None = None) -> None:
    if is_cancelled(job_id):
        raise AnalysisCancelled("보고서 생성을 취소했습니다.")


def cancel_job(job_id: str) -> bool:
    """Set the cancellation token and kill every CLI process tree the job owns."""
    with _job_lock:
        event = _cancel_events.get(job_id)
        processes = list(_active_processes.get(job_id) or ())
        if event is None:
            return False
        event.set()
    for process in processes:
        if process.poll() is None:
            _kill_process_tree(process)
    return True


def run_parallel(tasks: list, max_workers: int) -> list[tuple]:
    """작업을 병렬로 돌리고 ``(결과, 예외)`` 쌍을 **제출 순서 그대로** 돌려줍니다.

    이 함수가 따로 있는 이유는 병렬 실행 자체가 아니라 병렬 실행이 조용히 깨뜨리는 두 가지
    때문입니다. 호출부마다 다시 구현하면 한 곳에서만 빠뜨려도 증상이 드물게 나타나 찾기
    어렵습니다.

    **작업 바인딩.** 풀의 워커 스레드는 부모의 job을 물려받지 않습니다. 매어 두지 않으면
    취소 신호가 그 스레드에서 띄운 CLI 프로세스에 닿지 않아, 사용자가 멈춘 뒤에도 프로세스가
    끝까지 돕니다(_compare_cells가 같은 이유로 bind_job을 부릅니다).

    **순서.** 완료 순서로 모으면 같은 입력에서 실행마다 다른 산출물이 나옵니다. 구성대비
    표본은 순서가 곧 표본 번호이고(compare.consensus가 "앞선 두 표본이 일치했는가"를 셉니다),
    검증 노트는 순서가 곧 보고서의 줄 순서입니다.

    예외는 올리지 않고 쌍에 담아 돌려줍니다. 실패를 다루는 방식이 호출부마다 다르기
    때문입니다 — 표본 하나는 버리고 나머지로 다수결을 내지만, 배치 하나는 그 사실을 노트로
    남겨야 합니다. 취소(AnalysisCancelled)도 담아 보내므로 호출부가 먼저 가려내야 합니다.
    """
    if not tasks:
        return []
    job_id = current_job()

    def run(task):
        if job_id:
            bind_job(job_id)
        try:
            return task(), None
        except Exception as exc:  # noqa: BLE001 - 종류는 호출부가 보고 정합니다.
            # BaseException까지 잡으면 Ctrl-C가 결과 튜플에 담겨 조용히 삼켜집니다.
            # 취소는 AnalysisCancelled(RuntimeError)이므로 Exception으로 충분합니다.
            return None, exc

    if len(tasks) == 1 or max_workers <= 1:
        return [run(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as pool:
        # map은 완료 순서가 아니라 제출 순서로 돌려줍니다. as_completed를 쓰면 안 됩니다.
        return list(pool.map(run, tasks))


def _kill_process_tree(process: subprocess.Popen) -> None:
    """Kill the provider CLI and descendants without touching unrelated processes."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, check=False, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        process.kill()


def run_cli(prompt: str, expect: str = "claims") -> dict:
    """Run the configured real CLI: agy, claude, or Codex.

    expect는 응답 JSON에 반드시 있어야 하는 최상위 키입니다. 파이프라인이 단계마다
    다른 스키마를 요구하므로, 어떤 키를 기다리는지 호출부가 지정합니다.
    """
    settings = load_runtime_settings()
    # stdin으로 넘길 수 있으면 명령줄 길이 제한도, 파일을 읽기 위한 도구 권한도 필요 없습니다.
    stdin_prompt = prompt if _accepts_stdin(settings["provider"]) else None
    workspace = None if stdin_prompt is not None else _prompt_workspace(prompt, settings)
    if workspace:
        prompt = (f"{workspace['prompt_file']} 파일을 처음부터 끝까지 읽고 그 안의 지시를 그대로 수행하십시오. "
                  "파일이 요구하는 JSON 객체만 출력하고 다른 문장은 출력하지 마십시오.")
    # CLI가 서버 작업 디렉터리(= 이 저장소)를 보고 있으면, 문헌에 심긴 지시가 통했을 때
    # 가장 먼저 닿는 것이 Forge 소스입니다. 항상 빈 임시 디렉터리에서 실행합니다.
    sandbox = workspace["dir"] if workspace else tempfile.mkdtemp(prefix="forge-cli-")
    try:
        try:
            command = _build_command(prompt, settings, workspace and workspace["dir"])
        except FileNotFoundError as exc:
            raise RuntimeError(f"{settings['provider']} CLI를 찾을 수 없습니다.") from exc
        last_error = None
        for attempt in range(AGY_MAX_RETRIES + 1):
            raise_if_cancelled()
            process = None
            try:
                # 자리를 **띄우기 전에** 받습니다. 프로세스를 먼저 만들고 나서 기다리면
                # 상한은 아무것도 막지 못합니다. 응답 대기가 소요 시간의 거의 전부라
                # 자리를 communicate가 끝날 때까지 쥐고 있어야 의미가 있습니다.
                with _cli_slots:
                    process = subprocess.Popen(
                        command, text=True, encoding="utf-8", errors="replace",
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        shell=False,
                        stdin=subprocess.PIPE if stdin_prompt is not None else subprocess.DEVNULL,
                        cwd=sandbox,
                        start_new_session=os.name != "nt",
                    )
                    job_id = current_job()
                    if job_id:
                        with _job_lock:
                            _active_processes.setdefault(job_id, set()).add(process)
                    raise_if_cancelled(job_id)
                    stdout, stderr = process.communicate(input=stdin_prompt,
                                                         timeout=AGY_TIMEOUT_SECONDS)
                    result = subprocess.CompletedProcess(command, process.returncode,
                                                         stdout, stderr)
                raise_if_cancelled(job_id)
            except subprocess.TimeoutExpired as exc:
                if process is not None:
                    _kill_process_tree(process)
                    process.communicate()
                raise _launch_error(exc, settings) from exc
            except OSError as exc:
                raise _launch_error(exc, settings) from exc
            finally:
                job_id = current_job()
                if job_id and process is not None:
                    with _job_lock:
                        _active_processes.get(job_id, set()).discard(process)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
                last_error = f"exit code {result.returncode}: {detail[-1000:]}"
                continue
            try:
                return _parse_response(result.stdout, result.stderr, settings, expect)
            except RuntimeError as exc:
                # 응답이 잘리거나 스키마가 어긋나는 것은 재시도로 회복되는 경우가 많습니다.
                # CLI 내부 저장소를 뒤져 복구하지 않고 그냥 다시 호출합니다.
                last_error = str(exc)
                continue
        hint = ""
        if workspace and not ALLOW_TOOL_BYPASS:
            # 긴 프롬프트는 파일로 넘기므로 CLI가 그 파일을 읽어야 합니다. 도구 권한을
            # 막아 둔 상태에서 실패했다면 원인을 짚어 줍니다. 그냥 켜라고 하지는 않습니다.
            hint = (" 긴 프롬프트를 파일로 전달했는데 CLI가 읽지 못했을 수 있습니다."
                    " 신뢰할 수 있는 문헌이라면 FORGE_CLI_ALLOW_TOOL_BYPASS=1로 도구 권한을 열 수 있으나,"
                    " 그 경우 문헌에 심긴 지시가 실행될 수 있습니다. PDF 수를 줄이는 편이 안전합니다.")
        raise RuntimeError(
            f"{settings['provider']} CLI 실행에 실패했습니다: {last_error or '알 수 없는 오류'}{hint}")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def _accepts_stdin(provider: str) -> bool:
    """프롬프트를 stdin으로 받는 CLI.

    claude는 -p에 인자를 주지 않으면 stdin을 프롬프트로 읽습니다. 그래서 명령줄 길이 제한을
    아예 벗어나고, 긴 프롬프트를 파일로 넘길 필요도(=파일 읽기 도구를 열어 줄 필요도) 없습니다.
    agy --print는 인자를 요구하고 stdin을 읽지 않아 이 경로를 쓸 수 없습니다.
    """
    return provider == "claude"


def _prompt_workspace(prompt: str, settings: dict) -> dict | None:
    """Spill an oversized prompt to a file the CLI can read from its workspace."""
    if len(prompt) <= ARGV_PROMPT_LIMIT:
        return None
    if settings["provider"] == "gpt":
        raise RuntimeError(
            f"프롬프트가 명령줄 길이 제한({ARGV_PROMPT_LIMIT}자)을 초과했습니다. "
            "Codex CLI는 이 방식의 파일 전달을 지원하지 않으므로 PDF 개수를 줄이거나 agy/claude를 사용하십시오."
        )
    directory = tempfile.mkdtemp(prefix="forge-prompt-")
    prompt_file = os.path.join(directory, "prompt.md")
    with open(prompt_file, "w", encoding="utf-8") as file:
        file.write(prompt)
    return {"dir": directory, "prompt_file": prompt_file}


def _launch_error(exc: Exception, settings: dict) -> RuntimeError:
    provider = settings["provider"]
    if isinstance(exc, subprocess.TimeoutExpired):
        return RuntimeError(f"{provider} CLI 실행 시간이 초과되었습니다.")
    if getattr(exc, "winerror", None) == 206:
        return RuntimeError(f"{provider} CLI 명령줄이 Windows 길이 제한을 초과했습니다.")
    if isinstance(exc, FileNotFoundError):
        return RuntimeError(f"{provider} CLI를 찾을 수 없습니다.")
    return RuntimeError(f"{provider} CLI 실행에 실패했습니다: {exc}")


def _parse_response(stdout: str, stderr: str, settings: dict, expect: str = "claims") -> dict:
    provider = settings["provider"]
    try:
        payload = _parse_json_output(stdout)
    except json.JSONDecodeError as exc:
        detail = stdout.strip() or stderr.strip() or "출력이 비어 있습니다."
        raise RuntimeError(f"{provider} CLI가 JSON을 반환하지 않았습니다: {detail[-500:]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{provider} CLI 응답이 JSON 객체가 아닙니다.")
    if payload.get("status") == "ERROR":
        raise RuntimeError(f"{provider} CLI가 오류를 반환했습니다: {str(payload.get('error', ''))[-500:]}")
    if expect in payload:
        return payload
    # agy는 응답 본문을 response, claude는 result 필드에 담아 반환합니다.
    for field in ("response", "result", "output", "text"):
        body = payload.get(field)
        if not isinstance(body, str):
            continue
        try:
            inner = _parse_json_output(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{provider} CLI의 {field} 필드가 JSON이 아닙니다: {body.strip()[-500:]}") from exc
        if isinstance(inner, dict) and expect in inner:
            return inner
    raise RuntimeError(f"{provider} CLI 응답에 {expect} 필드가 없습니다.")


def _build_command(prompt: str, settings: dict | None = None, workspace: str | None = None) -> list[str]:
    """provider별 실행 명령. 권한 우회 플래그는 기본적으로 붙이지 않습니다.

    이 단계에 필요한 것은 텍스트를 읽고 JSON을 내는 것뿐이라 도구가 필요 없습니다.
    반면 프롬프트에는 비신뢰 PDF 본문이 들어가므로, 도구를 열어 두면 문헌에 심긴 지시가
    그대로 실행 권한을 얻습니다.
    """
    settings = settings or load_runtime_settings()
    provider, model = settings["provider"], settings["model"]
    executable = _resolve_command(_provider_command(provider))
    workspace_args = ["--add-dir", workspace] if workspace else []
    # 프롬프트는 반드시 마지막에 옵니다. -p/--print가 값을 받는 플래그라
    # 앞에 두면 뒤따르는 플래그가 프롬프트 문자열로 흡수됩니다.
    if provider == "agy":
        # agy에는 도구 전면 비활성화 플래그가 없습니다. --sandbox로 터미널을 제한하고,
        # 긴 프롬프트를 파일로 넘길 때만(=파일 읽기가 필요할 때만) 자동 승인을 붙입니다.
        isolation = ["--sandbox"]
        if workspace and ALLOW_TOOL_BYPASS:
            isolation = ["--dangerously-skip-permissions"]
        return [*executable, "--model", model, "--output-format", "json",
                *isolation, *workspace_args, "-p", prompt]
    if provider == "claude":
        # --tools ""는 내장 도구를 전부 끕니다. 승인 요청 자체가 발생하지 않으므로
        # 권한 우회가 필요 없고, 프롬프트는 stdin으로 들어가 인자로 두지 않습니다.
        tools = ["--tools", ""] if not ALLOW_TOOL_BYPASS else ["--dangerously-skip-permissions"]
        return [*executable, "--model", model, "--output-format", "json", *tools, "-p"]
    if provider == "gpt":
        # codex exec는 기본이 승인 없는 제한 샌드박스입니다. 우회 플래그를 붙이지 않습니다.
        return [*executable, "exec", "--model", model, prompt]
    raise RuntimeError(f"지원하지 않는 LLM_PROVIDER입니다: {provider}. agy, claude, gpt 중 하나를 사용하세요.")


_model_cache: dict[str, list[str]] = {}
# agy 외 CLI는 모델 목록 조회 명령이 없어 알려진 모델만 제안하고 직접 입력을 허용합니다.
_KNOWN_MODELS = {"claude": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"], "gpt": []}


def available_models(settings: dict | None = None, refresh: bool = False) -> list[str]:
    """설정된 CLI가 지원하는 모델 목록. 조회할 수 없으면 빈 목록을 돌려주고 UI는 직접 입력으로 대체합니다."""
    settings = settings or load_runtime_settings()
    if settings["provider"] != "agy":
        return list(_KNOWN_MODELS.get(settings["provider"], []))
    try:
        executable = _resolve_command(_provider_command(settings["provider"]))[0]
    except FileNotFoundError:
        return []
    if refresh:
        _model_cache.pop(executable, None)
    return _agy_models(executable)


def _agy_models(executable: str) -> list[str]:
    """agy가 지원하는 모델 목록. 조회에 실패하면 빈 목록으로 취급합니다.

    출력은 "id<TAB>표시 이름" 한 줄씩이고 앞에 진행 메시지("Fetching available models...")가
    붙습니다. 줄을 통째로 모델 이름으로 쓰면 화면 목록에서 하나를 고르는 순간 --model에
    "id<TAB>표시 이름"이 그대로 실려 CLI가 통째로 거부합니다(invalid model selection).
    그러면 분해·구성대비·의미검증이 전부 실패해 중요도는 기본값 3으로, 셀은 미판정으로
    떨어지는데, 보고서 자체는 그대로 나오므로 사용자는 원인을 알기 어렵습니다.
    """
    if executable not in _model_cache:
        try:
            # 서버가 콘솔 없이 실행되면 CLI가 stdin을 기다리다 멈추므로 반드시 닫아 준다.
            listing = subprocess.run([executable, "models"], text=True, encoding="utf-8", errors="replace",
                                     capture_output=True, timeout=30, check=False, shell=False,
                                     stdin=subprocess.DEVNULL)
            lines = [line.strip() for line in listing.stdout.splitlines()
                     if line.strip()] if listing.returncode == 0 else []
            # 탭이 있는 줄만 모델 행입니다 — 진행 메시지는 이 조건에서 자연히 빠집니다.
            # 탭이 하나도 없으면 목록 형식이 다른 것이므로 종전대로 줄 전체를 씁니다.
            models = [line.split("\t", 1)[0].strip() for line in lines if "\t" in line]
            _model_cache[executable] = models or lines
        except (subprocess.TimeoutExpired, OSError):
            _model_cache[executable] = []
    return _model_cache[executable]


def _provider_command(provider: str) -> str:
    return "codex" if provider == "gpt" else provider


def _resolve_command(command: str) -> list[str]:
    resolved = shutil.which(command)
    if os.name == "nt" and not resolved and not os.path.splitext(command)[1]:
        resolved = shutil.which(f"{command}.cmd")
    if os.name == "nt" and not resolved and not os.path.splitext(command)[1]:
        common_dirs = [
            os.path.expandvars(r"%LOCALAPPDATA%\agy\bin"),
            os.path.expandvars(r"%APPDATA%\npm"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "agy", "bin"),
        ]
        for directory in common_dirs:
            for suffix in (".exe", ".cmd", ".bat", ""):
                candidate = os.path.join(directory, command + suffix)
                if os.path.isfile(candidate):
                    resolved = candidate
                    break
            if resolved:
                break
    if not resolved and os.path.exists(command):
        resolved = command
    if not resolved:
        raise FileNotFoundError(command)
    return [resolved]


def _parse_json_output(output: str) -> dict:
    output = output.strip()
    if output.startswith("```"):
        output = output.split("\n", 1)[1] if "\n" in output else output
        if output.endswith("```"):
            output = output[:-3].rstrip()
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        start, end = output.find("{"), output.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(output[start:end + 1])
