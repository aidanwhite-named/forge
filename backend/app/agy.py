import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading

from .config import AGY_TIMEOUT_SECONDS, AGY_MAX_RETRIES, load_runtime_settings

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
_active_processes: dict[str, subprocess.Popen] = {}


def register_job(job_id: str) -> None:
    """Create the cancellation token before upload/analysis work starts."""
    with _job_lock:
        _cancel_events[job_id] = threading.Event()


def bind_job(job_id: str) -> None:
    """Associate CLI calls from the current worker thread with a job."""
    _job_local.job_id = job_id


def finish_job(job_id: str) -> None:
    with _job_lock:
        _active_processes.pop(job_id, None)
        _cancel_events.pop(job_id, None)
    if getattr(_job_local, "job_id", None) == job_id:
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
    """Set the cancellation token and kill the exact CLI process tree, if any."""
    with _job_lock:
        event = _cancel_events.get(job_id)
        process = _active_processes.get(job_id)
        if event is None:
            return False
        event.set()
    if process is not None and process.poll() is None:
        _kill_process_tree(process)
    return True


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
                process = subprocess.Popen(
                    command, text=True, encoding="utf-8", errors="replace",
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    shell=False,
                    stdin=subprocess.PIPE if stdin_prompt is not None else subprocess.DEVNULL,
                    cwd=sandbox,
                    start_new_session=os.name != "nt",
                )
                job_id = getattr(_job_local, "job_id", None)
                if job_id:
                    with _job_lock:
                        _active_processes[job_id] = process
                raise_if_cancelled(job_id)
                stdout, stderr = process.communicate(input=stdin_prompt, timeout=AGY_TIMEOUT_SECONDS)
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                raise_if_cancelled(job_id)
            except subprocess.TimeoutExpired as exc:
                if process is not None:
                    _kill_process_tree(process)
                    process.communicate()
                raise _launch_error(exc, settings) from exc
            except OSError as exc:
                raise _launch_error(exc, settings) from exc
            finally:
                job_id = getattr(_job_local, "job_id", None)
                if job_id and process is not None:
                    with _job_lock:
                        if _active_processes.get(job_id) is process:
                            _active_processes.pop(job_id, None)
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
    """agy가 지원하는 모델 목록. 조회에 실패하면 빈 목록으로 취급합니다."""
    if executable not in _model_cache:
        try:
            # 서버가 콘솔 없이 실행되면 CLI가 stdin을 기다리다 멈추므로 반드시 닫아 준다.
            listing = subprocess.run([executable, "models"], text=True, encoding="utf-8", errors="replace",
                                     capture_output=True, timeout=30, check=False, shell=False,
                                     stdin=subprocess.DEVNULL)
            _model_cache[executable] = [line.strip() for line in listing.stdout.splitlines() if line.strip()] if listing.returncode == 0 else []
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
