import json
import os
import shutil
import subprocess
import tempfile

from .config import AGY_TIMEOUT_SECONDS, AGY_MAX_RETRIES, load_runtime_settings

# Windows CreateProcess는 명령줄 전체를 32767자로 제한하고, 초과하면
# WinError 206(FileNotFoundError)으로 실패합니다. 긴 프롬프트는 인자 대신
# 작업 디렉터리에 파일로 넘겨 CLI가 직접 읽게 합니다.
ARGV_PROMPT_LIMIT = 24000


def run_agy(prompt: str, effort: str) -> dict:
    """Run the configured real CLI: agy -p, claude -p, or gpt -p."""
    settings = load_runtime_settings()
    workspace = _prompt_workspace(prompt, settings)
    if workspace:
        prompt = (f"{workspace['prompt_file']} 파일을 처음부터 끝까지 읽고 그 안의 지시를 그대로 수행하십시오. "
                  "파일이 요구하는 JSON 객체만 출력하고 다른 문장은 출력하지 마십시오.")
    try:
        try:
            command = _build_command(prompt, effort, settings, workspace and workspace["dir"])
        except FileNotFoundError as exc:
            raise RuntimeError(f"{settings['provider']} CLI를 찾을 수 없습니다: {settings['command']}") from exc
        last_error = None
        for attempt in range(AGY_MAX_RETRIES + 1):
            try:
                result = subprocess.run(
                    command, text=True, encoding="utf-8", errors="replace",
                    capture_output=True, timeout=AGY_TIMEOUT_SECONDS,
                    check=False, shell=False, stdin=subprocess.DEVNULL,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise _launch_error(exc, settings) from exc
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
                last_error = f"exit code {result.returncode}: {detail[-1000:]}"
                continue
            return _parse_response(result.stdout, result.stderr, settings)
        raise RuntimeError(f"{settings['provider']} CLI 실행에 실패했습니다: {last_error or '알 수 없는 오류'}")
    finally:
        if workspace:
            shutil.rmtree(workspace["dir"], ignore_errors=True)


def _prompt_workspace(prompt: str, settings: dict) -> dict | None:
    """Spill an oversized prompt to a file the CLI can read from its workspace."""
    if len(prompt) <= ARGV_PROMPT_LIMIT:
        return None
    if settings["provider"] == "gpt":
        raise RuntimeError(
            f"프롬프트가 명령줄 길이 제한({ARGV_PROMPT_LIMIT}자)을 초과했습니다. "
            "gpt CLI는 파일 전달을 지원하지 않으므로 PDF 개수를 줄이거나 agy/claude를 사용하십시오."
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
        return RuntimeError(f"{provider} CLI를 찾을 수 없습니다: {settings['command']}")
    return RuntimeError(f"{provider} CLI 실행에 실패했습니다: {exc}")


def _parse_response(stdout: str, stderr: str, settings: dict) -> dict:
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
    if "claims" in payload:
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
        if isinstance(inner, dict) and "claims" in inner:
            return inner
    raise RuntimeError(f"{provider} CLI 응답에 claims 필드가 없습니다.")


def _build_command(prompt: str, effort: str, settings: dict | None = None, workspace: str | None = None) -> list[str]:
    settings = settings or load_runtime_settings()
    provider, model = settings["provider"], settings["model"]
    executable = _resolve_command(settings["command"])
    workspace_args = ["--add-dir", workspace] if workspace else []
    # 프롬프트는 반드시 마지막에 옵니다. -p/--print가 값을 받는 플래그라
    # 앞에 두면 뒤따르는 플래그가 프롬프트 문자열로 흡수됩니다.
    if provider == "agy":
        model, effort_args = _apply_effort(executable[0], model, effort)
        return [*executable, "--model", model, *effort_args, "--output-format", "json",
                "--dangerously-skip-permissions", *workspace_args, "-p", prompt]
    if provider == "claude":
        return [*executable, "--model", model, "--output-format", "json",
                "--dangerously-skip-permissions", *workspace_args, "-p", prompt]
    if provider == "gpt":
        return [*executable, "--model", model, "-p", prompt]
    raise RuntimeError(f"지원하지 않는 LLM_PROVIDER입니다: {provider}. agy, claude, gpt 중 하나를 사용하세요.")


_EFFORTS = ("low", "medium", "high")
_model_cache: dict[str, list[str]] = {}
# agy 외 CLI는 모델 목록 조회 명령이 없어 알려진 모델만 제안하고 직접 입력을 허용합니다.
_KNOWN_MODELS = {"claude": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"], "gpt": []}


def available_models(settings: dict | None = None, refresh: bool = False) -> list[str]:
    """설정된 CLI가 지원하는 모델 목록. 조회할 수 없으면 빈 목록을 돌려주고 UI는 직접 입력으로 대체합니다."""
    settings = settings or load_runtime_settings()
    if settings["provider"] != "agy":
        return list(_KNOWN_MODELS.get(settings["provider"], []))
    try:
        executable = _resolve_command(settings["command"])[0]
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


def _apply_effort(executable: str, model: str, effort: str) -> tuple[str, list[str]]:
    """agy 모델명은 -low/-medium/-high 접미사로 effort를 포함하고, 이때 --effort를 함께 주면 거부됩니다."""
    base, _, suffix = model.rpartition("-")
    if suffix not in _EFFORTS:
        return model, ["--effort", effort]
    candidate = f"{base}-{effort}"
    models = _agy_models(executable)
    if candidate != model and (not models or candidate in models):
        return candidate, []
    return model, []


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
