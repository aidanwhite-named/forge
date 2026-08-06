import json
import pytest
from app import agy

SETTINGS = {"provider": "agy", "model": "gemini-3.6-flash-medium"}


def test_prompt_is_the_last_argument(monkeypatch):
    """-p는 값을 받는 플래그라 프롬프트 뒤에 다른 플래그가 오면 프롬프트로 흡수된다."""
    monkeypatch.setattr(agy, "_resolve_command", lambda command: ["agy.exe"])
    command = agy._build_command("PROMPT", SETTINGS)
    assert command[-2:] == ["-p", "PROMPT"]
    assert command[command.index("--model") + 1] == "gemini-3.6-flash-medium"


def test_long_prompt_moves_to_a_workspace_file():
    """Windows 명령줄은 32767자 제한이라 긴 프롬프트는 파일로 넘어가야 한다."""
    prompt = "가" * (agy.ARGV_PROMPT_LIMIT + 1)
    workspace = agy._prompt_workspace(prompt, SETTINGS)
    try:
        with open(workspace["prompt_file"], encoding="utf-8") as file:
            assert file.read() == prompt
    finally:
        import shutil
        shutil.rmtree(workspace["dir"], ignore_errors=True)
    assert agy._prompt_workspace("짧은 프롬프트", SETTINGS) is None


def test_agy_response_envelope_is_unwrapped():
    payload = json.dumps({"status": "SUCCESS", "response": json.dumps({"claims": [], "summary": "s"})})
    assert agy._parse_response(payload, "", SETTINGS) == {"claims": [], "summary": "s"}


def test_cli_error_status_is_surfaced():
    payload = json.dumps({"status": "ERROR", "response": "", "error": "invalid model selection"})
    with pytest.raises(RuntimeError, match="invalid model selection"):
        agy._parse_response(payload, "", SETTINGS)


def test_available_models_falls_back_to_an_empty_list_when_the_cli_is_missing(monkeypatch):
    def missing(command):
        raise FileNotFoundError(command)

    monkeypatch.setattr(agy, "_resolve_command", missing)
    assert agy.available_models(SETTINGS) == []
    assert "claude-opus-5" in agy.available_models({**SETTINGS, "provider": "claude"})


def test_available_models_reads_the_agy_listing(monkeypatch):
    monkeypatch.setattr(agy, "_resolve_command", lambda command: ["agy.exe"])
    monkeypatch.setattr(agy, "_model_cache", {"agy.exe": ["gemini-3.6-flash-medium"]})
    assert agy.available_models(SETTINGS) == ["gemini-3.6-flash-medium"]


def test_command_line_too_long_is_not_reported_as_missing_cli():
    exc = FileNotFoundError(2, "too long")
    exc.winerror = 206
    assert "길이 제한" in str(agy._launch_error(exc, SETTINGS))
    assert "찾을 수 없습니다" in str(agy._launch_error(FileNotFoundError(2, "nope"), SETTINGS))


class _Process:
    def poll(self):
        return None


def test_cancel_job_kills_only_its_registered_process(monkeypatch):
    process = _Process()
    killed = []
    agy.register_job("job-a")
    agy.register_job("job-b")
    agy._active_processes["job-a"] = {process}
    monkeypatch.setattr(agy, "_kill_process_tree", lambda target: killed.append(target))

    assert agy.cancel_job("job-a") is True
    assert agy.is_cancelled("job-a") is True
    assert agy.is_cancelled("job-b") is False
    assert killed == [process]

    agy.finish_job("job-a")
    agy.finish_job("job-b")


def test_cancel_job_kills_every_process_the_job_started(monkeypatch):
    """구성대비 셀은 동시에 여러 개 돌아간다. 하나만 죽이면 나머지가 계속 돈다.

    사용자는 취소 버튼을 눌러 멈춘 줄 알지만, 살아남은 CLI가 끝까지 돌면서 요금과
    시간을 계속 쓴다. 작업이 띄운 프로세스는 전부 함께 정리해야 한다.
    """
    processes = {_Process() for _ in range(3)}
    killed = []
    agy.register_job("job-parallel")
    agy._active_processes["job-parallel"] = set(processes)
    monkeypatch.setattr(agy, "_kill_process_tree", lambda target: killed.append(target))

    assert agy.cancel_job("job-parallel") is True
    assert set(killed) == processes

    agy.finish_job("job-parallel")


# --- 비신뢰 입력을 다루는 CLI의 실행 조건 --------------------------------------

def test_no_provider_bypasses_permissions_by_default(monkeypatch):
    """프롬프트에는 사용자가 올린 PDF 본문이 그대로 들어간다.

    권한을 우회한 코딩 에이전트에 비신뢰 텍스트를 넘기면, 문헌에 심긴 지시가
    그대로 실행 권한을 얻는다. 이 단계는 텍스트 in / JSON out만 필요하다.
    """
    monkeypatch.setattr(agy, "_resolve_command", lambda command: ["cli.exe"])
    monkeypatch.setattr(agy, "ALLOW_TOOL_BYPASS", False)
    for provider in ("agy", "claude", "gpt"):
        command = agy._build_command("PROMPT", {**SETTINGS, "provider": provider})
        assert "--dangerously-skip-permissions" not in command
        assert "--dangerously-bypass-approvals-and-sandbox" not in command


def test_claude_disables_all_tools_and_takes_the_prompt_on_stdin(monkeypatch):
    monkeypatch.setattr(agy, "_resolve_command", lambda command: ["claude.exe"])
    monkeypatch.setattr(agy, "ALLOW_TOOL_BYPASS", False)
    command = agy._build_command("PROMPT", {**SETTINGS, "provider": "claude"})
    assert command[command.index("--tools") + 1] == ""      # 내장 도구 전면 비활성화
    assert command[-1] == "-p" and "PROMPT" not in command  # 프롬프트는 stdin으로
    assert agy._accepts_stdin("claude") and not agy._accepts_stdin("agy")


def test_cli_runs_in_a_throwaway_directory_not_the_repository(monkeypatch):
    """CLI가 서버 작업 디렉터리를 보고 있으면 인젝션의 첫 사정권이 이 저장소가 된다."""
    captured: dict = {}

    class Process:
        returncode = 0

        def communicate(self, input=None, timeout=None):
            return json.dumps({"claims": []}), ""

        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(agy, "load_runtime_settings", lambda: SETTINGS)
    monkeypatch.setattr(agy, "_resolve_command", lambda command: ["agy.exe"])
    monkeypatch.setattr(agy.subprocess, "Popen", fake_popen)

    agy.run_cli("짧은 프롬프트", expect="claims")

    cwd = captured["cwd"]
    assert cwd and "forge-cli-" in cwd
    assert "Forge" not in cwd.replace("forge-cli-", "")
