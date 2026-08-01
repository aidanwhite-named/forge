import json
import pytest
from app import agy

SETTINGS = {"provider": "agy", "command": "agy", "model": "gemini-3.6-flash-medium"}


def test_prompt_is_the_last_argument(monkeypatch):
    """-p는 값을 받는 플래그라 프롬프트 뒤에 다른 플래그가 오면 프롬프트로 흡수된다."""
    monkeypatch.setattr(agy, "_resolve_command", lambda command: ["agy.exe"])
    command = agy._build_command("PROMPT", "low", SETTINGS)
    assert command[-2:] == ["-p", "PROMPT"]
    assert command[command.index("--model") + 1] == "gemini-3.6-flash-low"
    assert "--effort" not in command  # 모델명이 이미 effort를 포함한다


def test_effort_flag_used_when_model_has_no_suffix(monkeypatch):
    monkeypatch.setattr(agy, "_resolve_command", lambda command: ["agy.exe"])
    command = agy._build_command("PROMPT", "high", {**SETTINGS, "model": "claude-sonnet-4-6"})
    assert command[command.index("--effort") + 1] == "high"


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
