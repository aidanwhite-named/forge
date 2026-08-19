import json

from app import config


def test_a_stored_model_keeps_only_the_identifier(tmp_path, monkeypatch):
    """목록 파싱을 고쳐도 이미 저장된 설정은 낫지 않는다. 읽는 쪽에서 꼬리를 떼어야 한다.

    `agy models` 출력을 줄째로 저장하던 동안 설정 파일에는 "id<TAB>표시 이름"이 그대로
    들어갔고, 그 값은 --model에 실리는 순간 CLI가 통째로 거부한다.
    """
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps({
        "provider": "agy", "model": "gemini-3.7-flash-medium\tGemini 3.7 Flash (Medium)",
        "prompt": "지침"}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_FILE", settings_file)

    assert config.load_runtime_settings()["model"] == "gemini-3.7-flash-medium"


def test_saving_strips_the_display_name(tmp_path, monkeypatch):
    """다음 저장에서도 같은 값이 다시 들어오지 않도록 쓰는 쪽에서도 떼어 낸다."""
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", settings_file)

    saved = config.save_runtime_settings({
        "provider": "agy", "model": "claude-opus-4-6-thinking\tClaude Opus 4.6 (Thinking)",
        "prompt": "지침"})

    assert saved["model"] == "claude-opus-4-6-thinking"
    assert json.loads(settings_file.read_text(encoding="utf-8"))["model"] == "claude-opus-4-6-thinking"
