import json

import pytest
import requests

from src.ollama_client import OllamaClient, OllamaError, SYSTEM_PROMPT


class FakeResponse:
    def __init__(self, chunks=None, status_code=200, payload=None):
        self.chunks = chunks or []
        self.status_code = status_code
        self._payload = payload or {}
        self.ok = 200 <= status_code < 400
        self.closed = False

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        yield from self.chunks

    def close(self):
        self.closed = True


def test_model_list_reads_installed_ollama_models(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client.session,
        "get",
        lambda *args, **kwargs: FakeResponse(
            payload={"models": [{"name": "qwen2.5:1.5b"}, {"model": "llama3.2:1b"}]}
        ),
    )

    assert client.check_connection() == {
        "available": True,
        "models": ["llama3.2:1b", "qwen2.5:1.5b"],
        "error": None,
    }


def test_client_rejects_non_local_ollama_endpoints():
    with pytest.raises(ValueError, match="only connects to a local Ollama endpoint"):
        OllamaClient(base_url="https://example.com")


def test_connection_error_explains_how_to_install_ollama(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client.session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.ConnectionError("connection refused")
        ),
    )
    monkeypatch.setattr("src.ollama_client._ollama_is_installed", lambda: False)

    with pytest.raises(OllamaError, match="Ollama is not installed"):
        client.check_connection()


def test_connection_error_explains_how_to_start_ollama(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client.session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.ConnectionError("connection refused")
        ),
    )
    monkeypatch.setattr("src.ollama_client._ollama_is_installed", lambda: True)

    with pytest.raises(OllamaError, match="service is not running"):
        client.check_connection()


def test_streaming_generation_uses_local_chat_api_and_temperature(monkeypatch):
    response = FakeResponse(
        chunks=[
            json.dumps({"message": {"content": "Hello "}}).encode(),
            json.dumps({"message": {"content": "there!"}}).encode(),
            json.dumps({"done": True}).encode(),
        ]
    )
    client = OllamaClient()
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(client.session, "post", fake_post)
    result = "".join(
        client.stream_chat(
            [{"role": "user", "content": "Hi"}],
            model="qwen2.5:1.5b",
            temperature=0.4,
            max_tokens=256,
        )
    )

    assert result == "Hello there!"
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["stream"] is True
    assert captured["json"]["options"]["temperature"] == 0.4
    assert response.closed


def test_closing_stream_closes_local_request(monkeypatch):
    response = FakeResponse(
        chunks=[json.dumps({"message": {"content": "Partial answer"}}).encode()]
    )
    client = OllamaClient()
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: response)
    stream = client.stream_chat([{"role": "user", "content": "Explain this"}])

    assert next(stream) == "Partial answer"
    stream.close()

    assert response.closed


def test_prepare_messages_keeps_system_prompt_and_follow_up_context():
    messages = OllamaClient.prepare_messages(
        [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language."},
            {"role": "user", "content": "Who created it?"},
        ]
    )

    assert messages[0]["role"] == "system"
    assert "conversation history" in messages[0]["content"].lower()
    assert "complete, working solution" in SYSTEM_PROMPT
    assert [message["content"] for message in messages[1:]] == [
        "What is Python?",
        "A programming language.",
        "Who created it?",
    ]


def test_streaming_generation_reports_missing_model(monkeypatch):
    client = OllamaClient()
    response = FakeResponse(status_code=404, payload={"error": "model not found"})
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: response)

    with pytest.raises(OllamaError, match="ollama pull qwen2.5:1.5b"):
        list(client.stream_chat([{"role": "user", "content": "Hi"}]))


def test_streaming_generation_reports_memory_error(monkeypatch):
    client = OllamaClient()
    response = FakeResponse(
        chunks=[json.dumps({"error": "unable to allocate memory"}).encode()]
    )
    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: response)

    with pytest.raises(OllamaError, match="available memory"):
        list(client.stream_chat([{"role": "user", "content": "Hi"}]))


def test_streaming_generation_rejects_long_prompts():
    client = OllamaClient()

    with pytest.raises(ValueError, match="shorter than"):
        list(
            client.stream_chat(
                [{"role": "user", "content": "x" * 16001}]
            )
        )
