import json
import ipaddress
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib.parse import urlsplit

import requests
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from src.config import (
    DEFAULT_OLLAMA_MODEL,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_MESSAGES,
    MAX_USER_MESSAGE_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_READ_TIMEOUT,
)

SYSTEM_PROMPT = (
    "You are NexaBot, a helpful general-purpose AI assistant running locally. "
    "Answer in clear, understandable English. Explain difficult topics step by step "
    "and use examples when helpful. Answer the exact question first and avoid "
    "unnecessary preambles or unsupported extra facts. For programming requests, "
    "give a concise, complete, working solution before explaining it. Put code in "
    "correctly formatted Markdown code blocks and do not leave code unfinished. "
    "Be concise unless the user asks for detail. Follow the requested format. "
    "Admit uncertainty, do not invent facts, and ask a clarifying question when "
    "needed. Use the conversation history to resolve follow-up references. "
    "You have no live internet access."
)

ANSWER_STYLES = {
    "Short": "Keep the answer very brief and direct.",
    "Balanced": "Give a direct answer with enough detail to be useful and complete.",
    "Detailed": "Give a thorough step-by-step explanation with examples when useful.",
}


class OllamaError(RuntimeError):
    """A beginner-friendly error raised when local Ollama generation fails."""


def _model_names(payload: Dict[str, Any]) -> List[str]:
    models = payload.get("models", [])
    return sorted(
        {
            str(model.get("name") or model.get("model"))
            for model in models
            if model.get("name") or model.get("model")
        }
    )


def _is_local_endpoint(base_url: str) -> bool:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _ollama_is_installed() -> bool:
    if shutil.which("ollama"):
        return True
    local_app_data = os.environ.get("LOCALAPPDATA")
    candidates = [
        Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if local_app_data
        else None,
        Path(os.environ.get("ProgramFiles", "C:\\Program Files"))
        / "Ollama"
        / "ollama.exe",
    ]
    return any(path is not None and path.is_file() for path in candidates)


def _error_message(response: Response, model: str) -> OllamaError:
    try:
        detail = response.json().get("error", "")
    except (ValueError, AttributeError):
        detail = ""
    detail = str(detail).strip()
    lowered = detail.lower()
    if response.status_code == 404 or "not found" in lowered:
        return OllamaError(
            f"The model '{model}' is not downloaded. Open PowerShell and run: "
            f"ollama pull {model}"
        )
    if "memory" in lowered or "ram" in lowered or "allocate" in lowered:
        return OllamaError(
            "Ollama could not load this model with the available memory. Close other "
            "applications or choose a smaller model."
        )
    if detail:
        return OllamaError(f"Ollama could not generate a response: {detail}")
    return OllamaError(
        f"Ollama returned HTTP {response.status_code}. Check that the selected model "
        "is installed and that Ollama is running."
    )


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        connect_timeout: int = OLLAMA_CONNECT_TIMEOUT,
        read_timeout: int = OLLAMA_READ_TIMEOUT,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        if not _is_local_endpoint(self.base_url):
            raise ValueError(
                "For privacy, NexaBot only connects to a local Ollama endpoint "
                "(localhost or a loopback IP address)."
            )
        self.timeout = (connect_timeout, read_timeout)
        self.session = session or requests.Session()

    def installed_models(self) -> List[str]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=(OLLAMA_CONNECT_TIMEOUT, 10),
            )
            response.raise_for_status()
            return _model_names(response.json())
        except ConnectionError as error:
            if not _ollama_is_installed():
                raise OllamaError(
                    "Ollama is not installed. Install it from https://ollama.com/download, "
                    "then restart this app."
                ) from error
            raise OllamaError(
                "Ollama is installed but its local service is not running. Start Ollama "
                "from the Start menu and try again."
            ) from error
        except Timeout as error:
            raise OllamaError(
                "The local Ollama service did not respond in time. Check that it is running."
            ) from error
        except (HTTPError, RequestException, ValueError) as error:
            raise OllamaError(f"Could not read the local Ollama model list: {error}") from error

    def check_connection(self) -> Dict[str, Any]:
        models = self.installed_models()
        return {"available": True, "models": models, "error": None}

    @staticmethod
    def prepare_messages(
        history: Iterable[Dict[str, str]],
        answer_style: str = "Balanced",
    ) -> List[Dict[str, str]]:
        style_instruction = ANSWER_STYLES.get(answer_style, ANSWER_STYLES["Balanced"])
        system_message = {
            "role": "system",
            "content": f"{SYSTEM_PROMPT} {style_instruction}",
        }
        valid_messages = [
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        ]
        valid_messages = valid_messages[-MAX_CONTEXT_MESSAGES:]
        kept: List[Dict[str, str]] = []
        total_chars = 0
        for item in reversed(valid_messages):
            if total_chars + len(item["content"]) > MAX_CONTEXT_CHARS and kept:
                break
            kept.append(item)
            total_chars += len(item["content"])
        return [system_message, *reversed(kept)]

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        if not model or not model.strip():
            raise ValueError("Choose an Ollama model before sending a message.")
        if not messages or not any(
            item.get("role") == "user" and item.get("content", "").strip()
            for item in messages
        ):
            raise ValueError("Enter a message before starting a response.")
        if any(
            len(item.get("content", "")) > MAX_USER_MESSAGE_CHARS
            for item in messages
            if item.get("role") == "user"
        ):
            raise ValueError(
                f"Messages must be shorter than {MAX_USER_MESSAGE_CHARS:,} characters."
            )

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": max(0.0, min(float(temperature), 2.0)),
                "num_predict": max(32, min(int(max_tokens), 4096)),
            },
        }
        response: Optional[Response] = None
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            if not response.ok:
                raise _error_message(response, model)

            received_content = False
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (TypeError, json.JSONDecodeError) as error:
                    raise OllamaError("Ollama returned an unreadable streaming response.") from error
                if event.get("error"):
                    detail = str(event["error"])
                    if "memory" in detail.lower() or "allocate" in detail.lower():
                        raise OllamaError(
                            "Ollama could not load this model with the available memory. "
                            "Close other applications or choose a smaller model."
                        )
                    raise OllamaError(f"Ollama generation failed: {detail}")
                content = event.get("message", {}).get("content", "")
                if content:
                    received_content = True
                    yield content
            if not received_content:
                raise OllamaError(
                    "Ollama finished without returning text. Check the model and try again."
                )
        except ConnectionError as error:
            raise OllamaError(
                "The Ollama service disconnected. Start Ollama and try again."
            ) from error
        except Timeout as error:
            raise OllamaError(
                "Ollama took too long to respond. Try a shorter prompt or a smaller model."
            ) from error
        except RequestException as error:
            raise OllamaError(f"Could not connect to Ollama: {error}") from error
        finally:
            if response is not None:
                response.close()
