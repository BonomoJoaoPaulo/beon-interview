import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from app.main import GUARDRAIL_PROMPT, OLLAMA_MODEL, app, build_messages

client = TestClient(app)


def mock_ollama(content: str | None = None, error: Exception | None = None):
    """Patch httpx.AsyncClient so no real Ollama call is made."""
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"message": {"content": content}}
    http = MagicMock()
    http.post = AsyncMock(side_effect=error, return_value=response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=http)
    cm.__aexit__ = AsyncMock(return_value=False)
    return patch("app.main.httpx.AsyncClient", return_value=cm), http


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_in_scope():
    content = json.dumps({"answer": "It places top tech talent in U.S. companies.", "in_scope": True})
    patcher, http = mock_ollama(content)
    with patcher:
        r = client.post("/ask", json={"question": "What is BEON's mission?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "It places top tech talent in U.S. companies.", "in_scope": True}

    payload = http.post.call_args.kwargs["json"]
    assert payload["model"] == OLLAMA_MODEL
    assert payload["messages"][-1] == {"role": "user", "content": "What is BEON's mission?"}


def test_ask_out_of_scope():
    content = json.dumps({"answer": "I can only answer questions about BEON.tech.", "in_scope": False})
    patcher, _ = mock_ollama(content)
    with patcher:
        r = client.post("/ask", json={"question": "Who is the president of the US?"})
    assert r.status_code == 200
    assert r.json()["in_scope"] is False


def test_prompt_includes_knowledge_base_and_guardrail():
    system = build_messages("hi")[0]["content"]
    assert "BEON.tech mission" in system
    assert GUARDRAIL_PROMPT in system


def test_ask_llm_unavailable_returns_502():
    patcher, _ = mock_ollama(error=httpx.ConnectError("refused"))
    with patcher:
        r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 502


def test_ask_invalid_llm_output_returns_502():
    patcher, _ = mock_ollama("not json")
    with patcher:
        r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 502


def test_ask_rejects_missing_or_empty_question():
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422
