from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.code_execution import router as code_router
from app.config import settings


def test_code_execute_endpoint_returns_stdout(monkeypatch):
    app = FastAPI()
    app.include_router(code_router)
    client = TestClient(app)

    # Enable execution for the test
    monkeypatch.setattr(settings, "enable_code_execution", True, raising=False)

    # Avoid network calls
    import app.routers.code_execution as r

    class _DummySandbox:
        async def execute_code(self, code, language, test_cases=None, stdin="", trace=False, trace_max_events=2000):
            return {
                "success": True,
                "output": "hello\n",
                "error": "",
                "execution_time": 0.12,
                "memory_used": 1024,
                "status": "Accepted",
            }

    monkeypatch.setattr(r, "CodeExecutionSandbox", lambda **kwargs: _DummySandbox())

    resp = client.post(
        "/api/code/execute",
        json={"language": "python", "code": "print('hello')", "stdin": ""},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["stdout"].strip() == "hello"
    assert payload["stderr"] == ""
    assert payload["status"] == "Accepted"


def test_code_execute_endpoint_trace_events(monkeypatch):
    app = FastAPI()
    app.include_router(code_router)
    client = TestClient(app)

    monkeypatch.setattr(settings, "enable_code_execution", True, raising=False)

    import app.routers.code_execution as r

    class _DummySandbox:
        async def execute_code(self, code, language, test_cases=None, stdin="", trace=False, trace_max_events=2000):
            assert trace is True
            assert int(trace_max_events) == 3
            return {
                "success": True,
                "output": "ok\n",
                "error": "",
                "execution_time": 0.01,
                "memory_used": 123,
                "status": "Accepted",
                "trace_events": [
                    {"step": 1, "line": 1, "event": "line", "locals": {"x": "1"}},
                    {"step": 2, "line": 2, "event": "line", "locals": {"x": "2"}},
                ],
            }

    monkeypatch.setattr(r, "CodeExecutionSandbox", lambda **kwargs: _DummySandbox())

    resp = client.post(
        "/api/code/execute",
        json={
            "language": "python",
            "code": "print('ok')",
            "stdin": "",
            "trace": True,
            "trace_max_events": 3,
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["stdout"].strip() == "ok"
    assert isinstance(payload.get("trace_events"), list)
    assert payload["trace_events"][0]["line"] == 1
    assert payload["trace_events"][0]["locals"]["x"] == "1"


def test_code_execute_endpoint_trace_explanations(monkeypatch):
    app = FastAPI()
    app.include_router(code_router)
    client = TestClient(app)

    monkeypatch.setattr(settings, "enable_code_execution", True, raising=False)

    import app.routers.code_execution as r

    class _DummySandbox:
        async def execute_code(self, code, language, test_cases=None, stdin="", trace=False, trace_max_events=2000):
            return {
                "success": True,
                "output": "ok\n",
                "error": "",
                "execution_time": 0.01,
                "memory_used": 123,
                "status": "Accepted",
                "trace_events": [
                    {"step": 1, "line": 1, "event": "line", "locals": {"x": "1"}},
                    {"step": 2, "line": 2, "event": "line", "locals": {"x": "2"}},
                ],
            }

    monkeypatch.setattr(r, "CodeExecutionSandbox", lambda **kwargs: _DummySandbox())
    monkeypatch.setattr(r, "explain_lines", lambda **kwargs: {1: "Explains line 1", 2: "Explains line 2"})

    resp = client.post(
        "/api/code/execute",
        json={
            "language": "python",
            "code": "x=1\nprint(x)\n",
            "trace": True,
            "explain_trace": True,
            "trace_max_events": 10,
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["trace_events"][0]["explanation"] == "Explains line 1"
    assert payload["line_explanations"]["1"] == "Explains line 1"


def test_code_execute_endpoint_trace_fallback_for_other_languages(monkeypatch):
    app = FastAPI()
    app.include_router(code_router)
    client = TestClient(app)

    monkeypatch.setattr(settings, "enable_code_execution", True, raising=False)

    import app.routers.code_execution as r

    class _DummySandbox:
        async def execute_code(self, code, language, test_cases=None, stdin="", trace=False, trace_max_events=2000):
            # Simulate the sandbox returning trace_events from stdout timeline.
            return {
                "success": True,
                "output": "a\nb\n",
                "error": "",
                "execution_time": 0.01,
                "memory_used": 123,
                "status": "Accepted",
                "trace_events": [
                    {"step": 1, "line": 0, "event": "stdout", "locals": None},
                    {"step": 2, "line": 0, "event": "stdout", "locals": None},
                ],
            }

    monkeypatch.setattr(r, "CodeExecutionSandbox", lambda **kwargs: _DummySandbox())

    resp = client.post(
        "/api/code/execute",
        json={
            "language": "javascript",
            "code": "console.log('a'); console.log('b');",
            "trace": True,
            "trace_max_events": 10,
        },
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["stdout"].strip() == "a\nb"
    assert payload["trace_events"][0]["event"] == "stdout"


def test_code_execute_endpoint_disabled(monkeypatch):
    app = FastAPI()
    app.include_router(code_router)
    client = TestClient(app)

    monkeypatch.setattr(settings, "enable_code_execution", False, raising=False)

    resp = client.post(
        "/api/code/execute",
        json={"language": "python", "code": "print('x')"},
    )

    assert resp.status_code == 503
