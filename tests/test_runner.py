import asyncio
import math
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from runner import MAX_RETRIES, benchmark_request, run_benchmark


def _make_chunk(content: str | None = None, completion_tokens: int | None = None) -> Any:
    choices: list[Any] = []
    if content is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]
    usage = None
    if completion_tokens is not None:
        usage = SimpleNamespace(completion_tokens=completion_tokens)
    return SimpleNamespace(choices=choices, usage=usage)


def _make_client(chunks: list[Any]) -> Any:
    async def create(**kwargs: Any) -> Any:
        async def _gen() -> Any:
            for c in chunks:
                yield c

        return _gen()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _make_error_client(message: str) -> Any:
    async def create(**kwargs: Any) -> Any:
        raise RuntimeError(message)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _make_patched_client(chunks: list[Any]) -> Any:
    return _make_client(chunks)


async def test_benchmark_request_returns_correct_model() -> None:
    client = _make_client(
        [
            _make_chunk(content="Hello"),
            _make_chunk(completion_tokens=1),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["model"] == "gpt-4o"


async def test_benchmark_request_returns_correct_index() -> None:
    client = _make_client(
        [
            _make_chunk(content="Hi"),
            _make_chunk(completion_tokens=1),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 7)
    assert result["request_index"] == 7


async def test_benchmark_request_latency_is_positive() -> None:
    client = _make_client(
        [
            _make_chunk(content="Hello"),
            _make_chunk(completion_tokens=5),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["latency_s"] > 0
    assert not math.isnan(result["latency_s"])


async def test_benchmark_request_ttft_le_latency() -> None:
    client = _make_client(
        [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(completion_tokens=2),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["ttft_s"] <= result["latency_s"]
    assert not math.isnan(result["ttft_s"])


async def test_benchmark_request_token_rate_uses_decode_time() -> None:
    # token_rate = (tokens - 1) / (latency - ttft): excludes the prefill phase
    client = _make_client(
        [
            _make_chunk(content="Hi"),
            _make_chunk(completion_tokens=10),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["completion_tokens"] == 10
    decode_s = result["latency_s"] - result["ttft_s"]
    expected_rate = (result["completion_tokens"] - 1) / decode_s
    assert abs(result["token_rate_tok_s"] - expected_rate) < 0.001


async def test_benchmark_request_ttft_ignores_empty_content_chunk() -> None:
    # First chunk from OpenAI carries role="assistant", content="" — not a real token.
    role_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=""))],
        usage=None,
    )
    client = _make_client(
        [
            role_chunk,
            _make_chunk(content="Hello"),
            _make_chunk(completion_tokens=1),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    # TTFT should still be measured (on the "Hello" chunk), not nan
    assert not math.isnan(result["ttft_s"])


async def test_benchmark_request_no_error_on_success() -> None:
    client = _make_client(
        [
            _make_chunk(content="Hi"),
            _make_chunk(completion_tokens=1),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["error"] == ""


async def test_benchmark_request_captures_error_message() -> None:
    client = _make_error_client("connection refused")
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["error"] == "connection refused"


async def test_benchmark_request_nan_metrics_on_error() -> None:
    client = _make_error_client("timeout")
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert math.isnan(result["latency_s"])
    assert math.isnan(result["ttft_s"])
    assert math.isnan(result["token_rate_tok_s"])


async def test_benchmark_request_zero_tokens_on_error() -> None:
    client = _make_error_client("timeout")
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["completion_tokens"] == 0


async def test_run_benchmark_result_count(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    instance = _make_patched_client(chunks)
    monkeypatch.setattr("runner.AsyncOpenAI", lambda **kwargs: instance)

    results = await run_benchmark(
        base_url="http://localhost:4000",
        api_key="sk-test",
        models=["model-a", "model-b"],
        concurrency=2,
        n_requests=3,
        prompt="test",
    )

    assert len(results) == 6


async def test_run_benchmark_contains_both_models(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    instance = _make_patched_client(chunks)
    monkeypatch.setattr("runner.AsyncOpenAI", lambda **kwargs: instance)

    results = await run_benchmark(
        base_url="http://localhost:4000",
        api_key="sk-test",
        models=["model-a", "model-b"],
        concurrency=1,
        n_requests=2,
        prompt="test",
    )

    models_in_results = {r["model"] for r in results}
    assert models_in_results == {"model-a", "model-b"}


async def test_run_benchmark_passes_base_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    instance = _make_patched_client(chunks)
    captured: list[dict[str, str]] = []

    def fake_async_openai(**kwargs: str) -> Any:
        captured.append(kwargs)
        return instance

    monkeypatch.setattr("runner.AsyncOpenAI", fake_async_openai)

    await run_benchmark(
        base_url="http://my-proxy:4000",
        api_key="sk-secret",
        models=["m"],
        concurrency=1,
        n_requests=1,
        prompt="test",
    )

    assert captured == [{"base_url": "http://my-proxy:4000", "api_key": "sk-secret"}]


async def test_run_benchmark_prints_model_progress(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    instance = _make_patched_client(chunks)
    monkeypatch.setattr("runner.AsyncOpenAI", lambda **kwargs: instance)

    await run_benchmark(
        base_url="http://localhost:4000",
        api_key="sk-test",
        models=["model-a", "model-b"],
        concurrency=1,
        n_requests=1,
        prompt="test",
    )

    err = capsys.readouterr().err
    assert "model-a" in err
    assert "model-b" in err


# --- Rate-limit / retry helpers ---


def _make_rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limit exceeded", response=response, body=None)


def _make_flaky_client(n_fails: int, chunks: list[Any]) -> Any:
    """Raises RateLimitError for the first n_fails calls, then streams chunks."""
    call_count = 0

    async def create(**kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count <= n_fails:
            raise _make_rate_limit_error()

        async def _gen() -> Any:
            for c in chunks:
                yield c

        return _gen()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _make_always_rate_limit_client() -> Any:
    async def create(**kwargs: Any) -> Any:
        raise _make_rate_limit_error()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


# --- Rate-limit tests ---


async def test_benchmark_request_zero_retries_on_success() -> None:
    client = _make_client([_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)])
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["retries"] == 0


async def test_benchmark_request_retries_once_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    client = _make_flaky_client(1, chunks)
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["retries"] == 1
    assert result["error"] == ""
    assert not math.isnan(result["latency_s"])


async def test_benchmark_request_exhausts_retries_on_persistent_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_sleep(monkeypatch)
    client = _make_always_rate_limit_client()
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["retries"] == MAX_RETRIES
    assert result["error"] != ""
    assert math.isnan(result["latency_s"])


async def test_benchmark_request_no_retry_on_generic_error() -> None:
    client = _make_error_client("connection refused")
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["retries"] == 0
    assert result["error"] == "connection refused"


async def test_run_benchmark_staggers_requests_by_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    monkeypatch.setattr("runner.AsyncOpenAI", lambda **kwargs: _make_patched_client(chunks))
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr("runner.asyncio.sleep", fake_sleep)

    await run_benchmark(
        base_url="http://localhost",
        api_key="k",
        models=["m"],
        concurrency=5,
        n_requests=3,
        prompt="p",
        delay_s=1.5,
    )

    assert sleep_calls == [1.5, 1.5]  # n_requests-1 sleeps, no sleep before request 0


async def test_run_benchmark_warmup_excluded_from_results(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    monkeypatch.setattr("runner.AsyncOpenAI", lambda **kwargs: _make_patched_client(chunks))

    results = await run_benchmark(
        base_url="http://localhost:4000",
        api_key="sk-test",
        models=["model-a"],
        concurrency=1,
        n_requests=3,
        prompt="test",
        warmup=2,
    )

    assert len(results) == 3


async def test_run_benchmark_warmup_request_indices_start_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    monkeypatch.setattr("runner.AsyncOpenAI", lambda **kwargs: _make_patched_client(chunks))

    results = await run_benchmark(
        base_url="http://localhost:4000",
        api_key="sk-test",
        models=["model-a"],
        concurrency=1,
        n_requests=2,
        prompt="test",
        warmup=2,
    )

    indices = sorted(r["request_index"] for r in results)
    assert indices == [0, 1]


async def test_run_benchmark_no_sleep_on_zero_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_make_chunk(content="Hi"), _make_chunk(completion_tokens=1)]
    monkeypatch.setattr("runner.AsyncOpenAI", lambda **kwargs: _make_patched_client(chunks))
    sleep_calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr("runner.asyncio.sleep", fake_sleep)

    await run_benchmark(
        base_url="http://localhost",
        api_key="k",
        models=["m"],
        concurrency=5,
        n_requests=3,
        prompt="p",
        delay_s=0.0,
    )

    assert sleep_calls == []


async def test_benchmark_request_sends_unique_prompt_and_no_store() -> None:
    captured: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        captured.append(kwargs)

        async def _gen() -> Any:
            yield _make_chunk(content="Hi")
            yield _make_chunk(completion_tokens=1)

        return _gen()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    await benchmark_request(client, "gpt-4o", "hello", asyncio.Semaphore(1), 3)

    assert len(captured) == 1
    msg_content = captured[0]["messages"][0]["content"]
    assert "hello" in msg_content
    assert "3" in msg_content, "request index must be embedded to bust the LiteLLM cache"
    assert captured[0].get("extra_body", {}).get("no-store") is True
