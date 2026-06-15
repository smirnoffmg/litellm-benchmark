import asyncio
import math
from types import SimpleNamespace
from typing import Any

import pytest

from runner import benchmark_request, run_benchmark


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


async def test_benchmark_request_token_rate_equals_tokens_over_latency() -> None:
    client = _make_client(
        [
            _make_chunk(content="Hi"),
            _make_chunk(completion_tokens=10),
        ]
    )
    result = await benchmark_request(client, "gpt-4o", "prompt", asyncio.Semaphore(1), 0)
    assert result["completion_tokens"] == 10
    expected_rate = 10 / result["latency_s"]
    assert abs(result["token_rate_tok_s"] - expected_rate) < 0.01


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

    out = capsys.readouterr().out
    assert "Benchmarking model-a" in out
    assert "Benchmarking model-b" in out
