from typing import Any

import pytest

from main import get_config_from_env, parse_args


def test_parse_args_single_model() -> None:
    ns = parse_args(["--model", "gpt-4o"])
    assert ns.models == ["gpt-4o"]


def test_parse_args_multiple_models() -> None:
    ns = parse_args(["--model", "gpt-4o", "--model", "claude-3-5-sonnet"])
    assert ns.models == ["gpt-4o", "claude-3-5-sonnet"]


def test_parse_args_defaults() -> None:
    ns = parse_args(["--model", "gpt-4o"])
    assert ns.concurrency == 5
    assert ns.requests == 20
    assert ns.prompt == "Write a short poem about benchmarks"
    assert ns.output == "results.csv"
    assert ns.chart == "chart.png"


def test_parse_args_custom_values() -> None:
    ns = parse_args(
        [
            "--model",
            "gpt-4o",
            "--concurrency",
            "2",
            "--requests",
            "10",
            "--prompt",
            "Say hello",
            "--output",
            "out.csv",
            "--chart",
            "out.png",
        ]
    )
    assert ns.concurrency == 2
    assert ns.requests == 10
    assert ns.prompt == "Say hello"
    assert ns.output == "out.csv"
    assert ns.chart == "out.png"


def test_get_config_raises_when_base_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LITELLM_BASE_URL"):
        get_config_from_env()


def test_get_config_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LITELLM_API_KEY"):
        get_config_from_env()


def test_get_config_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    cfg = get_config_from_env()
    assert cfg == {"base_url": "http://localhost:4000", "api_key": "sk-test"}


def test_main_calls_run_benchmark_and_writes_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from main import main

    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")

    fake_results = [
        {
            "model": "gpt-4o",
            "request_index": 0,
            "latency_s": 1.0,
            "ttft_s": 0.2,
            "completion_tokens": 10,
            "token_rate_tok_s": 10.0,
            "error": "",
        },
    ]

    run_calls: list[dict[str, Any]] = []
    csv_calls: list[tuple[list[dict[str, Any]], str]] = []
    chart_calls: list[tuple[list[dict[str, Any]], str]] = []

    async def fake_run_benchmark(**kwargs: Any) -> list[dict[str, Any]]:
        run_calls.append(kwargs)
        return fake_results

    def fake_write_csv(results: list[dict[str, Any]], path: str) -> None:
        csv_calls.append((results, path))

    def fake_write_chart(results: list[dict[str, Any]], path: str) -> None:
        chart_calls.append((results, path))

    monkeypatch.setattr("main.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("main.write_csv", fake_write_csv)
    monkeypatch.setattr("main.write_chart", fake_write_chart)

    main(["--model", "gpt-4o", "--requests", "1"])

    assert len(run_calls) == 1
    assert run_calls[0]["models"] == ["gpt-4o"]
    assert run_calls[0]["base_url"] == "http://localhost:4000"
    assert run_calls[0]["api_key"] == "sk-test"
    assert run_calls[0]["n_requests"] == 1

    assert csv_calls == [(fake_results, "results.csv")]
    assert chart_calls == [(fake_results, "chart.png")]
