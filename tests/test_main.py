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
    assert ns.output is None
    assert ns.chart is None
    assert ns.warmup == 3


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
    monkeypatch.setattr("main._default_stem", lambda: "20260615_120000")

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
    summary_calls: list[tuple[list[dict[str, Any]], str]] = []
    chart_calls: list[tuple[list[dict[str, Any]], str]] = []
    percentile_calls: list[tuple[list[dict[str, Any]], str]] = []

    async def fake_run_benchmark(**kwargs: Any) -> list[dict[str, Any]]:
        run_calls.append(kwargs)
        return fake_results

    def fake_write_csv(results: list[dict[str, Any]], path: str) -> None:
        csv_calls.append((results, path))

    def fake_write_summary_csv(results: list[dict[str, Any]], path: str) -> None:
        summary_calls.append((results, path))

    def fake_write_chart(results: list[dict[str, Any]], path: str) -> None:
        chart_calls.append((results, path))

    def fake_write_percentile_chart(results: list[dict[str, Any]], path: str) -> None:
        percentile_calls.append((results, path))

    monkeypatch.setattr("main.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("main.write_csv", fake_write_csv)
    monkeypatch.setattr("main.write_summary_csv", fake_write_summary_csv)
    monkeypatch.setattr("main.write_chart", fake_write_chart)
    monkeypatch.setattr("main.write_percentile_chart", fake_write_percentile_chart)

    main(
        [
            "--model",
            "gpt-4o",
            "--requests",
            "1",
            "--concurrency",
            "3",
            "--prompt",
            "hello",
            "--delay",
            "0.5",
            "--warmup",
            "1",
        ]
    )

    assert len(run_calls) == 1
    kw = run_calls[0]
    assert kw["models"] == ["gpt-4o"]
    assert kw["base_url"] == "http://localhost:4000"
    assert kw["api_key"] == "sk-test"
    assert kw["n_requests"] == 1
    assert kw["concurrency"] == 3
    assert kw["prompt"] == "hello"
    assert kw["delay_s"] == 0.5
    assert kw["warmup"] == 1
    assert kw["timeout_s"] == 60.0
    assert kw["max_tokens"] is None

    assert csv_calls[0] == (fake_results, "results_20260615_120000.csv")
    assert summary_calls[0][1] == "results_20260615_120000_summary.csv"
    assert chart_calls[0] == (fake_results, "chart_20260615_120000.png")
    assert percentile_calls[0][1] == "percentile_20260615_120000.png"


def test_main_custom_output_overrides_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    from main import main

    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setattr("main._default_stem", lambda: "20260615_120000")

    csv_calls: list[str] = []
    chart_calls: list[str] = []

    async def fake_run(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr("main.run_benchmark", fake_run)
    monkeypatch.setattr("main.write_csv", lambda r, p: csv_calls.append(p))
    monkeypatch.setattr("main.write_summary_csv", lambda r, p: None)
    monkeypatch.setattr("main.write_chart", lambda r, p: chart_calls.append(p))
    monkeypatch.setattr("main.write_percentile_chart", lambda r, p: None)

    main(["--model", "gpt-4o", "--requests", "1", "--output", "my.csv", "--chart", "my.png"])

    assert csv_calls == ["my.csv"]
    assert chart_calls == ["my.png"]


def test_parse_args_warmup_custom() -> None:
    ns = parse_args(["--model", "gpt-4o", "--warmup", "5"])
    assert ns.warmup == 5


def test_parse_args_timeout_default() -> None:
    ns = parse_args(["--model", "gpt-4o"])
    assert ns.timeout == 60.0


def test_parse_args_timeout_custom() -> None:
    ns = parse_args(["--model", "gpt-4o", "--timeout", "30"])
    assert ns.timeout == 30.0


def test_parse_args_max_tokens_default() -> None:
    ns = parse_args(["--model", "gpt-4o"])
    assert ns.max_tokens is None


def test_parse_args_max_tokens_custom() -> None:
    ns = parse_args(["--model", "gpt-4o", "--max-tokens", "200"])
    assert ns.max_tokens == 200


def test_main_passes_timeout_and_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    from main import main

    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")

    run_calls: list[dict[str, Any]] = []

    async def fake_run(**kwargs: Any) -> list[dict[str, Any]]:
        run_calls.append(kwargs)
        return []

    monkeypatch.setattr("main.run_benchmark", fake_run)
    monkeypatch.setattr("main.write_csv", lambda r, p: None)
    monkeypatch.setattr("main.write_summary_csv", lambda r, p: None)
    monkeypatch.setattr("main.write_chart", lambda r, p: None)
    monkeypatch.setattr("main.write_percentile_chart", lambda r, p: None)

    main(["--model", "gpt-4o", "--requests", "1", "--timeout", "45", "--max-tokens", "100"])

    assert run_calls[0]["timeout_s"] == 45.0
    assert run_calls[0]["max_tokens"] == 100


# --- Result-quality warnings ---


def _ok_row(completion_tokens: int = 10) -> dict[str, Any]:
    return {"model": "m", "completion_tokens": completion_tokens, "error": ""}


def test_result_warnings_empty_for_large_clean_run() -> None:
    from main import result_warnings

    assert result_warnings([_ok_row()] * 100, n_requests=100) == []


def test_result_warnings_flags_small_sample_for_p99() -> None:
    from main import result_warnings

    warnings = result_warnings([_ok_row()] * 20, n_requests=20)
    assert len(warnings) == 1
    assert "p99" in warnings[0]


def test_result_warnings_flags_missing_usage() -> None:
    from main import result_warnings

    rows = [_ok_row(0), _ok_row(0), _ok_row(10)] * 50
    warnings = result_warnings(rows, n_requests=150)
    assert len(warnings) == 1
    assert "100 of 150" in warnings[0]
    assert "usage" in warnings[0]


def test_result_warnings_ignores_failed_rows_for_usage() -> None:
    from main import result_warnings

    failed = {"model": "m", "completion_tokens": 0, "error": "timeout after 60.0s"}
    assert result_warnings([failed] * 100, n_requests=100) == []


def test_main_prints_warnings_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from main import main

    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")

    async def fake_run(**kwargs: Any) -> list[dict[str, Any]]:
        return [_ok_row()]

    monkeypatch.setattr("main.run_benchmark", fake_run)
    for name in ("write_csv", "write_summary_csv", "write_chart", "write_percentile_chart"):
        monkeypatch.setattr(f"main.{name}", lambda r, p: None)

    main(["--model", "m", "--requests", "1"])

    assert "p99" in capsys.readouterr().err


def _capture_chart_paths(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    paths: dict[str, str] = {}

    async def fake_run(**kwargs: Any) -> list[dict[str, Any]]:
        return []

    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    monkeypatch.setattr("main._default_stem", lambda: "20260615_120000")
    monkeypatch.setattr("main.run_benchmark", fake_run)
    monkeypatch.setattr("main.write_csv", lambda r, p: None)
    monkeypatch.setattr("main.write_summary_csv", lambda r, p: None)
    monkeypatch.setattr("main.write_chart", lambda r, p: paths.update(chart=p))
    monkeypatch.setattr("main.write_percentile_chart", lambda r, p: paths.update(percentile=p))
    return paths


def test_main_percentile_chart_follows_custom_chart_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from main import main

    paths = _capture_chart_paths(monkeypatch)
    main(["--model", "m", "--requests", "100", "--chart", "out/my.png"])

    assert paths["percentile"] == "out/my_percentile.png"


def test_main_percentile_chart_default_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from main import main

    paths = _capture_chart_paths(monkeypatch)
    main(["--model", "m", "--requests", "100"])

    assert paths["percentile"] == "percentile_20260615_120000.png"


def test_parse_args_sequential_default_off() -> None:
    assert parse_args(["--model", "m"]).sequential is False


def test_parse_args_sequential_flag() -> None:
    assert parse_args(["--model", "m", "--sequential"]).sequential is True


def test_main_passes_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    from main import main

    run_calls: list[dict[str, Any]] = []

    async def fake_run(**kwargs: Any) -> list[dict[str, Any]]:
        run_calls.append(kwargs)
        return []

    _capture_chart_paths(monkeypatch)
    monkeypatch.setattr("main.run_benchmark", fake_run)
    main(["--model", "m", "--requests", "100", "--sequential"])

    assert run_calls[0]["sequential"] is True
