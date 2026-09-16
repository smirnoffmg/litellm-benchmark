import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from report import write_chart, write_csv, write_percentile_chart, write_summary_csv
from runner import run_benchmark


def _default_stem() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a LiteLLM installation")
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        metavar="MODEL",
        help="Model name to benchmark (repeatable)",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent requests")
    parser.add_argument("--requests", type=int, default=20, help="Total requests per model")
    parser.add_argument("--prompt", default="Write a short poem about benchmarks")
    parser.add_argument(
        "--delay", type=float, default=1.0, metavar="SECONDS", help="Delay between request launches"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, metavar="N", help="Warmup requests per model (discarded)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Per-request timeout in seconds (0 to disable)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        metavar="N",
        help="Max completion tokens per request (controls output length for fair comparison)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Benchmark models one after another instead of interleaving "
        "(use when models share one backend, e.g. a local Ollama)",
    )
    parser.add_argument("--output", default=None, help="CSV output path")
    parser.add_argument("--chart", default=None, help="Chart output path")
    return parser.parse_args(argv)


_MIN_SAMPLES_FOR_P99 = 100


def result_warnings(results: list[dict[str, Any]], n_requests: int) -> list[str]:
    warnings = []
    if n_requests < _MIN_SAMPLES_FOR_P99:
        warnings.append(
            f"p99 from {n_requests} requests per model is effectively the maximum; "
            f"use --requests {_MIN_SAMPLES_FOR_P99} or more for a meaningful tail"
        )
    succeeded = [r for r in results if not r["error"]]
    no_usage = sum(1 for r in succeeded if r["completion_tokens"] == 0)
    if no_usage:
        warnings.append(
            f"{no_usage} of {len(succeeded)} successful requests returned no token usage; "
            "token rate is unavailable for them (does the proxy honour include_usage?)"
        )
    return warnings


def get_config_from_env() -> dict[str, str]:
    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM_API_KEY")
    if not base_url:
        raise ValueError("LITELLM_BASE_URL environment variable is required")
    if not api_key:
        raise ValueError("LITELLM_API_KEY environment variable is required")
    return {"base_url": base_url, "api_key": api_key}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = get_config_from_env()
    stem = _default_stem()
    output = args.output or f"results_{stem}.csv"
    chart = args.chart or f"chart_{stem}.png"
    summary = str(Path(output).with_name(Path(output).stem + "_summary.csv"))
    percentile_chart = (
        str(Path(chart).with_name(Path(chart).stem + "_percentile" + Path(chart).suffix))
        if args.chart
        else f"percentile_{stem}.png"
    )
    results: list[dict[str, Any]] = asyncio.run(
        run_benchmark(
            base_url=config["base_url"],
            api_key=config["api_key"],
            models=args.models,
            concurrency=args.concurrency,
            n_requests=args.requests,
            prompt=args.prompt,
            delay_s=args.delay,
            warmup=args.warmup,
            timeout_s=args.timeout,
            max_tokens=args.max_tokens,
            sequential=args.sequential,
        )
    )
    for warning in result_warnings(results, args.requests):
        print(f"Warning: {warning}", file=sys.stderr)
    write_csv(results, output)
    write_summary_csv(results, summary)
    write_chart(results, chart)
    write_percentile_chart(results, percentile_chart)
    print(
        f"Done. \n"
        f"Results: {output}\n"
        f"Summary: {summary}\n"
        f"Chart: {chart}\n"
        f"Percentiles: {percentile_chart}"
    )


if __name__ == "__main__":
    main()
