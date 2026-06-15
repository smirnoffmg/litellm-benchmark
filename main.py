import argparse
import asyncio
import os
from typing import Any

from report import write_chart, write_csv
from runner import run_benchmark


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
    parser.add_argument("--output", default="results.csv", help="CSV output path")
    parser.add_argument("--chart", default="chart.png", help="Chart output path")
    return parser.parse_args(argv)


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
    results: list[dict[str, Any]] = asyncio.run(
        run_benchmark(
            base_url=config["base_url"],
            api_key=config["api_key"],
            models=args.models,
            concurrency=args.concurrency,
            n_requests=args.requests,
            prompt=args.prompt,
            delay_s=args.delay,
        )
    )
    write_csv(results, args.output)
    write_chart(results, args.chart)
    print(f"Done. Results: {args.output}  Chart: {args.chart}")


if __name__ == "__main__":
    main()
