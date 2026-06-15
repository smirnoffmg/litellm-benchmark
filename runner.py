import asyncio
import time
from typing import Any

from openai import AsyncOpenAI


async def benchmark_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    request_index: int,
) -> dict[str, Any]:
    async with semaphore:
        start = time.perf_counter()
        ttft_s = float("nan")
        completion_tokens = 0
        error = ""
        first_chunk = True

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if first_chunk and chunk.choices:
                    ttft_s = time.perf_counter() - start
                    first_chunk = False
                if chunk.usage:
                    completion_tokens = chunk.usage.completion_tokens or 0
        except Exception as exc:
            error = str(exc)

        latency_s = time.perf_counter() - start

        if error:
            return {
                "model": model,
                "request_index": request_index,
                "latency_s": float("nan"),
                "ttft_s": float("nan"),
                "completion_tokens": 0,
                "token_rate_tok_s": float("nan"),
                "error": error,
            }

        token_rate = completion_tokens / latency_s if completion_tokens else float("nan")

        return {
            "model": model,
            "request_index": request_index,
            "latency_s": latency_s,
            "ttft_s": ttft_s,
            "completion_tokens": completion_tokens,
            "token_rate_tok_s": token_rate,
            "error": error,
        }


async def run_benchmark(
    base_url: str,
    api_key: str,
    models: list[str],
    concurrency: int,
    n_requests: int,
    prompt: str,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    for model in models:
        print(f"Benchmarking {model} ({n_requests} requests, concurrency={concurrency})...")
        tasks = [benchmark_request(client, model, prompt, semaphore, i) for i in range(n_requests)]
        model_results = await asyncio.gather(*tasks)
        results.extend(model_results)
    return results
