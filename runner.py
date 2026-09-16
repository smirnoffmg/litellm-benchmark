import asyncio
import time
from collections.abc import Coroutine
from typing import Any, Protocol, runtime_checkable

import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

MAX_RETRIES = 3


@runtime_checkable
class _ChatClient(Protocol):
    @property
    def chat(self) -> Any: ...


def _row(
    model: str,
    request_index: int,
    start_time: float,
    queue_s: float,
    error: str,
    retries: int,
    latency_s: float = float("nan"),
    ttft_s: float = float("nan"),
    completion_tokens: int = 0,
    token_rate_tok_s: float = float("nan"),
) -> dict[str, Any]:
    return {
        "model": model,
        "request_index": request_index,
        "start_time": start_time,
        "queue_s": queue_s,
        "latency_s": latency_s,
        "ttft_s": ttft_s,
        "completion_tokens": completion_tokens,
        "token_rate_tok_s": token_rate_tok_s,
        "error": error,
        "retries": retries,
    }


async def benchmark_request(
    client: _ChatClient,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    request_index: int,
    timeout_s: float = 60.0,
    max_tokens: int | None = None,
    bench_start: float | None = None,
) -> dict[str, Any]:
    # Timing starts before the semaphore: queue wait and retry backoff are part of what
    # the caller experiences, and excluding them hides saturation (coordinated omission).
    submitted = time.perf_counter()
    start_time = submitted - bench_start if bench_start is not None else 0.0
    async with semaphore:
        queue_s = time.perf_counter() - submitted
        last_error = ""
        # Append index so identical prompts never hit LiteLLM's semantic cache.
        unique_prompt = f"{prompt} [req={request_index}]"
        extra: dict[str, int] = {"max_tokens": max_tokens} if max_tokens is not None else {}

        for attempt in range(MAX_RETRIES + 1):
            ttft_s = float("nan")
            completion_tokens = 0
            first_chunk = True

            try:
                async with asyncio.timeout(timeout_s if timeout_s > 0 else None):
                    stream = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": unique_prompt}],
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body={"cache": {"no-cache": True, "no-store": True}},
                        **extra,
                    )
                    async for chunk in stream:
                        if first_chunk and chunk.choices and chunk.choices[0].delta.content:
                            ttft_s = time.perf_counter() - submitted
                            first_chunk = False
                        if chunk.usage:
                            completion_tokens = chunk.usage.completion_tokens or 0
            except openai.RateLimitError as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2**attempt)
                continue
            except TimeoutError:
                # Elapsed time is a lower bound on the true latency; keeping it (instead of
                # NaN) puts timeouts into the percentile tail where they belong.
                return _row(
                    model,
                    request_index,
                    start_time,
                    queue_s,
                    f"timeout after {timeout_s}s",
                    attempt,
                    latency_s=time.perf_counter() - submitted,
                )
            except Exception as exc:
                return _row(model, request_index, start_time, queue_s, str(exc), attempt)

            latency_s = time.perf_counter() - submitted
            decode_s = latency_s - ttft_s
            token_rate = (
                (completion_tokens - 1) / decode_s
                if completion_tokens > 1 and decode_s > 0
                else float("nan")
            )
            return _row(
                model,
                request_index,
                start_time,
                queue_s,
                "",
                attempt,
                latency_s=latency_s,
                ttft_s=ttft_s,
                completion_tokens=completion_tokens,
                token_rate_tok_s=token_rate,
            )

        return _row(model, request_index, start_time, queue_s, last_error, MAX_RETRIES)


async def _launch_staggered(
    requests: list[Coroutine[Any, Any, dict[str, Any]]], delay_s: float
) -> list[asyncio.Task[dict[str, Any]]]:
    # Launches don't wait for responses, so a slow backend builds a queue
    # instead of quietly throttling the load.
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    for request in requests:
        if tasks and delay_s > 0:
            await asyncio.sleep(delay_s)
        tasks.append(asyncio.create_task(request))
    return tasks


async def run_benchmark(
    base_url: str,
    api_key: str,
    models: list[str],
    concurrency: int,
    n_requests: int,
    prompt: str,
    delay_s: float = 1.0,
    warmup: int = 3,
    timeout_s: float = 60.0,
    max_tokens: int | None = None,
    sequential: bool = False,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphores = {model: asyncio.Semaphore(concurrency) for model in models}
    bench_start = time.perf_counter()

    def timed(model: str, index: int) -> Coroutine[Any, Any, dict[str, Any]]:
        return benchmark_request(
            client, model, prompt, semaphores[model], index, timeout_s, max_tokens, bench_start
        )

    async def warm_up(model: str) -> None:
        if warmup <= 0:
            return
        requests = [
            benchmark_request(
                client, model, prompt, semaphores[model], -(i + 1), timeout_s, max_tokens
            )
            for i in range(warmup)
        ]
        tasks = await _launch_staggered(requests, delay_s)
        await tqdm.gather(*tasks, desc=f"{model} [warmup]", unit="req")

    if sequential:
        # For models sharing one backend (e.g. a single local GPU): interleaving would
        # make them wait on each other, and warming B before A runs is wasted.
        results: list[dict[str, Any]] = []
        for model in models:
            await warm_up(model)
            tasks = await _launch_staggered([timed(model, i) for i in range(n_requests)], delay_s)
            results.extend(await tqdm.gather(*tasks, desc=model, unit="req"))
        return results

    for model in models:
        await warm_up(model)
    # Round-robin so drift in provider load hits every model equally.
    requests = [timed(model, i) for i in range(n_requests) for model in models]
    tasks = await _launch_staggered(requests, delay_s)
    return list(await tqdm.gather(*tasks, desc=", ".join(models), unit="req"))
