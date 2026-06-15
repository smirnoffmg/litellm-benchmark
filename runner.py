import asyncio
import time
from typing import Any, Protocol, runtime_checkable

import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

MAX_RETRIES = 3


@runtime_checkable
class _ChatClient(Protocol):
    @property
    def chat(self) -> Any: ...


async def benchmark_request(
    client: _ChatClient,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    request_index: int,
) -> dict[str, Any]:
    async with semaphore:
        last_error = ""
        # Append index so identical prompts never hit LiteLLM's semantic cache.
        unique_prompt = f"{prompt} [req={request_index}]"

        for attempt in range(MAX_RETRIES + 1):
            start = time.perf_counter()
            ttft_s = float("nan")
            completion_tokens = 0
            first_chunk = True

            try:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": unique_prompt}],
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"no-store": True},  # bypass LiteLLM cache
                )
                async for chunk in stream:
                    if first_chunk and chunk.choices and chunk.choices[0].delta.content:
                        ttft_s = time.perf_counter() - start
                        first_chunk = False
                    if chunk.usage:
                        completion_tokens = chunk.usage.completion_tokens or 0
            except openai.RateLimitError as exc:
                last_error = str(exc)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2**attempt)
                continue
            except Exception as exc:
                return {
                    "model": model,
                    "request_index": request_index,
                    "latency_s": float("nan"),
                    "ttft_s": float("nan"),
                    "completion_tokens": 0,
                    "token_rate_tok_s": float("nan"),
                    "error": str(exc),
                    "retries": attempt,
                }

            latency_s = time.perf_counter() - start
            decode_s = latency_s - ttft_s
            token_rate = (
                (completion_tokens - 1) / decode_s
                if completion_tokens > 1 and decode_s > 0
                else float("nan")
            )
            return {
                "model": model,
                "request_index": request_index,
                "latency_s": latency_s,
                "ttft_s": ttft_s,
                "completion_tokens": completion_tokens,
                "token_rate_tok_s": token_rate,
                "error": "",
                "retries": attempt,
            }

        return {
            "model": model,
            "request_index": request_index,
            "latency_s": float("nan"),
            "ttft_s": float("nan"),
            "completion_tokens": 0,
            "token_rate_tok_s": float("nan"),
            "error": last_error,
            "retries": MAX_RETRIES,
        }


async def run_benchmark(
    base_url: str,
    api_key: str,
    models: list[str],
    concurrency: int,
    n_requests: int,
    prompt: str,
    delay_s: float = 1.0,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    for model in models:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        for i in range(n_requests):
            if i > 0 and delay_s > 0:
                await asyncio.sleep(delay_s)
            tasks.append(
                asyncio.create_task(benchmark_request(client, model, prompt, semaphore, i))
            )
        model_results = await tqdm.gather(*tasks, desc=model, unit="req")
        results.extend(model_results)
    return results
