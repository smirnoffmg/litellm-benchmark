# litellm-benchmark

Benchmarks a [LiteLLM](https://github.com/BerriAI/litellm) proxy across one or more models, measuring latency, time to first token (TTFT), and token rate. Outputs a CSV and a line-chart PNG.

## Example

![chart](./doc/img/example.png)

## Install

```bash
uv sync
```

## Configuration

Set two environment variables before running:

```bash
export LITELLM_BASE_URL=http://localhost:4000
export LITELLM_API_KEY=sk-your-key
```

## Usage

```bash
uv run python main.py \
  --model gpt-4o \
  --model claude-3-5-sonnet \
  --concurrency 5 \
  --requests 20 \
  --prompt "Write a short poem about benchmarks" \
  --output results.csv \
  --chart chart.png
```

| Flag            | Default                               | Description                       |
| --------------- | ------------------------------------- | --------------------------------- |
| `--model`       | *(required, repeatable)*              | Model name(s) to benchmark        |
| `--concurrency` | 5                                     | Max concurrent requests per model |
| `--requests`    | 20                                    | Total requests per model          |
| `--prompt`      | "Write a short poem about benchmarks" | Prompt sent to each model         |
| `--output`      | `results.csv`                         | CSV output path                   |
| `--chart`       | `chart.png`                           | Chart output path (PNG)           |

## Output

- **CSV**: one row per request with columns `model`, `request_index`, `latency_s`, `ttft_s`, `completion_tokens`, `token_rate_tok_s`, `error`, `retries`
- **Chart**: three line plots (latency, TTFT, token rate) per model with a 3-point rolling-average overlay and color-coded Smooth/Usable/Slow bands

## Metrics

LLM inference has two distinct computational phases ([BentoML inference metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics), [infercom.ai explainer](https://infercom.ai/blog/llm-inference-speed-explained)):

1. **Prefill** — the model processes the entire prompt in one forward pass and builds the KV cache. Duration scales with prompt length. No tokens are emitted yet.
2. **Decode** — the model autoregressively samples one token at a time, reusing the KV cache. Duration scales with output length.

Each measurement captures exactly one of these phases or their sum:

### `latency_s`

```
latency_s = t_last_chunk − t_request
```

End-to-end wall time. Includes both phases plus network round-trip. Useful for user-facing SLA budgets.

### `ttft_s` — Time to First Token

```
ttft_s = t_first_content_chunk − t_request
```

Time until the server emits the first real content token. Dominated by the prefill phase, so it scales with prompt length — TTFT benchmarks without a fixed prompt length are not comparable ([DigitalOcean benchmarking guide](https://www.digitalocean.com/blog/llm-inference-benchmarking)).

> **Implementation note:** OpenAI-compatible servers send an initial chunk with `delta.role = "assistant"` and `delta.content = ""` before any token is generated. This chunk is skipped; `ttft_s` starts the clock on the first chunk where `delta.content` is non-empty ([OpenAI streaming reference](https://developers.openai.com/api/docs/guides/streaming-responses)).

### `token_rate_tok_s` — Decode throughput

The standard industry metric is **TPOT** (Time Per Output Token), defined as ([BentoML](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)):

```
TPOT = (latency_s − ttft_s) / (completion_tokens − 1)
```

`token_rate_tok_s` is its inverse — tokens per second during the decode phase:

```
token_rate_tok_s = (completion_tokens − 1) / (latency_s − ttft_s)
```

The `− 1` accounts for the first token being already captured by `ttft_s`; the remaining `completion_tokens − 1` tokens are produced during the decode window `latency_s − ttft_s`. Using total `latency_s` as the denominator would dilute the rate with prefill time and make it prompt-length-dependent. Returns `nan` when output is ≤ 1 token or the decode window is zero.

### `retries`

Number of HTTP 429 responses received before the request succeeded (or was abandoned). Requests are retried up to 3 times with exponential backoff: 1 s, 2 s, 4 s. A non-zero value means the proxy was rate-limiting during the run and the reported latency is understated relative to what a client without retry logic would see.