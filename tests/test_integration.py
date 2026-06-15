import asyncio
from pathlib import Path
from typing import Any

import pytest

from report import write_chart, write_csv
from runner import run_benchmark


@pytest.mark.integration
def test_full_pipeline_against_ollama(
    ollama_base_url: str, ollama_model: str, tmp_path: Path
) -> None:
    results: list[dict[str, Any]] = asyncio.run(
        run_benchmark(
            base_url=f"{ollama_base_url}/v1",
            api_key="ollama",
            models=[ollama_model],
            concurrency=1,
            n_requests=1,
            prompt="Say hi",
        )
    )

    assert len(results) == 1
    row = results[0]
    assert row["error"] == ""
    assert row["latency_s"] > 0
    assert row["ttft_s"] > 0
    assert row["completion_tokens"] > 0

    csv_path = str(tmp_path / "results.csv")
    chart_path = str(tmp_path / "chart.png")
    write_csv(results, csv_path)
    write_chart(results, chart_path)

    assert Path(csv_path).stat().st_size > 0
    assert Path(chart_path).stat().st_size > 0
