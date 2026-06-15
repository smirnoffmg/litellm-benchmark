import os
from pathlib import Path
from typing import Any

import pandas as pd

from report import write_chart, write_csv, write_percentile_chart, write_summary_csv

SAMPLE: list[dict[str, Any]] = [
    {
        "model": "gpt-4o",
        "request_index": 0,
        "latency_s": 1.2,
        "ttft_s": 0.3,
        "completion_tokens": 50,
        "token_rate_tok_s": 41.7,
        "error": "",
    },
    {
        "model": "gpt-4o",
        "request_index": 1,
        "latency_s": 1.5,
        "ttft_s": 0.4,
        "completion_tokens": 60,
        "token_rate_tok_s": 40.0,
        "error": "",
    },
    {
        "model": "claude",
        "request_index": 0,
        "latency_s": 0.9,
        "ttft_s": 0.2,
        "completion_tokens": 45,
        "token_rate_tok_s": 50.0,
        "error": "",
    },
]


def test_write_csv_creates_file(tmp_path: Path) -> None:
    path = str(tmp_path / "results.csv")
    write_csv(SAMPLE, path)
    assert os.path.exists(path)


def test_write_csv_correct_row_count(tmp_path: Path) -> None:
    path = str(tmp_path / "results.csv")
    write_csv(SAMPLE, path)
    df = pd.read_csv(path)
    assert len(df) == 3


def test_write_csv_correct_columns(tmp_path: Path) -> None:
    path = str(tmp_path / "results.csv")
    write_csv(SAMPLE, path)
    df = pd.read_csv(path)
    expected = {
        "model",
        "request_index",
        "latency_s",
        "ttft_s",
        "completion_tokens",
        "token_rate_tok_s",
        "error",
    }
    assert set(df.columns) == expected


def test_write_csv_preserves_values(tmp_path: Path) -> None:
    path = str(tmp_path / "results.csv")
    write_csv(SAMPLE, path)
    df = pd.read_csv(path)
    assert df.iloc[0]["model"] == "gpt-4o"
    assert abs(df.iloc[0]["latency_s"] - 1.2) < 0.001


def test_write_chart_creates_file(tmp_path: Path) -> None:
    path = str(tmp_path / "chart.png")
    write_chart(SAMPLE, path)
    assert os.path.exists(path)


def test_write_chart_file_is_nonempty(tmp_path: Path) -> None:
    path = str(tmp_path / "chart.png")
    write_chart(SAMPLE, path)
    assert os.path.getsize(path) > 0


_SUMMARY_SAMPLE: list[dict[str, Any]] = [
    {
        "model": "gpt-4o",
        "request_index": i,
        "latency_s": float(i + 1),
        "ttft_s": float(i + 1) * 0.2,
        "completion_tokens": 10,
        "token_rate_tok_s": 50.0 - i * 10,
        "error": "",
    }
    for i in range(3)
]


def test_write_summary_csv_creates_file(tmp_path: Path) -> None:
    path = str(tmp_path / "summary.csv")
    write_summary_csv(_SUMMARY_SAMPLE, path)
    assert os.path.exists(path)


def test_write_summary_csv_columns(tmp_path: Path) -> None:
    path = str(tmp_path / "summary.csv")
    write_summary_csv(_SUMMARY_SAMPLE, path)
    df = pd.read_csv(path)
    assert {"model", "metric", "p50", "p95", "p99"}.issubset(set(df.columns))


def test_write_summary_csv_p50_latency(tmp_path: Path) -> None:
    path = str(tmp_path / "summary.csv")
    write_summary_csv(_SUMMARY_SAMPLE, path)
    df = pd.read_csv(path)
    row = df[(df["model"] == "gpt-4o") & (df["metric"] == "latency_s")].iloc[0]
    assert abs(row["p50"] - 2.0) < 0.01  # median of [1.0, 2.0, 3.0]


def test_write_percentile_chart_creates_file(tmp_path: Path) -> None:
    path = str(tmp_path / "percentile.png")
    write_percentile_chart(SAMPLE, path)
    assert os.path.exists(path)


def test_write_percentile_chart_file_is_nonempty(tmp_path: Path) -> None:
    path = str(tmp_path / "percentile.png")
    write_percentile_chart(SAMPLE, path)
    assert os.path.getsize(path) > 0


def test_write_summary_csv_row_count(tmp_path: Path) -> None:
    path = str(tmp_path / "summary.csv")
    write_summary_csv(_SUMMARY_SAMPLE, path)
    df = pd.read_csv(path)
    assert len(df) == 3  # 1 model × 3 metrics
