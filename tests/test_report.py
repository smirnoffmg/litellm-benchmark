import os
from pathlib import Path
from typing import Any

import matplotlib.axes
import pandas as pd
import pytest

from report import (
    _PERCENTILE_DEFS,
    write_chart,
    write_csv,
    write_percentile_chart,
    write_summary_csv,
)

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


def test_write_percentile_chart_file_is_nonempty(tmp_path: Path) -> None:
    path = str(tmp_path / "percentile.png")
    write_percentile_chart(SAMPLE, path)
    assert os.path.getsize(path) > 0


def test_write_summary_csv_row_count(tmp_path: Path) -> None:
    path = str(tmp_path / "summary.csv")
    write_summary_csv(_SUMMARY_SAMPLE, path)
    df = pd.read_csv(path)
    assert len(df) == 3  # 1 model × 3 metrics


# --- Percentile consistency tests ---


def test_percentile_defs_labels() -> None:
    labels = [label for label, _ in _PERCENTILE_DEFS]
    assert labels == ["p50", "p95", "p99"]


def test_percentile_defs_excludes_p80() -> None:
    labels = [label for label, _ in _PERCENTILE_DEFS]
    assert "p80" not in labels


def test_write_percentile_chart_xticklabels_match_defs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels_seen: list[str] = []
    orig = matplotlib.axes.Axes.set_xticklabels

    def capture(self: matplotlib.axes.Axes, labels: list[str], *args: Any, **kwargs: Any) -> Any:
        labels_seen.extend(list(labels))
        return orig(self, labels, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticklabels", capture)
    write_percentile_chart(SAMPLE, str(tmp_path / "p.png"))

    expected = [label for label, _ in _PERCENTILE_DEFS]
    assert "p80" not in labels_seen
    for label in expected:
        assert label in labels_seen


# --- Failures must stay visible in the summary ---


def _row(latency: float, error: str = "") -> dict[str, Any]:
    return {
        "model": "m",
        "request_index": 0,
        "latency_s": latency,
        "ttft_s": float("nan") if error else 0.1,
        "completion_tokens": 0 if error else 10,
        "token_rate_tok_s": float("nan") if error else 20.0,
        "error": error,
    }


def test_write_summary_csv_reports_error_rate_and_sample_count(tmp_path: Path) -> None:
    rows = [_row(1.0), _row(2.0), _row(3.0), _row(float("nan"), "connection refused")]
    path = str(tmp_path / "summary.csv")
    write_summary_csv(rows, path)
    df = pd.read_csv(path)
    latency = df[df["metric"] == "latency_s"].iloc[0]
    assert latency["n"] == 3
    assert latency["error_rate"] == 0.25


def test_write_summary_csv_timeouts_land_in_latency_tail(tmp_path: Path) -> None:
    rows = [_row(1.0)] * 9 + [_row(60.0, "timeout after 60.0s")]
    path = str(tmp_path / "summary.csv")
    write_summary_csv(rows, path)
    df = pd.read_csv(path)
    latency = df[df["metric"] == "latency_s"].iloc[0]
    assert latency["p99"] > 50.0
    assert latency["n"] == 10
