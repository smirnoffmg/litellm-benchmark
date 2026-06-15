from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd
import seaborn as sns


def write_csv(results: list[dict[str, Any]], path: str) -> None:
    pd.DataFrame(results).to_csv(path, index=False)


# Comfort tiers per metric: (lower_bound, upper_bound, color, label)
# For token_rate higher is better, so tiers are inverted.
_GATES: dict[str, list[tuple[float, float, str, str]]] = {
    "latency_s": [
        (0, 5, "#2ecc71", "Smooth"),  # ≤ 5s
        (5, 15, "#f39c12", "Usable"),  # 5–15s
        (15, 999, "#e74c3c", "Slow"),  # > 15s
    ],
    "ttft_s": [
        (0, 2, "#2ecc71", "Smooth"),
        (2, 8, "#f39c12", "Usable"),
        (8, 999, "#e74c3c", "Slow"),
    ],
    "token_rate_tok_s": [
        (15, 999, "#2ecc71", "Smooth"),  # ≥ 15 tok/s
        (7, 15, "#f39c12", "Usable"),  # 7–15 tok/s
        (0, 7, "#e74c3c", "Slow"),  # < 7 tok/s
    ],
}


_LOG_SCALE_METRICS = {"ttft_s", "token_rate_tok_s"}


def _draw_gates(ax: matplotlib.axes.Axes, col: str, ymax: float, log_scale: bool = False) -> None:
    floor = ymax * 0.001 if log_scale else 0
    for lo, hi, color, tier_label in _GATES[col]:
        band_lo = max(lo, floor)
        band_hi = min(hi, ymax * 1.15)
        if band_lo >= band_hi:
            continue
        ax.axhspan(band_lo, band_hi, color=color, alpha=0.08, zorder=0)
        ax.axhline(band_lo, color=color, linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
        if lo > 0:
            ax.text(
                0.01,
                band_lo,
                tier_label,
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                color=color,
                va="bottom",
                alpha=0.85,
            )


def write_chart(results: list[dict[str, Any]], path: str) -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    df = pd.DataFrame(results)
    metrics = [
        ("latency_s", "Latency (s)"),
        ("ttft_s", "TTFT (s)"),
        ("token_rate_tok_s", "Token Rate (tok/s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.patch.set_facecolor("#f8f9fa")

    palette = sns.color_palette("muted", n_colors=df["model"].nunique())
    model_colors = dict(zip(df["model"].unique(), palette))

    for ax, (col, label) in zip(axes, metrics):
        ax.set_facecolor("#f0f2f5")
        sns.lineplot(
            data=df,
            x="request_index",
            y=col,
            hue="model",
            palette=model_colors,
            marker="o",
            markersize=5,
            linewidth=2,
            ax=ax,
        )
        for model, grp in df.groupby("model"):
            rolled = grp.set_index("request_index")[col].rolling(3, min_periods=1).mean()
            ax.plot(
                rolled.index,
                rolled.values,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                color=model_colors[model],
            )

        log_scale = col in _LOG_SCALE_METRICS
        if log_scale:
            ax.set_yscale("log")
        _draw_gates(ax, col, df[col].max(), log_scale=log_scale)

        ax.set_title(label, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Request #", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(title="Model", fontsize=9, title_fontsize=9)
        sns.despine(ax=ax, left=False, bottom=False)

    plt.tight_layout(pad=2.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
