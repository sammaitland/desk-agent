"""Chart rendering.

Deliberately dumb: this plots what it is given and computes nothing. Keeping
calculation out of the rendering layer means every number on a chart traces
back to an analytical tool that was separately tested.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in a server or agent context
import matplotlib.pyplot as plt  # noqa: E402

from src.tools.base import ToolResult, empty  # noqa: E402

CHART_DIR = Path(__file__).resolve().parent.parent.parent / "charts"
CHART_TYPES = ("line", "bar", "barh", "scatter")


def make_chart(
    labels: list,
    values: list[float],
    title: str,
    chart_type: str = "bar",
    x_label: str = "",
    y_label: str = "",
    series_label: str = "",
) -> ToolResult:
    """Render a chart from values you already have and return its file path.

    This tool does no analysis. Pass it the labels and values returned by an
    analytical tool — it will not recompute or verify them. Call an analytical
    tool first, then chart its output.

    chart_type: 'bar' (comparison across categories), 'barh' (same, long
    labels), 'line' (change over time), or 'scatter' (relationship between two
    numeric series, where labels are the x values).
    """
    if chart_type not in CHART_TYPES:
        return empty(f"Unknown chart_type '{chart_type}'. Use {', '.join(CHART_TYPES)}.")
    if not labels or not values:
        return empty("Nothing to chart: labels and values must both be non-empty.")
    if len(labels) != len(values):
        return empty(f"Mismatched lengths: {len(labels)} labels against {len(values)} values.")

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / f"chart_{uuid.uuid4().hex[:10]}.png"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    colours = ["#c0392b" if v < 0 else "#2c6fbb" for v in values]

    if chart_type == "bar":
        ax.bar([str(x) for x in labels], values, color=colours)
        if len(labels) > 6:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    elif chart_type == "barh":
        ax.barh([str(x) for x in labels], values, color=colours)
        ax.invert_yaxis()
    elif chart_type == "line":
        ax.plot([str(x) for x in labels], values, marker="o", linewidth=1.8, color="#2c6fbb",
                label=series_label or None)
        if len(labels) > 10:
            step = max(len(labels) // 10, 1)
            ax.set_xticks(range(0, len(labels), step))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        if series_label:
            ax.legend(frameon=False)
    else:  # scatter
        ax.scatter(labels, values, color="#2c6fbb", alpha=0.75)

    if any(v < 0 for v in values):
        ax.axhline(0, color="#555", linewidth=0.8)

    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y" if chart_type in ("bar", "line", "scatter") else "x",
            alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return ToolResult(
        data={"chart_path": str(path), "points": len(values)},
        provenance={"chart_type": chart_type, "points": len(values), "title": title},
        summary=f"Rendered a {chart_type} chart with {len(values)} points: {title}",
    )
