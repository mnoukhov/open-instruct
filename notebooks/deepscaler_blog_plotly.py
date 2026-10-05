"""Plotly exports for the DeepScaler section of the NGU blog post."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import to_hex


SERIF = "et-book, Palatino, 'Palatino Linotype', Georgia, serif"
BLOG_FONT_SIZE = 15
BLOG_TITLE_FONT_SIZE = 16
BLOG_LEGEND_TITLE_FONT_SIZE = 15


def write_deepscaler_improvement_by_difficulty(
    *,
    improvement_df: pd.DataFrame,
    setting_order: list[str],
    palette: dict,
    difficulty_order: list[str],
    difficulty_labels: list[str],
    output_dir: Path,
) -> Path:
    """Export the combined AIME+BRUMO improvement bars used in the blog."""

    summary = (
        improvement_df.groupby(["setting", "difficulty"], observed=True)["improvement"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    summary["se"] = (summary["sd"] / np.sqrt(summary["n"].clip(lower=1))).fillna(0.0)
    difficulty_labels = [label.replace("\n", "<br>") for label in difficulty_labels]

    fig = go.Figure()
    for setting in setting_order:
        subset = (
            summary[summary["setting"].astype(str) == setting]
            .set_index("difficulty")
            .reindex(difficulty_order)
        )
        is_ngu = setting.startswith("+ NGU ")
        fig.add_trace(
            go.Bar(
                x=difficulty_labels,
                y=subset["mean"].tolist(),
                error_y=dict(type="data", array=subset["se"].tolist(), thickness=1.2, width=3),
                marker=dict(color=to_hex(palette[setting]), line=dict(width=0)),
                name=setting.removeprefix("+ NGU "),
                legend="legend2" if is_ngu else "legend",
                hovertemplate=(
                    setting.removeprefix("+ ")
                    + "<br>%{x}<br>improvement %{y:.1f} pp<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="DeepScaler AIME + BRUMO, by difficulty",
            x=0.02,
            xanchor="left",
            font=dict(size=BLOG_TITLE_FONT_SIZE),
        ),
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, size=BLOG_FONT_SIZE, color="#111111"),
        autosize=True,
        height=430,
        margin=dict(l=68, r=180, t=52, b=64),
        barmode="group",
        bargap=0.22,
        bargroupgap=0.04,
        xaxis=dict(title_text=""),
        yaxis=dict(title_text="Pass@1 improvement (pp)", rangemode="tozero", dtick=5),
        legend=dict(
            title_text="Standard GRPO",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.015,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
        legend2=dict(
            title_text="NGU",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.015,
            xanchor="left",
            y=0.48,
            yanchor="top",
        ),
    )

    figure = json.loads(pio.to_json(fig))
    figure["blogResponsive"] = {
        "breakpoint": 480,
        "wide": {
            "legend.orientation": "v", "legend.x": 1.015, "legend.xanchor": "left",
            "legend.y": 1.0, "legend.yanchor": "top",
            "legend2.orientation": "v", "legend2.x": 1.015, "legend2.xanchor": "left",
            "legend2.y": 0.48, "legend2.yanchor": "top",
            "margin.r": 180, "margin.b": 64, "height": 430,
        },
        "narrow": {
            "legend.orientation": "v", "legend.x": 0.02, "legend.xanchor": "left",
            "legend.y": -0.25, "legend.yanchor": "top",
            "legend2.orientation": "v", "legend2.x": 0.58, "legend2.xanchor": "left",
            "legend2.y": -0.25, "legend2.yanchor": "top",
            "margin.l": 58, "margin.r": 16, "margin.b": 210, "height": 550,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "deepscaler_gradnorm1_ngu_pass_at_1_improvement_by_difficulty_combined.json"
    output.write_text(json.dumps(figure))
    print("wrote", output, f"({output.stat().st_size / 1024:.0f} KB)")
    return output
