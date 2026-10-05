"""Plotly exports for the NGU section of the blog post."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import to_rgb
from plotly.subplots import make_subplots


SERIF = "et-book, Palatino, 'Palatino Linotype', Georgia, serif"
BLOG_FONT_SIZE = 15
BLOG_TITLE_FONT_SIZE = 16
BLOG_LEGEND_TITLE_FONT_SIZE = 15
GRPO_GRAY_SHADES = ["#d9d9d9", "#bdbdbd", "#969696", "#636363"]
NGU_COLOR = "#1f6fb4"
DIFFICULTY_FIGURE_HEIGHT = 430
DIFFICULTY_FIGURE_NARROW_HEIGHT = 540
DIFFICULTY_FIGURE_RIGHT_MARGIN = 170


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = to_rgb(color)
    return f"rgba({red * 255:.0f},{green * 255:.0f},{blue * 255:.0f},{alpha})"


def _add_band_and_line(
    fig: go.Figure,
    data: pd.DataFrame,
    *,
    y_column: str,
    name: str,
    color: str,
    spread: str,
    line_width: float = 2,
    row: int | None = None,
    col: int | None = None,
) -> None:
    grouped = (
        data.groupby("Step", observed=True)[y_column]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
        .sort_values("Step")
    )
    if grouped.empty:
        return
    if spread == "se":
        error = (grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))).fillna(0.0)
    elif spread == "sd":
        error = grouped["sd"].fillna(0.0)
    else:
        raise ValueError(f"Unknown spread: {spread}")

    x = grouped["Step"].tolist()
    mean = grouped["mean"].tolist()
    upper = (grouped["mean"] + error).tolist()
    lower = (grouped["mean"] - error).tolist()
    target = {} if row is None else {"row": row, "col": col}

    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=_rgba(color, 0.12),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        **target,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=name,
            showlegend=False,
            hovertemplate=f"{name}<br>step %{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ),
        **target,
    )


def _write_figure(fig: go.Figure, output_dir: Path, name: str, responsive: dict) -> Path:
    figure = json.loads(pio.to_json(fig))
    figure["blogResponsive"] = responsive
    output = output_dir / f"{name}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(figure))
    print("wrote", output, f"({output.stat().st_size / 1024:.0f} KB)")
    return output


def _blend_with_white(color: str, amount: float) -> str:
    rgb = np.asarray(to_rgb(color))
    mixed = (1 - amount) + amount * rgb
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in mixed)


def _summarize_difficulty(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["metric", "setting", "Step"], observed=True)["pass_at_1_percent"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    summary["se"] = (summary["sd"] / np.sqrt(summary["n"].clip(lower=1))).fillna(0.0)
    return summary


def _add_summary_band_and_line(
    fig: go.Figure,
    summary: pd.DataFrame,
    *,
    name: str,
    color: str,
    dash: str = "solid",
) -> None:
    summary = summary.sort_values("Step")
    x = summary["Step"].tolist()
    mean = summary["mean"].tolist()
    upper = (summary["mean"] + summary["se"]).tolist()
    lower = (summary["mean"] - summary["se"]).tolist()
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=_rgba(color, 0.12),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            line=dict(color=color, width=2, dash=dash),
            name=name,
            showlegend=False,
            hovertemplate=f"{name}<br>step %{{x}}<br>%{{y:.1f}}%<extra></extra>",
        )
    )


def _finish_difficulty_figure(
    fig: go.Figure,
    *,
    title: str,
    max_step: int,
    first_legend_title: str,
    first_legend_y: float,
) -> dict:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=BLOG_TITLE_FONT_SIZE)),
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, size=BLOG_FONT_SIZE, color="#111111"),
        autosize=True,
        height=DIFFICULTY_FIGURE_HEIGHT,
        margin=dict(l=58, r=DIFFICULTY_FIGURE_RIGHT_MARGIN, t=52, b=56),
        xaxis=dict(title_text="Step", range=[0, max_step], dtick=400, minor=dict(dtick=200, showgrid=True)),
        yaxis=dict(title_text="pass@1 (%)", range=[0, 100], dtick=20),
        legend=dict(
            title_text=first_legend_title,
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.015,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
        legend2=dict(
            title_text="Eval Difficulty",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.015,
            xanchor="left",
            y=first_legend_y,
            yanchor="top",
        ),
    )
    return {
        "breakpoint": 480,
        "wide": {
            "legend.orientation": "v", "legend.x": 1.015, "legend.xanchor": "left",
            "legend.y": 1.0, "legend.yanchor": "top",
            "legend2.orientation": "v", "legend2.x": 1.015, "legend2.xanchor": "left",
            "legend2.y": first_legend_y, "legend2.yanchor": "top",
            "margin.r": DIFFICULTY_FIGURE_RIGHT_MARGIN, "margin.b": 56,
            "height": DIFFICULTY_FIGURE_HEIGHT,
        },
        "narrow": {
            "legend.orientation": "h", "legend.x": 0.5, "legend.xanchor": "center",
            "legend.y": -0.28, "legend.yanchor": "top",
            "legend2.orientation": "h", "legend2.x": 0.5, "legend2.xanchor": "center",
            "legend2.y": -0.55, "legend2.yanchor": "top",
            "xaxis.dtick": 800, "margin.r": 16, "margin.b": 230,
            "height": DIFFICULTY_FIGURE_NARROW_HEIGHT,
        },
    }


def write_manufactoria_matthew_after_adaptation(
    *,
    adapted_df: pd.DataFrame,
    bucket_order: list[str],
    bucket_colors: dict[str, str],
    output_dir: Path,
) -> Path:
    """Export the adapted-difficulty Manufactoria panel, omitting its first point."""

    fig = go.Figure()
    plotted_steps: list[float] = []
    for bucket in bucket_order:
        series = (
            adapted_df[adapted_df["bucket"].astype(str) == bucket]
            .groupby("Step", observed=True)["pass_rate"]
            .mean()
            .sort_index()
            .iloc[1:]
        )
        if series.empty:
            continue
        x = series.index.tolist()
        y = (series * 100).tolist()
        plotted_steps.extend(x)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=bucket_colors[bucket], width=2.5),
                name=bucket,
                hovertemplate=f"{bucket}<br>step %{{x}}<br>%{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Matthew Effect on Code RL in Manufactoria",
            x=0.02,
            xanchor="left",
            font=dict(size=BLOG_TITLE_FONT_SIZE),
        ),
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, size=BLOG_FONT_SIZE, color="#111111"),
        autosize=True,
        height=DIFFICULTY_FIGURE_HEIGHT,
        margin=dict(l=58, r=130, t=52, b=56),
        xaxis=dict(
            title_text="Training Steps",
            range=[min(plotted_steps), max(plotted_steps)],
            dtick=500,
            minor=dict(dtick=250, showgrid=True),
        ),
        yaxis=dict(title_text="pass@1 (%)", range=[0, 100], dtick=20),
        legend=dict(
            title_text="Test Difficulty",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.015,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
    )
    responsive = {
        "breakpoint": 480,
        "wide": {
            "legend.orientation": "v", "legend.x": 1.015, "legend.xanchor": "left",
            "legend.y": 1.0, "legend.yanchor": "top",
            "margin.r": 130, "margin.b": 56, "height": DIFFICULTY_FIGURE_HEIGHT,
        },
        "narrow": {
            "legend.orientation": "h", "legend.x": 0.5, "legend.xanchor": "center",
            "legend.y": -0.32, "legend.yanchor": "top",
            "xaxis.dtick": 1000, "margin.r": 16, "margin.b": 145,
            "height": 500,
        },
    }
    return _write_figure(
        fig,
        output_dir,
        "manufactoria_matthew_after_adaptation",
        responsive,
    )


def write_manufactoria_grpo_vs_ngu(
    *,
    objective_df: pd.DataFrame,
    metric_titles: dict[str, str],
    method_order: list[str],
    method_colors: dict[str, str],
    output_dir: Path,
) -> Path:
    """Export the three Manufactoria metrics with a centered third panel."""

    fig = make_subplots(
        rows=2,
        cols=4,
        specs=[
            [{"colspan": 2}, None, {"colspan": 2}, None],
            [None, {"colspan": 2}, None, None],
        ],
        subplot_titles=list(metric_titles.values()),
        horizontal_spacing=0.12,
        vertical_spacing=0.24,
    )
    panel_positions = [(1, 1), (1, 3), (2, 2)]
    max_step = int(objective_df["Step"].max())
    legend_labels = {"GRPO": "Standard GRPO", "NGU": "NGU"}

    for panel_index, ((metric, _), (row, col)) in enumerate(
        zip(metric_titles.items(), panel_positions)
    ):
        for method in method_order:
            series = (
                objective_df[
                    (objective_df["method"].astype(str) == method)
                    & objective_df[metric].notna()
                ]
                .groupby("Step", observed=True)[metric]
                .mean()
                .sort_index()
                * 100
            )
            fig.add_trace(
                go.Scatter(
                    x=series.index.tolist(),
                    y=series.tolist(),
                    mode="lines",
                    line=dict(color=method_colors[method], width=2.5),
                    name=legend_labels.get(method, method),
                    legendgroup=method,
                    legendrank=0 if method == "NGU" else 1,
                    showlegend=panel_index == 0,
                    hovertemplate=(
                        f"{legend_labels.get(method, method)}<br>step %{{x}}"
                        f"<br>%{{y:.1f}}%<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    fig.update_xaxes(
        title_text="Training Steps",
        range=[0, max_step],
        dtick=1000,
        minor=dict(dtick=500, showgrid=True),
    )
    fig.update_yaxes(title_text="pass@1 (%)", range=[0, 100], dtick=20)
    fig.update_layout(
        title=dict(
            text="NGU on Manufactoria",
            x=0.02,
            xanchor="left",
            font=dict(size=BLOG_TITLE_FONT_SIZE),
        ),
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, size=BLOG_FONT_SIZE, color="#111111"),
        autosize=True,
        height=720,
        margin=dict(l=58, r=24, t=100, b=60),
        legend=dict(
            orientation="v",
            x=0.43,
            xanchor="right",
            y=0.64,
            yanchor="bottom",
        ),
    )
    fig.update_annotations(font=dict(size=BLOG_FONT_SIZE))

    responsive = {
        "breakpoint": 480,
        "wide": {
            "xaxis.domain": [0.0, 0.44], "yaxis.domain": [0.62, 1.0],
            "xaxis2.domain": [0.56, 1.0], "yaxis2.domain": [0.62, 1.0],
            "xaxis3.domain": [0.28, 0.72], "yaxis3.domain": [0.0, 0.38],
            "annotations[0].x": 0.22, "annotations[0].y": 1.0,
            "annotations[1].x": 0.78, "annotations[1].y": 1.0,
            "annotations[2].x": 0.50, "annotations[2].y": 0.38,
            "legend.orientation": "v", "legend.x": 0.43, "legend.xanchor": "right",
            "legend.y": 0.64, "legend.yanchor": "bottom",
            "height": 720, "margin.l": 58, "margin.r": 24,
        },
        "narrow": {
            "xaxis.domain": [0.0, 1.0], "yaxis.domain": [0.76, 1.0],
            "xaxis2.domain": [0.0, 1.0], "yaxis2.domain": [0.38, 0.62],
            "xaxis3.domain": [0.0, 1.0], "yaxis3.domain": [0.0, 0.24],
            "annotations[0].x": 0.50, "annotations[0].y": 1.0,
            "annotations[1].x": 0.50, "annotations[1].y": 0.62,
            "annotations[2].x": 0.50, "annotations[2].y": 0.24,
            "legend.orientation": "v", "legend.x": 0.98, "legend.xanchor": "right",
            "legend.y": 0.77, "legend.yanchor": "bottom",
            "xaxis.dtick": 1000, "xaxis2.dtick": 1000, "xaxis3.dtick": 1000,
            "height": 1200, "margin.l": 58, "margin.r": 16,
        },
    }
    return _write_figure(fig, output_dir, "manufactoria_grpo_vs_ngu", responsive)


def write_manufactoria_reward_compute_time(
    *,
    reward_df: pd.DataFrame,
    series_order: list[str],
    series_colors: dict[str, str],
    output_dir: Path,
) -> Path:
    """Export Manufactoria recovery performance against total compute time."""

    fig = go.Figure()
    for series_name in series_order:
        series = reward_df[reward_df["series"].astype(str) == series_name].sort_values(
            "compute_hours"
        )
        fig.add_trace(
            go.Scatter(
                x=series["compute_hours"].tolist(),
                y=series["pass_at_1"].tolist(),
                mode="lines",
                line=dict(
                    color=series_colors[series_name],
                    width=3.5 if series_name == "Per-Test Warmup" else 2.5,
                ),
                name=series_name,
                hovertemplate=(
                    f"{series_name}<br>%{{x:.1f}} compute hours"
                    "<br>pass@1 %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, size=BLOG_FONT_SIZE, color="#111111"),
        autosize=True,
        height=430,
        margin=dict(l=62, r=170, t=24, b=62),
        xaxis=dict(
            title_text="Compute Time (hours)",
            range=[0, 88],
            dtick=10,
            minor=dict(dtick=5, showgrid=True),
        ),
        yaxis=dict(title_text="pass@1", range=[0, 0.5], dtick=0.1),
        legend=dict(
            orientation="v",
            x=1.015,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
    )
    responsive = {
        "breakpoint": 480,
        "wide": {
            "legend.orientation": "v", "legend.x": 1.015, "legend.xanchor": "left",
            "legend.y": 1.0, "legend.yanchor": "top",
            "xaxis.dtick": 10, "margin.l": 62, "margin.r": 170,
            "height": 430,
        },
        "narrow": {
            "legend.orientation": "h", "legend.x": 0.5, "legend.xanchor": "center",
            "legend.y": -0.30, "legend.yanchor": "top",
            "xaxis.dtick": 20, "margin.l": 58, "margin.r": 16, "margin.b": 180,
            "height": 480,
        },
    }
    return _write_figure(fig, output_dir, "manufactoria_reward_compute_time", responsive)


def write_ngu_ablation_blog_figures(
    *,
    baseline_difficulty_plot_df: pd.DataFrame,
    age_difficulty_plot_df: pd.DataFrame,
    baseline_settings: list[str],
    age_settings: list[str],
    difficulty_order: list[str],
    difficulty_colors: dict[str, str],
    max_step: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write right-panel Plotly versions of the NGU baseline and max-age ablations."""

    baseline_summary = _summarize_difficulty(baseline_difficulty_plot_df)
    baseline_dashes = dict(zip(baseline_settings, ["dash", "dot", "solid"]))
    baseline_fig = go.Figure()
    for setting in baseline_settings:
        for difficulty in difficulty_order:
            subset = baseline_summary[
                (baseline_summary["setting"].astype(str) == setting)
                & (baseline_summary["metric"].astype(str) == difficulty)
            ]
            _add_summary_band_and_line(
                baseline_fig,
                subset,
                name=f"{difficulty} | {setting}",
                color=difficulty_colors[difficulty],
                dash=baseline_dashes[setting],
            )
    for setting in baseline_settings:
        baseline_fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=NGU_COLOR, width=3, dash=baseline_dashes[setting]),
            name=setting, legend="legend",
        ))
    for difficulty in difficulty_order:
        baseline_fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=difficulty_colors[difficulty], width=3),
            name=difficulty, legend="legend2",
        ))
    baseline_responsive = _finish_difficulty_figure(
        baseline_fig,
        title="GSM8K Platinum NGU baselines, by difficulty",
        max_step=max_step,
        first_legend_title="NGU Baseline",
        first_legend_y=0.50,
    )
    baseline_output = _write_figure(
        baseline_fig, output_dir, "gsm8k_per_difficulty_eval_ngu_baseline", baseline_responsive
    )

    age_summary = _summarize_difficulty(age_difficulty_plot_df)
    age_strengths = dict(zip(age_settings, [1.0, 0.75, 0.5, 0.35]))
    age_fig = go.Figure()
    for difficulty in difficulty_order:
        for setting in age_settings:
            subset = age_summary[
                (age_summary["setting"].astype(str) == setting)
                & (age_summary["metric"].astype(str) == difficulty)
            ]
            _add_summary_band_and_line(
                age_fig,
                subset,
                name=f"{difficulty} | {setting}",
                color=_blend_with_white(difficulty_colors[difficulty], age_strengths[setting]),
            )
    age_grays = dict(zip(age_settings, ["#525252", "#737373", "#969696", "#bdbdbd"]))
    for setting in age_settings:
        age_fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=age_grays[setting], width=3),
            name=setting, legend="legend",
        ))
    for difficulty in difficulty_order:
        age_fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=difficulty_colors[difficulty], width=3),
            name=difficulty, legend="legend2",
        ))
    age_responsive = _finish_difficulty_figure(
        age_fig,
        title="GSM8K Platinum NGU max age, by difficulty",
        max_step=max_step,
        first_legend_title="NGU Max Age",
        first_legend_y=0.43,
    )
    age_output = _write_figure(
        age_fig, output_dir, "gsm8k_per_difficulty_eval_ngu_max_age", age_responsive
    )
    return baseline_output, age_output


def write_ngu_blog_figures(
    *,
    ext_ratio_df: pd.DataFrame,
    ext_difficulty_plot_df: pd.DataFrame,
    grpo_settings: list[str],
    ngu_setting: str,
    difficulty_order: list[str],
    difficulty_colors: dict[str, str],
    max_step: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the NGU batch-composition and per-difficulty blog figures."""

    setting_order = [*grpo_settings, ngu_setting]
    setting_colors = dict(zip(grpo_settings, GRPO_GRAY_SHADES)) | {ngu_setting: NGU_COLOR}

    ratio_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Extra Hard Prompts", "Easy Prompts"],
        horizontal_spacing=0.07,
        shared_yaxes=True,
    )
    for column, y_column in enumerate(["q3_ratio", "q0_ratio"], start=1):
        for setting in setting_order:
            subset = ext_ratio_df[ext_ratio_df["setting"] == setting]
            subset = subset[subset["Step"] % 25 == 0]
            _add_band_and_line(
                ratio_fig,
                subset,
                y_column=y_column,
                name=setting,
                color=setting_colors[setting],
                spread="se",
                line_width=3.5 if setting == ngu_setting else 2,
                row=1,
                col=column,
            )

    for setting in reversed(setting_order):
        ratio_fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(
                    color=setting_colors[setting],
                    width=3.5 if setting == ngu_setting else 3,
                ),
                name=setting.removeprefix("NGU "),
                legend="legend" if setting == ngu_setting else "legend2",
            ),
            row=1,
            col=1,
        )

    ratio_fig.update_xaxes(
        title_text="Step",
        range=[0, max_step],
        dtick=400,
        minor=dict(dtick=200, showgrid=True),
    )
    ratio_fig.update_yaxes(rangemode="tozero")
    ratio_fig.update_yaxes(title_text="% of training batch", row=1, col=1)
    ratio_fig.update_annotations(font_size=BLOG_TITLE_FONT_SIZE)
    ratio_fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, size=BLOG_FONT_SIZE, color="#111111"),
        height=320,
        margin=dict(l=54, r=160, t=38, b=48),
        legend=dict(
            title_text="NGU",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
        legend2=dict(
            title_text="Standard GRPO",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.02,
            xanchor="left",
            y=0.70,
            yanchor="top",
        ),
    )
    ratio_responsive = {
        "breakpoint": 480,
        "wide": {
            "legend.orientation": "v",
            "legend.x": 1.02,
            "legend.xanchor": "left",
            "legend.y": 1.0,
            "legend.yanchor": "top",
            "legend2.orientation": "v",
            "legend2.x": 1.02,
            "legend2.xanchor": "left",
            "legend2.y": 0.70,
            "legend2.yanchor": "top",
            "margin.r": 160,
            "margin.b": 48,
            "height": 320,
        },
        "narrow": {
            "legend.orientation": "h",
            "legend.x": 0.5,
            "legend.xanchor": "center",
            "legend.y": -0.32,
            "legend.yanchor": "top",
            "legend2.orientation": "h",
            "legend2.x": 0.5,
            "legend2.xanchor": "center",
            "legend2.y": -0.54,
            "legend2.yanchor": "top",
            "xaxis.dtick": 800,
            "xaxis2.dtick": 800,
            "margin.r": 14,
            "margin.b": 190,
            "height": 430,
        },
    }
    ratio_output = _write_figure(
        ratio_fig,
        output_dir,
        "gsm8k_nonzero_prompt_ratios_with_ngu",
        ratio_responsive,
    )

    difficulty_fig = go.Figure()
    for difficulty in difficulty_order:
        for setting in setting_order:
            series = f"{difficulty} | {setting}"
            subset = ext_difficulty_plot_df[ext_difficulty_plot_df["series"] == series]
            color = difficulty_colors[difficulty] if setting == ngu_setting else setting_colors[setting]
            _add_band_and_line(
                difficulty_fig,
                subset,
                y_column="pass_at_1_percent",
                name=series,
                color=color,
                spread="sd",
                line_width=3 if setting == ngu_setting else 2,
            )

    for difficulty in difficulty_order:
        difficulty_fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=difficulty_colors[difficulty], width=3),
                name=difficulty,
                legend="legend",
            )
        )
    for setting in reversed(grpo_settings):
        difficulty_fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=setting_colors[setting], width=3),
                name=setting,
                legend="legend2",
            )
        )

    difficulty_fig.update_layout(
        title=dict(
            text="GSM8K Platinum with NGU, by difficulty",
            x=0.02,
            xanchor="left",
            font=dict(size=BLOG_TITLE_FONT_SIZE),
        ),
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, size=BLOG_FONT_SIZE, color="#111111"),
        autosize=True,
        height=DIFFICULTY_FIGURE_HEIGHT,
        margin=dict(l=58, r=DIFFICULTY_FIGURE_RIGHT_MARGIN, t=52, b=56),
        xaxis=dict(
            title_text="Step",
            range=[0, max_step],
            dtick=400,
            minor=dict(dtick=200, showgrid=True),
        ),
        yaxis=dict(title_text="pass@1 (%)", range=[0, 100], dtick=20),
        legend=dict(
            title_text="NGU",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.015,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
        legend2=dict(
            title_text="Standard GRPO",
            title_font=dict(size=BLOG_LEGEND_TITLE_FONT_SIZE),
            orientation="v",
            x=1.015,
            xanchor="left",
            y=0.50,
            yanchor="top",
        ),
    )
    difficulty_responsive = {
        "breakpoint": 480,
        "wide": {
            "legend.orientation": "v",
            "legend.x": 1.015,
            "legend.xanchor": "left",
            "legend.y": 1.0,
            "legend.yanchor": "top",
            "legend2.orientation": "v",
            "legend2.x": 1.015,
            "legend2.xanchor": "left",
            "legend2.y": 0.50,
            "legend2.yanchor": "top",
            "margin.r": DIFFICULTY_FIGURE_RIGHT_MARGIN,
            "margin.b": 56,
            "height": DIFFICULTY_FIGURE_HEIGHT,
        },
        "narrow": {
            "legend.orientation": "h",
            "legend.x": 0.5,
            "legend.xanchor": "center",
            "legend.y": -0.28,
            "legend.yanchor": "top",
            "legend2.orientation": "h",
            "legend2.x": 0.5,
            "legend2.xanchor": "center",
            "legend2.y": -0.55,
            "legend2.yanchor": "top",
            "margin.r": 16,
            "margin.b": 230,
            "height": DIFFICULTY_FIGURE_NARROW_HEIGHT,
        },
    }
    difficulty_output = _write_figure(
        difficulty_fig,
        output_dir,
        "gsm8k_per_difficulty_eval_with_ngu",
        difficulty_responsive,
    )
    return ratio_output, difficulty_output
