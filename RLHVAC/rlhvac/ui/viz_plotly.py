from __future__ import annotations
import math
import plotly.graph_objects as go
from rlhvac.spec import SceneSchema


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n)) if n else 1
    rows = math.ceil(n / cols) if cols else 1
    return rows, cols


def _unit_names_from_frame(frame: dict) -> list[str]:
    return list((frame or {}).get("scene", {}).keys())


def _hover_for_unit(schema: SceneSchema, unit_name: str, values: dict) -> str:
    lines = [f"<b>{unit_name}</b>"]
    for var_name, val in values.items():
        meta = schema.variable_meta(var_name)
        label = meta.label if meta else var_name
        unit = f" {meta.unit}" if meta and meta.unit else ""
        shown = "n/a" if val is None else (f"{val:.3g}" if isinstance(val, (int, float)) else val)
        lines.append(f"{label}: {shown}{unit}")
    return "<br>".join(lines)


def heatmap_figure(schema: SceneSchema, frame: dict) -> go.Figure:
    scene = (frame or {}).get("scene", {})
    names = _unit_names_from_frame(frame) or [u.name for u in schema.units]
    rows, cols = _grid_shape(len(names))
    z, text, labels = [], [], []
    for r in range(rows):
        zr, tr, lr = [], [], []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(names):
                name = names[idx]
                vals = scene.get(name, {})
                cval = vals.get(schema.color_by)
                zr.append(cval if isinstance(cval, (int, float)) else None)
                tr.append(_hover_for_unit(schema, name, vals))
                lr.append(f"{name}<br>{'' if not isinstance(cval, (int, float)) else f'{cval:.3g}'}")
            else:
                zr.append(None); tr.append(""); lr.append("")
        z.append(zr); text.append(tr); labels.append(lr)
    fig = go.Figure(go.Heatmap(z=z, text=text, hoverinfo="text",
                               zmin=schema.color_range[0], zmax=schema.color_range[1],
                               colorscale="RdYlBu_r", showscale=True))
    fig.update_traces(texttemplate="%{customdata}", customdata=labels)
    fig.update_layout(yaxis=dict(autorange="reversed", showticklabels=False),
                      xaxis=dict(showticklabels=False),
                      margin=dict(l=10, r=10, t=30, b=10),
                      title=f"Colored by {schema.color_by}")
    return fig


def variable_timeseries(frames: list[dict], schema: SceneSchema, var: str) -> go.Figure:
    steps = [f.get("step") for f in frames]
    names: list[str] = []
    for f in frames:
        for n in f.get("scene", {}):
            if n not in names:
                names.append(n)
    fig = go.Figure()
    for name in names:
        ys = [f.get("scene", {}).get(name, {}).get(var) for f in frames]
        if any(y is not None for y in ys):
            fig.add_trace(go.Scatter(x=steps, y=ys, mode="lines", name=name))
    fig.update_layout(title=var, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def reward_timeseries(frames: list[dict]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[f.get("step") for f in frames],
                               y=[f.get("reward") for f in frames], mode="lines", name="reward"))
    fig.update_layout(title="Reward", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def episode_bar_figure(summary: dict) -> go.Figure:
    items = [(k, v) for k, v in (summary or {}).items()
             if isinstance(v, (int, float)) and math.isfinite(v)]
    fig = go.Figure(go.Bar(x=[k for k, _ in items], y=[v for _, v in items]))
    fig.update_layout(title="Episode metrics", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def rollup_curve_figure(rollup: list[dict], metric: str = "total_reward") -> go.Figure:
    rows = [r for r in rollup
            if metric in r and isinstance(r[metric], (int, float)) and math.isfinite(r[metric])]
    fig = go.Figure(go.Scatter(x=[r.get("episode") for r in rows],
                               y=[r.get(metric) for r in rows], mode="lines+markers", name=metric))
    fig.update_layout(title=f"{metric} per episode", xaxis_title="episode",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig
