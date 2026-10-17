"""Plotly chart helpers in one visual system: validated categorical slots (fixed order, never
cycled), thin 2px lines, recessive grid, hover on everything, direct end labels where the light
palette is under 3:1 contrast."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Categorical slots, light / dark steps (validated: adjacent CVD and normal-vision floors pass)
SLOTS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
CHROME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", band="rgba(42,120,214,0.07)"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", band="rgba(57,135,229,0.10)"),
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def slots() -> list[str]:
    return SLOTS[mode()]


def chrome() -> dict:
    return CHROME[mode()]


def base_layout(fig: go.Figure, title: str | None = None, height: int = 300, y_title: str = "",
                x_title: str = "", legend: bool = True) -> go.Figure:
    c = chrome()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=c["ink"]), x=0, xanchor="left") if title else None,
        height=height, margin=dict(l=12, r=72, t=48 if title else 16, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=c["ink2"]),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(color=c["ink2"])),
        hoverlabel=dict(bgcolor=c["surface"], font=dict(color=c["ink"])),
    )
    axis = dict(gridcolor=c["grid"], gridwidth=1, zeroline=False, linecolor=c["axis"], showline=True, automargin=True,
                tickfont=dict(color=c["muted"]), title_font=dict(color=c["ink2"], size=12))
    fig.update_xaxes(**axis, title_text=x_title, showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor=c["axis"], spikedash="solid")
    fig.update_yaxes(**axis, title_text=y_title)
    return fig


def end_labels(fig: go.Figure, x_last, ys: list[float], names: list[str], colors: list[str],
               axis_span: float | None = None) -> None:
    """Direct labels at line ends, nudged apart so they don't collide. Spacing is a share of the
    axis range, so lines that end on the same value still get readable labels."""
    ys = np.asarray(ys, float)
    order = np.argsort(ys)
    span = axis_span or (np.nanmax(ys) - np.nanmin(ys)) or 1.0
    gap, placed = span * 0.08, {}
    last = -np.inf
    for i in order:
        y = ys[i] if np.isfinite(ys[i]) else last + gap
        y = max(y, last + gap)
        placed[i], last = y, y
    for i in range(len(names)):
        fig.add_annotation(x=x_last, y=placed[i], text=names[i], showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=chrome()["ink2"]), bgcolor="rgba(0,0,0,0)")
        fig.add_annotation(x=x_last, y=placed[i], text="●", showarrow=False, xanchor="left", xshift=-2,
                           font=dict(size=9, color=colors[i]))


def engine_lines(df: pd.DataFrame, fam: str, x: str, unit: str, title: str, x_title: str,
                 shade: pd.Series | None = None, flags: pd.DataFrame | None = None, height: int = 260) -> go.Figure:
    """Four engines of one family over time, with optional cruise shading and dropped-sample rug."""
    colors = slots()[:4]
    fig = go.Figure()
    names = [f"Eng {k}" for k in range(1, 5)]
    for k, (name, col) in enumerate(zip(names, colors), start=1):
        c = f"{fam}_{k}"
        if c not in df:
            continue
        fig.add_trace(go.Scattergl(x=df[x], y=df[c], name=name, mode="lines",
                                   line=dict(width=1.6, color=col),
                                   hovertemplate=f"{name}: %{{y:.4g}} {unit}<extra></extra>"))
    if shade is not None:
        _shade(fig, df[x], shade)
    if flags is not None and len(flags):
        _rug(fig, flags, x_scale=1 / 60 if x == "minutes" else 1)
    base_layout(fig, title, height, unit, x_title)
    last = df.dropna(subset=[f"{fam}_1"]).iloc[-1] if df[f"{fam}_1"].notna().any() else None
    if last is not None:
        vals = df[[f"{fam}_{k}" for k in range(1, 5) if f"{fam}_{k}" in df]].to_numpy(float)
        span = float(np.nanmax(vals) - np.nanmin(vals)) if np.isfinite(vals).any() else None
        end_labels(fig, last[x], [last.get(f"{fam}_{k}", np.nan) for k in range(1, 5)], names, colors, span)
    return fig


def _shade(fig: go.Figure, xs: pd.Series, mask: pd.Series) -> None:
    m = mask.to_numpy(bool)
    if not m.any():
        return
    edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    xv = xs.to_numpy()
    for s, e in zip(edges[::2], edges[1::2]):
        fig.add_vrect(x0=xv[s], x1=xv[e - 1], fillcolor=chrome()["band"], line_width=0, layer="below")


def _rug(fig: go.Figure, flags: pd.DataFrame, x_scale: float) -> None:
    """Small ticks along the bottom where samples were dropped as bad, with the reason on hover."""
    fig.add_trace(go.Scatter(
        x=flags.t_start_s * x_scale, y=[0] * len(flags), yaxis="y2", mode="markers", name="dropped samples",
        marker=dict(symbol="line-ns-open", size=10, color=chrome()["muted"], line=dict(width=1.5)),
        customdata=np.stack([flags.param, flags.check, flags.detail], axis=1),
        hovertemplate="dropped: %{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(yaxis2=dict(overlaying="y", range=[0, 20], visible=False, fixedrange=True))


def phase_bounds(fig: go.Figure, phases: pd.DataFrame, x_scale: float = 1 / 60) -> None:
    """Thin vertical rules at phase changes, labelled at the top (phases carry no color)."""
    c = chrome()
    last_label = -np.inf
    for r in phases.itertuples():
        x = r.t_start_s * x_scale
        fig.add_vline(x=x, line_width=1, line_color=c["axis"], line_dash="dot")
        if x - last_label < 4:  # minutes; skip labels that would overlap the previous one
            continue
        last_label = x
        fig.add_annotation(x=x, y=1, yref="paper", text=r.phase.replace("_", " "), showarrow=False,
                           xanchor="left", yanchor="bottom", textangle=-30, font=dict(size=10, color=c["muted"]))
