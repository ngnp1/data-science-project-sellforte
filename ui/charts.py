"""Plotly figures for the findings viewer.

Colours follow the dataviz reference palette: event types and channels each
take categorical slots in a fixed order, so a colour always names the same
thing whatever the filters show. Neutral ink, not a series colour, marks this
market, its comparison markets and the shaded windows, so a channel's hue is
never mistaken for a finding's. Every function takes `dark` to pick the steps
validated for a dark surface; Streamlit's own theme handles the chrome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.score import subject_channels
from ui.labels import SCORE_NAMES, TYPES, event_id, title
from ui.readout import Readout

SLOTS_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948")
SLOTS_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500",
              "#d55181", "#008300", "#9085e9", "#e66767")
INK = {False: "#0b0b0b", True: "#f0efec"}
MUTED = "#898781"
SHADE = {False: "#52514e", True: "#c3c2b7"}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
DAY = pd.Timedelta(days=1)
CARRYOVER_DAYS = (-14, 28)


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8),
                      font=dict(family=FONT), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0, title_text=""))
    return fig


def _windows(event: DetectedEvent) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """A pulse's separate off-windows; other types' components are per-channel
    detail inside one window, so they draw as that one window."""
    if event.event_type == "channel_pulse" and event.components:
        return event.components
    return ((event.start, event.end),)


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous True stretches of a daily mask, inclusive."""
    values = mask.to_numpy()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], values.astype(int), [0]])))
    return [(mask.index[a], mask.index[b - 1]) for a, b in zip(edges[::2], edges[1::2])]


def _fmt(day: pd.Timestamp) -> str:
    return f"{day.day} {day:%b %Y}"


def period_map(events: list[DetectedEvent], markets: list[str], selected: str | None,
               period: tuple[pd.Timestamp, pd.Timestamp] | None = None,
               dark: bool = False) -> go.Figure:
    """One row per market, one bar per finding window, coloured by type."""
    fig = go.Figure()
    for kind, info in TYPES.items():
        bars = [(e, a, b) for e in events if e.event_type == kind for a, b in _windows(e)]
        if not bars:
            continue
        ids = [event_id(e) for e, _, _ in bars]
        fig.add_trace(go.Bar(
            orientation="h", name=info.name,
            y=[e.country_code for e, _, _ in bars],
            base=[a for _, a, _ in bars],
            x=[(b + DAY - a) / pd.Timedelta(milliseconds=1) for _, a, b in bars],
            customdata=[[i, title(e), f"{_fmt(e.start)} – {_fmt(e.end)} · {e.n_days} days"]
                        for i, (e, _, _) in zip(ids, bars)],
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]}<extra></extra>",
            marker=dict(color=info.color_dark if dark else info.color,
                        line=dict(color=INK[dark],
                                  width=[2.5 if i == selected else 0 for i in ids])),
            width=0.6))
    fig.update_yaxes(categoryorder="array", categoryarray=markets[::-1],
                     type="category", fixedrange=True)
    fig.update_xaxes(type="date", range=list(period) if period else None)
    fig.update_layout(barmode="overlay", clickmode="event+select", showlegend=True)
    return _base_layout(fig, 90 + 34 * len(markets))


def channel_colors(panel: Panel, dark: bool = False) -> dict[str, str]:
    """Largest channels across the whole panel take the slots in order; past
    the seventh they fold into 'Other' rather than cycling a hue."""
    slots = SLOTS_DARK if dark else SLOTS_LIGHT
    totals = panel.spend.T.groupby(level=1).sum().sum(axis=1).sort_values(ascending=False)
    names = list(totals.index)
    if len(names) <= len(slots):
        return dict(zip(names, slots))
    colors = dict(zip(names[:len(slots) - 1], slots))
    colors.update({ch: MUTED for ch in names[len(slots) - 1:]})
    return colors


def _sales_index(series: pd.Series, reference: pd.Series) -> pd.Series:
    smooth = series.rolling(7, center=True, min_periods=4).mean()
    return smooth / series[reference].mean() * 100


def event_chart(event: DetectedEvent, panel: Panel, rd: Readout | None,
                dark: bool = False) -> go.Figure:
    country = event.country_code
    has_sales = country in panel.sales.columns and bool(panel.sales[country].notna().any())
    rows = 2 if has_sales else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        row_heights=[0.55, 0.45] if has_sales else None,
        subplot_titles=("Daily spend by channel",
                        "Sales, 7-day average, reference window = 100")[:rows])

    colors = channel_colors(panel, dark)
    # The finding's own channel must stay visible to be checked by eye: when
    # it would fold into "Other", it takes the one slot folding leaves free.
    subjects = subject_channels(event, panel)
    if len(subjects) == 1 and colors.get(subjects[0]) == MUTED:
        colors[subjects[0]] = (SLOTS_DARK if dark else SLOTS_LIGHT)[-1]
    channels = sorted(panel.channels_in(country),
                      key=lambda ch: list(colors).index(ch))
    named = [ch for ch in channels if colors[ch] != MUTED]
    other = [ch for ch in channels if colors[ch] == MUTED]
    stack = [(ch, panel.series(country, ch), colors[ch]) for ch in named]
    if other:
        stack.append(("Other", sum(panel.series(country, ch) for ch in other), MUTED))
    for name, series, color in stack:
        fig.add_trace(go.Scatter(
            x=panel.dates, y=series, name=name, stackgroup="spend", mode="lines",
            line=dict(width=1.5, color=color), opacity=0.9,
            hovertemplate=f"{name}: %{{y:,.0f}}<extra></extra>"), row=1, col=1)
    # The stack draws a missing row as zero spend, which is exactly the
    # confusion the panel's present mask exists to prevent, so mark those days.
    if channels:
        present = pd.concat([panel.present_mask(country, ch) for ch in channels],
                            axis=1).all(axis=1)
        gap_days = panel.dates[~present.to_numpy()]
        if len(gap_days):
            fig.add_trace(go.Scatter(
                x=gap_days, y=[0.0] * len(gap_days), name="Missing data rows",
                mode="markers", marker=dict(symbol="line-ns-open", size=10,
                                            line=dict(width=1.5, color=MUTED)),
                hovertemplate="No data rows on %{x|%d %b %Y}<extra></extra>"),
                row=1, col=1)

    if has_sales:
        reference = rd.reference if rd is not None else pd.Series(True, index=panel.dates)
        fig.add_trace(go.Scatter(
            x=panel.dates, y=_sales_index(panel.sales[country], reference),
            name="This market", mode="lines", line=dict(width=2, color=INK[dark]),
            hovertemplate="This market: %{y:.0f}<extra></extra>"), row=2, col=1)
        peers = rd.comparison_markets if rd is not None else ()
        if peers:
            average = pd.concat([_sales_index(panel.sales[p], reference) for p in peers],
                                axis=1).mean(axis=1)
            fig.add_trace(go.Scatter(
                x=panel.dates, y=average, name="Comparison markets (average)",
                mode="lines", line=dict(width=2, color=MUTED, dash="dash"),
                hovertemplate="Comparison markets: %{y:.0f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=100, line=dict(width=1, color=MUTED), row=2, col=1)
        fig.update_yaxes(title_text="Index", row=2, col=1)

    during = _runs(rd.during) if rd is not None else list(_windows(event))
    reference_runs = _runs(rd.reference) if rd is not None else []
    for runs, opacity, label in ((during, 0.18, "Finding"), (reference_runs, 0.07, "Reference")):
        for i, (a, b) in enumerate(runs):
            fig.add_vrect(x0=a, x1=b + DAY, fillcolor=SHADE[dark], opacity=opacity,
                          line_width=0, layer="below", row="all", col=1)
            if i == 0:
                fig.add_annotation(x=a, y=1.0, xref="x", yref="paper", text=label,
                                   showarrow=False, xanchor="left", yanchor="top",
                                   xshift=4, font=dict(size=11, color=MUTED))

    pad = pd.Timedelta(days=max(event.n_days, 60))
    lo = max(panel.dates[0], event.start - pad)
    hi = min(panel.dates[-1], event.end + pad)
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(title_text="Spend", row=1, col=1)
    fig.update_layout(hovermode="x unified")
    _base_layout(fig, 520 if has_sales else 320)
    # Many channels make a long legend: below the plot it cannot cover the
    # subplot titles or the window labels.
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.08,
                                  xanchor="left", x=0))
    fig.for_each_annotation(lambda a: a.update(x=0, xanchor="left")
                            if a.text in ("Daily spend by channel",
                                          "Sales, 7-day average, reference window = 100")
                            else None)
    return fig


def carryover_chart(event: DetectedEvent, panel: Panel, rd: Readout | None,
                    dark: bool = False) -> go.Figure | None:
    """Sales around each moment spend stopped, averaged over a pulse's off-windows."""
    country = event.country_code
    if (event.event_type != "channel_pulse" or len(event.components) < 2 or rd is None
            or country not in panel.sales.columns or not panel.sales[country].notna().any()):
        return None
    offsets = np.arange(CARRYOVER_DAYS[0], CARRYOVER_DAYS[1] + 1)

    def curve(market: str) -> np.ndarray:
        # A light 3-day average: daily noise otherwise hides the shape of the decay.
        values = (panel.sales[market].rolling(3, center=True, min_periods=1).mean()
                  .to_numpy(dtype=float))
        rows = []
        for start, _ in event.components:
            i = panel.dates.get_loc(start)
            before = values[max(0, i + CARRYOVER_DAYS[0]):i]
            if not len(before) or np.isnan(before).all():
                continue
            base = np.nanmean(before)
            rows.append([values[i + o] / base * 100 if 0 <= i + o < len(values) else np.nan
                         for o in offsets])
        return np.nanmean(rows, axis=0) if rows else np.full(len(offsets), np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=offsets, y=curve(country), name="This market", mode="lines",
                             line=dict(width=2, color=INK[dark]),
                             hovertemplate="Day %{x}: %{y:.0f}<extra></extra>"))
    if rd.comparison_markets:
        peers = np.nanmean([curve(p) for p in rd.comparison_markets], axis=0)
        fig.add_trace(go.Scatter(x=offsets, y=peers, name="Comparison markets (average)",
                                 mode="lines", line=dict(width=2, color=MUTED, dash="dash"),
                                 hovertemplate="Day %{x}: %{y:.0f}<extra></extra>"))
    off_days = float(np.median([(b - a).days + 1 for a, b in event.components]))
    fig.add_vrect(x0=0, x1=off_days, fillcolor=SHADE[dark], opacity=0.18,
                  line_width=0, layer="below")
    fig.add_annotation(x=0, y=1.0, yref="paper", text="Spend off", showarrow=False,
                       xanchor="left", yanchor="top", xshift=4,
                       font=dict(size=11, color=MUTED))
    fig.add_hline(y=100, line=dict(width=1, color=MUTED))
    fig.update_xaxes(title_text="Days since spend stopped")
    fig.update_yaxes(title_text="3-day avg (2 weeks before = 100)")
    fig.update_layout(hovermode="x unified")
    _base_layout(fig, 340)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.25,
                                  xanchor="left", x=0))
    return fig


def score_bars(parts: dict[str, float], dark: bool = False) -> go.Figure:
    names = [SCORE_NAMES.get(k, (k, ""))[0] for k in parts]
    meanings = [SCORE_NAMES.get(k, (k, ""))[1] for k in parts]
    fig = go.Figure(go.Bar(
        orientation="h", y=names, x=list(parts.values()), customdata=meanings,
        marker=dict(color=(SLOTS_DARK if dark else SLOTS_LIGHT)[0]), width=0.6,
        text=[f"{v:.2f}" for v in parts.values()], textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.2f}<br>%{customdata}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], fixedrange=True, showticklabels=False)
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    fig.update_layout(showlegend=False)
    return _base_layout(fig, 50 + 30 * len(parts))
