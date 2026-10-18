"""Plotly figures for the findings viewer.

The figures are drawn the way an analyst's report would draw them: thin
marks, hairline grids, direct labels and numbers on the chart instead of
colour alone. Event types keep their categorical palette slots on the period
map, so a colour always names the same pattern. In the spend and sales
chart, colour carries one meaning only -- the finding's own channels in the
accent, everything else in grey -- and each channel gets its own labelled
strip, so "this channel stopped while the others ran" is read straight off
the chart. Every function takes `dark` to pick steps validated for a dark
surface; Streamlit's own theme handles the chrome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.model import subject_channels
from ui.labels import TYPES, event_id, short_label, title
from ui.readout import Readout

INK = {False: "#18181b", True: "#fafafa"}
MUTED = {False: "#a1a1aa", True: "#71717a"}
GRID = {False: "#e4e4e7", True: "#27272a"}
FAINT = {False: "#f4f4f5", True: "#18181b"}
ACCENT = {False: "#2a78d6", True: "#3987e5"}
FONT = "Inter, system-ui, sans-serif"
DAY = pd.Timedelta(days=1)
CARRYOVER_DAYS = (-14, 28)
MAX_STRIPS = 8


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _base_layout(fig: go.Figure, height: int, dark: bool = False) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8),
                      font=dict(family=FONT, size=12), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="left", x=0, title_text="", font=dict(size=11)))
    fig.update_xaxes(showgrid=False, linecolor=GRID[dark], ticks="outside",
                     tickcolor=GRID[dark], tickfont=dict(size=11, color=MUTED[dark]))
    fig.update_yaxes(gridcolor=GRID[dark], gridwidth=1, zeroline=False,
                     tickfont=dict(size=11, color=MUTED[dark]))
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
    """One row per market: a faint track for the data span, a thin bar per
    finding window coloured by type, and a direct label naming each finding."""
    fig = go.Figure()
    if period is not None:
        lo, hi = period
        fig.add_trace(go.Bar(
            orientation="h", y=markets, base=[lo] * len(markets),
            x=[(hi + DAY - lo) / pd.Timedelta(milliseconds=1)] * len(markets),
            marker=dict(color=FAINT[dark], line=dict(width=0)), width=0.34,
            hoverinfo="skip", showlegend=False))
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
                                  width=[2 if i == selected else 0 for i in ids])),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
            width=0.34))
    for e in events:
        chosen = event_id(e) == selected
        fig.add_annotation(x=e.start, y=e.country_code, xanchor="left", yanchor="bottom",
                           yshift=7, showarrow=False,
                           text=f"<b>{short_label(e)}</b>" if chosen else short_label(e),
                           font=dict(size=11, color=INK[dark] if chosen else MUTED[dark]))
    _base_layout(fig, 70 + 46 * len(markets), dark)
    fig.update_yaxes(categoryorder="array", categoryarray=markets[::-1],
                     type="category", fixedrange=True, showgrid=False)
    fig.update_xaxes(type="date", range=list(period) if period else None, dtick="M3",
                     tickformat="%b %Y", showgrid=True, gridcolor=GRID[dark])
    fig.update_layout(barmode="overlay", clickmode="event+select", showlegend=True)
    return fig


def _strips(event: DetectedEvent, panel: Panel
            ) -> list[tuple[str, list[str], bool]]:
    """(label, channels summed into the strip, highlighted) per spend strip.

    Largest channels first; the finding's own channels always get a strip;
    past MAX_STRIPS the smallest of the rest fold into one "Other channels"
    strip instead of the chart growing without bound."""
    country = event.country_code
    subjects = set(subject_channels(event, panel))
    channels = sorted(panel.channels_in(country),
                      key=lambda ch: -float(panel.series(country, ch).sum()))
    if len(channels) <= MAX_STRIPS:
        return [(ch, [ch], ch in subjects) for ch in channels]
    keep = [ch for ch in channels if ch in subjects][:MAX_STRIPS - 1]
    keep += [ch for ch in channels if ch not in keep][:MAX_STRIPS - 1 - len(keep)]
    rest = [ch for ch in channels if ch not in keep]
    strips = [(ch, [ch], ch in subjects) for ch in channels if ch in keep]
    strips.append(("Other channels", rest, all(ch in subjects for ch in rest)))
    return strips


def _sales_index(series: pd.Series, reference: pd.Series) -> pd.Series:
    smooth = series.rolling(7, center=True, min_periods=4).mean()
    return smooth / series[reference].mean() * 100


def event_chart(event: DetectedEvent, panel: Panel, rd: Readout | None,
                dark: bool = False) -> go.Figure:
    """Sales against comparison markets on top, with the reference and finding
    averages drawn as labelled steps; below, one weekly spend strip per
    channel with the finding's own channels in the accent."""
    country = event.country_code
    has_sales = country in panel.sales.columns and bool(panel.sales[country].notna().any())
    strips = _strips(event, panel)
    top = 1 if has_sales else 0
    rows = top + len(strips)
    heights = ([0.42] + [0.58 / len(strips)] * len(strips)) if has_sales else None
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.012,
                        row_heights=heights)

    if has_sales:
        sales = panel.sales[country]
        reference = rd.reference if rd is not None else pd.Series(True, index=panel.dates)
        fig.add_trace(go.Scatter(
            x=panel.dates, y=_sales_index(sales, reference), name="This market",
            mode="lines", line=dict(width=1.6, color=INK[dark]),
            hovertemplate="This market %{y:.0f}<extra></extra>"), row=1, col=1)
        peers = rd.comparison_markets if rd is not None else ()
        if peers:
            average = pd.concat([_sales_index(panel.sales[p], reference) for p in peers],
                                axis=1).mean(axis=1)
            fig.add_trace(go.Scatter(
                x=panel.dates, y=average, name="Comparison markets (average)",
                mode="lines", line=dict(width=1.2, color=MUTED[dark], dash="dot"),
                hovertemplate="Comparison markets %{y:.0f}<extra></extra>"), row=1, col=1)
        if rd is not None:
            # The readout's two averages, drawn where they were measured.
            base = sales[reference].mean()
            for mask, label in ((rd.reference, "Reference"), (rd.during, "Finding")):
                level = sales[mask].mean() / base * 100
                for a, b in _runs(mask):
                    fig.add_trace(go.Scatter(
                        x=[a, b + DAY], y=[level, level], mode="lines", showlegend=False,
                        line=dict(width=2.5, color=ACCENT[dark]), hoverinfo="skip"),
                        row=1, col=1)
                fig.add_annotation(x=_runs(mask)[0][0], y=level, xanchor="left",
                                   yanchor="bottom", yshift=3, showarrow=False,
                                   text=f"{label} avg {level:.0f}",
                                   font=dict(size=11, color=ACCENT[dark]), row=1, col=1)
        fig.update_yaxes(title_text="Sales index (reference = 100)" if rd is not None
                         else "Sales index (mean = 100)",
                         title_font=dict(size=11, color=MUTED[dark]), row=1, col=1)

    missing_shown = False
    for row, (label, channels, highlighted) in enumerate(strips, start=top + 1):
        color = ACCENT[dark] if highlighted else MUTED[dark]
        daily = sum(panel.series(country, ch) for ch in channels)
        weekly = daily.resample("W-MON", label="left", closed="left").mean()
        fig.add_trace(go.Scatter(
            x=weekly.index, y=weekly.values, name=label, mode="lines",
            line=dict(width=1.3, color=color, shape="hv"), fill="tozeroy",
            fillcolor=_rgba(color, 0.14), showlegend=False,
            hovertemplate=f"{label} %{{y:,.0f}}/day (weekly avg)<extra></extra>"),
            row=row, col=1)
        # A missing row would otherwise draw as zero spend: tick those days.
        present = pd.concat([panel.present_mask(country, ch) for ch in channels],
                            axis=1).all(axis=1)
        gap_days = panel.dates[~present.to_numpy()]
        if len(gap_days):
            fig.add_trace(go.Scatter(
                x=gap_days, y=[0.0] * len(gap_days), name="Missing data rows",
                mode="markers", legendgroup="missing", showlegend=not missing_shown,
                marker=dict(symbol="line-ns-open", size=9,
                            line=dict(width=1.5, color=INK[dark])),
                hovertemplate="No data rows on %{x|%d %b %Y}<extra></extra>"),
                row=row, col=1)
            missing_shown = True
        peak = float(weekly.max()) if len(weekly) else 0.0
        fig.update_yaxes(range=[0, peak * 1.25 or 1.0], showticklabels=False,
                         showgrid=False, fixedrange=True, row=row, col=1)
        fig.add_annotation(xref="x domain", yref="y domain", x=0.005, y=0.95,
                           xanchor="left", yanchor="top", showarrow=False, row=row, col=1,
                           text=f"<b>{label}</b>" if highlighted else label,
                           font=dict(size=11, color=INK[dark] if highlighted else MUTED[dark]))

    during = _runs(rd.during) if rd is not None else list(_windows(event))
    reference_runs = _runs(rd.reference) if rd is not None else []
    for runs, colour in ((during, ACCENT[dark]), (reference_runs, MUTED[dark])):
        for a, b in runs:
            fig.add_vrect(x0=a, x1=b + DAY, fillcolor=colour, opacity=0.07,
                          line_width=0, layer="below", row="all", col=1)
    for x in (event.start, event.end + DAY):
        fig.add_vline(x=x, line=dict(width=1, color=ACCENT[dark], dash="dot"))

    pad = pd.Timedelta(days=max(event.n_days, 60))
    lo = max(panel.dates[0], event.start - pad)
    hi = min(panel.dates[-1], event.end + pad)
    _base_layout(fig, (260 if has_sales else 40) + 40 * len(strips), dark)
    fig.update_xaxes(range=[lo, hi])
    fig.update_layout(hovermode="x unified")
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
                             line=dict(width=1.6, color=INK[dark]),
                             hovertemplate="Day %{x}: %{y:.0f}<extra></extra>"))
    if rd.comparison_markets:
        peers = np.nanmean([curve(p) for p in rd.comparison_markets], axis=0)
        fig.add_trace(go.Scatter(x=offsets, y=peers, name="Comparison markets (average)",
                                 mode="lines", line=dict(width=1.2, color=MUTED[dark], dash="dot"),
                                 hovertemplate="Day %{x}: %{y:.0f}<extra></extra>"))
    off_days = float(np.median([(b - a).days + 1 for a, b in event.components]))
    fig.add_vrect(x0=0, x1=off_days, fillcolor=ACCENT[dark], opacity=0.07,
                  line_width=0, layer="below")
    fig.add_annotation(x=0, y=1.0, yref="paper", text="Spend off", showarrow=False,
                       xanchor="left", yanchor="top", xshift=4,
                       font=dict(size=11, color=ACCENT[dark]))
    fig.add_hline(y=100, line=dict(width=1, color=GRID[dark]))
    fig.update_xaxes(title_text="Days since spend stopped")
    fig.update_yaxes(title_text="3-day avg (2 weeks before = 100)")
    fig.update_layout(hovermode="x unified")
    _base_layout(fig, 340, dark)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.25,
                                  xanchor="left", x=0))
    return fig
