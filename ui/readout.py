"""Rough impact readings for one detected event.

The detector finds periods; it deliberately does not estimate what they did to
sales. The viewer still has to answer the brief's "what was it worth" question
for a reader who is not a statistician, so this module produces a small set of
descriptive numbers and is explicit about how far each can be trusted:

* The market's sales during the event are compared with a reference window
  (the weeks just before it, the weeks just after it when the data starts too
  late, or a pulse's own on-days).
* The same ratio in comparison markets -- markets with sales data and no
  finding of their own across those weeks -- is averaged with equal weight per
  market, so a large market cannot drown a small one, and divided out. What is
  left is the change that the other markets did not share.
* The same calculation repeated at every other weekly offset in the series
  gives the size of an ordinary swing. A change smaller than twice that is
  reported as not readable, and no rough return is computed from it.

None of this is a causal estimate; the viewer labels it as a rough reading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.model import subject_channels

REFERENCE_MAX_DAYS = 90
REFERENCE_MIN_DAYS = 7
PLACEBO_STEP_DAYS = 7
MIN_PLACEBOS = 5
READABLE_MULTIPLE = 2.0
MIN_SPEND_CHANGE = 0.10
OTHER_SPEND_SHARE = 0.5


@dataclass(frozen=True)
class Readout:
    during: pd.Series
    reference: pd.Series
    reference_side: str
    subject_channels: tuple[str, ...]
    spend_reference: float
    spend_during: float
    comparison_markets: tuple[str, ...]
    sales_reference: float | None
    sales_during: float | None
    effect: float | None
    swing: float | None
    readable: bool
    sales_difference: float | None
    rough_return: float | None
    baseline: float | None
    note: str


def _window(dates: pd.DatetimeIndex, start: pd.Timestamp,
            end: pd.Timestamp) -> pd.Series:
    return pd.Series((dates >= start) & (dates <= end), index=dates)


def _other_findings(event: DetectedEvent, panel: Panel,
                    events: list[DetectedEvent]) -> pd.Series:
    """Days covered by any other finding in the same market."""
    blocked = pd.Series(False, index=panel.dates)
    for other in events:
        if other.country_code == event.country_code and other != event:
            blocked |= _window(panel.dates, other.start, other.end)
    return blocked


def masks(event: DetectedEvent, panel: Panel, events: list[DetectedEvent] = ()
          ) -> tuple[pd.Series, pd.Series, str, bool] | None:
    """(during, reference, side, trimmed) as boolean masks on panel.dates, or
    None when no reference window of at least REFERENCE_MIN_DAYS exists.

    A reference window that runs into another finding in the same market
    would compare the event with a second unusual period -- a step change
    read against the pause just before it, say. So a clean before-window is
    preferred, then a clean after-window; only when neither is clean are the
    other finding's days cut out of the longer side, and `trimmed` says so.
    """
    dates = panel.dates
    blocked = _other_findings(event, panel, events)
    span = _window(dates, event.start, event.end)
    if event.event_type == "channel_pulse" and event.components:
        during = pd.Series(False, index=dates)
        for start, end in event.components:
            during |= _window(dates, start, end)
        on_days = span & ~during
        if (on_days & ~blocked).sum() >= REFERENCE_MIN_DAYS:
            return during, on_days & ~blocked, "on-days", bool((on_days & blocked).any())
    else:
        during = span

    length = min(event.n_days, REFERENCE_MAX_DAYS)
    first = int(np.flatnonzero(span.values)[0])
    last = int(np.flatnonzero(span.values)[-1])
    positions = np.arange(len(dates))
    sides = [
        ("before", pd.Series((positions >= first - length) & (positions < first), index=dates)),
        ("after", pd.Series((positions > last) & (positions <= last + length), index=dates)),
    ]
    for side, window in sides:
        if window.sum() >= REFERENCE_MIN_DAYS and not (window & blocked).any():
            return during, window, side, False
    side, window = max(sides, key=lambda s: int((s[1] & ~blocked).sum()))
    if (window & ~blocked).sum() >= REFERENCE_MIN_DAYS:
        return during, window & ~blocked, side, True
    return None


def _has_sales(panel: Panel, country: str) -> bool:
    return country in panel.sales.columns and bool(panel.sales[country].notna().any())


def comparison_markets(event: DetectedEvent, panel: Panel,
                       events: list[DetectedEvent], span: pd.Series) -> list[str]:
    """Other markets with sales data and no finding of their own in `span`."""
    days = panel.dates[span.values]
    lo, hi = days.min(), days.max()
    busy = {e.country_code for e in events if e.start <= hi and e.end >= lo}
    return [c for c in panel.countries
            if c != event.country_code and c not in busy and _has_sales(panel, c)]


def _ratio(values: np.ndarray, during: np.ndarray,
           reference: np.ndarray) -> float | None:
    d, r = values[during], values[reference]
    if np.isnan(d).all() or np.isnan(r).all():
        return None
    mean_d, mean_r = float(np.nanmean(d)), float(np.nanmean(r))
    return mean_d / mean_r if mean_r > 0 else None


def _effect(sales: dict[str, np.ndarray], country: str, peers: list[str],
            during: np.ndarray, reference: np.ndarray
            ) -> tuple[float, float, list[str]] | None:
    """(effect, peer ratio, peers actually used), or None without a market ratio."""
    a = _ratio(sales[country], during, reference)
    if a is None:
        return None
    used = {p: _ratio(sales[p], during, reference) for p in peers}
    used = {p: r for p, r in used.items() if r is not None}
    b = float(np.mean(list(used.values()))) if used else 1.0
    return a / b - 1.0, b, list(used)


def _swing(sales: dict[str, np.ndarray], country: str, peers: list[str],
           during: np.ndarray, reference: np.ndarray,
           blocked: np.ndarray) -> float | None:
    """Typical size of the same effect computed at every other weekly offset."""
    n = len(blocked)
    d_idx, r_idx = np.flatnonzero(during), np.flatnonzero(reference)
    both = np.concatenate([d_idx, r_idx])
    lo, hi = int(both.min()), int(both.max())
    values = []
    first_shift = -(lo // PLACEBO_STEP_DAYS) * PLACEBO_STEP_DAYS
    for k in range(first_shift, n - hi, PLACEBO_STEP_DAYS):
        if k == 0 or blocked[both + k].any():
            continue
        d, r = np.zeros(n, bool), np.zeros(n, bool)
        d[d_idx + k], r[r_idx + k] = True, True
        result = _effect(sales, country, peers, d, r)
        if result is not None:
            values.append(result[0])
    if len(values) < MIN_PLACEBOS:
        return None
    # Root-mean-square around zero, not the standard deviation around the
    # placebos' own mean: a market drifting against its peers gives every
    # placebo the same bias, and that drift is exactly what must not pass
    # for a real change.
    return float(np.sqrt(np.mean(np.square(values))))


def readout(event: DetectedEvent, panel: Panel,
            events: list[DetectedEvent]) -> Readout | None:
    windows = masks(event, panel, events)
    if windows is None:
        return None
    during, reference, side, trimmed = windows
    country = event.country_code
    channels = tuple(subject_channels(event, panel))
    spend = sum((panel.series(country, ch) for ch in channels),
                pd.Series(0.0, index=panel.dates))
    spend_reference = float(spend[reference].mean())
    spend_during = float(spend[during].mean())
    others = [ch for ch in panel.channels_in(country) if ch not in channels]
    other_spend = sum((panel.series(country, ch) for ch in others),
                      pd.Series(0.0, index=panel.dates))
    other_change = float(other_spend[during].mean() - other_spend[reference].mean())

    empty = dict(comparison_markets=(), sales_reference=None, sales_during=None,
                 effect=None, swing=None, readable=False, sales_difference=None,
                 rough_return=None, baseline=None)
    base = dict(during=during, reference=reference, reference_side=side,
                subject_channels=channels, spend_reference=spend_reference,
                spend_during=spend_during)
    if not _has_sales(panel, country):
        return Readout(**base, **empty,
                       note="No sales data for this market, so sales readings are unavailable.")

    peers = comparison_markets(event, panel, events, during | reference)
    sales = {c: panel.sales[c].to_numpy(dtype=float) for c in [country, *peers]}
    d, r = during.to_numpy(), reference.to_numpy()
    result = _effect(sales, country, peers, d, r)
    if result is None:
        return Readout(**base, **empty,
                       note="Sales are missing for these weeks, so sales readings are unavailable.")
    effect, b, used = result

    blocked = np.zeros(len(panel.dates), bool)
    for other in events:
        if other.country_code == country:
            blocked |= _window(panel.dates, other.start, other.end).to_numpy()
    swing = _swing(sales, country, used, d, r, blocked)
    readable = swing is not None and abs(effect) >= READABLE_MULTIPLE * swing

    sales_reference = float(np.nanmean(sales[country][r]))
    sales_during = float(np.nanmean(sales[country][d]))
    sales_difference = sales_reference * b * effect
    spend_change = spend_during - spend_reference
    notes = []
    rough_return = None
    if (readable and spend_change != 0
            and abs(spend_change) >= MIN_SPEND_CHANGE * max(spend_reference, spend_during)):
        # A return divides the whole sales change by one channel's spend
        # change, so it is withheld whenever that attribution is plainly wrong.
        if abs(other_change) > OTHER_SPEND_SHARE * abs(spend_change):
            notes.append("Spend on other channels also changed, so the sales change "
                         "cannot be put down to this one alone; no rough return.")
        elif sales_difference * spend_change < 0:
            notes.append("Sales moved the opposite way to spend, so something else "
                         "probably drove them; no rough return.")
        elif "confounded" in event.tags:
            notes.append("Another finding overlaps this one, so their effects cannot "
                         "be separated; no rough return.")
        else:
            rough_return = sales_difference / spend_change
    if trimmed:
        notes.append("The reference window skips days covered by another finding "
                     "in this market.")
    if not used:
        notes.append("No comparison markets without their own findings in these weeks, "
                     "so the sales change also includes any seasonality or trend.")
    if swing is None:
        notes.append("Too little history to judge how much sales normally swing.")
    elif not readable:
        notes.append("The change is within normal swings in sales, so it cannot be read clearly.")
    return Readout(**base, comparison_markets=tuple(used),
                   sales_reference=sales_reference, sales_during=sales_during,
                   effect=effect, swing=swing, readable=readable,
                   sales_difference=sales_difference, rough_return=rough_return,
                   baseline=sales_during if event.event_type == "dark_period" else None,
                   note=" ".join(notes))
