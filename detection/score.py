"""Spec section 8 -- the sub-scores behind detection_confidence.

Every sub-score is in [0, 1] and is reported alongside the result, so an analyst
can see WHY a number is what it is rather than being handed one opaque figure.
That is also why they are computed separately and combined by a weighted mean
instead of being folded into one expression: a term that misfires is visible.

The scores read the panel rather than trusting the event's evidence dict. The
composition layer writes evidence for human consumption and its contents vary by
event type; recomputing here keeps the six sub-scores defined for every event.

`subject_channels` is PUBLIC (no leading underscore) because it is shared across
a module boundary: detection/validity.py and detection/explain.py both need the
same "which channels does this event actually make a claim about" logic, and it
must not be duplicated there. It encodes an inversion that has already produced
a real defect in this codebase: a `single_channel` event's `channel` field names
the channel that is still RUNNING, so the event's subjects are every OTHER
channel in the market, not the named one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detection import params
from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.primitives.zero_runs import find_off_runs

COUNTRY_LEVEL_TYPES = frozenset({"dark_period", "single_channel"})


def _covering_run(panel: Panel, country: str, channel: str,
                  start: pd.Timestamp, end: pd.Timestamp):
    """The off-run this event sits on, or None for events that never stop."""
    series = panel.series(country, channel)
    present = panel.present_mask(country, channel)
    best, best_overlap = None, 0
    for run in find_off_runs(series, present):
        overlap = (min(run.end, end) - max(run.start, start)).days + 1
        if overlap > best_overlap:
            best, best_overlap = run, overlap
    return best


def subject_channels(event: DetectedEvent, panel: Panel) -> list[str]:
    """The channels this event actually makes a claim about.

    A dark period claims every channel in the market. A single_channel event
    names the channel still RUNNING, so its subjects are all the others -- the
    same inversion that produced a real defect in the cross-market layer.
    """
    country = event.country_code
    if event.event_type == "dark_period":
        return panel.channels_in(country)
    if event.event_type == "single_channel":
        return [ch for ch in panel.channels_in(country) if ch != event.channel]
    return [event.channel] if event.channel else []


def _magnitude_evidence(event: DetectedEvent, panel: Panel,
                        run) -> float:
    if event.event_type == "step_change":
        z = abs(float(event.evidence.get("z", 0.0)))
        return min(1.0, z / params.Z_SATURATION)
    if run is None:
        return 0.0
    # depth is the fraction by which spend fell; an exact zero is depth 1.
    return float(min(1.0, max(0.0, run.depth)))


def _duration_evidence(event: DetectedEvent) -> float:
    span = params.DURATION_SATURATION_MULT * params.MIN_DAYS
    return float(min(1.0, event.n_days / span))


def _distinctiveness(event: DetectedEvent, panel: Panel, run) -> float:
    """This run's length against the p90 of the series' OTHER off-runs.

    Per series, never global: a 30-day stop means something quite different on a
    channel that never pauses than on one that flights every fortnight.
    """
    if run is None or not event.channel:
        return 0.0
    series = panel.series(event.country_code, event.channel)
    others = [r.n_days for r in find_off_runs(series,
                                              panel.present_mask(event.country_code,
                                                                 event.channel))
              if not (r.start == run.start and r.end == run.end)]
    if not others:
        return 1.0
    p90 = float(np.percentile(np.array(others, dtype=float), 90))
    if p90 <= 0:
        return 1.0
    ratio = run.n_days / p90
    return float(min(1.0, ratio / params.DISTINCTIVENESS_SATURATION))


def _corroboration(event: DetectedEvent, panel: Panel) -> float:
    """Did impressions stop when spend did?

    Spend at zero with impressions still flowing is the signature of tracking
    loss rather than a deliberate pause. It scores low but not zero, because a
    spend feed that simply arrives late produces exactly the same shape.
    """
    if event.event_type == "step_change":
        return params.CORROBORATION_UNKNOWN
    channels = [ch for ch in subject_channels(event, panel) if ch]
    if not channels:
        return params.CORROBORATION_UNKNOWN
    window = slice(event.start, event.end)
    scores = []
    for ch in channels:
        key = (event.country_code, ch)
        if key not in panel.impressions.columns:
            scores.append(params.CORROBORATION_UNKNOWN)
            continue
        imps = panel.impressions.loc[window, key]
        total = float(np.nansum(imps.values))
        outside = panel.impressions[key].drop(panel.impressions.loc[window].index)
        if float(np.nansum(outside.values)) <= 0:
            # This series never reports impressions at all; silence inside the
            # window corroborates nothing.
            scores.append(params.CORROBORATION_UNKNOWN)
        elif total <= 0:
            scores.append(1.0)
        else:
            scores.append(params.CORROBORATION_CONTRADICTED)
    return float(np.mean(scores))


def _edge_sharpness(event: DetectedEvent, panel: Panel, run) -> float:
    if run is not None:
        return float(min(1.0, max(0.0, run.edge_sharpness)))
    if event.event_type == "step_change":
        return float(min(1.0, max(0.0, event.evidence.get("sharpness", 0.0))))
    return 0.0


def _consistency(event: DetectedEvent, panel: Panel) -> float:
    """For a country-level event, the fraction of the channels it claims that
    actually went off. A channel-level event claims nothing about its
    neighbours, so there is nothing to disagree and it scores full."""
    if event.event_type not in COUNTRY_LEVEL_TYPES:
        return 1.0
    channels = subject_channels(event, panel)
    if not channels:
        return 0.0
    agreeing = 0
    for ch in channels:
        run = _covering_run(panel, event.country_code, ch, event.start, event.end)
        if run is not None:
            agreeing += 1
    return agreeing / len(channels)


def sub_scores(event: DetectedEvent, panel: Panel) -> dict[str, float]:
    """The six named sub-scores from spec section 8, each in [0, 1]."""
    run = None
    if event.channel and event.country_code:
        run = _covering_run(panel, event.country_code, event.channel,
                            event.start, event.end)
    elif event.country_code and event.event_type == "dark_period":
        channels = subject_channels(event, panel)
        if channels:
            run = _covering_run(panel, event.country_code, channels[0],
                                event.start, event.end)
    return {
        "magnitude_evidence": _magnitude_evidence(event, panel, run),
        "duration_evidence": _duration_evidence(event),
        "distinctiveness": _distinctiveness(event, panel, run),
        "edge_sharpness": _edge_sharpness(event, panel, run),
        "corroboration": _corroboration(event, panel),
        "consistency": _consistency(event, panel),
    }
