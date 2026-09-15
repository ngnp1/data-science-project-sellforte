"""Spec section 8's validity gate.

Three verdicts: `ok`, `suspect_data_gap`, `suspect_tracking_loss`. The gate
exists because the two most dangerous real-data failures produce EXACTLY the
shape of a genuine event. An export that omits rows instead of writing zeros
looks like a perfect holdout. A tracking pixel that keeps firing after spend
stops looks like a pause that did not happen.

Neither can be settled from spend alone, so the gate reports suspicion rather
than filtering: a detector that silently dropped these would hide the single
largest risk in spec section 11 instead of surfacing it. This module never
drops or suppresses an event -- it only annotates one.

Spec section 8 names a fourth trigger -- a spend drop with no sales response
in a window where sales SNR was adequate to show one -- that is deliberately
not implemented here. On this benchmark sales are generated from the spend
the detector already reads, so the check could not be validated without
scoring the detector against its own answer key; it is recorded as
not-implemented rather than added on faith.
"""
from __future__ import annotations

import numpy as np

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.primitives.zero_runs import off_mask
from detection.score import subject_channels

OK = "ok"
SUSPECT_DATA_GAP = "suspect_data_gap"
SUSPECT_TRACKING_LOSS = "suspect_tracking_loss"

# A step change never stops the channel, so absence and tracking checks have no
# subject there -- neither trigger has anything to say about a level shift.
_STOPPING_TYPES = frozenset({"dark_period", "single_channel", "natural_holdout",
                             "channel_pulse", "staggered_launch"})


def assess(event: DetectedEvent, panel: Panel) -> tuple[str, tuple[str, ...]]:
    """Spec section 8's validity gate: report suspicion, never filter.

    Returns the verdict and every reason that fired, so an analyst triaging
    the output sees all of them rather than only the first.
    """
    reasons: list[str] = []
    verdict = OK
    if event.event_type not in _STOPPING_TYPES:
        return verdict, ()

    channels = [ch for ch in subject_channels(event, panel) if ch]
    window = slice(event.start, event.end)

    for ch in channels:
        key = (event.country_code, ch)
        if key not in panel.present.columns:
            continue
        present = panel.present.loc[window, key]
        if not bool(present.all()):
            missing = int((~present).sum())
            reasons.append(
                f"{ch}: {missing} of {len(present)} days in the window are "
                f"missing rows rather than zero-spend rows -- the export may "
                f"be omitting rows rather than reporting a real stop")
            verdict = SUSPECT_DATA_GAP

    for ch in channels:
        key = (event.country_code, ch)
        if key not in panel.impressions.columns:
            continue
        # This trigger's whole claim is "spend stopped but impressions did
        # not", so it may only fire where spend ACTUALLY stopped. Without this
        # check it fires on any subject channel that has impressions at all --
        # and a dark_period's subjects are every channel in the market,
        # including ones that never paused. It then prints "spend is zero" about
        # a channel spending normally, which is a false statement in output an
        # analyst is meant to trust. Reuses off_mask so "off" means here exactly
        # what it means everywhere else in the detector.
        spent = panel.series(event.country_code, ch)
        off = off_mask(spent, panel.present_mask(event.country_code, ch))
        if not bool(off.loc[window].any()):
            continue
        inside = float(np.nansum(panel.impressions.loc[window, key].values))
        outside_frame = panel.impressions[key].drop(
            panel.impressions.loc[window].index)
        outside = float(np.nansum(outside_frame.values))
        if inside > 0 and outside > 0:
            reasons.append(
                f"{ch}: spend is zero but impressions continue in the window "
                f"-- this may be tracking loss rather than a pause")
            if verdict == OK:
                verdict = SUSPECT_TRACKING_LOSS

    if _everything_off(panel, event):
        reasons.append(
            "every channel in every market is off simultaneously -- far more "
            "likely a feed outage than a coordinated global pause")
        if verdict == OK:
            verdict = SUSPECT_DATA_GAP

    return verdict, tuple(reasons)


def _everything_off(panel: Panel, event: DetectedEvent) -> bool:
    """True when the ENTIRE panel -- every country, every channel -- is zero
    across the event window. A panel-wide stop is far more likely to be a
    feed outage than a coordinated global pause, so this check reads the
    whole panel rather than the event's own subject channels."""
    window = panel.spend.loc[event.start:event.end]
    if window.empty or window.shape[1] <= 1:
        return False
    return bool(np.nansum(window.values) == 0)
