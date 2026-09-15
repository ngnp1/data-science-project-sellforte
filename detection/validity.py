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
import pandas as pd

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.primitives.zero_runs import find_off_runs, off_mask
from detection.score import subject_channels

OK = "ok"
SUSPECT_DATA_GAP = "suspect_data_gap"
SUSPECT_TRACKING_LOSS = "suspect_tracking_loss"

# A step change never stops the channel, so the tracking-loss trigger -- whose
# claim is "spend stopped but impressions did not" -- has no subject there.
# The missing-rows trigger is a weaker fit for this exclusion: absent rows are
# an export defect whatever the channel was doing, and a step episode with a
# hole in it is still a hole. It is excluded here anyway, deliberately, because
# widening it is a change to WHAT THE GATE REPORTS rather than a correction of
# something it states falsely, and this branch does not add reach to a gate on
# the way out the door. Recorded as an open design question in REPORT.md.
# tests/detection/test_validity.py pins the exclusion behaviourally, on a step
# window that really does have missing rows.
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

    # The missing-rows trigger reads the SPAN, not the claimed windows the
    # tracking-loss trigger below reads. Its claim is about export integrity
    # over the interval the event reports -- "these days have no rows at all"
    # is true of a day whatever the channel was doing on it -- so widening it
    # to a grouped event's active days states nothing false. The tracking-loss
    # claim is specifically about days spend STOPPED, which is why only that
    # one needs the narrower windows.
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

    claimed = _claimed_mask(panel.dates, event)
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
        #
        # Both halves of the test are per-DAY, not per-window. An existential
        # "some day in the window was off" plus a total "the window has
        # impressions" is satisfied by a window that merely MIXES off and on
        # days: the impressions are counted on the on-days and the claim is
        # made about the off-days, which is the same false sentence the
        # off_mask check above exists to prevent. The claim needs the
        # INTERSECTION -- impressions landing on the days spend was off --
        # over the days this event actually reports as off (`_claimed_mask`).
        spent = panel.series(event.country_code, ch)
        present = panel.present_mask(event.country_code, ch)
        off = off_mask(spent, present)
        stopped = off & claimed
        if not bool(stopped.any()):
            continue
        impressions = panel.impressions[key]
        inside = float(np.nansum(impressions[stopped].values))
        outside = float(np.nansum(impressions[~claimed].values))
        if inside > 0 and outside > 0:
            reasons.append(_tracking_loss_reason(ch, spent, present, claimed))
            if verdict == OK:
                verdict = SUSPECT_TRACKING_LOSS

    if _everything_off(panel, event):
        reasons.append(
            "every channel in every market is off simultaneously -- far more "
            "likely a feed outage than a coordinated global pause")
        if verdict == OK:
            verdict = SUSPECT_DATA_GAP

    return verdict, tuple(reasons)


def _claimed_mask(dates: pd.DatetimeIndex, event: DetectedEvent) -> pd.Series:
    """The days this event actually reports as off, as a boolean day mask.

    A GROUPED event -- a pulse train -- spans first start to last end, so its
    [start, end] interval also covers the active days BETWEEN its windows,
    where the channel was spending normally and impressions are supposed to
    be flowing. Its real windows are carried in `components` (the spec's
    output shape for a pulse; see detection/primitives/pulse.py), and a
    trigger whose claim is about the days spend stopped has to read those
    rather than the span. Reading the span instead is what let the
    tracking-loss trigger print "spend is zero but impressions continue"
    about a pulsing channel whose off-days carried exactly zero impressions
    and whose every impression fell on a day it was spending normally.

    Ungrouped types carry no components, and there the span IS the window.
    """
    mask = pd.Series(False, index=dates)
    windows = event.components or ((event.start, event.end),)
    for start, end in windows:
        mask.loc[start:end] = True
    return mask


def _tracking_loss_reason(channel: str, spent: pd.Series, present: pd.Series,
                          claimed: pd.Series) -> str:
    """The trigger's sentence, worded to match what the spend actually did.

    `off_mask` calls a day off at a small FRACTION of the channel's active
    level, not at zero, so a near-zero event -- spend cut to a trickle and
    held there -- is "off" while still buying. Saying "spend is zero" about
    a window that spent five figures is a false statement in output an
    analyst is meant to trust, and it is the same defect detection/explain.py
    fixed by replacing "stopped" with "cut to a trickle" for a near-zero run.
    The wording here is taken from the same source of truth those sentences
    use -- `OffRun.kind` on the runs covering the claimed days -- so the two
    modules cannot drift into describing one window two different ways.

    Impressions continuing in proportion to residual spend is ordinary rather
    than suspicious, so the near-zero sentence says what it saw and rates its
    own evidence down instead of repeating the stop wording.
    """
    kinds = {run.kind for run in find_off_runs(spent, present)
             if bool(claimed.loc[run.start:run.end].any())}
    if "near_zero" in kinds:
        return (f"{channel}: spend was cut to a trickle rather than to zero "
                f"on the days this event reports as off, and impressions "
                f"continue there -- impressions tracking residual spend is "
                f"ordinary, so this is far weaker evidence of tracking loss "
                f"than a stop at exactly zero")
    return (f"{channel}: spend is zero on the days this event reports as off "
            f"but impressions continue on those same days -- this may be "
            f"tracking loss rather than a pause")


def _everything_off(panel: Panel, event: DetectedEvent) -> bool:
    """True when the ENTIRE panel -- every country, every channel -- is zero
    across the event window. A panel-wide stop is far more likely to be a
    feed outage than a coordinated global pause, so this check reads the
    whole panel rather than the event's own subject channels."""
    window = panel.spend.loc[event.start:event.end]
    if window.empty or window.shape[1] <= 1:
        return False
    return bool(np.nansum(window.values) == 0)
