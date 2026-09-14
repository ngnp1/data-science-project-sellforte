"""Compose primitives into labelled events by segmenting each market's timeline.

Not interval clustering, which would merge unrelated concurrent events. Instead,
per market, compute the ACTIVE-CHANNEL SET for each day and cut the timeline
wherever it changes. Each maximal run of a constant active set is a regime, and
regimes label themselves:

    A = {}                       -> dark_period
    |A| = 1, 2+ channels went off -> single_channel (naming the LIVE channel)
    0 < |K \\ A| < |K|           -> natural_holdout per off channel
    A = K, a step episode spans  -> step_change

This resolves the nesting of event types structurally rather than by precedence
rules: a dark period simply IS the regime where nothing is active, and its
constituent per-channel holdouts ride along as components instead of competing
with the event that contains them.

Three claims are settled BEFORE the regime pass, because each of them owns
windows that the regime pass would otherwise report a second time:

- a pulse train claims every off-window it groups, so neither the per-window
  holdout nor the "only the other channel is live" reading of those windows is
  emitted again;
- a channel that was never live on one side of a regime is a launch or a
  discontinuation, not a holdout, and P4 plus the cross-market layer own it;
- a step episode overlapping any notable off-run of its own channel is the
  level shift that the off-run already explains, and is dropped.

A step episode with no matching reversal is dropped as well; its end is the
series end by construction rather than by measurement. See _step_events.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detection import params
from detection.model import DetectedEvent
from detection.io.panel import Panel
from detection.primitives.level_shift import find_step_episodes
from detection.primitives.pulse import find_pulse_trains
from detection.primitives.zero_runs import find_off_runs, off_mask

Span = tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class Regime:
    start: pd.Timestamp
    end: pd.Timestamp
    active: frozenset[str]


def active_matrix(panel: Panel, country: str) -> pd.DataFrame:
    """Bool frame, index=dates, columns=the channels this market actually ran.

    A channel with no spend anywhere in the market is excluded outright: it
    means the market never ran that channel, and carrying it here would make
    every market look as though it were holding out everything it does not buy.
    """
    channels = panel.channels_in(country)
    data = {}
    for ch in channels:
        s = panel.series(country, ch)
        data[ch] = ~off_mask(s, panel.present_mask(country, ch))
    return pd.DataFrame(data, index=panel.dates)


def segment_regimes(panel: Panel, country: str) -> list[Regime]:
    matrix = active_matrix(panel, country)
    if matrix.empty or not len(matrix.columns):
        return []

    sets = [frozenset(matrix.columns[row.values]) for _, row in matrix.iterrows()]
    regimes: list[Regime] = []
    start_idx = 0
    for i in range(1, len(sets) + 1):
        if i == len(sets) or sets[i] != sets[start_idx]:
            regimes.append(Regime(start=panel.dates[start_idx],
                                  end=panel.dates[i - 1],
                                  active=sets[start_idx]))
            start_idx = i
    return regimes


def _was_active_around(matrix: pd.DataFrame, channel: str,
                       regime: Regime) -> bool:
    """Did this channel run before AND after the regime?

    Without this, a launch (off at the start) or a discontinuation (off at the
    end) would be mislabelled as a holdout.
    """
    before = matrix.loc[:regime.start, channel]
    after = matrix.loc[regime.end:, channel]
    return bool(before.iloc[:-1].any()) and bool(after.iloc[1:].any())


def _overlaps(span: Span, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return span[0] <= end and start <= span[1]


def _inside_a_pulse(spans: dict[str, list[Span]], channel: str,
                    regime: Regime) -> bool:
    """Is this off-window one of the windows a pulse train already grouped?"""
    return any(a <= regime.start and regime.end <= b
               for a, b in spans.get(channel, ()))


def _pulse_events(panel: Panel, country: str, sid: str, channels: list[str],
                  ) -> tuple[list[DetectedEvent], dict[str, list[Span]]]:
    """Pulse trains, plus the span each one claims, per channel.

    find_pulse_trains deliberately merges every notable off-run on a series
    into ONE train however far apart the runs sit, and this layer must not
    re-split it: the truth for a pulsing channel is grouped the same way, so a
    detector emitting one event per off-window scores an overlap far below the
    matcher's threshold and matches nothing at all.
    """
    events: list[DetectedEvent] = []
    spans: dict[str, list[Span]] = {}
    for ch in channels:
        runs = find_off_runs(panel.series(country, ch),
                             panel.present_mask(country, ch))
        for train in find_pulse_trains(runs):
            spans.setdefault(ch, []).append((train.start, train.end))
            events.append(DetectedEvent(
                sid=sid, country_code=country, channel=ch,
                event_type="channel_pulse", start=train.start, end=train.end,
                components=train.components,
                evidence={"n_pulses": train.n_pulses},
            ))
    return events, spans


def _step_events(panel: Panel, country: str, sid: str,
                 channels: list[str]) -> list[DetectedEvent]:
    """Step changes, on channels that stayed on for the whole episode.

    A channel that drops to zero and back produces an enormous level shift at
    BOTH edges -- clearing the z, persistence and sharpness gates easily -- so
    an unfiltered step pass would re-report every holdout and every dark period
    as a step change too. That is the same window twice, and it also breaks the
    rule that a dark period emits exactly one event type. A step is only a step
    if the channel never went off during it.
    """
    events: list[DetectedEvent] = []
    for ch in channels:
        off_windows = [(r.start, r.end) for r
                       in find_off_runs(panel.series(country, ch),
                                        panel.present_mask(country, ch))
                       if r.notable]
        for ep in find_step_episodes(panel.series(country, ch),
                                     exclude=off_windows):
            # No MIN_DAYS floor here, and none is reachable: find_level_shifts
            # keeps only one shift per W-wide neighbourhood, so a closed
            # episode spans at least W days and an open-ended one runs to the
            # series end from at least W days out. W is a whole multiple of
            # MIN_DAYS' week, so every episode already clears it -- a floor
            # would be a gate no input could ever trip, and an untestable
            # gate is worse than none. The invariant that makes that true is
            # pinned in tests/detection/test_label.py.
            if any(_overlaps(w, ep.start, ep.end) for w in off_windows):
                continue
            if ep.open_ended:
                # An episode with no matching reversal runs to the last
                # observed day BY CONSTRUCTION, not by measurement. Reporting
                # it asserts an extent this layer never established, and on the
                # development split every such episode overlapped its window so
                # loosely that none of them could match anything -- while each
                # one still cost a false positive. Four of them did sit on a
                # real step; each was missed because find_level_shifts never
                # detected the CLOSING shift, and the opening shift was dated
                # correctly. That is a recall defect one layer down, and
                # guessing a duration here to cover it would hide the defect
                # behind a number nothing measured. See the Task 7 report.
                continue
            events.append(DetectedEvent(
                sid=sid, country_code=country, channel=ch,
                event_type="step_change", start=ep.start, end=ep.end,
                magnitude_ratio=ep.ratio,
                evidence={"z": ep.z, "rose_from_zero": ep.ratio is None},
            ))
    return events


def label_market(panel: Panel, country: str, sid: str) -> list[DetectedEvent]:
    channels = panel.channels_in(country)
    if not channels:
        return []

    matrix = active_matrix(panel, country)
    all_channels = frozenset(channels)

    events, pulse_spans = _pulse_events(panel, country, sid, channels)

    for regime in segment_regimes(panel, country):
        n_days = int((regime.end - regime.start).days) + 1
        if n_days < params.MIN_DAYS:
            continue
        off = all_channels - regime.active

        if not regime.active:
            events.append(DetectedEvent(
                sid=sid, country_code=country, channel=None,
                event_type="dark_period", start=regime.start, end=regime.end,
                components=tuple((regime.start, regime.end) for _ in channels),
                evidence={"n_channels_off": len(channels)},
            ))
        elif (len(regime.active) == 1 and len(off) > 1
                and not all(_inside_a_pulse(pulse_spans, ch, regime)
                            for ch in off)
                and all(_was_active_around(matrix, ch, regime) for ch in off)):
            # MORE THAN ONE channel has to have gone dark.
            #
            # This differs from the spec's "at least 2 channels exist" ONLY in
            # a two-channel market -- everywhere else, one active channel
            # already implies two or more off. And in a two-channel market the
            # label is genuinely AMBIGUOUS FROM SPEND ALONE: one channel
            # stopping and one continuing is the same shape either way, and
            # which name it carries depends only on which scenario family drew
            # it, not on anything observable in the data.
            #
            # An earlier version of this comment claimed the generator has a
            # convention here, naming the channel that stopped. It does not,
            # and that claim was false. Choosing to report the window as a
            # holdout is a PRIOR, not a reading of the generator: it is right
            # about as often as it is wrong, and it is deliberately not tuned
            # to the handful of development scenarios that would flip it.
            #
            # Requiring two off channels also stops a market that runs exactly
            # one channel and never stops it from reporting its whole history
            # as a single-channel period -- that part is unambiguous.
            events.append(DetectedEvent(
                sid=sid, country_code=country,
                channel=next(iter(regime.active)),
                event_type="single_channel", start=regime.start,
                end=regime.end,
                evidence={"n_channels_off": len(off)},
            ))
        elif off:
            for ch in sorted(off):
                if _inside_a_pulse(pulse_spans, ch, regime):
                    continue
                if not _was_active_around(matrix, ch, regime):
                    continue
                events.append(DetectedEvent(
                    sid=sid, country_code=country, channel=ch,
                    event_type="natural_holdout", start=regime.start,
                    end=regime.end, evidence={},
                ))

    events.extend(_step_events(panel, country, sid, channels))
    events.sort(key=lambda e: (e.start, str(e.channel), e.event_type))
    return events
