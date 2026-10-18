"""Peer-market comparison: identify potential comparison groups.

This layer decides an event's worth more than anything about its own shape. A
holdout whose peers kept running has a potential comparison group; causal comparability still needs
analyst validation. The same holdout with every peer
also off has no control at all -- and looks exactly like a pipeline outage.

Two rules govern which peers get a vote:

- a market that never bought a channel is not a control for it. Its series is
  all zeros, and reading that as "off during the window" would manufacture a
  global pause out of markets that simply do not use the channel, exactly as
  it would manufacture a control out of them if read the other way. Such a
  market abstains.
- the question is always asked about the channels that STOPPED. For a holdout
  or a pulse train that is the event's own channel; for a dark period it is
  every channel the market runs; for a single-channel period the event names
  the channel still LIVE, so the subject is every OTHER channel in the market.
  Asking the peers about the surviving channel asks the opposite question and
  answers "peers" whenever one channel is untouched everywhere.

It also owns staggered launches, and FANS THEM OUT to one event per market.
The benchmark's truth is per-market, so a country-less panel event matches
nothing; benchmark/eval/README.md states this is the detector's obligation.

WHY PEER COMPARISON IS DONE ON off_mask AND NOT ON A RESCALED SERIES
--------------------------------------------------------------------
Spec section 7 describes the cross-market layer over the market-scale-normalized
series, and this module compares binary off/on masks on raw spend instead. The
deviation is in the letter, not the intent, and the intent is the requirement
that a 15x larger market must not dominate the comparison.

off_mask thresholds each series at RHO * active_level(s), where active_level is
that series' OWN median positive spend. Market size therefore cancels before the
comparison begins: dividing by market_scale first and then applying a threshold
proportional to the same series' own level is arithmetically a no-op. The
question this layer asks -- "were my peers running while I was dark?" -- is
already scale-free.

That equivalence holds only for the on/off question. A cross-market comparison
of MAGNITUDES (did this market cut spend harder than its peers?) is not
scale-free and would need the normalized series. cross_market_series() and
channel_share() exist and are tested for exactly that, and have no caller in the
current pipeline: they are for the magnitude-aware comparisons in the next plan,
not dead code left behind.
"""
from __future__ import annotations

import pandas as pd

from detection import params
from detection.model import DetectedEvent
from detection.io.panel import Panel
from detection.primitives.onset import find_onset
from detection.primitives.zero_runs import off_mask

# Events whose worth depends on whether some other market kept spending. A
# step change is deliberately not among them: this layer measures peers by
# whether they were OFF, which says nothing about whether a peer's budget
# moved, so any control claim attached to a step here would be unmeasured.
_CONTROL_TYPES = frozenset({"dark_period", "single_channel",
                            "natural_holdout", "channel_pulse"})


def _subject_channels(panel: Panel, event: DetectedEvent) -> list[str]:
    """The channels that went off, which is what the peers are asked about."""
    ran = panel.channels_in(event.country_code)
    if event.event_type == "single_channel":
        return [ch for ch in ran if ch != event.channel]
    if event.channel is None:
        return ran
    return [event.channel]


def _peer_status(panel: Panel, country: str, channels: list[str],
                 start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    """Count peers continuously active and peers continuously off.

    Partial-window activity or missing rows qualifies as neither. A running
    peer must carry every subject channel throughout the event window.
    """
    running = off = 0
    for peer in panel.countries:
        if peer == country:
            continue
        ran = panel.channels_in(peer)
        subject = [ch for ch in channels if ch in ran]
        if not subject:
            # This market never bought any of these channels, so it can speak
            # neither for nor against a control group. It abstains.
            continue
        peer_off = True
        peer_running = len(subject) == len(channels)
        for ch in subject:
            mask = off_mask(panel.series(peer, ch),
                            panel.present_mask(peer, ch))
            window = mask.loc[start:end]
            known = panel.present_mask(peer, ch).loc[start:end]
            peer_off = peer_off and bool(window.all() and known.all())
            peer_running = peer_running and bool((~window).all() and known.all())
        if peer_off:
            off += 1
        elif peer_running:
            running += 1
    return running, off


def annotate(events: list[DetectedEvent], panel: Panel) -> list[DetectedEvent]:
    out: list[DetectedEvent] = []
    for e in events:
        if e.event_type not in _CONTROL_TYPES:
            out.append(e)
            continue

        running, peers_off = _peer_status(panel, e.country_code,
                                          _subject_channels(panel, e),
                                          e.start, e.end)
        tags = list(e.tags)
        if running > 0:
            control = "peers"
            if e.event_type == "natural_holdout":
                tags.append("cross_market_holdout")
        elif peers_off > 0 and peers_off == sum(
                bool(set(_subject_channels(panel, e)) & set(panel.channels_in(peer)))
                for peer in panel.countries if peer != e.country_code):
            control = "none"
            tags.append("global_pause")
        else:
            # No continuously active peer is available. Only credit siblings
            # that actually exist and stay active throughout the window.
            siblings = set(panel.channels_in(e.country_code)) - set(_subject_channels(panel, e))
            healthy = any(not off_mask(panel.series(e.country_code, ch),
                          panel.present_mask(e.country_code, ch)).loc[e.start:e.end].any()
                          for ch in siblings)
            control = "sibling_channels" if healthy else "none"

        evidence = dict(e.evidence)
        evidence.update(control_available=control, peers_running=running,
                        peers_off=peers_off)
        # Rebuilt rather than mutated, and `components` is carried through
        # untouched: a pulse train is ONE event with its windows attached, and
        # re-splitting it here would make a grouped truth interval unmatchable.
        out.append(DetectedEvent(
            sid=e.sid, country_code=e.country_code, channel=e.channel,
            event_type=e.event_type, start=e.start, end=e.end,
            magnitude_ratio=e.magnitude_ratio, components=e.components,
            tags=tuple(tags), evidence=evidence,
        ))
    return out


def find_staggered_launches(panel: Panel, sid: str) -> list[DetectedEvent]:
    """One event per market that started this channel late.

    A dormant start is only a launch if the channel was demonstrably runnable
    at the time, which is what the peers establish: some market was already
    live during this market's dormancy. The earliest market to start has no
    such witness -- nothing was live anywhere before it -- so it is the
    comparison group, not an event. On its own its dormancy is a censored
    holdout, and calling it a launch would be a guess.
    """
    out: list[DetectedEvent] = []
    for channel in panel.channels:
        onsets: dict[str, pd.Timestamp] = {}
        for country in panel.countries:
            if channel not in panel.channels_in(country):
                # The market never bought this channel; it is not a market
                # waiting to launch it.
                continue
            s = panel.series(country, channel)
            onset = find_onset(s, panel.present_mask(country, channel))
            onsets[country] = onset.first_active if onset else panel.dates[0]

        if not onsets:
            continue
        spread = (max(onsets.values()) - min(onsets.values())).days
        if spread < params.ONSET_SPREAD:
            # Either a channel live everywhere from day one, or start-up
            # jitter. A lone market always lands here: its spread is zero.
            continue

        earliest = min(onsets.values())
        for country, first_active in sorted(onsets.items()):
            if first_active == earliest:
                continue
            idx = panel.dates.get_loc(first_active)
            running, _ = _peer_status(panel, country, [channel], panel.dates[0], panel.dates[idx - 1])
            out.append(DetectedEvent(
                sid=sid, country_code=country, channel=channel,
                event_type="staggered_launch", start=panel.dates[0],
                end=panel.dates[idx - 1],
                tags=("cross_market_control",) if running else (),
                evidence={"control_available": "peers" if running else "none",
                          "peers_running": running, "censored_start": True,
                          "onset": str(first_active.date()),
                          "earliest_market_onset": str(earliest.date()),
                          "n_markets": len(onsets)},
            ))
    return out
