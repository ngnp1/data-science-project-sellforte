"""Compose market-level onsets into one event per late-starting market."""
from __future__ import annotations

import pandas as pd

from detection import params
from detection.compose.cross_market import _peer_status
from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.primitives.onset import find_onset, sustained_activity


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
        candidates = {}
        for country in panel.countries:
            if channel not in panel.channels_in(country):
                # The market never bought this channel; it is not a market
                # waiting to launch it.
                continue
            s = panel.series(country, channel)
            present = panel.present_mask(country, channel)
            onset = find_onset(s, present)
            if onset:
                candidates[country] = onset
                onsets[country] = onset.first_active
            elif sustained_activity(s, present):
                onsets[country] = panel.dates[0]

        if not onsets:
            continue
        spread = (max(onsets.values()) - min(onsets.values())).days
        if spread < params.ONSET_SPREAD:
            # Either a channel live everywhere from day one, or start-up
            # jitter. A lone market always lands here: its spread is zero.
            continue

        earliest = min(onsets.values())
        for country, onset in sorted(candidates.items()):
            first_active = onset.first_active
            separation = (first_active - earliest).days
            if separation < params.ONSET_SPREAD:
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
                          "observed_inactive_days": onset.observed_inactive_days,
                          "active_confirmation_days": onset.active_confirmation_days,
                          "onset_separation_days": separation,
                          "n_markets": len(onsets)},
            ))
    return out
