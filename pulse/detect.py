"""Score hook: media.csv in, one channel_pulse Event per pulse train out.

    python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data     # layer A
    python -m benchmark.eval.run_dev --detector pulse.detect:detect_b --load-data   # layer B
"""
from __future__ import annotations

import pandas as pd

from benchmark.eval.model import Event
from pulse.onoff import RHO, active_level, p_off, p_off_hmm
from pulse.pulse import find_pulse_trains, off_runs, pulse_probability


def run_series(s: pd.Series, present: pd.Series | None = None, layer: str = "A"):
    """[(train, pulse probability, absorbed spike days)] for one channel's daily spend.

    Layer A: per-day soft cutoff; a day with no data counts as spend 0 (off).
    Layer B: sticky HMM; a day with no data is unknown and its neighbours decide.
    """
    if layer == "A":
        p, present = p_off(s), None
    else:
        p = p_off_hmm(s, present)
    runs = off_runs(p >= 0.5, present)
    cut = RHO * active_level(s)
    out = []
    for t in find_pulse_trains(runs):
        # Days the layer called off although spend was above the cutoff: absorbed spikes.
        spikes = tuple(d for a, b in t.components for d, v in s.loc[a:b].items() if v > cut)
        out.append((t, pulse_probability(p, t.components, runs), spikes))
    return out


def detect(media_df, sales_df, sid: str, layer: str = "A") -> list[Event]:
    if media_df is None:
        raise ValueError("pass --load-data so the detector gets media.csv")
    m = media_df.assign(date=pd.to_datetime(media_df["date"]))
    days = pd.date_range(m["date"].min(), m["date"].max())
    events = []
    for (country, channel), g in m.groupby(["country_code", "advertising_channel"]):
        daily = g.groupby("date")["media_investment"].sum()
        s, present = daily.reindex(days, fill_value=0.0), pd.Series(days.isin(daily.index), index=days)
        for t, prob, spikes in run_series(s, present, layer):
            events.append(Event(sid=sid, country_code=country, channel=channel,
                                event_type="channel_pulse", start=t.start, end=t.end,
                                components=t.components, detection_confidence=prob,
                                tags=tuple(f"spike_absorbed:{d.date()}" for d in spikes)))
    return events


def detect_b(media_df, sales_df, sid: str) -> list[Event]:
    return detect(media_df, sales_df, sid, layer="B")
