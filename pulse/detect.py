"""Score hook: media.csv in, one channel_pulse Event per pulse train out.

    python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data
"""
from __future__ import annotations

import pandas as pd

from benchmark.eval.model import Event
from pulse.onoff import off_mask, p_off
from pulse.pulse import find_pulse_trains, off_runs, pulse_probability


def detect(media_df, sales_df, sid: str) -> list[Event]:
    if media_df is None:
        raise ValueError("pass --load-data so the detector gets media.csv")
    m = media_df.assign(date=pd.to_datetime(media_df["date"]))
    days = pd.date_range(m["date"].min(), m["date"].max())
    events = []
    for (country, channel), g in m.groupby(["country_code", "advertising_channel"]):
        # A day with no rows is spend 0, i.e. off, same as the missing-day rule.
        s = g.groupby("date")["media_investment"].sum().reindex(days, fill_value=0.0)
        p, runs = p_off(s), off_runs(off_mask(s))
        for t in find_pulse_trains(runs):
            events.append(Event(sid=sid, country_code=country, channel=channel,
                                event_type="channel_pulse", start=t.start, end=t.end,
                                components=t.components,
                                detection_confidence=pulse_probability(p, t.components, runs)))
    return events
