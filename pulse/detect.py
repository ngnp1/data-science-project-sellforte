"""Score hook: media.csv in, channel_pulse, natural_holdout and single_channel Events out.

    python -m benchmark.eval.run_dev --detector pulse.detect:detect --load-data     # layer C (default)
    python -m benchmark.eval.run_dev --detector pulse.detect:detect_b --load-data   # layer B
    python -m benchmark.eval.run_dev --detector pulse.detect:detect_a --load-data   # layer A
"""
from __future__ import annotations

import pandas as pd

from benchmark.eval.model import Event
from pulse.hsmm import p_off_hsmm
from pulse.onoff import RHO, active_level, p_off, p_off_hmm
from pulse.holdout import find_holdouts, find_single_channel, market_p_off
from pulse.pulse import find_pulse_trains, off_spells, pulse_probability


def run_series(s: pd.Series, present: pd.Series | None = None, layer: str = "A"):
    """[(train, pulse probability, absorbed spike days)] for one channel's daily spend.

    Layer A: per-day soft cutoff; a day with no data counts as spend 0 (off).
    Layer B: sticky HMM; a day with no data is unknown and its neighbours decide.
    Layer C: semi-Markov model with explicit spell lengths; missing days as in B.
    """
    if layer == "A":
        p, present = p_off(s), None
    else:
        p = (p_off_hmm if layer == "B" else p_off_hsmm)(s, present)
    cut = RHO * active_level(s)
    runs = off_spells(p, s, present, cut)
    out = []
    for t in find_pulse_trains(runs):
        # Days the layer called off although spend was above the cutoff: absorbed spikes.
        spikes = tuple(d for a, b in t.components for d, v in s.loc[a:b].items() if v > cut)
        out.append((t, pulse_probability(p, t.components, runs), spikes))
    return out


def detect(media_df, sales_df, sid: str, layer: str = "C") -> list[Event]:
    if media_df is None:
        raise ValueError("pass --load-data so the detector gets media.csv")
    m = media_df.assign(date=pd.to_datetime(media_df["date"]))
    days = pd.date_range(m["date"].min(), m["date"].max())
    events = []
    for country, mc in m.groupby("country_code"):
        market, spans = {}, {}
        for channel, g in mc.groupby("advertising_channel"):
            daily = g.groupby("date")["media_investment"].sum()
            s, present = daily.reindex(days, fill_value=0.0), pd.Series(days.isin(daily.index), index=days)
            market[channel] = (s, present)
            for t, prob, spikes in run_series(s, present, layer):
                spans.setdefault(channel, []).extend(t.components)
                events.append(Event(sid=sid, country_code=country, channel=channel,
                                    event_type="channel_pulse", start=t.start, end=t.end,
                                    components=t.components, detection_confidence=prob,
                                    tags=tuple(f"spike_absorbed:{d.date()}" for d in spikes)))
        p = market_p_off(market)
        for kind, found in (("natural_holdout", find_holdouts(market, spans, p)),
                            ("single_channel", find_single_channel(market, p))):
            events += [Event(sid=sid, country_code=country, channel=channel, event_type=kind,
                             start=a, end=b, detection_confidence=conf)
                       for channel, a, b, conf in found]
    return events


def detect_a(media_df, sales_df, sid: str) -> list[Event]:
    return detect(media_df, sales_df, sid, layer="A")


def detect_b(media_df, sales_df, sid: str) -> list[Event]:
    return detect(media_df, sales_df, sid, layer="B")
