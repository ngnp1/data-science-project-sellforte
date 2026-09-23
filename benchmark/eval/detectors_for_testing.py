"""Detectors with known-correct scores, used to validate the harness itself.

These live in benchmark/eval/ and NOT in detection/, deliberately: the oracles
read ground truth, which detection code is never permitted to do. They exist to
answer one question -- does the harness recognise a correct detector as correct,
and an incorrect one as incorrect? -- and they are the only thing standing
between a broken metric and a report full of flattering nonsense.

They are DEV-ONLY. Nothing here may be pointed at the test split.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from benchmark.eval import truth as T
from benchmark.eval.model import Event

# (media_df, sales_df, sid) -> detections. media/sales may be None for the
# oracles, which ignore the data entirely.
DetectorFn = Callable[[pd.DataFrame | None, pd.DataFrame | None, str], list[Event]]


def never_detect(media, sales, sid: str) -> list[Event]:
    """Reports nothing. Recall 0, but a flawless false-positive rate -- which is
    why the FP rate cannot be the only headline number."""
    return []


def perfect_oracle(media, sales, sid: str) -> list[Event]:
    """Emits exactly the normalised truth, at full confidence. Must score 1.0
    on every metric, and must produce a perfectly calibrated reliability curve
    (everything in the top bin, empirical precision 1.0)."""
    return [
        Event(**{**e.__dict__, "detection_confidence": 1.0,
                 "informativeness": 1.0})
        for e in T.load_truth("dev", sid)
    ]


def ungrouped_pulse_oracle(media, sales, sid: str) -> list[Event]:
    """Perfect except that pulse trains are emitted as one event per off-window
    rather than one grouped event -- the shape a detector would produce if the
    loader's grouping were removed. Must NOT score 1.0; it is the negative
    control proving the grouping is actually exercised.
    """
    out = []
    for e in T.load_truth("dev", sid):
        if e.event_type == "channel_pulse" and e.components:
            for start, end in e.components:
                out.append(Event(
                    sid=e.sid, country_code=e.country_code, channel=e.channel,
                    event_type=e.event_type, start=start, end=end,
                ))
        else:
            out.append(e)
    return out


def panel_launch_oracle(media, sales, sid: str) -> list[Event]:
    """Perfect except that staggered launches are emitted the way spec section
    7 describes them -- ONE panel-level event with no `country_code`, spanning
    the whole launch group -- rather than fanned out to one event per market.

    Must NOT score 1.0. It is the negative control for the staggered_launch
    reconciliation, the twin of `ungrouped_pulse_oracle`: truth rows are
    already per-market and the loader copies `country` through unchanged, so
    the fan-out is the DETECTOR's obligation and nothing in the loader can
    satisfy it. Measured on dev before this control existed, a detector of
    exactly this shape scored 0.000/0.000/0.000 on staggered_launch -- 16 of
    the 127 matchable test events -- with the whole suite green.
    """
    out = []
    groups: dict[tuple, list[Event]] = {}
    for e in T.load_truth("dev", sid):
        if e.event_type == "staggered_launch":
            groups.setdefault((e.channel,), []).append(e)
        else:
            out.append(e)

    for (channel,), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out.append(Event(
            sid=sid,
            country_code=None,          # panel-level: spec section 7's shape
            channel=channel,
            event_type="staggered_launch",
            start=min(g.start for g in group),
            end=max(g.end for g in group),
        ))
    return out


def shifted_oracle(days: int) -> DetectorFn:
    """Perfect, but every interval slid `days` forward. Degrades IoU and shows
    up in boundary error, so it validates both."""
    delta = pd.Timedelta(days=days)

    def detector(media, sales, sid: str) -> list[Event]:
        return [
            Event(sid=e.sid, country_code=e.country_code, channel=e.channel,
                  event_type=e.event_type, start=e.start + delta,
                  end=e.end + delta, tags=e.tags)
            for e in T.load_truth("dev", sid)
        ]

    return detector


def wrong_channel_oracle(media, sales, sid: str) -> list[Event]:
    """Correct windows, correct markets, wrong channel. Under the strict match
    these become FN+FP; only the relaxed match reveals the channel error."""
    return [
        Event(sid=e.sid, country_code=e.country_code,
              channel=("__wrong__" if e.channel is not None else None),
              event_type=e.event_type, start=e.start, end=e.end)
        for e in T.load_truth("dev", sid)
    ]


def half_confident_oracle(media, sales, sid: str) -> list[Event]:
    """Correct detections, but confidence alternates high/low. Its reliability
    curve must therefore be MIScalibrated -- low-confidence events are just as
    correct as high-confidence ones -- which is what proves the curve measures
    calibration rather than merely reporting the scores back."""
    out = []
    for i, e in enumerate(T.load_truth("dev", sid)):
        out.append(Event(**{**e.__dict__,
                            "detection_confidence": 0.95 if i % 2 else 0.15}))
    return out


def detect_everything(media, sales, sid: str) -> list[Event]:
    """One dark_period spanning the whole series in every market. Maximises
    recall-by-brute-force and should be destroyed by precision and by the
    false-positive rate on the null scenarios."""
    meta = T.load_meta("dev", sid)
    scen = T.load_scenario("dev", sid)
    start = T.START_DATE
    end = T.START_DATE + pd.Timedelta(days=int(meta["n_days"]) - 1)
    return [
        Event(sid=sid, country_code=c["code"], channel=None,
              event_type="dark_period", start=start, end=end)
        for c in scen["countries"]
    ]
