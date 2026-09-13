"""Load ground truth into normalised, matchable intervals.

Three reconciliations happen here, all forced by decisions frozen into the
sealed data and documented in benchmark/BENCHMARK.md. They belong in the
LOADER, not in any detector: the detector output shape is fixed by spec
section 7, and the truth shape is fixed by the generator, so this module is the
only place the two can be brought together.

1. channel_pulse truth is one row per off-window; spec section 7's P3 emits ONE
   grouped event. Ungrouped, a grouped detection scores IoU ~0.118 against any
   single row, so a PERFECT pulse detector would match nothing -- a third of the
   test split's matchable events.
2. global_pause is a truth pattern_type but a detector TAG on a dark_period
   regime.
3. staggered_launch truth is per-market; the detector emits a panel-level event.
   Convention chosen here, once: fan out to one event per market, which keeps
   the section 9 matcher unchanged.

Interval dates come from scenario.json's `end_day`, never from the CSV's
`end_date`. `reformat.py` clamps `end_idx` at the series end, so `end_date` is
slice-exclusive MOST of the time and already-inclusive at the boundary, and the
two are indistinguishable from the CSV alone. `end_day` is the unclamped
exclusive day number and needs no special case.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from benchmark.eval.model import NON_EVENT_TYPES, TYPE_MAP, Event
from benchmark.harness.runner import DATASETS_DIR, truth_dir

# Every scenario starts here; benchmark/harness/config_writer.py pins it.
START_DATE = pd.Timestamp("2024-01-01")

_PULSE_ORDINAL = re.compile(r"pulse (\d+) of (\d+)")


def list_scenarios(split: str, root: Path | None = None) -> list[str]:
    base = Path(root or DATASETS_DIR) / split
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def load_meta(split: str, sid: str, root: Path | None = None) -> dict:
    return json.loads((truth_dir(split, sid, root or DATASETS_DIR)
                       / "meta.json").read_text())


def load_scenario(split: str, sid: str, root: Path | None = None) -> dict:
    return json.loads((truth_dir(split, sid, root or DATASETS_DIR)
                       / "scenario.json").read_text())


def _interval(raw_event: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive [start, end] from the unclamped slice-exclusive day offsets."""
    start = START_DATE + pd.Timedelta(days=int(raw_event["start_day"]))
    end = START_DATE + pd.Timedelta(days=int(raw_event["end_day"]) - 1)
    return start, end


def _raw_events(split: str, sid: str, root: Path | None) -> list[dict]:
    return list(load_scenario(split, sid, root).get("events", []))


def _to_event(sid: str, raw: dict, *, event_type: str,
              tags: tuple[str, ...] = ()) -> Event:
    start, end = _interval(raw)
    channel = raw["channel"]
    return Event(
        sid=sid,
        country_code=raw["country"],
        channel=None if channel == "ALL" else channel,
        event_type=event_type,
        start=start,
        end=end,
        pattern_id=raw["pattern_id"],
        multiplier=(None if raw.get("multiplier") is None
                    else float(raw["multiplier"])),
        tags=tags,
    )


def _group_pulses(sid: str, rows: list[dict]) -> list[Event]:
    """Collapse a pulse train into one event per (country, channel).

    The individual windows are kept as `components` so a component-level
    secondary view stays possible. `description` carries "pulse i of n", which
    gives the intended group size -- we assert the count matches so a partially
    written truth file is loud rather than silently mis-scored.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for raw in rows:
        by_key.setdefault((raw["country"], raw["channel"]), []).append(raw)

    out = []
    for (country, channel), group in sorted(by_key.items()):
        group.sort(key=lambda r: int(r["start_day"]))
        spans = [_interval(r) for r in group]

        declared = {int(m.group(2))
                    for m in (_PULSE_ORDINAL.search(r.get("description", ""))
                              for r in group) if m}
        if declared and declared != {len(group)}:
            raise ValueError(
                f"{sid}: pulse train {country}/{channel} has {len(group)} rows "
                f"but its descriptions declare {sorted(declared)}")

        out.append(Event(
            sid=sid,
            country_code=country,
            channel=channel,
            event_type="channel_pulse",
            start=min(s for s, _ in spans),
            end=max(e for _, e in spans),
            pattern_id=group[0]["pattern_id"],
            multiplier=(None if group[0].get("multiplier") is None
                        else float(group[0]["multiplier"])),
            components=tuple(spans),
        ))
    return out


def load_truth(split: str, sid: str, root: Path | None = None) -> list[Event]:
    """Matchable ground-truth events, normalised to the detector's shape."""
    raws = _raw_events(split, sid, root)

    pulse_rows = [r for r in raws if r["pattern_type"] == "channel_pulse"]
    events: list[Event] = _group_pulses(sid, pulse_rows)

    for raw in raws:
        ptype = raw["pattern_type"]
        if ptype in NON_EVENT_TYPES or ptype == "channel_pulse":
            continue
        label = TYPE_MAP.get(ptype)
        if label is None:
            raise ValueError(f"{sid}: unmapped truth pattern_type {ptype!r}")
        tags = ("global_pause",) if ptype == "global_pause" else ()
        events.append(_to_event(sid, raw, event_type=label, tags=tags))

    events.sort(key=lambda e: (e.start, str(e.country_code), str(e.channel)))
    return events


def load_non_events(split: str, sid: str, root: Path | None = None) -> list[Event]:
    """The negative-control windows. Not matchable -- a detection inside one of
    these is a false positive, and the report needs to be able to say so."""
    return [
        _to_event(sid, raw, event_type=raw["pattern_type"])
        for raw in _raw_events(split, sid, root)
        if raw["pattern_type"] in NON_EVENT_TYPES
    ]
