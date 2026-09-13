# Evaluation Harness Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the evaluation harness that scores a detector against the frozen benchmark, and prove it correct against deliberately trivial detectors *before* any real detection algorithm exists.

**Architecture:** `benchmark/eval/` loads ground truth into normalised intervals, reconciling three granularity mismatches the sealed data froze in; matches detections to truth greedily by temporal IoU; computes the ten metrics spec §9 requires; and exposes two runners — `run_dev.py`, free to run and appending to a development history, and `run_final.py`, gated behind an explicit flag, a seal verification, and an append-only audit record. A set of synthetic detectors with known-correct scores (a perfect oracle, a never-detect, a channel-swapper) is the harness's own test suite.

**Tech Stack:** Python 3.13, pandas 3.0.5, numpy 2.5.2, pytest. Virtualenv at `synthetic_data_generator/.venv`.

**Spec:** `docs/superpowers/specs/2026-09-13-informative-periods-detection-design.md` (sections 9 and 10), plus `benchmark/BENCHMARK.md`, which documents the loader traps discovered while freezing the benchmark and is binding for this plan.

## Global Constraints

- **Pure pandas and numpy.** No scipy, sklearn, or ruptures anywhere in the project. pytest and PyYAML are permitted; pytest is test-only.
- **Python interpreter is always `synthetic_data_generator/.venv/bin/python`.** Never bare `python3`.
- **Never read `benchmark/datasets/test_truth/*/ground_truth.csv` during development.** The whole project rests on the algorithms being developed without seeing the sealed answers. Development and validation use the `dev` split only. `run_final.py` is the single sanctioned reader of test truth, and it is not run during this plan.
- **`detection/` must have no import path to any truth loader.** Truth reading lives only in `benchmark/eval/`. A test asserts this.
- **Never modify, regenerate, reseal, or delete anything under `benchmark/datasets/`.** The test split is sealed; `verify_seal('test')` must return OK at the end of every task.
- **Intervals are inclusive on both ends**, `[start, end]`, as `pandas.Timestamp` at day resolution.
- Dataset directories hold only `media.csv` and `sales.csv`; truth lives in `<split>_truth/<sid>/`.
- Split sizes: dev 45 (`dev_001`..`dev_045`), test 55 (`test_001`..`test_055`). Sids carry no family suffix.
- All paths relative to the project root, `/Users/user123/Desktop/projects/datascience-project`.
- Work happens on branch `evaluation-harness`, created off `benchmark-generation`. Do not merge to `main`.

## File Structure

| file | responsibility |
|---|---|
| `benchmark/eval/model.py` | the `Event` dataclass and `NON_EVENT_TYPES` / `TYPE_MAP` constants — shared vocabulary, no logic |
| `benchmark/eval/truth.py` | load `ground_truth.csv` + `scenario.json` + `meta.json` into normalised matchable `Event`s, applying all three granularity reconciliations |
| `benchmark/eval/matching.py` | temporal IoU, greedy one-to-one matching, strict and relaxed modes |
| `benchmark/eval/metrics.py` | the ten metrics of spec §9 |
| `benchmark/eval/breakdowns.py` | per-axis slices, with the confounded axes flagged rather than silently reported |
| `benchmark/eval/report.py` | render a metrics dict to readable Markdown |
| `benchmark/eval/detectors_for_testing.py` | synthetic detectors with known-correct scores; the harness's own oracle |
| `benchmark/eval/runner.py` | shared evaluate-a-detector-over-a-split logic |
| `benchmark/eval/run_dev.py` | free runner, appends `dev_history.jsonl` |
| `benchmark/eval/run_final.py` | gated runner, appends `final_runs.jsonl` |
| `tests/eval/…` | one test module per source module |

---

### Task 1: Branch, scaffold, and the Event model

**Files:**
- Create: `benchmark/eval/__init__.py`, `benchmark/eval/model.py`
- Create: `tests/eval/__init__.py`, `tests/eval/test_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) Event` with fields `sid: str`, `country_code: str | None`, `channel: str | None`, `event_type: str`, `start: pd.Timestamp`, `end: pd.Timestamp`, `pattern_id: str | None = None`, `multiplier: float | None = None`, `detection_confidence: float | None = None`, `informativeness: float | None = None`, `tags: tuple[str, ...] = ()`, `components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()`
  - `Event.n_days -> int` property (inclusive length)
  - `NON_EVENT_TYPES: frozenset[str]` = `{"ramp_block", "intermittent_baseline"}`
  - `TYPE_MAP: dict[str, str]` mapping truth `pattern_type` to the detector label it should match
  - `MATCHABLE_TYPES: frozenset[str]`
  - Every later task consumes `Event`.

- [ ] **Step 1: Create the branch and directories**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git checkout benchmark-generation
git checkout -b evaluation-harness
mkdir -p benchmark/eval tests/eval
touch benchmark/eval/__init__.py tests/eval/__init__.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/eval/test_model.py`:

```python
import pandas as pd
import pytest

from benchmark.eval.model import (
    Event, NON_EVENT_TYPES, TYPE_MAP, MATCHABLE_TYPES,
)


def ev(**kw):
    base = dict(
        sid="dev_001", country_code="DE", channel="TV",
        event_type="natural_holdout",
        start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-03-10"),
    )
    base.update(kw)
    return Event(**base)


def test_n_days_is_inclusive():
    assert ev().n_days == 10
    assert ev(start=pd.Timestamp("2024-03-01"),
              end=pd.Timestamp("2024-03-01")).n_days == 1


def test_event_is_hashable_and_frozen():
    e = ev()
    assert hash(e) is not None
    with pytest.raises(Exception):
        e.start = pd.Timestamp("2024-01-01")


def test_negative_controls_are_exactly_the_two_intended_types():
    assert NON_EVENT_TYPES == frozenset({"ramp_block", "intermittent_baseline"})


def test_type_map_covers_every_truth_type_the_generator_emits():
    """These are the pattern_types benchmark/spec/events.py can produce."""
    produced = {
        "dark_period", "single_channel", "natural_holdout", "step_change",
        "channel_pulse", "staggered_launch", "global_pause",
        "ramp_block", "intermittent_baseline",
    }
    assert produced <= set(TYPE_MAP)


def test_global_pause_maps_to_dark_period():
    """BENCHMARK.md: truth records global_pause as a pattern_type, but the
    detector labels the regime dark_period and applies global_pause as a tag."""
    assert TYPE_MAP["global_pause"] == "dark_period"


def test_identity_mappings_are_identity():
    for t in ["dark_period", "single_channel", "natural_holdout",
              "step_change", "channel_pulse", "staggered_launch"]:
        assert TYPE_MAP[t] == t


def test_negative_controls_are_not_matchable():
    assert MATCHABLE_TYPES.isdisjoint(NON_EVENT_TYPES)
    assert "dark_period" in MATCHABLE_TYPES


def test_tags_and_components_default_empty_and_are_tuples():
    e = ev()
    assert e.tags == () and e.components == ()


def test_scores_default_to_none_so_truth_events_carry_no_confidence():
    e = ev()
    assert e.detection_confidence is None and e.informativeness is None
    scored = ev(detection_confidence=0.9, informativeness=0.4)
    assert scored.detection_confidence == 0.9
```

- [ ] **Step 3: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_model.py -v`
Expected: FAIL — `No module named 'benchmark.eval.model'`.

- [ ] **Step 4: Write `benchmark/eval/model.py`**

```python
"""Shared vocabulary for the evaluation harness.

Deliberately logic-free: every other eval module imports these names, so
keeping them in one dependency-free place stops the import graph tangling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Pattern types the generator injects that change the data but must NEVER be
# reported as detections. Anything a detector finds inside their windows is a
# false positive -- that is the point of them.
NON_EVENT_TYPES: frozenset[str] = frozenset({"ramp_block", "intermittent_baseline"})

# Truth pattern_type -> the detector label that should match it.
# Per BENCHMARK.md, `global_pause` is the one genuine rename: truth records it
# as a pattern_type, while spec section 7 labels the regime `dark_period` and
# applies `global_pause` as a cross-market TAG. Matching on the string would
# fail on every one of those events.
TYPE_MAP: dict[str, str] = {
    "dark_period": "dark_period",
    "single_channel": "single_channel",
    "natural_holdout": "natural_holdout",
    "step_change": "step_change",
    "channel_pulse": "channel_pulse",
    "staggered_launch": "staggered_launch",
    "global_pause": "dark_period",
    # Negative controls map to themselves so the loader can recognise and drop
    # them explicitly rather than by silently failing a lookup.
    "ramp_block": "ramp_block",
    "intermittent_baseline": "intermittent_baseline",
}

MATCHABLE_TYPES: frozenset[str] = frozenset(
    label for truth_type, label in TYPE_MAP.items()
    if truth_type not in NON_EVENT_TYPES
)


@dataclass(frozen=True)
class Event:
    """One interval, either from ground truth or from a detector.

    Intervals are INCLUSIVE on both ends. `country_code` may be None for a
    panel-level event that spans markets; `channel` may be None for an event
    that covers every channel (a dark period).
    """
    sid: str
    country_code: str | None
    channel: str | None
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    pattern_id: str | None = None
    multiplier: float | None = None
    # Spec section 8's three scores. Truth events leave these None; detections
    # set them, and spec section 9 items 8 and 9 (the reliability and operating
    # curves) are computed from detection_confidence.
    detection_confidence: float | None = None
    informativeness: float | None = None
    tags: tuple[str, ...] = ()
    # For grouped events (a pulse train), the individual windows that compose it.
    components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()

    @property
    def n_days(self) -> int:
        return int((self.end - self.start).days) + 1
```

- [ ] **Step 5: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_model.py -v`
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval tests/eval
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(eval): add the Event model and truth-type vocabulary"
```

---

### Task 2: Truth loader

The task that decides whether the whole benchmark measures anything. Three reconciliations, all mandated by `BENCHMARK.md`.

**Files:**
- Create: `benchmark/eval/truth.py`
- Test: `tests/eval/test_truth.py`

**Interfaces:**
- Consumes: `Event`, `NON_EVENT_TYPES`, `TYPE_MAP` (Task 1); `benchmark.harness.runner.truth_dir`.
- Produces:
  - `START_DATE: pd.Timestamp` = `2024-01-01`
  - `load_meta(split, sid, root=None) -> dict`
  - `load_scenario(split, sid, root=None) -> dict`
  - `load_truth(split, sid, root=None) -> list[Event]` — matchable events, fully normalised
  - `load_non_events(split, sid, root=None) -> list[Event]` — the negative-control windows
  - `list_scenarios(split) -> list[str]`
  - Tasks 5, 6, 7, 8 and 9 all consume `load_truth`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_truth.py`:

```python
"""Truth-loader tests run against the DEV split only. Reading test-split
ground truth during development would defeat the entire project."""
import pandas as pd
import pytest

from benchmark.eval import truth as T
from benchmark.eval.model import NON_EVENT_TYPES


def test_lists_all_dev_scenarios():
    sids = T.list_scenarios("dev")
    assert len(sids) == 45
    assert sids[0] == "dev_001" and sids[-1] == "dev_045"


def test_meta_carries_the_breakdown_axes():
    m = T.load_meta("dev", "dev_001")
    assert m["sid"] == "dev_001"
    for key in ["family", "noise_level", "trend_p", "market_spread",
                "n_countries", "n_channels", "n_days", "years", "n_events"]:
        assert key in m


def test_null_scenarios_load_zero_events():
    for sid in T.list_scenarios("dev"):
        if T.load_meta("dev", sid)["family"] == "null":
            assert T.load_truth("dev", sid) == []


def test_intervals_are_inclusive_and_derived_from_end_day():
    """BENCHMARK.md: end_date in the CSV is slice-exclusive EXCEPT where it is
    clamped at the series end. We sidestep that entirely by taking end_day from
    scenario.json, where it is always the unclamped exclusive day number."""
    for sid in T.list_scenarios("dev"):
        scen = T.load_scenario("dev", sid)
        by_id = {e["pattern_id"]: e for e in scen["events"]}
        for e in T.load_truth("dev", sid) + T.load_non_events("dev", sid):
            for pid in ([e.pattern_id] if e.pattern_id else []):
                if pid in by_id:
                    src = by_id[pid]
                    expected_start = T.START_DATE + pd.Timedelta(days=src["start_day"])
                    expected_end = T.START_DATE + pd.Timedelta(days=src["end_day"] - 1)
                    assert e.start == expected_start, sid
                    assert e.end == expected_end, sid


def test_censored_end_reaches_the_last_day_of_the_series():
    """dev_037 is the censored_end edge case and is one of the four scenarios
    where the CSV's end_date is already inclusive. Getting this wrong shortens
    the window by a day and the edge case stops testing what it was built for."""
    meta = T.load_meta("dev", "dev_037")
    last_day = T.START_DATE + pd.Timedelta(days=meta["n_days"] - 1)
    events = T.load_truth("dev", "dev_037")
    assert any(e.end == last_day for e in events), \
        f"no event reaches {last_day.date()}"


def test_pulse_rows_are_grouped_into_one_event_per_country_channel():
    """THE critical reconciliation. dev_019 has four 14-day pulse rows; the
    detector emits ONE grouped event. Ungrouped, a perfect pulse detector scores
    IoU ~0.118 against any single row and matches nothing."""
    events = T.load_truth("dev", "dev_019")
    pulses = [e for e in events if e.event_type == "channel_pulse"]
    assert len(pulses) == 1, f"expected 1 grouped pulse, got {len(pulses)}"
    p = pulses[0]
    assert p.country_code == "DE" and p.channel == "Affiliate"
    assert p.start == pd.Timestamp("2024-05-16")
    assert p.end == pd.Timestamp("2024-09-11")   # end_day - 1, inclusive
    assert len(p.components) == 4
    assert all(isinstance(a, pd.Timestamp) and isinstance(b, pd.Timestamp)
               for a, b in p.components)


def test_pulse_grouping_is_per_country_channel_not_per_scenario():
    """Two pulse trains on different channels must stay two events."""
    for sid in T.list_scenarios("dev"):
        events = T.load_truth("dev", sid)
        pulses = [e for e in events if e.event_type == "channel_pulse"]
        keys = [(e.country_code, e.channel) for e in pulses]
        assert len(keys) == len(set(keys)), f"{sid}: duplicate pulse group"


def test_negative_controls_are_excluded_from_matchable_truth():
    for sid in T.list_scenarios("dev"):
        for e in T.load_truth("dev", sid):
            assert e.event_type not in NON_EVENT_TYPES, sid


def test_negative_controls_are_available_separately():
    """They must be loadable, because a detection inside one is a false
    positive and the reporting needs to say so."""
    hits = [sid for sid in T.list_scenarios("dev")
            if T.load_non_events("dev", sid)]
    assert hits, "no scenario exposed any negative-control window"
    for sid in hits:
        for e in T.load_non_events("dev", sid):
            assert e.event_type in NON_EVENT_TYPES


def test_global_pause_is_relabelled_to_dark_period_and_tagged():
    found = False
    for sid in T.list_scenarios("dev"):
        scen = T.load_scenario("dev", sid)
        if any(e["pattern_type"] == "global_pause" for e in scen["events"]):
            found = True
            events = T.load_truth("dev", sid)
            gp = [e for e in events if "global_pause" in e.tags]
            assert gp, sid
            assert all(e.event_type == "dark_period" for e in gp), sid
    assert found, "no dev scenario carries a global_pause"


def test_staggered_launch_is_fanned_out_per_market():
    """Convention chosen once and stated: the panel-level detector event is
    fanned out to one per market, so the section 9 matcher stays unchanged."""
    found = False
    for sid in T.list_scenarios("dev"):
        launches = [e for e in T.load_truth("dev", sid)
                    if e.event_type == "staggered_launch"]
        if launches:
            found = True
            assert all(e.country_code is not None for e in launches), sid
    assert found, "no dev scenario carries a staggered_launch"


def test_multiplier_is_carried_from_scenario_json_not_the_csv():
    """ground_truth.csv blanks multiplier for everything except step_change, so
    magnitude breakdowns must come from scenario.json."""
    seen = set()
    for sid in T.list_scenarios("dev"):
        for e in T.load_truth("dev", sid):
            if e.event_type == "natural_holdout" and e.multiplier is not None:
                seen.add(e.multiplier)
    assert 0.0 in seen, "no exact-zero holdout magnitude reached the loader"
    assert any(0 < m < 0.1 for m in seen), "no near-zero magnitude"


def test_every_matchable_event_has_a_positive_length():
    for sid in T.list_scenarios("dev"):
        for e in T.load_truth("dev", sid):
            assert e.n_days >= 1, (sid, e.pattern_id)


def test_loader_never_touches_the_test_split_truth(monkeypatch):
    """A guard against the single worst accident this project can have."""
    import builtins
    real_open = builtins.open

    def guarded(path, *a, **kw):
        if "test_truth" in str(path):
            raise AssertionError(f"loader touched sealed truth: {path}")
        return real_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded)
    T.load_truth("dev", "dev_019")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_truth.py -v`
Expected: FAIL — `No module named 'benchmark.eval.truth'`.

- [ ] **Step 3: Write `benchmark/eval/truth.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_truth.py -v`
Expected: 14 passed.

If `test_censored_end_reaches_the_last_day_of_the_series` fails, the interval derivation is wrong — do not adjust the test, which encodes the rule `BENCHMARK.md` verified on disk.

- [ ] **Step 5: Print a truth summary and eyeball it**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python - <<'PY'
from collections import Counter
from benchmark.eval import truth as T
c, n = Counter(), 0
for sid in T.list_scenarios("dev"):
    evs = T.load_truth("dev", sid)
    n += len(evs)
    c.update(e.event_type for e in evs)
print(f"dev matchable events: {n}")
for k, v in sorted(c.items()):
    print(f"  {k:20} {v}")
print("negative-control windows:",
      sum(len(T.load_non_events("dev", s)) for s in T.list_scenarios("dev")))
PY
```
Expected: roughly 92 matchable dev events per `BENCHMARK.md`, with `channel_pulse` collapsed to about 10 grouped events rather than 27 rows. Record the actual numbers in your report.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval/truth.py tests/eval/test_truth.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(eval): load ground truth, reconciling the three granularity mismatches"
```

---

### Task 3: Interval matching

**Files:**
- Create: `benchmark/eval/matching.py`
- Test: `tests/eval/test_matching.py`

**Interfaces:**
- Consumes: `Event` (Task 1).
- Produces:
  - `iou(a: Event, b: Event) -> float`
  - `overlap_days(a: Event, b: Event) -> int`
  - `@dataclass Match` with `truth: Event`, `pred: Event`, `iou: float`
  - `@dataclass MatchResult` with `matches: list[Match]`, `unmatched_truth: list[Event]`, `unmatched_pred: list[Event]`
  - `match_events(truth, pred, min_iou=0.5, strict=True) -> MatchResult`
  - Tasks 4, 5, 6 and 7 consume `match_events`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_matching.py`:

```python
import pandas as pd

from benchmark.eval.matching import iou, match_events, overlap_days
from benchmark.eval.model import Event


def e(start, end, *, country="DE", channel="TV", etype="natural_holdout",
      sid="dev_001"):
    return Event(sid=sid, country_code=country, channel=channel,
                 event_type=etype,
                 start=pd.Timestamp(start), end=pd.Timestamp(end))


def test_identical_intervals_have_iou_one():
    assert iou(e("2024-03-01", "2024-03-10"), e("2024-03-01", "2024-03-10")) == 1.0


def test_disjoint_intervals_have_iou_zero():
    assert iou(e("2024-03-01", "2024-03-10"), e("2024-04-01", "2024-04-10")) == 0.0


def test_adjacent_intervals_do_not_overlap():
    """Inclusive intervals: [1..10] and [11..20] touch but share no day."""
    assert overlap_days(e("2024-03-01", "2024-03-10"),
                        e("2024-03-11", "2024-03-20")) == 0


def test_iou_is_computed_on_inclusive_days():
    # [1..10] vs [6..15]: overlap 5 days, union 15 days
    got = iou(e("2024-03-01", "2024-03-10"), e("2024-03-06", "2024-03-15"))
    assert abs(got - 5 / 15) < 1e-9


def test_ungrouped_pulse_reproduces_the_documented_failure():
    """BENCHMARK.md's worked example: a grouped 119-day detection against a
    single 14-day truth window scores about 0.118 and cannot match at 0.5."""
    grouped = e("2024-05-16", "2024-09-11")
    single = e("2024-05-16", "2024-05-29")
    assert abs(iou(grouped, single) - 14 / 119) < 0.01
    assert iou(grouped, single) < 0.5


def test_strict_match_requires_same_country_type_and_channel():
    t = [e("2024-03-01", "2024-03-10")]
    assert len(match_events(t, [e("2024-03-01", "2024-03-10")]).matches) == 1
    assert not match_events(t, [e("2024-03-01", "2024-03-10", country="AT")]).matches
    assert not match_events(t, [e("2024-03-01", "2024-03-10", channel="Radio")]).matches
    assert not match_events(t, [e("2024-03-01", "2024-03-10",
                                  etype="step_change")]).matches


def test_relaxed_match_ignores_type_country_and_channel():
    t = [e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-01", "2024-03-10", country="AT", channel="Radio",
           etype="step_change")]
    assert len(match_events(t, p, strict=False).matches) == 1


def test_below_threshold_overlap_does_not_match():
    t = [e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-09", "2024-03-20")]      # 2/20 = 0.1
    assert not match_events(t, p).matches
    assert len(match_events(t, p).unmatched_truth) == 1
    assert len(match_events(t, p).unmatched_pred) == 1


def test_matching_is_one_to_one_and_greedy_by_best_iou():
    """Two predictions overlap one truth; only the better one may match."""
    t = [e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-01", "2024-03-09"), e("2024-03-01", "2024-03-10")]
    res = match_events(t, p)
    assert len(res.matches) == 1
    assert res.matches[0].iou == 1.0
    assert len(res.unmatched_pred) == 1


def test_each_truth_event_matches_at_most_once():
    t = [e("2024-03-01", "2024-03-10"), e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-01", "2024-03-10")]
    res = match_events(t, p)
    assert len(res.matches) == 1
    assert len(res.unmatched_truth) == 1


def test_empty_inputs_are_handled():
    assert match_events([], []).matches == []
    assert len(match_events([e("2024-03-01", "2024-03-10")], []).unmatched_truth) == 1
    assert len(match_events([], [e("2024-03-01", "2024-03-10")]).unmatched_pred) == 1


def test_matching_is_deterministic_regardless_of_input_order():
    t = [e("2024-03-01", "2024-03-10"), e("2024-06-01", "2024-06-10")]
    p = [e("2024-06-01", "2024-06-10"), e("2024-03-01", "2024-03-10")]
    a = match_events(t, p)
    b = match_events(list(reversed(t)), list(reversed(p)))
    assert sorted(m.truth.start for m in a.matches) == \
           sorted(m.truth.start for m in b.matches)


def test_events_from_different_scenarios_never_match():
    t = [e("2024-03-01", "2024-03-10", sid="dev_001")]
    p = [e("2024-03-01", "2024-03-10", sid="dev_002")]
    assert not match_events(t, p).matches
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_matching.py -v`
Expected: FAIL — `No module named 'benchmark.eval.matching'`.

- [ ] **Step 3: Write `benchmark/eval/matching.py`**

```python
"""Greedy one-to-one matching of detections to ground truth by temporal IoU.

Spec section 9: matched greedily by descending temporal IoU, one-to-one,
requiring the same country, type and channel, with IoU at least 0.5.

Greedy rather than optimal (Hungarian) on purpose: with a 0.5 threshold no
truth interval can be claimed by two predictions that also overlap each other
enough to change the assignment, so greedy and optimal agree in practice, and
greedy is auditable by hand -- which matters when a reviewer has to explain why
a particular event was scored the way it was.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from benchmark.eval.model import Event


def overlap_days(a: Event, b: Event) -> int:
    """Inclusive-interval overlap, in whole days."""
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    return max(0, int((hi - lo).days) + 1)


def iou(a: Event, b: Event) -> float:
    inter = overlap_days(a, b)
    if inter == 0:
        return 0.0
    union = a.n_days + b.n_days - inter
    return inter / union if union else 0.0


@dataclass
class Match:
    truth: Event
    pred: Event
    iou: float


@dataclass
class MatchResult:
    matches: list[Match] = field(default_factory=list)
    unmatched_truth: list[Event] = field(default_factory=list)
    unmatched_pred: list[Event] = field(default_factory=list)

    @property
    def n_tp(self) -> int:
        return len(self.matches)

    @property
    def n_fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def n_fn(self) -> int:
        return len(self.unmatched_truth)


def _compatible(t: Event, p: Event, strict: bool) -> bool:
    if t.sid != p.sid:
        return False
    if not strict:
        return True
    return (t.country_code == p.country_code
            and t.channel == p.channel
            and t.event_type == p.event_type)


def match_events(truth: list[Event], pred: list[Event], *,
                 min_iou: float = 0.5, strict: bool = True) -> MatchResult:
    """One-to-one assignment, best IoU first.

    `strict=False` ignores country, channel and type, which is what spec
    section 9 item 4 needs: under the strict match a channel mix-up vanishes
    into a false negative plus a false positive and channel accuracy reads a
    meaningless 100%.
    """
    candidates = []
    for ti, t in enumerate(truth):
        for pi, p in enumerate(pred):
            if not _compatible(t, p, strict):
                continue
            score = iou(t, p)
            if score >= min_iou:
                candidates.append((score, ti, pi))

    # Sort by descending IoU, then by index so ties resolve deterministically
    # regardless of the order the caller supplied.
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_t: set[int] = set()
    used_p: set[int] = set()
    result = MatchResult()
    for score, ti, pi in candidates:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        result.matches.append(Match(truth=truth[ti], pred=pred[pi], iou=score))

    result.unmatched_truth = [t for i, t in enumerate(truth) if i not in used_t]
    result.unmatched_pred = [p for i, p in enumerate(pred) if i not in used_p]
    return result
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_matching.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval/matching.py tests/eval/test_matching.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(eval): add temporal IoU and greedy one-to-one matching"
```

---

### Task 4: The ten metrics

**Files:**
- Create: `benchmark/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `Event` (Task 1), `match_events`, `MatchResult` (Task 3).
- Produces:
  - `prf(n_tp, n_fp, n_fn) -> dict` with keys `precision`, `recall`, `f1`
  - `event_level(truth, pred) -> dict`
  - `per_type(truth, pred) -> dict[str, dict]`
  - `iou_stats(result) -> dict` with `mean_iou`, `median_iou`
  - `boundary_error(result) -> dict` with `start_median`, `start_p90`, `end_median`, `end_p90`
  - `channel_and_market_accuracy(truth, pred) -> dict`
  - `day_level(truth, pred) -> dict`
  - `type_confusion(truth, pred) -> dict[tuple[str, str], int]`
  - `false_positive_rate(pred, country_years) -> float`
  - `reliability_curve(truth, pred, n_bins=10) -> list[dict]` — spec §9 item 8
  - `operating_curve(truth, pred, cuts=None) -> list[dict]` — spec §9 item 9
  - `evaluate_scenario(truth, pred) -> dict` — everything above for one scenario
  - Tasks 5, 6 and 7 consume `evaluate_scenario`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_metrics.py`:

```python
import pandas as pd

from benchmark.eval import metrics as M
from benchmark.eval.matching import match_events
from benchmark.eval.model import Event


def e(start, end, *, country="DE", channel="TV", etype="natural_holdout",
      sid="dev_001"):
    return Event(sid=sid, country_code=country, channel=channel,
                 event_type=etype,
                 start=pd.Timestamp(start), end=pd.Timestamp(end))


def test_prf_on_a_clean_split():
    got = M.prf(n_tp=3, n_fp=1, n_fn=1)
    assert got["precision"] == 0.75
    assert got["recall"] == 0.75
    assert got["f1"] == 0.75


def test_prf_handles_empty_denominators_without_dividing_by_zero():
    got = M.prf(n_tp=0, n_fp=0, n_fn=0)
    assert got["precision"] == 0.0 and got["recall"] == 0.0 and got["f1"] == 0.0


def test_perfect_prediction_scores_one_across_the_board():
    t = [e("2024-03-01", "2024-03-10"), e("2024-06-01", "2024-06-20")]
    got = M.event_level(t, list(t))
    assert got["precision"] == 1.0 and got["recall"] == 1.0 and got["f1"] == 1.0


def test_empty_prediction_scores_zero_recall_and_no_false_positives():
    t = [e("2024-03-01", "2024-03-10")]
    got = M.event_level(t, [])
    assert got["recall"] == 0.0
    assert got["n_fp"] == 0 and got["n_fn"] == 1


def test_prediction_on_a_null_scenario_is_all_false_positives():
    got = M.event_level([], [e("2024-03-01", "2024-03-10")])
    assert got["n_fp"] == 1 and got["n_tp"] == 0
    assert got["precision"] == 0.0


def test_per_type_splits_the_score():
    t = [e("2024-03-01", "2024-03-10", etype="dark_period"),
         e("2024-06-01", "2024-06-20", etype="step_change")]
    p = [e("2024-03-01", "2024-03-10", etype="dark_period")]
    got = M.per_type(t, p)
    assert got["dark_period"]["recall"] == 1.0
    assert got["step_change"]["recall"] == 0.0


def test_iou_stats_over_matched_pairs():
    t = [e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-01", "2024-03-10")]
    assert M.iou_stats(match_events(t, p))["mean_iou"] == 1.0


def test_boundary_error_in_days():
    t = [e("2024-03-01", "2024-03-20")]
    p = [e("2024-03-03", "2024-03-18")]      # start +2, end -2
    got = M.boundary_error(match_events(t, p))
    assert got["start_median"] == 2 and got["end_median"] == 2


def test_channel_error_is_visible_rather_than_vanishing():
    """Under the STRICT match a channel mix-up becomes FN+FP and channel
    accuracy would read 100%. The relaxed match is what exposes it."""
    t = [e("2024-03-01", "2024-03-10", channel="TV")]
    p = [e("2024-03-01", "2024-03-10", channel="Radio")]
    got = M.channel_and_market_accuracy(t, p)
    assert got["channel_accuracy"] == 0.0
    assert got["market_accuracy"] == 1.0
    assert got["n_relaxed_matches"] == 1


def test_market_error_is_visible():
    t = [e("2024-03-01", "2024-03-10", country="DE")]
    p = [e("2024-03-01", "2024-03-10", country="AT")]
    got = M.channel_and_market_accuracy(t, p)
    assert got["market_accuracy"] == 0.0


def test_day_level_counts_days_not_events():
    """Robust to split/merge disagreements: two adjacent predictions covering
    one truth interval score well at day level even though event-level
    matching may reject them."""
    t = [e("2024-03-01", "2024-03-20")]
    p = [e("2024-03-01", "2024-03-10"), e("2024-03-11", "2024-03-20")]
    got = M.day_level(t, p)
    assert got["recall"] == 1.0
    assert got["precision"] == 1.0


def test_day_level_penalises_over_prediction():
    t = [e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-01", "2024-03-20")]
    got = M.day_level(t, p)
    assert got["recall"] == 1.0
    assert got["precision"] == 0.5


def test_type_confusion_records_the_substitution():
    t = [e("2024-03-01", "2024-03-10", etype="dark_period")]
    p = [e("2024-03-01", "2024-03-10", etype="single_channel")]
    got = M.type_confusion(t, p)
    assert got[("dark_period", "single_channel")] == 1


def test_false_positive_rate_is_per_country_year():
    assert M.false_positive_rate([e("2024-03-01", "2024-03-10")] * 4,
                                 country_years=2) == 2.0
    assert M.false_positive_rate([], country_years=2) == 0.0


def test_reliability_curve_bins_confidence_against_correctness():
    """Spec section 9 item 8. A confident-and-right detection and an
    unconfident-and-wrong one must land in different bins."""
    t = [e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-01", "2024-03-10"),
         e("2024-08-01", "2024-08-10")]
    p = [Event(**{**p[0].__dict__, "detection_confidence": 0.95}),
         Event(**{**p[1].__dict__, "detection_confidence": 0.15})]
    curve = M.reliability_curve(t, p, n_bins=2)
    hi = [b for b in curve if b["bin_lo"] >= 0.5][0]
    lo = [b for b in curve if b["bin_lo"] < 0.5][0]
    assert hi["empirical_precision"] == 1.0 and hi["n"] == 1
    assert lo["empirical_precision"] == 0.0 and lo["n"] == 1


def test_reliability_curve_is_empty_when_nothing_is_scored():
    t = [e("2024-03-01", "2024-03-10")]
    assert M.reliability_curve(t, [e("2024-03-01", "2024-03-10")]) == []


def test_operating_curve_trades_precision_for_recall():
    """Spec section 9 item 9: sweeping the confidence cut must move precision
    and recall in opposite directions, so the handover can state a trade-off
    rather than a single point."""
    t = [e("2024-03-01", "2024-03-10"), e("2024-06-01", "2024-06-10")]
    p = [Event(**{**e("2024-03-01", "2024-03-10").__dict__,
                  "detection_confidence": 0.9}),
         Event(**{**e("2024-06-01", "2024-06-10").__dict__,
                  "detection_confidence": 0.3}),
         Event(**{**e("2024-09-01", "2024-09-10").__dict__,
                  "detection_confidence": 0.2})]
    curve = M.operating_curve(t, p, cuts=[0.0, 0.5])
    low, high = curve[0], curve[1]
    assert low["recall"] >= high["recall"]
    assert high["precision"] >= low["precision"]
    assert high["n_pred"] < low["n_pred"]


def test_evaluate_scenario_returns_every_metric_family():
    t = [e("2024-03-01", "2024-03-10")]
    got = M.evaluate_scenario(t, list(t))
    for key in ["event_level", "per_type", "iou", "boundary",
                "accuracy", "day_level", "confusion"]:
        assert key in got
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `No module named 'benchmark.eval.metrics'`.

- [ ] **Step 3: Write `benchmark/eval/metrics.py`**

```python
"""The ten metrics of spec section 9.

Two of them are easy to get quietly wrong and are worth naming here:

- Channel and market accuracy are computed under a RELAXED match that ignores
  type, country and channel. Under the strict match a channel mix-up becomes a
  false negative plus a false positive, the pair never appears as a matched
  couple, and the accuracy reads a meaningless 100%.
- False-positive rate is per country-year, not per scenario, so scenarios of
  different sizes are comparable. It is only measurable at all because the
  benchmark carries 9 null scenarios with no events.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from benchmark.eval.matching import MatchResult, match_events
from benchmark.eval.model import Event


def prf(n_tp: int, n_fp: int, n_fn: int) -> dict:
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def event_level(truth: list[Event], pred: list[Event]) -> dict:
    res = match_events(truth, pred)
    out = prf(res.n_tp, res.n_fp, res.n_fn)
    out.update(n_tp=res.n_tp, n_fp=res.n_fp, n_fn=res.n_fn)
    return out


def per_type(truth: list[Event], pred: list[Event]) -> dict[str, dict]:
    types = sorted({e.event_type for e in truth} | {e.event_type for e in pred})
    return {
        t: event_level([e for e in truth if e.event_type == t],
                       [e for e in pred if e.event_type == t])
        for t in types
    }


def iou_stats(result: MatchResult) -> dict:
    vals = [m.iou for m in result.matches]
    if not vals:
        return {"mean_iou": 0.0, "median_iou": 0.0, "n": 0}
    return {"mean_iou": float(np.mean(vals)),
            "median_iou": float(np.median(vals)),
            "n": len(vals)}


def boundary_error(result: MatchResult) -> dict:
    if not result.matches:
        return {"start_median": 0.0, "start_p90": 0.0,
                "end_median": 0.0, "end_p90": 0.0, "n": 0}
    starts = np.array([abs((m.pred.start - m.truth.start).days)
                       for m in result.matches], dtype=float)
    ends = np.array([abs((m.pred.end - m.truth.end).days)
                     for m in result.matches], dtype=float)
    return {
        "start_median": float(np.median(starts)),
        "start_p90": float(np.percentile(starts, 90)),
        "end_median": float(np.median(ends)),
        "end_p90": float(np.percentile(ends, 90)),
        "n": len(result.matches),
    }


def channel_and_market_accuracy(truth: list[Event], pred: list[Event]) -> dict:
    res = match_events(truth, pred, strict=False)
    if not res.matches:
        return {"channel_accuracy": 0.0, "market_accuracy": 0.0,
                "n_relaxed_matches": 0}
    same_channel = sum(m.truth.channel == m.pred.channel for m in res.matches)
    same_market = sum(m.truth.country_code == m.pred.country_code
                      for m in res.matches)
    n = len(res.matches)
    return {"channel_accuracy": same_channel / n,
            "market_accuracy": same_market / n,
            "n_relaxed_matches": n}


def _day_keys(events: list[Event]) -> set[tuple]:
    """Every (sid, country, channel, day) an event covers."""
    out: set[tuple] = set()
    for e in events:
        for day in range(e.n_days):
            out.add((e.sid, e.country_code, e.channel,
                     (e.start + np.timedelta64(day, "D"))))
    return out


def day_level(truth: list[Event], pred: list[Event]) -> dict:
    t_days = _day_keys(truth)
    p_days = _day_keys(pred)
    tp = len(t_days & p_days)
    return prf(tp, len(p_days - t_days), len(t_days - p_days))


def type_confusion(truth: list[Event], pred: list[Event]) -> dict:
    """Which truth type got called which detector label, over relaxed matches
    that agree on country and channel -- so this isolates TYPE errors rather
    than mixing them with localisation errors."""
    res = match_events(truth, pred, strict=False)
    counts: Counter = Counter()
    for m in res.matches:
        if (m.truth.country_code == m.pred.country_code
                and m.truth.channel == m.pred.channel):
            counts[(m.truth.event_type, m.pred.event_type)] += 1
    return dict(counts)


def reliability_curve(truth: list[Event], pred: list[Event],
                      n_bins: int = 10) -> list[dict]:
    """Spec section 9 item 8: confidence bin against EMPIRICAL precision.

    This is what converts `detection_confidence` from an assertion into a
    testable claim -- "events at confidence 0.9 are correct about 90% of the
    time". Detections without a confidence are skipped rather than assumed
    confident, so an unscored detector yields an empty curve instead of a
    misleadingly perfect one.
    """
    scored = [p for p in pred if p.detection_confidence is not None]
    if not scored:
        return []

    matched = {id(m.pred) for m in match_events(truth, pred).matches}
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    out = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Last bin is closed on the right so confidence 1.0 is counted.
        in_bin = [p for p in scored
                  if lo <= p.detection_confidence < hi
                  or (i == n_bins - 1 and p.detection_confidence == 1.0)]
        if not in_bin:
            continue
        correct = sum(id(p) in matched for p in in_bin)
        out.append({
            "bin_lo": lo, "bin_hi": hi, "n": len(in_bin),
            "mean_confidence": float(np.mean([p.detection_confidence
                                              for p in in_bin])),
            "empirical_precision": correct / len(in_bin),
        })
    return out


def operating_curve(truth: list[Event], pred: list[Event],
                    cuts: list[float] | None = None) -> list[dict]:
    """Spec section 9 item 9: sweep the confidence cut and report P/R at each.

    One operating point is not a result. This is what lets the handover say
    "at cut 0.5 you get P=x R=y; at 0.8, P=x' R=y'" instead of implying the
    detector has a single fixed accuracy.
    """
    if cuts is None:
        cuts = [round(c, 2) for c in np.arange(0.0, 1.0, 0.05)]

    out = []
    for cut in cuts:
        kept = [p for p in pred
                if p.detection_confidence is None
                or p.detection_confidence >= cut]
        scores = event_level(truth, kept)
        out.append({"cut": float(cut), "n_pred": len(kept), **scores})
    return out


def false_positive_rate(pred: list[Event], country_years: float) -> float:
    """Events reported per country-year. Used on the null scenarios, where every
    prediction is by construction a false positive."""
    if country_years <= 0:
        return 0.0
    return len(pred) / country_years


def evaluate_scenario(truth: list[Event], pred: list[Event]) -> dict:
    strict = match_events(truth, pred)
    return {
        "event_level": event_level(truth, pred),
        "reliability": reliability_curve(truth, pred),
        "operating": operating_curve(truth, pred),
        "per_type": per_type(truth, pred),
        "iou": iou_stats(strict),
        "boundary": boundary_error(strict),
        "accuracy": channel_and_market_accuracy(truth, pred),
        "day_level": day_level(truth, pred),
        "confusion": type_confusion(truth, pred),
    }
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_metrics.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval/metrics.py tests/eval/test_metrics.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(eval): add the ten metrics from spec section 9"
```

---

### Task 5: Synthetic detectors and harness validation

**The most important task in this plan.** Everything so far is unverified machinery. These detectors have known-correct scores, so they are the only thing that can prove the harness measures what it claims. In particular, the perfect oracle must score 1.0 *including on pulse scenarios* — that is what catches the grouping bug.

**Files:**
- Create: `benchmark/eval/detectors_for_testing.py`
- Test: `tests/eval/test_harness_validation.py`

**Interfaces:**
- Consumes: `Event` (Task 1), `load_truth` (Task 2), `evaluate_scenario` (Task 4).
- Produces:
  - `DetectorFn = Callable[[pd.DataFrame, pd.DataFrame, str], list[Event]]` — signature `(media_df, sales_df, sid) -> list[Event]`
  - `never_detect`, `detect_everything`, `perfect_oracle`, `shifted_oracle(days)`, `wrong_channel_oracle`, `ungrouped_pulse_oracle`, `half_confident_oracle`
  - Task 6 consumes `DetectorFn`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_harness_validation.py`:

```python
"""Validate the harness against detectors whose correct scores are known in
advance. Without this, a broken metric would silently flatter every later
result and nothing would catch it."""
import pandas as pd
import pytest

from benchmark.eval import detectors_for_testing as D
from benchmark.eval import truth as T
from benchmark.eval.metrics import evaluate_scenario

DEV_WITH_EVENTS = [sid for sid in T.list_scenarios("dev")
                   if T.load_truth("dev", sid)]


def score(sid, detector):
    truth = T.load_truth("dev", sid)
    pred = detector(None, None, sid)
    return evaluate_scenario(truth, pred)


def test_there_are_dev_scenarios_with_events_to_test_against():
    assert len(DEV_WITH_EVENTS) >= 30


@pytest.mark.parametrize("sid", DEV_WITH_EVENTS)
def test_perfect_oracle_scores_exactly_one(sid):
    """THE harness validation. If this fails anywhere, the harness cannot
    recognise a correct detector and every later number is meaningless."""
    got = score(sid, D.perfect_oracle)["event_level"]
    assert got["precision"] == 1.0, sid
    assert got["recall"] == 1.0, sid
    assert got["f1"] == 1.0, sid


def test_perfect_oracle_scores_one_on_pulse_scenarios_specifically():
    """Called out separately because ungrouped pulse truth is the documented
    trap: a perfect detector would score zero on a third of the test events."""
    pulse_sids = [sid for sid in DEV_WITH_EVENTS
                  if any(e.event_type == "channel_pulse"
                         for e in T.load_truth("dev", sid))]
    assert pulse_sids, "no dev pulse scenarios found"
    for sid in pulse_sids:
        got = score(sid, D.perfect_oracle)["per_type"]["channel_pulse"]
        assert got["f1"] == 1.0, sid


def test_ungrouped_pulse_oracle_fails_exactly_as_documented():
    """The negative control for the grouping fix: emitting one event per
    off-window instead of one grouped event must NOT score 1.0. If this passes
    at 1.0, the loader is not grouping and the trap is still live."""
    pulse_sids = [sid for sid in DEV_WITH_EVENTS
                  if any(e.event_type == "channel_pulse"
                         for e in T.load_truth("dev", sid))]
    sid = pulse_sids[0]
    got = score(sid, D.ungrouped_pulse_oracle)["per_type"]["channel_pulse"]
    assert got["f1"] < 1.0, \
        "ungrouped pulses scored perfectly -- grouping is not being exercised"


@pytest.mark.parametrize("sid", DEV_WITH_EVENTS[:10])
def test_never_detect_scores_zero_recall_and_no_false_positives(sid):
    got = score(sid, D.never_detect)["event_level"]
    assert got["recall"] == 0.0 and got["n_fp"] == 0


def test_never_detect_is_perfect_on_null_scenarios():
    """A detector that reports nothing has a flawless false-positive rate. This
    is exactly why precision alone is not a sufficient headline."""
    nulls = [sid for sid in T.list_scenarios("dev")
             if T.load_meta("dev", sid)["family"] == "null"]
    assert nulls
    for sid in nulls:
        assert D.never_detect(None, None, sid) == []


def test_detect_everything_is_heavily_penalised_on_nulls():
    nulls = [sid for sid in T.list_scenarios("dev")
             if T.load_meta("dev", sid)["family"] == "null"]
    sid = nulls[0]
    pred = D.detect_everything(None, None, sid)
    assert len(pred) > 0
    got = evaluate_scenario([], pred)["event_level"]
    assert got["n_fp"] == len(pred)
    assert got["precision"] == 0.0


@pytest.mark.parametrize("shift", [1, 3, 10])
def test_shifted_oracle_degrades_iou_monotonically(shift):
    sid = DEV_WITH_EVENTS[0]
    base = score(sid, D.perfect_oracle)["iou"]["mean_iou"]
    got = score(sid, D.shifted_oracle(shift))["iou"]["mean_iou"]
    assert got <= base


def test_shifted_oracle_shows_up_in_boundary_error():
    sid = DEV_WITH_EVENTS[0]
    got = score(sid, D.shifted_oracle(3))["boundary"]
    assert got["start_median"] == 3.0


def test_perfect_oracle_produces_a_perfectly_calibrated_reliability_curve():
    """Everything in the top bin, and everything in it correct."""
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.perfect_oracle)["reliability"]
    assert curve, "oracle produced no reliability curve"
    assert all(b["empirical_precision"] == 1.0 for b in curve)
    assert curve[-1]["bin_hi"] == 1.0


def test_half_confident_oracle_is_visibly_miscalibrated():
    """Low-confidence detections here are just as correct as high-confidence
    ones, so the low bin must show high empirical precision. If the curve just
    echoed the confidence values back, this would not show up."""
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.half_confident_oracle)["reliability"]
    low = [b for b in curve if b["bin_hi"] <= 0.3]
    assert low, "no low-confidence bin produced"
    assert all(b["empirical_precision"] == 1.0 for b in low)


def test_operating_curve_keeps_everything_at_cut_zero():
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.perfect_oracle)["operating"]
    at_zero = [c for c in curve if c["cut"] == 0.0][0]
    assert at_zero["recall"] == 1.0


def test_operating_curve_drops_recall_as_the_cut_rises():
    sid = DEV_WITH_EVENTS[0]
    curve = score(sid, D.half_confident_oracle)["operating"]
    recalls = [c["recall"] for c in sorted(curve, key=lambda c: c["cut"])]
    assert recalls[0] >= recalls[-1]


def test_wrong_channel_oracle_is_caught_by_channel_accuracy_not_by_f1():
    """The whole reason accuracy uses a relaxed match: under the strict match
    this detector's errors become FN+FP and channel accuracy would read 100%."""
    sid = next(s for s in DEV_WITH_EVENTS
               if any(e.channel for e in T.load_truth("dev", s)))
    got = score(sid, D.wrong_channel_oracle)
    assert got["accuracy"]["channel_accuracy"] < 1.0
    assert got["accuracy"]["market_accuracy"] == 1.0
    assert got["event_level"]["f1"] < 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_harness_validation.py -v`
Expected: FAIL — `No module named 'benchmark.eval.detectors_for_testing'`.

- [ ] **Step 3: Write `benchmark/eval/detectors_for_testing.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_harness_validation.py -v`
Expected: all pass. The parametrized oracle test runs once per dev scenario with events, so expect roughly 40–50 test cases in total.

If `test_perfect_oracle_scores_exactly_one` fails on any scenario, **stop and investigate the harness, not the test.** That test failing means the harness cannot recognise a correct detector, which invalidates everything downstream. Report the failing sids and what the mismatch was.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval/detectors_for_testing.py tests/eval/test_harness_validation.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "test(eval): validate the harness against detectors with known scores"
```

---

### Task 6: Split runner, breakdowns, and report rendering

**Files:**
- Create: `benchmark/eval/runner.py`, `benchmark/eval/breakdowns.py`, `benchmark/eval/report.py`
- Test: `tests/eval/test_runner.py`, `tests/eval/test_breakdowns.py`

**Interfaces:**
- Consumes: `DetectorFn` (Task 5), `load_truth`/`load_meta`/`list_scenarios` (Task 2), `evaluate_scenario` (Task 4).
- Produces:
  - `evaluate_split(detector, split, root=None, sids=None, load_data=False) -> dict` with keys `overall`, `per_type`, `iou`, `boundary`, `accuracy`, `day_level`, `confusion`, `reliability`, `operating`, `null_fp_rate`, `per_scenario`, `breakdowns`, `n_scenarios`
  - `breakdown_by(per_scenario, meta_by_sid, axis) -> dict`
  - `CONFOUNDED_AXES: frozenset[str]`
  - `render_markdown(results) -> str`
  - Tasks 7 and 8 consume `evaluate_split` and `render_markdown`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_runner.py`:

```python
from benchmark.eval import detectors_for_testing as D
from benchmark.eval import truth as T
from benchmark.eval.report import render_markdown
from benchmark.eval.runner import evaluate_split

SOME = T.list_scenarios("dev")[:8]


def test_evaluate_split_covers_every_requested_scenario():
    got = evaluate_split(D.never_detect, "dev", sids=SOME)
    assert got["n_scenarios"] == len(SOME)
    assert set(got["per_scenario"]) == set(SOME)


def test_perfect_oracle_scores_one_over_a_whole_split():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    assert got["overall"]["f1"] == 1.0


def test_never_detect_has_a_zero_false_positive_rate_on_nulls():
    got = evaluate_split(D.never_detect, "dev")
    assert got["null_fp_rate"] == 0.0


def test_detect_everything_has_a_nonzero_false_positive_rate_on_nulls():
    got = evaluate_split(D.detect_everything, "dev")
    assert got["null_fp_rate"] > 0.0


def test_breakdowns_are_present_for_the_interpretable_axis():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    assert "noise_level" in got["breakdowns"]


def test_report_renders_without_crashing_and_names_the_headline_numbers():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    md = render_markdown(got)
    assert "Precision" in md and "Recall" in md and "F1" in md
    assert "false positive" in md.lower()
    assert "Reliability" in md and "Operating curve" in md


def test_report_says_so_plainly_when_nothing_carries_a_confidence():
    got = evaluate_split(D.never_detect, "dev", sids=SOME)
    md = render_markdown(got)
    assert "calibration cannot be assessed" in md.lower()
```

Create `tests/eval/test_breakdowns.py`:

```python
from benchmark.eval.breakdowns import CONFOUNDED_AXES, breakdown_by


def test_confounded_axes_are_named_exactly():
    """BENCHMARK.md: only noise_level is stratified within family. trend_p and
    market_spread are family-confounded on BOTH splits and must not be reported
    as axis effects."""
    assert CONFOUNDED_AXES == frozenset({"trend_p", "market_spread"})
    assert "noise_level" not in CONFOUNDED_AXES


def test_breakdown_groups_scenarios_by_axis_value():
    per_scenario = {
        "dev_001": {"event_level": {"n_tp": 1, "n_fp": 0, "n_fn": 0}},
        "dev_002": {"event_level": {"n_tp": 0, "n_fp": 1, "n_fn": 1}},
    }
    meta = {"dev_001": {"noise_level": "low"}, "dev_002": {"noise_level": "high"}}
    got = breakdown_by(per_scenario, meta, "noise_level")
    assert got["low"]["f1"] == 1.0
    assert got["high"]["f1"] == 0.0
    assert got["low"]["n_scenarios"] == 1


def test_confounded_axis_carries_a_warning_in_its_result():
    per_scenario = {"dev_001": {"event_level": {"n_tp": 1, "n_fp": 0, "n_fn": 0}}}
    meta = {"dev_001": {"trend_p": 0.0}}
    got = breakdown_by(per_scenario, meta, "trend_p")
    assert got["_warning"], "confounded axis reported without a warning"
    assert "confounded" in got["_warning"].lower()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_runner.py tests/eval/test_breakdowns.py -v`
Expected: FAIL — missing modules.

- [ ] **Step 3: Write `benchmark/eval/breakdowns.py`**

```python
"""Per-axis slices of a split's results.

Only `noise_level` is stratified within family, so it is the only axis whose
breakdown is causally interpretable. `trend_p` and `market_spread` are
confounded with family on BOTH splits -- 8 of 10 dev families and 7 of 10 test
families sit at a single trend level, and dev's trend=1.0 bucket contains no
null scenarios at all, so precision there has no false-positive denominator.
Reporting those as axis effects would be reporting family difficulty under
another name, so they carry a warning that travels with the numbers.
"""
from __future__ import annotations

from collections import defaultdict

from benchmark.eval.metrics import prf

CONFOUNDED_AXES: frozenset[str] = frozenset({"trend_p", "market_spread"})

_WARNING = (
    "CONFOUNDED WITH FAMILY on both splits -- most families sit at a single "
    "value of this axis, so differences here measure family difficulty, not "
    "the axis. Do not report as an axis effect. See BENCHMARK.md."
)


def breakdown_by(per_scenario: dict, meta_by_sid: dict, axis: str) -> dict:
    buckets: dict = defaultdict(lambda: {"n_tp": 0, "n_fp": 0, "n_fn": 0,
                                         "n_scenarios": 0})
    for sid, result in per_scenario.items():
        meta = meta_by_sid.get(sid)
        if meta is None or axis not in meta:
            continue
        b = buckets[meta[axis]]
        ev = result["event_level"]
        b["n_tp"] += ev["n_tp"]
        b["n_fp"] += ev["n_fp"]
        b["n_fn"] += ev["n_fn"]
        b["n_scenarios"] += 1

    out = {}
    for value, b in buckets.items():
        entry = dict(b)
        entry.update(prf(b["n_tp"], b["n_fp"], b["n_fn"]))
        out[value] = entry
    out["_warning"] = _WARNING if axis in CONFOUNDED_AXES else ""
    return out
```

- [ ] **Step 4: Write `benchmark/eval/runner.py`**

```python
"""Run a detector over a split and aggregate the results."""
from __future__ import annotations

from pathlib import Path

from benchmark.eval import truth as T
from benchmark.eval.breakdowns import breakdown_by
from benchmark.eval.matching import match_events
from benchmark.eval.metrics import evaluate_scenario, prf
from benchmark.eval.model import Event

BREAKDOWN_AXES = ("noise_level", "family", "n_countries", "n_channels",
                  "years", "trend_p", "market_spread")


def _load_frames(split: str, sid: str, root: Path | None):
    """A detector reads ONLY these two files. Loaded lazily so the oracles,
    which ignore them, do not pay for 157 MB of CSV parsing."""
    import pandas as pd
    from benchmark.harness.runner import DATASETS_DIR, dataset_dir
    d = dataset_dir(split, sid, root or DATASETS_DIR)
    media = pd.read_csv(d / "media.csv", parse_dates=["date"])
    sales = pd.read_csv(d / "sales.csv", parse_dates=["date"])
    return media, sales


def evaluate_split(detector, split: str, root: Path | None = None,
                   sids: list[str] | None = None,
                   load_data: bool = False) -> dict:
    sids = list(sids or T.list_scenarios(split, root))

    per_scenario: dict = {}
    meta_by_sid: dict = {}
    all_truth: list[Event] = []
    all_pred: list[Event] = []
    null_preds = 0
    null_country_years = 0.0

    for sid in sids:
        meta = T.load_meta(split, sid, root)
        meta_by_sid[sid] = meta
        truth = T.load_truth(split, sid, root)

        media, sales = (_load_frames(split, sid, root)
                        if load_data else (None, None))
        pred = list(detector(media, sales, sid))

        per_scenario[sid] = evaluate_scenario(truth, pred)
        all_truth.extend(truth)
        all_pred.extend(pred)

        if meta["family"] == "null":
            null_preds += len(pred)
            null_country_years += meta["n_countries"] * meta["years"]

    overall_match = match_events(all_truth, all_pred)
    overall = prf(overall_match.n_tp, overall_match.n_fp, overall_match.n_fn)
    overall.update(n_tp=overall_match.n_tp, n_fp=overall_match.n_fp,
                   n_fn=overall_match.n_fn)

    from benchmark.eval.metrics import (boundary_error, channel_and_market_accuracy,
                                        day_level, iou_stats, operating_curve,
                                        per_type, reliability_curve,
                                        type_confusion)

    return {
        "split": split,
        "n_scenarios": len(sids),
        "overall": overall,
        "per_type": per_type(all_truth, all_pred),
        "iou": iou_stats(overall_match),
        "boundary": boundary_error(overall_match),
        "accuracy": channel_and_market_accuracy(all_truth, all_pred),
        "day_level": day_level(all_truth, all_pred),
        "confusion": type_confusion(all_truth, all_pred),
        "reliability": reliability_curve(all_truth, all_pred),
        "operating": operating_curve(all_truth, all_pred),
        "null_fp_rate": (null_preds / null_country_years
                         if null_country_years else 0.0),
        "null_country_years": null_country_years,
        "per_scenario": per_scenario,
        "breakdowns": {ax: breakdown_by(per_scenario, meta_by_sid, ax)
                       for ax in BREAKDOWN_AXES},
    }
```

- [ ] **Step 5: Write `benchmark/eval/report.py`**

```python
"""Render an evaluation result as readable Markdown."""
from __future__ import annotations

from benchmark.eval.breakdowns import CONFOUNDED_AXES


def _pct(x: float) -> str:
    return f"{x:.3f}"


def render_markdown(results: dict) -> str:
    o = results["overall"]
    lines = [
        f"# Evaluation — {results['split']} split",
        "",
        f"{results['n_scenarios']} scenarios.",
        "",
        "## Headline",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Precision | {_pct(o['precision'])} |",
        f"| Recall | {_pct(o['recall'])} |",
        f"| F1 | {_pct(o['f1'])} |",
        f"| TP / FP / FN | {o['n_tp']} / {o['n_fp']} / {o['n_fn']} |",
        f"| Mean IoU | {_pct(results['iou']['mean_iou'])} |",
        f"| Median IoU | {_pct(results['iou']['median_iou'])} |",
        f"| False positives per country-year (null scenarios) "
        f"| {results['null_fp_rate']:.3f} |",
        f"| Channel accuracy (relaxed match) "
        f"| {_pct(results['accuracy']['channel_accuracy'])} |",
        f"| Market accuracy (relaxed match) "
        f"| {_pct(results['accuracy']['market_accuracy'])} |",
        f"| Day-level F1 | {_pct(results['day_level']['f1'])} |",
        "",
        "## Boundary error (days)",
        "",
        "| | median | p90 |",
        "|---|---|---|",
        f"| start | {results['boundary']['start_median']:.1f} "
        f"| {results['boundary']['start_p90']:.1f} |",
        f"| end | {results['boundary']['end_median']:.1f} "
        f"| {results['boundary']['end_p90']:.1f} |",
        "",
        "## Per event type",
        "",
        "| type | precision | recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, m in sorted(results["per_type"].items()):
        lines.append(
            f"| {t} | {_pct(m['precision'])} | {_pct(m['recall'])} "
            f"| {_pct(m['f1'])} | {m['n_tp']} | {m['n_fp']} | {m['n_fn']} |")

    if results["confusion"]:
        lines += ["", "## Type confusion (truth → predicted)", "",
                  "| truth | predicted | n |", "|---|---|---|"]
        for (a, b), n in sorted(results["confusion"].items()):
            flag = "" if a == b else "  ← substitution"
            lines.append(f"| {a} | {b}{flag} | {n} |")

    if results.get("reliability"):
        lines += ["", "## Reliability — is the confidence score honest?", "",
                  "| confidence bin | n | mean confidence | empirical precision |",
                  "|---|---|---|---|"]
        for b in results["reliability"]:
            lines.append(
                f"| {b['bin_lo']:.1f}–{b['bin_hi']:.1f} | {b['n']} "
                f"| {_pct(b['mean_confidence'])} "
                f"| {_pct(b['empirical_precision'])} |")
    else:
        lines += ["", "## Reliability", "",
                  "_No detection carried a confidence score, so calibration "
                  "cannot be assessed._"]

    if results.get("operating"):
        lines += ["", "## Operating curve — the precision/recall trade-off", "",
                  "| confidence cut | detections kept | precision | recall | F1 |",
                  "|---|---|---|---|---|"]
        for c in results["operating"]:
            lines.append(
                f"| {c['cut']:.2f} | {c['n_pred']} | {_pct(c['precision'])} "
                f"| {_pct(c['recall'])} | {_pct(c['f1'])} |")

    lines += ["", "## Breakdowns", ""]
    for axis, data in results["breakdowns"].items():
        warning = data.get("_warning", "")
        lines.append(f"### {axis}" + (" ⚠️" if axis in CONFOUNDED_AXES else ""))
        lines.append("")
        if warning:
            lines += [f"> **{warning}**", ""]
        lines += ["| value | scenarios | precision | recall | F1 |",
                  "|---|---|---|---|---|"]
        for value, m in sorted((k, v) for k, v in data.items()
                               if k != "_warning"):
            lines.append(
                f"| {value} | {m['n_scenarios']} | {_pct(m['precision'])} "
                f"| {_pct(m['recall'])} | {_pct(m['f1'])} |")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 6: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_runner.py tests/eval/test_breakdowns.py -v`
Expected: 11 passed.

- [ ] **Step 7: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval/runner.py benchmark/eval/breakdowns.py benchmark/eval/report.py \
        tests/eval/test_runner.py tests/eval/test_breakdowns.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(eval): add split runner, confound-aware breakdowns, and report rendering"
```

---

### Task 7: The two runners and the black-box gate

**Files:**
- Create: `benchmark/eval/run_dev.py`, `benchmark/eval/run_final.py`
- Test: `tests/eval/test_gating.py`

**Interfaces:**
- Consumes: `evaluate_split`, `render_markdown` (Task 6); `seal.verify_seal` (Plan 1).
- Produces:
  - `python -m benchmark.eval.run_dev --detector <dotted.path> [--out PATH] [--label TEXT]`
  - `python -m benchmark.eval.run_final --detector <dotted.path> --finalize [--out PATH]`
  - `benchmark/eval/dev_history.jsonl`, `benchmark/eval/final_runs.jsonl`
  - `source_hash(path) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_gating.py`:

```python
import json
import subprocess
from pathlib import Path

PY = "synthetic_data_generator/.venv/bin/python"
ROOT = Path(__file__).resolve().parents[2]


def run(module, *args):
    return subprocess.run([PY, "-m", module, *args], cwd=ROOT,
                          capture_output=True, text=True)


def test_final_refuses_without_the_finalize_flag():
    """The gate. Running the sealed test split must be a deliberate act."""
    p = run("benchmark.eval.run_final",
            "--detector", "benchmark.eval.detectors_for_testing:never_detect")
    assert p.returncode != 0
    assert "--finalize" in (p.stdout + p.stderr)


def test_final_names_the_seal_check_in_its_help():
    p = run("benchmark.eval.run_final", "--help")
    assert p.returncode == 0
    assert "seal" in (p.stdout + p.stderr).lower()


def test_dev_runs_freely_and_appends_history():
    hist_before = 0
    hist = ROOT / "benchmark/eval/dev_history.jsonl"
    if hist.is_file():
        hist_before = len(hist.read_text().splitlines())

    p = run("benchmark.eval.run_dev",
            "--detector", "benchmark.eval.detectors_for_testing:perfect_oracle",
            "--limit", "5", "--label", "harness-selftest")
    assert p.returncode == 0, p.stderr
    assert "f1" in (p.stdout.lower())

    after = len(hist.read_text().splitlines())
    assert after == hist_before + 1
    last = json.loads(hist.read_text().splitlines()[-1])
    for key in ["timestamp", "label", "detector", "source_hash", "metrics"]:
        assert key in last
    assert last["metrics"]["f1"] == 1.0


def test_detection_package_has_no_import_path_to_truth():
    """Spec section 4: detector code must never be able to import a truth
    loader. detection/ does not exist yet -- when Plan 3 creates it, this test
    is what keeps the boundary real."""
    det = ROOT / "detection"
    if not det.is_dir():
        return
    offenders = []
    for py in det.rglob("*.py"):
        text = py.read_text()
        if "benchmark.eval" in text or "_truth" in text or "ground_truth" in text:
            offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, f"detection/ reaches for truth: {offenders}"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_gating.py -v`
Expected: FAIL — `No module named benchmark.eval.run_final`.

- [ ] **Step 3: Write `benchmark/eval/run_dev.py`**

```python
"""Evaluate a detector against the DEVELOPMENT split. Free to run, as often as
you like -- that is what the dev split is for.

Every run appends to dev_history.jsonl, which becomes the development
trajectory in the final report: evidence that tuning happened where it was
permitted, and only there.

    python -m benchmark.eval.run_dev --detector module.path:function
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path

from benchmark.eval import truth as T
from benchmark.eval.report import render_markdown
from benchmark.eval.runner import evaluate_split

HISTORY = Path(__file__).resolve().parent / "dev_history.jsonl"


def source_hash(path: Path) -> str:
    """SHA-256 over every .py under `path`, so a report can name the exact code
    that produced it."""
    h = hashlib.sha256()
    if not path.is_dir():
        return "absent"
    for py in sorted(path.rglob("*.py")):
        h.update(py.relative_to(path).as_posix().encode())
        h.update(py.read_bytes())
    return h.hexdigest()


def load_detector(spec: str):
    """`module.path:function` -> the callable."""
    if ":" not in spec:
        raise SystemExit(f"--detector must be module.path:function, got {spec!r}")
    mod_name, func_name = spec.split(":", 1)
    return getattr(importlib.import_module(mod_name), func_name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True,
                   help="module.path:function returning list[Event]")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N scenarios")
    p.add_argument("--label", default="",
                   help="short note recorded in dev_history.jsonl")
    p.add_argument("--out", default=None, help="write the Markdown report here")
    p.add_argument("--load-data", action="store_true",
                   help="pass media/sales frames to the detector")
    args = p.parse_args(argv)

    detector = load_detector(args.detector)
    sids = T.list_scenarios("dev")
    if args.limit:
        sids = sids[: args.limit]

    results = evaluate_split(detector, "dev", sids=sids,
                             load_data=args.load_data)
    report = render_markdown(results)
    print(report)

    if args.out:
        Path(args.out).write_text(report)

    o = results["overall"]
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "detector": args.detector,
        "n_scenarios": results["n_scenarios"],
        "source_hash": source_hash(Path(__file__).resolve().parents[2] / "detection"),
        "metrics": {"precision": o["precision"], "recall": o["recall"],
                    "f1": o["f1"], "mean_iou": results["iou"]["mean_iou"],
                    "null_fp_rate": results["null_fp_rate"]},
    }
    with HISTORY.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"\nAppended to {HISTORY.name} "
          f"(f1={o['f1']:.3f}, label={args.label or '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write `benchmark/eval/run_final.py`**

```python
"""Evaluate a detector against the SEALED HOLD-OUT TEST SPLIT.

This is the one sanctioned reader of test-split ground truth, and running it is
meant to be a deliberate, recorded act rather than a convenience. It therefore:

  * refuses to run without an explicit --finalize flag;
  * verifies the test split's seal before reading anything, so a tampered or
    regenerated benchmark cannot quietly produce a number;
  * appends an immutable record to final_runs.jsonl -- timestamp, the SHA-256
    of the entire detection/ tree, the sealed spec_hash, and the metrics.

Run it ONCE, after the algorithms are final. A second run is possible but is
permanently visible in final_runs.jsonl, and the report is expected to say so.

    python -m benchmark.eval.run_final --detector module.path:function --finalize
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from benchmark.eval import truth as T
from benchmark.eval.report import render_markdown
from benchmark.eval.run_dev import load_detector, source_hash
from benchmark.eval.runner import evaluate_split
from benchmark.harness import seal

FINAL_RUNS = Path(__file__).resolve().parent / "final_runs.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector", required=True,
                   help="module.path:function returning list[Event]")
    p.add_argument("--finalize", action="store_true",
                   help="required. Confirms the algorithms are frozen and you "
                        "intend to read the sealed test split.")
    p.add_argument("--out", default=None, help="write the Markdown report here")
    p.add_argument("--load-data", action="store_true",
                   help="pass media/sales frames to the detector")
    args = p.parse_args(argv)

    if not args.finalize:
        print("ERROR: run_final reads the SEALED hold-out test split. Pass "
              "--finalize to confirm the algorithms are frozen.\n"
              "Use `python -m benchmark.eval.run_dev` while iterating.",
              file=sys.stderr)
        return 2

    ok, problems = seal.verify_seal("test")
    if not ok:
        print("ERROR: the test split's seal does not verify. Refusing to "
              "produce a number from a benchmark that has changed:",
              file=sys.stderr)
        for problem in problems[:10]:
            print(f"  {problem}", file=sys.stderr)
        return 3

    marker = json.loads(
        (PROJECT_ROOT / "benchmark/datasets/test/SEALED").read_text())

    if FINAL_RUNS.is_file() and FINAL_RUNS.read_text().strip():
        n = len(FINAL_RUNS.read_text().splitlines())
        print(f"*** NOTE: final_runs.jsonl already has {n} record(s). "
              f"This is not the first final evaluation, and the report must "
              f"say so. ***\n", file=sys.stderr)

    results = evaluate_split(load_detector(args.detector), "test",
                             load_data=args.load_data)
    report = render_markdown(results)
    print(report)
    if args.out:
        Path(args.out).write_text(report)

    o = results["overall"]
    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "detector": args.detector,
        "detection_source_hash": source_hash(PROJECT_ROOT / "detection"),
        "sealed_spec_hash": marker["spec_hash"],
        "sealed_at": marker["sealed_at"],
        "n_scenarios": results["n_scenarios"],
        "metrics": {"precision": o["precision"], "recall": o["recall"],
                    "f1": o["f1"], "mean_iou": results["iou"]["mean_iou"],
                    "null_fp_rate": results["null_fp_rate"]},
    }
    with FINAL_RUNS.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"\nRecorded in {FINAL_RUNS.name}. detection/ hash "
          f"{record['detection_source_hash'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_gating.py -v`
Expected: 4 passed.

**Do NOT run `run_final.py` with `--finalize` during this plan.** Its first real execution belongs at the end of Plan 3, once the detectors are frozen. The test above only checks that it refuses without the flag.

- [ ] **Step 6: Verify the seal is still intact**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -c "
from benchmark.harness import seal
ok, p = seal.verify_seal('test')
print('SEAL OK' if ok else f'SEAL BROKEN: {p[:3]}')"
```
Expected: `SEAL OK`.

- [ ] **Step 7: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval/run_dev.py benchmark/eval/run_final.py tests/eval/test_gating.py
git add -f benchmark/eval/dev_history.jsonl 2>/dev/null || true
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(eval): add the free dev runner and the gated final runner"
```

---

### Task 8: Harness documentation and a baseline record

**Files:**
- Create: `benchmark/eval/README.md`
- Modify: `benchmark/BENCHMARK.md` (add a pointer to the harness)
- Test: `tests/eval/test_baseline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: a documented, recorded baseline that Plan 3 measures against.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_baseline.py`:

```python
"""Standing guarantees about the harness itself. These run on every suite
invocation, so a later change that breaks the harness's ability to recognise a
correct detector fails immediately rather than silently flattering Plan 3."""
from benchmark.eval import detectors_for_testing as D
from benchmark.eval.runner import evaluate_split


def test_perfect_oracle_still_scores_one_over_the_whole_dev_split():
    got = evaluate_split(D.perfect_oracle, "dev")
    assert got["overall"]["precision"] == 1.0
    assert got["overall"]["recall"] == 1.0
    assert got["overall"]["f1"] == 1.0
    assert got["iou"]["mean_iou"] == 1.0
    assert got["accuracy"]["channel_accuracy"] == 1.0
    assert got["accuracy"]["market_accuracy"] == 1.0
    assert got["day_level"]["f1"] == 1.0


def test_never_detect_is_the_floor():
    got = evaluate_split(D.never_detect, "dev")
    assert got["overall"]["recall"] == 0.0
    assert got["overall"]["n_fp"] == 0
    assert got["null_fp_rate"] == 0.0


def test_detect_everything_is_punished_where_it_should_be():
    got = evaluate_split(D.detect_everything, "dev")
    assert got["overall"]["precision"] < 0.2
    assert got["null_fp_rate"] > 0.0
```

- [ ] **Step 2: Run it**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_baseline.py -v`
Expected: 3 passed. If the first fails, the harness cannot recognise a perfect detector over the full split — stop and investigate.

- [ ] **Step 3: Record the three baselines**

```bash
cd /Users/user123/Desktop/projects/datascience-project
for d in perfect_oracle never_detect detect_everything; do
  echo "=== $d ==="
  synthetic_data_generator/.venv/bin/python -m benchmark.eval.run_dev \
    --detector "benchmark.eval.detectors_for_testing:$d" \
    --label "baseline-$d" 2>&1 | grep -E "^\| (Precision|Recall|F1|False positives)"
done
```
Record the output in your report. These are the numbers Plan 3's detectors must beat.

- [ ] **Step 4: Write `benchmark/eval/README.md`**

```markdown
# The Evaluation Harness

Scores a detector against the frozen benchmark. Built and validated **before**
any real detection algorithm existed, so a broken metric could not silently
flatter the results it was later used to judge.

## Running it

```bash
# Free. Run as often as you like; every run appends to dev_history.jsonl.
python -m benchmark.eval.run_dev --detector detection.pipeline:run_detection

# The sealed hold-out split. Deliberate, gated, recorded. Run ONCE.
python -m benchmark.eval.run_final --detector detection.pipeline:run_detection --finalize
```

A detector is any callable `(media_df, sales_df, sid) -> list[Event]`.

## What the harness guarantees

`tests/eval/test_baseline.py` asserts, on every suite run, that a perfect
oracle scores exactly 1.0 across precision, recall, F1, IoU, channel accuracy,
market accuracy and day-level F1 over the whole dev split. If that ever breaks,
the harness can no longer recognise a correct detector and nothing it reports
can be trusted.

Three synthetic detectors define the range:

| detector | what it proves |
|---|---|
| `perfect_oracle` | the harness recognises a correct detector |
| `never_detect` | the floor, and that a do-nothing detector still has a flawless false-positive rate — which is why FP rate alone is not a sufficient headline |
| `detect_everything` | brute-force recall is destroyed by precision and by the null-scenario FP rate |
| `ungrouped_pulse_oracle` | the pulse-grouping reconciliation is actually exercised |
| `shifted_oracle(n)` | IoU and boundary error respond to localisation error |
| `wrong_channel_oracle` | channel errors surface in channel accuracy rather than vanishing into FN+FP |

## Three things the loader reconciles

Ground-truth shape and detector output shape were fixed independently — by the
generator and by spec §7 — and they do not agree. `benchmark/eval/truth.py` is
where they are brought together, and `BENCHMARK.md` documents why:

1. **Pulse trains** are one truth row per off-window but one detector event.
   Ungrouped, a perfect pulse detector scores zero on a third of the test split.
2. **`global_pause`** is a truth `pattern_type` but a detector *tag* on a
   `dark_period` regime.
3. **`staggered_launch`** truth is per-market; the detector emits a panel event.
   Convention chosen here: fan out to one event per market.

Interval ends come from `scenario.json`'s `end_day`, never the CSV's
`end_date`, which is slice-exclusive except where it is clamped at the series
end.

## Which breakdowns to trust

Only `noise_level` is stratified within family. `trend_p` and `market_spread`
are confounded with family on both splits, and `breakdown_by` attaches a
warning to them that travels with the numbers. Reporting them as axis effects
would be reporting family difficulty under another name.

## The black-box gate

`run_final.py` refuses without `--finalize`, verifies the test split's seal
before reading anything, and appends the SHA-256 of the entire `detection/`
tree plus the sealed `spec_hash` to `final_runs.jsonl`. A second final run is
possible but permanently visible there.
```

- [ ] **Step 5: Add a pointer in `benchmark/BENCHMARK.md`**

Append this section to the end of `benchmark/BENCHMARK.md`:

```markdown
## The evaluation harness

`benchmark/eval/` implements every reconciliation this document describes, and
`benchmark/eval/README.md` explains how to run it. The loader traps above are
not advisory — `tests/eval/` asserts on every suite run that a perfect oracle
scores exactly 1.0 over the whole dev split, which is only true if all three
reconciliations are applied.
```

- [ ] **Step 6: Run the full fast suite**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests -m "not slow" -q`
Expected: all green, with the eval tests added to Plan 1's 110.

- [ ] **Step 7: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add benchmark/eval/README.md benchmark/BENCHMARK.md tests/eval/test_baseline.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "docs(eval): document the harness and record the baseline detectors"
```

---

## Verification

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m pytest tests -m "not slow" -q
synthetic_data_generator/.venv/bin/python -c "
from benchmark.harness import seal
ok, p = seal.verify_seal('test')
print('SEAL OK' if ok else f'SEAL BROKEN: {p}')"
git status --short
```

Expected: every test passes, `SEAL OK`, clean tree.

## What Plan 3 picks up

Plan 3 builds `detection/` — the panel loader and normalization, the four
primitives, regime composition, the cross-market layer, the three scores with
PAV calibration, and the explanation strings. It iterates against
`run_dev.py` only, logging each change to `dev_history.jsonl`, then freezes and
runs `run_final.py --finalize` exactly once before writing `REPORT.md`.
