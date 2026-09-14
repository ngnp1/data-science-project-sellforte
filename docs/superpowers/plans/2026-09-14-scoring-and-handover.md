# Scoring, Calibration, Explanation and Handover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every detected event an interpretable confidence score, an informativeness rank, a validity verdict and a written explanation; calibrate confidence on the development split; then run the sealed benchmark exactly once and write the handover report.

**Architecture:** Three new pure modules under `detection/` — `score.py` (sub-scores → `detection_confidence` and `informativeness`), `validity.py` (the data-quality gate), `explain.py` (templated prose) — plus `calibrate.py`, a hand-rolled pool-adjacent-violators isotonic fit whose output is frozen into a generated Python module. `pipeline.run_detection` composes them after the cross-market layer, so scoring sees finished events. The harness adapter, which already has `detection_confidence` and `informativeness` fields waiting on `Event`, stops dropping them.

**Tech Stack:** Python 3, pandas, numpy. No scipy, no sklearn — the isotonic regression is ~20 lines and writing it keeps the dependency list at two.

**Spec:** `docs/superpowers/specs/2026-09-13-informative-periods-detection-design.md` — sections 8 (scoring and explanation), 9 items 8–9 (reliability and operating curves), 10 (iteration protocol), 11 (handover).

## Global Constraints

- Python is ALWAYS `synthetic_data_generator/.venv/bin/python`. Never bare `python3`.
- Pure pandas/numpy. No scipy, sklearn or ruptures. PAV is hand-implemented.
- No file under `detection/` may contain `benchmark.eval`, `benchmark.spec`, `benchmark.harness`, `_truth` or `ground_truth` — in code, comments or docstrings. `tests/eval/test_gating.py` scans **every file** under `detection/`, not just `*.py`.
- `detection/` may contain **nothing but `.py` source**. `test_the_detection_package_contains_only_source` fails on any JSON, CSV or Parquet file there. A frozen calibration therefore ships as a generated Python module, never as a data file.
- Every threshold lives in `detection/params.py`. `tests/detection/test_params.py` does a naive substring scan for each float parameter's string form across `detection/` — writing `0.58` in a docstring trips the `0.5` needle. Keep float literals out of `detection/` entirely, prose included.
- NEVER read anything under `benchmark/datasets/test_truth/`. Never modify anything under `benchmark/datasets/`. `verify_seal("test")` must return `True` at the end of every task.
- `benchmark/eval/run_final.py --finalize` runs EXACTLY ONCE, in Task 11, after the algorithms are frozen. `benchmark/eval/final_runs.jsonl` must not exist before that moment. Every run is permanently recorded and a second run is banner-flagged forever.
- Iterate only with `benchmark/eval/run_dev.py`. It appends to `dev_history.jsonl`, which is the audit trail and goes into the report.
- Commit with `git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com"`. The user is sole author: no `Co-Authored-By`, no `Claude-Session` trailers.
- Work continues on branch `detection-library`. Do not merge to `main`.

## Testing discipline

This branch's dominant defect, caught eight times: **a test that passes for the wrong reason.** Recorded instances — a filter's named test whose input died at an earlier gate; fixtures sized `MIN_DAYS - 1` and `PERSIST - 4`, so the input moved with the constant under test; an unfalsifiable sort test; a guard patching `pathlib.Path.open` while pandas reads through `builtins.open`; two unreachable gates.

Rules, binding on every task:

1. Verify a filter by **DELETING** the code under test, never by zeroing its constant. Zeroing relocates a check rather than disabling it — that is how the persistence gate sat uncovered through a whole mutation battery that reported it healthy.
2. Clear bytecode before every mutation run: `find detection tests -name __pycache__ -type d -exec rm -rf {} +`. A stale `.pyc` faked a survivor once.
3. Never size a test input from the constant it tests. Use a literal and assert the constant sits on the expected side.
4. Every task's report carries a mutation table. A mutation with no failing test is disclosed, not quietly patched with a weak test.

## File structure

| File | Responsibility |
|---|---|
| `detection/model.py` (modify) | `DetectedEvent` gains `detection_confidence`, `informativeness`, `validity`, `validity_reasons` |
| `detection/params.py` (modify) | Sub-score weights per event type, informativeness weights, type priors, calibration bin count |
| `detection/score.py` (create) | Six confidence sub-scores, seven informativeness drivers, the two weighted means |
| `detection/validity.py` (create) | `ok` / `suspect_data_gap` / `suspect_tracking_loss` with reasons |
| `detection/calibrate.py` (create) | PAV isotonic fit, `apply`, and codegen for the frozen module |
| `detection/calibration_fit.py` (generated, Task 10) | The frozen dev-fitted mapping as Python literals |
| `detection/explain.py` (create) | Templated per-event prose from evidence |
| `detection/pipeline.py` (modify) | Compose scoring, validity, calibration, explanation |
| `benchmark/eval/adapter.py` (modify) | Carry the two scores across into `Event` |
| `REPORT.md` (create, Task 12) | The handover |

---

### Task 1: `DetectedEvent` carries scores

**Files:**
- Modify: `detection/model.py`
- Test: `tests/detection/test_model.py`

**Interfaces:**
- Consumes: existing frozen `DetectedEvent(sid, country_code, channel, event_type, start, end, magnitude_ratio=None, components=(), tags=(), evidence={})`
- Produces: the same dataclass with four new optional fields — `detection_confidence: float | None`, `informativeness: float | None`, `validity: str`, `validity_reasons: tuple[str, ...]`. Every later task relies on these names.

- [ ] **Step 1: Write the failing test**

```python
def test_scores_default_to_unscored_not_to_a_number():
    """An unscored event must be distinguishable from a low-confidence one.
    Defaulting to 0.0 would put every unscored detection in the bottom
    reliability bin and make an unscored detector look badly calibrated
    rather than unscored."""
    e = ev()
    assert e.detection_confidence is None
    assert e.informativeness is None


def test_validity_defaults_to_ok_with_no_reasons():
    e = ev()
    assert e.validity == "ok"
    assert e.validity_reasons == ()


def test_scores_round_trip_through_the_constructor():
    e = DetectedEvent(
        sid="dev_001", country_code="DE", channel="TV",
        event_type="dark_period",
        start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-04-01"),
        detection_confidence=0.87, informativeness=0.42,
        validity="suspect_tracking_loss",
        validity_reasons=("spend zero while impressions continue",),
    )
    assert e.detection_confidence == 0.87
    assert e.informativeness == 0.42
    assert e.validity == "suspect_tracking_loss"
    assert e.validity_reasons == ("spend zero while impressions continue",)
```

- [ ] **Step 2: Run to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_model.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'detection_confidence'`

- [ ] **Step 3: Add the fields**

In `detection/model.py`, inside the frozen `DetectedEvent` dataclass, after `evidence`:

```python
    # Spec section 8's three scores. None means UNSCORED, which is not the same
    # as zero: the reliability curve skips unscored detections rather than
    # binning them at the bottom, so a placeholder here would be
    # indistinguishable from a real low-confidence result.
    detection_confidence: float | None = None
    informativeness: float | None = None
    # Spec section 8's validity gate. "ok" is the default because most events
    # are fine; the reasons tuple is empty unless something tripped.
    validity: str = "ok"
    validity_reasons: tuple[str, ...] = ()
```

- [ ] **Step 4: Run to verify it passes**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Confirm nothing else broke**

Run: `synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"`
Expected: all pass (482 before this task)

- [ ] **Step 6: Commit**

```bash
git add detection/model.py tests/detection/test_model.py
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): DetectedEvent carries scores and a validity verdict"
```

---

### Task 2: Confidence sub-scores

**Files:**
- Create: `detection/score.py`
- Modify: `detection/params.py`
- Test: `tests/detection/test_score.py`

**Interfaces:**
- Consumes: `Panel` (`panel.series(country, channel)`, `panel.present_mask(country, channel)`, `panel.impressions`, `panel.channels_in(country)`, `panel.dates`), `find_off_runs`, `OffRun(start, end, n_days, depth, kind, notable, edge_sharpness, touches_start, touches_end)`, `DetectedEvent`
- Produces: `sub_scores(event: DetectedEvent, panel: Panel) -> dict[str, float]` returning exactly the six keys `magnitude_evidence`, `duration_evidence`, `distinctiveness`, `edge_sharpness`, `corroboration`, `consistency`, each in [0, 1].

Spec section 8's table, implemented literally:

| sub-score | source |
|---|---|
| `magnitude_evidence` | run depth, or `min(1, abs(z) / Z_SATURATION)` for steps |
| `duration_evidence` | `min(1, n_days / (DURATION_SATURATION_MULT * MIN_DAYS))` |
| `distinctiveness` | run length against that series' own p90 off-run |
| `edge_sharpness` | transition concentration |
| `corroboration` | spend 0 **and** impressions 0 scores 1.0; spend 0 with impressions still flowing scores `CORROBORATION_CONTRADICTED` |
| `consistency` | for country-level events, the fraction of channels agreeing |

- [ ] **Step 1: Add the parameters**

In `detection/params.py`, at the end, a new section:

```python
# --- Section 8, confidence sub-scores ------------------------------------

# |z| at which magnitude evidence for a step change saturates. A z of this size
# is already overwhelming; above it the extra certainty is not worth reporting
# as a difference.
Z_SATURATION = 8.0

# Duration evidence saturates at this multiple of MIN_DAYS. At 2 a 14-day event
# is fully evidenced; lower makes every reportable event look equally long.
DURATION_SATURATION_MULT = 2.0

# Distinctiveness saturates at this ratio of the run's length to the series' own
# p90 off-run. At 3 a run three times the usual gap is maximally distinctive.
DISTINCTIVENESS_SATURATION = 3.0

# Spend at zero while impressions keep flowing is the signature of tracking
# loss, not a real pause. It is not zero, because the spend feed may simply be
# late, but it must sit far below a corroborated stop.
CORROBORATION_CONTRADICTED = 0.2

# No impressions column, or impressions that are all zero for this series, means
# corroboration is unavailable rather than contradicted.
CORROBORATION_UNKNOWN = 0.6
```

- [ ] **Step 2: Write the failing tests**

Create `tests/detection/test_score.py`:

```python
import numpy as np
import pandas as pd
import pytest

from detection import params
from detection.io.panel import build_panel
from detection.model import DetectedEvent
from detection.score import sub_scores

COLS = ["date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"]


def build(spend_by_channel, country="DE", start="2024-01-01",
          impressions_by_channel=None):
    """spend_by_channel: {channel: [daily spend]}. Impressions default to
    tracking spend (zero spend -> zero impressions), which is the corroborated
    case; pass impressions_by_channel to break that link."""
    n = len(next(iter(spend_by_channel.values())))
    dates = pd.date_range(start, periods=n)
    rows = []
    for ch, values in spend_by_channel.items():
        imps = (impressions_by_channel or {}).get(ch)
        for i, (d, v) in enumerate(zip(dates, values)):
            impression = imps[i] if imps is not None else (100.0 if v > 0 else 0.0)
            rows.append([d, "P", ch, "c", 1, float(v), 1.0, float(impression),
                         0, 0.0, country])
    return build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")


def event(panel, channel, start, end, event_type="natural_holdout",
          country="DE", **kw):
    return DetectedEvent(
        sid="dev_test", country_code=country, channel=channel,
        event_type=event_type,
        start=panel.dates[start], end=panel.dates[end], **kw)


def test_every_sub_score_is_present_and_bounded():
    """All six are reported alongside the result, so all six must exist for
    every event type -- an absent key would silently drop a term from the
    weighted mean."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    s = sub_scores(event(p, "TV", 60, 89), p)
    assert set(s) == {"magnitude_evidence", "duration_evidence",
                      "distinctiveness", "edge_sharpness", "corroboration",
                      "consistency"}
    for name, value in s.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_a_deep_exact_zero_run_scores_full_magnitude_evidence():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    assert sub_scores(event(p, "TV", 60, 89), p)["magnitude_evidence"] == 1.0


def test_a_shallow_near_zero_run_scores_less_than_an_exact_zero():
    """Depth is the discriminator: spend cut to a trickle is weaker evidence of
    a deliberate pause than spend cut to nothing."""
    deep = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                  "Radio": [50.0] * 150})
    shallow = build({"TV": [100.0] * 60 + [8.0] * 30 + [100.0] * 60,
                     "Radio": [50.0] * 150})
    assert (sub_scores(event(shallow, "TV", 60, 89), shallow)["magnitude_evidence"]
            < sub_scores(event(deep, "TV", 60, 89), deep)["magnitude_evidence"])


def test_a_step_takes_its_magnitude_evidence_from_z():
    """Steps have no run depth -- the channel never stopped -- so the spec
    routes them through |z| / Z_SATURATION instead."""
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    e = event(p, "TV", 75, 149, event_type="step_change",
              evidence={"z": params.Z_SATURATION / 2})
    assert sub_scores(e, p)["magnitude_evidence"] == pytest.approx(0.5)


def test_step_magnitude_evidence_saturates_rather_than_exceeding_one():
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    e = event(p, "TV", 75, 149, event_type="step_change",
              evidence={"z": params.Z_SATURATION * 10})
    assert sub_scores(e, p)["magnitude_evidence"] == 1.0


def test_duration_evidence_grows_with_length_then_saturates():
    """Sized with literals 10 and 40, with the constant asserted to sit between
    them -- deriving the fixture from MIN_DAYS would make it move with the
    parameter and assert nothing."""
    assert 5 < params.MIN_DAYS <= 20
    short = build({"TV": [100.0] * 60 + [0.0] * 10 + [100.0] * 80,
                   "Radio": [50.0] * 150})
    long = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                  "Radio": [50.0] * 150})
    s_short = sub_scores(event(short, "TV", 60, 69), short)["duration_evidence"]
    s_long = sub_scores(event(long, "TV", 60, 99), long)["duration_evidence"]
    assert s_short < s_long
    assert s_long == 1.0


def test_distinctiveness_compares_the_run_against_the_series_own_gaps():
    """A 30-day stop on a channel that never otherwise pauses is far more
    distinctive than the same stop on a channel that goes dark every fortnight.
    This is why the comparison is per series and not global."""
    clean = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                   "Radio": [50.0] * 150})
    flighting_tv = []
    for _ in range(4):
        flighting_tv += [100.0] * 20 + [0.0] * 10
    flighting_tv = flighting_tv[:60] + [0.0] * 30 + [100.0] * 60
    flighting = build({"TV": flighting_tv, "Radio": [50.0] * 150})
    assert (sub_scores(event(flighting, "TV", 60, 89), flighting)["distinctiveness"]
            < sub_scores(event(clean, "TV", 60, 89), clean)["distinctiveness"])


def test_corroboration_is_full_when_impressions_stop_with_spend():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    assert sub_scores(event(p, "TV", 60, 89), p)["corroboration"] == 1.0


def test_corroboration_collapses_when_impressions_keep_flowing():
    """Spend at zero while impressions continue is tracking loss, not a pause.
    The spec gives this case a specifically low score rather than zero, because
    a late spend feed produces the same shape."""
    imps = [100.0] * 150          # impressions never stop
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150},
              impressions_by_channel={"TV": imps})
    assert (sub_scores(event(p, "TV", 60, 89), p)["corroboration"]
            == params.CORROBORATION_CONTRADICTED)


def test_consistency_is_the_fraction_of_channels_agreeing_for_a_dark_period():
    """A dark period claims the whole market. If one of four channels kept
    running, the claim is three-quarters supported."""
    p = build({"A": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "B": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "C": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "D": [100.0] * 150})
    e = event(p, None, 60, 89, event_type="dark_period")
    assert sub_scores(e, p)["consistency"] == pytest.approx(0.75)


def test_consistency_is_full_for_a_single_channel_event():
    """A channel-level event makes no claim about other channels, so there is
    nothing to disagree with it."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    assert sub_scores(event(p, "TV", 60, 89), p)["consistency"] == 1.0


def test_an_event_on_a_channel_with_no_off_run_still_scores():
    """Step changes never stop the channel, so the run-derived sub-scores have
    no run to read. They must degrade to a defined value, not raise."""
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    s = sub_scores(event(p, "TV", 75, 149, event_type="step_change",
                         evidence={"z": 5.0}), p)
    assert all(0.0 <= v <= 1.0 for v in s.values())
```

- [ ] **Step 3: Run to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'detection.score'`

- [ ] **Step 4: Implement `detection/score.py`**

```python
"""Spec section 8 -- the sub-scores behind detection_confidence.

Every sub-score is in [0, 1] and is reported alongside the result, so an analyst
can see WHY a number is what it is rather than being handed one opaque figure.
That is also why they are computed separately and combined by a weighted mean
instead of being folded into one expression: a term that misfires is visible.

The scores read the panel rather than trusting the event's evidence dict. The
composition layer writes evidence for human consumption and its contents vary by
event type; recomputing here keeps the six sub-scores defined for every event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detection import params
from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.primitives.zero_runs import find_off_runs

COUNTRY_LEVEL_TYPES = frozenset({"dark_period", "single_channel"})


def _covering_run(panel: Panel, country: str, channel: str,
                  start: pd.Timestamp, end: pd.Timestamp):
    """The off-run this event sits on, or None for events that never stop."""
    series = panel.series(country, channel)
    present = panel.present_mask(country, channel)
    best, best_overlap = None, 0
    for run in find_off_runs(series, present):
        overlap = (min(run.end, end) - max(run.start, start)).days + 1
        if overlap > best_overlap:
            best, best_overlap = run, overlap
    return best


def _subject_channels(event: DetectedEvent, panel: Panel) -> list[str]:
    """The channels this event actually makes a claim about.

    A dark period claims every channel in the market. A single_channel event
    names the channel still RUNNING, so its subjects are all the others -- the
    same inversion that produced a real defect in the cross-market layer.
    """
    country = event.country_code
    if event.event_type == "dark_period":
        return panel.channels_in(country)
    if event.event_type == "single_channel":
        return [ch for ch in panel.channels_in(country) if ch != event.channel]
    return [event.channel] if event.channel else []


def _magnitude_evidence(event: DetectedEvent, panel: Panel,
                        run) -> float:
    if event.event_type == "step_change":
        z = abs(float(event.evidence.get("z", 0.0)))
        return min(1.0, z / params.Z_SATURATION)
    if run is None:
        return 0.0
    # depth is the fraction by which spend fell; an exact zero is depth 1.
    return float(min(1.0, max(0.0, run.depth)))


def _duration_evidence(event: DetectedEvent) -> float:
    span = params.DURATION_SATURATION_MULT * params.MIN_DAYS
    return float(min(1.0, event.n_days / span))


def _distinctiveness(event: DetectedEvent, panel: Panel, run) -> float:
    """This run's length against the p90 of the series' OTHER off-runs.

    Per series, never global: a 30-day stop means something quite different on a
    channel that never pauses than on one that flights every fortnight.
    """
    if run is None or not event.channel:
        return 0.0
    series = panel.series(event.country_code, event.channel)
    others = [r.n_days for r in find_off_runs(series,
                                              panel.present_mask(event.country_code,
                                                                 event.channel))
              if not (r.start == run.start and r.end == run.end)]
    if not others:
        return 1.0
    p90 = float(np.percentile(np.array(others, dtype=float), 90))
    if p90 <= 0:
        return 1.0
    ratio = run.n_days / p90
    return float(min(1.0, ratio / params.DISTINCTIVENESS_SATURATION))


def _corroboration(event: DetectedEvent, panel: Panel) -> float:
    """Did impressions stop when spend did?

    Spend at zero with impressions still flowing is the signature of tracking
    loss rather than a deliberate pause. It scores low but not zero, because a
    spend feed that simply arrives late produces exactly the same shape.
    """
    if event.event_type == "step_change":
        return params.CORROBORATION_UNKNOWN
    channels = [ch for ch in _subject_channels(event, panel) if ch]
    if not channels:
        return params.CORROBORATION_UNKNOWN
    window = slice(event.start, event.end)
    scores = []
    for ch in channels:
        key = (event.country_code, ch)
        if key not in panel.impressions.columns:
            scores.append(params.CORROBORATION_UNKNOWN)
            continue
        imps = panel.impressions.loc[window, key]
        total = float(np.nansum(imps.values))
        outside = panel.impressions[key].drop(panel.impressions.loc[window].index)
        if float(np.nansum(outside.values)) <= 0:
            # This series never reports impressions at all; silence inside the
            # window corroborates nothing.
            scores.append(params.CORROBORATION_UNKNOWN)
        elif total <= 0:
            scores.append(1.0)
        else:
            scores.append(params.CORROBORATION_CONTRADICTED)
    return float(np.mean(scores))


def _edge_sharpness(event: DetectedEvent, panel: Panel, run) -> float:
    if run is not None:
        return float(min(1.0, max(0.0, run.edge_sharpness)))
    if event.event_type == "step_change":
        return float(min(1.0, max(0.0, event.evidence.get("sharpness", 0.0))))
    return 0.0


def _consistency(event: DetectedEvent, panel: Panel) -> float:
    """For a country-level event, the fraction of the channels it claims that
    actually went off. A channel-level event claims nothing about its
    neighbours, so there is nothing to disagree and it scores full."""
    if event.event_type not in COUNTRY_LEVEL_TYPES:
        return 1.0
    channels = _subject_channels(event, panel)
    if not channels:
        return 0.0
    agreeing = 0
    for ch in channels:
        run = _covering_run(panel, event.country_code, ch, event.start, event.end)
        if run is not None:
            agreeing += 1
    return agreeing / len(channels)


def sub_scores(event: DetectedEvent, panel: Panel) -> dict[str, float]:
    """The six named sub-scores from spec section 8, each in [0, 1]."""
    run = None
    if event.channel and event.country_code:
        run = _covering_run(panel, event.country_code, event.channel,
                            event.start, event.end)
    elif event.country_code and event.event_type == "dark_period":
        channels = _subject_channels(event, panel)
        if channels:
            run = _covering_run(panel, event.country_code, channels[0],
                                event.start, event.end)
    return {
        "magnitude_evidence": _magnitude_evidence(event, panel, run),
        "duration_evidence": _duration_evidence(event),
        "distinctiveness": _distinctiveness(event, panel, run),
        "edge_sharpness": _edge_sharpness(event, panel, run),
        "corroboration": _corroboration(event, panel),
        "consistency": _consistency(event, panel),
    }
```

- [ ] **Step 5: Run to verify they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_score.py -v`
Expected: PASS

- [ ] **Step 6: Mutation-verify every sub-score**

For each of the six, DELETE its line from the returned dict (not zero its constant) and confirm a named test fails. Clear `__pycache__` first each time. Record the table in the report. Then verify the three new parameters:

```bash
find detection tests -name __pycache__ -type d -exec rm -rf {} +
# CORROBORATION_CONTRADICTED -> 1.0 must fail the impressions-keep-flowing test
# Z_SATURATION -> 1.0 must fail the step-z test
# DISTINCTIVENESS_SATURATION -> 1e9 must fail the flighting comparison
```

Any parameter whose only failing test is `test_no_logic_module_hardcodes_a_threshold` or `test_every_documented_parameter_exists_with_the_spec_value` is NOT covered — add a behavioural test or disclose it.

- [ ] **Step 7: Add the value assertions**

In `tests/detection/test_params.py`, in the value-assertion test:

```python
    assert params.Z_SATURATION == 8.0
    assert params.DURATION_SATURATION_MULT == 2.0
    assert params.DISTINCTIVENESS_SATURATION == 3.0
    assert params.CORROBORATION_CONTRADICTED == 0.2
    assert params.CORROBORATION_UNKNOWN == 0.6
```

- [ ] **Step 8: Run the full suite and commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/score.py detection/params.py tests/detection/test_score.py tests/detection/test_params.py
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): the six confidence sub-scores from spec section 8"
```

---

### Task 3: `detection_confidence` and `informativeness`

**Files:**
- Modify: `detection/score.py`, `detection/params.py`
- Test: `tests/detection/test_score.py`

**Interfaces:**
- Consumes: `sub_scores(event, panel) -> dict[str, float]` from Task 2
- Produces: `confidence(event, panel) -> tuple[float, dict[str, float]]` (score plus the sub-scores that produced it) and `informativeness(event, panel) -> tuple[float, dict[str, float]]`

Weights differ by event type, per spec section 8 ("Weights differ by event type and are stated in `params.py`").

- [ ] **Step 1: Add the weights**

In `detection/params.py`:

```python
# Confidence sub-score weights per event type. They differ because the evidence
# differs: a dark period's whole claim is that every channel stopped together,
# so consistency carries real weight there and none at all for a step change,
# which makes no claim about its neighbours. Each row sums to 1.
CONFIDENCE_WEIGHTS = {
    "dark_period": {"magnitude_evidence": 0.2, "duration_evidence": 0.15,
                    "distinctiveness": 0.15, "edge_sharpness": 0.1,
                    "corroboration": 0.15, "consistency": 0.25},
    "single_channel": {"magnitude_evidence": 0.2, "duration_evidence": 0.15,
                       "distinctiveness": 0.15, "edge_sharpness": 0.1,
                       "corroboration": 0.15, "consistency": 0.25},
    "natural_holdout": {"magnitude_evidence": 0.25, "duration_evidence": 0.2,
                        "distinctiveness": 0.25, "edge_sharpness": 0.1,
                        "corroboration": 0.2, "consistency": 0.0},
    "channel_pulse": {"magnitude_evidence": 0.25, "duration_evidence": 0.1,
                      "distinctiveness": 0.3, "edge_sharpness": 0.15,
                      "corroboration": 0.2, "consistency": 0.0},
    "staggered_launch": {"magnitude_evidence": 0.2, "duration_evidence": 0.2,
                         "distinctiveness": 0.2, "edge_sharpness": 0.1,
                         "corroboration": 0.3, "consistency": 0.0},
    "step_change": {"magnitude_evidence": 0.45, "duration_evidence": 0.2,
                    "distinctiveness": 0.0, "edge_sharpness": 0.25,
                    "corroboration": 0.1, "consistency": 0.0},
}

# --- Section 8, informativeness ------------------------------------------

# Type prior. Dark periods read the baseline directly, single-channel periods
# give unambiguous attribution, and pulses are the only place adstock decay is
# observable -- so all three outrank a plain step change for an analyst
# choosing what to look at.
TYPE_PRIOR = {
    "dark_period": 1.0,
    "single_channel": 0.9,
    "channel_pulse": 0.9,
    "natural_holdout": 0.75,
    "staggered_launch": 0.6,
    "step_change": 0.45,
}

# Control availability, spec section 8: peers, then sibling channels, then none.
CONTROL_SCORE = {"peers": 1.0, "sibling_channels": 0.7, "none": 0.3}

# Informativeness driver weights. One row, not per type -- the type's own
# influence enters through TYPE_PRIOR.
INFORMATIVENESS_WEIGHTS = {
    "duration_adequacy": 0.25,
    "contrast": 0.15,
    "cleanliness": 0.15,
    "control_availability": 0.2,
    "type_prior": 0.25,
}

# Assumed adstock half-life in days. A window shorter than a few half-lives
# cannot show the decay it is supposed to reveal. Stated as an assumption
# because the real value is a modelling question, not a measurement.
ADSTOCK_HALF_LIFE = 7.0

# Informativeness saturates once the window covers this many half-lives.
ADSTOCK_WINDOWS_FOR_FULL_CREDIT = 4.0

# Multiplier applied when the event is censored at a series edge -- its true
# extent is unknown, so it is worth less than the same event fully observed.
CENSORING_PENALTY = 0.7

# Multiplier applied when another event overlaps the window and confounds it.
CONFOUNDED_PENALTY = 0.6
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/detection/test_score.py`:

```python
from detection.score import confidence, informativeness


def test_confidence_weights_sum_to_one_for_every_event_type():
    """A row that does not sum to 1 silently rescales that type's confidence
    against every other type, which would corrupt the calibration."""
    from detection.model import EVENT_TYPES
    assert set(params.CONFIDENCE_WEIGHTS) == set(EVENT_TYPES)
    for event_type, weights in params.CONFIDENCE_WEIGHTS.items():
        assert sum(weights.values()) == pytest.approx(1.0), event_type


def test_confidence_weights_name_exactly_the_six_sub_scores():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    names = set(sub_scores(event(p, "TV", 60, 89), p))
    for event_type, weights in params.CONFIDENCE_WEIGHTS.items():
        assert set(weights) == names, event_type


def test_confidence_is_the_weighted_mean_of_the_reported_sub_scores():
    """The returned sub-scores must be the ones that produced the number.
    Reporting one set and scoring another is how an explanation becomes a lie."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89)
    score, parts = confidence(e, p)
    weights = params.CONFIDENCE_WEIGHTS["natural_holdout"]
    expected = sum(parts[k] * weights[k] for k in weights)
    assert score == pytest.approx(expected)
    assert 0.0 <= score <= 1.0


def test_a_clean_long_exact_zero_outscores_a_short_shallow_one():
    """The ordering is the whole point of the score. If this inverts, ranking
    by confidence is worse than not ranking."""
    strong = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                    "Radio": [50.0] * 150})
    weak = build({"TV": [100.0] * 60 + [9.0] * 8 + [100.0] * 82,
                  "Radio": [50.0] * 150})
    assert (confidence(event(strong, "TV", 60, 99), strong)[0]
            > confidence(event(weak, "TV", 60, 67), weak)[0])


def test_tracking_loss_lowers_confidence_against_an_identical_clean_event():
    """Same window, same depth, same duration -- the ONLY difference is that
    impressions kept flowing. Confidence must notice."""
    clean = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                   "Radio": [50.0] * 150})
    tracked = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
                     "Radio": [50.0] * 150},
                    impressions_by_channel={"TV": [100.0] * 150})
    assert (confidence(event(tracked, "TV", 60, 89), tracked)[0]
            < confidence(event(clean, "TV", 60, 89), clean)[0])


def test_informativeness_prefers_a_dark_period_to_a_step_of_equal_length():
    """Spec section 8's type prior: a dark period lets the baseline be read
    directly, a step change does not."""
    p = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
               "Radio": [50.0] * 150})
    dark = event(p, None, 60, 99, event_type="dark_period")
    step = event(p, "TV", 60, 99, event_type="step_change",
                 evidence={"z": 6.0})
    assert informativeness(dark, p)[0] > informativeness(step, p)[0]


def test_a_window_too_short_to_show_adstock_decay_scores_low_duration():
    """Sized with literals against ADSTOCK_HALF_LIFE asserted, not derived."""
    assert params.ADSTOCK_HALF_LIFE >= 5.0
    p = build({"TV": [100.0] * 60 + [0.0] * 8 + [100.0] * 82,
               "Radio": [50.0] * 150})
    long_p = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                    "Radio": [50.0] * 150})
    _, short_parts = informativeness(event(p, "TV", 60, 67), p)
    _, long_parts = informativeness(event(long_p, "TV", 60, 99), long_p)
    assert short_parts["duration_adequacy"] < long_parts["duration_adequacy"]


def test_control_availability_follows_the_cross_market_tag():
    """The cross-market layer already worked out whether peers were running;
    informativeness must use that answer rather than guessing again."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    with_peers = event(p, "TV", 60, 89, evidence={"control_available": "peers"})
    without = event(p, "TV", 60, 89, evidence={"control_available": "none"})
    assert informativeness(with_peers, p)[0] > informativeness(without, p)[0]


def test_a_censored_event_is_worth_less_than_the_same_event_fully_observed():
    """An event running to the edge of the series has unknown true extent."""
    p = build({"TV": [0.0] * 40 + [100.0] * 110, "Radio": [50.0] * 150})
    censored = event(p, "TV", 0, 39, evidence={"censored_start": True})
    clean_p = build({"TV": [100.0] * 60 + [0.0] * 40 + [100.0] * 50,
                     "Radio": [50.0] * 150})
    clean = event(clean_p, "TV", 60, 99)
    assert informativeness(censored, p)[0] < informativeness(clean, clean_p)[0]


def test_informativeness_is_bounded_and_reports_its_drivers():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    score, parts = informativeness(event(p, "TV", 60, 89), p)
    assert 0.0 <= score <= 1.0
    assert set(parts) == set(params.INFORMATIVENESS_WEIGHTS)
```

- [ ] **Step 3: Run to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_score.py -v`
Expected: FAIL — `ImportError: cannot import name 'confidence'`

- [ ] **Step 4: Implement**

Append to `detection/score.py`:

```python
def confidence(event: DetectedEvent, panel: Panel
               ) -> tuple[float, dict[str, float]]:
    """Spec section 8's detection_confidence, plus the sub-scores behind it.

    Both are returned together deliberately: the number and its justification
    must come from the same computation, or the explanation stops being an
    explanation.
    """
    parts = sub_scores(event, panel)
    weights = params.CONFIDENCE_WEIGHTS[event.event_type]
    score = sum(parts[name] * weight for name, weight in weights.items())
    return float(min(1.0, max(0.0, score))), parts


def _duration_adequacy(event: DetectedEvent) -> float:
    """Can this window show adstock decay at all?

    A pause shorter than a couple of half-lives cannot reveal the decay it is
    supposed to expose, however confident we are that it happened.
    """
    span = params.ADSTOCK_HALF_LIFE * params.ADSTOCK_WINDOWS_FOR_FULL_CREDIT
    return float(min(1.0, event.n_days / span))


def _contrast(event: DetectedEvent, panel: Panel) -> float:
    """How far the window departs from the series' normal level."""
    if event.event_type == "step_change":
        ratio = event.magnitude_ratio
        if ratio is None or ratio <= 0:
            return 1.0        # out of zero: maximal contrast
        return float(min(1.0, abs(np.log(ratio)) / np.log(params.DISTINCTIVENESS_SATURATION)))
    run = None
    if event.channel and event.country_code:
        run = _covering_run(panel, event.country_code, event.channel,
                            event.start, event.end)
    return float(min(1.0, max(0.0, run.depth))) if run is not None else 0.0


def _cleanliness(event: DetectedEvent) -> float:
    """Penalty when another event overlaps and confounds the window."""
    return params.CONFOUNDED_PENALTY if "confounded" in event.tags else 1.0


def _control_availability(event: DetectedEvent) -> float:
    control = event.evidence.get("control_available", "none")
    return params.CONTROL_SCORE.get(control, params.CONTROL_SCORE["none"])


def informativeness(event: DetectedEvent, panel: Panel
                    ) -> tuple[float, dict[str, float]]:
    """Spec section 8's informativeness, plus its drivers.

    This is what ranking uses. It answers "how useful is this?", which is a
    different question from "how sure am I?" -- a genuine but uninformative
    global pause scores high on confidence and low here, and collapsing the two
    is the standard mistake the spec calls out.
    """
    parts = {
        "duration_adequacy": _duration_adequacy(event),
        "contrast": _contrast(event, panel),
        "cleanliness": _cleanliness(event),
        "control_availability": _control_availability(event),
        "type_prior": params.TYPE_PRIOR[event.event_type],
    }
    weights = params.INFORMATIVENESS_WEIGHTS
    score = sum(parts[name] * weight for name, weight in weights.items())
    if event.evidence.get("censored_start") or event.evidence.get("censored_end"):
        score *= params.CENSORING_PENALTY
    return float(min(1.0, max(0.0, score))), parts
```

- [ ] **Step 5: Run to verify they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_score.py -v`
Expected: PASS

- [ ] **Step 6: Mutation-verify**

DELETE each informativeness driver from the `parts` dict in turn and confirm a named test fails. Set `CENSORING_PENALTY` and `CONFOUNDED_PENALTY` to 1.0 and confirm the censoring test fails. Add value assertions for every new parameter in `tests/detection/test_params.py`. Record the table.

- [ ] **Step 7: Commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/score.py detection/params.py tests/detection/test_score.py tests/detection/test_params.py
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): detection_confidence and informativeness"
```

---

### Task 4: The validity gate

**Files:**
- Create: `detection/validity.py`
- Test: `tests/detection/test_validity.py`

**Interfaces:**
- Consumes: `Panel`, `DetectedEvent`
- Produces: `assess(event: DetectedEvent, panel: Panel) -> tuple[str, tuple[str, ...]]` returning one of `"ok"`, `"suspect_data_gap"`, `"suspect_tracking_loss"` and the reasons that triggered it.

Spec section 8 lists exactly four triggers: rows missing rather than zero; spend at zero while impressions continue; every channel in every country off at once; a spend drop with no sales response where sales SNR was adequate to show one.

- [ ] **Step 1: Write the failing tests**

Create `tests/detection/test_validity.py` (reuse the `build`/`event` helpers by importing them):

```python
import pandas as pd
import pytest

from detection import params
from detection.validity import assess
from tests.detection.test_score import build, event


def test_a_clean_event_is_ok_with_no_reasons():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    verdict, reasons = assess(event(p, "TV", 60, 89), p)
    assert verdict == "ok"
    assert reasons == ()


def test_missing_rows_are_a_data_gap_not_a_holdout():
    """The single largest real-data risk in spec section 11: an export that
    omits a row rather than writing a zero. The shape is identical; the meaning
    is the opposite."""
    n = 150
    dates = pd.date_range("2024-01-01", periods=n)
    rows = []
    for i, d in enumerate(dates):
        if 60 <= i < 90:
            continue                      # rows ABSENT, not zero
        rows.append([d, "P", "TV", "c", 1, 100.0, 1.0, 100.0, 0, 0.0, "DE"])
        rows.append([d, "P", "Radio", "c", 1, 50.0, 1.0, 50.0, 0, 0.0, "DE"])
    from detection.io.panel import build_panel
    from tests.detection.test_score import COLS
    p = build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")
    verdict, reasons = assess(event(p, "TV", 60, 89), p)
    assert verdict == "suspect_data_gap"
    assert any("missing" in r for r in reasons)


def test_spend_zero_with_impressions_flowing_is_tracking_loss():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150},
              impressions_by_channel={"TV": [100.0] * 150})
    verdict, reasons = assess(event(p, "TV", 60, 89), p)
    assert verdict == "suspect_tracking_loss"
    assert any("impression" in r for r in reasons)


def test_every_channel_in_every_country_off_at_once_is_suspect():
    """A panel-wide stop is far more likely to be a feed outage than a
    coordinated global marketing pause."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [100.0] * 60 + [0.0] * 30 + [100.0] * 60})
    verdict, reasons = assess(
        event(p, None, 60, 89, event_type="dark_period"), p)
    assert verdict != "ok"
    assert any("every channel" in r for r in reasons)


def test_a_verdict_lists_every_reason_that_fired():
    """Reasons accumulate: an analyst triaging needs all of them, not the
    first one found."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [100.0] * 60 + [0.0] * 30 + [100.0] * 60},
              impressions_by_channel={"TV": [100.0] * 150,
                                      "Radio": [100.0] * 150})
    _, reasons = assess(event(p, None, 60, 89, event_type="dark_period"), p)
    assert len(reasons) >= 2


def test_a_step_change_is_not_assessed_for_missing_rows():
    """A step change never stops the channel, so there is no absence to
    explain and the gate must not invent one."""
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    verdict, _ = assess(event(p, "TV", 75, 149, event_type="step_change",
                              evidence={"z": 6.0}), p)
    assert verdict == "ok"
```

- [ ] **Step 2: Run to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_validity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'detection.validity'`

- [ ] **Step 3: Implement `detection/validity.py`**

```python
"""Spec section 8's validity gate.

Three verdicts: `ok`, `suspect_data_gap`, `suspect_tracking_loss`. The gate
exists because the two most dangerous real-data failures produce EXACTLY the
shape of a genuine event. An export that omits rows instead of writing zeros
looks like a perfect holdout. A tracking pixel that keeps firing after spend
stops looks like a pause that did not happen.

Neither can be settled from spend alone, so the gate reports suspicion rather
than filtering: a detector that silently dropped these would hide the single
largest risk in spec section 11 instead of surfacing it.
"""
from __future__ import annotations

import numpy as np

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.score import _subject_channels

OK = "ok"
SUSPECT_DATA_GAP = "suspect_data_gap"
SUSPECT_TRACKING_LOSS = "suspect_tracking_loss"

# A step change never stops the channel, so absence and tracking checks have no
# subject there.
_STOPPING_TYPES = frozenset({"dark_period", "single_channel", "natural_holdout",
                             "channel_pulse", "staggered_launch"})


def assess(event: DetectedEvent, panel: Panel) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    verdict = OK
    if event.event_type not in _STOPPING_TYPES:
        return verdict, ()

    channels = [ch for ch in _subject_channels(event, panel) if ch]
    window = slice(event.start, event.end)

    for ch in channels:
        key = (event.country_code, ch)
        if key in panel.present.columns:
            present = panel.present.loc[window, key]
            if not bool(present.all()):
                missing = int((~present).sum())
                reasons.append(
                    f"{ch}: {missing} of {len(present)} days have no row at all "
                    f"rather than a zero -- the export may be omitting rows")
                verdict = SUSPECT_DATA_GAP

    for ch in channels:
        key = (event.country_code, ch)
        if key not in panel.impressions.columns:
            continue
        inside = float(np.nansum(panel.impressions.loc[window, key].values))
        outside_frame = panel.impressions[key].drop(
            panel.impressions.loc[window].index)
        outside = float(np.nansum(outside_frame.values))
        if inside > 0 and outside > 0:
            reasons.append(
                f"{ch}: spend is zero but impressions continue in the window -- "
                f"this may be tracking loss rather than a pause")
            if verdict == OK:
                verdict = SUSPECT_TRACKING_LOSS

    if _everything_off(panel, event):
        reasons.append(
            "every channel in every market is off simultaneously -- more likely "
            "a feed outage than a coordinated global pause")
        if verdict == OK:
            verdict = SUSPECT_DATA_GAP

    return verdict, tuple(reasons)


def _everything_off(panel: Panel, event: DetectedEvent) -> bool:
    window = panel.spend.loc[event.start:event.end]
    if window.empty:
        return False
    return bool((np.nansum(window.values) == 0) and window.shape[1] > 1)
```

- [ ] **Step 4: Run to verify they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_validity.py -v`
Expected: PASS

- [ ] **Step 5: Mutation-verify**

DELETE each of the three checks in turn and confirm its named test fails. Record the table.

- [ ] **Step 6: Commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/validity.py tests/detection/test_validity.py
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): the validity gate from spec section 8"
```

---

### Task 5: PAV isotonic calibration

**Files:**
- Create: `detection/calibrate.py`
- Test: `tests/detection/test_calibrate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure numpy)
- Produces: `fit_pav(scores: list[float], correct: list[bool], n_bins: int) -> list[tuple[float, float]]` returning `(bin_upper_edge, calibrated_value)` knots, monotone non-decreasing; `apply_calibration(score: float, knots: list[tuple[float, float]]) -> float`; `render_module(knots) -> str` emitting the frozen module source.

The spec: "confidence is binned into deciles, empirical precision measured per bin, a monotone pool-adjacent-violators mapping fitted, and that mapping frozen before the final run."

- [ ] **Step 1: Write the failing tests**

Create `tests/detection/test_calibrate.py`:

```python
import numpy as np
import pytest

from detection.calibrate import apply_calibration, fit_pav, render_module


def test_a_perfectly_calibrated_input_is_left_alone():
    """If raw confidence already matches empirical precision, calibration must
    be close to the identity -- a mapping that moves well-calibrated scores is
    doing harm."""
    rng = np.random.default_rng(0)
    scores, correct = [], []
    for s in np.linspace(0.05, 0.95, 10):
        for _ in range(200):
            scores.append(float(s))
            correct.append(bool(rng.random() < s))
    knots = fit_pav(scores, correct, n_bins=10)
    for s in [0.15, 0.45, 0.75, 0.95]:
        assert abs(apply_calibration(s, knots) - s) < 0.12


def test_an_overconfident_detector_is_pulled_down():
    """Every event scored 0.9 but only 30% are right. Calibration must report
    about 0.3, because the whole purpose is to make the number a testable
    claim."""
    scores = [0.9] * 100
    correct = [True] * 30 + [False] * 70
    knots = fit_pav(scores, correct, n_bins=10)
    assert apply_calibration(0.9, knots) == pytest.approx(0.3, abs=0.05)


def test_the_mapping_is_monotone_non_decreasing():
    """Pool-adjacent-violators exists to guarantee this. A calibration that
    inverts anywhere would rank a worse event above a better one."""
    rng = np.random.default_rng(1)
    scores = list(rng.random(500))
    correct = [bool(rng.random() < s ** 2) for s in scores]
    knots = fit_pav(scores, correct, n_bins=10)
    values = [v for _, v in knots]
    assert values == sorted(values), values


def test_violating_bins_are_pooled_rather_than_left_inverted():
    """The defining behaviour of PAV, with a hand-built inversion: the 0.3 bin
    outperforms the 0.5 bin. They must merge to their shared rate, not stay
    crossed."""
    scores = [0.35] * 100 + [0.55] * 100
    correct = ([True] * 80 + [False] * 20) + ([True] * 40 + [False] * 60)
    knots = fit_pav(scores, correct, n_bins=10)
    low = apply_calibration(0.35, knots)
    high = apply_calibration(0.55, knots)
    assert low <= high
    assert low == pytest.approx(0.6, abs=0.05)
    assert high == pytest.approx(0.6, abs=0.05)


def test_an_empty_fit_returns_the_identity():
    """No data is not the same as 'everything is wrong'. With nothing to learn
    from, calibration must not alter the score."""
    knots = fit_pav([], [], n_bins=10)
    for s in [0.0, 0.25, 0.5, 1.0]:
        assert apply_calibration(s, knots) == pytest.approx(s)


def test_calibration_output_stays_in_range():
    rng = np.random.default_rng(2)
    scores = list(rng.random(200))
    correct = [bool(rng.random() < 0.5) for _ in scores]
    knots = fit_pav(scores, correct, n_bins=10)
    for s in np.linspace(0, 1, 21):
        assert 0.0 <= apply_calibration(float(s), knots) <= 1.0


def test_the_rendered_module_is_importable_and_round_trips():
    """The frozen mapping ships as Python because detection/ may contain
    nothing but source -- a JSON file there fails the gating test."""
    knots = [(0.5, 0.2), (1.0, 0.8)]
    src = render_module(knots)
    namespace = {}
    exec(compile(src, "calibration_fit.py", "exec"), namespace)
    assert namespace["KNOTS"] == knots
```

- [ ] **Step 2: Run to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_calibrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'detection.calibrate'`

- [ ] **Step 3: Implement `detection/calibrate.py`**

```python
"""Isotonic calibration by pool-adjacent-violators, hand-rolled.

Spec section 8: confidence is binned, empirical precision measured per bin, a
monotone mapping fitted, and that mapping FROZEN before the final run. This is
what converts the score from an assertion into a testable claim -- "events at
confidence 0.9 are correct about 90% of the time on data the algorithm has
never seen".

PAV in twenty lines rather than a scikit-learn dependency. The algorithm: take
the per-bin rates left to right; wherever a bin's rate is lower than the one
before it, the pair is inverted, so pool them into their combined rate and
re-check leftwards. What remains is the closest non-decreasing fit.

Fitted on the DEVELOPMENT split only. Fitting on the test split would be
grading your own exam.
"""
from __future__ import annotations

import numpy as np


def fit_pav(scores, correct, n_bins: int):
    """Return (bin_upper_edge, calibrated_value) knots, monotone in value."""
    scores = np.asarray(list(scores), dtype=float)
    correct = np.asarray(list(correct), dtype=bool)
    if scores.size == 0:
        return []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index for every score; the top edge is inclusive so 1.0 lands in the
    # last bin rather than falling off the end.
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)

    blocks = []          # (weight, sum_correct, upper_edge)
    for b in range(n_bins):
        members = correct[idx == b]
        if members.size == 0:
            continue
        blocks.append([float(members.size), float(members.sum()),
                       float(edges[b + 1])])

    # Pool adjacent violators.
    pooled: list[list[float]] = []
    for block in blocks:
        pooled.append(block)
        while len(pooled) >= 2:
            prev, cur = pooled[-2], pooled[-1]
            if prev[1] / prev[0] <= cur[1] / cur[0]:
                break
            merged = [prev[0] + cur[0], prev[1] + cur[1], cur[2]]
            pooled[-2:] = [merged]

    return [(upper, total / weight) for weight, total, upper in pooled]


def apply_calibration(score: float, knots) -> float:
    """Map a raw score through the fitted mapping.

    With no knots the mapping is the identity: an uncalibrated detector must
    report its raw score, not zero.
    """
    if not knots:
        return float(min(1.0, max(0.0, score)))
    for upper, value in knots:
        if score <= upper:
            return float(min(1.0, max(0.0, value)))
    return float(min(1.0, max(0.0, knots[-1][1])))


def render_module(knots) -> str:
    """Emit the frozen mapping as Python source.

    It ships as a module, not a data file, because `detection/` may contain
    nothing but source -- a JSON blob there would be neither scanned by the
    import gate nor covered by the final-run audit hash, which is exactly the
    hole that check exists to close.
    """
    lines = [
        '"""Frozen confidence calibration, fitted on the development split.',
        "",
        "GENERATED by detection/calibrate.py -- do not edit by hand.",
        "Refitting is a development-split operation and must never be run",
        "against the sealed test split.",
        '"""',
        "",
        "KNOTS = [",
    ]
    for upper, value in knots:
        lines.append(f"    ({upper!r}, {value!r}),")
    lines.append("]")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run to verify they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_calibrate.py -v`
Expected: PASS

- [ ] **Step 5: Mutation-verify the pooling**

DELETE the `while len(pooled) >= 2:` pooling loop and confirm `test_violating_bins_are_pooled_rather_than_left_inverted` and `test_the_mapping_is_monotone_non_decreasing` both fail. A calibration without pooling is just binning.

- [ ] **Step 6: Commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/calibrate.py tests/detection/test_calibrate.py
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): hand-rolled PAV isotonic calibration"
```

---

### Task 6: Templated explanations

**Files:**
- Create: `detection/explain.py`
- Test: `tests/detection/test_explain.py`

**Interfaces:**
- Consumes: `DetectedEvent` (with scores set), `Panel`
- Produces: `explain(event: DetectedEvent, panel: Panel) -> str`

The register to match, from spec section 8 — concrete numbers, named comparisons, no hedging:

> Facebook spend in AT fell from a typical €1,240/day to exactly €0 for 42 consecutive days (2025-02-04 to 2025-03-17) — the longest off-run in this series by a factor of 21, the next longest being 2 days. All five other AT channels ran at normal levels throughout... Label: natural_holdout (cross-market). Confidence 0.94, informativeness 0.81, validity ok.

- [ ] **Step 1: Write the failing tests**

Create `tests/detection/test_explain.py`:

```python
import pandas as pd
import pytest

from detection.explain import explain
from detection.model import DetectedEvent
from tests.detection.test_score import build, event


def test_the_explanation_names_the_channel_market_dates_and_length():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89, detection_confidence=0.9,
              informativeness=0.5)
    text = explain(e, p)
    assert "TV" in text
    assert "DE" in text
    assert str(p.dates[60].date()) in text
    assert str(p.dates[89].date()) in text
    assert "30" in text


def test_the_explanation_states_the_normal_level_and_the_window_level():
    """'Fell from a typical X to Y' is the sentence an analyst checks against
    a briefed budget. Without both numbers it is not checkable."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    text = explain(event(p, "TV", 60, 89, detection_confidence=0.9,
                         informativeness=0.5), p)
    assert "100" in text


def test_the_explanation_reports_all_three_scores():
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    text = explain(event(p, "TV", 60, 89, detection_confidence=0.94,
                         informativeness=0.81), p)
    assert "0.94" in text
    assert "0.81" in text
    assert "ok" in text


def test_a_suspect_event_says_so_and_gives_the_reason():
    """A validity warning that does not reach the prose is a warning nobody
    reads."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    e = event(p, "TV", 60, 89, detection_confidence=0.9, informativeness=0.5,
              validity="suspect_tracking_loss",
              validity_reasons=("TV: spend is zero but impressions continue",))
    text = explain(e, p)
    assert "suspect_tracking_loss" in text
    assert "impressions continue" in text


def test_a_pulse_train_explanation_states_how_many_windows():
    p = build({"TV": ([100.0] * 20 + [0.0] * 10) * 4 + [100.0] * 30,
               "Radio": [50.0] * 150})
    e = DetectedEvent(
        sid="dev_test", country_code="DE", channel="TV",
        event_type="channel_pulse",
        start=p.dates[20], end=p.dates[109],
        components=((p.dates[20], p.dates[29]), (p.dates[50], p.dates[59])),
        evidence={"n_pulses": 4},
        detection_confidence=0.8, informativeness=0.7)
    text = explain(e, p)
    assert "4" in text
    assert "pulse" in text.lower()


def test_a_step_change_explanation_states_the_ratio():
    p = build({"TV": [100.0] * 75 + [300.0] * 75, "Radio": [50.0] * 150})
    e = event(p, "TV", 75, 149, event_type="step_change",
              magnitude_ratio=3.02, evidence={"z": 6.0},
              detection_confidence=0.7, informativeness=0.4)
    text = explain(e, p)
    assert "3.0" in text


def test_a_step_out_of_zero_does_not_claim_a_ratio():
    """magnitude_ratio is None when the level before the step was zero. The
    prose must say so rather than printing 'None' or inventing a number."""
    p = build({"TV": [0.0] * 40 + [300.0] * 110, "Radio": [50.0] * 150})
    e = event(p, "TV", 40, 149, event_type="step_change",
              magnitude_ratio=None, evidence={"z": 6.0},
              detection_confidence=0.7, informativeness=0.4)
    text = explain(e, p)
    assert "None" not in text


def test_every_event_type_produces_prose_without_raising():
    """The explanation is part of the output schema, so no type may be left
    without one."""
    from detection.model import EVENT_TYPES
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    for event_type in sorted(EVENT_TYPES):
        e = event(p, "TV", 60, 89, event_type=event_type,
                  evidence={"z": 5.0}, detection_confidence=0.5,
                  informativeness=0.5)
        text = explain(e, p)
        assert len(text) > 40, event_type


def test_an_unscored_event_still_explains_itself():
    """Explanation must not require scoring to have run -- it is also the
    debugging surface when scoring is what went wrong."""
    p = build({"TV": [100.0] * 60 + [0.0] * 30 + [100.0] * 60,
               "Radio": [50.0] * 150})
    text = explain(event(p, "TV", 60, 89), p)
    assert len(text) > 40
```

- [ ] **Step 2: Run to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_explain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'detection.explain'`

- [ ] **Step 3: Implement `detection/explain.py`**

```python
"""Spec section 8 -- the templated natural-language explanation.

Every number in the prose comes from the event or the panel. Nothing is
softened and nothing is invented: the register is a colleague pointing at the
data, not a model expressing an opinion. An analyst must be able to check every
claim in the sentence against the export.
"""
from __future__ import annotations

import numpy as np

from detection.io.panel import Panel
from detection.model import DetectedEvent
from detection.score import _subject_channels

_TYPE_PHRASE = {
    "dark_period": "every channel in the market stopped together",
    "single_channel": "all channels but one stopped",
    "natural_holdout": "one channel stopped while the rest kept running",
    "step_change": "spend moved to a new level and held there",
    "channel_pulse": "the channel switched on and off repeatedly",
    "staggered_launch": "the channel started later here than elsewhere",
}


def _typical_level(panel: Panel, country: str, channel: str,
                   start, end) -> float:
    """Median spend on active days OUTSIDE the event window."""
    if channel is None or country is None:
        return float("nan")
    key = (country, channel)
    if key not in panel.spend.columns:
        return float("nan")
    series = panel.spend[key]
    outside = series.drop(series.loc[start:end].index)
    active = outside[outside > 0]
    return float(active.median()) if not active.empty else float("nan")


def explain(event: DetectedEvent, panel: Panel) -> str:
    where = event.country_code or "the panel"
    what = event.channel or "every channel"
    phrase = _TYPE_PHRASE.get(event.event_type, event.event_type)

    parts = [
        f"{what} in {where}: {phrase} "
        f"for {event.n_days} consecutive days "
        f"({event.start.date()} to {event.end.date()})."
    ]

    typical = _typical_level(panel, event.country_code, event.channel,
                             event.start, event.end)
    if np.isfinite(typical):
        parts.append(
            f"Normal spend on this series is about {typical:,.0f} per day "
            f"outside the window.")

    if event.event_type == "step_change":
        if event.magnitude_ratio is None:
            parts.append(
                "Spend rose from zero, so there is no finite ratio to report.")
        else:
            parts.append(
                f"Spend moved to {event.magnitude_ratio:.2f}x its previous "
                f"level.")
        z = event.evidence.get("z")
        if z is not None:
            parts.append(f"The shift measures {abs(float(z)):.1f} robust "
                         f"standard deviations.")

    if event.event_type == "channel_pulse":
        n = event.evidence.get("n_pulses", len(event.components))
        parts.append(
            f"This is one pulse train of {n} separate off-windows, reported "
            f"as a single event because the windows belong to one flighting "
            f"pattern.")

    if event.event_type in {"dark_period", "single_channel"}:
        subjects = _subject_channels(event, panel)
        parts.append(f"{len(subjects)} channels are covered by this claim.")

    control = event.evidence.get("control_available")
    if control == "peers":
        parts.append(
            "Other markets ran this channel throughout, so they are available "
            "as controls.")
    elif control == "sibling_channels":
        parts.append(
            "No peer market ran this channel; the market's other channels are "
            "the only available control.")
    elif control == "none":
        parts.append("No control group is available for this window.")

    label = event.event_type
    if event.tags:
        label += " (" + ", ".join(event.tags) + ")"
    scored = []
    if event.detection_confidence is not None:
        scored.append(f"confidence {event.detection_confidence:.2f}")
    if event.informativeness is not None:
        scored.append(f"informativeness {event.informativeness:.2f}")
    scored.append(f"validity {event.validity}")
    parts.append(f"Label: {label}. " + ", ".join(scored).capitalize() + ".")

    if event.validity_reasons:
        parts.append("Caveat: " + "; ".join(event.validity_reasons) + ".")

    return " ".join(parts)
```

- [ ] **Step 4: Run to verify they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_explain.py -v`
Expected: PASS

- [ ] **Step 5: Print one real explanation and read it**

```bash
PYTHONPATH=. synthetic_data_generator/.venv/bin/python -c "
import pandas as pd
from detection.pipeline import run_detection
media = pd.read_csv('benchmark/datasets/dev/dev_005/media.csv', parse_dates=['date'])
for e in run_detection(media, None, 'dev_005')[:3]:
    print(e.event_type, '->', getattr(e, 'explanation', '(not wired yet)'))
"
```

Read the output. If a sentence would not survive an analyst asking "how do you know?", fix the template. Paste the real output into the task report.

- [ ] **Step 6: Commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/explain.py tests/detection/test_explain.py
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): templated per-event explanations"
```

---

### Task 7: Wire scoring into the pipeline and the adapter

**Files:**
- Modify: `detection/pipeline.py`, `detection/model.py`, `benchmark/eval/adapter.py`
- Test: `tests/detection/test_pipeline.py`, `tests/eval/test_adapter.py`

**Interfaces:**
- Consumes: `confidence`, `informativeness` (Task 3), `assess` (Task 4), `apply_calibration` (Task 5), `explain` (Task 6)
- Produces: every `DetectedEvent` from `run_detection` carries `detection_confidence`, `informativeness`, `validity`, `validity_reasons` and `evidence["explanation"]`; every `Event` from `adapter.detect` carries the two scores.

`DetectedEvent` needs one more field: `explanation: str = ""`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/detection/test_pipeline.py`:

```python
def test_every_detected_event_carries_all_three_scores():
    media, sales = load("dev_005")
    events = run_detection(media, sales, "dev_005")
    assert events
    for e in events:
        assert e.detection_confidence is not None
        assert 0.0 <= e.detection_confidence <= 1.0
        assert e.informativeness is not None
        assert 0.0 <= e.informativeness <= 1.0
        assert e.validity in {"ok", "suspect_data_gap", "suspect_tracking_loss"}


def test_every_detected_event_carries_an_explanation():
    media, sales = load("dev_005")
    for e in run_detection(media, sales, "dev_005"):
        assert len(e.explanation) > 40, e.event_type


def test_events_can_be_ranked_by_informativeness():
    """The brief asks for ranking explicitly. If every event scores the same,
    ranking is decorative."""
    media, sales = load("dev_017")
    events = run_detection(media, sales, "dev_017")
    scores = {round(e.informativeness, 4) for e in events}
    assert len(scores) > 1, "informativeness does not discriminate at all"
```

Append to `tests/eval/test_adapter.py`:

```python
def test_the_adapter_carries_both_scores_across():
    """The harness's reliability and operating curves read
    Event.detection_confidence. Dropping it here renders both degenerate --
    which is exactly what happened before scoring existed."""
    from detection.model import DetectedEvent
    d = DetectedEvent(
        sid="dev_001", country_code="DE", channel="TV",
        event_type="natural_holdout",
        start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-04-01"),
        detection_confidence=0.83, informativeness=0.41)
    e = to_event(d)
    assert e.detection_confidence == 0.83
    assert e.informativeness == 0.41
```

- [ ] **Step 2: Run to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_pipeline.py tests/eval/test_adapter.py -v`
Expected: FAIL — scores are `None`

- [ ] **Step 3: Add the `explanation` field**

In `detection/model.py`, after `validity_reasons`:

```python
    # Spec section 8's output schema. Empty until the pipeline scores the
    # event; the composition layers do not write prose.
    explanation: str = ""
```

- [ ] **Step 4: Wire the pipeline**

In `detection/pipeline.py`, replace the sort block with scoring then sorting:

```python
from dataclasses import replace

from detection.calibrate import apply_calibration
from detection.explain import explain
from detection.score import confidence, informativeness
from detection.validity import assess

try:                                  # pragma: no cover - the frozen fit is
    from detection.calibration_fit import KNOTS   # generated in Task 10
except ImportError:                   # pragma: no cover
    KNOTS = []
```

and, after `events.extend(find_staggered_launches(panel, sid))`:

```python
    scored = []
    for e in events:
        raw, parts = confidence(e, panel)
        info, drivers = informativeness(e, panel)
        verdict, reasons = assess(e, panel)
        evidence = dict(e.evidence)
        evidence.update(confidence_sub_scores=parts,
                        informativeness_drivers=drivers,
                        raw_confidence=raw)
        e = replace(
            e,
            detection_confidence=apply_calibration(raw, KNOTS),
            informativeness=info,
            validity=verdict,
            validity_reasons=reasons,
            evidence=evidence,
        )
        scored.append(replace(e, explanation=explain(e, panel)))
    events = scored
```

Keep the existing `events.sort(...)` after this block.

- [ ] **Step 5: Wire the adapter**

In `benchmark/eval/adapter.py`, in `to_event`, add the two fields and update the module docstring, which currently claims the scores are left `None` because nothing computes them:

```python
        detection_confidence=d.detection_confidence,
        informativeness=d.informativeness,
```

- [ ] **Step 6: Run to verify they pass**

Run: `synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS

- [ ] **Step 7: Confirm the curves are no longer degenerate**

```bash
PYTHONPATH=. synthetic_data_generator/.venv/bin/python benchmark/eval/run_dev.py \
  --detector benchmark.eval.adapter:detect --load-data \
  --label "scores wired, uncalibrated" --out /tmp/dev_scored.md
grep -A12 "Reliability" /tmp/dev_scored.md
grep -A12 "Operating curve" /tmp/dev_scored.md
```

Expected: both tables have rows. Paste them into the task report. The headline F1 must not move — scoring adds fields, it does not change which events are emitted. If F1 moves, something in the wiring changed detection behaviour; find it before continuing.

- [ ] **Step 8: Commit**

```bash
git add detection/ benchmark/eval/adapter.py tests/
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): score, assess and explain every event in the pipeline"
```

---

### Task 8: Recover the missing closing shift (backlog item 1)

**Files:**
- Modify: `detection/primitives/level_shift.py`
- Test: `tests/detection/test_level_shift.py`

Step recall is 2/13 on dev. In 4 of 6 dev step changes the opening shift is dated correctly and the CLOSING shift is never detected by `find_level_shifts` at all. This is the largest single lever in the plan.

**Diagnosis first, fix second.** Do not guess.

- [ ] **Step 1: Reproduce and locate the failure**

```bash
PYTHONPATH=. synthetic_data_generator/.venv/bin/python -c "
import pandas as pd, numpy as np
from detection.io.panel import build_panel
from detection.primitives.level_shift import find_level_shifts
from detection import params
media = pd.read_csv('benchmark/datasets/dev/dev_012/media.csv', parse_dates=['date'])
p = build_panel(media, None, 'dev_012')
for country in p.countries:
    for ch in p.channels_in(country):
        s = p.series(country, ch)
        shifts = find_level_shifts(s)
        if shifts:
            print(country, ch, [(str(x.at.date()), round(x.z,1), round(x.sharpness,2)) for x in shifts])
"
```

Then, for a series whose closing shift is missing, instrument the gate chain exactly as the Plan 3 fix wave did: for every `t`, print `z`, whether persistence passed, and `sharpness`. Identify **which gate rejects the closing shift**. Record the measurement in the task report before writing any fix.

The likely causes, in order — confirm which applies rather than assuming:
1. **Non-maximum suppression.** `find_level_shifts` keeps the strongest candidate in each `W`-wide neighbourhood. A step's closing shift sits `n_days` after its opening shift; if the event is shorter than `W = 21` days the two are in the same neighbourhood and the weaker one is discarded. Dev steps are 42/56/90 days, so check whether the suppression window is genuinely the cause before changing it.
2. **Sharpness at the closing edge.** Adstock decay smears the fall back to baseline over several days, so the close may be a ramp where the open was a cliff.
3. **The `t` range.** `range(params.W, n - params.W)` cannot see a shift within `W` days of either end.

- [ ] **Step 2: Write the failing test**

Write a test reproducing the shape you measured, with the true window asserted. Example form — adjust the fixture to the shape your diagnosis found:

```python
def test_both_edges_of_a_bounded_step_are_detected():
    """Step recall was 2/13 on the development split because the CLOSING shift
    was never detected on four of six events, leaving the episode open-ended
    and dropped. Both edges must be found for the episode to be bounded."""
    rng = np.random.default_rng(31)
    values = (noisy(100.0, 150, rng) + noisy(300.0, 56, rng)
              + noisy(100.0, 150, rng))
    shifts = find_level_shifts(s(values))
    ats = [sh.index for sh in shifts]
    assert any(abs(i - 150) <= 2 for i in ats), f"opening shift missing: {ats}"
    assert any(abs(i - 206) <= 2 for i in ats), f"closing shift missing: {ats}"
```

- [ ] **Step 3: Run to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_level_shift.py -k both_edges -v`
Expected: FAIL, naming the missing shift

- [ ] **Step 4: Fix the cause you identified**

Whatever the fix, it must be expressible as a named parameter in `params.py` with its risk documented, and it must not weaken the ramp or spike defences. Re-run the full `test_level_shift.py` after the change: if `test_a_gradual_ramp_is_rejected_by_sharpness` or `test_a_decaying_excursion_is_rejected_by_persistence` now fails, the fix traded precision for recall and must be reconsidered.

- [ ] **Step 5: Measure on dev**

```bash
find detection tests -name __pycache__ -type d -exec rm -rf {} +
PYTHONPATH=. synthetic_data_generator/.venv/bin/python benchmark/eval/run_dev.py \
  --detector benchmark.eval.adapter:detect --load-data \
  --label "closing-shift fix" --out /tmp/dev_step.md
grep -E "^\| (Precision|Recall|F1|False positives) |^\| step_change " /tmp/dev_step.md
```

Report step_change precision/recall/F1 before and after, and the overall F1 and null FP rate. **A recall gain that costs null-scenario false positives is not a gain** — the null FP rate is the number that separates a real detector from an indiscriminate one, and it is currently 0.000.

- [ ] **Step 6: Commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/ tests/
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "fix(detection): detect the closing shift of a bounded step change"
```

---

### Task 9: Collapse market-wide steps (backlog item 2)

**Files:**
- Modify: `detection/compose/label.py`, `detection/params.py`
- Test: `tests/detection/test_label.py`

Task 8 will bring back a precision problem it currently masks: `dev_035/AT` fires a step on all 11 channels on the same day (`dev_030/DE` 7, `dev_033/FI` 3). These are suppressed today only as a side effect of dropping open-ended episodes. A move that hits every channel in a market on one day is a **market-level event**, not N independent channel events.

Do this task **after** Task 8 and re-measure, because Task 8 is what makes the problem live again.

- [ ] **Step 1: Confirm the problem is live**

```bash
PYTHONPATH=. synthetic_data_generator/.venv/bin/python -c "
import pandas as pd, collections
from benchmark.eval.adapter import detect
for sid in ['dev_035','dev_030','dev_033']:
    media = pd.read_csv(f'benchmark/datasets/dev/{sid}/media.csv', parse_dates=['date'])
    events = detect(media, None, sid)
    groups = collections.Counter(
        (e.country_code, e.start) for e in events if e.event_type=='step_change')
    worst = groups.most_common(3)
    print(sid, worst)
"
```

If no `(market, day)` group has 3 or more channels, the problem is not live: record that measurement and SKIP this task rather than building a rule with no input. Say so in the report.

- [ ] **Step 2: Add the parameter**

```python
# A step that lands on this fraction of a market's channels on the same day is
# a market-level budget move, not N independent channel decisions. Reporting it
# N times is both wrong and a precision disaster on wide markets.
MARKET_WIDE_STEP_FRACTION = 0.6
```

- [ ] **Step 3: Write the failing test**

```python
def test_a_step_hitting_most_of_a_market_is_reported_once():
    """dev_035/AT fired a step on all eleven channels on one day. That is one
    market-level budget move; eleven events is eleven chances to be wrong and
    one right answer buried among them."""
    n = 300
    spend = {}
    for ch in ["A", "B", "C", "D", "E"]:
        spend[ch] = [100.0] * 150 + [300.0] * 90 + [100.0] * 60
    p = build(spend)
    steps = [e for e in label_market(p, "DE", "dev_test")
             if e.event_type == "step_change"]
    assert len(steps) == 1, f"expected one market-level step, got {len(steps)}"
    assert steps[0].channel is None
    assert steps[0].evidence.get("n_channels") == 5


def test_a_step_on_one_channel_of_five_stays_channel_level():
    """The collapse must not swallow a genuine single-channel step."""
    spend = {ch: [100.0] * 300 for ch in ["A", "B", "C", "D"]}
    spend["E"] = [100.0] * 150 + [300.0] * 90 + [100.0] * 60
    p = build(spend)
    steps = [e for e in label_market(p, "DE", "dev_test")
             if e.event_type == "step_change"]
    assert len(steps) == 1
    assert steps[0].channel == "E"
```

- [ ] **Step 4: Run to verify they fail, implement the collapse in `_step_events`, run to verify they pass**

Group the step episodes by `(start day, direction)`; when a group covers at least `MARKET_WIDE_STEP_FRACTION` of `panel.channels_in(country)`, emit ONE event with `channel=None` and `evidence["n_channels"]` set, instead of one per channel.

- [ ] **Step 5: Mutation-verify and measure on dev**

DELETE the collapse branch and confirm `test_a_step_hitting_most_of_a_market_is_reported_once` fails. Then re-run `run_dev` and report step precision/recall and the null FP rate before and after.

- [ ] **Step 6: Commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/ tests/
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "fix(detection): a market-wide step is one event, not one per channel"
```

---

### Task 10: Fit and freeze the calibration

**Files:**
- Create: `detection/calibration_fit.py` (generated), `scripts/fit_calibration.py`
- Test: `tests/detection/test_calibration_fit.py`

This is the last task that may change behaviour. After it, the algorithms are FROZEN.

- [ ] **Step 1: Write the fitting script**

Create `scripts/fit_calibration.py`. It lives outside `detection/` because it imports the harness to learn which detections matched — something no file under `detection/` is allowed to do.

```python
"""Fit the confidence calibration on the DEVELOPMENT split and freeze it.

Spec section 8: bin confidence, measure empirical precision per bin, fit a
monotone PAV mapping, freeze it before the final run. Fitting on the test split
would be grading your own exam, so this script reads the development split only.

Usage:
    PYTHONPATH=. synthetic_data_generator/.venv/bin/python scripts/fit_calibration.py
"""
import pathlib
import sys

import pandas as pd

from benchmark.eval.matching import match_events
from benchmark.eval.truth import list_scenarios, load_scenario, load_truth
from benchmark.eval.adapter import detect
from detection.calibrate import fit_pav, render_module
from detection import params

OUT = pathlib.Path("detection/calibration_fit.py")


def main() -> int:
    scores, correct = [], []
    for sid in list_scenarios("dev"):
        media, sales = load_scenario(sid, "dev")
        pred = detect(media, sales, sid)
        truth = load_truth(sid, "dev")
        result = match_events(truth, pred, min_iou=0.5, strict=True)
        matched = {id(m.pred) for m in result.matches}
        for p in pred:
            if p.detection_confidence is None:
                continue
            scores.append(p.detection_confidence)
            correct.append(id(p) in matched)

    knots = fit_pav(scores, correct, n_bins=params.CALIBRATION_BINS)
    OUT.write_text(render_module(knots))
    print(f"fitted on {len(scores)} detections, {sum(correct)} correct")
    print(f"wrote {OUT} with {len(knots)} knots")
    for upper, value in knots:
        print(f"  <= {upper:.2f} -> {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add to `detection/params.py`:

```python
# Calibration bins. Spec section 8 says deciles; fewer bins makes each estimate
# steadier but the mapping coarser.
CALIBRATION_BINS = 10
```

**Important:** the script must run against the pipeline with `KNOTS = []` (the raw scores), or it calibrates an already-calibrated score. Before fitting, confirm `detection/calibration_fit.py` does not exist, or delete it.

- [ ] **Step 2: Run the fit**

```bash
rm -f detection/calibration_fit.py
find detection tests -name __pycache__ -type d -exec rm -rf {} +
PYTHONPATH=. synthetic_data_generator/.venv/bin/python scripts/fit_calibration.py
```

Paste the printed knots into the task report.

- [ ] **Step 3: Write the tests**

Create `tests/detection/test_calibration_fit.py`:

```python
import pathlib

from detection.calibrate import apply_calibration
from detection.calibration_fit import KNOTS


def test_the_frozen_fit_is_monotone():
    values = [v for _, v in KNOTS]
    assert values == sorted(values), values


def test_the_frozen_fit_is_python_source_not_a_data_file():
    """detection/ may contain nothing but .py -- a JSON fit would be invisible
    to both the import gate and the final-run audit hash."""
    assert pathlib.Path("detection/calibration_fit.py").suffix == ".py"


def test_calibrated_confidence_stays_in_range():
    for s in [0.0, 0.1, 0.35, 0.6, 0.85, 1.0]:
        assert 0.0 <= apply_calibration(s, KNOTS) <= 1.0
```

- [ ] **Step 4: Verify the reliability curve improved**

```bash
PYTHONPATH=. synthetic_data_generator/.venv/bin/python benchmark/eval/run_dev.py \
  --detector benchmark.eval.adapter:detect --load-data \
  --label "calibrated" --out /tmp/dev_calibrated.md
grep -A14 "Reliability" /tmp/dev_calibrated.md
```

Compare each bin's stated confidence against its empirical precision, before and after. Report the mean absolute gap for both. Calibration that does not reduce it has failed and must be investigated, not shipped.

Note honestly in the report: this curve is measured on the same split the mapping was fitted on, so it is optimistic by construction. The test-split curve in Task 11 is the real one.

- [ ] **Step 5: Commit**

```bash
synthetic_data_generator/.venv/bin/python -m pytest -q -m "not slow"
git add detection/ scripts/fit_calibration.py tests/
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "feat(detection): freeze the dev-fitted confidence calibration"
```

---

### Task 11: The single gated final run

**Files:**
- Creates: `benchmark/eval/final_runs.jsonl` (by running the gate), `REPORT_final_metrics.md`

**THIS TASK RUNS ONCE AND CANNOT BE UNDONE.** Every run is permanently recorded in `final_runs.jsonl` and a second run is banner-flagged in the report forever. Do not reach this task until Tasks 1–10 are complete, reviewed, and the suite is green.

- [ ] **Step 1: Pre-flight — verify the algorithms are frozen**

```bash
git status --short                      # must be empty
synthetic_data_generator/.venv/bin/python -m pytest -q                 # full suite incl. slow
PYTHONPATH=. synthetic_data_generator/.venv/bin/python -c "
from benchmark.harness.seal import verify_seal
import pathlib
ok, problems = verify_seal('test')
print('seal:', ok, problems)
print('final_runs.jsonl exists:', pathlib.Path('benchmark/eval/final_runs.jsonl').exists())
"
```

ALL of these must hold: working tree clean, full suite green, `seal: True` with no problems, `final_runs.jsonl exists: False`. If any fails, STOP and fix it. Running the gate with a dirty tree records a hash that does not correspond to any commit.

- [ ] **Step 2: Record the frozen state**

```bash
git rev-parse HEAD
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  tag -a algorithms-frozen -m "State of the detector at the single final benchmark run"
```

- [ ] **Step 3: Run the gate — ONCE**

```bash
PYTHONPATH=. synthetic_data_generator/.venv/bin/python benchmark/eval/run_final.py \
  --detector benchmark.eval.adapter:detect --load-data --finalize \
  --out REPORT_final_metrics.md
```

- [ ] **Step 4: Verify the audit record**

```bash
cat benchmark/eval/final_runs.jsonl
```

Expected: exactly ONE line, carrying the detector source hash, the params hash and the metrics. If there is more than one line, say so in the report and in `REPORT.md` — the banner is the honest outcome, not something to work around.

- [ ] **Step 5: Extract the numbers for the handover**

```bash
grep -E "^\| " REPORT_final_metrics.md | head -60
```

Record: overall P/R/F1; per-type P/R/F1; mean and median IoU; boundary error median and p90; channel and market accuracy; day-level F1; type confusion; null-scenario FP per country-year; the reliability curve; the operating curve; and every breakdown. Compare each against the dev numbers and note where the test split is worse — that gap is the generalization result and it is the single most informative thing in the report.

- [ ] **Step 6: Commit**

```bash
git add benchmark/eval/final_runs.jsonl REPORT_final_metrics.md
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "chore(eval): the single gated final run against the sealed test split"
```

---

### Task 12: `REPORT.md`, the handover

**Files:**
- Create: `REPORT.md`

Delivers the six items the brief asked for, plus spec section 11's requirements. Written for a Sellforte reader who has not seen the code.

- [ ] **Step 1: Write the report**

Required sections:

1. **What this detects and how to run it** — the end-to-end pipeline in one diagram and one command.
2. **Final algorithm design, per event type** — dark period, single-channel, natural holdout, step change, channel pulse, staggered launch. For each: the primitives it composes, the parameters that govern it, and its test-split numbers.
3. **Benchmark results on the unseen test set** — the full section 9 metric set from Task 11, with the dev-versus-test comparison stated plainly. Lead with the **null-scenario false-positive rate**, not F1: a silent detector and an indiscriminate one both score 0.000 F1, and only that rate separates them.
4. **Reliability and operating curves** — the calibration claim ("events at confidence X are right about X% of the time on data never seen") tested against the test split, plus the recommended operating cut with its precision/recall trade-off.
5. **Important parameters and their sensitivity** — all parameters in `params.py`, what each controls, and the measured cost of getting it wrong.
6. **Failure modes** — every item on the Plan 4 backlog that remains open, with its measured cost.
7. **Edge-case behaviour** — censored windows, the 5-day event below `MIN_DAYS`, near-zero versus exact-zero, back-to-back and overlapping events, single-channel markets, the all-zero channel.
8. **Production-readiness verdict, per detector** — graded honestly against the evidence, not against the expectation. Spec section 11's going-in expectation was: dark, holdout and single-channel production-ready; step change production-ready with a ramp caveat; pulse and staggered launch promising. State where the measurements disagree with that.
9. **Development trajectory** — `dev_history.jsonl` as evidence that tuning happened on the dev split and only there.
10. **Recommendations for the real Sellforte dataset** — spec section 11's five points: missing-row versus zero-spend as the largest risk; aggregate campaigns to channel level before detection; holiday calendars producing all-channel pauses; validating without ground truth; running at a chosen confidence cut and triaging top-N by informativeness.

- [ ] **Step 2: Write the honesty section**

A section titled "What this benchmark cannot tell you", stating plainly:

- **The edge family's event windows are hardcoded offsets** (`holdout(c0, ch0, 0, 60)`, `holdout(c0, ch0, n_days-60, 60)`, `holdout(…, 200, 42) + step(…, 242, 56)`, ramp at 200, global pause at 300) and always target `countries[0]` / `channels[0..1]`. Only country codes, channel names and noise differ between splits. **Ten of the 55 test scenarios therefore have event locations the dev split already discloses.** Not exploited, but it caps what "unseen" means for that family.
- **`RHO` was changed from the spec's value** based on dev measurement — the spec set it equal to the top of the near-zero band the spec itself defines, so near-zero holdouts sat exactly on the threshold. Raising it took holdout recall 0.550 → 0.800 and step false positives 4 → 0. Logged in `dev_history.jsonl`.
- **Two-channel markets are label-ambiguous in the truth itself.** The same spend shape is `single_channel` in `dev_008/009/010` and `natural_holdout` in `dev_011/013`, depending only on which scenario family drew it. The detector's rule is a prior, not a reading, and it is deliberately not fitted to those scenarios.
- **`dev_036` and `dev_022` are the same data with different truth labels** (`natural_holdout` with `censored_start` versus `staggered_launch`). No spend-only rule separates them.
- **A regular pulse train of six or more windows is suppressed entirely** by the P1 intermittent guard. Faithful to spec section 7; the conflict is in the spec. The benchmark's pulse family draws 2–4 windows so no score reveals it — **but on real flighting data this is the normal case.** Highest-priority question before production use.
- **Open-ended step episodes are dropped.** Safe on this benchmark, where every generated step reverts inside the series. A real budget change that never reverts would be missed.
- **Adstock half-life is assumed, not measured** (`ADSTOCK_HALF_LIFE`), and feeds informativeness ranking only — never detection.

- [ ] **Step 3: Verify every number in the report**

Every figure in `REPORT.md` must be traceable to `REPORT_final_metrics.md`, `dev_history.jsonl`, or `params.py`. Re-read the report and check each one. A wrong number in a handover is worse than a missing one.

- [ ] **Step 4: Commit**

```bash
git add REPORT.md
git -c user.name="pipiland2612" -c user.email="nguyen.t.dang.minh@gmail.com" \
  commit -m "docs: handover report"
```

---

## Self-review

**Spec coverage.** Section 8: the three scores are Tasks 2–4, calibration is Tasks 5 and 10, explanation is Task 6, the output schema is Task 7. Section 9 items 8–9 (reliability and operating curves) are unblocked by Task 7 and reported in Tasks 10–11. Section 10's iteration protocol governs Tasks 8–9 (dev only) and Task 11 (the single gated run). Section 11's handover is Task 12, with its five real-data recommendations enumerated. Section 12's build order step 6 (scoring) and step 7 (final run and report) are Tasks 1–10 and 11–12 respectively.

**One gap accepted deliberately.** Spec section 8's `validity` lists a fourth trigger — "a spend drop with no sales response in a window where sales SNR was adequate to show one". Task 4 implements the first three. The fourth needs a sales-SNR estimator that exists nowhere in the codebase, and building one to serve a single validity flag is scope this plan does not need: the benchmark's sales series are generated from the same spend the detector already reads, so the check would be circular here and could not be validated. Task 12 records it as not implemented, with that reasoning, rather than shipping an unvalidated flag.

**Placeholder scan.** No TBDs. Task 8 deliberately specifies *diagnosis before fix* rather than prescribing a fix, because the cause is measurable and guessing it is how the Plan 3 pairing fix went wrong twice; the three candidate causes are enumerated with the measurement that distinguishes them. Task 9 carries an explicit skip condition with the command that decides it.

**Type consistency.** `sub_scores(event, panel) -> dict[str, float]` (Task 2) is consumed by `confidence` (Task 3), which returns `(float, dict)`; `informativeness` matches that shape. `assess(event, panel) -> (str, tuple[str, ...])` (Task 4) feeds `validity` / `validity_reasons` (Task 1). `fit_pav(scores, correct, n_bins) -> list[tuple[float, float]]` (Task 5) feeds `apply_calibration(score, knots)` (Task 5) and `render_module(knots)` (Task 5), whose output is `detection/calibration_fit.py`'s `KNOTS` (Task 10), imported by `pipeline.py` (Task 7). `explain(event, panel) -> str` (Task 6) fills `DetectedEvent.explanation` (Task 7). `_subject_channels` is defined once in `score.py` and reused by `validity.py` and `explain.py` rather than reimplemented.

**Ordering constraint.** Task 9 must follow Task 8 — Task 8 is what makes the market-wide step problem live again. Task 10 must follow both, since calibration must be fitted on the final behaviour. Task 11 must follow Task 10 and must be the last behaviour-affecting action taken.
