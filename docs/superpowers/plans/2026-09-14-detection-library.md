# Detection Library Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the detector — the actual deliverable of this project — that finds informative periods in marketing data using only spend, and score it against the frozen benchmark through the harness built in Plan 2.

**Architecture:** `detection/` loads `media.csv` into a complete daily panel with a presence mask, runs four independent per-series primitives (off-runs, level shifts, pulses, onsets), then composes them into labelled events by segmenting each market's timeline on its active-channel set, and finally annotates them with cross-market control availability. It has **no dependency on the evaluation harness**; a thin adapter in `benchmark/eval/` converts its output for scoring.

**Tech Stack:** Python 3.14, pandas 3.0.5, numpy 2.5.2, pytest. Virtualenv at `synthetic_data_generator/.venv`.

**Spec:** `docs/superpowers/specs/2026-09-13-informative-periods-detection-design.md` — sections 5 and 7 are binding for this plan. Also read `benchmark/eval/README.md`, which states the output conventions the harness expects.

## Global Constraints

- **Pure pandas and numpy.** No scipy, sklearn, or ruptures anywhere. pytest is test-only.
- **`detection/` must never import from `benchmark.eval`.** `tests/eval/test_gating.py` asserts no file under `detection/` contains the substrings `benchmark.eval`, `_truth` or `ground_truth`, and that guard activates the moment `detection/` exists. `detection/` defines its own `DetectedEvent`; the harness adapts it. The dependency points harness → detector, never the reverse.
- **Never read anything under `benchmark/datasets/test_truth/`.** Development and iteration use the `dev` split only.
- **Never run `run_final.py --finalize`.** `benchmark/eval/final_runs.jsonl` must not exist when this plan ends. The final evaluation belongs to Plan 4.
- **Never modify, regenerate or delete anything under `benchmark/datasets/`.** The test split is sealed; `verify_seal('test')` must return OK at the end of every task.
- **Intervals are inclusive on both ends**, `[start, end]`, as `pandas.Timestamp` at day resolution.
- **Every threshold lives in `detection/params.py`.** No magic numbers in logic modules.
- Python interpreter is always `synthetic_data_generator/.venv/bin/python`. Never bare `python3`.
- All paths relative to the project root. Work happens on branch `detection-library`, created off `evaluation-harness`. Do not merge to `main`.

## What the detector actually sees

Verified on disk. `media.csv` has one row per campaign per day: `date, ad_platform, advertising_channel, campaign_name, campaign_id, media_investment, clicks, impressions, conversions, conversion_value, country_code`. In the synthetic benchmark there is exactly one campaign per `(country, channel, date)` — but the panel loader still aggregates, because the real Sellforte export is campaign-grained and spec §11 names that as the top real-data risk.

Series run 730 days, 2024-01-01 to 2025-12-30. Scenarios carry 1–8 countries and 2–12 channels. An injected `dark_period` on `dev_005` shows **exact zeros for every channel** across the window, with spend around 15,000 the day before and 11,800 the day after — sharp edges, no ramp.

## Two spec items deliberately NOT in this plan

Stating them so they are not silently missing:

- **Binary-segmentation boundary refinement.** Spec §7 says P2's boundaries "are
  then refined by binary segmentation under the same robust cost". This plan
  emits boundaries at the change-point index instead. That costs boundary
  accuracy, which is a reported metric — so it belongs with Plan 4's iteration,
  where the dev-split boundary error will show whether it is worth the code.
- **`censored_start` / `censored_end` flags.** Spec §8's output schema carries
  them. `find_discontinuation` is built here (Task 6) and consumed by Plan 4,
  which owns the output schema and the scores.

## Baselines to beat

Measured on the dev split by Plan 2:

| detector | F1 | null FP / country-year |
|---|---|---|
| `perfect_oracle` | 1.000 | 0.000 |
| `never_detect` | 0.000 | 0.000 |
| `detect_everything` | 0.000 | 0.654 |

`never_detect` has a flawless false-positive rate, so beating the FP rate alone is meaningless — F1 and FP rate must both be read.

## File Structure

| file | responsibility |
|---|---|
| `detection/model.py` | `DetectedEvent` — the detector's own output type, no harness dependency |
| `detection/params.py` | every threshold, one file, each documented with its failure mode |
| `detection/io/panel.py` | `media.csv` + `sales.csv` → a complete daily panel with a presence mask |
| `detection/io/normalize.py` | the three scales of spec §5: within-series, within-market, cross-market |
| `detection/primitives/zero_runs.py` | P1 — off-runs, distribution-relative notability |
| `detection/primitives/level_shift.py` | P2 — robust change points, persistence and sharpness filters |
| `detection/primitives/pulse.py` | P3 — grouping repeated off-runs into one pulse train |
| `detection/primitives/onset.py` | P4 — series-start and series-end runs |
| `detection/compose/label.py` | regime segmentation on the active-channel set, and labelling |
| `detection/compose/cross_market.py` | peer comparison, control availability, launch fan-out |
| `detection/pipeline.py` | `run_detection(media_df, sales_df, sid) -> list[DetectedEvent]` |
| `benchmark/eval/adapter.py` | `DetectedEvent` → harness `Event`; the scoreable detector callable |
| `tests/detection/…` | one test module per source module |

---

### Task 1: Branch, scaffold, event model and parameters

**Files:**
- Create: `detection/__init__.py`, `detection/model.py`, `detection/params.py`
- Create: `detection/io/__init__.py`, `detection/primitives/__init__.py`, `detection/compose/__init__.py`
- Create: `tests/detection/__init__.py`, `tests/detection/test_model.py`, `tests/detection/test_params.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) DetectedEvent` with `sid: str`, `country_code: str | None`, `channel: str | None`, `event_type: str`, `start: pd.Timestamp`, `end: pd.Timestamp`, `magnitude_ratio: float | None = None`, `components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()`, `tags: tuple[str, ...] = ()`, `evidence: dict = field(default_factory=dict)`, plus an `n_days` property
  - `EVENT_TYPES: frozenset[str]`
  - `detection.params` module-level constants: `RHO`, `EPS_ABS`, `MIN_DAYS`, `RUN_RATIO`, `MIN_RUNS_FOR_RATIO`, `W`, `Z_THRESH`, `SIGMA_FLOOR`, `PERSIST`, `SHARPNESS`, `SHARPNESS_WINDOW`, `ONSET_SPREAD`, `PULSE_MIN_RUNS`, `PULSE_LEN_IQR_RATIO`, `ROLLING`
  - Every later task consumes both.

- [ ] **Step 1: Create the branch and scaffold**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git checkout evaluation-harness
git checkout -b detection-library
mkdir -p detection/io detection/primitives detection/compose tests/detection
touch detection/__init__.py detection/io/__init__.py \
      detection/primitives/__init__.py detection/compose/__init__.py \
      tests/detection/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/detection/test_model.py`:

```python
import pandas as pd
import pytest

from detection.model import EVENT_TYPES, DetectedEvent


def ev(**kw):
    base = dict(sid="dev_001", country_code="DE", channel="TV",
                event_type="natural_holdout",
                start=pd.Timestamp("2024-03-01"), end=pd.Timestamp("2024-03-10"))
    base.update(kw)
    return DetectedEvent(**base)


def test_n_days_is_inclusive():
    assert ev().n_days == 10
    assert ev(end=pd.Timestamp("2024-03-01")).n_days == 1


def test_is_frozen():
    with pytest.raises(Exception):
        ev().start = pd.Timestamp("2024-01-01")


def test_optional_fields_default_sensibly():
    e = ev()
    assert e.magnitude_ratio is None
    assert e.components == () and e.tags == ()
    assert e.evidence == {}


def test_evidence_is_per_instance_not_shared():
    """A mutable default shared across instances would silently merge the
    evidence of every detection in a run."""
    a, b = ev(), ev()
    a.evidence["x"] = 1
    assert b.evidence == {}


def test_event_types_cover_everything_the_composer_can_emit():
    assert EVENT_TYPES == frozenset({
        "dark_period", "single_channel", "natural_holdout",
        "step_change", "channel_pulse", "staggered_launch",
    })


def test_channel_is_none_for_market_wide_events():
    e = ev(channel=None, event_type="dark_period")
    assert e.channel is None


def test_detection_model_does_not_import_the_harness():
    """The black-box boundary: detection/ must never reach into benchmark.eval."""
    import inspect

    import detection.model as m
    src = inspect.getsource(m)
    assert "benchmark.eval" not in src
    assert "ground_truth" not in src
```

Create `tests/detection/test_params.py`:

```python
from detection import params


def test_every_documented_parameter_exists_with_the_spec_value():
    """Spec section 7's parameter table. These are the defaults the design was
    reasoned about; changing one is a deliberate act, not a typo."""
    assert params.RHO == 0.05
    assert params.MIN_DAYS == 7
    assert params.RUN_RATIO == 3.0
    assert params.W == 21
    assert params.Z_THRESH == 3.5
    assert params.SIGMA_FLOOR == 0.05
    assert params.PERSIST == 14
    assert params.SHARPNESS == 0.6
    assert params.ONSET_SPREAD == 14
    assert params.MAD_TO_SIGMA == 1.4826


def test_window_parameters_are_whole_weeks():
    """W and ROLLING are multiples of 7 so day-of-week structure cancels
    instead of aliasing into the level estimate."""
    assert params.W % 7 == 0
    assert params.ROLLING % 7 == 0


def test_no_logic_module_hardcodes_a_threshold():
    """Every threshold lives here. A magic number in a primitive is how two
    parameters silently drift apart."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "detection"
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "params.py":
            continue
        text = py.read_text()
        for needle in ["0.05", "3.5", "0.6", "1.4826"]:
            if needle in text:
                offenders.append(f"{py.name}: {needle}")
    assert not offenders, f"hardcoded thresholds: {offenders}"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection -v`
Expected: FAIL — `No module named 'detection.model'`.

- [ ] **Step 4: Write `detection/model.py`**

```python
"""The detector's own output type.

Deliberately independent of the evaluation harness. `detection/` is the
deliverable; the harness is scaffolding that scores it, and a detector that
imported its own scorer's vocabulary could not be trusted to be blind to the
answers. tests/eval/test_gating.py enforces that boundary by asserting no file
here mentions benchmark.eval or any truth loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EVENT_TYPES: frozenset[str] = frozenset({
    "dark_period",
    "single_channel",
    "natural_holdout",
    "step_change",
    "channel_pulse",
    "staggered_launch",
})


@dataclass(frozen=True)
class DetectedEvent:
    """One informative period the detector believes it has found.

    Intervals are INCLUSIVE on both ends. `channel` is None for an event that
    covers every channel in a market (a dark period); `country_code` is None
    only for a panel-level event, which the cross-market layer fans out before
    the event reaches the harness.
    """
    sid: str
    country_code: str | None
    channel: str | None
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    magnitude_ratio: float | None = None
    components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = ()
    tags: tuple[str, ...] = ()
    evidence: dict = field(default_factory=dict)

    @property
    def n_days(self) -> int:
        return int((self.end - self.start).days) + 1
```

- [ ] **Step 5: Write `detection/params.py`**

```python
"""Every threshold the detector uses, with the failure mode of getting it wrong.

Defaults come from spec section 7's parameter table. They were chosen from
first principles -- weekly multiples, a conventional robust cut -- and are to be
adjusted ONLY against the development split, never the sealed test split.
"""

# --- P1, off-runs -----------------------------------------------------------

# A day counts as "off" when spend <= max(EPS_ABS, RHO * active_level).
# Too low misses near-zero events (the benchmark injects holdouts at 0.02-0.08x);
# too high reads ordinary low-spend days as off.
RHO = 0.05

# Absolute floor, so float noise around zero cannot register as spend.
EPS_ABS = 1e-6

# Shortest reportable event. At 7 the benchmark's deliberate 5-day edge case is
# undetectable and should surface as an honest false negative; lowering it to
# catch that case floods the output with noise.
MIN_DAYS = 7

# A run is notable if it is also RUN_RATIO times the series' own p90 off-run.
# This is the intermittent-channel guard: a flighting channel whose normal gaps
# are 3 days needs ~9 off-days to register, while a channel that is otherwise
# never off needs only MIN_DAYS. A single global threshold fails one of those
# two cases whichever value is picked.
RUN_RATIO = 3.0

# Below this many other runs the series has no usable distribution, so the
# ratio rule is skipped and MIN_DAYS alone applies.
MIN_RUNS_FOR_RATIO = 5

# --- P2, level shifts -------------------------------------------------------

# Half-width of the comparison windows, in days. A multiple of 7 so day-of-week
# structure cancels. Shorter gives a noisier z; longer misses short steps.
W = 21

# Robust z threshold. The dominant precision/recall lever for step changes.
Z_THRESH = 3.5

# Lower bound on the MAD scale, so a near-constant window cannot produce an
# unbounded z.
SIGMA_FLOOR = 0.05

# The new level must still hold this many days later, at half the original
# delta. Rejects spikes.
PERSIST = 14

# Fraction of the total level change that must land inside SHARPNESS_WINDOW.
# THIS is the gradual-ramp defence: a genuine step concentrates its change into
# a couple of days (sharpness near 1) while a 50-day ramp spreads it out
# (near 0.1). Persistence alone does NOT reject a ramp, because a ramp's new
# level genuinely does hold.
SHARPNESS = 0.6
SHARPNESS_WINDOW = 3

# Centred rolling median width used for level work, to remove day-of-week
# structure. Run detection stays on raw daily values so boundaries land on
# exact dates.
ROLLING = 7

# Median-absolute-deviation to standard-deviation factor for a normal
# distribution. Lives here because tests/detection/test_params.py asserts no
# logic module hardcodes a constant.
MAD_TO_SIGMA = 1.4826

# --- P3, pulses -------------------------------------------------------------

# Minimum notable off-runs on one series to call it a pulse train.
PULSE_MIN_RUNS = 2

# Runs must be of similar length: IQR of lengths divided by median <= this.
PULSE_LEN_IQR_RATIO = 0.5

# --- P4, onsets and the cross-market layer ----------------------------------

# Spread in onset dates across markets, in days, above which the channel is
# reported as a staggered launch rather than coincidental start-up jitter.
ONSET_SPREAD = 14
```

- [ ] **Step 6: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection -v`
Expected: 10 passed.

`test_no_logic_module_hardcodes_a_threshold` passes vacuously now (no logic modules exist) and becomes load-bearing from Task 4 onward. That is deliberate — it is a standing guard, not a one-off check.

- [ ] **Step 7: Confirm the black-box guard now activates**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/eval/test_gating.py -v`
Expected: all pass. `test_detection_package_has_no_import_path_to_truth` previously returned early because `detection/` did not exist; it now actually scans. Confirm in the output that it ran rather than skipped.

- [ ] **Step 8: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection tests/detection
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): add the detector's event model and parameter table"
```

---

### Task 2: The panel

**Files:**
- Create: `detection/io/panel.py`
- Test: `tests/detection/test_panel.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `@dataclass(frozen=True) Panel` with `sid: str`, `dates: pd.DatetimeIndex`, `spend: pd.DataFrame`, `present: pd.DataFrame`, `impressions: pd.DataFrame`, `clicks: pd.DataFrame`, `sales: pd.DataFrame`
  - `Panel.countries -> list[str]`, `Panel.channels -> list[str]`, `Panel.series(country, channel) -> pd.Series`, `Panel.present_mask(country, channel) -> pd.Series`, `Panel.channels_in(country) -> list[str]`
  - `build_panel(media_df, sales_df=None, sid="") -> Panel`
  - Tasks 3, 4, 5, 6, 7, 8 and 9 all consume `Panel`.

- [ ] **Step 1: Write the failing test**

Create `tests/detection/test_panel.py`:

```python
import numpy as np
import pandas as pd

from detection.io.panel import build_panel


def media(rows):
    return pd.DataFrame(rows, columns=[
        "date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code",
    ]).astype({"date": "datetime64[ns]"})


def row(date, country, channel, spend, clicks=1.0, impr=10.0, campaign="c1"):
    return [pd.Timestamp(date), "P", channel, campaign, 1, spend, clicks,
            impr, 0, 0.0, country]


def test_campaigns_are_aggregated_to_channel_level():
    """Synthetic data has one campaign per channel-day, but the real Sellforte
    export is campaign-grained and spec section 11 calls this the top real-data
    risk: a campaign ending is NOT a channel holdout."""
    m = media([
        row("2024-01-01", "DE", "TV", 100.0, campaign="a"),
        row("2024-01-01", "DE", "TV", 150.0, campaign="b"),
    ])
    p = build_panel(m)
    assert p.series("DE", "TV").loc[pd.Timestamp("2024-01-01")] == 250.0


def test_a_campaign_ending_is_not_a_channel_holdout():
    """Two campaigns; one stops. The channel keeps spending, so the panel must
    show continuous spend rather than a gap."""
    rows = []
    for d in pd.date_range("2024-01-01", periods=4):
        rows.append(row(d, "DE", "TV", 100.0, campaign="a"))
        if d < pd.Timestamp("2024-01-03"):
            rows.append(row(d, "DE", "TV", 50.0, campaign="b"))
    p = build_panel(media(rows))
    assert list(p.series("DE", "TV")) == [150.0, 150.0, 100.0, 100.0]
    assert p.present_mask("DE", "TV").all()


def test_missing_days_are_reindexed_and_flagged_absent():
    """The single biggest real-data failure mode: a missing row and a zero-spend
    row look identical once reindexed, unless presence is tracked separately."""
    m = media([
        row("2024-01-01", "DE", "TV", 100.0),
        row("2024-01-04", "DE", "TV", 120.0),
    ])
    p = build_panel(m)
    s = p.series("DE", "TV")
    assert len(s) == 4
    assert s.loc[pd.Timestamp("2024-01-02")] == 0.0
    present = p.present_mask("DE", "TV")
    assert present.loc[pd.Timestamp("2024-01-01")]
    assert not present.loc[pd.Timestamp("2024-01-02")]


def test_an_explicit_zero_is_present_but_a_missing_row_is_not():
    m = media([
        row("2024-01-01", "DE", "TV", 0.0),
        row("2024-01-03", "DE", "TV", 5.0),
    ])
    p = build_panel(m)
    present = p.present_mask("DE", "TV")
    assert present.loc[pd.Timestamp("2024-01-01")]      # explicit zero
    assert not present.loc[pd.Timestamp("2024-01-02")]  # absent row


def test_the_date_grid_is_complete_and_shared_across_series():
    m = media([
        row("2024-01-01", "DE", "TV", 10.0),
        row("2024-01-05", "AT", "Radio", 20.0),
    ])
    p = build_panel(m)
    assert len(p.dates) == 5
    for c, ch in [("DE", "TV"), ("AT", "Radio")]:
        assert p.series(c, ch).index.equals(p.dates)


def test_channels_in_lists_only_that_markets_channels():
    m = media([
        row("2024-01-01", "DE", "TV", 10.0),
        row("2024-01-01", "AT", "Radio", 20.0),
    ])
    p = build_panel(m)
    assert p.countries == ["AT", "DE"]
    assert p.channels_in("DE") == ["TV"]
    assert p.channels_in("AT") == ["Radio"]


def test_a_channel_absent_from_a_market_reads_as_all_zero_not_missing_key():
    """Cross-market comparison asks about channels a market may not run."""
    m = media([
        row("2024-01-01", "DE", "TV", 10.0),
        row("2024-01-01", "AT", "Radio", 20.0),
    ])
    p = build_panel(m)
    s = p.series("AT", "TV")
    assert (s == 0.0).all()
    assert not p.present_mask("AT", "TV").any()


def test_sales_are_summed_to_one_daily_series_per_country():
    m = media([row("2024-01-01", "DE", "TV", 10.0)])
    s = pd.DataFrame({
        "country": ["Germany"] * 4,
        "turnover": [10.0, 20.0, 30.0, 40.0],
        "customer_type": ["New", "New", "Returning", "Returning"],
        "country_code": ["DE"] * 4,
        "granted_discounts": [0] * 4,
        "sales_channel": ["Ecom", "Stores", "Ecom", "Stores"],
        "date": [pd.Timestamp("2024-01-01")] * 4,
    })
    p = build_panel(m, s)
    assert p.sales.loc[pd.Timestamp("2024-01-01"), "DE"] == 100.0


def test_sales_are_optional():
    p = build_panel(media([row("2024-01-01", "DE", "TV", 10.0)]))
    assert p.sales.empty or p.sales.shape[1] == 0


def test_impressions_and_clicks_are_carried_for_corroboration():
    """Spend zero AND impressions zero is a real pause; spend zero with
    impressions still flowing is a billing or tracking artifact."""
    m = media([row("2024-01-01", "DE", "TV", 0.0, clicks=5.0, impr=500.0)])
    p = build_panel(m)
    assert p.impressions.loc[pd.Timestamp("2024-01-01"), ("DE", "TV")] == 500.0
    assert p.clicks.loc[pd.Timestamp("2024-01-01"), ("DE", "TV")] == 5.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_panel.py -v`
Expected: FAIL — `No module named 'detection.io.panel'`.

- [ ] **Step 3: Write `detection/io/panel.py`**

```python
"""media.csv and sales.csv into a complete daily panel.

Two things happen here that matter more on real data than on the benchmark:

1. Campaigns are aggregated to channel level. The synthetic export happens to
   carry one campaign per channel-day, so this is a no-op there -- but the real
   Sellforte export is campaign-grained, and a campaign ending is not a channel
   holdout. Spec section 11 names this as the top real-data risk.
2. Every series is reindexed onto a complete date grid, and a `present` mask
   records whether a row actually existed. Once reindexed, a missing row and a
   zero-spend row are indistinguishable -- and that is the difference between an
   eight-week dark period and eight weeks of broken ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Panel:
    sid: str
    dates: pd.DatetimeIndex
    spend: pd.DataFrame        # index=dates, columns=MultiIndex(country, channel)
    present: pd.DataFrame      # same shape, bool
    impressions: pd.DataFrame
    clicks: pd.DataFrame
    sales: pd.DataFrame        # index=dates, columns=country_code

    @property
    def countries(self) -> list[str]:
        return sorted({c for c, _ in self.spend.columns})

    @property
    def channels(self) -> list[str]:
        return sorted({ch for _, ch in self.spend.columns})

    def channels_in(self, country: str) -> list[str]:
        """Channels this market actually ran -- i.e. spent on at least once."""
        out = []
        for c, ch in self.spend.columns:
            if c == country and (self.spend[(c, ch)] > 0).any():
                out.append(ch)
        return sorted(out)

    def series(self, country: str, channel: str) -> pd.Series:
        key = (country, channel)
        if key not in self.spend.columns:
            return pd.Series(0.0, index=self.dates, name=f"{country}/{channel}")
        return self.spend[key]

    def present_mask(self, country: str, channel: str) -> pd.Series:
        key = (country, channel)
        if key not in self.present.columns:
            return pd.Series(False, index=self.dates)
        return self.present[key]


def _pivot(df: pd.DataFrame, values: str, dates: pd.DatetimeIndex,
           fill: float) -> pd.DataFrame:
    wide = df.pivot_table(index="date", columns=["country_code",
                                                 "advertising_channel"],
                          values=values, aggfunc="sum")
    return wide.reindex(dates).fillna(fill)


def build_panel(media_df: pd.DataFrame, sales_df: pd.DataFrame | None = None,
                sid: str = "") -> Panel:
    media = media_df.copy()
    media["date"] = pd.to_datetime(media["date"])

    dates = pd.date_range(media["date"].min(), media["date"].max(), freq="D")

    spend = _pivot(media, "media_investment", dates, 0.0)
    impressions = _pivot(media, "impressions", dates, 0.0)
    clicks = _pivot(media, "clicks", dates, 0.0)

    # Presence is counted BEFORE any fill, so a row that existed with spend 0.0
    # is distinguishable from a row that never existed at all.
    present = (media.pivot_table(index="date",
                                 columns=["country_code", "advertising_channel"],
                                 values="media_investment", aggfunc="size")
               .reindex(dates).fillna(0) > 0)
    present = present.reindex(columns=spend.columns, fill_value=False)

    if sales_df is not None and len(sales_df):
        sales = sales_df.copy()
        sales["date"] = pd.to_datetime(sales["date"])
        sales = (sales.pivot_table(index="date", columns="country_code",
                                   values="turnover", aggfunc="sum")
                 .reindex(dates).fillna(0.0))
    else:
        sales = pd.DataFrame(index=dates)

    return Panel(sid=sid, dates=dates, spend=spend, present=present,
                 impressions=impressions, clicks=clicks, sales=sales)
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_panel.py -v`
Expected: 10 passed.

- [ ] **Step 5: Smoke it against a real scenario**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python - <<'PY'
import pandas as pd
from detection.io.panel import build_panel
m = pd.read_csv("benchmark/datasets/dev/dev_005/media.csv", parse_dates=["date"])
s = pd.read_csv("benchmark/datasets/dev/dev_005/sales.csv", parse_dates=["date"])
p = build_panel(m, s, "dev_005")
print("days:", len(p.dates), p.dates[0].date(), "->", p.dates[-1].date())
print("countries:", p.countries)
print("channels:", p.channels)
print("all present?", bool(p.present.all().all()))
ser = p.series("FR", p.channels_in("FR")[0])
print("FR zeros:", int((ser == 0).sum()), "of", len(ser))
PY
```
Expected: 730 days, 2024-01-01 → 2025-12-30, 5 countries, 2 channels, all present, and FR showing 14 zero days — the injected dark period. Record the output in your report.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/io/panel.py tests/detection/test_panel.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): build a complete daily panel with a presence mask"
```

---

### Task 3: Normalization

**Files:**
- Create: `detection/io/normalize.py`
- Test: `tests/detection/test_normalize.py`

**Interfaces:**
- Consumes: `Panel` (Task 2).
- Produces:
  - `active_level(s: pd.Series) -> float` — median of positive spend, `nan` if never positive
  - `scale_free(s: pd.Series) -> pd.Series` — `log1p(spend / active_level)`
  - `channel_share(panel, country) -> pd.DataFrame` — each channel's share of that market's daily total
  - `market_scale(panel, country) -> float` — long-run median of the market's daily total spend
  - `cross_market_series(panel, country, channel) -> pd.Series` — spend ÷ market scale
  - Tasks 4, 5, 7, 8 consume these.

- [ ] **Step 1: Write the failing test**

Create `tests/detection/test_normalize.py`:

```python
import numpy as np
import pandas as pd

from detection.io.normalize import (active_level, channel_share,
                                    cross_market_series, market_scale,
                                    scale_free)
from detection.io.panel import build_panel


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def media(rows):
    return pd.DataFrame(rows, columns=[
        "date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code",
    ]).astype({"date": "datetime64[ns]"})


def row(date, country, channel, spend):
    return [pd.Timestamp(date), "P", channel, "c", 1, spend, 1.0, 10.0, 0,
            0.0, country]


def test_active_level_ignores_off_days():
    """The level must describe what the channel spends when it is RUNNING,
    otherwise a long holdout drags the baseline down and hides itself."""
    assert active_level(s([100.0, 100.0, 0.0, 0.0, 100.0])) == 100.0


def test_active_level_is_nan_when_the_channel_never_ran():
    assert np.isnan(active_level(s([0.0, 0.0, 0.0])))


def test_active_level_is_robust_to_a_spike():
    assert active_level(s([100.0, 100.0, 100.0, 100.0, 9999.0])) == 100.0


def test_scale_free_makes_two_markets_of_different_size_identical():
    """A Finnish holdout and a US holdout must look the same to the detector.
    This is the whole reason detection runs on a log ratio."""
    small = s([100.0, 100.0, 0.0, 0.0, 100.0])
    big = s([13000.0, 13000.0, 0.0, 0.0, 13000.0])
    assert np.allclose(scale_free(small).values, scale_free(big).values)


def test_scale_free_of_zero_is_zero():
    assert scale_free(s([100.0, 0.0]))[1] == 0.0


def test_channel_share_sums_to_one_on_days_with_spend():
    m = media([
        row("2024-01-01", "DE", "TV", 75.0),
        row("2024-01-01", "DE", "Radio", 25.0),
    ])
    share = channel_share(build_panel(m), "DE")
    assert abs(share.loc[pd.Timestamp("2024-01-01")].sum() - 1.0) < 1e-9
    assert abs(share.loc[pd.Timestamp("2024-01-01"), "TV"] - 0.75) < 1e-9


def test_channel_share_is_zero_not_nan_on_a_dark_day():
    """A dark day has no total to divide by. It must not poison the series
    with NaN, because composition logic reads it downstream."""
    m = media([
        row("2024-01-01", "DE", "TV", 0.0),
        row("2024-01-01", "DE", "Radio", 0.0),
    ])
    share = channel_share(build_panel(m), "DE")
    assert (share.loc[pd.Timestamp("2024-01-01")] == 0.0).all()


def test_market_scale_is_the_long_run_median_of_total_spend():
    rows = []
    for d in pd.date_range("2024-01-01", periods=5):
        rows.append(row(d, "DE", "TV", 60.0))
        rows.append(row(d, "DE", "Radio", 40.0))
    assert market_scale(build_panel(media(rows)), "DE") == 100.0


def test_cross_market_series_divides_out_market_size():
    """Two markets 10x apart in size, running the same relative pattern, must
    produce the same cross-market series -- otherwise the large market dominates
    every comparison, which spec section 5 exists to prevent."""
    rows = []
    for d in pd.date_range("2024-01-01", periods=5):
        rows.append(row(d, "US", "TV", 1000.0))
        rows.append(row(d, "FI", "TV", 100.0))
    p = build_panel(media(rows))
    assert np.allclose(cross_market_series(p, "US", "TV").values,
                       cross_market_series(p, "FI", "TV").values)


def test_market_scale_of_a_silent_market_does_not_divide_by_zero():
    m = media([row("2024-01-01", "DE", "TV", 0.0)])
    p = build_panel(m)
    assert market_scale(p, "DE") == 0.0
    assert not cross_market_series(p, "DE", "TV").isna().any()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_normalize.py -v`
Expected: FAIL — `No module named 'detection.io.normalize'`.

- [ ] **Step 3: Write `detection/io/normalize.py`**

```python
"""The three scales of spec section 5.

Raw euros are never compared across markets: the benchmark spans a 15x range
between its largest and smallest, so a threshold tuned on Germany would be
meaningless in Finland.

- within-series : spend / the series' own active level, then log1p. Scale-free,
  and multiplicative spend noise becomes additive.
- within-market : each channel's share of the market's daily total, so
  composition changes are visible independently of budget swings.
- cross-market  : spend / a market-scale proxy, the long-run median of that
  market's total daily spend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detection.io.panel import Panel


def active_level(s: pd.Series) -> float:
    """Median spend over days the channel was actually running.

    Conditioning on positive days is what keeps a long holdout from dragging
    the baseline down and concealing itself.
    """
    positive = s[s > 0]
    if positive.empty:
        return float("nan")
    return float(positive.median())


def scale_free(s: pd.Series) -> pd.Series:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        return pd.Series(0.0, index=s.index)
    return np.log1p(s / level)


def channel_share(panel: Panel, country: str) -> pd.DataFrame:
    cols = [(c, ch) for c, ch in panel.spend.columns if c == country]
    block = panel.spend[cols]
    block.columns = [ch for _, ch in cols]
    total = block.sum(axis=1)
    # A dark day has no total; share is 0 rather than NaN so downstream
    # composition logic never has to special-case it.
    return block.div(total.where(total > 0), axis=0).fillna(0.0)


def market_scale(panel: Panel, country: str) -> float:
    cols = [(c, ch) for c, ch in panel.spend.columns if c == country]
    if not cols:
        return 0.0
    total = panel.spend[cols].sum(axis=1)
    positive = total[total > 0]
    if positive.empty:
        return 0.0
    return float(positive.median())


def cross_market_series(panel: Panel, country: str, channel: str) -> pd.Series:
    scale = market_scale(panel, country)
    s = panel.series(country, channel)
    if scale <= 0:
        return pd.Series(0.0, index=s.index)
    return s / scale
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_normalize.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/io/normalize.py tests/detection/test_normalize.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): add within-series, within-market and cross-market scaling"
```

---

### Task 4: P1 — off-runs

**Files:**
- Create: `detection/primitives/zero_runs.py`
- Test: `tests/detection/test_zero_runs.py`

**Interfaces:**
- Consumes: `params`, `active_level` (Task 3).
- Produces:
  - `@dataclass(frozen=True) OffRun` with `start: pd.Timestamp`, `end: pd.Timestamp`, `n_days: int`, `depth: float`, `kind: str`, `notable: bool`, `edge_sharpness: float`, `touches_start: bool`, `touches_end: bool`
  - `off_mask(s, present=None) -> pd.Series`
  - `find_off_runs(s, present=None) -> list[OffRun]` — all runs, each flagged `notable`
  - `notable_runs(s, present=None) -> list[OffRun]`
  - Tasks 6, 7, 9 consume these.

- [ ] **Step 1: Write the failing test**

Create `tests/detection/test_zero_runs.py`:

```python
import pandas as pd

from detection import params
from detection.primitives.zero_runs import find_off_runs, notable_runs, off_mask


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def run_of(level, n_on, n_off, n_on2):
    return s([level] * n_on + [0.0] * n_off + [level] * n_on2)


def test_exact_zero_is_off():
    assert list(off_mask(s([100.0, 0.0, 100.0]))) == [False, True, False]


def test_near_zero_is_off_at_the_rho_threshold():
    """The benchmark injects holdouts at 0.02-0.08x, so near-zero must count."""
    assert off_mask(s([100.0, 3.0, 100.0]))[1]       # 0.03 * level
    assert not off_mask(s([100.0, 50.0, 100.0]))[1]  # ordinary low-spend day


def test_a_run_shorter_than_min_days_is_not_notable():
    runs = find_off_runs(run_of(100.0, 20, params.MIN_DAYS - 1, 20))
    assert len(runs) == 1 and not runs[0].notable


def test_a_run_at_min_days_is_notable_when_the_series_is_otherwise_never_off():
    runs = find_off_runs(run_of(100.0, 20, params.MIN_DAYS, 20))
    assert len(runs) == 1 and runs[0].notable


def test_an_intermittent_channel_needs_a_much_longer_run():
    """THE guard. A flighting channel whose normal gaps are 3 days must not
    report every gap; a fixed 7-day rule would fire on all of them."""
    values = ([100.0] * 11 + [0.0] * 3) * 8          # 8 routine 3-day gaps
    values += [100.0] * 11 + [0.0] * 8 + [100.0] * 20  # one 8-day gap
    runs = find_off_runs(s(values))
    routine = [r for r in runs if r.n_days == 3]
    assert len(routine) == 8
    assert not any(r.notable for r in routine)
    long_run = [r for r in runs if r.n_days == 8]
    assert len(long_run) == 1 and long_run[0].notable


def test_a_never_otherwise_off_channel_needs_only_min_days():
    """Complement of the test above: the ratio rule must not punish a series
    that has no gap distribution to compare against."""
    runs = notable_runs(run_of(100.0, 100, params.MIN_DAYS, 100))
    assert len(runs) == 1


def test_depth_is_one_for_an_exact_zero_and_lower_for_near_zero():
    exact = find_off_runs(run_of(100.0, 20, 10, 20))[0]
    assert exact.depth == 1.0
    near = find_off_runs(s([100.0] * 20 + [5.0] * 10 + [100.0] * 20))[0]
    assert 0.9 < near.depth < 1.0


def test_kind_distinguishes_exact_zero_from_near_zero():
    assert find_off_runs(run_of(100.0, 20, 10, 20))[0].kind == "exact_zero"
    assert find_off_runs(s([100.0] * 20 + [5.0] * 10 + [100.0] * 20))[0].kind \
        == "near_zero"


def test_a_missing_row_is_reported_as_missing_not_as_a_pause():
    """A missing row and a zero-spend row are the same number after reindexing.
    Only the presence mask can tell them apart, and the difference is a real
    dark period versus broken ingestion."""
    values = [100.0] * 20 + [0.0] * 10 + [100.0] * 20
    present = pd.Series(True, index=pd.date_range("2024-01-01",
                                                  periods=len(values)))
    present.iloc[20:30] = False
    r = find_off_runs(s(values), present=present)[0]
    assert r.kind == "missing"


def test_runs_carry_inclusive_dates():
    r = find_off_runs(run_of(100.0, 3, 4, 3))[0]
    assert r.start == pd.Timestamp("2024-01-04")
    assert r.end == pd.Timestamp("2024-01-07")
    assert r.n_days == 4


def test_a_run_at_the_series_start_is_flagged_censored():
    r = find_off_runs(s([0.0] * 10 + [100.0] * 20))[0]
    assert r.touches_start and not r.touches_end


def test_a_run_at_the_series_end_is_flagged_censored():
    r = find_off_runs(s([100.0] * 20 + [0.0] * 10))[0]
    assert r.touches_end and not r.touches_start


def test_a_channel_that_never_ran_yields_no_runs():
    """All-zero means the market does not use this channel. Reporting the whole
    series as one enormous holdout would be a false positive on every market
    that simply does not run a channel."""
    assert find_off_runs(s([0.0] * 100)) == []


def test_edge_sharpness_is_high_for_a_clean_stop():
    r = find_off_runs(run_of(100.0, 20, 10, 20))[0]
    assert r.edge_sharpness > 0.9
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_zero_runs.py -v`
Expected: FAIL — `No module named 'detection.primitives.zero_runs'`.

- [ ] **Step 3: Write `detection/primitives/zero_runs.py`**

```python
"""P1 -- maximal runs of days where a channel was effectively off.

Notability is judged RELATIVE TO THE SERIES' OWN HISTORY, which is what lets
one rule serve two opposite cases: a flighting channel whose normal gaps are
three days needs a much longer run before anything is reported, while a channel
that is otherwise never off needs only MIN_DAYS. A single global threshold
fails one of those two whichever value is chosen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detection import params
from detection.io.normalize import active_level


@dataclass(frozen=True)
class OffRun:
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    depth: float
    kind: str            # "exact_zero" | "near_zero" | "missing"
    notable: bool
    edge_sharpness: float
    touches_start: bool
    touches_end: bool


def off_mask(s: pd.Series, present: pd.Series | None = None) -> pd.Series:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        return pd.Series(False, index=s.index)
    threshold = max(params.EPS_ABS, params.RHO * level)
    mask = s <= threshold
    if present is not None:
        mask = mask | ~present.astype(bool)
    return mask


def _spans(mask: pd.Series) -> list[tuple[int, int]]:
    """Maximal [start, end] index pairs where mask is True."""
    out, start = [], None
    values = list(mask.values)
    for i, flag in enumerate(values):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(values) - 1))
    return out


def _edge_sharpness(s: pd.Series, lo: int, hi: int, level: float) -> float:
    """How cleanly spend stops and restarts, as a fraction of the active level.

    A clean stop scores near 1; a gradual wind-down scores lower.
    """
    before = s.iloc[max(0, lo - params.ROLLING):lo]
    after = s.iloc[hi + 1:hi + 1 + params.ROLLING]
    flank = pd.concat([before, after])
    if flank.empty or level <= 0:
        return 0.0
    return float(min(1.0, flank.median() / level))


def find_off_runs(s: pd.Series, present: pd.Series | None = None) -> list[OffRun]:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0:
        # The market never ran this channel. Reporting the whole series as one
        # enormous holdout would be a false positive on every market that
        # simply does not use a channel.
        return []

    mask = off_mask(s, present)
    spans = _spans(mask)
    if not spans:
        return []

    lengths = np.array([hi - lo + 1 for lo, hi in spans], dtype=float)

    runs: list[OffRun] = []
    for idx, (lo, hi) in enumerate(spans):
        n_days = int(lengths[idx])
        window = s.iloc[lo:hi + 1]

        others = np.delete(lengths, idx)
        if others.size >= params.MIN_RUNS_FOR_RATIO:
            floor = max(params.MIN_DAYS,
                        params.RUN_RATIO * float(np.percentile(others, 90)))
        else:
            floor = params.MIN_DAYS
        notable = n_days >= floor

        if present is not None and not present.iloc[lo:hi + 1].any():
            kind = "missing"
        elif float(window.max()) <= params.EPS_ABS:
            kind = "exact_zero"
        else:
            kind = "near_zero"

        runs.append(OffRun(
            start=s.index[lo], end=s.index[hi], n_days=n_days,
            depth=float(1.0 - window.mean() / level),
            kind=kind, notable=bool(notable),
            edge_sharpness=_edge_sharpness(s, lo, hi, level),
            touches_start=lo == 0, touches_end=hi == len(s) - 1,
        ))
    return runs


def notable_runs(s: pd.Series, present: pd.Series | None = None) -> list[OffRun]:
    return [r for r in find_off_runs(s, present) if r.notable]
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_zero_runs.py -v`
Expected: 14 passed.

- [ ] **Step 5: Check it against the real injected dark period**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python - <<'PY'
import pandas as pd
from detection.io.panel import build_panel
from detection.primitives.zero_runs import notable_runs
m = pd.read_csv("benchmark/datasets/dev/dev_005/media.csv", parse_dates=["date"])
p = build_panel(m, sid="dev_005")
for ch in p.channels_in("FR"):
    for r in notable_runs(p.series("FR", ch), p.present_mask("FR", ch)):
        print(f"FR/{ch}: {r.start.date()} -> {r.end.date()} "
              f"({r.n_days}d, depth {r.depth:.2f}, {r.kind}, "
              f"edge {r.edge_sharpness:.2f})")
PY
```
Expected: one 14-day `exact_zero` run per FR channel, 2024-12-18 → 2024-12-31, depth 1.00. That is the injected dark period, found from spend alone. Record the output.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/primitives/zero_runs.py tests/detection/test_zero_runs.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): add P1 off-run detection with a distribution-relative guard"
```

---

### Task 5: P2 — level shifts

**Files:**
- Create: `detection/primitives/level_shift.py`
- Test: `tests/detection/test_level_shift.py`

**Interfaces:**
- Consumes: `params`, `active_level`/`scale_free` (Task 3).
- Produces:
  - `@dataclass(frozen=True) LevelShift` with `at: pd.Timestamp`, `index: int`, `z: float`, `delta: float`, `ratio: float`, `sharpness: float`
  - `@dataclass(frozen=True) StepEpisode` with `start: pd.Timestamp`, `end: pd.Timestamp`, `ratio: float`, `z: float`, `open_ended: bool`
  - `find_level_shifts(s) -> list[LevelShift]`
  - `find_step_episodes(s) -> list[StepEpisode]`
  - Tasks 7 and 9 consume `find_step_episodes`.

- [ ] **Step 1: Write the failing test**

Create `tests/detection/test_level_shift.py`:

```python
import numpy as np
import pandas as pd

from detection import params
from detection.primitives.level_shift import find_level_shifts, find_step_episodes


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def noisy(level, n, rng, scale=0.05):
    return list(level * (1.0 + rng.normal(0, scale, n)))


def test_a_clean_step_up_is_found_with_the_right_ratio():
    rng = np.random.default_rng(0)
    series = s(noisy(100.0, 120, rng) + noisy(300.0, 120, rng))
    episodes = find_step_episodes(series)
    assert episodes, "no step found"
    best = max(episodes, key=lambda e: abs(np.log(e.ratio)))
    assert 2.6 < best.ratio < 3.4


def test_a_clean_step_down_is_found():
    rng = np.random.default_rng(1)
    series = s(noisy(300.0, 120, rng) + noisy(100.0, 120, rng))
    episodes = find_step_episodes(series)
    assert episodes
    best = max(episodes, key=lambda e: abs(np.log(e.ratio)))
    assert 0.25 < best.ratio < 0.45


def test_a_flat_noisy_series_produces_no_step():
    rng = np.random.default_rng(2)
    assert find_step_episodes(s(noisy(100.0, 240, rng))) == []


def test_a_gradual_ramp_is_rejected_by_sharpness():
    """THE ramp defence. The benchmark injects a 50-day ramp rising 1.2x to
    2.0x in blocks and records NO step change; a detector that fires here is
    wrong. Persistence alone does not reject it, because a ramp's new level
    genuinely does hold -- only sharpness does."""
    rng = np.random.default_rng(3)
    values = noisy(100.0, 120, rng)
    for mult in [1.2, 1.4, 1.6, 1.8, 2.0]:
        values += noisy(100.0 * mult, 10, rng)
    values += noisy(200.0, 120, rng)
    assert find_step_episodes(s(values)) == []


def test_a_one_day_spike_is_rejected_by_persistence():
    rng = np.random.default_rng(4)
    values = noisy(100.0, 120, rng) + [900.0] + noisy(100.0, 120, rng)
    assert find_step_episodes(s(values)) == []


def test_a_short_excursion_below_persist_is_rejected():
    rng = np.random.default_rng(5)
    values = (noisy(100.0, 120, rng) + noisy(300.0, params.PERSIST - 4, rng)
              + noisy(100.0, 120, rng))
    assert find_step_episodes(s(values)) == []


def test_a_bounded_step_episode_has_both_ends():
    rng = np.random.default_rng(6)
    values = noisy(100.0, 120, rng) + noisy(300.0, 60, rng) + noisy(100.0, 120, rng)
    episodes = [e for e in find_step_episodes(s(values)) if not e.open_ended]
    assert episodes, "expected a bounded episode"
    e = episodes[0]
    assert 40 <= (e.end - e.start).days <= 80


def test_an_unpaired_shift_runs_to_the_series_end():
    rng = np.random.default_rng(7)
    values = noisy(100.0, 150, rng) + noisy(300.0, 150, rng)
    episodes = find_step_episodes(s(values))
    assert any(e.open_ended for e in episodes)


def test_shifts_report_a_ratio_in_original_units():
    """3.02x must be readable straight from the output -- it is what an analyst
    compares against a briefed budget change."""
    rng = np.random.default_rng(8)
    shifts = find_level_shifts(s(noisy(100.0, 120, rng) + noisy(300.0, 120, rng)))
    assert shifts
    assert any(2.6 < abs(sh.ratio) < 3.4 for sh in shifts)


def test_a_series_that_never_ran_produces_nothing():
    assert find_step_episodes(s([0.0] * 200)) == []


def test_a_short_series_does_not_crash():
    assert find_step_episodes(s([100.0] * 5)) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_level_shift.py -v`
Expected: FAIL — `No module named 'detection.primitives.level_shift'`.

- [ ] **Step 3: Write `detection/primitives/level_shift.py`**

```python
"""P2 -- robust level shifts on a scale-free series.

Everything is medians and a MAD-scaled z on log1p(spend / level). A Gaussian z
on raw euros breaks on both the 15x market-size spread and the heavy right tail
of daily spend; a robust z on a log ratio is scale-free, tolerant of
multiplicative noise, and prints legibly.

Two filters do different jobs, and BOTH are needed:

- persistence rejects spikes -- a one-day excursion does not hold.
- sharpness rejects gradual ramps -- a genuine step concentrates its change
  into a couple of days, while a 50-day ramp spreads it out. Persistence alone
  does NOT reject a ramp, because a ramp's new level genuinely does hold. The
  benchmark injects exactly such a ramp and records no step change for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detection import params
from detection.io.normalize import active_level, scale_free


@dataclass(frozen=True)
class LevelShift:
    at: pd.Timestamp
    index: int
    z: float
    delta: float
    ratio: float
    sharpness: float


@dataclass(frozen=True)
class StepEpisode:
    start: pd.Timestamp
    end: pd.Timestamp
    ratio: float
    z: float
    open_ended: bool


def _robust_sigma(values: np.ndarray) -> float:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return max(float(params.MAD_TO_SIGMA * mad), params.SIGMA_FLOOR)


def find_level_shifts(s: pd.Series) -> list[LevelShift]:
    level = active_level(s)
    if not np.isfinite(level) or level <= 0 or len(s) < 2 * params.W + 1:
        return []

    y = scale_free(s).rolling(params.ROLLING, center=True,
                              min_periods=1).median().values
    raw = s.values
    n = len(y)

    candidates: list[LevelShift] = []
    for t in range(params.W, n - params.W):
        before, after = y[t - params.W:t], y[t:t + params.W]
        delta = float(np.median(after) - np.median(before))
        sigma = _robust_sigma(np.concatenate([before, after]))
        z = delta / sigma
        if abs(z) < params.Z_THRESH:
            continue

        # Persistence: the new level must still hold PERSIST days later.
        tail_end = min(n, t + params.PERSIST + params.W)
        if tail_end - (t + params.PERSIST) < 2:
            continue
        held = float(np.median(y[t + params.PERSIST:tail_end])
                     - np.median(before))
        if abs(held) < abs(delta) / 2 or np.sign(held) != np.sign(delta):
            continue

        # Sharpness: how much of the change lands within a few days.
        k = params.SHARPNESS_WINDOW
        near = float(np.median(y[t:t + k]) - np.median(y[max(0, t - k):t]))
        sharpness = abs(near) / abs(delta) if delta else 0.0
        if sharpness < params.SHARPNESS:
            continue

        before_raw = np.median(raw[t - params.W:t])
        after_raw = np.median(raw[t:t + params.W])
        ratio = float(after_raw / before_raw) if before_raw > 0 else float("inf")

        candidates.append(LevelShift(at=s.index[t], index=t, z=float(z),
                                     delta=delta, ratio=ratio,
                                     sharpness=float(sharpness)))

    # Keep the strongest candidate in each W-wide neighbourhood, so one step
    # does not report as a cluster of adjacent change points.
    kept: list[LevelShift] = []
    for c in sorted(candidates, key=lambda c: -abs(c.z)):
        if all(abs(c.index - k.index) >= params.W for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c.index)


def find_step_episodes(s: pd.Series) -> list[StepEpisode]:
    shifts = find_level_shifts(s)
    if not shifts:
        return []

    episodes: list[StepEpisode] = []
    used: set[int] = set()
    for i, up in enumerate(shifts):
        if i in used:
            continue
        partner = None
        for j in range(i + 1, len(shifts)):
            if j in used:
                continue
            other = shifts[j]
            # Opposite sign and comparable magnitude closes an episode.
            if (np.sign(other.delta) != np.sign(up.delta)
                    and 0.5 <= abs(other.delta / up.delta) <= 2.0):
                partner = j
                break
        if partner is not None:
            other = shifts[partner]
            used.update({i, partner})
            episodes.append(StepEpisode(
                start=up.at, end=s.index[other.index - 1], ratio=up.ratio,
                z=up.z, open_ended=False))
        else:
            used.add(i)
            episodes.append(StepEpisode(
                start=up.at, end=s.index[-1], ratio=up.ratio, z=up.z,
                open_ended=True))
    return episodes
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_level_shift.py -v`
Expected: 11 passed.

If `test_a_gradual_ramp_is_rejected_by_sharpness` fails, the sharpness filter is wrong — **do not relax the test**. It encodes a negative control the benchmark deliberately contains, and a detector that fires on a ramp loses precision on the sealed split.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/primitives/level_shift.py tests/detection/test_level_shift.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): add P2 level shifts with persistence and ramp defences"
```

---

### Task 6: P3 and P4 — pulses and onsets

**Files:**
- Create: `detection/primitives/pulse.py`, `detection/primitives/onset.py`
- Test: `tests/detection/test_pulse.py`, `tests/detection/test_onset.py`

**Interfaces:**
- Consumes: `OffRun`, `find_off_runs`, `notable_runs` (Task 4).
- Produces:
  - `@dataclass(frozen=True) PulseTrain` with `start`, `end`, `components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]`, `n_pulses: int`
  - `find_pulse_trains(runs: list[OffRun]) -> list[PulseTrain]`
  - `@dataclass(frozen=True) Onset` with `first_active: pd.Timestamp`, `dormant_days: int`
  - `find_onset(s, present=None) -> Onset | None`
  - `find_discontinuation(s, present=None) -> OffRun | None` — built here, consumed by Plan 4's `censored_end` flag
  - Tasks 7, 8 and 9 consume `PulseTrain`, `find_pulse_trains` and `find_onset`.

- [ ] **Step 1: Write the failing tests**

Create `tests/detection/test_pulse.py`:

```python
import pandas as pd

from detection.primitives.pulse import find_pulse_trains
from detection.primitives.zero_runs import find_off_runs


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def pulsed(on, off, n):
    values = []
    for _ in range(n):
        values += [100.0] * on + [0.0] * off
    return s(values + [100.0] * on)


def test_a_repeating_on_off_pattern_becomes_one_train():
    """Spec section 7's P3: the detector emits ONE grouped event spanning first
    start to last end, with the individual windows as components. The harness
    matches against grouped truth, so emitting four separate events scores
    zero."""
    trains = find_pulse_trains(find_off_runs(pulsed(21, 14, 4)))
    assert len(trains) == 1
    t = trains[0]
    assert t.n_pulses == 4
    assert len(t.components) == 4
    assert t.start == t.components[0][0]
    assert t.end == t.components[-1][1]


def test_a_single_holdout_is_not_a_pulse_train():
    runs = find_off_runs(s([100.0] * 40 + [0.0] * 14 + [100.0] * 40))
    assert find_pulse_trains(runs) == []


def test_runs_of_wildly_different_length_are_not_a_train():
    """A 7-day gap and a 90-day shutdown are not the same phenomenon."""
    values = ([100.0] * 30 + [0.0] * 7 + [100.0] * 30 + [0.0] * 90
              + [100.0] * 30)
    assert find_pulse_trains(find_off_runs(s(values))) == []


def test_only_notable_runs_are_grouped():
    """An intermittent channel's routine 3-day gaps must not become a train."""
    values = ([100.0] * 11 + [0.0] * 3) * 8 + [100.0] * 20
    assert find_pulse_trains(find_off_runs(s(values))) == []


def test_components_are_inclusive_and_ordered():
    t = find_pulse_trains(find_off_runs(pulsed(21, 14, 3)))[0]
    starts = [a for a, _ in t.components]
    assert starts == sorted(starts)
    for a, b in t.components:
        assert (b - a).days == 13    # 14 inclusive days
```

Create `tests/detection/test_onset.py`:

```python
import pandas as pd

from detection import params
from detection.primitives.onset import find_discontinuation, find_onset


def s(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)),
                     dtype=float)


def test_a_dormant_start_is_an_onset():
    o = find_onset(s([0.0] * 90 + [100.0] * 200))
    assert o is not None
    assert o.dormant_days == 90
    assert o.first_active == pd.Timestamp("2024-03-31")


def test_a_channel_running_from_day_one_has_no_onset():
    assert find_onset(s([100.0] * 200)) is None


def test_a_dormant_period_shorter_than_min_days_is_not_an_onset():
    assert find_onset(s([0.0] * (params.MIN_DAYS - 1) + [100.0] * 200)) is None


def test_a_mid_series_holdout_is_not_an_onset():
    assert find_onset(s([100.0] * 50 + [0.0] * 30 + [100.0] * 50)) is None


def test_a_channel_that_never_runs_has_no_onset():
    """All-zero is a market that does not use the channel, not a launch."""
    assert find_onset(s([0.0] * 200)) is None


def test_a_run_at_the_series_end_is_a_discontinuation():
    d = find_discontinuation(s([100.0] * 200 + [0.0] * 40))
    assert d is not None and d.touches_end and d.n_days == 40


def test_a_channel_running_to_the_end_has_no_discontinuation():
    assert find_discontinuation(s([100.0] * 200)) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_pulse.py tests/detection/test_onset.py -v`
Expected: FAIL — missing modules.

- [ ] **Step 3: Write `detection/primitives/pulse.py`**

```python
"""P3 -- repeated off-runs on one series, grouped into a single pulse train.

Spec section 7 emits ONE event spanning first start to last end, with the
individual windows attached as components. That shape is not cosmetic: the
benchmark's truth is grouped the same way, and a detector that emits one event
per off-window scores IoU about 0.118 against the grouped truth -- under the 0.5
threshold -- so a perfectly working pulse detector would match nothing.

Pulse trains are also the only place adstock decay is observable, which is why
they carry the highest informativeness weight in spec section 8.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detection import params
from detection.primitives.zero_runs import OffRun


@dataclass(frozen=True)
class PulseTrain:
    start: pd.Timestamp
    end: pd.Timestamp
    components: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    n_pulses: int


def find_pulse_trains(runs: list[OffRun]) -> list[PulseTrain]:
    notable = [r for r in runs if r.notable and not r.touches_start
               and not r.touches_end]
    if len(notable) < params.PULSE_MIN_RUNS:
        return []

    lengths = np.array([r.n_days for r in notable], dtype=float)
    median = float(np.median(lengths))
    if median <= 0:
        return []
    iqr = float(np.percentile(lengths, 75) - np.percentile(lengths, 25))
    if iqr / median > params.PULSE_LEN_IQR_RATIO:
        # Runs of wildly different length are not one phenomenon: a 7-day gap
        # and a 90-day shutdown do not belong in the same train.
        return []

    ordered = sorted(notable, key=lambda r: r.start)
    return [PulseTrain(
        start=ordered[0].start,
        end=ordered[-1].end,
        components=tuple((r.start, r.end) for r in ordered),
        n_pulses=len(ordered),
    )]
```

- [ ] **Step 4: Write `detection/primitives/onset.py`**

```python
"""P4 -- runs anchored at the start or end of a series.

A dormant start followed by sustained activity is a launch CANDIDATE only. It
becomes a staggered launch when the cross-market layer confirms the channel was
live elsewhere during that window; on its own it is a censored holdout, and
calling it a launch without peers would be a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detection import params
from detection.primitives.zero_runs import OffRun, find_off_runs


@dataclass(frozen=True)
class Onset:
    first_active: pd.Timestamp
    dormant_days: int


def find_onset(s: pd.Series, present: pd.Series | None = None) -> Onset | None:
    runs = find_off_runs(s, present)          # empty if the channel never ran
    for r in runs:
        if r.touches_start and not r.touches_end and r.n_days >= params.MIN_DAYS:
            idx = s.index.get_loc(r.end)
            return Onset(first_active=s.index[idx + 1], dormant_days=r.n_days)
    return None


def find_discontinuation(s: pd.Series,
                         present: pd.Series | None = None) -> OffRun | None:
    for r in find_off_runs(s, present):
        if r.touches_end and not r.touches_start and r.n_days >= params.MIN_DAYS:
            return r
    return None
```

- [ ] **Step 5: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_pulse.py tests/detection/test_onset.py -v`
Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/primitives/pulse.py detection/primitives/onset.py \
        tests/detection/test_pulse.py tests/detection/test_onset.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): add P3 pulse grouping and P4 onset detection"
```

---

### Task 7: Regime composition

**Files:**
- Create: `detection/compose/label.py`
- Test: `tests/detection/test_label.py`

**Interfaces:**
- Consumes: `Panel` (Task 2), `off_mask`/`notable_runs` (Task 4), `find_step_episodes` (Task 5), `find_pulse_trains` (Task 6), `DetectedEvent` (Task 1).
- Produces:
  - `active_matrix(panel, country) -> pd.DataFrame` — bool, index=dates, columns=channels
  - `@dataclass(frozen=True) Regime` with `start`, `end`, `active: frozenset[str]`
  - `segment_regimes(panel, country) -> list[Regime]`
  - `label_market(panel, country, sid) -> list[DetectedEvent]`
  - Task 9 consumes `label_market`.

- [ ] **Step 1: Write the failing test**

Create `tests/detection/test_label.py`:

```python
import pandas as pd

from detection.compose.label import label_market, segment_regimes
from detection.io.panel import build_panel

COLS = ["date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"]


def build(spend_by_channel, country="DE", start="2024-01-01"):
    """spend_by_channel: {channel: [daily spend]}"""
    n = len(next(iter(spend_by_channel.values())))
    dates = pd.date_range(start, periods=n)
    rows = []
    for ch, values in spend_by_channel.items():
        for d, v in zip(dates, values):
            rows.append([d, "P", ch, "c", 1, float(v), 1.0, 10.0, 0, 0.0,
                         country])
    return build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")


def types(events):
    return sorted({e.event_type for e in events})


def test_all_channels_off_is_a_dark_period():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40})
    events = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "dark_period"]
    assert len(events) == 1
    e = events[0]
    assert e.channel is None
    assert e.start == pd.Timestamp("2024-02-10")
    assert e.n_days == 20


def test_one_channel_off_is_a_natural_holdout_naming_the_off_channel():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 100})
    events = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "natural_holdout"]
    assert len(events) == 1
    assert events[0].channel == "TV"


def test_exactly_one_channel_left_running_is_a_single_channel_period():
    """The channel named is the one that stays ON -- the opposite convention to
    natural_holdout, inherited from the generator and asserted by the harness."""
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40,
               "Search": [30] * 100})
    events = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "single_channel"]
    assert len(events) == 1
    assert events[0].channel == "Search"


def test_a_dark_period_does_not_also_emit_its_component_holdouts():
    """Regime segmentation resolves the nesting structurally: a dark period IS
    the regime where nothing is active, and its constituent per-channel
    holdouts ride along as components rather than competing events."""
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40})
    events = label_market(p, "DE", "dev_test")
    assert types(events) == ["dark_period"]
    assert len(events[0].components) == 2


def test_a_regime_shorter_than_min_days_is_ignored():
    p = build({"TV": [100] * 40 + [0] * 3 + [100] * 40,
               "Radio": [50] * 83})
    assert label_market(p, "DE", "dev_test") == []


def test_a_channel_never_run_in_this_market_is_not_a_holdout():
    """A market that simply does not use a channel must not be reported as
    holding it out for the entire series."""
    p = build({"TV": [100] * 80, "Radio": [0] * 80})
    assert label_market(p, "DE", "dev_test") == []


def test_a_channel_off_only_at_the_start_is_not_a_holdout():
    """That is a launch, and P4 plus the cross-market layer own it."""
    p = build({"TV": [0] * 30 + [100] * 60, "Radio": [50] * 90})
    holdouts = [e for e in label_market(p, "DE", "dev_test")
                if e.event_type == "natural_holdout"]
    assert holdouts == []


def test_a_step_change_is_labelled_with_its_ratio():
    import numpy as np
    rng = np.random.default_rng(11)
    tv = list(100 * (1 + rng.normal(0, 0.05, 120))) + \
         list(300 * (1 + rng.normal(0, 0.05, 120)))
    p = build({"TV": tv, "Radio": [50] * 240})
    steps = [e for e in label_market(p, "DE", "dev_test")
             if e.event_type == "step_change"]
    assert steps
    assert 2.6 < steps[0].magnitude_ratio < 3.4


def test_a_pulse_train_is_one_event_with_components():
    values = []
    for _ in range(4):
        values += [100] * 21 + [0] * 14
    values += [100] * 21
    p = build({"TV": values, "Radio": [50] * len(values)})
    pulses = [e for e in label_market(p, "DE", "dev_test")
              if e.event_type == "channel_pulse"]
    assert len(pulses) == 1
    assert len(pulses[0].components) == 4


def test_a_pulse_train_suppresses_its_individual_holdouts():
    """Otherwise the same windows are reported twice, once grouped and once
    per-window, and precision collapses."""
    values = []
    for _ in range(4):
        values += [100] * 21 + [0] * 14
    values += [100] * 21
    p = build({"TV": values, "Radio": [50] * len(values)})
    events = label_market(p, "DE", "dev_test")
    assert not [e for e in events
                if e.event_type == "natural_holdout" and e.channel == "TV"]


def test_a_holdout_is_not_also_reported_as_a_step_change():
    """A channel dropping to zero and back is a huge level shift at both edges.
    Without a guard the same window is reported twice -- once as a holdout and
    once as a step -- and precision collapses."""
    p = build({"TV": [100] * 60 + [0] * 30 + [100] * 60,
               "Radio": [50] * 150})
    events = label_market(p, "DE", "dev_test")
    assert [e for e in events
            if e.event_type == "step_change" and e.channel == "TV"] == []
    assert [e for e in events if e.event_type == "natural_holdout"]


def test_a_dark_period_is_not_also_reported_as_step_changes():
    p = build({"TV": [100] * 60 + [0] * 30 + [100] * 60,
               "Radio": [50] * 60 + [0] * 30 + [50] * 60})
    assert types(label_market(p, "DE", "dev_test")) == ["dark_period"]


def test_regimes_partition_the_timeline_without_gaps_or_overlaps():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 100})
    regimes = segment_regimes(p, "DE")
    assert regimes[0].start == p.dates[0]
    assert regimes[-1].end == p.dates[-1]
    for a, b in zip(regimes, regimes[1:]):
        assert (b.start - a.end).days == 1


def test_events_carry_the_sid():
    p = build({"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 100})
    assert all(e.sid == "dev_test" for e in label_market(p, "DE", "dev_test"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_label.py -v`
Expected: FAIL — `No module named 'detection.compose.label'`.

- [ ] **Step 3: Write `detection/compose/label.py`**

```python
"""Compose primitives into labelled events by segmenting each market's timeline.

Not interval clustering, which would merge unrelated concurrent events. Instead,
per market, compute the ACTIVE-CHANNEL SET for each day and cut the timeline
wherever it changes. Each maximal run of a constant active set is a regime, and
regimes label themselves:

    A = {}                      -> dark_period
    |A| = 1, others were active -> single_channel (naming the LIVE channel)
    0 < |K \\ A| < |K|          -> natural_holdout per off channel
    A = K, a step episode spans -> step_change

This resolves the nesting of event types structurally rather than by precedence
rules: a dark period simply IS the regime where nothing is active, and its
constituent per-channel holdouts ride along as components instead of competing
with the event that contains them.
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


@dataclass(frozen=True)
class Regime:
    start: pd.Timestamp
    end: pd.Timestamp
    active: frozenset[str]


def active_matrix(panel: Panel, country: str) -> pd.DataFrame:
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


def label_market(panel: Panel, country: str, sid: str) -> list[DetectedEvent]:
    channels = panel.channels_in(country)
    if not channels:
        return []

    matrix = active_matrix(panel, country)
    all_channels = frozenset(channels)
    events: list[DetectedEvent] = []

    # --- pulse trains first: they claim their own windows, so the regime pass
    # below must not also report each window as a separate holdout.
    pulsed_channels: set[str] = set()
    for ch in channels:
        runs = find_off_runs(panel.series(country, ch),
                             panel.present_mask(country, ch))
        for train in find_pulse_trains(runs):
            pulsed_channels.add(ch)
            events.append(DetectedEvent(
                sid=sid, country_code=country, channel=ch,
                event_type="channel_pulse", start=train.start, end=train.end,
                components=train.components,
                evidence={"n_pulses": train.n_pulses},
            ))

    # --- regimes
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
        elif len(regime.active) == 1 and len(all_channels) >= 2 and all(
                _was_active_around(matrix, ch, regime) for ch in off):
            events.append(DetectedEvent(
                sid=sid, country_code=country,
                channel=next(iter(regime.active)),
                event_type="single_channel", start=regime.start,
                end=regime.end,
                evidence={"n_channels_off": len(off)},
            ))
        elif off:
            for ch in sorted(off):
                if ch in pulsed_channels:
                    continue
                if not _was_active_around(matrix, ch, regime):
                    continue
                events.append(DetectedEvent(
                    sid=sid, country_code=country, channel=ch,
                    event_type="natural_holdout", start=regime.start,
                    end=regime.end, evidence={},
                ))

    # --- step changes, on channels that stayed on throughout
    #
    # A channel that drops to zero and back produces an enormous level shift at
    # BOTH edges, so an unfiltered step pass would re-report every holdout and
    # dark period as a step change too -- the same window twice, wrecking
    # precision. A step is only a step if the channel never went off during it.
    for ch in channels:
        off_windows = [(r.start, r.end) for r
                       in find_off_runs(panel.series(country, ch),
                                        panel.present_mask(country, ch))
                       if r.notable]
        for ep in find_step_episodes(panel.series(country, ch)):
            n_days = int((ep.end - ep.start).days) + 1
            if n_days < params.MIN_DAYS:
                continue
            if any(a <= ep.end and ep.start <= b for a, b in off_windows):
                continue
            events.append(DetectedEvent(
                sid=sid, country_code=country, channel=ch,
                event_type="step_change", start=ep.start, end=ep.end,
                magnitude_ratio=ep.ratio,
                evidence={"z": ep.z, "open_ended": ep.open_ended},
            ))

    events.sort(key=lambda e: (e.start, str(e.channel), e.event_type))
    return events
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_label.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/compose/label.py tests/detection/test_label.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): compose events by regime segmentation on the active-channel set"
```

---

### Task 8: Cross-market layer

**Files:**
- Create: `detection/compose/cross_market.py`
- Test: `tests/detection/test_cross_market.py`

**Interfaces:**
- Consumes: `Panel` (Task 2), `find_onset` (Task 6), `DetectedEvent` (Task 1), `off_mask` (Task 4).
- Produces:
  - `annotate(events, panel) -> list[DetectedEvent]` — adds `control_available` to evidence and `cross_market_holdout` / `global_pause` tags
  - `find_staggered_launches(panel, sid) -> list[DetectedEvent]` — already fanned out, one event per late market
  - Task 9 consumes both.

- [ ] **Step 1: Write the failing test**

Create `tests/detection/test_cross_market.py`:

```python
import pandas as pd

from detection.compose.cross_market import annotate, find_staggered_launches
from detection.compose.label import label_market
from detection.io.panel import build_panel
from detection.model import DetectedEvent

COLS = ["date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"]


def build(by_country, start="2024-01-01"):
    """by_country: {country: {channel: [daily spend]}}"""
    rows = []
    for country, channels in by_country.items():
        for ch, values in channels.items():
            dates = pd.date_range(start, periods=len(values))
            for d, v in zip(dates, values):
                rows.append([d, "P", ch, "c", 1, float(v), 1.0, 10.0, 0, 0.0,
                             country])
    return build_panel(pd.DataFrame(rows, columns=COLS), sid="dev_test")


def test_a_holdout_with_healthy_peers_is_tagged_cross_market():
    """The most MMM-valuable finding the system can produce: the peers are a
    ready-made control group."""
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
        "AT": {"TV": [100] * 100, "Radio": [50] * 100},
    })
    events = annotate(label_market(p, "DE", "dev_test"), p)
    holdout = [e for e in events if e.event_type == "natural_holdout"][0]
    assert "cross_market_holdout" in holdout.tags
    assert holdout.evidence["control_available"] == "peers"


def test_a_holdout_with_every_peer_also_off_is_tagged_global_pause():
    """Real, but with no control group -- and this is also what a pipeline
    outage looks like, so validity suspicion belongs here."""
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
        "AT": {"TV": [100] * 40 + [0] * 20 + [100] * 40, "Radio": [50] * 100},
    })
    events = annotate(label_market(p, "DE", "dev_test"), p)
    holdout = [e for e in events if e.event_type == "natural_holdout"][0]
    assert "global_pause" in holdout.tags
    assert holdout.evidence["control_available"] == "none"


def test_a_dark_period_with_healthy_peers_reports_peer_control():
    p = build({
        "DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40,
               "Radio": [50] * 40 + [0] * 20 + [50] * 40},
        "AT": {"TV": [100] * 100, "Radio": [50] * 100},
    })
    dark = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
            if e.event_type == "dark_period"][0]
    assert dark.evidence["control_available"] == "peers"
    assert "global_pause" not in dark.tags


def test_a_single_market_scenario_reports_no_peer_control():
    p = build({"DE": {"TV": [100] * 40 + [0] * 20 + [100] * 40,
                      "Radio": [50] * 100}})
    holdout = [e for e in annotate(label_market(p, "DE", "dev_test"), p)
               if e.event_type == "natural_holdout"][0]
    assert holdout.evidence["control_available"] == "sibling_channels"


def test_staggered_launch_is_emitted_once_per_LATE_market():
    """The harness's truth is PER-MARKET, so a country-less panel event scores
    zero. benchmark/eval/README.md states the fan-out is the detector's job."""
    p = build({
        "DE": {"TV": [100] * 200},
        "AT": {"TV": [0] * 60 + [100] * 140},
        "CH": {"TV": [0] * 120 + [100] * 80},
    })
    launches = find_staggered_launches(p, "dev_test")
    assert {e.country_code for e in launches} == {"AT", "CH"}
    assert all(e.event_type == "staggered_launch" for e in launches)
    assert all(e.country_code is not None for e in launches)


def test_a_launch_window_runs_from_series_start_to_the_day_before_activity():
    p = build({"DE": {"TV": [100] * 200}, "AT": {"TV": [0] * 60 + [100] * 140}})
    at = [e for e in find_staggered_launches(p, "dev_test")
          if e.country_code == "AT"][0]
    assert at.start == p.dates[0]
    assert at.end == p.dates[59]


def test_markets_starting_together_are_not_a_staggered_launch():
    """Coincidental start-up jitter is not a rollout."""
    p = build({
        "DE": {"TV": [0] * 30 + [100] * 170},
        "AT": {"TV": [0] * 32 + [100] * 168},
    })
    assert find_staggered_launches(p, "dev_test") == []


def test_a_channel_live_everywhere_from_day_one_is_not_a_launch():
    p = build({"DE": {"TV": [100] * 200}, "AT": {"TV": [100] * 200}})
    assert find_staggered_launches(p, "dev_test") == []


def test_annotate_preserves_events_it_cannot_classify():
    e = DetectedEvent(sid="dev_test", country_code="DE", channel="TV",
                      event_type="step_change",
                      start=pd.Timestamp("2024-02-01"),
                      end=pd.Timestamp("2024-03-01"))
    p = build({"DE": {"TV": [100] * 100}, "AT": {"TV": [100] * 100}})
    out = annotate([e], p)
    assert len(out) == 1 and out[0].event_type == "step_change"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_cross_market.py -v`
Expected: FAIL — `No module named 'detection.compose.cross_market'`.

- [ ] **Step 3: Write `detection/compose/cross_market.py`**

```python
"""Peer-market comparison: what control group, if any, does an event have?

This layer decides an event's worth more than anything about its own shape. A
holdout whose peers kept running has a ready-made control group and is the most
MMM-valuable finding the system can produce. The same holdout with every peer
also off has no control at all -- and looks exactly like a pipeline outage.

It also owns staggered launches, and FANS THEM OUT to one event per market.
The benchmark's truth is per-market, so a country-less panel event matches
nothing; benchmark/eval/README.md states this is the detector's obligation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from detection import params
from detection.model import DetectedEvent
from detection.io.panel import Panel
from detection.primitives.onset import find_onset
from detection.primitives.zero_runs import off_mask


def _peer_status(panel: Panel, country: str, channel: str | None,
                 start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    """(peers running normally, peers also off) over the window."""
    running = off = 0
    for peer in panel.countries:
        if peer == country:
            continue
        channels = ([channel] if channel is not None
                    else panel.channels_in(peer))
        channels = [ch for ch in channels if ch in panel.channels_in(peer)]
        if not channels:
            continue
        peer_off = True
        for ch in channels:
            if not off_mask(panel.series(peer, ch)).loc[start:end].all():
                peer_off = False
                break
        if peer_off:
            off += 1
        else:
            running += 1
    return running, off


def annotate(events: list[DetectedEvent], panel: Panel) -> list[DetectedEvent]:
    out: list[DetectedEvent] = []
    for e in events:
        if e.event_type not in {"dark_period", "single_channel",
                                "natural_holdout", "channel_pulse"}:
            out.append(e)
            continue

        running, peers_off = _peer_status(panel, e.country_code, e.channel,
                                          e.start, e.end)
        tags = list(e.tags)
        if running > 0:
            control = "peers"
            if e.event_type == "natural_holdout":
                tags.append("cross_market_holdout")
        elif peers_off > 0:
            control = "none"
            tags.append("global_pause")
        else:
            # No peers carry this channel at all; the market's other channels
            # are the only available control.
            control = "sibling_channels"

        evidence = dict(e.evidence)
        evidence.update(control_available=control, peers_running=running,
                        peers_off=peers_off)
        out.append(DetectedEvent(
            sid=e.sid, country_code=e.country_code, channel=e.channel,
            event_type=e.event_type, start=e.start, end=e.end,
            magnitude_ratio=e.magnitude_ratio, components=e.components,
            tags=tuple(tags), evidence=evidence,
        ))
    return out


def find_staggered_launches(panel: Panel, sid: str) -> list[DetectedEvent]:
    out: list[DetectedEvent] = []
    for channel in panel.channels:
        onsets: dict[str, pd.Timestamp | None] = {}
        for country in panel.countries:
            if channel not in panel.channels_in(country):
                continue
            s = panel.series(country, channel)
            onset = find_onset(s, panel.present_mask(country, channel))
            onsets[country] = onset.first_active if onset else panel.dates[0]

        if len(onsets) < 2:
            continue
        days = np.array([(d - panel.dates[0]).days for d in onsets.values()],
                        dtype=float)
        if days.max() - days.min() < params.ONSET_SPREAD:
            continue

        # Fan out: one event per LATE market. The earliest market is the
        # comparison group and is not itself a launch event.
        earliest = min(onsets.values())
        for country, first_active in sorted(onsets.items()):
            if first_active == earliest:
                continue
            idx = panel.dates.get_loc(first_active)
            out.append(DetectedEvent(
                sid=sid, country_code=country, channel=channel,
                event_type="staggered_launch", start=panel.dates[0],
                end=panel.dates[idx - 1],
                tags=("cross_market_control",),
                evidence={"onset": str(first_active.date()),
                          "earliest_market_onset": str(earliest.date()),
                          "n_markets": len(onsets)},
            ))
    return out
```

- [ ] **Step 4: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_cross_market.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/compose/cross_market.py tests/detection/test_cross_market.py
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): add the cross-market layer and staggered-launch fan-out"
```

---

### Task 9: Pipeline, adapter, and the first real score

The task that turns nine modules into a number.

**Files:**
- Create: `detection/pipeline.py`, `benchmark/eval/adapter.py`
- Test: `tests/detection/test_pipeline.py`, `tests/eval/test_adapter.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `run_detection(media_df, sales_df=None, sid="") -> list[DetectedEvent]`
  - `benchmark.eval.adapter.to_event(d: DetectedEvent) -> Event`
  - `benchmark.eval.adapter.detect(media, sales, sid) -> list[Event]` — the scoreable detector callable
  - Plan 4 consumes `run_detection` and `detect`.

- [ ] **Step 1: Write the failing tests**

Create `tests/detection/test_pipeline.py`:

```python
import pandas as pd
import pytest

from detection.model import EVENT_TYPES
from detection.pipeline import run_detection

DEV = "benchmark/datasets/dev"


def load(sid):
    media = pd.read_csv(f"{DEV}/{sid}/media.csv", parse_dates=["date"])
    sales = pd.read_csv(f"{DEV}/{sid}/sales.csv", parse_dates=["date"])
    return media, sales


def test_runs_end_to_end_on_a_real_scenario():
    media, sales = load("dev_005")
    events = run_detection(media, sales, "dev_005")
    assert all(e.sid == "dev_005" for e in events)
    assert all(e.event_type in EVENT_TYPES for e in events)


def test_finds_the_injected_dark_period_in_dev_005():
    """dev_005 injects a dark period in FR from 2024-12-18 for 14 days. The
    detector sees only spend, and must find it."""
    media, sales = load("dev_005")
    events = run_detection(media, sales, "dev_005")
    dark = [e for e in events
            if e.event_type == "dark_period" and e.country_code == "FR"]
    assert dark, "the injected dark period was not found"
    e = dark[0]
    assert abs((e.start - pd.Timestamp("2024-12-18")).days) <= 2
    assert abs((e.end - pd.Timestamp("2024-12-31")).days) <= 2


def test_reports_nothing_on_a_null_scenario():
    """dev_001 carries no events at all. Anything reported is a false positive,
    and null scenarios are the only way to measure a false-positive rate."""
    media, sales = load("dev_001")
    assert run_detection(media, sales, "dev_001") == []


def test_every_interval_is_inclusive_and_ordered():
    media, sales = load("dev_019")
    for e in run_detection(media, sales, "dev_019"):
        assert e.start <= e.end
        assert e.n_days >= 1


def test_sales_are_optional():
    media, _ = load("dev_005")
    assert run_detection(media, None, "dev_005") is not None


def test_no_duplicate_events():
    media, sales = load("dev_005")
    events = run_detection(media, sales, "dev_005")
    keys = [(e.country_code, e.channel, e.event_type, e.start, e.end)
            for e in events]
    assert len(keys) == len(set(keys))


def test_the_detector_never_reads_ground_truth(monkeypatch):
    """The black-box guarantee, enforced at runtime rather than by inspection."""
    import pathlib
    real = pathlib.Path.open

    def guarded(self, *a, **kw):
        if "_truth" in str(self):
            raise AssertionError(f"detector reached for truth: {self}")
        return real(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "open", guarded)
    media, sales = load("dev_005")
    run_detection(media, sales, "dev_005")
```

Create `tests/eval/test_adapter.py`:

```python
import pandas as pd

from benchmark.eval.adapter import detect, to_event
from benchmark.eval.model import Event
from detection.model import DetectedEvent


def test_to_event_maps_every_field():
    d = DetectedEvent(sid="dev_001", country_code="DE", channel="TV",
                      event_type="natural_holdout",
                      start=pd.Timestamp("2024-03-01"),
                      end=pd.Timestamp("2024-03-10"),
                      magnitude_ratio=3.0,
                      components=((pd.Timestamp("2024-03-01"),
                                   pd.Timestamp("2024-03-05")),),
                      tags=("cross_market_holdout",),
                      evidence={"z": 4.2})
    e = to_event(d)
    assert isinstance(e, Event)
    assert (e.sid, e.country_code, e.channel, e.event_type) == \
           ("dev_001", "DE", "TV", "natural_holdout")
    assert e.start == d.start and e.end == d.end
    assert e.components == d.components
    assert e.tags == d.tags


def test_detect_returns_harness_events_for_a_real_scenario():
    media = pd.read_csv("benchmark/datasets/dev/dev_005/media.csv",
                        parse_dates=["date"])
    sales = pd.read_csv("benchmark/datasets/dev/dev_005/sales.csv",
                        parse_dates=["date"])
    events = detect(media, sales, "dev_005")
    assert events and all(isinstance(e, Event) for e in events)


def test_detect_requires_the_media_frame():
    """run_dev and run_final pass None unless --load-data is given. A detector
    that silently returned [] there would burn the one-shot final gate and
    record a 0.0 audit line."""
    import pytest
    with pytest.raises(ValueError, match="--load-data"):
        detect(None, None, "dev_005")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_pipeline.py tests/eval/test_adapter.py -v`
Expected: FAIL — missing modules.

- [ ] **Step 3: Write `detection/pipeline.py`**

```python
"""End to end: two CSV frames in, labelled events out.

Detection runs on SPEND ONLY. Sales is accepted and carried on the panel for a
later scoring layer, but it never participates in finding an event: it is
noisier by an order of magnitude, carries seasonality and promotions, and
coupling the two makes every failure harder to explain.
"""
from __future__ import annotations

import pandas as pd

from detection.compose.cross_market import annotate, find_staggered_launches
from detection.compose.label import label_market
from detection.io.panel import build_panel
from detection.model import DetectedEvent


def run_detection(media_df: pd.DataFrame, sales_df: pd.DataFrame | None = None,
                  sid: str = "") -> list[DetectedEvent]:
    panel = build_panel(media_df, sales_df, sid)

    events: list[DetectedEvent] = []
    for country in panel.countries:
        events.extend(label_market(panel, country, sid))

    events = annotate(events, panel)
    events.extend(find_staggered_launches(panel, sid))

    events.sort(key=lambda e: (str(e.country_code), e.start, str(e.channel),
                               e.event_type))
    return events
```

- [ ] **Step 4: Write `benchmark/eval/adapter.py`**

```python
"""Convert the detector's output into the harness's Event type.

The dependency points THIS way on purpose. `detection/` is the deliverable and
must stay blind to the harness -- tests/eval/test_gating.py asserts no file
there even mentions `benchmark.eval`. So the detector emits its own
`DetectedEvent` and the harness, which is allowed to know everything, adapts.
"""
from __future__ import annotations

import pandas as pd

from benchmark.eval.model import Event
from detection.model import DetectedEvent
from detection.pipeline import run_detection


def to_event(d: DetectedEvent) -> Event:
    return Event(
        sid=d.sid,
        country_code=d.country_code,
        channel=d.channel,
        event_type=d.event_type,
        start=d.start,
        end=d.end,
        multiplier=d.magnitude_ratio,
        components=d.components,
        tags=d.tags,
    )


def detect(media_df, sales_df, sid: str) -> list[Event]:
    """The scoreable detector callable.

    `run_dev` and `run_final` pass None for both frames unless --load-data is
    given. Returning [] there would look like a detector that found nothing
    rather than a misconfigured run -- and on the one-shot final gate that
    would burn the run and record a 0.0 audit line. So it raises instead.
    """
    if media_df is None:
        raise ValueError(
            "detect() needs the media frame; pass --load-data to run_dev "
            "or run_final")
    return [to_event(d) for d in run_detection(media_df, sales_df, sid)]
```

- [ ] **Step 5: Run the tests**

Run: `synthetic_data_generator/.venv/bin/python -m pytest tests/detection/test_pipeline.py tests/eval/test_adapter.py -v`
Expected: 10 passed.

`test_reports_nothing_on_a_null_scenario` is the one most likely to fail first. If it does, the detector is producing false positives on a scenario with no events — investigate which primitive fired and why, and report the cause. Do NOT relax the test; the null scenarios are the only way the benchmark can measure a false-positive rate.

- [ ] **Step 6: Score the detector on the dev split for the first time**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m benchmark.eval.run_dev \
  --detector benchmark.eval.adapter:detect \
  --load-data --label "plan3-first-run" \
  --out /tmp/dev_report.md
```

This is the moment the whole project has been building toward. Record in your report: the headline precision, recall and F1; the null-scenario false-positive rate; the per-type table; and the type-confusion matrix.

For context, the baselines from Plan 2: `perfect_oracle` scores F1 1.000 with a 0.000 null FP rate; `never_detect` scores F1 0.000 with a 0.000 null FP rate; `detect_everything` scores F1 0.000 with a 0.654 null FP rate. **Do not tune anything to improve this number** — Plan 4 owns iteration, and a first honest reading is more useful than a good one.

- [ ] **Step 7: Run the whole suite and confirm the seal**

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m pytest tests -m "not slow" -q
synthetic_data_generator/.venv/bin/python -c "
from benchmark.harness import seal
ok, p = seal.verify_seal('test')
print('SEAL OK' if ok else f'SEAL BROKEN: {p[:3]}')"
ls benchmark/eval/final_runs.jsonl 2>&1 | tail -1
```
Expected: all tests pass, `SEAL OK`, and `final_runs.jsonl` still absent.

- [ ] **Step 8: Commit**

```bash
cd /Users/user123/Desktop/projects/datascience-project
git add detection/pipeline.py benchmark/eval/adapter.py \
        tests/detection/test_pipeline.py tests/eval/test_adapter.py
git add benchmark/eval/dev_history.jsonl
git -c user.name="user123" -c user.email="airin.sunshine@gmail.com" \
  commit -m "feat(detection): wire the pipeline and score it against the dev split"
```

---

## Verification

```bash
cd /Users/user123/Desktop/projects/datascience-project
synthetic_data_generator/.venv/bin/python -m pytest tests -m "not slow" -q
synthetic_data_generator/.venv/bin/python -c "
from benchmark.harness import seal
ok, p = seal.verify_seal('test'); print('SEAL OK' if ok else f'BROKEN: {p}')"
git status --short
```

Expected: every test passes, `SEAL OK`, clean tree, and `benchmark/eval/final_runs.jsonl` absent.

## What Plan 4 picks up

Plan 4 adds spec §8's three scores — detection confidence with PAV calibration on the dev split, informativeness for ranking, and the validity gate that separates a real dark period from a data outage — plus the templated explanations, the dev iteration loop logging to `dev_history.jsonl`, the single gated final run against the sealed test split, and `REPORT.md` with the six handover items.
