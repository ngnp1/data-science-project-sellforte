"""End-to-end tests for detection.pipeline.

Paths are resolved from this file rather than from the process working
directory: the rest of the suite (tests/eval/test_gating.py) already does that,
and a test that only passes when pytest happens to be invoked from the project
root is a test that fails for the wrong reason.
"""
import builtins
import pathlib

import pandas as pd
import pytest

from detection.model import EVENT_TYPES, DetectedEvent
from detection.pipeline import run_detection

_MEDIA_COLS = ["date", "ad_platform", "advertising_channel", "campaign_name",
        "campaign_id", "media_investment", "clicks", "impressions",
        "conversions", "conversion_value", "country_code"]

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEV = ROOT / "benchmark" / "datasets" / "dev"

# The substring that marks a ground-truth path in this repository:
# benchmark/datasets/dev_truth/ and benchmark/datasets/test_truth/.
TRUTH_MARKER = "_truth"


def load(sid):
    media = pd.read_csv(DEV / sid / "media.csv", parse_dates=["date"])
    sales = pd.read_csv(DEV / sid / "sales.csv", parse_dates=["date"])
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


def test_sales_do_not_participate_in_detection():
    """The module docstring claims detection runs on SPEND ONLY. That is a
    causal claim about this pipeline, so it is asserted rather than repeated:
    dropping the sales frame entirely must not change a single event.

    The brief's version of this test asserted only `is not None`, which a
    detector that silently returned [] without sales would also satisfy.
    """
    media, sales = load("dev_005")
    with_sales = run_detection(media, sales, "dev_005")
    without_sales = run_detection(media, None, "dev_005")
    assert with_sales, "nothing detected at all; the comparison proves nothing"
    assert with_sales == without_sales


def test_the_sales_frame_reaches_the_panel():
    """Sales does not participate in DETECTION (see above), but the module
    docstring also claims it is carried onto the panel for a later scoring
    layer. Nothing downstream reads it yet, so without this the frame could be
    dropped on the floor and every existing test would stay green.
    """
    seen = {}
    import detection.pipeline as pipeline
    real = pipeline.build_panel

    def spy(media_df, sales_df=None, sid=""):
        panel = real(media_df, sales_df, sid)
        seen["sales_arg_is_none"] = sales_df is None
        seen["panel_sales_columns"] = list(panel.sales.columns)
        return panel

    media, sales = load("dev_005")
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(pipeline, "build_panel", spy)
        pipeline.run_detection(media, sales, "dev_005")
    finally:
        monkey.undo()

    assert seen["sales_arg_is_none"] is False
    assert seen["panel_sales_columns"], "the panel carries no sales series"


def test_no_duplicate_events():
    media, sales = load("dev_005")
    events = run_detection(media, sales, "dev_005")
    keys = [(e.country_code, e.channel, e.event_type, e.start, e.end)
            for e in events]
    assert len(keys) == len(set(keys))


def test_a_pulse_train_stays_one_grouped_event():
    """dev_019 pulses DE/Affiliate four times. Truth for a pulsing channel is
    grouped into ONE interval by the harness's loader, so a pipeline that
    emitted one event per off-window would score an IoU far below the matcher's
    threshold and match nothing. The grouping has to survive to the output.
    """
    media, sales = load("dev_019")
    events = run_detection(media, sales, "dev_019")
    pulses = [e for e in events
              if e.event_type == "channel_pulse"
              and e.country_code == "DE" and e.channel == "Affiliate"]
    assert len(pulses) == 1
    assert len(pulses[0].components) >= 2
    assert all(a <= b for a, b in pulses[0].components)


def test_every_event_names_a_market():
    """A staggered launch is fanned out to one event per launching market. A
    panel-level event with country_code None matches no truth row at all --
    benchmark/eval/truth.py names that as the detector's obligation.
    """
    media, sales = load("dev_022")
    events = run_detection(media, sales, "dev_022")
    assert any(e.event_type == "staggered_launch" for e in events)
    assert all(e.country_code is not None for e in events)


def synthetic_two_market_frame():
    """Two markets whose events land in the order the pipeline does NOT emit.

    BB goes dark in the middle of the year; AA starts TV late and is therefore
    the staggered launch. Per-market labelling emits BB's event first (markets
    are walked in sorted order and BB's regime pass runs before the launch
    pass), while the launches are appended last -- so unsorted output reads
    BB, AA and the final sort is load-bearing.

    Both magnitudes are chosen far away from the thresholds they must clear, so
    this input cannot be accidentally sized from the constants it exercises: a
    months-long dormancy against a fortnight of onset spread, and a month of
    dark against a week of minimum duration.
    """
    dates = pd.date_range("2024-01-01", periods=366, freq="D")
    rows = []
    for i, day in enumerate(dates):
        for country, channel in [("AA", "TV"), ("AA", "Search"),
                                 ("BB", "TV"), ("BB", "Search")]:
            spend = 1000.0
            if country == "AA" and channel == "TV" and i < 150:
                spend = 0.0          # AA launches TV late
            if country == "BB" and 200 <= i < 230:
                spend = 0.0          # BB goes dark for a month
            rows.append({"date": day, "country_code": country,
                         "advertising_channel": channel,
                         "media_investment": spend,
                         "clicks": spend / 10, "impressions": spend * 100})
    return pd.DataFrame(rows)


def test_events_come_back_sorted():
    """On every scenario in the dev split the output happens to be sorted
    already -- per-market lists arrive in market order and no dev scenario
    mixes a staggered launch with an earlier-sorting market's event -- so the
    dev split cannot falsify the sort. Measured, not assumed: removing the
    sort leaves all 45 scenarios byte-identical. This synthetic frame is the
    input that can tell the difference.
    """
    events = run_detection(synthetic_two_market_frame(), None, "synthetic")
    types = {e.event_type for e in events}
    assert types == {"staggered_launch", "dark_period"}, types
    keys = [(str(e.country_code), e.start, str(e.channel), e.event_type)
            for e in events]
    assert keys == sorted(keys)


def _install_truth_guard(monkeypatch):
    """Fail loudly on any attempt to open a path carrying the truth marker.

    BOTH `builtins.open` and `pathlib.Path.open` are patched, and both are
    needed. Measured on this interpreter and this pandas build:

        pd.read_csv(str | Path)  -> builtins.open      YES
        pd.read_csv(str | Path)  -> pathlib.Path.open  NO
        Path.read_text()         -> builtins.open      NO
        Path.read_text()         -> pathlib.Path.open  YES

    Every loader in benchmark/eval reads its CSVs with pandas and its JSON
    with Path.read_text, so a guard patching only one of the two sits green
    while a detector reads truth through the other. See
    test_the_truth_guard_actually_fires, which is the positive control for
    exactly that.
    """
    real_builtin_open = builtins.open
    real_path_open = pathlib.Path.open

    def guarded_builtin_open(file, *a, **kw):
        if TRUTH_MARKER in str(file):
            raise AssertionError(f"detector reached for truth: {file}")
        return real_builtin_open(file, *a, **kw)

    def guarded_path_open(self, *a, **kw):
        if TRUTH_MARKER in str(self):
            raise AssertionError(f"detector reached for truth: {self}")
        return real_path_open(self, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(pathlib.Path, "open", guarded_path_open)


# A path that does not exist and never will, so the positive control can
# prove the guard fires without any risk of actually reading a truth file.
MISSING_TRUTH = "/nonexistent/benchmark/datasets/dev_truth/never_created.csv"


@pytest.mark.parametrize("vector", ["builtin_open", "path_open",
                                    "path_read_text", "read_csv_str",
                                    "read_csv_path"])
def test_the_truth_guard_actually_fires(monkeypatch, vector):
    """POSITIVE CONTROL: a guard that cannot fire is worse than no guard.

    Each vector below is a way a detector could actually reach ground truth.
    The target path is nonexistent, so nothing is read even if the guard were
    removed -- the failure would then be FileNotFoundError, not a pass.
    """
    _install_truth_guard(monkeypatch)
    attempts = {
        "builtin_open": lambda: open(MISSING_TRUTH),
        "path_open": lambda: pathlib.Path(MISSING_TRUTH).open(),
        "path_read_text": lambda: pathlib.Path(MISSING_TRUTH).read_text(),
        "read_csv_str": lambda: pd.read_csv(MISSING_TRUTH),
        "read_csv_path": lambda: pd.read_csv(pathlib.Path(MISSING_TRUTH)),
    }
    with pytest.raises(AssertionError, match="reached for truth"):
        attempts[vector]()


def test_the_truth_guard_still_allows_ordinary_reads(monkeypatch):
    """The negative half of the control: the guard must not simply block
    everything, or the black-box test above would pass on a detector that
    cannot read anything at all."""
    _install_truth_guard(monkeypatch)
    media = pd.read_csv(DEV / "dev_001" / "media.csv", parse_dates=["date"])
    assert len(media)


def test_the_detector_never_reads_ground_truth(monkeypatch):
    """The black-box guarantee, enforced at runtime rather than by inspection."""
    media, sales = load("dev_005")
    _install_truth_guard(monkeypatch)
    events = run_detection(media, sales, "dev_005")
    assert all(isinstance(e, DetectedEvent) for e in events)


def test_an_empty_media_frame_returns_no_events_rather_than_raising():
    """A filtered export or a market with no bookings yet is an empty frame,
    not a bug. It used to reach pd.date_range and surface as "Neither start nor
    end can be NaT"."""
    empty = pd.DataFrame(columns=_MEDIA_COLS)
    assert run_detection(empty, None, "dev_test") == []
