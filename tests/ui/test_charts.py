from dataclasses import replace

import pandas as pd

from ui.charts import carryover_chart, event_chart, period_map
from ui.labels import SCORE_NAMES, TYPES, event_id
from ui.readout import readout

MARKETS = ["AT", "CH", "DE", "FI", "US"]


def test_period_map_bars_carry_event_ids(sample_events):
    fig = period_map(sample_events, MARKETS, None)
    ids = [cd[0] for trace in fig.data for cd in trace.customdata]
    assert set(ids) == {event_id(e) for e in sample_events}
    assert len(ids) == len(sample_events) + 2  # the pulse draws its 3 off-windows


def test_period_map_lists_every_market_and_colours_by_type(sample_events):
    fig = period_map(sample_events[:1], MARKETS, None)
    assert list(fig.layout.yaxis.categoryarray) == MARKETS[::-1]
    trace = fig.data[0]
    assert trace.marker.color == TYPES[sample_events[0].event_type].color
    assert period_map(sample_events[:1], MARKETS, None, dark=True).data[0].marker.color == \
        TYPES[sample_events[0].event_type].color_dark


def test_period_map_outlines_the_selected_finding(sample_events):
    chosen = event_id(sample_events[0])
    fig = period_map(sample_events, MARKETS, chosen)
    widths = {cd[0]: w for trace in fig.data
              for cd, w in zip(trace.customdata, trace.marker.line.width)}
    assert widths[chosen] > 0
    assert all(w == 0 for i, w in widths.items() if i != chosen)


def test_event_chart_has_sales_row_when_market_has_sales(sample_events, sample_panel):
    e = sample_events[0]
    fig = event_chart(e, sample_panel, readout(e, sample_panel, sample_events))
    assert "yaxis2" in fig.layout.to_plotly_json()
    names = {t.name for t in fig.data}
    assert {"This market", "Comparison markets (average)"} <= names


def test_event_chart_without_sales_is_spend_only(sample_events, sample_panel):
    no_sales = replace(sample_panel, sales=pd.DataFrame(index=sample_panel.dates))
    fig = event_chart(sample_events[0], no_sales, None)
    assert "yaxis2" not in fig.layout.to_plotly_json()
    assert len(fig.data) > 0


def test_event_chart_starts_zoomed_on_the_event(sample_events, sample_panel):
    dark = next(e for e in sample_events if e.event_type == "dark_period")
    fig = event_chart(dark, sample_panel, readout(dark, sample_panel, sample_events))
    lo, hi = (pd.Timestamp(x) for x in fig.layout.xaxis.range)
    assert lo <= dark.start - pd.Timedelta(days=55) and hi >= dark.end + pd.Timedelta(days=55)
    assert hi - lo < pd.Timedelta(days=400)


def test_carryover_only_for_pulses(sample_events, sample_panel):
    pulse = next(e for e in sample_events if e.event_type == "channel_pulse")
    fig = carryover_chart(pulse, sample_panel, readout(pulse, sample_panel, sample_events))
    assert fig is not None
    xs = list(fig.data[0].x)
    assert xs[0] == -14 and xs[-1] == 28
    dark = next(e for e in sample_events if e.event_type == "dark_period")
    assert carryover_chart(dark, sample_panel, readout(dark, sample_panel, sample_events)) is None




def test_event_chart_marks_missing_data_rows(sample_frames, sample_events):
    from detection.io.panel import build_panel
    media, sales = sample_frames
    dropped = pd.to_datetime(media["date"]).between("2024-06-01", "2024-06-03") & \
        (media["country_code"] == "DE") & (media["advertising_channel"] == "TV")
    panel = build_panel(media[~dropped], sales)
    dark = next(e for e in sample_events if e.event_type == "dark_period")
    fig = event_chart(dark, panel, readout(dark, panel, sample_events))
    gaps = [t for t in fig.data if t.name == "Missing data rows"]
    assert len(gaps) == 1
    assert list(pd.to_datetime(gaps[0].x)) == list(pd.date_range("2024-06-01", "2024-06-03"))
    complete = event_chart(dark, build_panel(media, sales), None)
    assert not [t for t in complete.data if t.name == "Missing data rows"]


def test_event_chart_never_folds_the_findings_own_channel():
    from detection.io.panel import build_panel
    from detection.model import DetectedEvent
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    rows = []
    for i in range(10):  # ten channels: the smallest would fold into "Other"
        spend = [100.0 - 9 * i] * 200
        if i == 9:
            spend[100:130] = [0.0] * 30
        rows.append(pd.DataFrame({"date": dates, "country_code": "AT",
                                  "advertising_channel": f"C{i}", "media_investment": spend}))
    panel = build_panel(pd.concat(rows))
    e = DetectedEvent(sid="t", country_code="AT", channel="C9", event_type="natural_holdout",
                      start=dates[100], end=dates[129])
    names = [t.name for t in event_chart(e, panel, readout(e, panel, [e])).data]
    assert "C9" in names and "Other" in names
    colours = {t.name: t.line.color for t in event_chart(e, panel, None).data}
    assert colours["C9"] != colours["Other"]
