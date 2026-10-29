import re
from dataclasses import replace

import pandas as pd

from detection.model import DetectedEvent
from ui.charts import ACCENT, MUTED, carryover_chart, event_chart, period_map
from ui.labels import TYPES, event_id, short_label
from ui.readout import readout

MARKETS = ["AT", "CH", "DE", "FI", "US"]


def finding_bars(fig):
    """The map's finding traces (the faint per-market tracks carry no ids)."""
    return [t for t in fig.data if t.customdata is not None]


def line_colours(fig):
    return {t.name: t.line.color for t in fig.data if t.type == "scatter"}


def test_period_map_bars_carry_event_ids(sample_events):
    fig = period_map(sample_events, MARKETS, None)
    ids = [cd[0] for trace in finding_bars(fig) for cd in trace.customdata]
    assert set(ids) == {event_id(e) for e in sample_events}
    assert len(ids) == len(sample_events) + 2  # the pulse draws its 3 off-windows


def test_period_map_lists_every_market_and_colours_by_type(sample_events):
    fig = period_map(sample_events[:1], MARKETS, None)
    assert list(fig.layout.yaxis.ticktext) == MARKETS
    info = TYPES[sample_events[0].event_type]
    assert finding_bars(fig)[0].name == info.name
    assert finding_bars(fig)[0].marker.color == info.color
    assert finding_bars(period_map(sample_events[:1], MARKETS, None, dark=True))[0].marker.color \
        == info.color_dark


def test_period_map_separates_overlapping_events_in_one_market():
    events = [
        DetectedEvent(sid="t", country_code="DE", channel=channel, event_type=kind,
                      start=pd.Timestamp(start), end=pd.Timestamp(end))
        for channel, kind, start, end in (
            (None, "dark_period", "2024-02-01", "2024-03-10"),
            ("TV", "natural_holdout", "2024-02-10", "2024-02-28"),
            ("Radio", "step_change", "2024-02-15", "2024-03-05"),
        )
    ]
    fig = period_map(events, ["DE"], None,
                     (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")))
    positions = {cd[0]: y for trace in finding_bars(fig)
                 for cd, y in zip(trace.customdata, trace.y)}
    assert len(set(positions.values())) == 3
    assert len({annotation.y for annotation in fig.layout.annotations}) == 3
    assert list(fig.layout.yaxis.ticktext) == ["DE"]


def test_period_map_reuses_a_lane_for_non_overlapping_events():
    events = [
        DetectedEvent(sid="t", country_code="DE", channel="TV", event_type="natural_holdout",
                      start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-10")),
        DetectedEvent(sid="t", country_code="DE", channel="Radio", event_type="step_change",
                      start=pd.Timestamp("2024-02-01"), end=pd.Timestamp("2024-02-10")),
    ]
    fig = period_map(events, ["DE"], None)
    assert len({y for trace in finding_bars(fig) for y in trace.y}) == 1


def test_period_map_outlines_the_selected_finding(sample_events):
    chosen = event_id(sample_events[0])
    fig = period_map(sample_events, MARKETS, chosen)
    widths = {cd[0]: w for trace in finding_bars(fig)
              for cd, w in zip(trace.customdata, trace.marker.line.width)}
    assert widths[chosen] > 0
    assert all(w == 0 for i, w in widths.items() if i != chosen)


def test_period_map_labels_every_finding(sample_events):
    chosen = event_id(sample_events[0])
    texts = [a.text for a in period_map(sample_events, MARKETS, chosen).layout.annotations]
    plain = {re.sub(r"</?b>", "", t) for t in texts}
    assert plain == {short_label(e) for e in sample_events}
    assert f"<b>{short_label(sample_events[0])}</b>" in texts


def test_event_chart_has_sales_row_when_market_has_sales(sample_events, sample_panel):
    e = sample_events[0]
    fig = event_chart(e, sample_panel, readout(e, sample_panel, sample_events))
    names = {t.name for t in fig.data}
    assert {"This market", "Other markets (avg)"} <= names


def test_event_chart_without_sales_is_spend_only(sample_events, sample_panel):
    no_sales = replace(sample_panel, sales=pd.DataFrame(index=sample_panel.dates))
    fig = event_chart(sample_events[0], no_sales, None)
    names = {t.name for t in fig.data}
    assert "This market" not in names
    assert set(sample_panel.channels_in(sample_events[0].country_code)) <= names


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
    assert "C9" in names and "Other channels" in names
    assert len([n for n in names if n.startswith("C") or n == "Other channels"]) == 8
    colours = line_colours(event_chart(e, panel, None))
    assert colours["C9"] == ACCENT[False] and colours["Other channels"] == MUTED[False]


def test_event_chart_draws_one_strip_per_channel_and_highlights_the_finding(sample_events,
                                                                          sample_panel):
    holdout = next(e for e in sample_events if e.event_type == "natural_holdout")
    fig = event_chart(holdout, sample_panel, readout(holdout, sample_panel, sample_events))
    colours = line_colours(fig)
    channels = sample_panel.channels_in("AT")
    assert set(channels) <= set(colours)
    assert colours["Facebook"] == ACCENT[False]
    assert all(colours[ch] == MUTED[False] for ch in channels if ch != "Facebook")
    dark = line_colours(event_chart(holdout, sample_panel, None, dark=True))
    assert dark["Facebook"] == ACCENT[True]


def test_event_chart_draws_the_window_averages(sample_events, sample_panel):
    dark = next(e for e in sample_events if e.event_type == "dark_period")
    r = readout(dark, sample_panel, sample_events)
    texts = [a.text for a in event_chart(dark, sample_panel, r).layout.annotations]
    assert "Before avg 100" in texts
    during = round(r.sales_during / r.sales_reference * 100)
    assert f"Period avg {during}" in texts
    bare = [a.text for a in event_chart(dark, sample_panel, None).layout.annotations]
    assert not any("avg" in t for t in bare)
