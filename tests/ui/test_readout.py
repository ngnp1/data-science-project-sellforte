import numpy as np
import pandas as pd
import pytest

from detection.io.panel import build_panel
from detection.model import DetectedEvent
from ui.readout import comparison_markets, masks, readout

N_DAYS = 400
DATES = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")


def make_panel(sales: dict[str, list[float]] | None,
               spend: dict[tuple[str, str], list[float]]):
    media = pd.concat([
        pd.DataFrame({"date": DATES[:len(values)], "country_code": country,
                      "advertising_channel": channel, "media_investment": values})
        for (country, channel), values in spend.items()])
    sales_df = None
    if sales is not None:
        sales_df = pd.concat([
            pd.DataFrame({"date": DATES[:len(values)], "country_code": country,
                          "turnover": values})
            for country, values in sales.items()])
    return build_panel(media, sales_df, sid="t")


def flat(value: float, n: int = N_DAYS) -> list[float]:
    return [float(value)] * n


def with_window(base: float, during: float, start: int, end: int,
                n: int = N_DAYS) -> list[float]:
    values = flat(base, n)
    values[start:end + 1] = [float(during)] * (end - start + 1)
    return values


def event(country, channel, kind, start, end, components=()):
    return DetectedEvent(sid="t", country_code=country, channel=channel,
                         event_type=kind, start=DATES[start], end=DATES[end],
                         components=tuple((DATES[a], DATES[b]) for a, b in components))


HOLDOUT_AT = event("AT", "Facebook", "natural_holdout", 200, 241)
BASE_SPEND = {
    ("AT", "Facebook"): with_window(10, 0, 200, 241),
    ("AT", "TV"): flat(20),
    ("CH", "Facebook"): flat(5),
    ("DE", "Facebook"): flat(7),
}


@pytest.fixture
def panel():
    return make_panel({"AT": with_window(100, 90, 200, 241),
                       "CH": flat(50), "DE": flat(70)}, BASE_SPEND)


def test_ten_percent_drop_against_flat_controls(panel):
    r = readout(HOLDOUT_AT, panel, [HOLDOUT_AT])
    assert r.reference_side == "before" and int(r.reference.sum()) == 42
    assert r.subject_channels == ("Facebook",)
    assert (r.spend_reference, r.spend_during) == (10.0, 0.0)
    assert r.comparison_markets == ("CH", "DE")
    assert (r.sales_reference, r.sales_during) == (100.0, 90.0)
    assert r.effect == pytest.approx(-0.10, abs=1e-9)
    assert r.sales_difference == pytest.approx(-10.0)
    assert r.readable
    assert r.rough_return == pytest.approx(1.0)
    assert r.baseline is None


def test_shared_seasonal_swing_cancels():
    seasonal = make_panel({"AT": with_window(100, 100 * 1.2 * 0.9, 200, 241),
                           "CH": with_window(50, 60, 200, 241),
                           "DE": with_window(70, 84, 200, 241)}, BASE_SPEND)
    r = readout(HOLDOUT_AT, seasonal, [HOLDOUT_AT])
    assert r.effect == pytest.approx(-0.10, abs=1e-9)


def test_reference_falls_back_to_after_window_at_data_start(panel):
    launch = event("AT", "Facebook", "staggered_launch", 0, 59)
    r = readout(launch, panel, [launch])
    assert r.reference_side == "after"
    assert r.reference[DATES[60]:DATES[119]].all() and int(r.reference.sum()) == 60


def test_reference_is_capped_at_ninety_days(panel):
    long_event = event("AT", "Facebook", "natural_holdout", 200, 330)
    assert int(readout(long_event, panel, [long_event]).reference.sum()) == 90


def test_no_reference_window_gives_none():
    tiny = make_panel({"AT": flat(100, 20)}, {("AT", "TV"): flat(5, 20)})
    whole = event("AT", "TV", "natural_holdout", 0, 19)
    assert masks(whole, tiny) is None and readout(whole, tiny, [whole]) is None


def test_pulse_uses_on_days_inside_span(panel):
    pulse = event("AT", "TV", "channel_pulse", 100, 141,
                  components=((100, 113), (128, 141)))
    during, reference, side = masks(pulse, panel)
    assert side == "on-days"
    assert int(during.sum()) == 28 and not (during & reference).any()
    assert reference[DATES[114]:DATES[127]].all() and int(reference.sum()) == 14


def test_no_sales_gives_none_with_reason():
    no_sales = make_panel(None, BASE_SPEND)
    r = readout(HOLDOUT_AT, no_sales, [HOLDOUT_AT])
    assert r.effect is None and r.sales_reference is None and r.rough_return is None
    assert "No sales data" in r.note
    assert r.spend_reference == 10.0


def test_single_market_flags_missing_controls():
    alone = make_panel({"AT": with_window(100, 90, 200, 241)},
                       {k: v for k, v in BASE_SPEND.items() if k[0] == "AT"})
    r = readout(HOLDOUT_AT, alone, [HOLDOUT_AT])
    assert r.comparison_markets == () and "No comparison markets" in r.note
    assert r.effect == pytest.approx(-0.10, abs=1e-9)


def test_tiny_spend_change_gives_no_rough_return(panel):
    step = event("AT", "TV", "step_change", 200, 241)
    stepped = make_panel({"AT": with_window(100, 90, 200, 241),
                          "CH": flat(50), "DE": flat(70)},
                         {**BASE_SPEND, ("AT", "TV"): with_window(20, 21, 200, 241)})
    r = readout(step, stepped, [step])
    assert r.effect is not None and r.rough_return is None


def test_swing_needs_five_placebos():
    n = 60
    short = make_panel({"AT": with_window(100, 90, 32, 45, n), "CH": flat(50, n)},
                       {("AT", "Facebook"): with_window(10, 0, 32, 45, n),
                        ("CH", "Facebook"): flat(5, n)})
    e = event("AT", "Facebook", "natural_holdout", 32, 45)
    r = readout(e, short, [e])
    assert r.swing is None and not r.readable and r.rough_return is None


def test_change_within_normal_swings_is_not_readable():
    rng = np.random.default_rng(7)
    noisy = make_panel({"AT": list(100 + rng.normal(0, 30, N_DAYS)),
                        "CH": list(50 + rng.normal(0, 1, N_DAYS))}, BASE_SPEND)
    r = readout(HOLDOUT_AT, noisy, [HOLDOUT_AT])
    assert r.swing is not None and r.swing > 0
    assert abs(r.effect) < 2 * r.swing
    assert not r.readable and r.rough_return is None


def test_markets_with_overlapping_findings_are_not_controls(panel):
    overlap = event("CH", "Facebook", "step_change", 180, 230)
    r = readout(HOLDOUT_AT, panel, [HOLDOUT_AT, overlap])
    assert r.comparison_markets == ("DE",)
    span = r.during | r.reference
    assert comparison_markets(HOLDOUT_AT, panel, [HOLDOUT_AT, overlap], span) == ["DE"]


def test_missing_sales_days_are_skipped(panel):
    gappy = with_window(100, 90, 200, 241)
    gappy[205] = float("nan")
    sparse = make_panel({"AT": gappy, "CH": flat(50), "DE": flat(70)}, BASE_SPEND)
    assert readout(HOLDOUT_AT, sparse, [HOLDOUT_AT]).sales_during == pytest.approx(90.0)


def test_dark_period_baseline(sample_events, sample_panel):
    dark = next(e for e in sample_events if e.event_type == "dark_period")
    r = readout(dark, sample_panel, sample_events)
    assert r.baseline == pytest.approx(sample_panel.sales["DE"][r.during].mean())
    assert r.spend_during == 0.0 and r.spend_reference > 0
