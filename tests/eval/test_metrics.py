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
