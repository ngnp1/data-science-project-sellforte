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


def test_reliability_curve_uses_identity_not_equality_to_credit_matches():
    """Two Events equal in content but distinct objects must not both be
    credited when only one of them can match under one-to-one assignment.
    A structural-equality-based matched set would wrongly credit both and
    report empirical_precision == 1.0 instead of 0.5."""
    t = [e("2024-03-01", "2024-03-10")]
    base = e("2024-03-01", "2024-03-10")
    p1 = Event(**{**base.__dict__, "detection_confidence": 0.9})
    p2 = Event(**{**base.__dict__, "detection_confidence": 0.9})
    assert p1 == p2 and p1 is not p2

    curve = M.reliability_curve(t, [p1, p2], n_bins=10)
    assert len(curve) == 1
    assert curve[0]["n"] == 2
    assert curve[0]["empirical_precision"] == 0.5


def test_reliability_curve_counts_every_bin_boundary_confidence():
    """Confidences sitting exactly on bin edges -- 0.0, an internal boundary,
    and 1.0 -- must all be counted somewhere. The last bin is closed on the
    right specifically so a confidence of 1.0 is not silently dropped."""
    t = [e("2024-03-01", "2024-03-10")]
    pred = [
        Event(**{**e("2024-04-01", "2024-04-10").__dict__,
                 "detection_confidence": 0.0}),
        Event(**{**e("2024-05-01", "2024-05-10").__dict__,
                 "detection_confidence": 0.5}),
        Event(**{**e("2024-06-01", "2024-06-10").__dict__,
                 "detection_confidence": 1.0}),
    ]
    curve = M.reliability_curve(t, pred, n_bins=10)
    assert sum(b["n"] for b in curve) == len(pred)


def test_operating_curve_keeps_unscored_predictions_at_every_cut():
    """Opposite of reliability_curve's null rule: an unscored detection must
    survive every confidence cut rather than vanishing, so a detector that
    never scores anything still produces a meaningful curve instead of one
    that empties out at the first threshold."""
    t = [e("2024-03-01", "2024-03-10")]
    p = [e("2024-03-01", "2024-03-10")]  # detection_confidence is None
    curve = M.operating_curve(t, p, cuts=[0.0, 0.5, 0.9, 1.0])
    assert all(c["n_pred"] == 1 for c in curve)


def test_day_level_and_event_level_diverge_on_split_predictions():
    """Two adjacent predictions covering one truth interval score perfectly
    at day level but not at strict event level, pinning the contrast that
    justifies having a day-level metric at all."""
    t = [e("2024-03-01", "2024-03-20")]
    p = [e("2024-03-01", "2024-03-10"), e("2024-03-11", "2024-03-20")]

    day = M.day_level(t, p)
    assert day["precision"] == 1.0 and day["recall"] == 1.0

    event = M.event_level(t, p)
    assert event["precision"] < 1.0 or event["recall"] < 1.0


def test_relaxed_match_is_what_makes_channel_error_visible():
    """Under strict matching a channel mix-up is invisible as a channel
    error -- it collapses into a false negative plus a false positive.
    Pinning both sides together so a refactor can't silently lose the
    contrast that makes channel accuracy meaningful."""
    t = [e("2024-03-01", "2024-03-10", channel="TV")]
    p = [e("2024-03-01", "2024-03-10", channel="Radio")]

    strict = match_events(t, p, strict=True)
    assert strict.n_tp == 0 and strict.n_fp == 1 and strict.n_fn == 1

    got = M.channel_and_market_accuracy(t, p)
    assert got["channel_accuracy"] == 0.0
    assert got["market_accuracy"] == 1.0


def test_operating_curve_default_sweep_reaches_one_and_has_21_points():
    """The default cut sweep must include 1.0 -- np.arange's half-open end
    would otherwise silently drop the strictest cut from a curve that goes
    straight into the project's report."""
    t = [e("2024-03-01", "2024-03-10")]
    p = [Event(**{**e("2024-03-01", "2024-03-10").__dict__,
                  "detection_confidence": 0.9})]
    curve = M.operating_curve(t, p)
    assert len(curve) == 21
    assert curve[-1]["cut"] == 1.0


def test_f1_is_the_harmonic_mean_not_the_arithmetic_one():
    """I2. F1 is the benchmark's headline number, and every other assertion in
    this file sits at precision == recall, where the harmonic and arithmetic
    means agree exactly. Replacing the harmonic mean with the arithmetic one
    therefore left the whole suite green. This case is deliberately
    asymmetric: P = 0.25, R = 2/3. The harmonic mean is 0.364; the arithmetic
    mean would be 0.458 -- a 26% overstatement of the project's headline."""
    got = M.prf(n_tp=2, n_fp=6, n_fn=1)
    assert got["precision"] == 0.25
    assert abs(got["recall"] - 2 / 3) < 1e-9
    assert abs(got["f1"] - 0.36363636363636365) < 1e-9
    assert abs(got["f1"] - (0.25 + 2 / 3) / 2) > 0.09, (
        "F1 equals the arithmetic mean of P and R -- the harmonic mean is gone")


def test_boundary_p90_is_the_ninetieth_percentile_not_the_median():
    """I3a. Spec section 9 item 3 requires median AND p90, and p90 is what
    shows the tail -- the occasional badly-localised detection a median hides.
    Every prior assertion used a fixture whose median and p90 coincide, so
    changing `np.percentile(..., 90)` to `..., 50)` left the suite green.
    Ten matched pairs with start offsets 0..9: median 4.5, p90 8.1."""
    # One 40-day truth window per scenario, so no two pairs can contend.
    t = [e("2024-03-01", "2024-04-09", sid=f"dev_{i:03d}") for i in range(9)]
    # Start offsets 1..9 days; end offsets 0,0,0,1,1,1,2,2,2 -- so the start
    # and end columns cannot coincide by accident either.
    p = [Event(**{**t[i].__dict__,
                  "start": t[i].start + pd.Timedelta(days=i + 1),
                  "end": t[i].end - pd.Timedelta(days=i // 3)})
         for i in range(9)]
    got = M.boundary_error(match_events(t, p))

    assert got["n"] == 9
    assert got["start_median"] == 5.0                 # offsets 1..9
    assert abs(got["start_p90"] - 8.2) < 1e-9         # 90th percentile of 1..9
    assert got["start_p90"] > got["start_median"], (
        "p90 collapsed onto the median -- the tail is no longer being reported")
    assert got["end_median"] == 1.0                   # offsets 0,0,0,1,1,1,2,2,2
    assert abs(got["end_p90"] - 2.0) < 1e-9
    assert got["end_p90"] > got["end_median"]


def test_boundary_error_reports_n_zero_when_nothing_matched():
    """I3b (the metric half). With no matched pairs there is no boundary error
    to report, and the zeros this returns are a convention, not a measurement.
    `n` is what lets the renderer tell the two apart."""
    t = [e("2024-03-01", "2024-03-10")]
    got = M.boundary_error(match_events(t, []))
    assert got["n"] == 0
    assert got["start_median"] == 0.0 and got["start_p90"] == 0.0


def test_operating_curve_keeps_full_confidence_detections_at_the_strictest_cut():
    """I6. The comparison at the cut is `>=`, not `>`. At cut == 1.0 a `>`
    would drop every confidence-1.0 detection, so a PERFECT detector's
    strictest published operating point would read 0.000/0.000 -- and the
    strictest point is exactly the one a reader quotes as the conservative
    operating regime. No prior assertion pinned the boundary."""
    t = [e("2024-03-01", "2024-03-10"), e("2024-06-01", "2024-06-10")]
    p = [Event(**{**x.__dict__, "detection_confidence": 1.0}) for x in t]

    curve = M.operating_curve(t, p, cuts=[1.0])
    assert len(curve) == 1
    row = curve[0]
    assert row["cut"] == 1.0
    assert row["n_pred"] == 2, (
        "confidence-1.0 detections were dropped at cut 1.0 -- the threshold "
        "comparison is exclusive")
    assert row["precision"] == 1.0 and row["recall"] == 1.0 and row["f1"] == 1.0

    # And the default sweep's last row agrees, since that is what gets rendered.
    assert M.operating_curve(t, p)[-1]["recall"] == 1.0


def test_market_accuracy_reads_the_same_whichever_market_is_dropped():
    """I5, on the real dev scenario that exposed it. `dev_045` is a
    `global_pause`: FR and NL share one identical dark window. A detector that
    is perfect except for one dropped market used to read
    market_accuracy 0.000 if FR was dropped and 1.000 if NL was dropped --
    identical detector quality, the reading decided by alphabetical order.
    """
    from benchmark.eval import truth as T

    truth = T.load_truth("dev", "dev_045")
    assert {x.country_code for x in truth} == {"FR", "NL"}, "fixture moved"
    assert len({(x.start, x.end) for x in truth}) == 1, "windows no longer identical"

    readings = {}
    for dropped in ("FR", "NL"):
        pred = [x for x in truth if x.country_code != dropped]
        readings[dropped] = M.channel_and_market_accuracy(truth, pred)

    assert readings["FR"]["market_accuracy"] == readings["NL"]["market_accuracy"], (
        f"same detector quality, different reading: {readings}")
    assert readings["FR"]["market_accuracy"] == 1.0, (
        "the one market that WAS found should read as correctly attributed")
