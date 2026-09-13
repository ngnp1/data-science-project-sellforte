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
    """Determinism test with two parts: non-contending pairs and tied IoU."""
    # Part 1: Two disjoint truth/pred pairs (no contention).
    t = [e("2024-03-01", "2024-03-10"), e("2024-06-01", "2024-06-10")]
    p = [e("2024-06-01", "2024-06-10"), e("2024-03-01", "2024-03-10")]
    a = match_events(t, p)
    b = match_events(list(reversed(t)), list(reversed(p)))
    assert sorted(m.truth.start for m in a.matches) == \
           sorted(m.truth.start for m in b.matches)

    # Part 2: Tied IoU values must resolve identically regardless of input order.
    # Reviewer's reproduction case: truth [Jan 1..10] vs two predictions that
    # both score IoU 0.5: [Jan 1..5] and [Jan 6..10]. Must match the same one.
    t_contention = [e("2024-03-01", "2024-03-10")]
    p1 = e("2024-03-01", "2024-03-05")  # IoU = 5 / 10 = 0.5
    p2 = e("2024-03-06", "2024-03-10")  # IoU = 5 / 10 = 0.5

    res_p1_first = match_events(t_contention, [p1, p2])
    res_p2_first = match_events(t_contention, [p2, p1])

    # Both should match exactly one pair
    assert len(res_p1_first.matches) == 1
    assert len(res_p2_first.matches) == 1

    # Same prediction must be credited in both orderings (deterministic)
    assert res_p1_first.matches[0].pred.start == res_p2_first.matches[0].pred.start
    assert res_p1_first.matches[0].pred.end == res_p2_first.matches[0].pred.end

    # Also test with both lists reversed
    res_reversed_both = match_events(list(reversed(t_contention)), [p2, p1])
    assert len(res_reversed_both.matches) == 1
    assert res_p1_first.matches[0].pred.start == res_reversed_both.matches[0].pred.start
    assert res_p1_first.matches[0].pred.end == res_reversed_both.matches[0].pred.end


def test_events_from_different_scenarios_never_match():
    t = [e("2024-03-01", "2024-03-10", sid="dev_001")]
    p = [e("2024-03-01", "2024-03-10", sid="dev_002")]
    assert not match_events(t, p).matches
