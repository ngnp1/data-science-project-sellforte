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
