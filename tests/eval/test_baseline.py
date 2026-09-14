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


def test_f1_alone_cannot_tell_never_detect_from_detect_everything():
    """Pins the fact the README explains: never_detect (reports nothing) and
    detect_everything (reports everything) are opposite pathologies but score
    identically on precision/recall/F1. Only the null-scenario false-positive
    rate separates them -- so F1 must never be read as a standalone headline
    for this benchmark. If this stops being true, the README's warning stops
    being accurate."""
    never = evaluate_split(D.never_detect, "dev")
    everything = evaluate_split(D.detect_everything, "dev")
    assert never["overall"]["f1"] == everything["overall"]["f1"]
    assert never["null_fp_rate"] != everything["null_fp_rate"]
