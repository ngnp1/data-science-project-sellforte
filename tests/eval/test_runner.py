from benchmark.eval import detectors_for_testing as D
from benchmark.eval import truth as T
from benchmark.eval.report import render_markdown
from benchmark.eval.runner import evaluate_split

SOME = T.list_scenarios("dev")[:8]


def test_evaluate_split_covers_every_requested_scenario():
    got = evaluate_split(D.never_detect, "dev", sids=SOME)
    assert got["n_scenarios"] == len(SOME)
    assert set(got["per_scenario"]) == set(SOME)


def test_perfect_oracle_scores_one_over_a_whole_split():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    assert got["overall"]["f1"] == 1.0


def test_never_detect_has_a_zero_false_positive_rate_on_nulls():
    got = evaluate_split(D.never_detect, "dev")
    assert got["null_fp_rate"] == 0.0


def test_detect_everything_has_a_nonzero_false_positive_rate_on_nulls():
    got = evaluate_split(D.detect_everything, "dev")
    assert got["null_fp_rate"] > 0.0


def test_breakdowns_are_present_for_the_interpretable_axis():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    assert "noise_level" in got["breakdowns"]


def test_report_renders_without_crashing_and_names_the_headline_numbers():
    got = evaluate_split(D.perfect_oracle, "dev", sids=SOME)
    md = render_markdown(got)
    assert "Precision" in md and "Recall" in md and "F1" in md
    assert "false positive" in md.lower()
    assert "Reliability" in md and "Operating curve" in md


def test_report_says_so_plainly_when_nothing_carries_a_confidence():
    got = evaluate_split(D.never_detect, "dev", sids=SOME)
    md = render_markdown(got)
    assert "calibration cannot be assessed" in md.lower()
