"""Standing guarantees about the harness itself. These run on every suite
invocation, so a later change that breaks the harness's ability to recognise a
correct detector fails immediately rather than silently flattering Plan 3."""
import pytest
from benchmark.eval import detectors_for_testing as D
from benchmark.eval.runner import evaluate_split


@pytest.mark.benchmark_data
def test_perfect_oracle_still_scores_one_over_the_whole_dev_split():
    got = evaluate_split(D.perfect_oracle, "dev")
    assert got["overall"]["precision"] == 1.0
    assert got["overall"]["recall"] == 1.0
    assert got["overall"]["f1"] == 1.0
    assert got["iou"]["mean_iou"] == 1.0
    assert got["accuracy"]["channel_accuracy"] == 1.0
    assert got["accuracy"]["market_accuracy"] == 1.0
    assert got["day_level"]["f1"] == 1.0


@pytest.mark.benchmark_data
def test_never_detect_is_the_floor():
    got = evaluate_split(D.never_detect, "dev")
    assert got["overall"]["recall"] == 0.0
    assert got["overall"]["n_fp"] == 0
    assert got["null_fp_rate"] == 0.0


@pytest.mark.benchmark_data
def test_detect_everything_is_punished_where_it_should_be():
    got = evaluate_split(D.detect_everything, "dev")
    assert got["overall"]["precision"] < 0.2
    assert got["null_fp_rate"] > 0.0


@pytest.mark.benchmark_data
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


@pytest.mark.benchmark_data
def test_null_fp_rate_is_the_published_per_country_year_baseline():
    """I4. `null_fp_rate` is spec section 9 item 7 -- "the cleanest headline
    number" -- and README.md publishes 0.654 for `detect_everything`. Until
    now nothing asserted it: `evaluate_split` computed it inline, and the
    tested `metrics.false_positive_rate` was called only from a unit test, so
    dropping the `* years` factor from the runner roughly DOUBLED the
    published rate (85 of 100 scenarios are 2-year) with the suite still
    green.

    Pinning both ends of the range on the real dev split, at the precision the
    README quotes.
    """
    everything = evaluate_split(D.detect_everything, "dev")
    never = evaluate_split(D.never_detect, "dev")

    assert never["null_fp_rate"] == 0.0
    assert abs(everything["null_fp_rate"] - 0.654) < 0.001, (
        f"the published null FP rate moved: {everything['null_fp_rate']:.3f} "
        f"vs the 0.654 in benchmark/eval/README.md")

    # The denominator is country-YEARS, not countries: a per-country rate
    # would come out at roughly double this.
    assert everything["null_country_years"] > 0
    n_preds = everything["null_fp_rate"] * everything["null_country_years"]
    assert abs(n_preds - round(n_preds)) < 1e-9


@pytest.mark.benchmark_data
def test_the_runner_publishes_the_tested_false_positive_rate():
    """The same number must come out of `metrics.false_positive_rate`, not out
    of a second copy of the arithmetic living in the runner where no test can
    reach it."""
    from benchmark.eval.metrics import false_positive_rate

    got = evaluate_split(D.detect_everything, "dev")
    fake_preds = [None] * int(round(got["null_fp_rate"]
                                    * got["null_country_years"]))
    assert abs(false_positive_rate(fake_preds, got["null_country_years"])
               - got["null_fp_rate"]) < 1e-12
