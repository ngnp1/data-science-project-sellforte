import numpy as np
import pytest

from detection.calibrate import apply_calibration, fit_pav, render_module


def test_a_perfectly_calibrated_input_is_left_alone():
    """If raw confidence already matches empirical precision, calibration must
    be close to the identity -- a mapping that moves well-calibrated scores is
    doing harm."""
    rng = np.random.default_rng(0)
    scores, correct = [], []
    for s in np.linspace(0.05, 0.95, 10):
        for _ in range(200):
            scores.append(float(s))
            correct.append(bool(rng.random() < s))
    knots = fit_pav(scores, correct, n_bins=10)
    for s in [0.15, 0.45, 0.75, 0.95]:
        assert abs(apply_calibration(s, knots) - s) < 0.12


def test_an_overconfident_detector_is_pulled_down():
    """Every event scored 0.9 but only 30% are right. Calibration must report
    about 0.3, because the whole purpose is to make the number a testable
    claim."""
    scores = [0.9] * 100
    correct = [True] * 30 + [False] * 70
    knots = fit_pav(scores, correct, n_bins=10)
    assert apply_calibration(0.9, knots) == pytest.approx(0.3, abs=0.05)


def test_the_mapping_is_monotone_non_decreasing():
    """Pool-adjacent-violators exists to guarantee this. A calibration that
    inverts anywhere would rank a worse event above a better one."""
    rng = np.random.default_rng(1)
    scores = list(rng.random(500))
    correct = [bool(rng.random() < s ** 2) for s in scores]
    knots = fit_pav(scores, correct, n_bins=10)
    values = [v for _, v in knots]
    assert values == sorted(values), values


def test_violating_bins_are_pooled_rather_than_left_inverted():
    """The defining behaviour of PAV, with a hand-built inversion: the 0.3 bin
    outperforms the 0.5 bin. They must merge to their shared rate, not stay
    crossed."""
    scores = [0.35] * 100 + [0.55] * 100
    correct = ([True] * 80 + [False] * 20) + ([True] * 40 + [False] * 60)
    knots = fit_pav(scores, correct, n_bins=10)
    low = apply_calibration(0.35, knots)
    high = apply_calibration(0.55, knots)
    assert low <= high
    assert low == pytest.approx(0.6, abs=0.05)
    assert high == pytest.approx(0.6, abs=0.05)


def test_an_empty_fit_returns_the_identity():
    """No data is not the same as 'everything is wrong'. With nothing to learn
    from, calibration must not alter the score."""
    knots = fit_pav([], [], n_bins=10)
    for s in [0.0, 0.25, 0.5, 1.0]:
        assert apply_calibration(s, knots) == pytest.approx(s)


def test_calibration_output_stays_in_range():
    rng = np.random.default_rng(2)
    scores = list(rng.random(200))
    correct = [bool(rng.random() < 0.5) for _ in scores]
    knots = fit_pav(scores, correct, n_bins=10)
    for s in np.linspace(0, 1, 21):
        assert 0.0 <= apply_calibration(float(s), knots) <= 1.0


def test_the_rendered_module_is_importable_and_round_trips():
    """The frozen mapping ships as Python because detection/ may contain
    nothing but source -- a JSON file there fails the gating test."""
    knots = [(0.5, 0.2), (1.0, 0.8)]
    src = render_module(knots)
    namespace = {}
    exec(compile(src, "calibration_fit.py", "exec"), namespace)
    assert namespace["KNOTS"] == knots


def test_a_score_of_exactly_one_lands_in_the_last_bin_not_off_the_end():
    """Off-by-one bin-edge arithmetic is the classic way to make apply_
    calibration index past the fitted range, or silently mis-map the top
    score into the wrong bucket."""
    scores = [0.05] * 50 + [1.0] * 50
    correct = [False] * 50 + [True] * 50
    knots = fit_pav(scores, correct, n_bins=10)
    # The score of exactly 1.0 must have been counted in its own top bin, not
    # dropped and not folded into the 0.05 bin.
    assert apply_calibration(1.0, knots) == pytest.approx(1.0, abs=0.05)
    assert apply_calibration(0.05, knots) == pytest.approx(0.0, abs=0.05)


def test_pooling_actually_merges_not_just_bins_a_three_bin_cascade():
    """A chain of three violating bins must pool into ONE block at their
    shared rate. Plain binning (no pooling) would leave three distinct,
    inverted values instead of one merged, monotone one -- this is the test
    that a per-bin histogram cannot pass by accident."""
    scores = [0.15] * 100 + [0.35] * 100 + [0.55] * 100
    # Rates 0.9, 0.5, 0.1 -- strictly decreasing, so PAV must pool all three
    # into a single block at the combined rate (90+50+10)/300 = 0.5.
    correct = (
        [True] * 90 + [False] * 10
        + [True] * 50 + [False] * 50
        + [True] * 10 + [False] * 90
    )
    knots = fit_pav(scores, correct, n_bins=10)
    v_low = apply_calibration(0.15, knots)
    v_mid = apply_calibration(0.35, knots)
    v_high = apply_calibration(0.55, knots)
    assert v_low == pytest.approx(0.5, abs=0.02)
    assert v_mid == pytest.approx(0.5, abs=0.02)
    assert v_high == pytest.approx(0.5, abs=0.02)
    # And the fitted knots themselves must contain exactly one distinct value
    # across these three original bins -- proof they were merged into one
    # block, not left as three separate (inverted) ones.
    distinct_values = {round(v, 6) for _, v in knots}
    assert len(distinct_values) == 1


def test_rendered_output_is_admissible_under_the_threshold_guard():
    """The round-trip test above proves render_module's output is
    importable; it does not prove it is admissible where it will actually
    live. Task 10 writes render_module's output to detection/
    calibration_fit.py, where tests/detection/test_params.py's
    no-hardcoded-threshold guard scans it. A realistic decile fit's bin
    upper edges (0.1, 0.2, ..., 1.0) collide on pure decimal coincidence with
    several of params.py's float values (e.g. "0.2" and "0.5" are always
    present as edges regardless of the data), so this must be checked
    against the guard's OWN needle-derivation and scan logic, not just
    against render_module's contract in isolation."""
    from tests.detection.test_params import (
        _threshold_guard_needles, _threshold_guard_offenders,
    )

    rng = np.random.default_rng(3)
    scores = list(rng.random(2000))
    correct = [bool(rng.random() < s) for s in scores]
    knots = fit_pav(scores, correct, n_bins=10)
    src = render_module(knots)

    needles = _threshold_guard_needles()
    offenders = _threshold_guard_offenders("calibration_fit.py", src, needles)
    assert not offenders, offenders
