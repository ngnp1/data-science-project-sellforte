"""Regression test for the fix-round-1 defect: scripts/fit_calibration.py
must fit on RAW confidence, not on `detection_confidence`, so that running it
again after `detection/calibration_fit.py` already exists does not silently
calibrate an already-calibrated score.

This does NOT delete detection/calibration_fit.py before either fit -- the
whole point is to prove the script no longer needs that manual step. If the
fix regressed and the script started reading the calibrated field again, the
first fit's PAV mapping would visibly warp the second fit's input and the two
knot lists would differ.
"""
import pytest

from scripts.fit_calibration import fit


@pytest.mark.slow
def test_fitting_twice_without_deleting_the_frozen_module_is_idempotent():
    knots_1, n_1, correct_1 = fit()
    knots_2, n_2, correct_2 = fit()
    assert knots_1 == knots_2, (knots_1, knots_2)
    assert n_1 == n_2
    assert correct_1 == correct_2
