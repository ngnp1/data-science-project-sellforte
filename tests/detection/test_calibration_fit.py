import pathlib

import pytest

from detection import params
from detection.calibrate import apply_calibration, fit_pav
from detection.calibration_fit import KNOTS


def test_the_frozen_fit_is_monotone():
    values = [v for _, v in KNOTS]
    assert values == sorted(values), values


def test_the_frozen_fit_is_python_source_not_a_data_file():
    """detection/ may contain nothing but .py -- a JSON fit would be invisible
    to both the import gate and the final-run audit hash."""
    assert pathlib.Path("detection/calibration_fit.py").suffix == ".py"


def test_calibrated_confidence_stays_in_range():
    for s in [0.0, 0.1, 0.35, 0.6, 0.85, 1.0]:
        assert 0.0 <= apply_calibration(s, KNOTS) <= 1.0


# The archived fit is not applied to the current heuristic scores.
