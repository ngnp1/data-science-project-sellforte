import pytest

from detection import params

# Files under detection/ that are generated DATA, not logic, and are
# therefore exempt from the "no hardcoded threshold" guard below -- exactly
# like params.py itself, which the guard already excludes because it is the
# source of the needles, not a place a needle could hide.
#
# detection/calibration_fit.py (Task 10) is a frozen calibration mapping
# rendered by detection/calibrate.py's render_module(). Its bin-upper-edge
# knots are fixed decile cuts (0.1, 0.2, ..., 1.0) that will collide with
# several of params.py's float values on pure coincidence of decimal
# formatting (e.g. "0.2", "0.5", "0.6", "0.7") no matter what the fit ever
# learns. That is not a hardcoded threshold; it is fitted numbers, the same
# category of thing params.py itself is.
#
# Keep this set to exactly the one file it exists for -- a broad exemption
# would let a real hardcoded threshold hide behind it. See
# test_the_threshold_guard_exemption_is_exactly_one_file below.
GUARD_EXEMPT_FILENAMES = {"calibration_fit.py"}


def _threshold_guard_needles():
    """The exact needle-derivation logic test_no_logic_module_hardcodes_a_
    threshold uses: every FLOAT-valued top-level attribute of params.py,
    formatted the way it would appear in source. Shared so any other test
    that needs to know whether text is admissible under the guard checks the
    same needles the guard itself checks."""
    needles = []
    for name in dir(params):
        if name.startswith("_"):
            continue
        val = getattr(params, name)
        if isinstance(val, float):
            needles.append(str(val))
    return needles


def _threshold_guard_offenders(filename: str, text: str, needles) -> list[str]:
    """The exact skip-and-scan logic test_no_logic_module_hardcodes_a_
    threshold applies to one file's text, given its needle list. Shared so a
    test can check whether some rendered text WOULD be admissible if it
    lived under detection/ as `filename`, without re-deriving the rule."""
    if filename == "params.py" or filename in GUARD_EXEMPT_FILENAMES:
        return []
    return [f"{filename}: {needle}" for needle in needles if needle in text]


def test_the_threshold_guard_exemption_is_exactly_one_file():
    """A broad exemption would let a real hardcoded threshold hide behind
    it -- pin the exemption list to precisely the one generated file it
    exists for."""
    assert GUARD_EXEMPT_FILENAMES == {"calibration_fit.py"}


def test_every_documented_parameter_exists_with_the_spec_value():
    """Spec section 7's parameter table. These are the defaults the design was
    reasoned about; changing one is a deliberate act, not a typo."""
    assert params.RHO == 0.15
    assert params.EPS_ABS == 1e-6
    assert params.MIN_DAYS == 7
    assert params.RUN_RATIO == 3.0
    assert params.MIN_RUNS_FOR_RATIO == 5
    assert params.W == 21
    assert params.Z_THRESH == 3.5
    assert params.SIGMA_FLOOR == 0.05
    assert params.PERSIST == 14
    assert params.SHARPNESS == 0.6
    # Extracted from literals during Task 5 and omitted from this list until the
    # final review found both unpinned by any test, behavioural or value.
    assert params.PERSIST_FRACTION == 0.5
    assert params.EPISODE_MATCH_BAND == 2.0
    assert params.SHARPNESS_WINDOW == 3
    assert params.ROLLING == 7
    assert params.MAD_TO_SIGMA == 1.4826
    assert params.PULSE_MIN_RUNS == 2
    assert params.PULSE_LEN_IQR_RATIO == 0.5
    assert params.ONSET_SPREAD == 14
    assert params.Z_SATURATION == 8.0
    assert params.DURATION_SATURATION_MULT == 2.0
    assert params.DISTINCTIVENESS_SATURATION == 3.0
    assert params.CORROBORATION_CONTRADICTED == 0.2
    assert params.CORROBORATION_UNKNOWN == 0.6
    assert params.ADSTOCK_HALF_LIFE == 7.0
    assert params.ADSTOCK_WINDOWS_FOR_FULL_CREDIT == 4.0
    assert params.CENSORING_PENALTY == 0.7
    assert params.CONFOUNDED_PENALTY == 0.6
    assert params.SALES_BASELINE_WEEKS == 8
    assert params.SALES_SIGMA_FLOOR_FRAC == 0.02
    assert params.SALES_SNR_SATURATION == 3.0
    assert params.SALES_SNR_UNKNOWN == 0.5


def test_confidence_and_informativeness_weight_tables_have_the_spec_values():
    """Section 8's weight tables, pinned the same way the scalar thresholds
    above are: a deliberate act to change, not a typo to slip past silently."""
    assert params.CONFIDENCE_WEIGHTS["dark_period"] == {
        "magnitude_evidence": 0.2, "duration_evidence": 0.15,
        "distinctiveness": 0.15, "edge_sharpness": 0.1,
        "corroboration": 0.15, "consistency": 0.25}
    assert params.CONFIDENCE_WEIGHTS["single_channel"] == \
        params.CONFIDENCE_WEIGHTS["dark_period"]
    assert params.CONFIDENCE_WEIGHTS["natural_holdout"] == {
        "magnitude_evidence": 0.25, "duration_evidence": 0.2,
        "distinctiveness": 0.25, "edge_sharpness": 0.1,
        "corroboration": 0.2, "consistency": 0.0}
    assert params.CONFIDENCE_WEIGHTS["channel_pulse"] == {
        "magnitude_evidence": 0.25, "duration_evidence": 0.1,
        "distinctiveness": 0.3, "edge_sharpness": 0.15,
        "corroboration": 0.2, "consistency": 0.0}
    assert params.CONFIDENCE_WEIGHTS["staggered_launch"] == {
        "magnitude_evidence": 0.2, "duration_evidence": 0.2,
        "distinctiveness": 0.2, "edge_sharpness": 0.1,
        "corroboration": 0.3, "consistency": 0.0}
    assert params.CONFIDENCE_WEIGHTS["step_change"] == {
        "magnitude_evidence": 0.45, "duration_evidence": 0.2,
        "distinctiveness": 0.0, "edge_sharpness": 0.25,
        "corroboration": 0.1, "consistency": 0.0}
    assert params.TYPE_PRIOR == {
        "dark_period": 1.0, "single_channel": 0.9, "channel_pulse": 0.9,
        "natural_holdout": 0.75, "staggered_launch": 0.6, "step_change": 0.45}
    assert params.CONTROL_SCORE == {
        "peers": 1.0, "sibling_channels": 0.7, "none": 0.3}
    assert params.INFORMATIVENESS_WEIGHTS == {
        "duration_adequacy": 0.25, "contrast": 0.15, "cleanliness": 0.15,
        "control_availability": 0.2, "sales_snr": 0.15, "type_prior": 0.1}


def test_confidence_weight_rows_name_exactly_the_six_sub_scores():
    from detection.model import EVENT_TYPES
    names = {"magnitude_evidence", "duration_evidence", "distinctiveness",
             "edge_sharpness", "corroboration", "consistency"}
    assert set(params.CONFIDENCE_WEIGHTS) == set(EVENT_TYPES)
    for event_type, weights in params.CONFIDENCE_WEIGHTS.items():
        assert set(weights) == names, event_type
        assert sum(weights.values()) == pytest.approx(1.0), event_type


def test_informativeness_weights_name_exactly_the_six_drivers_and_sum_to_one():
    """Spec section 8's sixth informativeness driver, sales_snr, was added in
    fix round 1 -- this pins the row's full key set (so a driver silently
    dropped from the row is caught here, not just wherever it happens to be
    read) and re-confirms the row still sums to 1 after the rebalance."""
    names = {"duration_adequacy", "contrast", "cleanliness",
             "control_availability", "sales_snr", "type_prior"}
    assert set(params.INFORMATIVENESS_WEIGHTS) == names
    assert sum(params.INFORMATIVENESS_WEIGHTS.values()) == pytest.approx(1.0)


def test_window_parameters_are_whole_weeks():
    """W and ROLLING are multiples of 7 so day-of-week structure cancels
    instead of aliasing into the level estimate."""
    assert params.W % 7 == 0
    assert params.ROLLING % 7 == 0


def test_no_logic_module_hardcodes_a_threshold():
    """Every threshold lives here. A magic number in a primitive is how two
    parameters silently drift apart.

    This test derives the list of forbidden literals from params.py itself,
    checking only FLOAT-valued parameters. Integer thresholds (7, 3, 2, 14, 21)
    are deliberately out of scope because their literals are indistinguishable
    from ordinary indices, ranges and slicing. A value written differently
    (e.g. .6 for 0.6, or 7/2 for 3.5) would also slip past this guard.

    detection/calibration_fit.py is exempt by name (GUARD_EXEMPT_FILENAMES,
    above) -- it is a generated data file, not logic, exactly like params.py
    itself. See test_the_threshold_guard_exemption_is_exactly_one_file for
    why that exemption cannot silently widen, and
    tests/detection/test_calibrate.py's guard-survival test for proof that a
    realistic rendered fit is actually clean under this exemption.
    """
    import pathlib

    needles = _threshold_guard_needles()

    root = pathlib.Path(__file__).resolve().parents[2] / "detection"
    offenders = []
    for py in root.rglob("*.py"):
        offenders.extend(
            _threshold_guard_offenders(py.name, py.read_text(), needles))
    assert not offenders, f"hardcoded thresholds: {offenders}"
