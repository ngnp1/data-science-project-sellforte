from detection import params


def test_every_documented_parameter_exists_with_the_spec_value():
    """Spec section 7's parameter table. These are the defaults the design was
    reasoned about; changing one is a deliberate act, not a typo."""
    assert params.RHO == 0.05
    assert params.MIN_DAYS == 7
    assert params.RUN_RATIO == 3.0
    assert params.W == 21
    assert params.Z_THRESH == 3.5
    assert params.SIGMA_FLOOR == 0.05
    assert params.PERSIST == 14
    assert params.SHARPNESS == 0.6
    assert params.ONSET_SPREAD == 14
    assert params.MAD_TO_SIGMA == 1.4826


def test_window_parameters_are_whole_weeks():
    """W and ROLLING are multiples of 7 so day-of-week structure cancels
    instead of aliasing into the level estimate."""
    assert params.W % 7 == 0
    assert params.ROLLING % 7 == 0


def test_no_logic_module_hardcodes_a_threshold():
    """Every threshold lives here. A magic number in a primitive is how two
    parameters silently drift apart."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2] / "detection"
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "params.py":
            continue
        text = py.read_text()
        for needle in ["0.05", "3.5", "0.6", "1.4826"]:
            if needle in text:
                offenders.append(f"{py.name}: {needle}")
    assert not offenders, f"hardcoded thresholds: {offenders}"
