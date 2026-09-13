from benchmark.eval.breakdowns import CONFOUNDED_AXES, breakdown_by


def test_confounded_axes_are_named_exactly():
    """BENCHMARK.md: only noise_level is stratified within family. trend_p and
    market_spread are family-confounded on BOTH splits and must not be reported
    as axis effects."""
    assert CONFOUNDED_AXES == frozenset({"trend_p", "market_spread"})
    assert "noise_level" not in CONFOUNDED_AXES


def test_breakdown_groups_scenarios_by_axis_value():
    per_scenario = {
        "dev_001": {"event_level": {"n_tp": 1, "n_fp": 0, "n_fn": 0}},
        "dev_002": {"event_level": {"n_tp": 0, "n_fp": 1, "n_fn": 1}},
    }
    meta = {"dev_001": {"noise_level": "low"}, "dev_002": {"noise_level": "high"}}
    got = breakdown_by(per_scenario, meta, "noise_level")
    assert got["low"]["f1"] == 1.0
    assert got["high"]["f1"] == 0.0
    assert got["low"]["n_scenarios"] == 1


def test_confounded_axis_carries_a_warning_in_its_result():
    per_scenario = {"dev_001": {"event_level": {"n_tp": 1, "n_fp": 0, "n_fn": 0}}}
    meta = {"dev_001": {"trend_p": 0.0}}
    got = breakdown_by(per_scenario, meta, "trend_p")
    assert got["_warning"], "confounded axis reported without a warning"
    assert "confounded" in got["_warning"].lower()
