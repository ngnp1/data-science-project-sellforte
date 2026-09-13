import pytest

from benchmark.spec import scenarios
from benchmark.spec.events import NON_EVENT_TYPES


@pytest.fixture(scope="module")
def all_scenarios():
    return scenarios.build_all()


def test_exactly_one_hundred_scenarios(all_scenarios):
    assert len(all_scenarios) == 100
    assert sum(s.split == "dev" for s in all_scenarios) == 45
    assert sum(s.split == "test" for s in all_scenarios) == 55


def test_family_counts_match_the_spec_table(all_scenarios):
    for family, (n_dev, n_test) in scenarios.FAMILY_COUNTS.items():
        got_dev = sum(s.family == family and s.split == "dev" for s in all_scenarios)
        got_test = sum(s.family == family and s.split == "test" for s in all_scenarios)
        assert (got_dev, got_test) == (n_dev, n_test), f"family {family}"


def test_seed_ranges_are_disjoint_between_splits(all_scenarios):
    dev = {s.seed for s in all_scenarios if s.split == "dev"}
    test = {s.seed for s in all_scenarios if s.split == "test"}
    assert dev.isdisjoint(test)
    assert all(1000 <= s < 2000 for s in dev)
    assert all(5000 <= s < 6000 for s in test)


def test_scenario_ids_are_unique(all_scenarios):
    ids = [s.sid for s in all_scenarios]
    assert len(ids) == len(set(ids))


def test_build_is_deterministic():
    a, b = scenarios.build_all(), scenarios.build_all()
    assert scenarios.spec_hash(a) == scenarios.spec_hash(b)
    assert [s.sid for s in a] == [s.sid for s in b]


def test_null_scenarios_carry_no_events(all_scenarios):
    nulls = [s for s in all_scenarios if s.family == "null"]
    assert len(nulls) == 9
    assert all(s.events == () for s in nulls)


def test_every_event_references_a_country_and_channel_in_its_scenario(all_scenarios):
    for s in all_scenarios:
        codes = {c["code"] for c in s.countries}
        names = {c["name"] for c in s.channels}
        for e in s.events:
            assert e["country"] in codes, f"{s.sid}: unknown country {e['country']}"
            assert e["channel"] == "ALL" or e["channel"] in names, \
                f"{s.sid}: unknown channel {e['channel']}"


def test_every_event_window_fits_inside_the_series(all_scenarios):
    for s in all_scenarios:
        n_days = 365 * s.years
        for e in s.events:
            assert 0 <= e["start_day"] < e["end_day"] <= n_days, f"{s.sid}: {e}"


def test_channel_spend_shares_are_valid_for_the_generator(all_scenarios):
    for s in all_scenarios:
        head = s.channels[:-1]
        assert "spend_share_min" not in s.channels[-1]
        assert sum(c["spend_share_max"] for c in head) < 0.92, s.sid


def test_mixed_scenarios_really_do_mix(all_scenarios):
    mixed = [s for s in all_scenarios if s.family == "mixed"]
    assert len(mixed) == 18
    for s in mixed:
        kinds = {e["pattern_type"] for e in s.events} - NON_EVENT_TYPES
        assert len(kinds) >= 3, f"{s.sid} has only {kinds}"


def test_every_variation_axis_is_exercised_on_both_splits(all_scenarios):
    for split in ["dev", "test"]:
        subset = [s for s in all_scenarios if s.split == split]
        assert {s.meta["noise_level"] for s in subset} == {"low", "med", "high"}
        assert {s.meta["market_spread"] for s in subset} == \
            {"tight", "moderate", "extreme"}
        assert {s.meta["n_countries"] for s in subset} >= {1, 2, 3, 5, 8}
        assert {s.meta["n_channels"] for s in subset} >= {2, 4, 6, 9, 12}
        assert {s.meta["trend_p"] for s in subset} == {0.0, 0.5, 1.0}


def test_eight_country_scenarios_are_capped_at_one_year(all_scenarios):
    """Runtime control: 8 countries x 2 years would blow the generation budget."""
    for s in all_scenarios:
        if s.meta["n_countries"] >= 8:
            assert s.years == 1, s.sid


def test_edge_case_family_covers_every_named_case(all_scenarios):
    for split in ["dev", "test"]:
        cases = {s.meta["edge_case"] for s in all_scenarios
                 if s.family == "edge" and s.split == split}
        assert cases == {
            "censored_start", "censored_end", "back_to_back", "overlapping",
            "too_short", "very_long", "single_channel_market",
            "intermittent_channel", "gradual_ramp", "global_pause",
        }, f"{split}: {cases}"


def test_near_zero_and_exact_zero_are_both_represented(all_scenarios):
    mults = {e["multiplier"] for s in all_scenarios for e in s.events
             if e["pattern_type"] == "natural_holdout"}
    assert 0 in mults
    assert any(0 < m < 0.1 for m in mults)


def test_meta_carries_no_event_locations(all_scenarios):
    """meta.json lands on the truth side, but keeping it free of day offsets
    means a leak would be harmless rather than fatal."""
    for s in all_scenarios:
        blob = repr(s.meta)
        assert "start_day" not in blob and "end_day" not in blob
