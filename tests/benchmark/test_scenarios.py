import os
import subprocess
import sys

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
        # siMMMulator rejects a one-channel config outright (Task 4 probe),
        # so this is not just a style preference -- it is load-bearing.
        assert len(s.channels) >= 2, s.sid
        head = s.channels[:-1]
        assert "spend_share_min" not in s.channels[-1]
        for c in head:
            assert "spend_share_min" in c and "spend_share_max" in c, s.sid
            assert c["spend_share_min"] > 0, s.sid
            assert c["spend_share_max"] > 0, s.sid
            assert c["spend_share_min"] < c["spend_share_max"], s.sid
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


def _is_country_level(e):
    return e["channel"] == "ALL" or e["pattern_type"] == "single_channel"


def test_mixed_events_never_contradict_within_a_country(all_scenarios):
    """A country-level event (dark_period/single_channel) must own its whole
    country; nothing else may touch that country, and two channel-level
    events sharing a country must target different channels. Otherwise R
    applies both in sequence and one of them becomes invisible in the data
    while ground truth still lists it as something to detect."""
    mixed = [s for s in all_scenarios if s.family == "mixed"]
    assert mixed
    for s in mixed:
        country_level = [e for e in s.events if _is_country_level(e)]
        channel_level = [e for e in s.events if not _is_country_level(e)]

        cl_countries = [e["country"] for e in country_level]
        assert len(cl_countries) == len(set(cl_countries)), \
            f"{s.sid}: more than one country-level event in the same country"

        cl_set = set(cl_countries)
        for e in channel_level:
            assert e["country"] not in cl_set, \
                f"{s.sid}: {e['pattern_id']} targets a country that also " \
                f"has a country-level event"

        # A single builder call (e.g. channel_pulse) can legitimately emit
        # several entries that all share (country, channel) -- one entry per
        # on/off window of the SAME logical event. That is not a collision.
        # A collision is two DIFFERENT event kinds landing on the same
        # (country, channel).
        seen = {}
        for e in channel_level:
            key = (e["country"], e["channel"])
            pt = e["pattern_type"]
            assert seen.get(key, pt) == pt, \
                f"{s.sid}: (country, channel) {key} is used by both " \
                f"{seen[key]} and {pt}"
            seen[key] = pt


def test_single_channel_market_leaves_exactly_one_channel_active(all_scenarios):
    targets = [s for s in all_scenarios if s.family == "edge"
               and s.meta.get("edge_case") == "single_channel_market"]
    assert targets
    for s in targets:
        n_days = 365 * s.years
        held_out_whole_series = {
            e["channel"] for e in s.events
            if e["pattern_type"] == "natural_holdout"
            and e["start_day"] == 0 and e["end_day"] == n_days
            and e["multiplier"] == 0
        }
        all_names = {c["name"] for c in s.channels}
        active = all_names - held_out_whole_series
        assert len(active) == 1, f"{s.sid}: active channels = {active}"


def test_launch_scenarios_are_actually_staggered(all_scenarios):
    launches = [s for s in all_scenarios if s.family == "launch"]
    assert launches
    for s in launches:
        onsets = [e["end_day"] for e in s.events
                  if e["pattern_type"] == "staggered_launch"]
        assert len(onsets) == len(set(onsets)), \
            f"{s.sid}: two markets launch on the same day {onsets}"
        launched = {e["country"] for e in s.events
                    if e["pattern_type"] == "staggered_launch"}
        assert len(launched) < len(s.countries), \
            f"{s.sid}: no comparison-group market carries the channel " \
            f"from day one"


def test_build_is_deterministic_across_processes():
    """test_build_is_deterministic only proves determinism within one
    interpreter. The sealed test split's whole value rests on determinism
    across process boundaries, so shell out to a fresh interpreter (twice,
    with different PYTHONHASHSEED values so hash-ordering bugs would surface)
    and compare its spec_hash against this process's."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    script = (
        "from benchmark.spec import scenarios\n"
        "print(scenarios.spec_hash(scenarios.build_all()))\n"
    )
    in_process_hash = scenarios.spec_hash(scenarios.build_all())

    for hash_seed in ("0", "998877"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == in_process_hash, \
            f"PYTHONHASHSEED={hash_seed}: {result.stdout!r} != {in_process_hash!r}"
