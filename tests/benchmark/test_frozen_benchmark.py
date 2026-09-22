"""Standing guarantees about the benchmark on disk. These run on every suite
invocation from here on, so a later change that quietly breaks the seal or
leaks ground truth into a dataset directory fails immediately."""
import json
from pathlib import Path

import pytest

from benchmark.harness import runner, seal
from benchmark.spec import scenarios

DATASETS = runner.DATASETS_DIR
pytestmark = pytest.mark.skipif(not (DATASETS / "test" / "SEALED").is_file(),
                                reason="benchmark not generated yet")


@pytest.mark.benchmark_data
@pytest.mark.parametrize("split,expected", [("dev", 45), ("test", 55)])
def test_every_scenario_was_generated(split, expected):
    dirs = [p for p in (DATASETS / split).iterdir() if p.is_dir()]
    assert len(dirs) == expected


@pytest.mark.benchmark_data
@pytest.mark.parametrize("split", ["dev", "test"])
def test_dataset_directories_leak_nothing(split):
    """The black-box guarantee, checked against what is actually on disk."""
    for d in (DATASETS / split).iterdir():
        if d.is_dir():
            assert sorted(p.name for p in d.iterdir()) == ["media.csv", "sales.csv"], d


@pytest.mark.benchmark_data
@pytest.mark.parametrize("split", ["dev", "test"])
def test_truth_directories_are_complete(split):
    for s in scenarios.build_split(split):
        t = runner.truth_dir(split, s.sid)
        assert (t / "ground_truth.csv").is_file(), s.sid
        assert (t / "meta.json").is_file(), s.sid


@pytest.mark.benchmark_data
def test_test_split_seal_still_verifies():
    ok, problems = seal.verify_seal("test")
    assert ok, problems


def test_seal_records_the_current_spec_hash():
    marker = json.loads((DATASETS / "test" / "SEALED").read_text())
    assert marker["spec_hash"] == scenarios.spec_hash(scenarios.build_split("test"))
    assert marker["n_scenarios"] == 55


def test_dev_split_is_not_sealed():
    """Dev must stay writable -- it is where iteration is allowed to happen."""
    assert not seal.is_sealed("dev")
