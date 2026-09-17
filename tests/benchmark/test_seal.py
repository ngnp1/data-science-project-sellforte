import json

import pytest

from benchmark.harness import seal


@pytest.fixture
def fake_split(tmp_path):
    """Two scenarios' worth of files, data and truth side."""
    for sid in ["test_001_dark", "test_002_step"]:
        d = tmp_path / "test" / sid
        t = tmp_path / "test_truth" / sid
        d.mkdir(parents=True)
        t.mkdir(parents=True)
        (d / "media.csv").write_text("date,media_investment\n2024-01-01,100\n")
        (d / "sales.csv").write_text("date,turnover\n2024-01-01,900\n")
        (t / "ground_truth.csv").write_text("pattern_id\nX\n")
        (t / "meta.json").write_text('{"family": "dark"}')
    return tmp_path


def test_manifest_covers_every_file(fake_split):
    path = seal.write_manifest(fake_split / "test")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4  # two scenarios x two CSVs
    assert all(len(l.split("  ")[0]) == 64 for l in lines)


def test_verify_passes_on_an_untouched_directory(fake_split):
    seal.write_manifest(fake_split / "test")
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert ok and problems == []


def test_verify_detects_a_modified_file(fake_split):
    seal.write_manifest(fake_split / "test")
    (fake_split / "test" / "test_001_dark" / "media.csv").write_text("tampered\n")
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert not ok
    assert any("media.csv" in p and "changed" in p for p in problems)


def test_verify_detects_a_deleted_file(fake_split):
    seal.write_manifest(fake_split / "test")
    (fake_split / "test" / "test_002_step" / "sales.csv").unlink()
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert not ok
    assert any("missing" in p for p in problems)


def test_verify_detects_an_added_file(fake_split):
    seal.write_manifest(fake_split / "test")
    (fake_split / "test" / "test_001_dark" / "extra.csv").write_text("surprise\n")
    ok, problems = seal.verify_manifest(fake_split / "test")
    assert not ok
    assert any("unexpected" in p for p in problems)


def test_seal_split_records_spec_hash_and_counts(fake_split):
    from benchmark.spec import scenarios
    scns = [s for s in scenarios.build_split("test")][:2]
    info = seal.seal_split("test", scns, root=fake_split)

    marker = json.loads((fake_split / "test" / "SEALED").read_text())
    assert marker["spec_hash"] == scenarios.spec_hash(scns)
    assert marker["n_scenarios"] == 2
    assert "sealed_at" in marker
    assert info == marker
    assert seal.is_sealed("test", root=fake_split)


def test_verify_seal_checks_both_data_and_truth(fake_split):
    from benchmark.spec import scenarios
    scns = scenarios.build_split("test")[:2]
    seal.seal_split("test", scns, root=fake_split)
    assert seal.verify_seal("test", root=fake_split, scenarios=scns)[0]

    (fake_split / "test_truth" / "test_001_dark" / "ground_truth.csv").write_text("X\n")
    ok, problems = seal.verify_seal("test", root=fake_split, scenarios=scns)
    assert not ok and problems


def test_verify_seal_detects_spec_drift_even_when_every_file_is_intact(fake_split):
    """The marker's spec_hash used to be written and never read, so a split
    whose scenario DEFINITIONS had changed still verified clean as long as the
    bytes on disk were untouched. Spec section 10 has the final evaluation
    "verify the seal", so that weaker check would have been inherited by the
    one run that must not be wrong."""
    from benchmark.spec import scenarios
    scns = scenarios.build_split("test")[:2]
    seal.seal_split("test", scns, root=fake_split)

    # Nothing on disk is touched; only the definitions we verify against.
    drifted = scenarios.build_split("test")[:3]
    ok, problems = seal.verify_seal("test", root=fake_split, scenarios=drifted)

    assert not ok
    assert any("spec drift" in p for p in problems), problems


def test_sealing_twice_is_refused(fake_split):
    from benchmark.spec import scenarios
    scns = scenarios.build_split("test")[:2]
    seal.seal_split("test", scns, root=fake_split)
    with pytest.raises(RuntimeError, match="already sealed"):
        seal.seal_split("test", scns, root=fake_split)


def test_unsealed_split_reports_false(fake_split):
    assert not seal.is_sealed("dev", root=fake_split)
