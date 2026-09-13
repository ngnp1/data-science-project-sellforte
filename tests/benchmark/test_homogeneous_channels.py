"""Regression test for the sapply()-on-empty-list bug (Task 10b).

`generate_with_simmmulator.R` used to build CHANNELS_IMPRESSIONS/
CHANNELS_CLICKS with `sapply()`, which returns `list()` (not `character(0)`)
when the source list is empty. Any scenario whose channels are entirely one
type -- 34 of the real 100 benchmark scenarios -- hit this: `c()` on a
character vector and an empty list coerces the whole result to a list,
corrupting the `channel` column several steps downstream and crashing
`step_3_generate_media`'s `case_when()` with "object 'channel' not found".

The fix (vapply(..., character(1))) makes the empty side a genuine
character(0), which siMMMulator's step_3_generate_media explicitly supports
(it has an `if (length(channels_impressions) == 0)` no-op guard). This test
proves the real R generator now succeeds end-to-end on a genuine
homogeneous-channel scenario from the frozen spec.
"""
import dataclasses
import subprocess

import pytest

from benchmark.harness.config_writer import write_scenario_configs
from benchmark.spec import scenarios


@pytest.mark.slow
def test_generator_succeeds_on_an_all_impression_scenario(tmp_path, generator_dir):
    """dev_004 is a real scenario from the frozen 100-scenario spec whose
    two channels (YouTube, Instagram) are both impression-type -- exactly the
    shape that used to crash the generator. Reduced to one country and one
    year (keeping the scenario's own 2-channel, all-impression composition,
    which is what is under test) so the run stays to roughly 4 seconds."""
    s = next(x for x in scenarios.build_all() if x.sid == "dev_004")
    assert {ch["type"] for ch in s.channels} == {"impression"}, \
        "dev_004 is expected to be all-impression; spec may have changed"

    minimal = dataclasses.replace(s, countries=(s.countries[0],), years=1)

    cfg_path, ev_path = write_scenario_configs(minimal, tmp_path)
    outdir = tmp_path / "out"

    proc = subprocess.run(
        ["Rscript", str(generator_dir / "generate_with_simmmulator.R"),
         "--seed", str(minimal.seed), "--config", str(cfg_path),
         "--events", str(ev_path), "--outdir", str(outdir)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert "object 'channel' not found" not in proc.stdout + proc.stderr

    raw_csv = outdir / "raw_daily_wide.csv"
    assert raw_csv.is_file()

    header = raw_csv.read_text().splitlines()[0]
    assert "impressions_YouTube" in header
    assert "impressions_Instagram" in header
    # a genuinely all-impression run must carry no click-type columns
    assert "clicks_" not in header
