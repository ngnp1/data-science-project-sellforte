import subprocess

from benchmark.harness import generate


def _run(venv_python, project_root, *args):
    return subprocess.run(
        [str(venv_python), "-m", "benchmark.harness.generate", *args],
        cwd=project_root, capture_output=True, text=True,
    )


def test_dry_run_reports_the_plan_without_generating(venv_python, project_root):
    proc = _run(venv_python, project_root, "--split", "all", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "100 scenarios" in proc.stdout
    assert "country-years" in proc.stdout
    assert "estimated" in proc.stdout.lower()


def test_dry_run_can_be_limited_to_one_split(venv_python, project_root):
    proc = _run(venv_python, project_root, "--split", "dev", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "45 scenarios" in proc.stdout


def test_seal_only_on_an_ungenerated_split_fails_loudly(venv_python, project_root,
                                                        tmp_path):
    """--seal-only must refuse immediately rather than starting a 45-minute
    generation run."""
    proc = _run(venv_python, project_root, "--split", "test", "--seal-only",
                "--root", str(tmp_path))
    assert proc.returncode != 0
    assert "not generated" in (proc.stdout + proc.stderr)


# --- channel-aware cost model ---------------------------------------------
#
# The flat SECONDS_PER_COUNTRY_YEAR constant from the original brief was
# replaced with a measured, channel-count-keyed cost table, because
# siMMMulator's runtime is dominated by channel count, not country-years.
# These tests pin that behaviour: the estimate must actually come from the
# table (not a constant), and it must grow with channel count.

def test_cost_table_has_a_measured_entry_for_every_channel_count_in_the_spec():
    from benchmark.spec import axes

    channel_counts = {2, 4, 6, 9, 12}
    assert channel_counts <= set(generate.COST_PER_COUNTRY_YEAR)
    # sanity: these are the only counts the scenario spec ever draws
    assert set(generate.COST_PER_COUNTRY_YEAR) == channel_counts
    del axes  # imported only to assert the module exists where expected


def test_cost_per_country_year_increases_with_channel_count():
    costs = [generate._cost_per_country_year(n) for n in (2, 4, 6, 9, 12)]
    assert costs == sorted(costs)
    assert costs[0] < costs[-1]


def test_cost_per_country_year_interpolates_between_measured_points():
    lo, hi = generate.COST_PER_COUNTRY_YEAR[4], generate.COST_PER_COUNTRY_YEAR[6]
    mid = generate._cost_per_country_year(5)
    assert lo < mid < hi


def test_cost_per_country_year_clamps_outside_the_measured_range():
    assert generate._cost_per_country_year(1) == generate.COST_PER_COUNTRY_YEAR[2]
    assert generate._cost_per_country_year(20) == generate.COST_PER_COUNTRY_YEAR[12]


def test_plan_scales_with_channel_count_not_just_country_years():
    """Two scenario-shaped objects with identical country-years but different
    channel counts must produce different estimates -- proof the estimate is
    actually reading the cost table instead of a flat per-country-year rate."""
    class _FakeScenario:
        def __init__(self, n_channels):
            self.years = 1
            self.meta = {"n_countries": 1, "n_channels": n_channels}

    cheap = [_FakeScenario(2)]
    pricey = [_FakeScenario(12)]

    cy_cheap, seconds_cheap = generate._plan(cheap)
    cy_pricey, seconds_pricey = generate._plan(pricey)

    # Same country-years, very different estimated cost.
    assert cy_cheap == cy_pricey == 1
    assert seconds_pricey > seconds_cheap * 10
    assert seconds_cheap == generate.COST_PER_COUNTRY_YEAR[2]
    assert seconds_pricey == generate.COST_PER_COUNTRY_YEAR[12]
