"""CLI for generating and sealing the benchmark.

    python -m benchmark.harness.generate --split all --dry-run
    python -m benchmark.harness.generate --split dev
    python -m benchmark.harness.generate --split test
    python -m benchmark.harness.generate --split test --seal
    python -m benchmark.harness.generate --split test --seal-only

--seal and --seal-only require an explicit --split; they refuse --split all,
which would otherwise seal the dev split.

Generation is resumable: a scenario whose outputs already exist is skipped
unless --force is passed, so an interrupted 45-minute run can be restarted.
"""
from __future__ import annotations

import argparse
import bisect
import sys
import time
from pathlib import Path

from benchmark.harness import runner, seal
from benchmark.spec import scenarios

# Measured, channel-count-keyed cost model (seconds of wall-clock per
# country-year), replacing an earlier flat SECONDS_PER_COUNTRY_YEAR = 25.0
# constant.
#
# The flat constant was derived from the shipped 6-channel config and does
# not hold: siMMMulator's runtime is dominated by CHANNEL COUNT, not
# country-years, and scenarios in this benchmark draw n_channels from
# [2, 4, 6, 9, 12] (see benchmark.spec.scenarios._shape). A flat per-country-
# -year constant therefore mis-estimates any split whose channel-count mix
# differs from 6.
#
# These five numbers were measured directly on this machine (macOS, R 4.5
# arm64, the checked-in siMMMulator + generate_with_simmmulator.R) by timing
# a real `Rscript synthetic_data_generator/generate_with_simmmulator.R`
# invocation against a synthetic ONE-country, ONE-year config (empty events
# list) for each channel count, built with
# `benchmark.spec.axes.pick_channels`/`pick_countries` and
# `benchmark.harness.config_writer.write_scenario_configs`. Channel draws
# that happened to be entirely one type (all-impression or all-click) were
# discarded and re-drawn -- an unrelated pre-existing issue in the generator
# rejects homogeneous-type channel lists, and that failure mode is not what
# this table is trying to measure:
#
#     n_channels=2   1 country  1 year  ->   3.62s
#     n_channels=4   1 country  1 year  ->   9.72s
#     n_channels=6   1 country  1 year  ->  21.76s
#     n_channels=9   1 country  1 year  ->  48.20s
#     n_channels=12  1 country  1 year  ->  84.05s
#
# MEASURED SINGLE-PROCESS -- this table UNDERESTIMATES BY ROUGHLY 2x under the
# default 6 workers. Each number above was timed with one Rscript alone on the
# machine; the real 100-scenario run took 99.3 min at --workers 6 against the
# 52 min this table predicts, a contention factor of >= 2.68x once six R
# processes compete for six performance cores. Treat the estimate the CLI
# prints as a floor, not a forecast.
#
# The same contention is what makes runner.R_TIMEOUT_S (1800 s) tighter than it
# looks: the most expensive point in the variation space (8 countries x 12
# channels x 1 year) models at 8 * 84.05 = 672 s here and measured ~840 s, only
# ~2.14x of headroom -- less than the contention factor already observed. It
# duly timed out at 1801.0 s during the real run and had to be retried with
# --workers 1 (see benchmark/BENCHMARK.md, "Runtime").
#
# Re-measure and update this table if the generator, its dependencies, or the
# machine running it changes materially.
COST_PER_COUNTRY_YEAR: dict[int, float] = {
    2: 3.62,
    4: 9.72,
    6: 21.76,
    9: 48.20,
    12: 84.05,
}


def _cost_per_country_year(n_channels: int) -> float:
    """Seconds of wall-clock per country-year for a given channel count.

    Exact hits return the measured value. A channel count between two
    measured points is linearly interpolated; below the smallest or above
    the largest measured point, the nearest measured value is used rather
    than extrapolated, since the table has no evidence of the trend
    continuing past its ends.
    """
    keys = sorted(COST_PER_COUNTRY_YEAR)
    if n_channels in COST_PER_COUNTRY_YEAR:
        return COST_PER_COUNTRY_YEAR[n_channels]
    if n_channels <= keys[0]:
        return COST_PER_COUNTRY_YEAR[keys[0]]
    if n_channels >= keys[-1]:
        return COST_PER_COUNTRY_YEAR[keys[-1]]

    i = bisect.bisect_left(keys, n_channels)
    lo, hi = keys[i - 1], keys[i]
    lo_cost, hi_cost = COST_PER_COUNTRY_YEAR[lo], COST_PER_COUNTRY_YEAR[hi]
    frac = (n_channels - lo) / (hi - lo)
    return lo_cost + frac * (hi_cost - lo_cost)


def _plan(scns):
    country_years = sum(s.meta["n_countries"] * s.years for s in scns)
    est_seconds = sum(
        s.meta["n_countries"] * s.years * _cost_per_country_year(s.meta["n_channels"])
        for s in scns
    )
    return country_years, est_seconds


def _missing_scenarios(split: str, sp_scns, root: Path) -> list[str]:
    """sids in `sp_scns` that are not fully generated on disk under `root`.

    The one completeness check both --seal and --seal-only must use: sealing
    is the mechanism the whole benchmark's black-box claim rests on, so
    nothing may certify a split as sealed on a weaker check (e.g. merely
    `(root / split).is_dir()`) than this one. A missing scenario can arise
    from --limit, a failed scenario, or a --split mismatch -- --seal runs
    right after generation, but that generation may not have covered every
    scenario in the split, so the check still applies there too.

    It delegates to `runner._is_complete` rather than re-deriving the rule:
    this used to check media.csv alone, one module away from a stricter
    definition of "generated" that also requires the truth side. Since
    `run_scenario` writes the data side first, a crash between the two left a
    tree that the weaker check would have happily sealed, with the answers
    missing or half-written. One definition, used by the resume logic and the
    seal alike, is the only way those two cannot drift apart again.
    """
    return [s.sid for s in sp_scns
            if not runner._is_complete(split, s.sid, root)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true",
                   help="regenerate scenarios that already exist")
    p.add_argument("--seal", action="store_true",
                   help="seal the split after generating (test split only in practice)")
    p.add_argument("--seal-only", action="store_true",
                   help="seal an already-generated split without regenerating; "
                        "fails immediately if the split is incomplete")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="generate only the first N scenarios, for smoke tests")
    p.add_argument("--root", default=str(runner.DATASETS_DIR))
    args = p.parse_args(argv)

    if args.split == "all" and (args.seal or args.seal_only):
        # --split defaults to "all", so a bare `--seal` would seal DEV first --
        # violating "dev must stay writable, it is where iteration happens" --
        # and only then hit the already-sealed test split and die on an
        # unhandled RuntimeError, with the damage already done.
        flag = "--seal-only" if args.seal_only else "--seal"
        print(f"ERROR: {flag} requires an explicit --split (dev or test). "
              f"With --split all it would seal the dev split too, and dev must "
              f"stay writable.", file=sys.stderr)
        return 2

    root = Path(args.root)
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    scns = [s for sp in splits for s in scenarios.build_split(sp)]
    if args.limit:
        scns = scns[: args.limit]

    country_years, est_seconds = _plan(scns)
    print(f"{len(scns)} scenarios, {country_years} country-years, "
          f"estimated {est_seconds / 60 / args.workers:.0f} min "
          f"at {args.workers} workers (channel-aware cost model)")
    print(f"spec_hash: {scenarios.spec_hash(scns)}")

    if args.dry_run:
        return 0

    if args.seal_only:
        for sp in splits:
            sp_scns = scenarios.build_split(sp)
            missing = _missing_scenarios(sp, sp_scns, root)
            if missing:
                print(f"ERROR: {sp} is not generated ({len(missing)} of "
                      f"{len(sp_scns)} scenarios missing); generate before "
                      f"sealing", file=sys.stderr)
                return 2
            info = seal.seal_split(sp, sp_scns, root=root)
            print(f"sealed {sp}: {info['n_scenarios']} scenarios, "
                  f"spec_hash {info['spec_hash'][:16]}")
        return 0

    started = time.time()

    def progress(res, done, total):
        flag = {"generated": "ok", "skipped": "--", "failed": "FAIL"}[res["status"]]
        print(f"[{done:3}/{total}] {flag:4} {res['sid']:32} {res['seconds']:6.1f}s")
        if res["status"] == "failed":
            print(f"        {res['error'][:400]}", file=sys.stderr)

    results = runner.run_many(scns, root=root, workers=args.workers,
                              force=args.force, on_done=progress)

    failed = [r for r in results if r["status"] == "failed"]
    print(f"\ndone in {(time.time() - started) / 60:.1f} min: "
          f"{sum(r['status'] == 'generated' for r in results)} generated, "
          f"{sum(r['status'] == 'skipped' for r in results)} skipped, "
          f"{len(failed)} failed")

    if failed:
        print("failed scenarios: " + ", ".join(r["sid"] for r in failed),
              file=sys.stderr)
        return 1

    if args.seal:
        for sp in splits:
            sp_scns = scenarios.build_split(sp)
            missing = _missing_scenarios(sp, sp_scns, root)
            if missing:
                print(f"ERROR: {sp} is not generated ({len(missing)} of "
                      f"{len(sp_scns)} scenarios missing); generate before "
                      f"sealing", file=sys.stderr)
                return 2
            info = seal.seal_split(sp, sp_scns, root=root)
            print(f"sealed {sp}: {info['n_scenarios']} scenarios, "
                  f"spec_hash {info['spec_hash'][:16]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
