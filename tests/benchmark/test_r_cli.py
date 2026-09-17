"""Argument parsing is tested by sourcing cli_args.R directly, so these tests
run in milliseconds instead of invoking siMMMulator."""
import json
import subprocess

import pytest


def _parse(generator_dir, argv):
    """Call parse_cli_args() in R and bring the result back as a dict."""
    quoted = ", ".join(f'"{a}"' for a in argv)
    script = (
        f'source("{generator_dir / "cli_args.R"}"); '
        f"o <- parse_cli_args(c({quoted})); "
        'cat(sprintf(\'{"seed":%d,"config":"%s","events":"%s","outdir":"%s"}\', '
        "o$seed, o$config, o$events, o$outdir))"
    )
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def test_defaults_reproduce_original_behaviour(generator_dir):
    assert _parse(generator_dir, []) == {
        "seed": 42,
        "config": "config.yaml",
        "events": "events_config.yaml",
        "outdir": ".",
    }


def test_all_four_options_are_honoured(generator_dir):
    got = _parse(generator_dir, ["--seed", "1007", "--config", "/tmp/c.yaml",
                                 "--events", "/tmp/e.yaml", "--outdir", "/tmp/out"])
    assert got == {"seed": 1007, "config": "/tmp/c.yaml",
                   "events": "/tmp/e.yaml", "outdir": "/tmp/out"}


def test_unknown_option_is_rejected(generator_dir):
    with pytest.raises(RuntimeError, match="unknown option"):
        _parse(generator_dir, ["--nope", "1"])


def test_option_without_value_is_rejected(generator_dir):
    with pytest.raises(RuntimeError, match="needs a value"):
        _parse(generator_dir, ["--seed"])


def test_non_integer_seed_is_rejected(generator_dir):
    with pytest.raises(RuntimeError, match="must be an integer"):
        _parse(generator_dir, ["--seed", "abc"])


@pytest.mark.slow
def test_seed_changes_the_data_and_is_reproducible(generator_dir, tmp_path):
    """Two runs at the same seed must be byte-identical; a different seed must
    produce different data. Without this, every benchmark scenario of the same
    shape would carry identical noise."""
    tiny_config = tmp_path / "config.yaml"
    tiny_config.write_text((generator_dir / "config.yaml").read_text()
                           .replace("years: 2", "years: 1"))
    no_events = tmp_path / "events.yaml"
    no_events.write_text("[]\n")

    outs = []
    for name, seed in [("a", "7"), ("b", "7"), ("c", "8")]:
        outdir = tmp_path / name
        subprocess.run(
            ["Rscript", str(generator_dir / "generate_with_simmmulator.R"),
             "--seed", seed, "--config", str(tiny_config),
             "--events", str(no_events), "--outdir", str(outdir)],
            check=True, capture_output=True, text=True,
        )
        outs.append((outdir / "raw_daily_wide.csv").read_bytes())

    assert outs[0] == outs[1], "same seed must reproduce byte-identically"
    assert outs[0] != outs[2], "different seed must produce different data"
