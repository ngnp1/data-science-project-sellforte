"""Freeze a split so that later claims about it are provable.

A sealed split carries a manifest.sha256 over every file on both the data and
truth sides, plus a SEALED marker recording when it was sealed and the hash of
the scenario definitions it was built from. Editing any file produces a
different manifest; regenerating the split with any parameter changed produces
a different spec_hash. `verify_seal()` checks BOTH: the manifests file by file,
and the recorded spec_hash against the scenario definitions the current code
produces. Spec drift used to be undetectable here -- the marker carried the
hash but nothing compared it -- while spec section 10 has the final evaluation
"verify the seal", so that gap would have been inherited by the run that
matters most.

What the seal does NOT prove: that nobody read the answers. It is a tamper
record, not an access control. The black-box guarantee is procedural (see
`benchmark/spec/scenarios.py` and BENCHMARK.md), and no hash can substitute
for it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from benchmark.harness.runner import DATASETS_DIR
from benchmark.spec.scenarios import Scenario, build_split, spec_hash

MANIFEST_NAME = "manifest.sha256"
SEAL_NAME = "SEALED"
_EXCLUDED = {MANIFEST_NAME, SEAL_NAME}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.name not in _EXCLUDED)


def write_manifest(directory: Path) -> Path:
    directory = Path(directory)
    lines = [f"{_sha256(p)}  {p.relative_to(directory).as_posix()}"
             for p in _files(directory)]
    out = directory / MANIFEST_NAME
    out.write_text("\n".join(lines) + "\n")
    return out


def verify_manifest(directory: Path) -> tuple[bool, list[str]]:
    directory = Path(directory)
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        return False, [f"{directory}: no manifest"]

    expected = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            expected[rel] = digest

    actual = {p.relative_to(directory).as_posix(): p for p in _files(directory)}
    problems = []

    for rel, digest in sorted(expected.items()):
        if rel not in actual:
            problems.append(f"missing: {rel}")
        elif _sha256(actual[rel]) != digest:
            problems.append(f"changed: {rel}")
    for rel in sorted(set(actual) - set(expected)):
        problems.append(f"unexpected: {rel}")

    return not problems, problems


def seal_split(split: str, scenarios: list[Scenario],
               root: Path = DATASETS_DIR) -> dict:
    """Write manifests for both sides and stamp the SEALED marker."""
    root = Path(root)
    data_root, truth_root = root / split, root / f"{split}_truth"
    marker = data_root / SEAL_NAME

    if marker.is_file():
        raise RuntimeError(
            f"{split} is already sealed (marker at {marker}). Sealing again "
            f"would destroy the record of what the original benchmark was. "
            f"Delete the marker deliberately if you really mean to reseal.")

    write_manifest(data_root)
    write_manifest(truth_root)

    info = {
        "split": split,
        "sealed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spec_hash": spec_hash(scenarios),
        "n_scenarios": len(scenarios),
    }
    marker.write_text(json.dumps(info, indent=2, sort_keys=True))
    return info


def is_sealed(split: str, root: Path = DATASETS_DIR) -> bool:
    return (Path(root) / split / SEAL_NAME).is_file()


def verify_seal(split: str, root: Path = DATASETS_DIR,
                scenarios: list[Scenario] | None = None) -> tuple[bool, list[str]]:
    """Check a sealed split against BOTH things the seal records.

    1. the manifests -- every file present, and unchanged byte for byte;
    2. the spec_hash in the SEALED marker, against the scenario definitions
       the current `benchmark/spec/` code produces.

    (2) is what catches spec drift: data that still matches its manifest but
    was built from a definition the repo no longer agrees with. Pass
    `scenarios` explicitly when the sealed tree was built from a subset (the
    tests do); by default the full split is rebuilt from the spec.
    """
    root = Path(root)
    if not is_sealed(split, root):
        return False, [f"{split} is not sealed"]

    problems = []
    for directory in (root / split, root / f"{split}_truth"):
        ok, found = verify_manifest(directory)
        if not ok:
            problems += [f"{directory.name}: {p}" for p in found]

    marker = Path(root) / split / SEAL_NAME
    try:
        recorded = json.loads(marker.read_text()).get("spec_hash")
    except (OSError, ValueError) as exc:
        problems.append(f"{SEAL_NAME}: unreadable ({type(exc).__name__}: {exc})")
    else:
        current = spec_hash(build_split(split) if scenarios is None else scenarios)
        if recorded != current:
            problems.append(
                f"spec drift: {SEAL_NAME} records spec_hash {recorded}, but the "
                f"current scenario definitions hash to {current}")

    return not problems, problems
