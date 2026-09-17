"""Serialise a Scenario into the two YAML files the generator reads.

The only subtlety is numeric types: numpy scalars survive from the sampling
code, and PyYAML tags them as python objects, which R's yaml reader rejects.
Everything is coerced to a plain int, float or str on the way out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from benchmark.spec.scenarios import Scenario

START_DATE = "2024/01/01"
REVENUE_PER_CONV = 40.0
CUSTOMER_TYPES = ["New", "Returning"]
SALES_CHANNELS = ["Ecom", "Stores"]


def _plain(obj):
    """Recursively strip numpy types so PyYAML emits plain scalars."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj


def write_scenario_configs(scenario: Scenario, dest: Path) -> tuple[Path, Path]:
    """Write config.yaml and events_config.yaml into `dest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    config = _plain({
        "years": scenario.years,
        "start_date": START_DATE,
        "revenue_per_conv": REVENUE_PER_CONV,
        "customer_types": CUSTOMER_TYPES,
        "sales_channels": SALES_CHANNELS,
        "baseline": scenario.baseline,
        "campaign_spend": scenario.campaign_spend,
        "countries": list(scenario.countries),
        "channels": list(scenario.channels),
    })

    cfg_path = dest / "config.yaml"
    ev_path = dest / "events_config.yaml"

    with cfg_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    with ev_path.open("w") as f:
        yaml.safe_dump(_plain(list(scenario.events)), f, sort_keys=False,
                       default_flow_style=False)

    return cfg_path, ev_path
