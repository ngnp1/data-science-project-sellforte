"""tvh-ds: staggered launch -- a channel starting later in one market than in others.

The detector itself lives in detection/compose/staggered_launch/, because our
rule-based library uses it too; this module is its entry point in the team
package. It finds the late launches and adds the readiness checks
(`launch_verification` in each finding's evidence).
"""
from __future__ import annotations

import pandas as pd

from detection.compose.staggered_launch.detector import find_staggered_launches
from detection.compose.staggered_launch.readiness import verify_staggered_launches
from detection.io.panel import build_panel
from detection.model import DetectedEvent


def run_detection(media_df: pd.DataFrame | None, sales_df: pd.DataFrame | None = None,
                  sid: str = "") -> list[DetectedEvent]:
    if media_df is None or media_df.empty:
        return []
    panel = build_panel(media_df, sales_df, sid)
    launches = find_staggered_launches(panel, sid)
    return verify_staggered_launches(launches, panel)
