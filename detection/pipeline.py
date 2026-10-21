"""End to end: two CSV frames in, labelled events out.

Detection runs on SPEND ONLY. Sales is accepted and carried on the panel for a
later scoring layer, but it never participates in finding an event: it is
noisier by an order of magnitude, carries seasonality and promotions, and
coupling the two makes every failure harder to explain. That claim is asserted
rather than asserted-about in tests/detection/test_pipeline.py, which runs the
whole pipeline with and without the sales frame and requires identical EVENT
IDENTITIES -- which events, of what type, over what boundaries, with what
confidence and validity. It does NOT require identical output: informativeness
is permitted to differ, because its sales_snr driver (detection/scoring/score.py)
reads sales on purpose, to judge how readable a market's response would be,
not to help find the event in the first place. Scoring an already-found event
is not the same thing as participating in finding it, and the test draws that
line explicitly rather than asserting full-object equality, which this driver
would fail by design.

`annotate` is a strictly 1:1 pass -- it adds tags and evidence and changes
nothing else -- and it ignores staggered launches outright, since a launch is
already the product of a cross-market comparison. Swapping the two lines below
was therefore measured to change nothing at all on any scenario; they are in
this order for reading, not for correctness, and no test pins the order because
none can.

Every event leaving here names a market. A panel-level event would match no
truth interval at all, so the fan-out to one event per launching market happens
one layer down, in compose.cross_market.
"""
from __future__ import annotations

import pandas as pd

from detection.compose.staggered_launch.detector import find_staggered_launches
from detection.compose.label import label_market
from detection.io.panel import build_panel
from detection.model import DetectedEvent
from detection.scoring import score_findings

# The historical calibration predates the current detector. Scores are raw
# heuristics until an independent calibration set supports a new mapping.


def run_detection(media_df: pd.DataFrame, sales_df: pd.DataFrame | None = None,
                  sid: str = "") -> list[DetectedEvent]:
    # An empty-but-present media frame is a real shape (a filtered export, a
    # market with no bookings yet). Without this it reached pd.date_range and
    # surfaced as "Neither start nor end can be NaT", which tells the caller
    # nothing. No rows means nothing to detect.
    if media_df is None or media_df.empty:
        return []

    panel = build_panel(media_df, sales_df, sid)

    events: list[DetectedEvent] = []
    for country in panel.countries:
        events.extend(label_market(panel, country, sid))

    events.extend(find_staggered_launches(panel, sid))
    return score_findings(events, panel)
