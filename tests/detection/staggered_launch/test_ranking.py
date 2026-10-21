from copy import deepcopy
from dataclasses import replace
import json

import pandas as pd
import pytest

from detection.compose.staggered_launch import rank_staggered_launches, summarize_staggered_launches
from detection.model import DetectedEvent


def launch(country="AT", status="analysis_ready", info=0.6, confidence=0.9):
    return DetectedEvent(
        sid="example", country_code=country, channel="TV",
        event_type="staggered_launch", start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-03-10"), informativeness=info,
        detection_confidence=confidence,
        evidence={"onset": "2024-03-11", "launch_verification": {
            "status": status, "reason_codes": [], "qualified_peers": ["CH"],
            "windows": {"pre_start": "2024-01-15", "pre_end": "2024-03-10",
                        "post_start": "2024-03-11", "post_end": "2024-04-07"},
        }},
    )


def test_readiness_precedes_scores_and_retains_every_launch_without_mutation():
    ready = launch(info=0.1, confidence=0.1)
    review = launch("CH", "needs_review", info=0.2)
    rejected = launch("DE", "not_analysis_ready", info=1.0, confidence=1.0)
    unassessed = replace(launch("FR", info=1.0), evidence={})
    holdout = replace(launch("NL"), event_type="natural_holdout")
    events = [rejected, unassessed, holdout, review, ready]
    before = deepcopy(events)

    ranked = rank_staggered_launches(events)
    summarize_staggered_launches(events)

    assert ranked == [ready, review, rejected, unassessed]
    assert events == before
    assert {id(e) for e in ranked} == {id(e) for e in events if e.event_type == "staggered_launch"}


def test_scores_then_identifiers_break_ties_deterministically():
    higher_info = launch("DE", info=0.8, confidence=0.1)
    higher_confidence = launch("FR", info=0.6, confidence=1.0)
    first_tie, second_tie = launch("AT"), launch("CH")
    events = [second_tie, higher_confidence, higher_info, first_tie]
    expected = [higher_info, higher_confidence, first_tie, second_tie]
    assert rank_staggered_launches(events) == expected
    assert rank_staggered_launches(reversed(events)) == expected


@pytest.mark.parametrize("field,value", [
    ("informativeness", None), ("informativeness", float("nan")),
    ("informativeness", float("inf")), ("informativeness", float("-inf")),
    ("detection_confidence", None), ("detection_confidence", float("nan")),
    ("detection_confidence", float("inf")),
])
def test_missing_or_nonfinite_scores_sort_last_at_their_sort_position(field, value):
    scored = launch(info=0.0, confidence=0.0)
    missing = replace(scored, country_code="AA", **{field: value})
    assert rank_staggered_launches([missing, scored]) == [scored, missing]
    record = summarize_staggered_launches([missing])[0]
    assert record[field] is None
    json.dumps(record, allow_nan=False)
    if value is None:
        assert getattr(missing, field) is None
    else:
        assert getattr(missing, field) is value


@pytest.mark.parametrize("evidence", [{}, {"launch_verification": None},
                                      {"launch_verification": {"status": "future_status"}}])
def test_missing_or_unrecognized_assessment_is_unassessed(evidence):
    event = replace(launch(), evidence=evidence)
    row = summarize_staggered_launches([event])[0]
    assert row["readiness"] == "unassessed"
    assert row["launch_date"] is None
    assert row["qualified_peers"] == []
    assert all(value is None for value in row["windows"].values())


def test_summary_uses_actual_onset_and_returns_independent_json_records():
    event = launch()
    row = summarize_staggered_launches([event])[0]
    assert row == {
        "rank": 1, "sid": "example", "country_code": "AT", "channel": "TV",
        "launch_date": "2024-03-11", "inactive_start": "2024-01-01",
        "inactive_end": "2024-03-10", "readiness": "analysis_ready",
        "reason_codes": [], "qualified_peers": ["CH"],
        "windows": {"pre_start": "2024-01-15", "pre_end": "2024-03-10",
                    "post_start": "2024-03-11", "post_end": "2024-04-07"},
        "detection_confidence": 0.9, "informativeness": 0.6,
    }
    assert json.loads(json.dumps(row, allow_nan=False)) == row
    row["qualified_peers"].append("NL")
    row["reason_codes"].append("changed")
    row["windows"]["pre_start"] = None
    original = event.evidence["launch_verification"]
    assert original["qualified_peers"] == ["CH"]
    assert original["reason_codes"] == []
    assert original["windows"]["pre_start"] == "2024-01-15"


def test_empty_mixed_and_generator_inputs():
    event = launch()
    holdout = replace(event, event_type="natural_holdout")
    assert rank_staggered_launches(iter([holdout, event])) == [event]
    assert summarize_staggered_launches(iter([holdout, event]))[0]["rank"] == 1
    assert rank_staggered_launches([]) == []
    assert summarize_staggered_launches([holdout]) == []


def test_rejection_reasons_and_failed_launch_are_exported():
    event = launch(status="not_analysis_ready")
    event.evidence["launch_verification"].update(
        reason_codes=["no_qualified_comparison"], qualified_peers=[])
    row = summarize_staggered_launches([event])[0]
    assert row["readiness"] == "not_analysis_ready"
    assert row["reason_codes"] == ["no_qualified_comparison"]
    assert row["qualified_peers"] == []
