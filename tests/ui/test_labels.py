from detection.model import EVENT_TYPES
from detection.scoring import ranked
from ui.labels import (SCORE_NAMES, TYPES, event_id, evidence_band,
                       priority_band, score_rows, short_label, title)


def test_every_event_type_has_copy_and_colours():
    assert set(TYPES) == set(EVENT_TYPES)
    for info in TYPES.values():
        assert info.name and info.measures
        assert info.color.startswith("#") and info.color_dark.startswith("#")


def test_type_colours_are_distinct():
    assert len({t.color for t in TYPES.values()}) == len(TYPES)
    assert len({t.color_dark for t in TYPES.values()}) == len(TYPES)


def test_band_edges():
    assert [priority_band(x) for x in (0.75, 0.7499, 0.5, 0.49, None)] == [
        "High", "Medium", "Medium", "Low", "Unscored"]
    assert [evidence_band(x) for x in (0.8, 0.6, 0.59, None)] == [
        "Strong", "Moderate", "Weak", "Unscored"]


def test_score_names_cover_pipeline_keys(sample_events):
    for e in sample_events:
        keys = (e.evidence["confidence_sub_scores"].keys()
                | e.evidence["informativeness_drivers"].keys())
        assert keys <= SCORE_NAMES.keys()
    assert all(name and meaning for name, meaning in SCORE_NAMES.values())


def test_titles_name_the_market_and_pattern(sample_events):
    assert {title(e) for e in sample_events} == {
        "DE · All advertising off",
        "FI · Only Google Search on",
        "AT · Facebook paused",
        "CH · Google Search budget step",
        "DE · Radio on–off pulsing",
        "US · Instagram launched late",
    }


def test_ranked_orders_by_informativeness_then_confidence(sample_events):
    top = ranked(sample_events)
    assert title(top[0]) == "DE · All advertising off"
    scores = [(e.informativeness, e.detection_confidence) for e in top]
    assert scores == sorted(scores, reverse=True)


def test_ranked_puts_unscored_last(sample_events):
    from dataclasses import replace
    unscored = replace(sample_events[0], informativeness=None)
    assert ranked([unscored, *sample_events[1:]])[-1] is unscored


def test_event_ids_are_unique_and_stable(sample_events):
    ids = [event_id(e) for e in sample_events]
    assert len(set(ids)) == len(ids)
    dark = next(e for e in sample_events if e.event_type == "dark_period")
    assert event_id(dark) == "DE|ALL|dark_period|2024-06-24"


def test_score_rows_keep_order_and_plain_names():
    rows = score_rows({"contrast": 1.0, "mystery": 0.2})
    assert rows == [(SCORE_NAMES["contrast"][0], SCORE_NAMES["contrast"][1], 1.0),
                    ("mystery", "", 0.2)]


def test_short_labels_say_what_and_how_long(sample_events):
    assert {short_label(e) for e in sample_events} == {
        "All channels off · 55d",
        "Only Google Search · 30d",
        "Facebook paused · 42d",
        "Google Search ×3.1 · 55d",
        "Radio ×3 pauses · 70d",
        "Instagram late · 90d",
    }
