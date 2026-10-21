from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app(detector: str | None = "rule_based") -> AppTest:
    # Most tests pin our library, whose findings on the sample they assert;
    # pass None to get the app's own default.
    app = AppTest.from_file(APP, default_timeout=60)
    if detector:
        app.session_state["detector"] = detector
    return app.run()


def inspected(app: AppTest) -> str:
    return app.subheader[-1].value


SAMPLE_IDS = {  # rank -> event id on the included sample
    1: "DE|ALL|dark_period|2024-06-24",
    2: "FI|Google Search|single_channel|2024-09-07",
    3: "AT|Facebook|natural_holdout|2025-02-04",
    4: "CH|Google Search|step_change|2025-05-16",
    5: "DE|Radio|channel_pulse|2024-02-10",
    6: "US|Instagram|staggered_launch|2024-01-01",
}


def inspect_finding(app: AppTest, rank: int) -> AppTest:
    # AppTest cannot click shadcn buttons; selecting through session state
    # exercises the same render path the card's Inspect button leads to.
    app.session_state["selected"] = SAMPLE_IDS[rank]
    return app.run()


def test_sample_loads_ranks_and_inspects_the_top_finding():
    app = run_app()
    assert not app.exception
    assert len(app.dataframe[0].value) == 6
    assert list(app.dataframe[0].value["Rank"]) == [1, 2, 3, 4, 5, 6]
    assert inspected(app).startswith("#1 · DE · All advertising off")


def test_selecting_a_finding_switches_the_inspect_panel():
    app = inspect_finding(run_app(), 2)
    assert not app.exception
    assert inspected(app).startswith("#2 · FI · Only Google Search on")


def test_market_filter_narrows_and_hidden_selection_falls_back():
    app = inspect_finding(run_app(), 2)
    app.sidebar.multiselect[0].set_value(["AT"]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    assert inspected(app).startswith("#3 · AT · Facebook paused")


def test_pattern_filter_uses_plain_names():
    app = run_app()
    kinds = app.sidebar.multiselect[1]
    assert "All advertising off" in kinds.options
    kinds.set_value(["step_change"]).run()
    assert not app.exception
    assert inspected(app).startswith("#4 · CH · Google Search budget step")


def test_empty_filters_show_a_message():
    app = run_app()
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Nothing matches these filters" in m.value for m in app.info)


def test_upload_mode_waits_for_media():
    app = run_app()
    app.sidebar.radio[0].set_value("Upload CSV files").run()
    assert not app.exception
    assert any("Upload a media CSV" in m.value for m in app.info)


def test_media_without_sales_renders_spend_only():
    script = f"""
import sys
sys.path.insert(0, {str(Path(APP).parent)!r})
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
media, _ = app.load_sample()
app.load_sample = lambda: (media, None)
app.main()
"""
    app = AppTest.from_string(script, default_timeout=60).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 6
    assert set(app.dataframe[0].value["Sales change"]) == {"No sales data"}


def _single_market_readouts():
    from dataclasses import replace  # noqa: F401  (used by callers)
    import pandas as pd
    from detection.io.panel import build_panel
    from detection.pipeline import run_detection
    from ui.readout import readout
    root = Path(APP).parent / "synthetic_data_generator/data"
    media, sales = pd.read_csv(root / "media.csv"), pd.read_csv(root / "sales.csv")
    media, sales = media[media.country_code == "DE"], sales[sales.country_code == "DE"]
    events = run_detection(media, sales)
    panel = build_panel(media, sales)
    return {e.event_type: (e, readout(e, panel, events)) for e in events}


def test_copy_never_claims_comparison_markets_that_do_not_exist():
    import app as viewer
    e, r = _single_market_readouts()["channel_pulse"]
    assert r.comparison_markets == () and r.effect is not None
    assert viewer.sales_column(r) == f"{r.effect:+.1%}"
    assert viewer.comparison_label(r) == "Sales change"
    assert any("No other market" in line for line in viewer.caveats(e, r))


def test_copy_when_history_is_too_short_to_judge_swings():
    from dataclasses import replace
    import app as viewer
    e, r = _single_market_readouts()["channel_pulse"]
    unknown = replace(r, swing=None, readable=False)
    assert viewer.sales_column(unknown) == f"{unknown.effect:+.1%}"


def test_baseline_help_only_claims_a_share_when_it_is_readable_and_negative():
    from dataclasses import replace
    import app as viewer
    e, r = _single_market_readouts()["dark_period"]
    assert "38%" in viewer.baseline_help(replace(r, effect=-0.38, readable=True))
    assert "came with advertising" not in viewer.baseline_help(replace(r, effect=0.05, readable=True))
    assert "came with advertising" not in viewer.baseline_help(replace(r, effect=-0.38, readable=False))


def test_table_starts_fresh_when_its_rows_change():
    # A keyed st.dataframe keeps its row ticks across reruns; a new key is the
    # only way to stop a stale tick pointing at a different finding.
    app = run_app()
    first = app.dataframe[0].key
    app.sidebar.multiselect[0].set_value(["AT", "FI"]).run()
    assert app.dataframe[0].key != first


def test_selecting_outside_the_table_renews_its_key():
    script = f"""
import sys
sys.path.insert(0, {str(Path(APP).parent)!r})
import streamlit as st
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
st.session_state["table_nonce"] = 0
app.select("a")
st.write(st.session_state["table_nonce"])
app.select("b", from_table=True)
st.write(st.session_state["table_nonce"])
"""
    app = AppTest.from_string(script, default_timeout=30).run()
    assert [m.value.strip("`") for m in app.markdown] == ["1", "1"]


def _analyzed(markets=None, with_sales=True):
    import pandas as pd
    from detection.io.panel import build_panel
    from detection.pipeline import run_detection
    from ui.readout import readout
    root = Path(APP).parent / "synthetic_data_generator/data"
    media, sales = pd.read_csv(root / "media.csv"), pd.read_csv(root / "sales.csv")
    if markets:
        media, sales = media[media.country_code.isin(markets)], sales[sales.country_code.isin(markets)]
    sales = sales if with_sales else None
    events = run_detection(media, sales)
    panel = build_panel(media, sales)
    return events, panel, {e.event_type: (e, readout(e, panel, events)) for e in events}


def test_kpi_tiles_labels_by_finding_type():
    import app as viewer
    events, panel, by_type = _analyzed()
    base = ["Spend/day", "Sales/day", "Vs other markets", "Return"]
    labels = {k: [t["label"] for t in viewer.kpi_tiles(e, r, panel)] for k, (e, r) in by_type.items()}
    assert labels["dark_period"] == base + ["Sales with ads off"]
    assert labels["step_change"] == base + ["Spend change"]
    assert labels["single_channel"] == base + ["Channel still on/day"]
    assert all(set(t) == {"label", "value", "description"}
               for e, r in by_type.values() for t in viewer.kpi_tiles(e, r, panel))
    _, bare_panel, bare = _analyzed(with_sales=False)
    e, r = bare["natural_holdout"]
    assert [t["label"] for t in viewer.kpi_tiles(e, r, bare_panel)] == ["Spend/day"]


def test_kpi_tiles_empty_without_readout():
    import app as viewer
    _, panel, by_type = _analyzed()
    e, _ = by_type["dark_period"]
    assert viewer.kpi_tiles(e, None, panel) == []


# --- shadcn props: AppTest exposes each component's kind and props as JSON. ---

def shadcn(app: AppTest, kind: str | None = None) -> list[dict]:
    import json
    comps = [json.loads(c.proto.json) for c in app.get("bidi_component") if c.proto.json]
    return [c for c in comps if kind is None or c["kind"] == kind]


def element_nodes(app: AppTest) -> list[dict]:
    def walk(nodes):
        for n in nodes:
            yield n
            yield from walk(n.get("children", []))
    return [n for c in shadcn(app, "elements") for n in walk(c["props"]["nodes"])]


ROOT = str(Path(APP).parent)
DE_ONLY = """
m, s = app.load_sample()
app.load_sample = lambda: (m[m.country_code == "DE"], s[s.country_code == "DE"])
"""


def run_patched(patch: str) -> AppTest:
    script = (f"import sys\nsys.path.insert(0, {ROOT!r})\nimport importlib, app\n"
              f"app = importlib.reload(app)\n{patch}\napp.main()\n")
    app = AppTest.from_string(script, default_timeout=60)
    app.session_state["detector"] = "rule_based"
    return app.run()


def test_panel_surfaces_data_warnings():
    app = run_patched(DE_ONLY)
    assert not app.exception
    alerts = [c["props"] for c in shadcn(app, "alert")]
    assert any(a["title"] == "Data quality" and a["variant"] == "destructive" for a in alerts)


def test_why_flagged_is_two_short_lines():
    app = run_app()
    app.session_state["eid"] = SAMPLE_IDS[1]
    script = f"""
import sys
sys.path.insert(0, {ROOT!r})
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
events, panel, reads = app.analyze(*app.load_sample())
app.why_flagged(next(e for e in events if e.event_type == "dark_period"))
"""
    flagged = AppTest.from_string(script, default_timeout=60).run()
    assert not flagged.exception
    assert not [n for n in element_nodes(flagged) if n["type"] == "progress"]
    texts = [m.value for m in flagged.markdown]
    assert texts[0] == "**How sure: Strong.** Nothing falls short."
    assert "**How useful: High.** Nothing falls short." in texts
    # Each line's Details button lists the criteria by name, ticked, with no scores.
    assert "✓ How completely spend changed" in texts
    assert "✓ Usefulness of this pattern" in texts
    assert not any(any(ch.isdigit() for ch in t) for t in texts if t.startswith(("✓", "✗")))


def test_score_checks_mark_what_falls_short():
    import app as viewer
    _, _, by_type = _analyzed()
    pulse, _ = by_type["channel_pulse"]
    checks = dict(viewer.score_checks(pulse))
    assert ("Unusual for this channel", False) in checks["How sure"]
    assert ("Long enough", True) in checks["How sure"]


def test_score_summary_names_only_what_falls_short():
    import app as viewer
    _, _, by_type = _analyzed()
    pulse, _ = by_type["channel_pulse"]
    sure, useful = viewer.score_summary(pulse)
    assert sure.startswith("**How sure: Moderate.** Falls short on: ")
    assert "unusual for this channel" in sure
    assert useful.startswith("**How useful: High.**")


def test_rough_return_description_says_when_it_was_withheld():
    from dataclasses import replace
    import app as viewer
    _, panel, by_type = _analyzed()
    e, r = by_type["dark_period"]
    tiles = {t["label"]: t for t in viewer.kpi_tiles(e, replace(r, readable=True, rough_return=None), panel)}
    assert "not shown" in tiles["Return"]["description"]
    tiles = {t["label"]: t for t in viewer.kpi_tiles(e, replace(r, readable=False, rough_return=None), panel)}
    assert "small change" in tiles["Return"]["description"]


def test_kpi_descriptions_keep_channel_names_as_written():
    import app as viewer
    _, panel, by_type = _analyzed()
    e, r = by_type["step_change"]
    assert viewer.kpi_tiles(e, r, panel)[0]["description"].startswith("Google Search:")


def test_table_shows_a_sortable_start_date():
    import datetime
    app = run_app()
    frame = app.dataframe[0].value
    assert list(frame.columns) == ["Rank", "Market", "Pattern", "Channel", "Start", "Days",
                                   "Priority", "Sales change"]
    assert isinstance(frame["Start"].iloc[0], datetime.date)


# --- detectors: the app runs whatever is registered. ---

STUB = '''
import pandas as pd
from detection.model import DetectedEvent
from detectors import registry
from detectors.base import Detector

class StubDetector(Detector):
    key, name = "stub", "Stub detector"
    event_types = frozenset({"dark_period", "natural_holdout"})
    def detect(self, media, sales):
        return [
            DetectedEvent(sid="", country_code="DE", channel=None, event_type="dark_period",
                          start=pd.Timestamp("2024-06-24"), end=pd.Timestamp("2024-08-17")),
            DetectedEvent(sid="", country_code="AT", channel="Facebook",
                          event_type="natural_holdout",
                          start=pd.Timestamp("2025-02-04"), end=pd.Timestamp("2025-03-17")),
        ]

if "stub" not in [d.key for d in registry.available()]:
    registry.register(StubDetector())
'''


@pytest.fixture
def only_ours(monkeypatch):
    """Exactly our detector registered for the test, restored afterwards --
    AppTest scripts run in this process and share the registry."""
    from detectors import registry
    monkeypatch.setattr(registry, "_DETECTORS", {"rule_based": registry.get("rule_based")})


def test_detector_picker_hidden_with_one_detector(only_ours):
    app = run_app()
    assert not app.exception
    assert "Detector" not in [s.label for s in app.sidebar.selectbox]


def test_switching_detector_changes_the_findings(only_ours):
    app = run_patched(STUB)
    assert not app.exception
    picker = next(s for s in app.sidebar.selectbox if s.label == "Detector")
    assert picker.options == ["Rule-based (ours)", "Stub detector"]
    assert len(app.dataframe[0].value) == 6
    picker.set_value("stub").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 2
    assert set(app.dataframe[0].value["Priority"]) == {"Unscored"}
    # The DE dark period exists in both detectors' output, so the selection
    # carries over, exactly as it does through a filter change.
    assert inspected(app) == "#2 · DE · All advertising off"
    kinds = app.sidebar.multiselect[1]
    assert set(kinds.options) == {"All advertising off", "One channel paused"}


def test_switching_detector_drops_a_selection_it_does_not_report(only_ours):
    app = run_patched(STUB)
    app.session_state["selected"] = SAMPLE_IDS[2]       # FI single channel: ours only
    app.run()
    assert inspected(app).startswith("#2 · FI")
    next(s for s in app.sidebar.selectbox if s.label == "Detector").set_value("stub").run()
    assert not app.exception
    assert inspected(app) == "#1 · AT · Facebook paused"


def test_unscored_detector_renders_the_panel(only_ours):
    script = f"""
import sys
sys.path.insert(0, {ROOT!r})
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
{STUB}
import streamlit as st
from detection.io.panel import build_panel
from ui.readout import readout
media, sales = app.load_sample()
events = registry.get("stub").detect(media, sales)
panel = build_panel(media, sales)
e = events[st.session_state["i"]]
app.inspect(e, readout(e, panel, events), 1, panel, False)
"""
    for i in (0, 1):
        app = AppTest.from_string(script, default_timeout=60)
        app.session_state["i"] = i
        app.run()
        assert not app.exception, i
        assert "No scores from this detector" in [c["props"]["title"] for c in shadcn(app, "alert")]
        assert any("This detector gives no explanation." in m.value for m in app.markdown), i



INSPECT = f"""
import sys
sys.path.insert(0, {ROOT!r})
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
import streamlit as st
from ui.labels import event_id
events, panel, reads = app.analyze(*app.load_sample())
e = next(x for x in events if event_id(x) == st.session_state["eid"])
app.inspect(e, reads[event_id(e)], 1, panel, False)
"""


def test_the_page_has_no_summary_cards_or_top_cards():
    app = run_app()
    assert not app.exception
    assert not [c for c in shadcn(app) if c["kind"] in ("metric_card", "elements", "tabs")
                and c["props"].get("label") in ("Findings", "Markets with findings")]
    assert not shadcn(app, "tabs")
    assert not [n for n in element_nodes(app) if n["type"] == "button"]


def test_inspect_shows_readout_chart_and_collapsed_details_for_every_finding():
    for eid in SAMPLE_IDS.values():
        app = AppTest.from_string(INSPECT, default_timeout=60)
        app.session_state["eid"] = eid
        app.run()
        assert not app.exception, eid
        assert shadcn(app, "metric_card"), eid
        assert app.get("plotly_chart"), eid
        assert [x.label for x in app.expander] == ["Details"], eid


def test_caveats_are_short_lines_not_a_tab():
    app = AppTest.from_string(INSPECT, default_timeout=60)
    app.session_state["eid"] = SAMPLE_IDS[6]          # US launch touches the data start
    app.run()
    assert any("first day of the data" in c.value for c in app.caption)



def test_sales_change_cells_show_one_thing():
    from dataclasses import replace
    import app as viewer
    _, _, by_type = _analyzed()
    _, r = by_type["dark_period"]                      # readable, with other markets
    assert viewer.sales_column(r) == f"{r.effect:+.1%}"
    assert viewer.sales_column(replace(r, readable=False)) == f"{r.effect:+.1%}"
    assert "chance" in viewer.SALES_CHANGE_HELP and "unclear" not in viewer.SALES_CHANGE_HELP.lower()



def test_the_app_uses_the_team_detector_by_default():
    app = run_app(detector=None)
    assert not app.exception
    picker = next(s for s in app.sidebar.selectbox if s.label == "Detector")
    assert picker.value == "team"
    assert any("Detector: Combined" in c.value for c in app.caption)
    assert len(app.dataframe[0].value) == 6
    assert inspected(app).startswith("#1 ·")


def test_periods_can_be_sorted_by_sales_change_without_changing_rank():
    app = run_app()
    sort = next(r for r in app.radio if r.label == "Sort by")
    assert sort.value == "Most useful"
    sort.set_value("Biggest sales change").run()
    assert not app.exception
    frame = app.dataframe[0].value
    cells = list(frame["Sales change"])
    known = [c for c in cells if c != "No sales data"]
    assert cells[:len(known)] == known  # periods without a reading go last
    sizes = [abs(float(c.rstrip("%"))) for c in known]
    assert sizes == sorted(sizes, reverse=True)
    assert sorted(frame["Rank"]) == [1, 2, 3, 4, 5, 6]
    assert list(frame["Rank"]) != [1, 2, 3, 4, 5, 6]
    assert any("Biggest sales change first" in c.value for c in app.caption)
    assert inspected(app).startswith("#1 ·")
