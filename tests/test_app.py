from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


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
    assert any("No findings match" in m.value for m in app.info)


def test_upload_mode_waits_for_media():
    app = run_app()
    app.sidebar.radio[0].set_value("Upload CSV files").run()
    assert not app.exception
    assert any("Upload media" in m.value for m in app.info)


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
    assert set(app.dataframe[0].value["Sales change"]) == {"No sales reading"}


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
    assert "comparison" not in viewer.headline(e, r).lower()
    assert "no comparison" in viewer.sales_column(r)
    assert viewer.comparison_label(r) == "Sales change"
    assert any("No comparison markets" in line for line in viewer.caveats(e, r))


def test_copy_when_history_is_too_short_to_judge_swings():
    from dataclasses import replace
    import app as viewer
    e, r = _single_market_readouts()["channel_pulse"]
    unknown = replace(r, swing=None, readable=False)
    assert "too little history" in viewer.headline(e, unknown).lower()
    assert "normal swings" not in viewer.headline(e, unknown)
    assert "too little history" in viewer.sales_column(unknown)


def test_baseline_help_only_claims_a_share_when_it_is_readable_and_negative():
    from dataclasses import replace
    import app as viewer
    e, r = _single_market_readouts()["dark_period"]
    assert "38%" in viewer.baseline_help(replace(r, effect=-0.38, readable=True))
    assert "went with advertising" not in viewer.baseline_help(replace(r, effect=0.05, readable=True))
    assert "went with advertising" not in viewer.baseline_help(replace(r, effect=-0.38, readable=False))


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


def test_card_badges_flag_data_warnings():
    from dataclasses import replace
    import app as viewer
    _, _, by_type = _analyzed()
    e, _ = by_type["dark_period"]
    assert viewer.card_badges(e)[:3] == [("All advertising off", "outline"),
                                         ("Priority High", "default"),
                                         ("Evidence Strong", "secondary")]
    assert ("Data warning", "destructive") not in viewer.card_badges(e)
    assert ("Data warning", "destructive") in viewer.card_badges(replace(e, validity="suspect_data_gap"))


def test_summary_cards_on_the_sample():
    import app as viewer
    events, panel, _ = _analyzed()
    assert viewer.summary_cards(events, panel.countries) == [
        ("Findings", "6", "in 5 markets"),
        ("Markets with findings", "5 of 5", "with a finding"),
        ("High priority", "5", "priority score ≥ 0.75"),
        ("With data warnings", "0", "check before using"),
    ]


def test_kpi_tiles_labels_by_finding_type():
    import app as viewer
    events, panel, by_type = _analyzed()
    base = ["Affected spend/day", "Market sales/day", "Vs comparison", "Rough return"]
    labels = {k: [t["label"] for t in viewer.kpi_tiles(e, r, panel)] for k, (e, r) in by_type.items()}
    assert labels["dark_period"] == base + ["Baseline sales/day"]
    assert labels["step_change"] == base + ["Spend multiple"]
    assert labels["single_channel"] == base + ["Running channel/day"]
    assert all(set(t) == {"label", "value", "description"}
               for e, r in by_type.values() for t in viewer.kpi_tiles(e, r, panel))
    _, bare_panel, bare = _analyzed(with_sales=False)
    e, r = bare["natural_holdout"]
    assert [t["label"] for t in viewer.kpi_tiles(e, r, bare_panel)] == ["Affected spend/day"]


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
    return AppTest.from_string(script, default_timeout=60).run()


def test_tab_default_never_moves_so_clicks_are_not_reset():
    # shadcn treats a changed default as a server reset and drops the click
    # the user just made, so the default must stay fixed across reruns.
    app = run_app()
    app.session_state["inspect_tab"] = "Caveats"
    inspect_finding(app, 2)
    tabs = shadcn(app, "tabs")[0]
    assert tabs["state"]["serverRevision"] == 0
    assert tabs["state"]["value"] == tabs["props"]["options"][0]["value"]


def test_inspect_body_renders_each_tab_for_every_finding():
    script = f"""
import sys
sys.path.insert(0, {ROOT!r})
import streamlit as st
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
from ui.labels import event_id
events, panel, reads = app.analyze(*app.load_sample())
e = next(x for x in events if event_id(x) == st.session_state["eid"])
app.inspect_body(st.session_state["tab"], e, reads[event_id(e)], panel, False)
"""
    expected = {"Readout": "metric_card", "Why flagged": "elements",
                "Caveats": "alert", "Explanation": "card"}
    for eid in SAMPLE_IDS.values():
        for tab, kind in expected.items():
            app = AppTest.from_string(script, default_timeout=60)
            app.session_state["eid"], app.session_state["tab"] = eid, tab
            app.run()
            assert not app.exception, (eid, tab)
            assert {c["kind"] for c in shadcn(app)} == {kind}, (eid, tab)
            assert bool(app.get("plotly_chart")) == (tab == "Readout"), (eid, tab)


def test_cards_mark_the_selected_finding():
    app = inspect_finding(run_app(), 2)
    buttons = {n["id"].split("/")[-1]: n["props"] for n in element_nodes(app) if n["type"] == "button"}
    assert buttons["inspect-2"]["text"] == "Inspecting" and buttons["inspect-2"]["variant"] == "default"
    assert buttons["inspect-1"]["text"] == "Inspect" and buttons["inspect-1"]["variant"] == "outline"


def test_cards_and_panel_surface_data_warnings():
    app = run_patched(DE_ONLY)
    assert not app.exception
    badges = [b for n in element_nodes(app) if n["type"] == "badge" for b in n["props"]["badges"]]
    assert {"text": "Data warning", "variant": "destructive"} in badges
    # The data-quality alert sits above the tabs, so it shows on the default tab.
    alerts = [c["props"] for c in shadcn(app, "alert")]
    assert any(a["title"] == "Data quality" and a["variant"] == "destructive" for a in alerts)


def test_why_flagged_draws_one_progress_row_per_score():
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
    rows = [n for n in element_nodes(flagged) if n["type"] == "progress"]
    assert len(rows) == 12 and all(0 <= r["props"]["value"] <= 100 for r in rows)


def test_inspect_handler_survives_shadcn_event_binding():
    script = f"""
import sys
sys.path.insert(0, {ROOT!r})
import streamlit as st
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
from streamlit_shadcn_ui.v2.elements import _invoke_callback
st.session_state["table_nonce"] = 0
_invoke_callback(app.inspect_handler("x"), object())
st.write(st.session_state["table_nonce"], st.session_state["selected"])
"""
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert [m.value.strip("`") for m in app.markdown] == ["1", "x"] or \
        "1" in app.markdown[0].value and "x" in app.markdown[0].value


def test_caption_counts_caveats_and_none_copy_matches_spec():
    app = inspect_finding(run_app(), 6)          # US launch touches the data start
    assert "1 caveat" in app.caption[-1].value or any("1 caveat" in c.value for c in app.caption)
    script = f"""
import sys
sys.path.insert(0, {ROOT!r})
import importlib, app
app = importlib.reload(app)  # earlier tests patch the cached module
events, panel, reads = app.analyze(*app.load_sample())
e = next(x for x in events if x.event_type == "dark_period")
app.caveat_alerts(e, reads[app.event_id(e)])
"""
    none = AppTest.from_string(script, default_timeout=60).run()
    assert [c["props"]["title"] for c in shadcn(none, "alert")] == ["No caveats found for this finding."]


def test_rough_return_description_says_when_it_was_withheld():
    from dataclasses import replace
    import app as viewer
    _, panel, by_type = _analyzed()
    e, r = by_type["dark_period"]
    tiles = {t["label"]: t for t in viewer.kpi_tiles(e, replace(r, readable=True, rough_return=None), panel)}
    assert "withheld" in tiles["Rough return"]["description"]
    tiles = {t["label"]: t for t in viewer.kpi_tiles(e, replace(r, readable=False, rough_return=None), panel)}
    assert "readable" in tiles["Rough return"]["description"]


def test_kpi_descriptions_keep_channel_names_as_written():
    import app as viewer
    _, panel, by_type = _analyzed()
    e, r = by_type["step_change"]
    assert viewer.kpi_tiles(e, r, panel)[0]["description"].startswith("Google Search:")


def test_table_shows_a_sortable_start_date():
    import datetime
    app = run_app()
    frame = app.dataframe[0].value
    assert list(frame.columns) == ["Rank", "Market", "Finding", "Channel", "Start", "Days",
                                   "Priority", "Evidence", "Sales change", "Data quality"]
    assert isinstance(frame["Start"].iloc[0], datetime.date)
