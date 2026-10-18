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


def test_inspect_tabs_render_for_every_sample_finding():
    app = run_app()
    for rank in SAMPLE_IDS:
        for tab in ("Readout", "Why flagged", "Caveats", "Explanation"):
            app.session_state["inspect_tab"] = tab
            inspect_finding(app, rank)
            assert not app.exception, (rank, tab)
            assert inspected(app).startswith(f"#{rank} ·")
            # The map is always there; the finding's own charts only on Readout.
            charts = len(app.get("plotly_chart"))
            expected = 1 + (1 + (rank == 5) if tab == "Readout" else 0)
            assert charts == expected, (rank, tab, charts)


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
import app
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
import app
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
