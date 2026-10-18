from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


def inspected(app: AppTest) -> str:
    return app.subheader[-1].value


def test_sample_loads_ranks_and_inspects_the_top_finding():
    app = run_app()
    assert not app.exception
    assert len(app.dataframe[0].value) == 6
    assert list(app.dataframe[0].value["Rank"]) == [1, 2, 3, 4, 5, 6]
    assert inspected(app).startswith("#1 · DE · All advertising off")
    assert [m.label for m in app.metric][:4] == [
        "Findings", "Markets with findings", "High priority", "With data warnings"]


def test_card_button_switches_the_inspected_finding():
    app = run_app()
    app.button(key="inspect-2").click().run()
    assert not app.exception
    assert inspected(app).startswith("#2 · FI · Only Google Search on")


def test_market_filter_narrows_and_hidden_selection_falls_back():
    app = run_app()
    app.button(key="inspect-2").click().run()
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


def test_every_sample_finding_renders_its_inspect_panel():
    app = run_app()
    for rank in range(1, 4):
        app.button(key=f"inspect-{rank}").click().run()
        assert not app.exception
    app.sidebar.multiselect[0].set_value(["DE", "US"]).run()
    for rank in (5, 6):
        app.button(key=f"inspect-{rank}").click().run()
        assert not app.exception
        assert inspected(app).startswith(f"#{rank} ·")


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
    assert [m.label for m in app.metric][4:] == ["Affected spend/day"]


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


def test_table_starts_fresh_when_the_selection_changes_elsewhere():
    # A keyed st.dataframe keeps its row ticks across reruns; a new key is the
    # only way to stop a stale tick pointing at a different finding.
    app = run_app()
    first = app.dataframe[0].key
    app.button(key="inspect-2").click().run()
    second = app.dataframe[0].key
    assert second != first
    app.sidebar.multiselect[0].set_value(["AT", "FI"]).run()
    assert app.dataframe[0].key not in (first, second)
