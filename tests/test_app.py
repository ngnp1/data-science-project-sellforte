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
