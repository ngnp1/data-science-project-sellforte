from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_sample_viewer_and_filters():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 6
    app.sidebar.multiselect[0].set_value(["DE"]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 2
    app.selectbox[0].set_value(1).run()
    assert not app.exception
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("No periods match" in message.value for message in app.info)


def test_upload_mode_waits_for_media():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Upload CSV files").run()
    assert not app.exception
    assert any("Upload media" in message.value for message in app.info)
