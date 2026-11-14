"""Streamlit app renders both views and an item without errors (findings 15, 16)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "outputs" / "app_products.parquet").exists(), reason="no outputs")


def test_app_item_and_portfolio():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    at.selectbox[0].select_index(0).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Recommended price" in labels and "Confidence" in labels
    at.number_input[1].set_value(100).run()          # stock cap goes through the parameter, not module state
    assert not at.exception
    at.radio[0].set_value("All recommendations").run()
    assert not at.exception and len(at.dataframe) == 1
