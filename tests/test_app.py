"""Every app page renders without errors, for every flight and every option."""
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["overview", "flight_explorer", "engine_trends", "detector_test", "maintenance_validation"]

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(not (APP / "data" / "demo_flights.parquet").exists(),
                                reason="app bundle not built (scripts/build_app_data.py)")


def _page(tmp_path, fn: str) -> AppTest:
    # run one page function on its own; navigation is stubbed out
    script = tmp_path / f"run_{fn}.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, r'{APP}')\n"
        "import streamlit as st\n"
        "st.navigation = lambda pages: type('P', (), {'run': lambda self: None})()\n"
        "import streamlit_app as app\n"
        f"app.{fn}()\n"
    )
    return AppTest.from_file(str(script), default_timeout=180).run()


@pytest.mark.parametrize("fn", PAGES)
def test_page_renders(tmp_path, fn):
    at = _page(tmp_path, fn)
    assert not at.exception, [e.message for e in at.exception]


def test_every_demo_flight_renders(tmp_path):
    at = _page(tmp_path, "flight_explorer")
    for fid in at.selectbox[0].options:
        at.selectbox[0].set_value(fid).run()
        assert not at.exception, (fid, [e.message for e in at.exception])


@pytest.mark.parametrize("fn", ["engine_trends", "detector_test"])
def test_every_option_renders(tmp_path, fn):
    at = _page(tmp_path, fn)
    for opt in at.radio[0].options:
        at.radio[0].set_value(opt).run()
        assert not at.exception, (opt, [e.message for e in at.exception])
