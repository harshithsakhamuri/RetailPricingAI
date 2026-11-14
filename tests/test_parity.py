"""The webpage's JS port must give the same decisions as pricing_core.py (review finding 17)."""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "parity_cases.json"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_matches_python():
    expected = {c["name"]: c["expect"] for c in json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]}
    run = subprocess.run(["node", str(ROOT / "tests" / "js" / "parity.js")], capture_output=True, text=True, encoding="utf-8", check=True)
    got = json.loads(run.stdout)
    assert len(got) == len(expected) >= 30
    for g in got:
        e = expected[g["name"]]
        for k, ev in e.items():
            gv = g[k]
            if isinstance(ev, float) and gv is not None:
                assert math.isclose(gv, ev, rel_tol=1e-6, abs_tol=1e-6), (g["name"], k, gv, ev)
            elif isinstance(ev, dict):
                for kk, vv in ev.items():
                    assert gv[kk] == pytest.approx(vv), (g["name"], k, kk)
            else:
                assert gv == ev, (g["name"], k, gv, ev)


def test_fixtures_cover_edge_cases():
    names = {c["name"] for c in json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]}
    for n in ["stock cap", "competitor 10% cheaper", "forced joint price", "margin floor above range",
              "narrow range, no grid price", "custom new item", "very low volume"]:
        assert n in names
