"""Decision maths (src/pricing_core.py). Each test names the review finding it guards."""
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

import pricing_core as core

ROOT = Path(__file__).resolve().parent.parent
Z = np.random.default_rng(0).standard_normal(500)


def item(**kw):
    r = {"StockCode": "X", "current_price": 2.95, "unit_cost": 1.77, "hist_min_price": 2.55, "hist_max_price": 3.25,
         "baseline_units": 300.0, "elasticity": -1.5, "elasticity_se": 0.4, "elasticity_source": core.SRC_OWN,
         "daily_log_sd": 0.8}
    r.update(kw)
    return r


def decide(r, **kw):
    o = core.optimise_product(r, Z, **kw)
    b = o["comp_same"]
    conf, why = core.confidence({**r, "recommended_price": b["price"], "downside": b["downside"],
                                 "p_improve": b["p_improve"], "flags": o["flags"]})
    return o, b, conf


# Finding 2: competitor level must not add to the own-price elasticity
@pytest.mark.parametrize("mult", [0.8, 1.0, 1.25])
def test_competitor_term_does_not_change_price_slope(mult):
    e = np.array([-1.3])
    u = core.demand_matrix([2.0, 2.4], 2.0, 100, e, comp_mult=mult)[0]
    assert math.isclose(np.log(u[1] / u[0]) / np.log(1.2), -1.3, rel_tol=1e-12)


def test_competitor_level_changes_profit_not_price():
    _, b, _ = decide(item())
    o = core.optimise_product(item(), Z)
    assert o["comp_minus10"]["price"] == o["comp_plus10"]["price"] == b["price"]
    assert o["comp_minus10"]["profit"] < b["profit"] < o["comp_plus10"]["profit"]


# Finding 3: high confidence needs the item's own price evidence
@pytest.mark.parametrize("src", [core.SRC_PRIOR, core.SRC_CONTRA])
def test_category_only_items_are_never_high(src):
    for e in (-3.0, -1.5, -0.5):
        for se in (0.05, 0.3, 1.0):
            assert decide(item(elasticity=e, elasticity_se=se, elasticity_source=src))[2] != "high"


def test_contradictory_items_are_low():
    assert decide(item(elasticity_source=core.SRC_CONTRA))[2] == "low"


def test_own_evidence_can_be_high():
    assert decide(item(elasticity=-3.0, elasticity_se=0.05))[2] == "high"


# Finding 1: the decision driver and at-bound are always reported
def test_decision_driver_reports_binding_limit():
    o = core.optimise_product(item(elasticity=-0.5, elasticity_se=0.05), Z)
    assert o["at_bound"] and o["decision_driver"].startswith("upper limit")
    o = core.optimise_product(item(elasticity=-3.0, elasticity_se=0.05), Z)
    assert o["decision_driver"].startswith("data") or o["decision_driver"].startswith("lower")


def test_lerner():
    assert core.lerner(-2.0, 1.0, 1.5)["unconstrained_price"] == pytest.approx(2.0)
    assert core.lerner(-0.8, 1.0, 1.5)["unconstrained_price"] == math.inf


# General: recommendations always respect the allowed range
def test_price_within_bounds_random():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p0 = float(rng.uniform(0.3, 20))
        r = item(current_price=round(p0, 2), unit_cost=p0 * rng.uniform(0.3, 0.9),
                 hist_min_price=p0 * rng.uniform(0.7, 1.0), hist_max_price=p0 * rng.uniform(1.0, 1.4),
                 elasticity=float(rng.uniform(-4, -0.2)), elasticity_se=float(rng.uniform(0.05, 2)))
        o = core.optimise_product(r, Z)
        lo, hi = o["bounds"]
        if not o["flags"]:
            assert lo - 1e-9 <= o["comp_same"]["price"] <= hi + 1e-9


# Finding 18: an empty price grid is flagged, never silent
def test_empty_grid_is_flagged():
    r = item(current_price=0.40, hist_min_price=0.40, hist_max_price=0.43, unit_cost=0.33, custom=True)
    o, b, conf = decide(r)
    assert any("price grid" in f for f in o["flags"]) and conf == "low" and b["price"] == 0.40


def test_infeasible_range_is_flagged():
    o, _, conf = decide(item(unit_cost=3.5))
    assert any("no feasible price" in f for f in o["flags"]) and conf == "low"


# Finding 4: cross-fitted gain is reported
def test_crossfit_gain_present_and_bounded():
    b = core.optimise_product(item(elasticity=-0.6), Z)["comp_same"]
    assert "profit_change_crossfit" in b
    assert b["profit_change_crossfit"] <= (b["profit"] - b["profit_current"]) * 1.5 + 1e-9


# Finding 12: profit range includes forecast error
def test_profit_range_widens_with_forecast_error():
    d = np.linspace(90, 110, 500)
    lo1, hi1 = core.profit_range(d, np.ones(500))
    lo2, hi2 = core.profit_range(d, np.random.default_rng(2).permutation(np.linspace(0.5, 1.5, 500)))
    assert lo2 < lo1 and hi2 > hi1


# Feature: test plan
def test_test_plan_scales_with_noise():
    a = core.test_plan(2.95, 2.95, 0.5, 2.36, 3.54)
    b = core.test_plan(2.95, 2.95, 1.0, 2.36, 3.54)
    assert b["test_days_per_arm"] > a["test_days_per_arm"] and a["test_price"] > 2.95


# Finding 15: no shared mutable state in the decision code or the app
def test_no_global_state_in_core_or_app():
    src = inspect.getsource(core)
    assert not re.search(r"^\s*global\s", src, re.M) and "STOCK_CAP" not in src
    assert "stock" in inspect.signature(core.optimise_product).parameters
    app = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(core|pp)\.[A-Z_]+\s*=", app, re.M), "the app must not assign module settings"
