"""Pipeline building blocks on small hand-made data (no dataset needed)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pricing_core as core
import pricing_profit_optimizer as pp

ROOT = Path(__file__).resolve().parent.parent


def lines(rows):
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                                       "UnitPrice", "CustomerID", "Country"]) \
        .assign(InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"]))


# Finding 10: cancellations are netted against earlier orders of the same customer
def test_cancellations_are_netted():
    raw = lines([
        ["1", "10001", "A", 10, "2011-01-03", 1.0, 1.0, "UK"],
        ["2", "10001", "A", 5, "2011-01-04", 1.0, 2.0, "UK"],
        ["C3", "10001", "A", -4, "2011-01-05", 1.0, 1.0, "UK"],    # customer 1 cancels 4
        ["C4", "10001", "A", -3, "2011-01-01", 1.0, 2.0, "UK"],    # before customer 2's order: unmatched
        ["5", "10001", "A", 6, "2011-01-06", 1.0, 1.0, "UK"],      # later order is untouched
    ])
    out, stats = pp.net_cancellations(raw)
    q = out.sort_values("InvoiceDate")["Quantity"].tolist()
    assert q == [6, 5, 6]
    assert stats["cancel_units_matched"] == 4 and stats["cancel_units_unmatched"] == 3


# Finding 11: dropping guest lines is a setting and is measured
def test_guest_lines_setting_and_disclosure():
    raw = lines([["1", "10001", "A", 10, "2011-01-03", 1.0, 1.0, "UK"],
                 ["2", "10001", "A", 10, "2011-01-03", 1.0, None, "UK"]])
    d0, s0 = pp.clean(raw, include_guests=False)
    d1, s1 = pp.clean(raw, include_guests=True)
    assert len(d0) == 1 and len(d1) == 2 and s0["guest_line_share"] == pytest.approx(0.5)


# Finding 13: whole-word categories
@pytest.mark.parametrize("desc,bad", [("PARTY BUNTING", "storage"), ("OPEN CLOSED METAL SIGN", "stationery_paper"),
                                      ("METAL 4 HOOK HANGER FRENCH CHATEAU", "kitchen_baking"),
                                      ("SET OF 6 SOLDIER SKITTLES", None), ("PAINTING ON CANVAS", "storage")])
def test_categories_match_whole_words(desc, bad):
    assert pp.categorise(desc) != bad if bad else pp.categorise(desc) == "toys_gifts"


def test_categories_still_match_plurals():
    assert pp.categorise("SET OF 3 CAKE TINS PANTRY DESIGN") == "kitchen_baking"
    assert pp.categorise("ASSORTED BOXES") == "storage"


def test_variant_key_groups_colours():
    assert pp.variant_key("PINK REGENCY TEACUP AND SAUCER") == pp.variant_key("GREEN REGENCY TEACUP AND SAUCER")
    assert pp.variant_key("JUMBO BAG RED RETROSPOT") != pp.variant_key("LUNCH BAG RED RETROSPOT")


# Finding 8: price levels are counted in WEEKS
def test_price_levels_count_weeks_not_days():
    dates = pd.date_range("2011-01-03", periods=70, freq="D")
    price = np.where((dates >= "2011-01-10") & (dates <= "2011-01-12"), 1.25, 1.00)   # 3 days at 1.25, one week
    g = pd.DataFrame({"Date": dates, "list_price": price, "sold": True, "x": np.log(price)})
    g["week"] = g["Date"].dt.to_period("W-SUN")
    assert pp.price_evidence(g)["n_price_levels"] == 1
    price2 = np.where(dates >= "2011-02-07", 1.25, 1.00)                              # 4 weeks at 1.25
    g2 = g.assign(list_price=price2, x=np.log(price2))
    assert pp.price_evidence(g2)["n_price_levels"] == 2


def test_every_setting_is_used():
    src = (ROOT / "src" / "pricing_profit_optimizer.py").read_text(encoding="utf-8")
    for name in re.findall(r"^([A-Z][A-Z_0-9]+)\s*(?::[^=]+)?=", src, re.M):
        assert len(re.findall(rf"\b{name}\b", src)) >= 2, f"setting {name} is defined but never used"


# Finding 9: one list-price definition, bulk lines excluded
def test_list_price_ignores_bulk_lines():
    df = pd.DataFrame({"StockCode": "10001", "Date": pd.to_datetime(["2011-01-03"] * 3),
                       "Quantity": [6, 6, 600], "UnitPrice": [2.10, 2.10, 1.65]})
    lp = pp.list_price_daily(df, ["10001"], pd.Timestamp("2011-12-31"))
    assert lp.iloc[0] == pytest.approx(2.10)


# Finding 7: PPML on the full calendar recovers a known elasticity (zeros included)
def _sim_panel(e_true, n_items=30, n_days=240, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    season = rng.normal(0, 0.3, n_days)
    for i in range(n_items):
        base = rng.uniform(0.2, 3)
        change = rng.integers(60, 180)
        p = np.where(np.arange(n_days) < change, 1.0, rng.choice([0.85, 1.15]))
        y = rng.poisson(base * np.exp(season) * p ** e_true)
        rows.append(pd.DataFrame({"item": i, "date": np.arange(n_days), "x": np.log(p), "y": y}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("e_true", [-0.5, -1.5, -2.5])
def test_ppml_recovers_elasticity(e_true):
    d = _sim_panel(e_true)
    b, se, _ = pp.ppml_fe(d["item"].to_numpy(), d["date"].to_numpy(), d["x"].to_numpy(), d["y"].to_numpy(float),
                          d["item"].nunique(), d["date"].nunique())
    assert abs(b - e_true) < 3 * se + 0.05 and se < 0.3
    assert (d["y"] == 0).mean() > 0.05      # the test really includes zero-sale days


def test_ppml_item_slope():
    d = _sim_panel(-1.5, n_items=1, n_days=400, seed=3)
    b, se = pp.ppml_item(d["y"].to_numpy(float), d["x"].to_numpy(), np.zeros(len(d)))
    assert abs(b + 1.5) < 3 * se + 0.1


# Findings 7 + 14: pooling never yields a positive elasticity; thin categories use the global prior
def test_pooling_rules():
    codes = [f"P{i}" for i in range(14)]
    cats = pd.Series(["big"] * 12 + ["thin"] * 2, index=codes)
    rng = np.random.default_rng(0)
    e_raw = np.r_[rng.normal(-1.2, 0.3, 11), 2.5, np.nan, np.nan]          # one wrong-signed item
    el = pd.DataFrame({"e_raw": e_raw, "se_raw": np.r_[np.full(12, 0.4), np.nan, np.nan]}, index=codes)
    cat_fit = {"big": (-1.2, 0.1, 12), "thin": (3.0, 0.2, 1)}                 # thin slope must be ignored
    out, tbl, _ = pp.pool_elasticities(el, cats, cat_fit)
    assert (out["elasticity"] <= core.ELASTICITY_CEILING + 1e-12).all()
    assert out.loc["P11", "elasticity_source"] == core.SRC_CONTRA
    assert "global prior" in tbl.loc["thin", "prior_basis"]
    assert out.loc["P13", "elasticity"] < 0


# Leakage guard: features at date t never use units from after t - FORECAST_LEAD
def test_features_have_no_leakage():
    dates = pd.date_range("2011-01-03", periods=120, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"StockCode": np.repeat(["10001", "10002"], len(dates)), "Date": np.tile(dates, 2),
                       "Quantity": rng.poisson(5, 2 * len(dates)) + 1, "UnitPrice": 1.0})
    cats = pd.Series({"10001": "other", "10002": "other"})
    train_end = dates[60]
    base = pp.build_panel(df, ["10001", "10002"], cats, train_end)
    shock_day = dates[90]
    df2 = df.copy()
    df2.loc[(df2["Date"] == shock_day) & (df2["StockCode"] == "10001"), "Quantity"] += 500
    shocked = pp.build_panel(df2, ["10001", "10002"], cats, train_end)
    safe = base["Date"] < shock_day + pd.Timedelta(days=pp.FORECAST_LEAD)
    feats = [f for f in pp.FEATURES if f not in pp.STATIC_FEATS]
    pd.testing.assert_frame_equal(base.loc[safe, feats].reset_index(drop=True),
                                  shocked.loc[safe, feats].reset_index(drop=True))


def test_weekly_totals_drop_partial_weeks():
    f = pd.DataFrame({"StockCode": "A", "Date": pd.date_range("2011-11-01", periods=10), "u_cap": 1.0})
    w = pp.weekly_totals(f, np.ones(10), pd.Timestamp("2011-11-01"))
    assert len(w) == 1 and w["u_cap"].iloc[0] == 7
