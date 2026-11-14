"""Invariants on the committed pipeline outputs, so a re-run that brings back a
reviewed problem fails CI. Each test names the review finding it guards."""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import pricing_core as core

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
pytestmark = pytest.mark.skipif(not (OUT / "app_products.parquet").exists(), reason="pipeline outputs not present")


@pytest.fixture(scope="module")
def rec():
    return pd.read_csv(OUT / "pricing_recommendations_all_products.csv", dtype={"StockCode": str})


@pytest.fixture(scope="module")
def summary():
    return (OUT / "summary.md").read_text(encoding="utf-8")


def test_no_positive_or_flat_elasticity(rec):                       # finding 7
    assert (rec["elasticity"] <= core.ELASTICITY_CEILING + 1e-9).all()


def test_confidence_rules_hold(rec):                                 # findings 3, 7
    assert not ((rec["elasticity_source"] != core.SRC_OWN) & (rec["confidence"] == "high")).any()
    assert (rec.loc[rec["elasticity_source"] == core.SRC_CONTRA, "confidence"] == "low").all()


def test_diagnostics_present(rec):                                   # finding 1
    for c in ["decision_driver", "at_bound", "lerner_margin", "unconstrained_price", "lerner_gap"]:
        assert c in rec.columns
    assert rec["decision_driver"].notna().all()


def test_prices_within_bounds(rec):
    ok = rec["flags"].isna()
    r = rec[ok]
    assert ((r["recommended_price"] >= r["bound_lo"] - 1e-6) & (r["recommended_price"] <= r["bound_hi"] + 1e-6)).all()


def test_single_price_definition(rec):                               # finding 9
    assert {"hist_min_price", "hist_max_price"} <= set(rec.columns)
    assert not any(c.startswith("train_") for c in rec.columns)
    assert (rec["hist_min_price"] <= rec["hist_max_price"] + 1e-9).all()


def test_headline_gain_split_and_crossfit(summary, rec):             # finding 4
    assert "profit_change_crossfit" in rec.columns
    assert "high + medium" in summary and "cross-fitted" in summary.lower()


def test_forecast_selection_includes_naive():                         # finding 5
    m = pd.read_csv(OUT / "model_metrics.csv", index_col=0)
    assert {"Naive: same day last week", "Naive: 28-day mean rate"} <= set(m.index)
    assert m["calibration"].notna().all()
    assert m.loc[m["selected"], "select_WAPE%_7day"].iloc[0] == m["select_WAPE%_7day"].min()


def test_production_calibration_is_measured_on_refit():              # finding 6
    meta = json.loads((OUT / "app_meta.json").read_text(encoding="utf-8"))
    assert meta["calibration_source"] == "refit-to-test check"


def test_summary_discloses_data_handling(summary):                   # findings 10, 11, 19
    assert "netted" in summary and "CustomerID" in summary
    assert "seasonality control" not in summary and "weekly median price" not in summary
    assert "Placebo test" in summary and "backtest" in summary.lower()


def test_profit_range_includes_forecast_error(summary):              # finding 12
    assert "baseline-forecast error" in summary


def test_backtest_and_placebo_files():                               # finding 7, feature 3
    assert (OUT / "backtest_price_changes.csv").exists() and (OUT / "elasticity_placebo.csv").exists()


def test_test_plan_written(rec):                                     # feature 2
    plan = pd.read_csv(OUT / "price_test_plan.csv", dtype={"StockCode": str})
    assert len(plan) and {"test_price", "test_weeks_total", "test_weeks_pooled_category"} <= set(plan.columns)


def test_webpage_data_matches_outputs(rec):                          # findings 16, 17
    text = (ROOT / "docs" / "data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rstrip().rindex("}") + 1])
    assert len(data["items"]) == len(rec)
    web = {i["StockCode"]: i["recommended_price"] for i in data["items"]}
    assert all(abs(web[c] - p) < 1e-6 for c, p in zip(rec["StockCode"], rec["recommended_price"]))


def test_portable_artifacts_and_pinned_requirements():                # finding 16
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout
    assert "app_artifacts.pkl" not in tracked
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            assert re.match(r"^[A-Za-z0-9_.\-\[\]]+==[0-9]", line), f"unpinned requirement: {line}"


def test_no_blanket_warning_suppression():                            # finding 19
    for f in (ROOT / "src").glob("*.py"):
        assert 'filterwarnings("ignore")' not in f.read_text(encoding="utf-8"), f


def test_elasticity_level_is_calibrated_to_real_price_changes(summary):
    meta = json.loads((OUT / "app_meta.json").read_text(encoding="utf-8"))
    lvl = meta["elasticity_level"]
    assert lvl and lvl["n_events"] >= 10 and meta["elasticity_level_scale"] >= 1.0
    assert "Level calibration" in summary
    rec = pd.read_csv(OUT / "pricing_recommendations_all_products.csv")
    assert abs(rec["elasticity"].median() - lvl["elasticity"]) < 2 * lvl["se"]


def test_no_suggestion_expected_to_earn_less(rec):
    """A suggested price must never have lower expected profit than today's price
    (it confused users when a 'raise' came with a profit fall)."""
    single = rec["joint_with"] == 0          # joint prices optimise the group total, not each item
    r = rec[single & rec["flags"].isna()]
    assert (r["expected_profit"] >= r["profit_at_current"] - 1e-6).all()


def test_elasticity_within_plausible_range(rec):
    assert (rec["elasticity"] >= core.ELASTICITY_FLOOR - 1e-9).all()


def test_no_negative_extra_profit(rec):
    assert (rec["profit_change_crossfit"] >= -1e-9).all()
    kept = rec["recommended_price"] == rec["current_price"]
    assert (rec.loc[kept, "profit_change_crossfit"].abs() < 1e-9).all()


def test_daily_forecast_month_ahead(rec):
    d = pd.read_csv(OUT / "daily_forecast.csv", dtype={"StockCode": str}, parse_dates=["Date"])
    meta = json.loads((OUT / "app_meta.json").read_text(encoding="utf-8"))
    assert d["Date"].nunique() == 30 and (d["baseline_units"] >= 0).all()
    closed = {pd.Timestamp(c["date"]) for c in meta["calendar"] if not c["open"]}
    assert (d.loc[d["Date"].isin(closed), "baseline_units"] == 0).all()
    week = d[d["Date"] <= d["Date"].min() + pd.Timedelta(days=6)].groupby("StockCode")["baseline_units"].sum()
    base = rec.set_index("StockCode")["baseline_units"]
    assert (abs(week.reindex(base.index) - base) < 1e-2 * base.clip(lower=1)).all()
    assert meta["holidays"]["calendar_wrong_open"] == [] and meta["holidays"]["calendar_wrong_closed"] == []


def test_webpage_has_daily_data():
    text = (ROOT / "docs" / "data.js").read_text(encoding="utf-8")
    data = json.loads(text[text.index("{"): text.rstrip().rindex("}") + 1])
    assert len(data["days"]) == len(data["calendar"]) == 30
    assert all(len(i["daily"]) == 30 for i in data["items"])
