"""
Export pipeline results to docs/data.js for the static webpage, and write
tests/fixtures/parity_cases.json: Python reference answers that
tests/test_parity.py replays through docs/pricing.js.

Run after the pipeline:   python src/export_web_data.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import pricing_core as core

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

ITEM_INPUTS = ["StockCode", "Description", "category", "current_price", "hist_min_price", "hist_max_price",
               "unit_cost", "baseline_units", "comp_price_SIMULATED", "e_raw", "elasticity", "elasticity_se",
               "elasticity_source", "n_weeks", "n_price_levels", "daily_log_sd", "variant_group", "joint_with"]
ITEM_RESULTS = ["recommended_price", "price_change_%", "expected_profit", "profit_at_current", "profit_change_crossfit",
                "profit_p10", "profit_p90", "confidence", "confidence_reason", "decision_driver", "lerner_gap",
                "test_price", "test_weeks_total", "test_weeks_pooled_category", "test_feasible"]


def clean_value(v):
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else round(float(v), 6)
    return v


def records(df):
    return [{k: clean_value(v) for k, v in r.items()} for r in df.to_dict("records")]


def category_defaults(p):
    out = {}
    for cat, g in p.groupby("category"):
        prior = g[g["elasticity_source"] == core.SRC_PRIOR]
        src = prior if len(prior) else g
        out[cat] = {"elasticity": round(float(src["elasticity"].median()), 6),
                    "elasticity_se": round(float(src["elasticity_se"].median()), 6),
                    "daily_log_sd": round(float(g["daily_log_sd"].median()), 6), "n_items": int(len(g))}
    return out


def python_case(row, z, ratios, edges, stock=math.inf, forced=None):
    """What pricing_core says for one input row (the parity reference)."""
    o = core.optimise_product(row, z, stock=stock, forced_price=forced)
    b = o["comp_same"]
    full = {**row, "recommended_price": b["price"], "downside": b["downside"], "p_improve": b["p_improve"],
            "flags": o["flags"]}
    conf, why = core.confidence(full)
    p10, p90 = core.profit_range(b["profit_draws"], ratios[int(np.searchsorted(edges, row["baseline_units"]))])
    return {"price": b["price"], "profit": b["profit"], "units": b["units"], "downside": b["downside"],
            "p_improve": None if math.isnan(b["p_improve"]) else b["p_improve"],
            "profit_change_crossfit": b["profit_change_crossfit"], "regret": b["regret"],
            "minus10_price": o["comp_minus10"]["price"], "plus10_profit": o["comp_plus10"]["profit"],
            "lo": o["bounds"][0], "hi": o["bounds"][1], "driver": o["decision_driver"], "flags": o["flags"],
            "confidence": conf, "reason": why, "p10": p10, "p90": p90,
            "test": core.test_plan(row["current_price"], b["price"], row["daily_log_sd"], *o["bounds"])}


def main():
    p = pd.read_parquet(OUT_DIR / "app_products.parquet")
    meta = json.loads((OUT_DIR / "app_meta.json").read_text(encoding="utf-8"))
    z = np.array(meta["z"])
    ratios = [np.array(r) for r in meta["forecast_ratio_samples"]]
    edges = meta["forecast_band_edges"]

    items = records(p[ITEM_INPUTS + ITEM_RESULTS])
    data = {"week": f"{pd.Timestamp(meta['horizon_start']):%d %b %Y} – {pd.Timestamp(meta['horizon_end']):%d %b %Y}",
            "model": meta["model"], "elasticity_level": meta.get("elasticity_level"), "wape": round(meta["test_wape_7day"], 1),
            "naive_wape": round(meta["best_naive_test_wape_7day"], 1), "z": meta["z"],
            "ratio_samples": meta["forecast_ratio_samples"], "band_edges": edges, "backtest": meta["backtest"],
            "gains_top": meta["gains_top"], "gains_all": meta["gains_all"], "cost_ratio": 0.60,
            "categories": category_defaults(p), "items": items}
    out = ROOT / "docs" / "data.js"
    out.write_text("window.PRICING_DATA = " + json.dumps(data, ensure_ascii=False, allow_nan=False) + ";\n",
                   encoding="utf-8")

    # ---- parity fixtures: a spread of real items plus edge cases ----
    rng = np.random.default_rng(0)
    pick = sorted(set(rng.choice(len(p), 30, replace=False).tolist())
                  | set(p.index[p["joint_with"] > 0][:3]) | set(p.index[p["elasticity_source"] == core.SRC_CONTRA][:3]))
    cases = []
    for i in pick:
        row = {k: clean_value(v) for k, v in p.loc[i, ITEM_INPUTS].items()}
        cases.append({"name": f"item {row['StockCode']}", "row": row, "stock": None, "forced": None,
                      "expect": python_case(row, z, ratios, edges)})
    base = {k: clean_value(v) for k, v in p.loc[0, ITEM_INPUTS].items()}
    edge = [
        ("stock cap", {**base}, 50.0, None),
        ("competitor 10% cheaper", {**base, "comp_mult": 0.9}, None, None),
        ("forced joint price", {**base}, None, round(base["current_price"] * 1.1, 2)),
        ("margin floor above range", {**base, "unit_cost": base["current_price"] * 1.2}, None, None),
        ("narrow range, no grid price", {**base, "current_price": 0.40, "hist_min_price": 0.40, "hist_max_price": 0.43,
                                         "unit_cost": 0.33, "custom": True}, None, None),
        ("custom new item", {"StockCode": "NEW", "current_price": 4.95, "hist_min_price": 3.96, "hist_max_price": 5.94,
                             "unit_cost": 2.97, "baseline_units": 100.0, "elasticity": -0.82, "elasticity_se": 1.0,
                             "elasticity_source": core.SRC_PRIOR, "daily_log_sd": 0.8, "custom": True}, None, None),
        ("very low volume", {**base, "baseline_units": 2.0}, None, None),
    ]
    for name, row, stock, forced in edge:
        cases.append({"name": name, "row": row, "stock": stock, "forced": forced,
                      "expect": python_case(row, z, ratios, edges, stock=stock if stock else math.inf, forced=forced)})
    fx = ROOT / "tests" / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    (fx / "parity_cases.json").write_text(json.dumps({"z": meta["z"], "ratios": meta["forecast_ratio_samples"],
                                                      "edges": edges, "cases": cases}, allow_nan=False),
                                          encoding="utf-8")
    print(f"Wrote {out} ({len(items)} items) and {len(cases)} parity cases")


if __name__ == "__main__":
    main()
