"""
Export the trained results to docs/data.js for the standalone web page (docs/index.html).
Run after the pipeline:   python src/export_web_data.py
"""
import lightgbm  # noqa: F401  must load before pandas/pyarrow
import json
import numpy as np
import pandas as pd

import pricing_profit_optimizer as pp

art = pd.read_pickle(pp.ARTIFACTS_PATH)
p = art["products"]

cols = ["StockCode", "Description", "category", "current_price", "train_min_price", "train_max_price",
        "unit_cost", "baseline_units", "comp_price_SIMULATED", "e_raw", "elasticity", "elasticity_se",
        "elasticity_source", "n_weeks", "n_price_levels"]
items = []
for r in p[cols].to_dict("records"):
    items.append({k: (None if isinstance(v, float) and np.isnan(v) else
                      round(v, 5) if isinstance(v, float) else
                      int(v) if isinstance(v, (np.integer,)) else v) for k, v in r.items()})

# Category elasticity priors (used for items the user types in themselves)
cats = {}
for cat, g in p.groupby("category"):
    prior = g[g["elasticity_source"] == "category prior only"]
    src = prior if len(prior) else g
    cats[cat] = {"elasticity": round(float(src["elasticity"].median()), 4),
                 "elasticity_se": round(float(src["elasticity_se"].median()), 4),
                 "n_items": int(len(g))}

data = {
    "week": f"{art['horizon_start']:%d %b %Y} – {art['horizon_end']:%d %b %Y}",
    "model": art["model"], "wape": round(float(art["test_wape_7day"]), 1),
    "z": [round(float(x), 5) for x in art["z"]],
    "categories": cats, "items": items,
    "cfg": {"ELASTICITY_CEILING": pp.ELASTICITY_CEILING, "CROSS_ELASTICITY": pp.CROSS_ELASTICITY,
            "COMP_REACTION_SHARE": pp.COMP_REACTION_SHARE, "COMP_REACTION_LAG_DAYS": pp.COMP_REACTION_LAG_DAYS,
            "HORIZON_DAYS": pp.HORIZON_DAYS, "COMP_SCENARIOS": pp.COMP_SCENARIOS,
            "COST_RATIO_DEFAULT": pp.COST_RATIO_DEFAULT, "MAX_PRICE_CHANGE": pp.MAX_PRICE_CHANGE,
            "MIN_MARGIN": pp.MIN_MARGIN, "PRICE_HEADROOM": pp.PRICE_HEADROOM, "PRICE_ENDINGS": list(pp.PRICE_ENDINGS),
            "ROBUST_QUANTILE": pp.ROBUST_QUANTILE},
}
out = pp.SCRIPT_DIR / "docs" / "data.js"
out.parent.mkdir(exist_ok=True)
out.write_text("window.PRICING_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print(f"Wrote {out} ({len(items)} items, {len(cats)} categories)")
