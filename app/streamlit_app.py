"""
Retail Pricing AI: Streamlit app.

Reads outputs/app_products.parquet + outputs/app_meta.json (written by
src/pricing_profit_optimizer.py) and re-runs only the fast decision maths in
src/pricing_core.py. No ML libraries are imported, and no module state is
modified, so concurrent sessions cannot affect each other.

Run:  streamlit run app/streamlit_app.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import pricing_core as core  # noqa: E402

OUT = ROOT / "outputs"
st.set_page_config(page_title="Retail Pricing AI", page_icon="🏷️", layout="wide")


@st.cache_data
def load():
    if not (OUT / "app_products.parquet").exists():
        return None, None
    return pd.read_parquet(OUT / "app_products.parquet"), json.loads((OUT / "app_meta.json").read_text(encoding="utf-8"))


products, meta = load()
if products is None:
    st.error("No results found. Run `python src/pricing_profit_optimizer.py` first, then refresh.")
    st.stop()
z = np.array(meta["z"])
ratios = [np.array(r) for r in meta["forecast_ratio_samples"]]
week = f"{pd.Timestamp(meta['horizon_start']):%d %b %Y} – {pd.Timestamp(meta['horizon_end']):%d %b %Y}"

st.title("🏷️ Retail Pricing AI")
st.caption(f"Profit-maximising price recommendations · decision week **{week}**")

view = st.radio("View", ["One item", "All recommendations"], horizontal=True, label_visibility="collapsed")

if view == "All recommendations":
    g = meta["gains_all"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Gain, high + medium confidence", f"£{g.get('high', 0) + g.get('medium', 0):,.0f}", help="Cross-fitted, 1 week")
    c2.metric("Gain, low confidence", f"£{g.get('low', 0):,.0f}")
    c3.metric("Prices set by a limit", f"{products['decision_driver'].str.contains('limit').sum()} / {len(products)}")
    f1, f2, f3 = st.columns(3)
    conf = f1.multiselect("Confidence", ["high", "medium", "low"])
    cat = f2.multiselect("Category", sorted(products["category"].unique()))
    ev = f3.multiselect("Evidence", sorted(products["elasticity_source"].unique()))
    d = products
    if conf:
        d = d[d["confidence"].isin(conf)]
    if cat:
        d = d[d["category"].isin(cat)]
    if ev:
        d = d[d["elasticity_source"].isin(ev)]
    cols = ["StockCode", "Description", "category", "current_price", "recommended_price", "price_change_%",
            "profit_change_crossfit", "confidence", "elasticity", "elasticity_source", "decision_driver",
            "test_weeks_total", "test_weeks_pooled_category", "confidence_reason"]
    d = d.sort_values("profit_change_crossfit", ascending=False)[cols]
    st.dataframe(d, hide_index=True, width="stretch")
    st.download_button("Download CSV", d.to_csv(index=False), "pricing_recommendations.csv", "text/csv")
    st.stop()

labels = (products["StockCode"] + " — " + products["Description"].str.title()).tolist()
choice = st.selectbox("Search for an item (type a name or stock code)", labels, index=None,
                      placeholder="e.g. lunch bag, 85123A, cake stand…")
if choice is None:
    st.info(f"Pick an item. {len(products)} items are available.")
    st.stop()
r = products.iloc[labels.index(choice)].to_dict()

with st.expander("Adjust assumptions (optional)"):
    c1, c2, c3 = st.columns(3)
    unit_cost = c1.number_input("Unit cost £", min_value=0.01, value=round(float(r["unit_cost"]), 2), step=0.05,
                                help="Default is an ASSUMPTION: 60% of the median list price.")
    stock = c2.number_input("Stock available for the week (0 = unlimited)", min_value=0, value=0, step=10)
    comp = c3.number_input("Competitor price £", min_value=0.01, value=round(float(r["comp_price_SIMULATED"]), 2),
                           step=0.05, help="Default is SIMULATED.")

row = {**r, "unit_cost": r["unit_cost"] if abs(unit_cost - r["unit_cost"]) < 0.005 else unit_cost,
       "comp_mult": 1.0 if abs(comp - r["comp_price_SIMULATED"]) < 0.005 else comp / r["comp_price_SIMULATED"]}
stock_cap = stock if stock else math.inf
indiv = core.optimise_product(row, z, stock=stock_cap)
joint = r["joint_with"] > 0
o = core.optimise_product(row, z, stock=stock_cap, forced_price=r["recommended_price"]) if joint else indiv
o["bounds"] = indiv["bounds"]
b = o["comp_same"]
conf, reason = core.confidence({**row, "recommended_price": b["price"], "downside": b["downside"],
                                "p_improve": b["p_improve"], "flags": o["flags"]})
p10, p90 = core.profit_range(b["profit_draws"], ratios[int(np.searchsorted(meta["forecast_band_edges"], row["baseline_units"]))])
plan = core.test_plan(row["current_price"], b["price"], row["daily_log_sd"], *o["bounds"])
p0, p_star = row["current_price"], b["price"]
chg = 100 * (p_star / p0 - 1)

st.subheader(r["Description"].title())
m1, m2, m3 = st.columns(3)
m1.metric("Forecast demand (at current price)", f"{b['units_current']:,.0f} units")
m2.metric("Recommended price", f"£{p_star:.2f}", f"{chg:+.1f}% vs £{p0:.2f}" if abs(chg) > 0.05 else "keep current price",
          delta_color="off")
m3.metric("Units at recommended price", f"{b['units']:,.0f}", f"{b['units'] - b['units_current']:+,.0f}", delta_color="off")
m4, m5, m6 = st.columns(3)
m4.metric("Expected profit", f"£{b['profit']:,.0f}", f"{b['profit'] - b['profit_current']:+,.0f} £ vs current")
m5.metric("Likely profit range (P10–P90)", f"£{p10:,.0f} – £{p90:,.0f}", help="Includes forecast error and price-sensitivity uncertainty.")
m6.metric("Confidence", {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}[conf])
if reason:
    st.warning(f"**Why the confidence is {conf}:** {reason}")
info = [f"**What sets this price:** {o['decision_driver']}."]
if joint:
    info.append(f"**Priced jointly** with {int(r['joint_with'])} variant(s); on its own the model would pick £{indiv['comp_same']['price']:.2f}.")
L = core.lerner(min(row["elasticity"], core.ELASTICITY_CEILING), row["unit_cost"], p0)
if not math.isfinite(L["unconstrained_price"]) or L["unconstrained_price"] > 1.5 * p0:
    info.append("**Sanity check:** this price sensitivity implies a far higher unconstrained optimal price than today's, "
                "which is implausible; the true sensitivity is probably stronger. Treat increases with caution.")
info.append(f"**Price test:** alternate weeks at £{p0:.2f} and £{plan['test_price']:.2f}; about {plan['test_days_per_arm']} "
            f"trading days per price (≈ {plan['test_weeks_total']} weeks"
            + ("" if plan["test_feasible"] else f"; ≈ {int(r['test_weeks_pooled_category'])} weeks if run across the category") + ").")
st.info("\n\n".join(info))

grid = np.linspace(p0 * 0.7, p0 * 1.3, 120)
units = np.minimum(core.demand_matrix(grid, p0, row["baseline_units"], o["e_draws"], row["comp_mult"]), stock_cap)
profit = (grid - row["unit_cost"])[None, :] * units
curve = pd.DataFrame({"price": grid, "expected profit £": profit.mean(0), "P10": np.quantile(profit, .1, 0),
                      "P90": np.quantile(profit, .9, 0), "expected units": units.mean(0)})
lo, hi = o["bounds"]
base_chart = alt.Chart(curve).encode(x=alt.X("price:Q", title="Price £"))
band = base_chart.mark_area(opacity=0.2).encode(y=alt.Y("P10:Q", title="Profit £ for the week"), y2="P90:Q")
line = base_chart.mark_line(strokeWidth=3).encode(y="expected profit £:Q", tooltip=["price:Q", "expected profit £:Q", "expected units:Q"])
allowed = alt.Chart(pd.DataFrame({"lo": [lo], "hi": [hi]})).mark_rect(opacity=0.08, color="green").encode(x="lo:Q", x2="hi:Q")
rules = alt.Chart(pd.DataFrame({"price": [p0, p_star], "label": ["current", "recommended"]})) \
    .mark_rule(strokeWidth=2, strokeDash=[4, 3]).encode(x="price:Q", color=alt.Color("label:N", title=None))
left, right = st.columns([3, 2])
with left:
    st.markdown("**Profit vs price** (green = allowed range; band = price-sensitivity uncertainty)")
    st.altair_chart(allowed + band + line + rules, width="stretch")
with right:
    st.markdown("**What if I charge…**")
    try_p = st.slider("Price £", float(round(p0 * 0.7, 2)), float(round(p0 * 1.3, 2)), float(p_star), 0.01)
    u_try = np.minimum(core.demand_matrix([try_p], p0, row["baseline_units"], o["e_draws"], row["comp_mult"])[:, 0], stock_cap)
    st.metric("Expected units", f"{u_try.mean():,.0f}")
    st.metric("Expected profit", f"£{((try_p - row['unit_cost']) * u_try).mean():,.0f}")
    st.markdown("**Competitor scenarios** (SIMULATED competitor)")
    st.dataframe(pd.DataFrame([{"Competitor": k, "Price £": v["price"], "Units": round(v["units"]), "Profit £": round(v["profit"])}
                               for k, v in o.items() if k.startswith("comp_")]), hide_index=True, width="stretch")

with st.expander("Details"):
    st.write({"Price sensitivity": f"{row['elasticity']:.2f} ± {row['elasticity_se']:.2f} ({row['elasticity_source']})",
              "Weeks with sales / weekly price levels": f"{int(row['n_weeks'])} / {int(row['n_price_levels'])}",
              "Allowed price range £": f"{lo:.2f} – {hi:.2f}",
              "Forecast model": f"{meta['model']} ({meta['test_wape_7day']:.0f}% weekly WAPE; best simple baseline "
                                f"{meta['best_naive_test_wape_7day']:.0f}%)"})
st.caption("Prototype. Unit costs are assumed and competitor prices are SIMULATED. Test before acting.")
