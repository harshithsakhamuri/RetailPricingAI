"""
Retail Pricing AI: prototype web app (Streamlit).

The user picks an item and sees:
  1. Forecast demand: expected units next week at today's price
  2. Ideal price: the profit-maximising price within business constraints
  3. Units sold at the ideal price

It does NOT retrain. It reads outputs/app_artifacts.pkl, which is produced by
    python pricing_profit_optimizer.py
and re-runs only the fast price optimisation when the user changes inputs.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lightgbm  # noqa: E402,F401  must load before pandas/pyarrow (see pricing_profit_optimizer.py)
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import pricing_profit_optimizer as pp

st.set_page_config(page_title="Retail Pricing AI", page_icon="🏷️", layout="wide")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    if not pp.ARTIFACTS_PATH.exists():
        return None
    return pd.read_pickle(pp.ARTIFACTS_PATH)


art = load_artifacts()
if art is None:
    st.error("No trained results found. Run `python pricing_profit_optimizer.py` first, then refresh this page.")
    st.stop()

products = art["products"]
z = art["z"]
week = f"{art['horizon_start']:%d %b %Y} – {art['horizon_end']:%d %b %Y}"


def run_optimiser(row: dict, stock):
    """Re-run the robust profit optimisation for one product with user inputs."""
    pp.STOCK_CAP = {row["StockCode"]: stock} if stock else {}
    o = pp.optimise_product(row, z)
    base = o["comp_same"]
    row = {**row, "recommended_price": base["price"], "p_improve": base["p_improve"], "regret": base["regret"], "downside": base["downside"], "at_bound": base["at_bound"],
           "flags": "; ".join(o["flags"])}
    conf, reason = pp.confidence(row)
    return o, base, conf, reason


# -----------------------------------------------------------------------------
# Header + item search
# -----------------------------------------------------------------------------
st.title("🏷️ Retail Pricing AI")
st.caption(f"Profit-maximising price recommendations · decision week **{week}** "
           f"(the 7 days after the sales data ends; the model cannot forecast further ahead)")

labels = (products["StockCode"] + " — " + products["Description"].str.title()).tolist()
choice = st.selectbox("Search for an item (type a name or stock code)", labels, index=None,
                      placeholder="e.g. lunch bag, 85123A, cake stand…")

if choice is None:
    st.info(f"Pick an item to see its demand forecast and ideal price. {len(products)} items are available.")
    st.stop()

r = products.iloc[labels.index(choice)].to_dict()

# -----------------------------------------------------------------------------
# Optional inputs (sensible defaults so the user can just pick an item)
# -----------------------------------------------------------------------------
with st.expander("Adjust assumptions (optional)"):
    c1, c2, c3 = st.columns(3)
    unit_cost = c1.number_input("Unit cost £", min_value=0.01, value=round(float(r["unit_cost"]), 2), step=0.05,
                                help=f"Default is an ASSUMPTION: {pp.COST_RATIO_DEFAULT:.0%} of median historical price.")
    stock = c2.number_input("Stock available for the week (0 = unlimited)", min_value=0, value=0, step=10)
    comp = c3.number_input("Competitor price £", min_value=0.01, value=round(float(r["comp_price_SIMULATED"]), 2),
                           step=0.05, help="Default is SIMULATED. No real competitor data exists yet.")

row = {**r, "unit_cost": unit_cost, "comp_price_SIMULATED": comp}
o, base, conf, reason = run_optimiser(row, stock)

p0, p_star = r["current_price"], base["price"]
chg = 100 * (p_star / p0 - 1)

# -----------------------------------------------------------------------------
# The three headline answers
# -----------------------------------------------------------------------------
st.subheader(r["Description"].title())
m1, m2, m3 = st.columns(3)
m1.metric("Forecast demand (at current price)", f"{base['units_current']:,.0f} units",
          help=f"Expected units for {week} at today's price of £{p0:.2f}.")
m2.metric("Ideal price", f"£{p_star:.2f}", f"{chg:+.1f}% vs current £{p0:.2f}" if abs(chg) > 0.05 else "keep current price",
          delta_color="off")
m3.metric("Units sold at ideal price", f"{base['units']:,.0f} units",
          f"{base['units'] - base['units_current']:+,.0f} vs current", delta_color="off")

m4, m5, m6 = st.columns(3)
m4.metric("Expected profit at ideal price", f"£{base['profit']:,.0f}",
          f"{base['profit'] - base['profit_current']:+,.0f} £ vs current price")
m5.metric("Likely profit range (P10–P90)", f"£{base['profit_p10']:,.0f} – £{base['profit_p90']:,.0f}")
badge = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}[conf]
m6.metric("Confidence", badge)
if reason:
    st.warning(f"**Why the confidence is {conf}:** {reason}")

# -----------------------------------------------------------------------------
# Profit / demand vs price chart
# -----------------------------------------------------------------------------
grid = np.linspace(p0 * 0.7, p0 * 1.3, 120)
e_draws = np.minimum(r["elasticity"] + r["elasticity_se"] * z, pp.ELASTICITY_CEILING)
units = pp.demand_matrix(grid, p0, r["baseline_units"], e_draws, comp, 1.0)
if stock:
    units = np.minimum(units, stock)
profit = (grid - unit_cost)[None, :] * units
curve = pd.DataFrame({"price": grid, "expected profit £": profit.mean(0),
                      "P10": np.quantile(profit, .1, 0), "P90": np.quantile(profit, .9, 0),
                      "expected units": units.mean(0)})
lo, hi = o["bounds"]

base_chart = alt.Chart(curve).encode(x=alt.X("price:Q", title="Price £"))
band = base_chart.mark_area(opacity=0.2).encode(y=alt.Y("P10:Q", title="Profit £ for the week"), y2="P90:Q")
line = base_chart.mark_line(strokeWidth=3).encode(y="expected profit £:Q",
                                                  tooltip=[alt.Tooltip("price:Q", format=".2f"),
                                                           alt.Tooltip("expected profit £:Q", format=",.0f"),
                                                           alt.Tooltip("expected units:Q", format=",.0f")])
allowed = alt.Chart(pd.DataFrame({"lo": [lo], "hi": [hi]})).mark_rect(opacity=0.08, color="green") \
    .encode(x="lo:Q", x2="hi:Q")
rules = alt.Chart(pd.DataFrame({"price": [p0, p_star], "label": ["current", "ideal"]})) \
    .mark_rule(strokeWidth=2, strokeDash=[4, 3]).encode(x="price:Q", color=alt.Color("label:N", title=None))

left, right = st.columns([3, 2])
with left:
    st.markdown("**Profit vs price** (shaded green = allowed price range; band = uncertainty)")
    st.altair_chart(allowed + band + line + rules, width="stretch")
with right:
    st.markdown("**What if I charge…**")
    try_p = st.slider("Price £", float(round(p0 * 0.7, 2)), float(round(p0 * 1.3, 2)), float(p_star), 0.01)
    u_try = pp.demand_matrix(np.array([try_p]), p0, r["baseline_units"], e_draws, comp, 1.0)[:, 0]
    if stock:
        u_try = np.minimum(u_try, stock)
    st.metric("Expected units", f"{u_try.mean():,.0f}")
    st.metric("Expected profit", f"£{((try_p - unit_cost) * u_try).mean():,.0f}")
    st.markdown("**Competitor scenarios** (SIMULATED competitor)")
    st.dataframe(pd.DataFrame([{"Competitor": k.replace("comp_", "").replace("minus10", "−10%")
                               .replace("plus10", "+10%"), "Ideal price £": v["price"],
                                "Units": round(v["units"]), "Profit £": round(v["profit"])}
                               for k, v in o.items() if k.startswith("comp_")]),
                 hide_index=True, width="stretch")

with st.expander("Details and model inputs"):
    st.write({
        "Elasticity (pooled)": f"{r['elasticity']:.2f} ± {r['elasticity_se']:.2f} ({r['elasticity_source']})",
        "Weeks of history / price levels": f"{int(r['n_weeks'])} / {int(r['n_price_levels'])}",
        "Allowed price range £": f"{lo:.2f} – {hi:.2f} (±{pp.MAX_PRICE_CHANGE:.0%}, train range, "
                                 f"min margin {pp.MIN_MARGIN:.0%})",
        "Forecast model": f"{art['model']} (test error on weekly totals: {art['test_wape_7day']:.0f}% WAPE)",
    })

st.caption("Prototype. Demand and elasticity come from real UCI Online Retail sales. Unit costs are assumed and "
           "competitor prices are SIMULATED. Treat results as directional, especially low-confidence items.")

