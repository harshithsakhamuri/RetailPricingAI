# Retail Pricing AI

[![CI](https://github.com/harshithsakhamuri/RetailPricingAI/actions/workflows/ci.yml/badge.svg)](https://github.com/harshithsakhamuri/RetailPricingAI/actions/workflows/ci.yml)

Profit-maximising price recommendations for retail items, learned from real sales data
([UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)), with an honest account of how much
the data actually supports each recommendation.

**Live webpage:** https://harshithsakhamuri.github.io/RetailPricingAI/ (runs entirely in the browser)

- **Existing item:** forecast demand, recommended price, units, profit, a P10–P90 range that includes forecast error,
  confidence, *what set the price* (the data or a business limit), a price-test plan, and **profit on any single day
  up to a month ahead** (defaults to the next day the shop is open; closed days show why)
- **New item:** the same for an item you describe
- **Batch upload:** price many of your own items from a CSV and download the results
- **All recommendations:** sortable, filterable table of all 500 items with CSV download

Reports: [`outputs/summary.md`](outputs/summary.md) (generated on every run) and the formal
[project report](reports/Project_Report.md) ([PDF](reports/Project_Report.pdf)).

## Project structure
```
├── app/streamlit_app.py           # Optional Streamlit app (one item + all recommendations)
├── data/README.md                 # Where to download the dataset (not committed)
├── docs/                          # Static webpage (GitHub Pages)
│   ├── index.html                 #   UI
│   ├── pricing.js                 #   Decision maths: JS port of src/pricing_core.py
│   └── data.js                    #   Exported results (generated)
├── outputs/                       # Generated results, plots and app data
├── reports/                       # Formal project report (Markdown + PDF)
├── scripts/                       # retrain / run_app / open_webpage / run_tests (.bat and .sh)
├── src/
│   ├── pricing_core.py            # Decision maths shared by pipeline and apps
│   ├── pricing_profit_optimizer.py# Data, forecasting, elasticity, backtest, outputs
│   └── export_web_data.py         # Writes docs/data.js + parity test fixtures
├── tests/                         # Regression tests (run in CI on every push)
└── requirements.txt               # Pinned versions (requirements-dev.txt adds pytest)
```

## Setup and usage
```bash
python -m venv .venv
.venv\Scripts\activate                     # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

python src/pricing_profit_optimizer.py     # ~1.5 min; writes outputs/
python src/pricing_profit_optimizer.py --as-of 2011-09-30   # any other decision week -> outputs/asof_2011-09-30/
python src/export_web_data.py              # refresh docs/data.js
pytest -q                                  # all checks
streamlit run app/streamlit_app.py         # optional app at http://localhost:8501
```
Or use `scripts/retrain.(bat|sh)`, which runs all three steps and fails if any check fails.

## How it works
1. **Data:** cancellations are netted against the customer's earlier orders; lines without a CustomerID are excluded
   by default (`INCLUDE_GUEST_LINES`). One price definition throughout: the list price from regular-size lines.
2. **Holidays:** a closure calendar (no trading on Saturdays, England & Wales bank holidays, Easter Sunday and
   24 Dec – 2 Jan) checked against all 374 days in the data with zero mismatches, plus UK gifting events
   (Valentine's, Mothering Sunday, Easter, Father's Day, Halloween, Bonfire Night, Christmas). Features: days
   to/since events, closures in the next/previous 7 days, and reopening days (which sell about half as much).
3. **Demand forecast:** LightGBM, XGBoost, Random Forest, an ensemble **and two naive baselines** all get the same
   level calibration and compete on held-out validation weeks. The production calibration is measured on a refit model.
   Days 1–7 come from the model; days 8–30 use next week's level × last year's pattern for the same days.
4. **Price sensitivity:** the overall level comes from an event study of the shop's 51 real list-price changes
   (sales before vs after, net of similar products): elasticity −2.2. A Poisson (PPML) panel with item and date
   fixed effects on every trading day decides how categories and items differ; its slopes are rescaled to the event
   level because noisy daily prices pull them toward zero. Wrong-signed items fall back to the category and are
   flagged. Checked with a **placebo test** and an **out-of-sample backtest**.
5. **Decision** (`src/pricing_core.py`): the price with the highest expected profit over the elasticity uncertainty, within ±20% of the current price, ≤ 10%
   above the highest price charged and a 20% minimum margin. Colour/design variants that share a price are priced
   jointly. A change is suggested only if its gain survives a cross-check (otherwise the price is kept), so no
   suggestion shows a negative extra profit. Items priced below cost get the largest allowed step up, with a warning.
   Each item reports what set its price, a Lerner sanity check, a cross-fitted gain and a test plan.
6. **Confidence:** "high" needs low profit-at-risk **and** the item's own price history; category-only items are
   capped at medium.

## What the results say (honestly)
- Real price changes show a 10% rise cuts sales by about 22%. At that sensitivity and the assumed 40% margin,
  today's prices are close to the best price: 223 items get a small rise, 215 stay the same and 62 get a small cut.
  Calibrating to real price changes cut the backtest error from 1.09 to 0.91 (1.41 when assuming price has no effect).
- Gains are small: about £391 a week across all 500 items from high- and medium-confidence suggestions.
- Many items show the same % change because we know the same things about them: most never changed price and all
  use the same assumed cost ratio, so items at the same price point (e.g. £2.95) get the same answer. Entering real
  costs makes each suggestion specific. The simulated competitor deliberately does not drive prices.
- Two items are priced below their estimated cost and still lose money after the largest allowed step; the page
  says so plainly.
- Only 7 items reach high confidence. Single-item price tests mostly need far more than 12 weeks; testing the
  same change across a category is practical. See `outputs/price_test_plan.csv`.
- The ML forecast is no better than a calibrated 28-day average on the test weeks; adding holiday features did not
  change weekly accuracy noticeably (53.3% vs 53.5% error), but the calendar now gets closed days exactly right.

## Tests and CI
`pytest -q` runs on every push (GitHub Actions). The tests cover the decision maths, every review finding as a
regression test, feature-leakage, recovery of a known elasticity, Python/JS parity on 41 cases, a guarantee that no suggestion earns less than today's price, the Streamlit app,
and invariants on the committed outputs. A manual workflow run with *full_pipeline* downloads the dataset,
re-trains and re-checks everything.

## Notes
- Forecasts cover 10–16 Dec 2011, the week after the data ends; use `--as-of` for other weeks.
- **Unit costs are assumed** (60% of median list price) and **competitor prices are simulated**; both can be changed.
- On Windows, LightGBM must be imported before pandas/pyarrow, and `pyarrow` is pinned to 18.1.0.
