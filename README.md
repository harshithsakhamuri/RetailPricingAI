# Retail Pricing AI

Profit-maximising price recommendations for retail items, learned from real sales data
([UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)).

Pick an item, or enter a new one, and get:
- **Forecast demand** for the coming week at the current price
- **Optimal price** (profit-maximising, within business rules)
- **Units you can sell** at that price, with expected profit and revenue
- **Confidence** and **profit at risk** compared with keeping the current price
- A **what-if slider** showing units, profit and risk at any price

No login and no server needed: the webpage runs entirely in the browser.

## Live webpage
Hosted with GitHub Pages from the `docs/` folder: `https://<your-username>.github.io/<repo-name>/`
(enable it in **Settings → Pages → Deploy from a branch → `main` / `/docs`**).

Locally, just open `docs/index.html` (or run `scripts\open_webpage.bat`).

## Project structure
```
├── app/
│   └── streamlit_app.py          # Optional Streamlit version of the app
├── data/
│   └── Online Retail.xlsx        # UCI Online Retail dataset (CC BY 4.0)
├── docs/                         # Static webpage (served by GitHub Pages)
│   ├── index.html                #   UI + in-browser price optimiser
│   └── data.js                   #   Exported model results (generated)
├── outputs/                      # Generated reports, plots and model artifacts
│   ├── summary.md                #   Full method + results write-up
│   ├── pricing_recommendations*.csv
│   └── app_artifacts.pkl         #   Read by the Streamlit app
├── scripts/                      # Windows helpers
│   ├── retrain.bat               #   Retrain + refresh docs/data.js
│   ├── run_app.bat               #   Launch the Streamlit app
│   └── open_webpage.bat          #   Open the static webpage
├── src/
│   ├── pricing_profit_optimizer.py   # Training + optimisation pipeline
│   └── export_web_data.py            # Export results for the webpage
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python src/pricing_profit_optimizer.py    # train (~1–2 min); writes outputs/
python src/export_web_data.py             # refresh docs/data.js for the webpage
streamlit run app/streamlit_app.py        # optional Streamlit app at http://localhost:8501
```

## How it works
1. **Demand forecast:** LightGBM, XGBoost and Random Forest (Tweedie loss) on a leakage-free daily panel of 500 items.
   The best model is chosen on validation. It forecasts the 7 days after the data ends.
2. **Price sensitivity (elasticity):** daily two-way fixed-effects panel regression per category, using item and date
   effects with HAC/clustered errors. Results are pooled global → category → item with empirical Bayes.
3. **Price optimisation:** robust profit maximisation over 500 elasticity draws, with these limits: ±20% vs current price,
   at most 10% above the highest price ever charged, a 20% minimum margin, realistic price endings and an optional stock cap.
4. **Confidence:** decision-based. It combines downside profit risk vs the current price with the probability that the
   change beats the current price.

See `outputs/summary.md` for full details and metrics, and [`reports/Project_Report.md`](reports/Project_Report.md) for the formal project report ([PDF](reports/Project_Report.pdf)).

## Notes and limitations
- Forecasts cover only 10–16 Dec 2011, the week after the data ends. Add newer sales data and retrain for current weeks.
- **Unit costs are assumed** (60% of median price) and **competitor prices are simulated**. Both can be changed in the app.
- Price changes in historical data are not experiments, so elasticities may be biased. A/B price tests give causal estimates.
- On this Windows machine LightGBM must be imported before pandas/pyarrow, and `pyarrow` is pinned to 18.1.0.
