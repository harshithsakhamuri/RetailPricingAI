"""
=============================================================================
RETAIL PROFIT-MAXIMISING PRICING  —  UCI Online Retail
=============================================================================
Two-stage system:

  STAGE 1  DEMAND MODEL
      Expected units(p) = Baseline  x  Own-price effect  x  Competitor effect
        Baseline           ML forecast (LightGBM / XGBoost / RF / ensemble,
                           chosen on validation vs naive baselines), trained on
                           real sales only, NO price features (so price effects
                           are not double counted).
        Own-price effect   (p / p_current) ^ elasticity, elasticity estimated
                           from REAL weekly price/demand variation with a
                           log-log regression + seasonality control, then
                           partially pooled (empirical Bayes) by category.
        Competitor effect  ((comp/p) / (comp0/p0)) ^ CROSS_ELASTICITY, using a
                           SIMULATED competitor price (no real data exists).

  STAGE 2  PROFIT OPTIMISATION
      For each product, search rounded candidate prices inside business
      constraints and pick the price that maximises a ROBUST (downside
      quantile) profit across the elasticity uncertainty; run competitor
      scenarios.

Objective is PROFIT = (price - unit cost) x expected units — not volume, not
revenue.

Every column / plot / report line that depends on the simulated competitor is
labelled SIMULATED. Set environment variable COMPETITOR_PATH to a CSV with
columns StockCode, Date, CompetitorPrice to use real competitor data instead.

Run:  python pricing_profit_optimizer.py
=============================================================================
"""

import os
import time
import zlib
import warnings
from pathlib import Path

# LightGBM MUST be imported before pandas: pandas loads pyarrow, whose bundled
# native runtime makes LightGBM crash ("access violation") if it loads first.
import lightgbm as lgb  # noqa: E402  (import order is deliberate)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # write plots to files, no GUI
import matplotlib.pyplot as plt
import holidays
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# =============================================================================
# CONFIGURATION  (everything tunable lives here)
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent.parent   # project root (this file lives in src/)
DATA_PATH = Path(os.environ.get("RETAIL_DATA_PATH", SCRIPT_DIR / "data" / "Online Retail.xlsx"))
ARTIFACTS_PATH = SCRIPT_DIR / "outputs" / "app_artifacts.pkl"   # read by app.py
OUTPUT_DIR = Path(os.environ.get("PRICING_OUTPUT_DIR", SCRIPT_DIR / "outputs"))
CACHE_PATH = OUTPUT_DIR / "_cache_online_retail.pkl"   # parsed xlsx (slow to read)
COMPETITOR_PATH = os.environ.get("COMPETITOR_PATH") or None  # real data hook

SEED = 42

# --- Time splits (strictly by date) ---------------------------------------
TRAIN_END = pd.Timestamp("2011-08-31")    # train : <= TRAIN_END
VAL_END = pd.Timestamp("2011-10-31")      # val   : (TRAIN_END, VAL_END]
                                          # test  : > VAL_END (Nov–9 Dec 2011)

# --- Forecasting -------------------------------------------------------------
FORECAST_LEAD = 7        # features only use info >= 7 days old -> honest 7-day-ahead
HORIZON_DAYS = FORECAST_LEAD   # pricing decision horizon = the 7 days after data end
TOP_N_MODEL = 500        # products modelled (top by TRAIN revenue)
MIN_TRAIN_SALE_DAYS = 30 # need enough history to model a product
TOP_N_REPORT = 50        # products in the main recommendations CSV
DEMAND_CAP_Q = 0.995     # per-product cap on daily units, quantile computed on TRAIN

# --- Elasticity ----------------------------------------------------------------
ELAST_MIN_WEEKS = 8          # min weeks with sales for a product-level estimate
ELAST_MIN_PRICE_LEVELS = 2   # min distinct weekly list prices, each held >= 3 weeks
                             # (one real list-price change = before/after natural experiment)
ELAST_MIN_WEEKS_PER_LEVEL = 3
ELAST_MIN_LOGP_STD = 0.01    # min price variation (std of log price)
ELAST_SE_FLOOR = 0.05        # avoid over-confident product estimates
TAU_MIN = 0.10               # min between-product spread inside a category
MIN_PRODUCTS_PER_CATEGORY = 4  # else the category borrows the global prior
ELASTICITY_CEILING = -0.10   # decision draws capped here (demand must fall with price)

# --- Competitor (SIMULATED unless COMPETITOR_PATH is given) ---------------
CROSS_ELASTICITY = 0.5
COMP_OFFSET_RANGE = (-0.15, 0.15)   # per-product permanent offset vs our price
COMP_AR_PHI = 0.90                  # mean-reverting drift: AR(1) coefficient (log scale)
COMP_AR_SIGMA = 0.02                # daily shock std of the drift
COMP_PROMO_DAILY_PROB = 0.02        # chance a promotion starts on a given day
COMP_PROMO_DEPTH = (0.10, 0.30)     # promo price drop
COMP_PROMO_DAYS = (3, 7)            # promo length (days)
COMP_REACTION_LAG_DAYS = 7          # competitor follows our price after this lag...
COMP_REACTION_SHARE = 0.5           # ...by this fraction (partial reaction)
COMP_SCENARIOS = {"comp_minus10": 0.90, "comp_same": 1.00, "comp_plus10": 1.10}

# --- Costs (no cost data exists -> assumption) ---------------------------
COST_RATIO_DEFAULT = 0.60           # unit cost = 60% of product's median TRAIN price
COST_RATIO_BY_CATEGORY = {}         # e.g. {"bags": 0.55}
COST_OVERRIDES_PRODUCT = {}         # absolute unit cost in £, e.g. {"85123A": 1.40}

# --- Pricing constraints ---------------------------------------------------
MAX_PRICE_CHANGE = 0.20             # max ±20% vs current price
MIN_MARGIN = 0.20                   # (price - cost) / price must be >= this
PRICE_HEADROOM = 0.10               # may test up to 10% above the highest price ever charged (flagged)
STOCK_CAP = {}                      # units available over the horizon, e.g. {"22423": 300}
PRICE_ENDINGS = (0.05, 0.10, 0.25, 0.45, 0.49, 0.55, 0.65, 0.75, 0.79, 0.85, 0.95, 0.99)

# --- Uncertainty -------------------------------------------------------------
N_ELASTICITY_DRAWS = 500
ROBUST_QUANTILE = 0.25    # choose price maximising the 25th-percentile profit

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
L = FORECAST_LEAD


# =============================================================================
# 1. LOAD & CLEAN TRANSACTIONS
# =============================================================================
def load_transactions() -> pd.DataFrame:
    """Read the xlsx once and cache a pickle (the xlsx takes ~1 min to parse)."""
    if CACHE_PATH.exists() and CACHE_PATH.stat().st_mtime > DATA_PATH.stat().st_mtime:
        return pd.read_pickle(CACHE_PATH)
    log(f"Reading {DATA_PATH} (first run only, cached afterwards)...")
    raw = pd.read_excel(DATA_PATH, dtype={"InvoiceNo": str, "StockCode": str})
    raw.to_pickle(CACHE_PATH)
    return raw


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    n0 = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C")]              # cancellations
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[df["CustomerID"].notna()]                           # consistent B2B price list
    df = df[df["StockCode"].str.match(r"^\d{5}")]               # drop POST, M, DOT, fees...
    # Drop price typos: lines priced >5x or <0.2x the product's median price
    med = df.groupby("StockCode")["UnitPrice"].transform("median")
    df = df[(df["UnitPrice"] <= 5 * med) & (df["UnitPrice"] >= 0.2 * med)]
    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    log(f"Cleaned transactions: {len(df):,} of {n0:,} raw lines")
    return df


# =============================================================================
# 2. PRODUCT CATEGORIES (keyword rules on Description) — used for pooling
# =============================================================================
CATEGORY_RULES = [
    ("christmas", ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER", "SNOWFLAKE", "BAUBLE"]),
    ("bags", ["BAG", "SHOPPER", "TOTE"]),
    ("lighting_candles", ["T-LIGHT", "TEALIGHT", "LANTERN", "CANDLE", "LIGHT", "NIGHT LIGHT"]),
    ("kitchen_baking", ["CAKE", "BAKING", "MUG", "CUP", "PLATE", "BOWL", "TEA", "JAM", "LUNCH",
                        "BOTTLE", "NAPKIN", "TRAY", "CUTLERY", "KITCHEN", "BREAD", "APRON", "SPOON"]),
    ("stationery_paper", ["CARD", "WRAP", "PAPER", "NOTEBOOK", "PEN", "PENCIL", "STICKER",
                          "CHALK", "BUNTING", "TISSUE"]),
    ("storage", ["BOX", "TIN", "JAR", "BASKET", "STORAGE", "CRATE", "CASE"]),
    ("home_decor", ["SIGN", "HEART", "FRAME", "CLOCK", "DOORMAT", "CUSHION", "DECORATION",
                    "HANGING", "MIRROR", "WALL", "ORNAMENT", "GARLAND", "HOOK"]),
    ("toys_gifts", ["TOY", "GAME", "DOLLY", "DOLL", "KIT", "PARTY", "BALLOON", "SPACEBOY",
                    "CHILDREN", "ALARM", "SKIPPING", "PLAYHOUSE"]),
]


def categorise(desc: str) -> str:
    for cat, kws in CATEGORY_RULES:
        if any(k in desc for k in kws):
            return cat
    return "other"


# =============================================================================
# 3. CALENDAR FEATURES (all known in advance -> no leakage)
# =============================================================================
UK_HOLIDAYS = sorted(pd.Timestamp(d) for d in
                     holidays.country_holidays("GB", subdiv="ENG", years=[2010, 2011, 2012]))


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame({"Date": dates})
    d = cal["Date"]
    hol = np.array(UK_HOLIDAYS, dtype="datetime64[ns]")
    dv = d.values
    nxt = np.searchsorted(hol, dv, side="left")
    prv = np.searchsorted(hol, dv, side="right") - 1
    day = np.timedelta64(1, "D")
    cal["dow"] = d.dt.dayofweek
    cal["month"] = d.dt.month
    cal["weekofyear"] = d.dt.isocalendar().week.astype(int).values
    doy = d.dt.dayofyear
    cal["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cal["is_holiday"] = d.isin(UK_HOLIDAYS).astype(int)
    cal["days_to_hol"] = np.clip((hol[np.minimum(nxt, len(hol) - 1)] - dv) / day, 0, 30)
    cal["days_since_hol"] = np.clip((dv - hol[np.maximum(prv, 0)]) / day, 0, 30)
    # Days to next Christmas (covers Dec 2010 as well as Dec 2011)
    xmas_this = pd.to_datetime(d.dt.year.astype(str) + "-12-25")
    xmas_next = pd.to_datetime((d.dt.year + 1).astype(str) + "-12-25")
    xmas = xmas_this.where(d <= xmas_this, xmas_next)
    cal["days_to_xmas"] = np.clip((xmas - d).dt.days, 0, 90)
    cal["trend"] = (d - pd.Timestamp("2010-12-01")).dt.days
    # Store calendar (planned, so usable for the future): no Saturday trading,
    # closed on bank holidays and over the Christmas break.
    xmas_break = ((d.dt.month == 12) & (d.dt.day >= 23)) | ((d.dt.month == 1) & (d.dt.day <= 3))
    cal["store_open"] = ((cal["dow"] != 5) & (cal["is_holiday"] == 0) & ~xmas_break).astype(int)
    return cal


CAL_FEATS = ["dow", "month", "weekofyear", "doy_sin", "doy_cos", "is_holiday", "days_to_hol",
             "days_since_hol", "days_to_xmas", "trend", "store_open"]


# =============================================================================
# 4. DAILY PANEL ON A FULL CALENDAR + LEAKAGE-FREE FEATURES
# =============================================================================
def build_panel(df: pd.DataFrame, products: list, cats: pd.Series) -> pd.DataFrame:
    sub = df[df["StockCode"].isin(products)]
    daily = (sub.groupby(["StockCode", "Date"])
                .agg(units=("Quantity", "sum"), price=("UnitPrice", "median"),
                     revenue=("Revenue", "sum")).reset_index())
    data_end = df["Date"].max()
    future_end = data_end + pd.Timedelta(days=HORIZON_DAYS)

    # Reindex every product to a full daily calendar from its first sale to the
    # end of the forecast horizon -> shift(k) really means "k calendar days ago".
    first_sale = daily.groupby("StockCode")["Date"].min()
    frames = [pd.DataFrame({"StockCode": c, "Date": pd.date_range(f, future_end, freq="D")})
              for c, f in first_sale.items()]
    p = pd.concat(frames, ignore_index=True).merge(daily, on=["StockCode", "Date"], how="left")
    p = p.sort_values(["StockCode", "Date"]).reset_index(drop=True)
    p["is_future"] = p["Date"] > data_end
    p["units"] = np.where(p["is_future"], np.nan, p["units"].fillna(0.0))
    p = p.merge(calendar_features(pd.date_range(p["Date"].min(), future_end)), on="Date", how="left")

    # Per-product outlier cap computed on TRAIN sale days only, applied everywhere.
    tr_sales = p[(p["Date"] <= TRAIN_END) & (p["units"] > 0)]
    cap = tr_sales.groupby("StockCode")["units"].quantile(DEMAND_CAP_Q)
    p["cap"] = p["StockCode"].map(cap)
    p["u_cap"] = np.minimum(p["units"], p["cap"])

    g = p.groupby("StockCode", sort=False)
    key = p["StockCode"]

    def roll(series, w, fn):
        return series.groupby(key, sort=False).transform(lambda x: x.rolling(w, min_periods=1).agg(fn))

    # --- Lags & rolling windows; everything is >= L days old --------------
    s = g["u_cap"].shift(L)                                  # units at t-L
    for k in (0, 7, 14, 21):
        p[f"lag_{L + k}"] = g["u_cap"].shift(L + k)
    open_s = g["store_open"].shift(L)
    for w in (7, 14, 28, 56):
        p[f"rate_{w}"] = roll(s, w, "sum") / roll(open_s, w, "sum").clip(lower=1)  # units / open day
    p["rstd_28"] = roll(s, 28, "std")
    p["nz_28"] = roll((s > 0).astype(float).where(s.notna()), 28, "mean")       # selling frequency
    p["ewm_7"] = s.groupby(key, sort=False).transform(lambda x: x.ewm(halflife=7).mean())
    p["momentum"] = (p["rate_7"] / (p["rate_56"] + 0.1)).clip(0, 10)            # recent vs longer
    # Level scale: models predict units / scale (a multiplier on the product's
    # recent selling rate). Trees cannot extrapolate absolute levels, so an
    # unscaled model under-forecasts when demand grows into the Q4 peak.
    p["scale"] = (0.5 * p["rate_28"] + 0.5 * p["rate_56"]).fillna(0).clip(lower=0.2)
    p["rel_lag"] = p[f"lag_{L}"] / p["scale"]
    p["rel_rate_7"] = p["rate_7"] / p["scale"]
    p["rel_rate_14"] = p["rate_14"] / p["scale"]
    last_sale = p["Date"].where(p["u_cap"] > 0).groupby(key).ffill()
    p["age"] = (p["Date"] - key.map(first_sale)).dt.days
    p["days_since_sale"] = (p["Date"] - last_sale.groupby(key).shift(L)).dt.days
    p["days_since_sale"] = p["days_since_sale"].fillna(p["age"]).clip(upper=180)

    # --- Static product attributes (train-only statistics) -----------------
    tr = p[(p["Date"] <= TRAIN_END) & (p["store_open"] == 1)]
    p["prod_mean"] = key.map(tr.groupby("StockCode")["u_cap"].mean())
    med_price = df[df["Date"] <= TRAIN_END].groupby("StockCode")["UnitPrice"].median()
    bands = pd.qcut(med_price.reindex(products), 5, labels=False, duplicates="drop")
    p["price_band"] = key.map(bands)
    cat_codes = {c: i for i, c in enumerate(sorted(cats.unique()))}
    p["cat_code"] = key.map(cats).map(cat_codes)

    # --- Store-level momentum (all modelled products) ---------------------
    store = p.groupby("Date")["u_cap"].sum(min_count=1).shift(L)
    st = pd.DataFrame({"s": store})
    st["store_r7"] = st["s"].rolling(7, min_periods=1).mean()
    st["store_r28"] = st["s"].rolling(28, min_periods=1).mean()
    st["store_mom"] = st["store_r7"] / (st["store_r28"] + 1)
    p = p.merge(st[["store_r7", "store_r28", "store_mom"]], left_on="Date", right_index=True, how="left")

    # Our price as a daily state (last observed price carried forward) — used
    # only to drive the competitor simulation, never as a baseline feature.
    p["our_price"] = g["price"].transform(lambda x: x.ffill().bfill())
    return p


LAG_FEATS = [f"lag_{L + k}" for k in (0, 7, 14, 21)] + \
            ["rate_7", "rate_14", "rate_28", "rate_56", "rstd_28", "nz_28", "ewm_7",
             "momentum", "days_since_sale", "age", "scale", "rel_lag", "rel_rate_7", "rel_rate_14"]
STATIC_FEATS = ["prod_mean", "price_band", "cat_code"]
STORE_FEATS = ["store_r7", "store_r28", "store_mom"]
FEATURES = CAL_FEATS + LAG_FEATS + STATIC_FEATS + STORE_FEATS   # NB: no price features


# =============================================================================
# 5. BASELINE DEMAND MODELS
# =============================================================================
def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)),
            "WAPE%": 100 * np.abs(err).sum() / max(y.sum(), 1e-9),
            "Bias%": 100 * err.sum() / max(y.sum(), 1e-9)}


def vec(s):
    """pandas Series -> writable contiguous float64 array. pandas 3 (copy-on-write)
    hands out READ-ONLY views, which crash LightGBM's native label setter."""
    return np.array(s.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64, copy=True, order="C")


def mat(df, rf=False):
    """Feature frame -> contiguous float64 matrix (column order = FEATURES).
    Models get plain NumPy: with pyarrow installed, pandas 3 + LightGBM crash
    in native code when given DataFrames. RF cannot take NaN -> fill -1."""
    m = np.array(df[FEATURES].to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64,
                 copy=True, order="C")
    return np.nan_to_num(m, nan=-1.0) if rf else m


def fit_models(tr, va):
    """Fit LightGBM, XGBoost and RF on train; early-stop boosters on val."""
    # Target = units / scale, weighted by scale -> the loss is still in units.
    Xtr, ytr, wtr = mat(tr), vec(tr["u_cap"] / tr["scale"]), vec(tr["scale"])
    Xva, yva, wva = mat(va), vec(va["u_cap"] / va["scale"]), vec(va["scale"])
    models = {}

    # Tweedie objective suits intermittent, zero-heavy counts and predicts the
    # mean directly (no log back-transform bias).
    lgbm = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=4000,
                             learning_rate=0.03, num_leaves=63, min_child_samples=50,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
    # Early-stop on Tweedie deviance (a MEAN-consistent loss). Stopping on MAE
    # pulls forecasts toward the median and under-forecasts units (~-40% bias).
    lgbm.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], eval_sample_weight=[wva],
             callbacks=[lgb.early_stopping(150, verbose=False)])
    models["LightGBM"] = lgbm

    xgbm = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3, n_estimators=4000,
                            learning_rate=0.03, max_depth=7, min_child_weight=20, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                            eval_metric="tweedie-nloglik@1.3", early_stopping_rounds=150,
                            tree_method="hist")
    xgbm.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], sample_weight_eval_set=[wva],
             verbose=False)
    models["XGBoost"] = xgbm

    rf = RandomForestRegressor(n_estimators=250, min_samples_leaf=20, max_features=0.5,
                               max_samples=0.5, n_jobs=-1, random_state=SEED)
    rf.fit(np.nan_to_num(Xtr, nan=-1.0), ytr, sample_weight=wtr)
    models["RandomForest"] = rf
    return models


def predict(models, name, X):
    """Predict daily units; closed-store days are forced to 0."""
    def one(n):
        m = models[n]
        return np.clip(m.predict(mat(X, rf=(n == "RandomForest"))), 0, None)
    if name == "Ensemble":
        pred = np.mean([one(n) for n in ("LightGBM", "XGBoost", "RandomForest")], axis=0)
    else:
        pred = one(name)
    return pred * X["scale"].values * X["store_open"].values


def naive_predictions(X):
    return {"Naive: same day last week": X[f"lag_{L}"].fillna(0).values * X["store_open"].values,
            "Naive: 28-day mean rate": X["rate_28"].fillna(0).values * X["store_open"].values}


def refit_full(models, name, full):
    """Production refit on all history with the validated iteration counts."""
    X, y, w = mat(full), vec(full["u_cap"] / full["scale"]), vec(full["scale"])
    names = ("LightGBM", "XGBoost", "RandomForest") if name == "Ensemble" else (name,)
    out = {}
    for n in names:
        m = models[n]
        if n == "LightGBM":
            r = lgb.LGBMRegressor(**{**m.get_params(), "n_estimators": max(m.best_iteration_, 50)})
            r.fit(X, y, sample_weight=w)
        elif n == "XGBoost":
            params = {**m.get_params(), "n_estimators": max(m.best_iteration + 1, 50),
                      "early_stopping_rounds": None}
            r = xgb.XGBRegressor(**params)
            r.fit(X, y, sample_weight=w, verbose=False)
        else:
            r = RandomForestRegressor(**m.get_params())
            r.fit(np.nan_to_num(X, nan=-1.0), y, sample_weight=w)
        out[n] = r
    return out


# =============================================================================
# 6. OWN-PRICE ELASTICITY FROM REAL DATA + HIERARCHICAL POOLING
# =============================================================================
HAC_LAGS = 14           # Newey-West lags (days) for product-level standard errors
MIN_SALE_DAYS_ELAST = 30


def _daily_price_panel(df, products):
    """Daily product panel (sale days only) on REAL data up to VAL_END.
    Wholesale tier pricing: bulk lines get a lower unit price, so a day with big
    orders shows BOTH a low average price and high units, a spurious, hugely
    negative "elasticity". The day's LIST price is measured only from
    regular-size lines (qty <= product's median line qty, train)."""
    sub = df[(df["Date"] <= VAL_END) & df["StockCode"].isin(products)].copy()
    med_q = sub[sub["Date"] <= TRAIN_END].groupby("StockCode")["Quantity"].median()
    sub["regular"] = sub["Quantity"] <= sub["StockCode"].map(med_q)
    units = sub.groupby(["StockCode", "Date"])["Quantity"].sum().rename("units")
    price = sub[sub["regular"]].groupby(["StockCode", "Date"])["UnitPrice"].median().rename("price")
    d = pd.concat([units, price], axis=1).dropna().reset_index()
    d["week"] = d["Date"].dt.to_period("W-SUN")
    d["y"], d["lp"] = np.log(d["units"]), np.log(d["price"])
    return d


def _two_way_demean(g, iters=30):
    """Remove product fixed effects (level of each item) and date fixed effects
    (seasonality, trend, day-of-week, store-wide shocks) by alternating projections."""
    y, x = g["y"].values.copy(), g["lp"].values.copy()
    keys = [g["StockCode"].values, g["Date"].values]
    for _ in range(iters):
        for k in keys:
            y -= pd.Series(y).groupby(k).transform("mean").values
            x -= pd.Series(x).groupby(k).transform("mean").values
    return y, x


def _hac_se(x, r):
    h = x * r
    s = h @ h
    for lag in range(1, HAC_LAGS + 1):
        s += 2 * (1 - lag / (HAC_LAGS + 1)) * (h[lag:] @ h[:-lag])
    return np.sqrt(max(s, h @ h)) / (x @ x)


def product_elasticities(df, products, cats):
    """
    Daily two-way fixed-effects log-log panel per category on REAL data:
        log(units_it) = a_i + d_t + e * log(price_it) + noise
    Identified purely from WITHIN-item price changes, compared against other
    items of the category on the same day. Returns
      * per-product slopes (e_raw, HAC se_raw) where the item's own price varied
      * per-category pooled slope with product-clustered se
    """
    d = _daily_price_panel(df, products)
    d["cat"] = d["StockCode"].map(cats)
    rows, cat_fit = [], {}
    for cat, g in d.groupby("cat"):
        g = g.reset_index(drop=True)
        y, x = _two_way_demean(g)
        b = (x @ y) / (x @ x)
        score = pd.Series(x * (y - b * x)).groupby(g["StockCode"].values).sum().values
        cat_fit[cat] = (b, np.sqrt((score ** 2).sum()) / (x @ x), g["StockCode"].nunique())
        for code, idx in g.groupby("StockCode").indices.items():
            gi = g.iloc[idx]
            levels = gi["price"].round(2).value_counts()
            rec = {"StockCode": code, "n_weeks": int(gi["week"].nunique()), "n_sale_days": len(idx),
                   "n_price_levels": int((levels >= ELAST_MIN_WEEKS_PER_LEVEL).sum()),
                   "logp_std": float(x[idx].std()), "e_raw": np.nan, "se_raw": np.nan}
            xx, yy = x[idx], y[idx]
            if (len(idx) >= MIN_SALE_DAYS_ELAST and rec["n_price_levels"] >= ELAST_MIN_PRICE_LEVELS
                    and rec["logp_std"] >= ELAST_MIN_LOGP_STD):
                bi = (xx @ yy) / (xx @ xx)
                rec["e_raw"], rec["se_raw"] = bi, max(_hac_se(xx, yy - bi * xx), ELAST_SE_FLOOR)
            rows.append(rec)
    el = pd.DataFrame(rows).set_index("StockCode").reindex(products)
    el[["n_weeks", "n_sale_days", "n_price_levels"]] = el[["n_weeks", "n_sale_days", "n_price_levels"]].fillna(0).astype(int)
    return el, cat_fit


def dersimonian_laird(e, se):
    """Random-effects pooled mean, between-product variance tau^2, and se of mean."""
    w = 1 / se ** 2
    mu_fe = np.sum(w * e) / np.sum(w)
    q = np.sum(w * (e - mu_fe) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max((q - (len(e) - 1)) / c if c > 0 else 0.0, TAU_MIN ** 2)
    w2 = 1 / (se ** 2 + tau2)
    return np.sum(w2 * e) / np.sum(w2), tau2, np.sqrt(1 / np.sum(w2))


def pool_elasticities(el, cats, cat_fit):
    """Three-level empirical Bayes (global -> category -> product).
    Category mean: the category's two-way FE panel slope, shrunk toward the
    global mean (DerSimonian-Laird across categories). Between-product spread
    tau: DerSimonian-Laird on the product slopes (global tau if the category has
    too few). Products with their own price variation are shrunk toward the
    category mean; the rest get the category mean with its full uncertainty."""
    el = el.copy()
    el["category"] = cats.reindex(el.index)
    ok = el["e_raw"].notna()
    cb = np.array([v[0] for v in cat_fit.values()]); cse = np.array([v[1] for v in cat_fit.values()])
    g_mu, g_tau2_cat, g_se = dersimonian_laird(cb, cse)
    _, g_tau2, _ = dersimonian_laird(el.loc[ok, "e_raw"].values, el.loc[ok, "se_raw"].values)
    priors = {}
    for cat in el["category"].unique():
        b, se, _ = cat_fit.get(cat, (g_mu, np.sqrt(g_tau2_cat + g_se ** 2), 0))
        prec = 1 / se ** 2 + 1 / g_tau2_cat
        mu, se_mu = (b / se ** 2 + g_mu / g_tau2_cat) / prec, np.sqrt(1 / prec)
        grp = el[ok & (el["category"] == cat)]
        tau2 = (dersimonian_laird(grp["e_raw"].values, grp["se_raw"].values)[1]
                if len(grp) >= MIN_PRODUCTS_PER_CATEGORY else g_tau2)
        priors[cat] = (mu, tau2, se_mu)
    post_e, post_se, source = [], [], []
    for code, r in el.iterrows():
        mu, tau2, se_mu = priors[r["category"]]
        if pd.notna(r["e_raw"]):
            v_prior = tau2 + se_mu ** 2
            prec = 1 / r["se_raw"] ** 2 + 1 / v_prior
            post_e.append((r["e_raw"] / r["se_raw"] ** 2 + mu / v_prior) / prec)
            post_se.append(np.sqrt(1 / prec))
            source.append("product+pooled")
        else:
            post_e.append(mu)
            post_se.append(np.sqrt(tau2 + se_mu ** 2))
            source.append("category prior only")
    el["elasticity"] = post_e
    el["elasticity_se"] = post_se
    el["elasticity_source"] = source
    el["prior_mean"] = el["category"].map(lambda c: priors[c][0])
    prior_tbl = pd.DataFrame({c: {"panel_slope": cat_fit.get(c, (np.nan,))[0], "prior_mean": v[0],
                                  "prior_se": v[2], "tau": np.sqrt(v[1]),
                                  "n_products": int((el.loc[ok, "category"] == c).sum())}
                              for c, v in priors.items()}).T
    return el, prior_tbl, (g_mu, np.sqrt(g_tau2))


# =============================================================================
# 7. COMPETITOR PRICE — SIMULATED (or real via COMPETITOR_PATH)
# =============================================================================
def simulate_competitor(panel, ref_price):
    """SIMULATED competitor price per product per day:
        comp_t = anchor_t * (1 + offset) * exp(drift_t) * promo_t
      anchor_t : ref^(1-share) * our_price_{t-lag}^share  (lagged partial reaction)
      offset   : per-product U(-15%, +15%)
      drift_t  : mean-reverting AR(1) on log scale
      promo_t  : occasional 10–30% drops lasting a few days
    Seeded per product (SEED + StockCode) so results are reproducible."""
    out = np.empty(len(panel))
    for code, idx in panel.groupby("StockCode", sort=False).indices.items():
        rng = np.random.default_rng([SEED, zlib.crc32(code.encode())])
        our = panel["our_price"].values[idx]
        n = len(our)
        offset = rng.uniform(*COMP_OFFSET_RANGE)
        eps = rng.normal(0, COMP_AR_SIGMA, n)
        drift = np.zeros(n)
        for t in range(1, n):
            drift[t] = COMP_AR_PHI * drift[t - 1] + eps[t]
        promo = np.ones(n)
        t = 0
        while t < n:
            if rng.random() < COMP_PROMO_DAILY_PROB:
                dur = rng.integers(COMP_PROMO_DAYS[0], COMP_PROMO_DAYS[1] + 1)
                promo[t:t + dur] = 1 - rng.uniform(*COMP_PROMO_DEPTH)
                t += dur
            else:
                t += 1
        lagged = np.r_[np.full(min(COMP_REACTION_LAG_DAYS, n), our[0]), our[:-COMP_REACTION_LAG_DAYS]][:n]
        ref = ref_price.get(code, np.median(our))
        anchor = ref ** (1 - COMP_REACTION_SHARE) * lagged ** COMP_REACTION_SHARE
        out[idx] = anchor * (1 + offset) * np.exp(drift) * promo
    return out


def attach_competitor(panel, ref_price):
    panel["comp_price"] = simulate_competitor(panel, ref_price)
    panel["comp_source"] = "SIMULATED"
    if COMPETITOR_PATH:
        real = pd.read_csv(COMPETITOR_PATH, dtype={"StockCode": str}, parse_dates=["Date"])
        real["StockCode"] = real["StockCode"].str.strip().str.upper()
        m = panel[["StockCode", "Date"]].merge(real, on=["StockCode", "Date"], how="left")
        m["CompetitorPrice"] = m.groupby("StockCode")["CompetitorPrice"].ffill()
        has = m["CompetitorPrice"].notna().values
        panel.loc[has, "comp_price"] = m.loc[has, "CompetitorPrice"].values
        panel.loc[has, "comp_source"] = "REAL"
        log(f"Competitor: REAL data used for {has.mean():.0%} of product-days from {COMPETITOR_PATH}")
    return panel


# =============================================================================
# 8. PROFIT OPTIMISATION
# =============================================================================
def price_grid(lo, hi):
    """Candidate prices with realistic endings: 5p steps under £1, typical
    retail endings (.25, .49, .95, .99 ...) above £1."""
    c = set(np.round(np.arange(0.05, 1.0, 0.05), 2))
    for whole in range(1, int(hi) + 2):
        c.update(round(whole + e, 2) for e in PRICE_ENDINGS)
    c.update(float(w) for w in range(1, int(hi) + 2))
    return np.array(sorted(x for x in c if lo - 1e-9 <= x <= hi + 1e-9))


def demand_matrix(prices, p0, base_units, e_draws, comp0, comp_mult):
    """Expected units over the horizon, shape (draws, prices).
    Baseline x (p/p0)^e x competitor effect; competitor partially follows our
    price change only for the part of the horizon beyond its reaction lag."""
    react = COMP_REACTION_SHARE * max(0, HORIZON_DAYS - COMP_REACTION_LAG_DAYS) / HORIZON_DAYS
    comp = comp0 * comp_mult * (prices / p0) ** react
    comp_effect = ((comp / prices) / (comp0 / p0)) ** CROSS_ELASTICITY
    own = (prices[None, :] / p0) ** e_draws[:, None]
    return base_units * own * comp_effect[None, :]


def optimise_product(r, z):
    """Robust profit-maximising price for one product row `r` (a dict)."""
    p0, cost = r["current_price"], r["unit_cost"]
    e_draws = np.minimum(r["elasticity"] + r["elasticity_se"] * z, ELASTICITY_CEILING)
    margin_floor = cost / (1 - MIN_MARGIN)
    lo = max(p0 * (1 - MAX_PRICE_CHANGE), min(r["train_min_price"], p0), margin_floor)
    hi = min(p0 * (1 + MAX_PRICE_CHANGE), max(r["train_max_price"] * (1 + PRICE_HEADROOM), p0))
    flags = []
    if lo > hi:
        flags.append("no feasible price (margin floor above allowed range)")
        cands = np.array([p0])
    else:
        cands = np.unique(np.r_[price_grid(lo, hi), p0 if lo <= p0 <= hi else []])
        if p0 < margin_floor:
            flags.append("current price below margin floor")
    stock = STOCK_CAP.get(r["StockCode"], np.inf)

    out = {}
    for scen, mult in COMP_SCENARIOS.items():
        prices = np.unique(np.r_[cands, p0])
        units = np.minimum(demand_matrix(prices, p0, r["baseline_units"], e_draws,
                                         r["comp_price_SIMULATED"], mult), stock)
        profit = (prices - cost)[None, :] * units
        feasible = np.isin(prices, cands)
        robust = np.where(feasible, np.quantile(profit, ROBUST_QUANTILE, axis=0), -np.inf)
        k = int(np.argmax(robust))
        k0 = int(np.where(np.isclose(prices, p0))[0][0])
        ev = np.where(feasible, profit.mean(0), -np.inf)
        # Expected regret: profit lost vs the price that would have been best had
        # we known the true elasticity (per draw), as a share of expected profit.
        best_draw = np.where(feasible[None, :], profit, -np.inf).max(1)
        regret = (float(np.mean(best_draw - profit[:, k]) / max(profit[:, k].mean(), 1e-9))
                  if feasible.any() else np.nan)
        # Downside risk: expected profit LOST vs keeping the current price
        # (only the draws where the new price does worse), as a share of current profit.
        downside = float(np.mean(np.maximum(profit[:, k0] - profit[:, k], 0)) / max(profit[:, k0].mean(), 1e-9))
        out[scen] = {
            "regret": regret, "downside": downside,
            "price": prices[k], "units": units[:, k].mean(), "profit": profit[:, k].mean(),
            "profit_p10": np.quantile(profit[:, k], 0.10), "profit_p90": np.quantile(profit[:, k], 0.90),
            "profit_current": profit[:, k0].mean(), "units_current": units[:, k0].mean(),
            "p_improve": float((profit[:, k] > profit[:, k0]).mean()) if k != k0 else np.nan,
            "ev_price": prices[int(np.argmax(ev))],
            "at_bound": len(cands) > 1 and (np.isclose(prices[k], cands.min()) or np.isclose(prices[k], cands.max())),
        }
    out["flags"] = flags
    out["bounds"] = (lo, hi)
    return out


RISK_HIGH = 0.02       # expected profit lost vs current price <= 2% -> high
RISK_MEDIUM = 0.05     # <= 5% -> medium


def confidence(r):
    """Decision-based confidence: how safe is acting on this price, given
    everything we do NOT know about elasticity? Uses downside risk (expected
    profit lost vs keeping the current price) and, when the price changes,
    the probability the change beats the current price. Regret (upside that
    may be missed) is reported but does not lower confidence."""
    hard = []
    if r["baseline_units"] < 5:
        hard.append("very low forecast volume")
    if r["flags"]:
        hard.append(r["flags"])
    if r["elasticity_source"] == "product+pooled" and pd.notna(r["e_raw"]) and r["e_raw"] > 0:
        hard.append("item's own data shows demand rising with price (likely confounded)")
    risk, pi = r["downside"], r["p_improve"]
    if hard:
        return "low", "; ".join(hard)
    untested = r.get("recommended_price", 0) > max(r["train_max_price"], r["current_price"]) + 1e-9
    if risk <= RISK_HIGH and (pd.isna(pi) or pi >= 0.8) and not untested:
        return "high", ""
    notes = []
    if untested:
        notes.append(f"price is above the highest ever charged (£{r['train_max_price']:.2f}), so test it before rolling out")
    if r["elasticity_source"] != "product+pooled":
        notes.append("price sensitivity taken from the category (item never changed price)")
    if risk > RISK_HIGH:
        notes.append(f"{risk:.1%} expected profit at risk vs current price")
    if pd.notna(pi) and pi < 0.8:
        notes.append(f"price change beats current price in {pi:.0%} of scenarios")
    if risk <= RISK_MEDIUM and (pd.isna(pi) or pi >= 0.6):
        return "medium", "; ".join(notes)
    return "low", "; ".join(notes)


# =============================================================================
# 9. PLOTS
# =============================================================================
def plot_forecast(res_tbl, te, best, pred_best):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    t = res_tbl.reset_index()
    xs = np.arange(len(t))
    ax[0].bar(xs - 0.2, t["val_WAPE%_7day"], 0.4, label="validation", color="#4C72B0")
    ax[0].bar(xs + 0.2, t["test_WAPE%_7day"], 0.4, label="test", color="#DD8452")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(t["model"], rotation=30, ha="right")
    ax[0].set_ylabel("WAPE % of 7-day product totals (lower is better)")
    ax[0].set_title(f"Forecast error ({L}-day-ahead) — selected on validation: {best}")
    ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
    d = te.assign(pred=pred_best, naive=te[f"lag_{L}"].fillna(0) * te["store_open"]) \
          .groupby("Date")[["u_cap", "pred", "naive"]].sum()
    ax[1].plot(d.index, d["u_cap"], color="black", lw=2, label="actual")
    ax[1].plot(d.index, d["pred"], color="#DD8452", lw=2, label=best)
    ax[1].plot(d.index, d["naive"], color="gray", lw=1, ls="--", label="naive (last week)")
    ax[1].set_title("Test period: total daily units, modelled products")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "forecast_accuracy.png", dpi=130); plt.close(fig)


def plot_elasticity(el):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    raw = el["e_raw"].dropna()
    bins = np.linspace(-6, 3, 61)
    ax[0].hist(raw.clip(-6, 3), bins=bins, alpha=0.5, label=f"raw product OLS (n={len(raw)})", color="gray")
    ax[0].hist(el["elasticity"].clip(-6, 3), bins=bins, alpha=0.7, label=f"pooled posterior (n={len(el)})", color="#4C72B0")
    ax[0].axvline(-1, color="orange", ls=":", label="unit elastic")
    ax[0].set_title("Own-price elasticity (REAL data)"); ax[0].set_xlabel("elasticity"); ax[0].legend()
    ok = el["e_raw"].notna()
    sc = ax[1].scatter(el.loc[ok, "e_raw"].clip(-8, 4), el.loc[ok, "elasticity"],
                       c=el.loc[ok, "se_raw"].clip(0, 3), cmap="viridis", s=14)
    ax[1].plot([-8, 4], [-8, 4], "r--", lw=1)
    plt.colorbar(sc, ax=ax[1], label="raw standard error")
    ax[1].set_xlabel("raw estimate"); ax[1].set_ylabel("pooled estimate")
    ax[1].set_title("Partial pooling: noisy estimates shrink to category prior")
    cats = el.groupby("category")["elasticity"].median().sort_values().index
    ax[2].boxplot([el.loc[el["category"] == c, "elasticity"] for c in cats], vert=False, tick_labels=list(cats))
    ax[2].axvline(-1, color="orange", ls=":")
    ax[2].set_title("Pooled elasticity by category")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "elasticity_distribution.png", dpi=130); plt.close(fig)


def plot_profit_curves(rec, z, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for ax, (_, r) in zip(axes.ravel(), rec.head(n).iterrows()):
        p0 = r["current_price"]
        grid = np.linspace(p0 * 0.7, p0 * 1.3, 200)
        e_draws = np.minimum(r["elasticity"] + r["elasticity_se"] * z, ELASTICITY_CEILING)
        units = demand_matrix(grid, p0, r["baseline_units"], e_draws, r["comp_price_SIMULATED"], 1.0)
        prof = (grid - r["unit_cost"])[None, :] * units
        ax.fill_between(grid, np.quantile(prof, 0.1, 0), np.quantile(prof, 0.9, 0), alpha=0.2,
                        color="#4C72B0", label="P10–P90 (elasticity uncertainty)")
        ax.plot(grid, prof.mean(0), color="#4C72B0", lw=2, label="expected profit")
        ax.plot(grid, np.quantile(prof, ROBUST_QUANTILE, 0), color="#4C72B0", ls="--", lw=1,
                label=f"P{int(ROBUST_QUANTILE * 100)} (robust objective)")
        ax.axvspan(r["bound_lo"], r["bound_hi"], color="green", alpha=0.06, label="allowed range")
        ax.axvline(p0, color="gray", ls="--", label=f"current £{p0:.2f}")
        ax.axvline(r["recommended_price"], color="red", lw=2, label=f"recommended £{r['recommended_price']:.2f}")
        ax.set_title(f"{r['StockCode']} {r['Description'][:30]}\n"
                     f"e={r['elasticity']:.2f}±{r['elasticity_se']:.2f}, conf={r['confidence']}", fontsize=9)
        ax.set_xlabel("price £"); ax.set_ylabel(f"profit £ over {HORIZON_DAYS} days")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Profit vs price (competitor effect uses SIMULATED competitor prices)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "profit_vs_price_curves.png", dpi=130); plt.close(fig)


def plot_scenarios(rec, panel, n=15):
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    top = rec.head(n)
    xs = np.arange(len(top))
    colors = {"comp_minus10": "#C44E52", "comp_same": "#4C72B0", "comp_plus10": "#55A868"}
    for i, scen in enumerate(COMP_SCENARIOS):
        chg = 100 * (top[f"{scen}_price"] / top["current_price"] - 1)
        ax[0].bar(xs + (i - 1) * 0.27, chg, 0.27, label=scen, color=colors[scen])
    ax[0].set_xticks(xs); ax[0].set_xticklabels(top["StockCode"], rotation=60, fontsize=8)
    ax[0].axhline(0, color="black", lw=0.8)
    ax[0].set_ylabel("recommended price change vs current (%)")
    ax[0].set_title("Optimal price by competitor scenario (SIMULATED)"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
    tot_cur = [rec[f"{s}_profit_at_current"].sum() for s in COMP_SCENARIOS]
    tot_opt = [rec[f"{s}_profit"].sum() for s in COMP_SCENARIOS]
    xs2 = np.arange(len(COMP_SCENARIOS))
    ax[1].bar(xs2 - 0.2, tot_cur, 0.4, label="keep current prices", color="gray")
    ax[1].bar(xs2 + 0.2, tot_opt, 0.4, label="re-optimised prices", color="#4C72B0")
    ax[1].set_xticks(xs2); ax[1].set_xticklabels(list(COMP_SCENARIOS))
    ax[1].set_ylabel(f"expected profit £, top {len(rec)} products, {HORIZON_DAYS} days")
    ax[1].set_title("Total profit by competitor scenario (SIMULATED)"); ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    for code in rec["StockCode"].head(3):
        g = panel[(panel["StockCode"] == code) & ~panel["is_future"]]
        line, = ax[2].plot(g["Date"], g["our_price"], lw=2, label=f"{code} our price")
        ax[2].plot(g["Date"], g["comp_price"], lw=1, ls="--", color=line.get_color(),
                   label=f"{code} competitor (SIMULATED)")
    ax[2].set_title("SIMULATED competitor price paths (offset, drift, promos, lagged reaction)")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "competitor_scenarios.png", dpi=130); plt.close(fig)


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================
def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


def main():
    df = clean(load_transactions())
    data_end = df["Date"].max()
    log(f"Data range {df['Date'].min().date()} -> {data_end.date()}; "
        f"train <= {TRAIN_END.date()}, val <= {VAL_END.date()}, test <= {data_end.date()}")

    desc = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    cats = desc.map(categorise)

    # --- Product selection on TRAIN only (no peeking at test performance) ----
    tr_lines = df[df["Date"] <= TRAIN_END]
    sale_days = tr_lines.groupby("StockCode")["Date"].nunique()
    tr_rev = tr_lines.groupby("StockCode")["Revenue"].sum()
    eligible = sale_days[sale_days >= MIN_TRAIN_SALE_DAYS].index
    products = tr_rev.reindex(eligible).sort_values(ascending=False).head(TOP_N_MODEL).index.tolist()
    log(f"Modelling {len(products)} products (top by train revenue, >= {MIN_TRAIN_SALE_DAYS} sale days)")

    # ---------------- STAGE 1a: baseline demand -------------------------------
    panel = build_panel(df, products, cats)
    hist = panel[~panel["is_future"] & (panel["age"] >= L)]
    tr = hist[(hist["Date"] <= TRAIN_END) & (hist["store_open"] == 1)]
    va = hist[(hist["Date"] > TRAIN_END) & (hist["Date"] <= VAL_END)]
    te = hist[hist["Date"] > VAL_END]
    assert tr["Date"].max() < va["Date"].min() and va["Date"].max() < te["Date"].min()
    log(f"Rows  train={len(tr):,}  val={len(va):,}  test={len(te):,}  features={len(FEATURES)}")

    models = fit_models(tr, va[va["store_open"] == 1])
    log("Models fitted: " + ", ".join(f"{k}" for k in models))

    rows, preds, calib = [], {}, {}
    for name in ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]:
        pv = predict(models, name, va)
        # Level calibration fitted on VALIDATION only (never on test): demand
        # grows into Q4 and models trained on Dec–Aug under-forecast the level.
        calib[name] = float(np.clip(va["u_cap"].sum() / max(pv.sum(), 1e-9), 0.8, 1.5))
        preds[name] = (pv * calib[name], predict(models, name, te) * calib[name])
    log("Validation level calibration factors: " + ", ".join(f"{k}={v:.3f}" for k, v in calib.items()))
    nv, nt = naive_predictions(va), naive_predictions(te)
    for name in nv:
        preds[name] = (nv[name], nt[name])
    def weekly_wape(frame, pred):
        """WAPE of 7-day product totals: the quantity the pricing decision uses."""
        w = frame.assign(pred=pred, wk=frame["Date"].dt.to_period("W-SUN")) \
                 .groupby(["StockCode", "wk"])[["u_cap", "pred"]].sum()
        return metrics(w["u_cap"], w["pred"])["WAPE%"]

    for name, (pv, pt) in preds.items():
        mv, mt = metrics(va["u_cap"], pv), metrics(te["u_cap"], pt)
        raw_t = metrics(te["units"], pt)
        rows.append({"model": name, **{f"val_{k}": v for k, v in mv.items()},
                     **{f"test_{k}": v for k, v in mt.items()}, "test_WAPE%_uncapped": raw_t["WAPE%"],
                     "val_WAPE%_7day": weekly_wape(va, pv), "test_WAPE%_7day": weekly_wape(te, pt)})
    res_tbl = pd.DataFrame(rows).set_index("model")
    ml_names = ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]
    # Chosen on VALIDATION only, on the 7-day product totals the optimiser consumes
    best = res_tbl.loc[ml_names, "val_WAPE%_7day"].idxmin()
    best_naive = res_tbl.drop(ml_names)["test_WAPE%_7day"].min()
    log(f"Selected baseline model on validation: {best}")
    print(res_tbl.round(2).to_string())
    res_tbl.round(4).to_csv(OUTPUT_DIR / "model_metrics.csv")
    plot_forecast(res_tbl, te, best, preds[best][1])

    # Production refit on all history, then forecast the next HORIZON_DAYS
    final = refit_full(models, best, hist[hist["store_open"] == 1])
    fut = panel[panel["is_future"]]
    fut_pred = predict(final, best, fut) * calib[best]
    baseline_units = pd.Series(fut_pred, index=fut.index).groupby(fut["StockCode"]).sum()
    log(f"Baseline forecast {fut['Date'].min().date()} -> {fut['Date'].max().date()}: "
        f"{baseline_units.sum():,.0f} units across {len(baseline_units)} products")

    # ---------------- STAGE 1b: own-price elasticity (real data) --------------
    el, cat_fit = product_elasticities(df, products, cats)
    el, prior_tbl, (g_mu, g_tau) = pool_elasticities(el, cats, cat_fit)
    n_prod_est = int(el["e_raw"].notna().sum())
    log(f"Elasticity: {n_prod_est}/{len(el)} products have usable price variation; "
        f"global prior {g_mu:.2f} (tau {g_tau:.2f}); pooled median {el['elasticity'].median():.2f}")
    plot_elasticity(el)

    # ---------------- STAGE 1c: competitor (SIMULATED) ------------------------
    tr_price = tr_lines.groupby("StockCode")["UnitPrice"]
    med_train_price = tr_price.median()
    panel = attach_competitor(panel, med_train_price.to_dict())
    last28 = panel[~panel["is_future"] & (panel["Date"] > data_end - pd.Timedelta(days=28))]
    comp0 = last28.groupby("StockCode")["comp_price"].median()

    # ---------------- STAGE 2: profit optimisation ----------------------------
    recent = df[df["Date"] > data_end - pd.Timedelta(days=28)]
    p_recent = recent.groupby("StockCode")["UnitPrice"].median()
    p_last = df.sort_values("Date").groupby("StockCode")["UnitPrice"].last()
    # Observed price range over the FULL history (the current price is recent,
    # so a train-only range wrongly excludes the latest, higher prices).
    daily_train_price = df.groupby(["StockCode", "Date"])["UnitPrice"].median()
    tr_min = daily_train_price.groupby("StockCode").min()
    tr_max = daily_train_price.groupby("StockCode").max()
    total_rev = df.groupby("StockCode")["Revenue"].sum()

    z = np.random.default_rng(SEED).standard_normal(N_ELASTICITY_DRAWS)   # common random numbers
    recs = []
    for code in products:
        cat = cats[code]
        r = {"StockCode": code, "Description": desc[code], "category": cat,
             "total_revenue": total_rev[code],
             "current_price": round(float(p_recent.get(code, p_last[code])), 2),
             "train_min_price": tr_min[code], "train_max_price": tr_max[code],
             "unit_cost": COST_OVERRIDES_PRODUCT.get(
                 code, COST_RATIO_BY_CATEGORY.get(cat, COST_RATIO_DEFAULT) * med_train_price[code]),
             "baseline_units": float(baseline_units.get(code, 0.0)),
             "comp_price_SIMULATED": float(comp0[code]),
             **el.loc[code, ["e_raw", "se_raw", "n_weeks", "n_price_levels", "elasticity",
                             "elasticity_se", "elasticity_source"]].to_dict()}
        o = optimise_product(r, z)
        base = o["comp_same"]
        r.update({"recommended_price": base["price"], "price_change_%": 100 * (base["price"] / r["current_price"] - 1),
                  "expected_units": base["units"], "expected_profit": base["profit"],
                  "profit_p10": base["profit_p10"], "profit_p90": base["profit_p90"],
                  "units_at_current": base["units_current"], "profit_at_current": base["profit_current"],
                  "profit_change": base["profit"] - base["profit_current"],
                  "p_improve": base["p_improve"], "regret": base["regret"], "downside": base["downside"], "expected_value_price": base["ev_price"],
                  "at_bound": base["at_bound"], "flags": "; ".join(o["flags"]),
                  "bound_lo": o["bounds"][0], "bound_hi": o["bounds"][1]})
        for scen in COMP_SCENARIOS:
            s = o[scen]
            r.update({f"{scen}_price": s["price"], f"{scen}_units": s["units"], f"{scen}_profit": s["profit"],
                      f"{scen}_profit_at_current": s["profit_current"]})
        r["confidence"], r["confidence_reason"] = confidence(r)   # r has recommended_price here
        recs.append(r)
    rec = pd.DataFrame(recs).sort_values("total_revenue", ascending=False).reset_index(drop=True)
    top = rec.head(TOP_N_REPORT).copy()

    csv_cols = (["StockCode", "Description", "category", "total_revenue", "current_price", "recommended_price",
                 "price_change_%", "expected_units", "expected_profit", "profit_p10", "profit_p90",
                 "units_at_current", "profit_at_current", "profit_change", "p_improve", "downside", "regret",
                 "elasticity", "elasticity_se", "e_raw", "n_weeks", "n_price_levels", "elasticity_source",
                 "confidence", "confidence_reason", "unit_cost", "baseline_units", "comp_price_SIMULATED"]
                + [f"{s}_{k}" for s in COMP_SCENARIOS for k in ("price", "units", "profit")])
    top[csv_cols].round(4).to_csv(OUTPUT_DIR / "pricing_recommendations.csv", index=False)
    rec[csv_cols].round(4).to_csv(OUTPUT_DIR / "pricing_recommendations_all_products.csv", index=False)
    el.round(4).to_csv(OUTPUT_DIR / "elasticity_estimates.csv")
    plot_profit_curves(top, z)
    plot_scenarios(top, panel)
    log(f"Wrote recommendations for top {len(top)} products (and all {len(rec)}) to {OUTPUT_DIR}")

    # ---------------- summary.md ----------------------------------------------
    mt = res_tbl.reset_index()[["model", "val_WAPE%_7day", "test_WAPE%_7day", "val_WAPE%", "test_WAPE%",
                                "test_MAE", "test_RMSE", "test_Bias%", "test_WAPE%_uncapped"]]
    conf_counts = top["confidence"].value_counts().reindex(["high", "medium", "low"]).fillna(0).astype(int)
    scen_tbl = pd.DataFrame([{"scenario (SIMULATED competitor)": s,
                              "profit @ current prices £": top[f"{s}_profit_at_current"].sum(),
                              "profit @ optimised prices £": top[f"{s}_profit"].sum(),
                              "mean price change %": (100 * (top[f"{s}_price"] / top["current_price"] - 1)).mean()}
                             for s in COMP_SCENARIOS])
    show = top.sort_values("profit_change", ascending=False).head(15)[
        ["StockCode", "Description", "current_price", "recommended_price", "expected_units",
         "expected_profit", "profit_change", "elasticity", "elasticity_se", "confidence"]]
    show["Description"] = show["Description"].str[:32]
    el_pooled = el["elasticity"]
    best_row = res_tbl.loc[best]
    lift = 100 * (best_naive - best_row["test_WAPE%_7day"]) / best_naive

    summary = f"""# Profit-maximising pricing: UCI Online Retail

Generated by `pricing_profit_optimizer.py`. Data {df['Date'].min().date()} to {data_end.date()}.
Decision horizon is the {HORIZON_DAYS} days after the data ends ({fut['Date'].min().date()} to {fut['Date'].max().date()}).
The objective is **profit = (price − unit cost) × expected units**.

> **SIMULATED data warning:** the competitor prices are **SIMULATED** (seeded, synthetic). Every competitor effect,
> every `comp_*` scenario column and the competitor plots depend on them. Unit costs are an **ASSUMPTION**
> ({COST_RATIO_DEFAULT:.0%} of each product's median train price). Treat absolute profit figures as illustrative.

## 1. Baseline demand model
- {len(products)} products (top by train revenue), each reindexed to a full daily calendar. Forecasts are {L}-day-ahead
  (all lag/rolling features are at least {L} days old) and **no price features are used**, so price effects come only
  from the elasticity stage.
- Split by date: train ≤ {TRAIN_END.date()}, validation ≤ {VAL_END.date()}, test ≤ {data_end.date()}. Demand caps
  (per-product {DEMAND_CAP_Q:.1%} quantile), product statistics and price bands are computed on train only.
- Targets are scaled by each product's recent selling rate, and a level-calibration factor fitted on validation
  ({calib[best]:.3f} for {best}) corrects the Q4 growth the trees cannot extrapolate. Because this factor is fitted on
  validation, validation scores are slightly optimistic; test scores are not affected.
- **Selected on validation 7-day WAPE: {best}.** On test, its 7-day product-total WAPE is {best_row['test_WAPE%_7day']:.1f}%, against
  {best_naive:.1f}% for the best naive baseline ({lift:+.1f}% relative improvement). Its test bias is {best_row['test_Bias%']:+.1f}%.

{md_table(mt)}

(WAPE = Σ|error| / Σ actual. `_7day` scores the weekly product totals that pricing uses; the other columns score
daily product-level units. The metrics use train-based capped actuals; the last column scores against raw uncapped
actuals.)

## 2. Own-price elasticity (REAL data)
- Daily two-way fixed-effects log-log panel per category (item effects + date effects) on real prices and units up to
  {VAL_END.date()}: elasticity is identified only from within-item price changes, compared with other items on the same day.
  Item-level slopes use Newey-West (HAC) standard errors; category slopes use item-clustered errors.
- {n_prod_est} of {len(el)} products had enough real price variation for a product-level estimate. The rest use their
  category prior.
- Confidence is decision-based: **high** if downside risk (expected profit lost vs keeping the current price, over the
  elasticity draws) is ≤ {RISK_HIGH:.0%} and any price change beats the current price in ≥ 80% of draws; **medium** if risk ≤ {RISK_MEDIUM:.0%}.
  Prices above the highest ever charged are capped at medium.
- The estimates are partially pooled (empirical Bayes, global → category → item). The global prior
  is {g_mu:.2f} (between-product sd {g_tau:.2f}).
- Pooled elasticity: median {el_pooled.median():.2f}, IQR [{el_pooled.quantile(.25):.2f}, {el_pooled.quantile(.75):.2f}]. Share elastic (< −1):
  {100 * (el_pooled < -1).mean():.0f}%. Share of raw estimates with the wrong sign (> 0): {100 * (el['e_raw'] > 0).mean() / max(el['e_raw'].notna().mean(), 1e-9):.0f}%.

{md_table(prior_tbl.reset_index().rename(columns={'index': 'category'}).sort_values('prior_mean'))}

## 3. Top pricing recommendations (largest expected profit gain, among top {TOP_N_REPORT} by revenue)
Confidence across the top {TOP_N_REPORT}: high {conf_counts['high']}, medium {conf_counts['medium']}, low {conf_counts['low']}.
Expected profit over the horizon across the top {TOP_N_REPORT}: £{top['profit_at_current'].sum():,.0f} at current prices,
£{top['expected_profit'].sum():,.0f} at recommended prices (Δ £{top['profit_change'].sum():,.0f}, competitor held at the same level).

{md_table(show)}

Each price is chosen to maximise the **P{int(ROBUST_QUANTILE * 100)} profit** across {N_ELASTICITY_DRAWS} elasticity draws, so it is robust
to elasticity uncertainty. The constraints are: within the observed price range (plus up to {PRICE_HEADROOM:.0%} above the highest price ever charged, flagged as untested), at most ±{MAX_PRICE_CHANGE:.0%} vs the current price, a margin of
at least {MIN_MARGIN:.0%}, the optional stock cap, and realistic price endings.

## 4. Competitor scenarios (SIMULATED competitor)
{md_table(scen_tbl)}

The competitor effect is `((comp/p)/(comp0/p0))^{CROSS_ELASTICITY}`. Within the price range it adds −{CROSS_ELASTICITY} to the effective
own-price elasticity. A competitor **level** shift (±10%) multiplies demand by the same factor at every price,
so with a constant-elasticity demand curve it changes **profit** (shown above) but **not the optimal price**. Only a binding
stock cap, or a competitor reacting to our price inside the horizon, can move the optimum. This is a property of the model,
not a bug. With real competitor data, a non-constant cross effect (e.g. a threshold when we are undercut) could be fitted.

## 5. What is simulated or assumed
| Item | Status |
|---|---|
| Sales, prices, baseline forecast, elasticity | REAL data |
| Competitor prices, cross-elasticity ({CROSS_ELASTICITY}), all `comp_*` scenario columns | **SIMULATED** / assumed |
| Unit cost ({COST_RATIO_DEFAULT:.0%} of median train price) | **ASSUMPTION** |
| Stock caps | none set (configure `STOCK_CAP`) |

## 6. Limitations
- **Price endogeneity:** price changes in this data are not experiments. Wholesale tier pricing (bulk lines priced lower),
  list-price changes timed with seasons, and customer mix can all bias elasticities. The weekly median price and the
  seasonality control reduce this bias but do not remove it. Pricing A/B tests would give causal estimates.
- Many products have little real price variation, so their elasticity relies on the category prior (see `elasticity_source`).
- The elasticity is a constant-elasticity curve, applied only within ±{MAX_PRICE_CHANGE:.0%} of current prices, where it is a reasonable local approximation.
- Baseline forecast uncertainty scales profit at every candidate price equally, so it does not move the optimal price.
  It does widen the absolute profit range, and that is not included in P10/P90.
- The decision horizon is the {HORIZON_DAYS} days before Christmas 2011, an unusual period. Re-run with fresh data before acting.
- The competitor stage is synthetic: **do not act on the scenario numbers** until `COMPETITOR_PATH` points to real data.
- Costs are assumed. Replace them with real landed costs via `COST_OVERRIDES_PRODUCT` / `COST_RATIO_BY_CATEGORY`.

## Output files
`pricing_recommendations.csv` (top {TOP_N_REPORT}), `pricing_recommendations_all_products.csv`, `elasticity_estimates.csv`,
`model_metrics.csv`, `forecast_accuracy.png`, `elasticity_distribution.png`, `profit_vs_price_curves.png`,
`competitor_scenarios.png`.
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    # Everything the web app needs, so it never retrains on a request.
    pd.to_pickle({"products": rec, "horizon_start": fut["Date"].min(), "horizon_end": fut["Date"].max(),
                  "data_end": data_end, "model": best, "test_wape_7day": float(best_row["test_WAPE%_7day"]),
                  "z": z}, ARTIFACTS_PATH)
    log("summary.md written. PIPELINE COMPLETE.")
    print(show.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
