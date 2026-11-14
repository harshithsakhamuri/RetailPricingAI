"""
=============================================================================
RETAIL PROFIT-MAXIMISING PRICING  —  UCI Online Retail
=============================================================================
  STAGE 1  DEMAND
      units(p) = baseline forecast x (p / p_current) ^ elasticity x competitor level effect
        Baseline     7-day-ahead forecast. Candidates: LightGBM, XGBoost, Random Forest,
                     ensemble AND two naive baselines, all given the same level
                     calibration; selected on held-out validation weeks. No price
                     features, so price effects come only from the elasticity.
        Elasticity   Level set by an event study of the shop's real list-price changes;
                     relative differences from a Poisson (PPML) two-way fixed-effects panel per category on the
                     full trading calendar (zero-sale days included), item + date
                     effects, list price from regular-size lines. Item slopes are
                     partially pooled (empirical Bayes); wrong-signed items fall
                     back to the category. Checked with a placebo test and an
                     out-of-sample backtest on later real price changes.
        Competitor   SIMULATED competitor (seeded). Enters only as a demand shift
                     for a competitor price level away from the reference.
  STAGE 2  DECISION  (src/pricing_core.py, shared with the apps)
      Price maximising EXPECTED profit over elasticity draws inside business
      limits; cross-fitted gains; Lerner diagnostics; decision driver; jointly
      priced colour/design variants; price-test plans.

Run:  python src/pricing_profit_optimizer.py [--as-of YYYY-MM-DD]
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import zlib
from pathlib import Path

# LightGBM MUST be imported before pandas on Windows: pandas loads pyarrow, whose
# bundled native runtime makes LightGBM crash ("access violation") if it loads first.
import lightgbm as lgb  # noqa: E402  (import order is deliberate)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import holidays
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import pricing_core as core

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.environ.get("RETAIL_DATA_PATH", ROOT / "data" / "Online Retail.xlsx"))
OUTPUT_DIR = Path(os.environ.get("PRICING_OUTPUT_DIR", ROOT / "outputs"))
CACHE_NAME = "_cache_online_retail.pkl"
COMPETITOR_PATH = os.environ.get("COMPETITOR_PATH") or None
SEED = 42

# Data
INCLUDE_GUEST_LINES = False  # lines without CustomerID (~25%, mostly retail/guest buyers)

# Splits, relative to the last data day (defaults reproduce 2011-08-31 / 2011-10-31)
TEST_DAYS = 39
VAL_DAYS = 61

# Forecasting
FORECAST_LEAD = 7
HORIZON_DAYS = FORECAST_LEAD   # the pricing decision covers the next 7 days
DAILY_HORIZON_DAYS = 30        # single-day forecasts up to a month ahead (recursive beyond 7 days)
TOP_N_MODEL = 500
MIN_TRAIN_SALE_DAYS = 30
TOP_N_REPORT = 50
DEMAND_CAP_Q = 0.995
CALIB_CLIP = (0.7, 1.5)

# Elasticity
ELAST_MIN_WEEKS = 8            # weeks with sales needed for an item-level estimate
ELAST_MIN_PRICE_LEVELS = 2     # distinct weekly list prices ...
ELAST_MIN_WEEKS_PER_LEVEL = 3  # ... each held for at least this many WEEKS
ELAST_MIN_LOGP_STD = 0.01
ELAST_SE_FLOOR = 0.05
HAC_LAGS = 14
TAU_MIN = 0.10
MIN_CLUSTERS_CATEGORY = 10     # items with own price variation needed to trust a category slope
MIN_PRODUCTS_TAU = 4
N_PLACEBO = 20
LEVEL_SCALE_CLIP = (1.0, 4.0)  # allowed correction of the panel's attenuation (event level / panel level)
EVENT_WINDOW_WEEKS = 2         # weeks either side of a real list-price change in the event study

# Competitor simulation (SIMULATED unless COMPETITOR_PATH is given)
COMP_OFFSET_RANGE = (-0.15, 0.15)
COMP_AR_PHI, COMP_AR_SIGMA = 0.90, 0.02
COMP_PROMO_DAILY_PROB, COMP_PROMO_DEPTH, COMP_PROMO_DAYS = 0.02, (0.10, 0.30), (3, 7)
COMP_REACTION_LAG_DAYS, COMP_REACTION_SHARE = 7, 0.5

# Costs (no cost data exists -> assumption)
COST_RATIO_DEFAULT = 0.60
COST_RATIO_BY_CATEGORY: dict = {}
COST_OVERRIDES_PRODUCT: dict = {}
STOCK_CAP: dict = {}           # units available over the horizon, e.g. {"22423": 300}

# Uncertainty
N_ELASTICITY_DRAWS = 500

L = FORECAST_LEAD

COLOUR_WORDS = {"RED", "BLUE", "PINK", "GREEN", "WHITE", "BLACK", "IVORY", "CREAM", "YELLOW", "PURPLE",
                "ORANGE", "GREY", "GOLD", "SILVER", "ROSE", "ROSES", "BROWN", "NATURAL", "PASTEL",
                "LILAC", "TURQUOISE", "AQUA", "MINT", "BLOSSOM", "SPOTTY", "POLKADOT", "SKULL", "SKULLS",
                "RETROSPOT", "VINTAGE", "SMALL", "LARGE", "MEDIUM", "MINI", "JUMBO", "ASSORTED", "COLOUR",
                "COLOURS", "DESIGN", "DESIGNS"}


# =============================================================================
# 1. LOAD, NET CANCELLATIONS, CLEAN
# =============================================================================
def load_transactions(data_path: Path = DATA_PATH, cache_path: Path | None = None) -> pd.DataFrame:
    """Read the xlsx once and cache a pickle. Works from the cache alone."""
    cache_path = cache_path or (OUTPUT_DIR / CACHE_NAME)
    if cache_path.exists() and (not data_path.exists() or cache_path.stat().st_mtime > data_path.stat().st_mtime):
        return pd.read_pickle(cache_path)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found at {data_path}. Download 'Online Retail.xlsx' from "
                         "https://archive.ics.uci.edu/dataset/352/online+retail (see data/README.md).")
    log(f"Reading {data_path} (first run only, cached afterwards)...")
    raw = pd.read_excel(data_path, dtype={"InvoiceNo": str, "StockCode": str})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(cache_path)
    return raw


def net_cancellations(raw: pd.DataFrame):
    """Subtract each cancellation from the same customer's EARLIER orders of the
    same item (most recent first). Returns (order lines with netted quantities,
    stats). Cancellations of orders placed before the data starts cannot be
    matched and are ignored."""
    df = raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    is_c = df["InvoiceNo"].str.startswith("C")
    orders = df[~is_c & (df["Quantity"] > 0)].sort_values("InvoiceDate").copy()
    canc = df[is_c & df["CustomerID"].notna()].sort_values("InvoiceDate")
    qty = orders["Quantity"].to_numpy(dtype=float).copy()
    dates = orders["InvoiceDate"].to_numpy()
    pos = {k: np.asarray(v) for k, v in orders.reset_index(drop=True)
           .groupby(["CustomerID", "StockCode"], sort=False).indices.items()}
    matched = unmatched = 0.0
    for cust, code, q, d in zip(canc["CustomerID"], canc["StockCode"], -canc["Quantity"].to_numpy(float),
                                canc["InvoiceDate"].to_numpy()):
        idx = pos.get((cust, code))
        left = q
        if idx is not None:
            for i in idx[::-1]:
                if left <= 0:
                    break
                if dates[i] > d or qty[i] <= 0:
                    continue
                take = min(qty[i], left)
                qty[i] -= take
                left -= take
        matched += q - left
        unmatched += left
    orders["Quantity"] = qty
    stats = {"cancel_lines": int(is_c.sum()), "cancel_units_matched": float(matched),
             "cancel_units_unmatched": float(unmatched),
             "order_lines_fully_cancelled": int((qty <= 0).sum())}
    return orders[orders["Quantity"] > 0], stats


def clean(raw: pd.DataFrame, include_guests: bool = INCLUDE_GUEST_LINES):
    n0 = len(raw)
    df, stats = net_cancellations(raw)
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df = df[df["UnitPrice"] > 0]
    df = df[df["StockCode"].str.match(r"^\d{5}")]                 # drop POST, M, DOT, fees...
    no_cust = df["CustomerID"].isna()
    rev = df["Quantity"] * df["UnitPrice"]
    stats.update(guest_line_share=float(no_cust.mean()), guest_revenue_share=float(rev[no_cust].sum() / rev.sum()),
                 guests_included=bool(include_guests))
    if not include_guests:
        df = df[~no_cust]
    med = df.groupby("StockCode")["UnitPrice"].transform("median")   # price typos
    df = df[(df["UnitPrice"] <= 5 * med) & (df["UnitPrice"] >= 0.2 * med)].copy()
    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    stats.update(raw_lines=n0, clean_lines=len(df))
    log(f"Cleaned: {len(df):,} of {n0:,} raw lines; {stats['cancel_units_matched']:,.0f} cancelled units netted; "
        f"lines without CustomerID {'kept' if include_guests else 'dropped'} "
        f"({stats['guest_line_share']:.0%} of lines, {stats['guest_revenue_share']:.0%} of revenue)")
    return df, stats


# =============================================================================
# 2. CATEGORIES AND VARIANT GROUPS
# =============================================================================
CATEGORY_RULES = [
    ("christmas", ["CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER", "SNOWFLAKE", "BAUBLE"]),
    ("bags", ["BAG", "SHOPPER", "TOTE"]),
    ("lighting_candles", ["T-LIGHT", "TEALIGHT", "LANTERN", "CANDLE", "LIGHT", "NIGHT LIGHT"]),
    ("kitchen_baking", ["CAKE", "CAKESTAND", "CAKE STAND", "CAKECASES", "BAKING", "MUG", "CUP", "TEACUP",
                        "PLATE", "BOWL", "TEA", "TEAPOT", "JAM", "LUNCH", "BOTTLE", "NAPKIN", "TRAY",
                        "CUTLERY", "KITCHEN", "BREAD", "APRON", "SPOON"]),
    ("stationery_paper", ["CARD", "WRAP", "PAPER", "NOTEBOOK", "PEN", "PENCIL", "STICKER",
                          "CHALK", "BUNTING", "TISSUE"]),
    ("storage", ["BOX", "TIN", "JAR", "BASKET", "STORAGE", "CRATE", "CASE", "CHEST"]),
    ("home_decor", ["SIGN", "HEART", "FRAME", "CLOCK", "DOORMAT", "CUSHION", "DECORATION",
                    "HANGING", "MIRROR", "WALL", "ORNAMENT", "GARLAND", "HOOK", "HANGER"]),
    ("toys_gifts", ["TOY", "GAME", "DOLLY", "DOLL", "KIT", "PARTY", "BALLOON", "SPACEBOY",
                    "CHILDREN", "ALARM", "SKIPPING", "PLAYHOUSE", "SKITTLES"]),
]
# Whole words only (optional plural S/ES): "TIN" must not match "BUNTING".
_CAT_RE = [(c, re.compile(r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")(?:S|ES)?\b"))
           for c, kws in CATEGORY_RULES]


def categorise(desc: str) -> str:
    for cat, rx in _CAT_RE:
        if rx.search(desc):
            return cat
    return "other"


def variant_key(desc: str) -> str:
    """Product name with colour/design/size words removed: variants share a key."""
    words = re.findall(r"[A-Z0-9'+&-]+", desc.upper())
    return " ".join(w for w in words if w not in COLOUR_WORDS)


# =============================================================================
# 3. CALENDAR: SHOP CLOSURES AND GIFTING EVENTS
# =============================================================================
# Verified against the data (see summary.md, section 0): the shop never trades on
# Saturdays and is closed on every England & Wales bank holiday (incl. substitute
# days and the 2011 Royal Wedding), on Easter Sunday, and from 24 Dec to 2 Jan.
# It DOES trade on 23 Dec and on ordinary Sundays.
CAL_YEARS = [2010, 2011, 2012]
UK_HOLIDAYS = sorted(pd.Timestamp(d) for d in holidays.country_holidays("GB", subdiv="ENG", years=CAL_YEARS))
UK_HOLIDAY_NAMES = {pd.Timestamp(d): n for d, n in holidays.country_holidays("GB", subdiv="ENG", years=CAL_YEARS).items()}


def easter_sunday(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    return pd.Timestamp(year, month, (h + l_ - 7 * m + 114) % 31 + 1)


def gift_events(years=CAL_YEARS) -> dict:
    """UK gifting occasions a gift wholesaler's customers stock up for."""
    ev = {}
    for y in years:
        easter = easter_sunday(y)
        june1 = pd.Timestamp(y, 6, 1)
        ev[pd.Timestamp(y, 2, 14)] = "Valentine's Day"
        ev[easter - pd.Timedelta(days=21)] = "Mothering Sunday (UK Mother's Day)"
        ev[easter] = "Easter"
        ev[june1 + pd.Timedelta(days=(6 - june1.day_of_week) % 7 + 14)] = "Father's Day"
        ev[pd.Timestamp(y, 10, 31)] = "Halloween"
        ev[pd.Timestamp(y, 11, 5)] = "Bonfire Night"
        ev[pd.Timestamp(y, 12, 25)] = "Christmas"
    return dict(sorted(ev.items()))


GIFT_EVENTS = gift_events()
GIFT_EVENT_CODES = {n: i for i, n in enumerate(sorted(set(GIFT_EVENTS.values())))}
CAL_FEATS = ["dow", "month", "weekofyear", "doy_sin", "doy_cos", "is_holiday", "days_to_hol",
             "days_since_hol", "days_to_xmas", "trend", "store_open", "days_to_event", "next_event",
             "days_since_event", "closed_next_7", "closed_prev_7", "reopening_day"]


def closure_reason(day: pd.Timestamp) -> str | None:
    """Why the shop is closed on `day` (None if it trades)."""
    day = pd.Timestamp(day).normalize()
    if (day.month == 12 and day.day >= 24) or (day.month == 1 and day.day <= 2):
        return "Christmas break"
    if day in UK_HOLIDAY_NAMES:
        return f"bank holiday ({UK_HOLIDAY_NAMES[day]})"
    if day == easter_sunday(day.year):
        return "Easter Sunday"
    if day.day_of_week == 5:
        return "the shop does not trade on Saturdays"
    return None


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Everything here is known in advance, so it is safe for future dates."""
    cal = pd.DataFrame({"Date": pd.DatetimeIndex(dates)})
    d = cal["Date"]
    hol = np.array(UK_HOLIDAYS, dtype="datetime64[ns]")
    dv = d.values.astype("datetime64[ns]")
    nxt = np.searchsorted(hol, dv, side="left")
    prv = np.searchsorted(hol, dv, side="right") - 1
    day = np.timedelta64(1, "D")
    cal["dow"] = d.dt.day_of_week
    cal["month"] = d.dt.month
    cal["weekofyear"] = d.dt.isocalendar().week.astype(int).values
    doy = d.dt.day_of_year
    cal["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cal["is_holiday"] = d.isin(UK_HOLIDAYS).astype(int)
    cal["days_to_hol"] = np.clip((hol[np.minimum(nxt, len(hol) - 1)] - dv) / day, 0, 30)
    cal["days_since_hol"] = np.clip((dv - hol[np.maximum(prv, 0)]) / day, 0, 30)
    xmas_this = pd.to_datetime(d.dt.year.astype(str) + "-12-25")
    xmas_next = pd.to_datetime((d.dt.year + 1).astype(str) + "-12-25")
    xmas = xmas_this.where(d <= xmas_this, xmas_next)
    cal["days_to_xmas"] = np.clip((xmas - d).dt.days, 0, 90)
    cal["trend"] = (d - pd.Timestamp("2010-12-01")).dt.days
    cal["store_open"] = [int(closure_reason(x) is None) for x in d]
    # Gifting events: wholesale customers order weeks ahead of these
    ev = np.array(list(GIFT_EVENTS), dtype="datetime64[ns]")
    names = list(GIFT_EVENTS.values())
    en = np.minimum(np.searchsorted(ev, dv, side="left"), len(ev) - 1)
    ep = np.maximum(np.searchsorted(ev, dv, side="right") - 1, 0)
    cal["days_to_event"] = np.clip((ev[en] - dv) / day, 0, 60)
    cal["next_event"] = [GIFT_EVENT_CODES[names[i]] for i in en]
    cal["days_since_event"] = np.clip((dv - ev[ep]) / day, 0, 30)
    # Closures around the day (planned, so known in advance). The data shows the
    # first day back after a closure is WEAKER than usual, not a catch-up day.
    full = pd.date_range(d.min() - pd.Timedelta(days=7), d.max() + pd.Timedelta(days=7))
    closed = pd.Series([int(closure_reason(x) is not None) for x in full], index=full)
    nxt7 = closed[::-1].rolling(8, min_periods=1).sum()[::-1] - closed
    prv7 = closed.rolling(8, min_periods=1).sum() - closed
    cal["closed_next_7"] = nxt7.reindex(d).values
    cal["closed_prev_7"] = prv7.reindex(d).values
    prev_closed = closed.shift(1).fillna(0).reindex(d).values
    prev2_closed = closed.shift(2).fillna(0).reindex(d).values
    cal["reopening_day"] = ((cal["store_open"].values == 1) & (prev_closed == 1) & (prev2_closed == 1)).astype(int)
    return cal


# =============================================================================
# 4. DAILY PANEL + LEAKAGE-FREE FEATURES
# =============================================================================
LAG_FEATS = [f"lag_{L + k}" for k in (0, 7, 14, 21)] + \
            ["rate_7", "rate_14", "rate_28", "rate_56", "rstd_28", "nz_28", "ewm_7",
             "momentum", "days_since_sale", "age", "scale", "rel_lag", "rel_rate_7", "rel_rate_14"]
STATIC_FEATS = ["prod_mean", "price_band", "cat_code"]
STORE_FEATS = ["store_r7", "store_r28", "store_mom"]
FEATURES = CAL_FEATS + LAG_FEATS + STATIC_FEATS + STORE_FEATS   # NB: no price features
BASE_COLS = ["StockCode", "Date", "units", "price", "is_future"] + CAL_FEATS


def build_panel(df, products, cats, train_end, horizon_days=HORIZON_DAYS):
    """Full daily calendar per product, from first sale to `horizon_days` after the
    data ends. Every lag/rolling feature at date t uses units from t-L or earlier
    only (tests/test_pipeline.py::test_features_have_no_leakage enforces this)."""
    sub = df[df["StockCode"].isin(products)]
    daily = (sub.groupby(["StockCode", "Date"])
                .agg(units=("Quantity", "sum"), price=("UnitPrice", "median")).reset_index())
    data_end = df["Date"].max()
    future_end = data_end + pd.Timedelta(days=horizon_days)
    first_sale = daily.groupby("StockCode")["Date"].min()
    frames = [pd.DataFrame({"StockCode": c, "Date": pd.date_range(f, future_end, freq="D")})
              for c, f in first_sale.items()]
    p = pd.concat(frames, ignore_index=True).merge(daily, on=["StockCode", "Date"], how="left")
    p = p.sort_values(["StockCode", "Date"]).reset_index(drop=True)
    p["is_future"] = p["Date"] > data_end
    p["units"] = np.where(p["is_future"], np.nan, p["units"].fillna(0.0))
    p = p.merge(calendar_features(pd.date_range(p["Date"].min(), future_end)), on="Date", how="left")
    p["our_price"] = p.groupby("StockCode", sort=False)["price"].transform(lambda x: x.ffill().bfill())
    med_price = df[df["Date"] <= train_end].groupby("StockCode")["UnitPrice"].median()
    ctx = {"train_end": train_end, "cats": cats,
           "price_band": pd.qcut(med_price.reindex(products), 5, labels=False, duplicates="drop"),
           "cat_codes": {c: i for i, c in enumerate(sorted(cats.unique()))}}
    return add_features(p, ctx)


def add_features(p, ctx):
    """Compute all demand features from p["units"] (lags are at least L days old)."""
    p = p[BASE_COLS + ["our_price"]].copy()
    train_end = ctx["train_end"]
    tr_sales = p[(p["Date"] <= train_end) & (p["units"] > 0)]
    p["cap"] = p["StockCode"].map(tr_sales.groupby("StockCode")["units"].quantile(DEMAND_CAP_Q))
    p["u_cap"] = np.minimum(p["units"], p["cap"].fillna(np.inf))

    g = p.groupby("StockCode", sort=False)
    key = p["StockCode"]

    def roll(series, w, fn):
        return series.groupby(key, sort=False).transform(lambda x: x.rolling(w, min_periods=1).agg(fn))

    s = g["u_cap"].shift(L)
    for k in (0, 7, 14, 21):
        p[f"lag_{L + k}"] = g["u_cap"].shift(L + k)
    open_s = g["store_open"].shift(L)
    for w in (7, 14, 28, 56):
        p[f"rate_{w}"] = roll(s, w, "sum") / roll(open_s, w, "sum").clip(lower=1)
    p["rstd_28"] = roll(s, 28, "std")
    p["nz_28"] = roll((s > 0).astype(float).where(s.notna()), 28, "mean")
    p["ewm_7"] = s.groupby(key, sort=False).transform(lambda x: x.ewm(halflife=7).mean())
    p["momentum"] = (p["rate_7"] / (p["rate_56"] + 0.1)).clip(0, 10)
    p["scale"] = (0.5 * p["rate_28"] + 0.5 * p["rate_56"]).fillna(0).clip(lower=0.2)
    p["rel_lag"] = p[f"lag_{L}"] / p["scale"]
    p["rel_rate_7"] = p["rate_7"] / p["scale"]
    p["rel_rate_14"] = p["rate_14"] / p["scale"]
    first_sale = g["Date"].transform("min")
    last_sale = p["Date"].where(p["u_cap"] > 0).groupby(key).ffill()
    p["age"] = (p["Date"] - first_sale).dt.days
    p["days_since_sale"] = (p["Date"] - last_sale.groupby(key).shift(L)).dt.days
    p["days_since_sale"] = p["days_since_sale"].fillna(p["age"]).clip(upper=180)

    tr = p[(p["Date"] <= train_end) & (p["store_open"] == 1)]
    p["prod_mean"] = key.map(tr.groupby("StockCode")["u_cap"].mean())
    p["price_band"] = key.map(ctx["price_band"])
    p["cat_code"] = key.map(ctx["cats"]).map(ctx["cat_codes"])

    store = p.groupby("Date")["u_cap"].sum(min_count=1).shift(L)
    st = pd.DataFrame({"s": store})
    st["store_r7"] = st["s"].rolling(7, min_periods=1).mean()
    st["store_r28"] = st["s"].rolling(28, min_periods=1).mean()
    st["store_mom"] = st["store_r7"] / (st["store_r28"] + 1)
    return p.merge(st[["store_r7", "store_r28", "store_mom"]], left_on="Date", right_index=True, how="left")


# =============================================================================
# 5. BASELINE DEMAND: CANDIDATES, CALIBRATION, SELECTION
# =============================================================================
ML_MODELS = ["LightGBM", "XGBoost", "RandomForest", "Ensemble"]
NAIVE_MODELS = ["Naive: same day last week", "Naive: 28-day mean rate"]


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    return {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(np.mean(err ** 2)),
            "WAPE%": 100 * np.abs(err).sum() / max(y.sum(), 1e-9), "Bias%": 100 * err.sum() / max(y.sum(), 1e-9)}


def vec(s):
    return np.array(s.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64, copy=True, order="C")


def mat(df, rf=False):
    m = np.array(df[FEATURES].to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64, copy=True, order="C")
    return np.nan_to_num(m, nan=-1.0) if rf else m


def fit_models(tr, es):
    Xtr, ytr, wtr = mat(tr), vec(tr["u_cap"] / tr["scale"]), vec(tr["scale"])
    Xes, yes, wes = mat(es), vec(es["u_cap"] / es["scale"]), vec(es["scale"])
    lgbm = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=4000,
                             learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                             n_jobs=-1, verbose=-1)
    lgbm.fit(Xtr, ytr, sample_weight=wtr, eval_X=(Xes,), eval_y=(yes,), eval_sample_weight=[wes],
             callbacks=[lgb.early_stopping(150, verbose=False)])
    xgbm = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3, n_estimators=4000,
                            learning_rate=0.03, max_depth=7, min_child_weight=20, subsample=0.8,
                            colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
                            eval_metric="tweedie-nloglik@1.3", early_stopping_rounds=150, tree_method="hist")
    xgbm.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xes, yes)], sample_weight_eval_set=[wes], verbose=False)
    rf = RandomForestRegressor(n_estimators=250, min_samples_leaf=20, max_features=0.5, max_samples=0.5,
                               n_jobs=-1, random_state=SEED)
    rf.fit(np.nan_to_num(Xtr, nan=-1.0), ytr, sample_weight=wtr)
    return {"LightGBM": lgbm, "XGBoost": xgbm, "RandomForest": rf}


def predict(models, name, X):
    """Daily units for any candidate (ML or naive); closed days are 0."""
    if name == NAIVE_MODELS[0]:
        return X[f"lag_{L}"].fillna(0).values * X["store_open"].values
    if name == NAIVE_MODELS[1]:
        return X["rate_28"].fillna(0).values * X["store_open"].values

    def one(n):
        return np.clip(models[n].predict(mat(X, rf=(n == "RandomForest"))), 0, None)
    pred = np.mean([one(n) for n in ("LightGBM", "XGBoost", "RandomForest")], axis=0) \
        if name == "Ensemble" else one(name)
    return pred * X["scale"].values * X["store_open"].values


def refit(models, name, data):
    """Refit the selected candidate on `data` with the validated iteration counts."""
    if name in NAIVE_MODELS:
        return {}
    X, y, w = mat(data), vec(data["u_cap"] / data["scale"]), vec(data["scale"])
    out = {}
    for n in (("LightGBM", "XGBoost", "RandomForest") if name == "Ensemble" else (name,)):
        m = models[n]
        if n == "LightGBM":
            r = lgb.LGBMRegressor(**{**m.get_params(), "n_estimators": max(m.best_iteration_, 50)})
            r.fit(X, y, sample_weight=w)
        elif n == "XGBoost":
            r = xgb.XGBRegressor(**{**m.get_params(), "n_estimators": max(m.best_iteration + 1, 50),
                                    "early_stopping_rounds": None})
            r.fit(X, y, sample_weight=w, verbose=False)
        else:
            r = RandomForestRegressor(**m.get_params())
            r.fit(np.nan_to_num(X, nan=-1.0), y, sample_weight=w)
        out[n] = r
    return out


def weekly_totals(frame, pred, start):
    """7-day product totals in complete blocks counted from `start` (partial blocks dropped)."""
    blk = (frame["Date"] - start).dt.days // 7
    n_days = frame.groupby(blk)["Date"].nunique()
    full = n_days[n_days == 7].index
    w = frame.assign(pred=pred, blk=blk)
    w = w[w["blk"].isin(full)]
    return w.groupby(["StockCode", "blk"])[["u_cap", "pred"]].sum()


def calib_factor(actual, pred):
    return float(np.clip(np.sum(actual) / max(np.sum(pred), 1e-9), *CALIB_CLIP))


def holiday_check(raw, df, products):
    """Compare the closure calendar with the days the shop actually traded, and
    measure sales around gifting events (all from the data)."""
    traded = set(pd.to_datetime(raw["InvoiceDate"]).dt.normalize().unique())
    days = pd.date_range(min(traded), max(traded))
    says_closed = {d for d in days if closure_reason(d) is not None}
    wrong_open = sorted(d for d in says_closed if d in traded)            # calendar says closed, shop traded
    wrong_closed = sorted(d for d in days if d not in traded and d not in says_closed)
    s = df[df["StockCode"].isin(products)].groupby("Date")["Quantity"].sum()
    rows = []
    for d, name in GIFT_EVENTS.items():
        if d - pd.Timedelta(days=84) < s.index.min() or d > s.index.max():
            continue
        before = s[(s.index > d - pd.Timedelta(days=28)) & (s.index < d)].mean()
        prior = s[(s.index > d - pd.Timedelta(days=84)) & (s.index <= d - pd.Timedelta(days=28))].mean()
        rows.append({"event": name, "date": str(d.date()), "sales 4 wks before vs 8 wks earlier": f"{before / prior - 1:+.0%}"})
    reopen = [d for d in days if d in traded and closure_reason(d - pd.Timedelta(days=1)) is not None
              and closure_reason(d - pd.Timedelta(days=2)) is not None]
    reopen_eff = [s.get(d, 0) / s[s.index < d].tail(10).mean() - 1 for d in reopen if len(s[s.index < d]) >= 10]
    return {"days_checked": len(days), "closed_days": len(says_closed), "calendar_wrong_open": [str(d.date()) for d in wrong_open],
            "calendar_wrong_closed": [str(d.date()) for d in wrong_closed], "events": rows,
            "reopening_days": len(reopen_eff), "reopening_effect": float(np.median(reopen_eff)) if reopen_eff else float("nan")}


def day_calendar(dates):
    """Plain-language notes for each forecast day (shop closed? event coming up?)."""
    out = []
    for d in sorted(pd.DatetimeIndex(dates)):
        reason = closure_reason(d)
        nxt = [(e, n) for e, n in GIFT_EVENTS.items() if 0 <= (e - d).days <= 21]
        note = ""
        if reason is None and closure_reason(d - pd.Timedelta(days=1)) and closure_reason(d - pd.Timedelta(days=2)):
            note = "First day back after a closure: usually a slow day."
        elif reason is None and any(closure_reason(d + pd.Timedelta(days=k)) == "Christmas break" for k in range(1, 8)):
            note = "Last days before the Christmas break: sales usually slow down."
        elif nxt and reason is None:
            e, n = nxt[0]
            k = (e - d).days
            note = f"{n} is in {k} day{'s' if k != 1 else ''}." if k else f"Today is {n}."
        out.append({"date": str(d.date()), "open": reason is None, "reason": reason or "", "note": note})
    return out


def last_year_profile(df, products, data_end, horizon=DAILY_HORIZON_DAYS, week=HORIZON_DAYS):
    """Store-wide daily pattern for days week+1..horizon after data_end, relative
    to the level of next week, taken from the same days one year earlier
    (364 days back = same weekday; if that day was closed but the target day is
    open, the nearest open day of last year is used). Days are smoothed with their
    open neighbours (+/-1 day) because single days are very noisy."""
    s = df[df["StockCode"].isin(products)].groupby("Date")["Quantity"].sum()
    def ly(day):
        for off in (364, 365, 363, 366, 362, 371, 357):
            d = day - pd.Timedelta(days=off)
            if closure_reason(d) is None and d >= s.index.min():
                near = [d + pd.Timedelta(days=k) for k in (-1, 0, 1)]
                vals = [s.get(x, 0.0) for x in near if closure_reason(x) is None]
                return float(np.mean(vals))
        return np.nan
    wk = [data_end + pd.Timedelta(days=k) for k in range(1, week + 1)]
    base = np.nanmean([ly(d) for d in wk if closure_reason(d) is None])
    prof = {}
    for k in range(week + 1, horizon + 1):
        d = data_end + pd.Timedelta(days=k)
        prof[d] = 0.0 if closure_reason(d) is not None else ly(d) / base
    src = (f"last year's same days relative to last year's equivalent of next week "
           f"(factors {min(prof.values()):.2f}–{max(prof.values()):.2f})")
    return pd.Series(prof).fillna(1.0), src


# =============================================================================
# 6. ELASTICITY: PPML TWO-WAY FIXED EFFECTS ON THE FULL CALENDAR
# =============================================================================
def list_price_daily(df, products, cutoff):
    """Daily LIST price: median unit price of regular-size lines (qty <= the item's
    median line quantity). Bulk lines get tier discounts, so mixing them in
    creates fake "price cuts" on big-order days."""
    sub = df[(df["Date"] <= cutoff) & df["StockCode"].isin(products)]
    med_q = sub.groupby("StockCode")["Quantity"].median()
    reg = sub[sub["Quantity"] <= sub["StockCode"].map(med_q)]
    return reg.groupby(["StockCode", "Date"])["UnitPrice"].median().rename("list_price")


def elasticity_panel(df, products, cats, cutoff):
    """One row per item per TRADING day from first sale to cutoff, zero-sale days
    included. Price on days without a regular-size sale = last list price."""
    lp = list_price_daily(df, products, cutoff)
    units = df[(df["Date"] <= cutoff) & df["StockCode"].isin(products)] \
        .groupby(["StockCode", "Date"])["Quantity"].sum()
    cal = calendar_features(pd.date_range(df["Date"].min(), cutoff))
    open_days = cal.loc[cal["store_open"] == 1, "Date"]
    first = units.reset_index().groupby("StockCode")["Date"].min()
    frames = [pd.DataFrame({"StockCode": c, "Date": open_days[open_days >= f].values}) for c, f in first.items()]
    p = pd.concat(frames, ignore_index=True)
    p = p.join(units.rename("y"), on=["StockCode", "Date"]).join(lp, on=["StockCode", "Date"])
    p["y"] = p["y"].fillna(0.0)
    p["sold"] = p["list_price"].notna()
    p["list_price"] = p.groupby("StockCode")["list_price"].transform(lambda s: s.ffill().bfill())
    p = p.dropna(subset=["list_price"])
    p["x"] = np.log(p["list_price"])
    p["cat"] = p["StockCode"].map(cats)
    p["week"] = p["Date"].dt.to_period("W-SUN")
    return p.reset_index(drop=True)


def _wdemean(x, w, ids, iters=50, tol=1e-10):
    """Weighted two-way demeaning (alternating projections)."""
    x = x.copy()
    for _ in range(iters):
        delta = 0.0
        for k, n in ids:
            m = np.bincount(k, w * x, n) / np.maximum(np.bincount(k, w, n), 1e-300)
            x -= m[k]
            delta = max(delta, np.abs(m).max())
        if delta < tol:
            break
    return x


def ppml_fe(item, date, x, y, n_item, n_date, b0=-1.0, max_iter=100):
    """Poisson pseudo-ML with item and date fixed effects and one common slope.
    Returns slope, item-clustered s.e., and the fitted date effects."""
    b = b0
    a = np.log(np.bincount(item, y, n_item) + 1e-9) - np.log(np.bincount(item, np.exp(b * x), n_item))
    d = np.zeros(n_date)
    ysum_d = np.bincount(date, y, n_date)
    for _ in range(max_iter):
        for _ in range(5):
            e = np.exp(d[date] + b * x)
            a = np.log(np.bincount(item, y, n_item) + 1e-12) - np.log(np.bincount(item, e, n_item) + 1e-300)
            e = np.exp(a[item] + b * x)
            with np.errstate(divide="ignore"):
                d = np.where(ysum_d > 0, np.log(ysum_d + 1e-300) - np.log(np.bincount(date, e, n_date) + 1e-300), -30.0)
        mu = np.exp(a[item] + d[date] + b * x)
        xt = _wdemean(x, mu, [(item, n_item), (date, n_date)])
        h = np.sum(mu * xt ** 2)
        step = np.sum(xt * (y - mu)) / max(h, 1e-12)
        b += float(np.clip(step, -1, 1))
        if abs(step) < 1e-7:
            break
    mu = np.exp(a[item] + d[date] + b * x)
    xt = _wdemean(x, mu, [(item, n_item), (date, n_date)])
    h = np.sum(mu * xt ** 2)
    score = np.bincount(item, xt * (y - mu), n_item)
    se = float(np.sqrt(np.sum(score ** 2)) / max(h, 1e-12))
    return float(b), se, d


def ppml_item(y, x, offset, b0=-1.0, max_iter=100):
    """Item-level PPML slope with the item effect profiled out and the category's
    date effects as offset. Newey-West (HAC) s.e. over consecutive trading days."""
    b = b0
    for _ in range(max_iter):
        e = np.exp(offset + b * x)
        a = np.log(y.sum() + 1e-12) - np.log(e.sum() + 1e-300)
        mu = np.exp(a) * e
        xt = x - np.sum(mu * x) / mu.sum()
        h = np.sum(mu * xt ** 2)
        step = np.sum(xt * (y - mu)) / max(h, 1e-12)
        b += float(np.clip(step, -1, 1))
        if abs(step) < 1e-8:
            break
    e = np.exp(offset + b * x)
    mu = e * y.sum() / max(e.sum(), 1e-300)
    xt = x - np.sum(mu * x) / mu.sum()
    h = max(np.sum(mu * xt ** 2), 1e-12)
    s = xt * (y - mu)
    v = s @ s
    for lag in range(1, HAC_LAGS + 1):
        v += 2 * (1 - lag / (HAC_LAGS + 1)) * (s[lag:] @ s[:-lag])
    return float(b), float(np.sqrt(max(v, s @ s)) / h)


def price_evidence(g: pd.DataFrame) -> dict:
    """Weeks with sales, and distinct weekly list prices held for >= N WEEKS each."""
    sold = g[g["sold"]]
    wk_price = sold.groupby("week")["list_price"].median().round(2)
    levels = wk_price.value_counts()
    return {"n_weeks": int(sold["week"].nunique()),
            "n_price_levels": int((levels >= ELAST_MIN_WEEKS_PER_LEVEL).sum()),
            "logp_std": float(g["x"].std(ddof=0))}


def _codes(s):
    c, u = pd.factorize(s)
    return c.astype(np.int64), len(u)


def category_fit(g, shuffle_rng=None):
    item, n_item = _codes(g["StockCode"])
    date, n_date = _codes(g["Date"])
    x = g["x"].to_numpy(float)
    if shuffle_rng is not None:   # placebo: circularly shift each item's price path in time
        x = x.copy()
        for idx in g.groupby("StockCode").indices.values():
            if len(idx) > 40:
                x[idx] = np.roll(x[idx], shuffle_rng.integers(20, len(idx) - 20))
    return ppml_fe(item, date, x, g["y"].to_numpy(float), n_item, n_date), date


def product_elasticities(df, products, cats, cutoff, n_placebo=0):
    d = elasticity_panel(df, products, cats, cutoff)
    rows, cat_fit, placebo = [], {}, []
    rng = np.random.default_rng(SEED)
    for cat, g in d.groupby("cat"):
        g = g.reset_index(drop=True)
        (b, se, dfe), date = category_fit(g)
        offset = dfe[date]
        n_clusters = 0
        for code, idx in g.groupby("StockCode").indices.items():
            gi = g.iloc[idx]
            rec = {"StockCode": code, **price_evidence(gi), "e_raw": np.nan, "se_raw": np.nan}
            ok = (rec["n_weeks"] >= ELAST_MIN_WEEKS and rec["n_price_levels"] >= ELAST_MIN_PRICE_LEVELS
                  and rec["logp_std"] >= ELAST_MIN_LOGP_STD and gi["y"].sum() > 0)
            if ok:
                n_clusters += 1
                bi, sei = ppml_item(gi["y"].to_numpy(float), gi["x"].to_numpy(float), offset[idx])
                rec["e_raw"], rec["se_raw"] = bi, max(sei, ELAST_SE_FLOOR)
            rows.append(rec)
        cat_fit[cat] = (b, se, n_clusters)
        # Placebo only where the category slope is actually used (unidentified
        # categories have no meaningful slope to compare against).
        for k in range(n_placebo if n_clusters >= MIN_CLUSTERS_CATEGORY else 0):
            (bp, _, _), _ = category_fit(g, shuffle_rng=rng)
            placebo.append({"category": cat, "draw": k, "placebo_slope": bp, "real_slope": b})
    el = pd.DataFrame(rows).set_index("StockCode").reindex(products)
    el[["n_weeks", "n_price_levels"]] = el[["n_weeks", "n_price_levels"]].fillna(0).astype(int)
    return el, cat_fit, pd.DataFrame(placebo)


def dersimonian_laird(e, se):
    w = 1 / se ** 2
    mu_fe = np.sum(w * e) / np.sum(w)
    q = np.sum(w * (e - mu_fe) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max((q - (len(e) - 1)) / c if c > 0 else 0.0, TAU_MIN ** 2)
    w2 = 1 / (se ** 2 + tau2)
    return np.sum(w2 * e) / np.sum(w2), tau2, np.sqrt(1 / np.sum(w2))


def event_elasticity(events):
    """Pooled elasticity from real list-price changes: slope through the origin of
    the item's log change in units (net of its category) on its log price change,
    with heteroskedasticity-robust s.e. These clean before/after comparisons are
    the least noisy evidence on the LEVEL of price sensitivity."""
    if len(events) < 10:
        return None
    x, y = events["dlogp"].to_numpy(float), events["actual_dlogq"].to_numpy(float)
    b = float(x @ y / (x @ x))
    se = float(np.sqrt(np.sum((x * (y - b * x)) ** 2)) / (x @ x))
    return {"elasticity": b, "se": se, "n_events": int(len(events)), "n_items": int(events["StockCode"].nunique())}


def _panel_level(cat_fit):
    good = {c: v for c, v in cat_fit.items() if v[2] >= MIN_CLUSTERS_CATEGORY}
    src = good if len(good) >= 2 else cat_fit
    return dersimonian_laird(np.array([v[0] for v in src.values()]), np.array([v[1] for v in src.values()]))[0]


def calibrate_level(el, cat_fit, level):
    """Measurement error in daily prices pulls panel slopes toward zero by a common
    factor. Rescale every panel slope (and its s.e.) so the overall level matches
    the event-study level; the panel still decides how categories and items DIFFER."""
    g = _panel_level(cat_fit)
    if level is None or g >= 0 or level["elasticity"] >= 0:
        return el, cat_fit, 1.0
    k = float(np.clip(level["elasticity"] / g, *LEVEL_SCALE_CLIP))
    el = el.copy()
    el["e_raw"], el["se_raw"] = el["e_raw"] * k, el["se_raw"] * k
    return el, {c: (b * k, se * k, n) for c, (b, se, n) in cat_fit.items()}, k


def pool_elasticities(el, cats, cat_fit, level=None):
    """Global -> category -> item empirical Bayes.
    * If `level` (event-study estimate) is given, panel slopes are first rescaled
      to that level, and its uncertainty is added to every prior.
    * A category slope is used only if >= MIN_CLUSTERS_CATEGORY items in it have
      their own price variation (clustered s.e. need clusters); others use the global prior.
    * Items whose own slope is above the ceiling (demand not falling with price)
      contribute no evidence: they get the category prior and are flagged.
    * Every prior mean is capped at the ceiling, so no posterior can be positive."""
    el, cat_fit, k = calibrate_level(el, cat_fit, level)
    el["category"] = cats.reindex(el.index)
    level_var = (level["se"] ** 2) if (level is not None and k != 1.0) else 0.0
    good = {c: v for c, v in cat_fit.items() if v[2] >= MIN_CLUSTERS_CATEGORY}
    src = good if len(good) >= 2 else cat_fit
    cb = np.array([v[0] for v in src.values()])
    cse = np.array([v[1] for v in src.values()])
    g_mu, g_tau2_cat, g_se = dersimonian_laird(cb, cse)
    g_mu = min(g_mu, core.ELASTICITY_CEILING)
    usable = el["e_raw"].notna() & (el["e_raw"] <= core.ELASTICITY_CEILING)
    _, g_tau2, _ = dersimonian_laird(el.loc[usable, "e_raw"].values, el.loc[usable, "se_raw"].values)
    priors = {}
    for cat in el["category"].unique():
        if cat in good:
            b, se, n = good[cat]
            prec = 1 / se ** 2 + 1 / g_tau2_cat
            mu, se_mu = (b / se ** 2 + g_mu / g_tau2_cat) / prec, np.sqrt(1 / prec)
            basis = f"category slope ({n} items)"
        else:
            mu, se_mu = g_mu, np.sqrt(g_tau2_cat + g_se ** 2)
            basis = f"global prior (only {cat_fit.get(cat, (0, 0, 0))[2]} items with price changes)"
        mu = min(mu, core.ELASTICITY_CEILING)
        grp = el[usable & (el["category"] == cat)]
        tau2 = (dersimonian_laird(grp["e_raw"].values, grp["se_raw"].values)[1]
                if len(grp) >= MIN_PRODUCTS_TAU else g_tau2)
        priors[cat] = (mu, tau2, float(np.sqrt(se_mu ** 2 + level_var)), basis)
    post_e, post_se, source = [], [], []
    for _, r in el.iterrows():
        mu, tau2, se_mu, _ = priors[r["category"]]
        v_prior = tau2 + se_mu ** 2
        if pd.notna(r["e_raw"]) and r["e_raw"] <= core.ELASTICITY_CEILING:
            prec = 1 / r["se_raw"] ** 2 + 1 / v_prior
            post_e.append((r["e_raw"] / r["se_raw"] ** 2 + mu / v_prior) / prec)
            post_se.append(np.sqrt(1 / prec))
            source.append(core.SRC_OWN)
        else:
            post_e.append(mu)
            post_se.append(np.sqrt(v_prior))
            source.append(core.SRC_CONTRA if pd.notna(r["e_raw"]) else core.SRC_PRIOR)
    el["elasticity"] = np.clip(post_e, core.ELASTICITY_FLOOR, core.ELASTICITY_CEILING)
    el["elasticity_se"], el["elasticity_source"] = post_se, source
    el["prior_mean"] = el["category"].map(lambda c: priors[c][0])
    prior_tbl = pd.DataFrame({c: {"category_slope": cat_fit.get(c, (np.nan,))[0],
                                  "items_with_price_changes": int(cat_fit.get(c, (0, 0, 0))[2]),
                                  "prior_mean": v[0], "prior_se": v[2], "tau": np.sqrt(v[1]), "prior_basis": v[3]}
                              for c, v in priors.items()}).T
    return el, prior_tbl, (g_mu, np.sqrt(g_tau2), k)


# =============================================================================
# 7. BACKTEST OF ELASTICITY ON LATER REAL PRICE CHANGES
# =============================================================================
def price_change_events(df, products, cats, start, end, min_change=0.02, window=2):
    """List-price changes after `start`: the weekly list price moves by >= 2% and
    holds for `window` weeks on both sides. Actual effect = log change in mean
    weekly units of the item minus the same change for the rest of its category."""
    lp = list_price_daily(df, products, end).reset_index()
    lp["week"] = lp["Date"].dt.to_period("W-SUN")
    wp = lp.groupby(["StockCode", "week"])["list_price"].median().round(2).unstack()
    u = df[(df["Date"] <= end) & df["StockCode"].isin(products)].assign(week=lambda f: f["Date"].dt.to_period("W-SUN"))
    wu = u.groupby(["StockCode", "week"])["Quantity"].sum().unstack().reindex(columns=wp.columns).fillna(0)
    weeks = list(wp.columns)
    first_wk = pd.Timestamp(start).to_period("W-SUN")
    cat_tot = wu.groupby(wu.index.map(cats)).sum()
    ev = []
    for code in wp.index:
        s = wp.loc[code]
        for i in range(window, len(weeks) - window + 1):
            if weeks[i] <= first_wk:
                continue
            before, after = s.iloc[i - window:i], s.iloc[i:i + window]
            if before.isna().any() or after.isna().any() or before.nunique() > 1 or after.nunique() > 1:
                continue
            p_b, p_a = before.iloc[0], after.iloc[0]
            if abs(np.log(p_a / p_b)) < np.log(1 + min_change):
                continue
            ub, ua = wu.loc[code].iloc[i - window:i].mean(), wu.loc[code].iloc[i:i + window].mean()
            c = cats[code]
            cb = cat_tot.loc[c].iloc[i - window:i].sum() - wu.loc[code].iloc[i - window:i].sum()
            ca = cat_tot.loc[c].iloc[i:i + window].sum() - wu.loc[code].iloc[i:i + window].sum()
            if min(ub, ua, cb, ca) <= 0:
                continue
            ev.append({"StockCode": code, "week": str(weeks[i]), "price_before": p_b, "price_after": p_a,
                       "dlogp": np.log(p_a / p_b), "actual_dlogq": np.log(ua / ub) - np.log(ca / cb)})
    return pd.DataFrame(ev)


def backtest(df, products, cats, train_end, data_end):
    """Re-estimate everything with data up to train_end (including the event-study
    level, from price changes BEFORE train_end only) and predict later changes."""
    el_bt, cf_bt, _ = product_elasticities(df, products, cats, train_end)
    level_bt = event_elasticity(price_change_events(df, products, cats, df["Date"].min(), train_end,
                                                    window=EVENT_WINDOW_WEEKS))
    el_bt, _, _ = pool_elasticities(el_bt, cats, cf_bt, level=level_bt)
    ev = price_change_events(df, products, cats, train_end, data_end, window=EVENT_WINDOW_WEEKS)
    if ev.empty:
        return ev, {"n_events": 0}
    ev["elasticity_at_cutoff"] = ev["StockCode"].map(el_bt["elasticity"])
    ev["source_at_cutoff"] = ev["StockCode"].map(el_bt["elasticity_source"])
    ev["pred_dlogq"] = ev["elasticity_at_cutoff"] * ev["dlogp"]
    err, null = ev["actual_dlogq"] - ev["pred_dlogq"], ev["actual_dlogq"]
    s = {"n_events": int(len(ev)), "n_items": int(ev["StockCode"].nunique()),
         "mae_model": float(err.abs().mean()), "mae_no_effect": float(null.abs().mean()),
         "direction_hit_rate": float((np.sign(ev["actual_dlogq"]) == np.sign(ev["pred_dlogq"])).mean()),
         "corr": float(ev[["actual_dlogq", "pred_dlogq"]].corr().iloc[0, 1]) if len(ev) > 2 else float("nan"),
         "implied_elasticity_median": float((ev["actual_dlogq"] / ev["dlogp"]).median())}
    return ev, s


# =============================================================================
# 8. COMPETITOR — SIMULATED (or real via COMPETITOR_PATH)
# =============================================================================
def simulate_competitor(panel, ref_price):
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
        log(f"Competitor: REAL data for {has.mean():.0%} of product-days")
    return panel


# =============================================================================
# 9. PLOTS
# =============================================================================
def plot_forecast(res_tbl, te_weeks, best):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    t = res_tbl.reset_index()
    xs = np.arange(len(t))
    ax[0].bar(xs - 0.2, t["select_WAPE%_7day"], 0.4, label="validation (selection weeks)", color="#4C72B0")
    ax[0].bar(xs + 0.2, t["test_WAPE%_7day"], 0.4, label="test", color="#DD8452")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(t["model"], rotation=30, ha="right")
    ax[0].set_ylabel("WAPE % of 7-day product totals (lower is better)")
    ax[0].set_title(f"All candidates calibrated identically — selected: {best}")
    ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
    w = te_weeks.groupby(level="blk")[["u_cap", "pred"]].sum()
    ax[1].bar(w.index - 0.2, w["u_cap"], 0.4, color="black", label="actual")
    ax[1].bar(w.index + 0.2, w["pred"], 0.4, color="#DD8452", label=best)
    ax[1].set_xlabel("test week"); ax[1].set_title("Test period: weekly units, modelled products")
    ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "forecast_accuracy.png", dpi=130); plt.close(fig)


def plot_elasticity(el, placebo, bt):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    raw = el["e_raw"].dropna()
    bins = np.linspace(-6, 3, 61)
    ax[0].hist(raw.clip(-6, 3), bins=bins, alpha=0.5, label=f"raw item PPML (n={len(raw)})", color="gray")
    ax[0].hist(el["elasticity"].clip(-6, 3), bins=bins, alpha=0.7, label=f"pooled (n={len(el)})", color="#4C72B0")
    ax[0].axvline(-1, color="orange", ls=":", label="unit elastic")
    ax[0].set_title("Own-price elasticity (REAL data)"); ax[0].legend()
    if not placebo.empty:
        cats = sorted(placebo["category"].unique())
        ax[1].boxplot([placebo.loc[placebo["category"] == c, "placebo_slope"] for c in cats],
                      orientation="horizontal", tick_labels=cats)
        real = placebo.groupby("category")["real_slope"].first().reindex(cats)
        ax[1].scatter(real.values, np.arange(1, len(cats) + 1), color="red", zorder=3, label="real slope")
        ax[1].set_title("Placebo test: slopes from time-shuffled prices vs real"); ax[1].legend()
    if len(bt):
        ax[2].scatter(bt["pred_dlogq"], bt["actual_dlogq"], s=14)
        lim = np.nanmax(np.abs(bt[["pred_dlogq", "actual_dlogq"]].values)) * 1.05
        ax[2].plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax[2].axhline(0, color="gray", lw=0.5); ax[2].axvline(0, color="gray", lw=0.5)
        ax[2].set_xlabel("predicted log change in units"); ax[2].set_ylabel("actual (vs category)")
        ax[2].set_title("Backtest: later real price changes")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "elasticity_distribution.png", dpi=130); plt.close(fig)


def plot_profit_curves(rec, z, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for ax, (_, r) in zip(axes.ravel(), rec.head(n).iterrows()):
        p0 = r["current_price"]
        grid = np.linspace(p0 * 0.7, p0 * 1.3, 200)
        prof = (grid - r["unit_cost"])[None, :] * core.demand_matrix(
            grid, p0, r["baseline_units"], core.elasticity_draws(r["elasticity"], r["elasticity_se"], z))
        ax.fill_between(grid, np.quantile(prof, 0.1, 0), np.quantile(prof, 0.9, 0), alpha=0.2, color="#4C72B0",
                        label="P10–P90 (elasticity)")
        ax.plot(grid, prof.mean(0), color="#4C72B0", lw=2, label="expected profit")
        ax.plot(grid, np.quantile(prof, 0.25, 0), color="#4C72B0", ls="--", lw=1, label="P25")
        ax.axvspan(r["bound_lo"], r["bound_hi"], color="green", alpha=0.06, label="allowed range")
        ax.axvline(p0, color="gray", ls="--", label=f"current £{p0:.2f}")
        ax.axvline(r["recommended_price"], color="red", lw=2, label=f"recommended £{r['recommended_price']:.2f}")
        ax.set_title(f"{r['StockCode']} {r['Description'][:30]}\ne={r['elasticity']:.2f}±{r['elasticity_se']:.2f}, "
                     f"conf={r['confidence']}, {r['decision_driver']}", fontsize=9)
        ax.set_xlabel("price £"); ax.set_ylabel(f"profit £ over {HORIZON_DAYS} days"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Profit vs price (largest-revenue items)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "profit_vs_price_curves.png", dpi=130); plt.close(fig)


def plot_scenarios(rec, panel):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    tot_cur = [rec[f"{s}_profit_at_current"].sum() for s in core.COMP_SCENARIOS]
    tot_opt = [rec[f"{s}_profit"].sum() for s in core.COMP_SCENARIOS]
    xs = np.arange(len(core.COMP_SCENARIOS))
    ax[0].bar(xs - 0.2, tot_cur, 0.4, label="keep current prices", color="gray")
    ax[0].bar(xs + 0.2, tot_opt, 0.4, label="recommended prices", color="#4C72B0")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(list(core.COMP_SCENARIOS))
    ax[0].set_title("Total profit by competitor scenario (SIMULATED)"); ax[0].legend(); ax[0].grid(alpha=0.3, axis="y")
    for code in rec["StockCode"].head(3):
        g = panel[(panel["StockCode"] == code) & ~panel["is_future"]]
        line, = ax[1].plot(g["Date"], g["our_price"], lw=2, label=f"{code} our price")
        ax[1].plot(g["Date"], g["comp_price"], lw=1, ls="--", color=line.get_color(), label=f"{code} competitor (SIMULATED)")
    ax[1].set_title("SIMULATED competitor price paths"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "competitor_scenarios.png", dpi=130); plt.close(fig)


def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


# =============================================================================
# 10. MAIN PIPELINE
# =============================================================================
def run(as_of: str | None = None):
    global OUTPUT_DIR
    raw = load_transactions()
    if as_of:
        raw = raw[pd.to_datetime(raw["InvoiceDate"]).dt.normalize() <= pd.Timestamp(as_of)]
        OUTPUT_DIR = OUTPUT_DIR / f"asof_{as_of}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, clean_stats = clean(raw)
    data_end = df["Date"].max()
    val_end = data_end - pd.Timedelta(days=TEST_DAYS)
    train_end = val_end - pd.Timedelta(days=VAL_DAYS)
    val_cal_end = train_end + pd.Timedelta(days=VAL_DAYS // 2)
    if train_end - df["Date"].min() < pd.Timedelta(days=150):
        raise SystemExit("Not enough history before the training cut-off; choose a later --as-of date.")
    log(f"Data {df['Date'].min().date()} -> {data_end.date()}; train <= {train_end.date()}, "
        f"calibration <= {val_cal_end.date()}, selection <= {val_end.date()}, test <= {data_end.date()}")

    desc = df.groupby("StockCode")["Description"].agg(lambda s: s.mode().iat[0])
    cats = desc.map(categorise)
    tr_lines = df[df["Date"] <= train_end]
    sale_days = tr_lines.groupby("StockCode")["Date"].nunique()
    tr_rev = tr_lines.groupby("StockCode")["Revenue"].sum()
    eligible = sale_days[sale_days >= MIN_TRAIN_SALE_DAYS].index
    products = tr_rev.reindex(eligible).sort_values(ascending=False).head(TOP_N_MODEL).index.tolist()
    log(f"Modelling {len(products)} products")
    hol = holiday_check(raw, df, products)
    log(f"Holiday calendar vs data: {hol['days_checked']} days, {len(hol['calendar_wrong_open'])} wrongly closed, "
        f"{len(hol['calendar_wrong_closed'])} unexplained closures")

    # ---------------- STAGE 1a: baseline demand ----------------
    panel = build_panel(df, products, cats, train_end, horizon_days=DAILY_HORIZON_DAYS)
    hist = panel[~panel["is_future"] & (panel["age"] >= L)]
    tr = hist[(hist["Date"] <= train_end) & (hist["store_open"] == 1)]
    va_cal = hist[(hist["Date"] > train_end) & (hist["Date"] <= val_cal_end)]
    va_sel = hist[(hist["Date"] > val_cal_end) & (hist["Date"] <= val_end)]
    te = hist[hist["Date"] > val_end]
    models = fit_models(tr, va_cal[va_cal["store_open"] == 1])

    rows, calib, te_preds = [], {}, {}
    for name in ML_MODELS + NAIVE_MODELS:
        # Every candidate gets the same treatment: level factor fitted on the
        # calibration weeks, selection on the later selection weeks, test untouched.
        calib[name] = calib_factor(va_cal["u_cap"], predict(models, name, va_cal))
        ps, pt = predict(models, name, va_sel) * calib[name], predict(models, name, te) * calib[name]
        te_preds[name] = pt
        ws, wt = weekly_totals(va_sel, ps, val_cal_end + pd.Timedelta(days=1)), weekly_totals(te, pt, val_end + pd.Timedelta(days=1))
        mt = metrics(te["u_cap"], pt)
        rows.append({"model": name, "kind": "naive" if name in NAIVE_MODELS else "ML", "calibration": calib[name],
                     "select_WAPE%_7day": metrics(ws["u_cap"], ws["pred"])["WAPE%"],
                     "test_WAPE%_7day": metrics(wt["u_cap"], wt["pred"])["WAPE%"],
                     "test_Bias%_7day": metrics(wt["u_cap"], wt["pred"])["Bias%"],
                     **{f"test_{k}_daily": v for k, v in mt.items()}})
    res_tbl = pd.DataFrame(rows).set_index("model")
    best = res_tbl["select_WAPE%_7day"].idxmin()
    res_tbl["selected"] = res_tbl.index == best
    log(f"Selected baseline (over ALL candidates incl. naive): {best}")
    print(res_tbl.round(2).to_string())
    res_tbl.round(4).to_csv(OUTPUT_DIR / "model_metrics.csv")
    te_weeks = weekly_totals(te, te_preds[best], val_end + pd.Timedelta(days=1))
    plot_forecast(res_tbl, te_weeks, best)

    # Production calibration: refit on data up to val_end (same lag as production),
    # measure the level error on test, apply that factor to the full-history refit.
    check = refit(models, best, hist[(hist["Date"] <= val_end) & (hist["store_open"] == 1)])
    calib_prod = calib_factor(te["u_cap"], predict(check, best, te))
    final = refit(models, best, hist[hist["store_open"] == 1])
    # Daily forecasts. Days 1-7: the validated 7-day-ahead model. Days 8-30: next
    # week's level x last year's pattern for the same days (store-wide, weekday-
    # matched). The model has seen only one Christmas, and on its own it misses
    # the pre-Christmas slowdown and the weak first days back in January.
    fut_mask = panel["is_future"].to_numpy()
    week_mask = fut_mask & (panel["Date"] <= data_end + pd.Timedelta(days=HORIZON_DAYS)).to_numpy()
    fut = panel[week_mask]
    week_pred = pd.Series(predict(final, best, fut) * calib_prod, index=fut.index)
    baseline_units = week_pred.groupby(fut["StockCode"]).sum()
    profile, profile_src = last_year_profile(df, products, data_end)
    open_week = fut.groupby("StockCode")["store_open"].sum().clip(lower=1)
    level = baseline_units / open_week                       # forecast units per open day, next week
    later = panel.loc[fut_mask & ~week_mask, ["StockCode", "Date"]]
    later_units = later["StockCode"].map(level).to_numpy() * later["Date"].map(profile).fillna(0).to_numpy()
    daily_fc = pd.concat([fut[["StockCode", "Date"]].assign(baseline_units=week_pred.to_numpy()),
                          later.assign(baseline_units=later_units)]).sort_values(["StockCode", "Date"])
    daily_fc.assign(baseline_units=daily_fc["baseline_units"].round(4)).to_csv(OUTPUT_DIR / "daily_forecast.csv", index=False)
    log(f"Day pattern beyond next week: {profile_src}")
    log(f"Production calibration {calib_prod:.3f} (refit-to-test check); next-week baseline "
        f"{baseline_units.sum():,.0f} units; daily forecasts to {daily_fc['Date'].max().date()}")

    # Forecast error distribution (actual / forecast, 7-day totals) by volume band
    tw = te_weeks[te_weeks["pred"] > 0.5]
    band_edges = np.quantile(tw["pred"], [1 / 3, 2 / 3]).tolist()
    rng = np.random.default_rng(SEED + 1)
    ratio_samples = []
    for b in range(3):
        lo_e = -np.inf if b == 0 else band_edges[b - 1]
        hi_e = np.inf if b == 2 else band_edges[b]
        rat = (tw["u_cap"] / tw["pred"])[(tw["pred"] > lo_e) & (tw["pred"] <= hi_e)].to_numpy()
        q = np.quantile(rat, (np.arange(N_ELASTICITY_DRAWS) + 0.5) / N_ELASTICITY_DRAWS)
        ratio_samples.append(rng.permutation(q))

    # ---------------- STAGE 1b: elasticity ----------------
    el, cat_fit, placebo = product_elasticities(df, products, cats, data_end, n_placebo=N_PLACEBO)
    level = event_elasticity(price_change_events(df, products, cats, df["Date"].min(), data_end,
                                                 window=EVENT_WINDOW_WEEKS))
    el, prior_tbl, (g_mu, g_tau, level_k) = pool_elasticities(el, cats, cat_fit, level=level)
    log(f"Event-study level: {level}; panel slopes scaled by {level_k:.2f}")
    placebo.round(4).to_csv(OUTPUT_DIR / "elasticity_placebo.csv", index=False)
    pl = placebo.groupby("category").agg(real=("real_slope", "first"), placebo_mean=("placebo_slope", "mean"),
                                         placebo_sd=("placebo_slope", "std"))
    pl["z_vs_placebo"] = (pl["real"] - pl["placebo_mean"]) / pl["placebo_sd"]
    log(f"Elasticity: {int(el['e_raw'].notna().sum())} items with own variation; pooled median {el['elasticity'].median():.2f}")
    bt_events, bt = backtest(df, products, cats, train_end, data_end)
    bt_events.round(4).to_csv(OUTPUT_DIR / "backtest_price_changes.csv", index=False)
    log(f"Backtest: {bt}")
    plot_elasticity(el, placebo, bt_events)

    # ---------------- STAGE 1c: competitor (SIMULATED) ----------------
    lp_all = list_price_daily(df, products, data_end)
    panel = attach_competitor(panel, lp_all.groupby("StockCode").median().to_dict())
    comp_ref = panel[~panel["is_future"] & (panel["Date"] > data_end - pd.Timedelta(days=28))] \
        .groupby("StockCode")["comp_price"].median()

    # ---------------- STAGE 2: decisions ----------------
    # ONE price definition everywhere: the regular-size-line list price.
    lp = lp_all.reset_index()
    lp["week"] = lp["Date"].dt.to_period("W-SUN")
    wk_lp = lp.groupby(["StockCode", "week"])["list_price"].median()
    hist_min, hist_max = wk_lp.groupby("StockCode").min(), wk_lp.groupby("StockCode").max()
    recent_lp = lp[lp["Date"] > data_end - pd.Timedelta(days=28)].groupby("StockCode")["list_price"].median()
    last_lp = lp.sort_values("Date").groupby("StockCode")["list_price"].last()
    med_lp = lp.groupby("StockCode")["list_price"].median()
    total_rev = df.groupby("StockCode")["Revenue"].sum()
    recent_units = panel[~panel["is_future"] & (panel["store_open"] == 1) & (panel["Date"] > data_end - pd.Timedelta(days=91))]
    log_sd = recent_units.groupby("StockCode")["units"].agg(lambda u: np.log1p(u).std(ddof=1))

    z = np.random.default_rng(SEED).standard_normal(N_ELASTICITY_DRAWS)
    recs, outs = [], {}
    for code in products:
        cat = cats[code]
        p0 = round(float(recent_lp.get(code, last_lp[code])), 2)
        r = {"StockCode": code, "Description": desc[code], "category": cat, "variant_group": variant_key(desc[code]),
             "total_revenue": float(total_rev[code]), "current_price": p0,
             "hist_min_price": float(hist_min[code]), "hist_max_price": float(hist_max[code]),
             "unit_cost": float(COST_OVERRIDES_PRODUCT.get(code, COST_RATIO_BY_CATEGORY.get(cat, COST_RATIO_DEFAULT) * med_lp[code])),
             "baseline_units": float(baseline_units.get(code, 0.0)),
             "comp_price_SIMULATED": float(comp_ref[code]), "daily_log_sd": float(log_sd.get(code, 1.0)),
             **el.loc[code, ["e_raw", "se_raw", "n_weeks", "n_price_levels", "elasticity", "elasticity_se",
                             "elasticity_source"]].to_dict()}
        outs[code] = core.optimise_product(r, z, stock=STOCK_CAP.get(code, math.inf))
        recs.append(r)

    # Jointly price colour/design variants that currently share a price.
    groups = {}
    for r in recs:
        groups.setdefault((r["variant_group"], r["current_price"]), []).append(r)
    joint = 0
    for (_, p0), members in groups.items():
        if len(members) < 2:
            continue
        b = [outs[m["StockCode"]]["bounds"] for m in members]
        lo, hi = max(x[0] for x in b), min(x[1] for x in b)
        if lo > hi:
            continue
        cands = np.unique(np.r_[core.price_grid(lo, hi), [p0] if lo <= p0 <= hi else []])
        if len(cands) == 0:
            continue
        tot = 0
        for m in members:
            e_d = core.elasticity_draws(m["elasticity"], m["elasticity_se"], z)
            tot = tot + (cands - m["unit_cost"])[None, :] * core.demand_matrix(cands, p0, m["baseline_units"], e_d)
        score = tot.mean(0) if core.ROBUST_QUANTILE is None else np.quantile(tot, core.ROBUST_QUANTILE, axis=0)
        pj = float(cands[int(np.argmax(score))])
        k0 = np.where(np.isclose(cands, p0))[0]
        if len(k0) and not np.isclose(pj, p0):     # same cross-check as single items, on the group total
            half = np.arange(len(z)) % 2 == 0
            g = [tot[~pick][:, int(np.argmax(tot[pick].mean(0)))].mean() - tot[~pick][:, k0[0]].mean()
                 for pick in (half, ~half)]
            if np.mean(g) <= 0:
                pj = p0
        for m in members:
            outs[m["StockCode"]] = {**core.optimise_product(m, z, stock=STOCK_CAP.get(m["StockCode"], math.inf),
                                                            forced_price=pj),
                                    "bounds": outs[m["StockCode"]]["bounds"]}
            m["joint_with"] = len(members) - 1
            joint += 1
    log(f"Jointly priced {joint} items in variant groups")

    band_of = lambda u: int(np.searchsorted(band_edges, u))  # noqa: E731
    for r in recs:
        o = outs[r["StockCode"]]
        b = o["comp_same"]
        p10, p90 = core.profit_range(b["profit_draws"], ratio_samples[band_of(r["baseline_units"])])
        r.update({"recommended_price": b["price"], "price_change_%": 100 * (b["price"] / r["current_price"] - 1),
                  "expected_units": b["units"], "expected_profit": b["profit"], "profit_p10": p10, "profit_p90": p90,
                  "units_at_current": b["units_current"], "profit_at_current": b["profit_current"],
                  "profit_change": b["profit"] - b["profit_current"],
                  "profit_change_crossfit": b.get("profit_change_crossfit", b["profit"] - b["profit_current"]),
                  "p_improve": b["p_improve"], "regret": b["regret"], "downside": b["downside"],
                  "flags": o["flags"], "decision_driver": o["decision_driver"], "at_bound": o["at_bound"],
                  "bound_lo": o["bounds"][0], "bound_hi": o["bounds"][1], "joint_with": r.get("joint_with", 0),
                  **core.lerner(min(r["elasticity"], core.ELASTICITY_CEILING), r["unit_cost"], r["current_price"])})
        for scen in core.COMP_SCENARIOS:
            s = o[scen]
            r.update({f"{scen}_price": s["price"], f"{scen}_units": s["units"], f"{scen}_profit": s["profit"],
                      f"{scen}_profit_at_current": s["profit_current"]})
        r["confidence"], r["confidence_reason"] = core.confidence(r)
        r.update(core.test_plan(r["current_price"], r["recommended_price"], r["daily_log_sd"], *o["bounds"]))
        r["flags"] = "; ".join(r["flags"])
    rec = pd.DataFrame(recs).sort_values("total_revenue", ascending=False).reset_index(drop=True)
    rec["lerner_gap"] = rec["unconstrained_price"] / rec["current_price"]
    # Single items are often too noisy to test alone; testing the same % change on
    # k similar items of the category at once cuts the duration roughly k-fold.
    n_cat = rec.groupby("category")["StockCode"].transform("size")
    rec["test_weeks_pooled_category"] = np.ceil(rec["test_weeks_total"] / n_cat).astype(int)
    top = rec.head(TOP_N_REPORT).copy()

    csv_cols = ["StockCode", "Description", "category", "variant_group", "joint_with", "total_revenue",
                "current_price", "recommended_price", "price_change_%", "decision_driver", "at_bound",
                "expected_units", "expected_profit", "profit_p10", "profit_p90", "units_at_current",
                "profit_at_current", "profit_change", "profit_change_crossfit", "p_improve", "downside", "regret",
                "elasticity", "elasticity_se", "e_raw", "n_weeks", "n_price_levels", "elasticity_source",
                "confidence", "confidence_reason", "flags", "lerner_margin", "current_margin", "unconstrained_price",
                "lerner_gap", "unit_cost", "baseline_units", "hist_min_price", "hist_max_price", "bound_lo",
                "bound_hi", "comp_price_SIMULATED", "test_price", "test_days_per_arm", "test_weeks_total",
                "test_feasible", "test_weeks_pooled_category", "daily_log_sd"] + \
               [f"{s}_{k}" for s in core.COMP_SCENARIOS for k in ("price", "units", "profit", "profit_at_current")]
    top[csv_cols].round(4).to_csv(OUTPUT_DIR / "pricing_recommendations.csv", index=False)
    rec[csv_cols].round(4).to_csv(OUTPUT_DIR / "pricing_recommendations_all_products.csv", index=False)
    plan = rec[rec["confidence"] != "high"].sort_values("profit_change_crossfit", ascending=False)
    plan[["StockCode", "Description", "category", "confidence", "current_price", "test_price", "test_days_per_arm",
          "test_weeks_total", "test_feasible", "test_weeks_pooled_category", "profit_change_crossfit"]].round(4) \
        .to_csv(OUTPUT_DIR / "price_test_plan.csv", index=False)
    el.round(4).to_csv(OUTPUT_DIR / "elasticity_estimates.csv")
    plot_profit_curves(top, z)
    plot_scenarios(top, panel)

    # ---------------- summary.md ----------------
    def gains(d):
        g = d.groupby("confidence")["profit_change_crossfit"].sum().reindex(["high", "medium", "low"]).fillna(0)
        return g
    g_top, g_all = gains(top), gains(rec)
    conf_counts = top["confidence"].value_counts().reindex(["high", "medium", "low"]).fillna(0).astype(int)
    drivers = rec["decision_driver"].str.split(":").str[0].value_counts()
    implausible = float((rec["lerner_gap"] > 1.5).mean())
    best_row = res_tbl.loc[best]
    best_naive = res_tbl.loc[NAIVE_MODELS, "test_WAPE%_7day"].min()
    best_ml = res_tbl.loc[ML_MODELS, "test_WAPE%_7day"].min()
    show = top.sort_values("profit_change_crossfit", ascending=False).head(15)[
        ["StockCode", "Description", "current_price", "recommended_price", "profit_change_crossfit",
         "elasticity", "elasticity_source", "decision_driver", "confidence"]].copy()
    show["Description"] = show["Description"].str[:30]
    mt = res_tbl.reset_index()[["model", "kind", "calibration", "select_WAPE%_7day", "test_WAPE%_7day",
                                "test_Bias%_7day", "selected"]]
    srcs = el["elasticity_source"].value_counts()
    bt_line = ("no qualifying price changes after the cut-off" if bt.get("n_events", 0) == 0 else
               f"{bt['n_events']} list-price changes on {bt['n_items']} items. Mean absolute error of the predicted log "
               f"change in units: **{bt['mae_model']:.3f}** with the model vs **{bt['mae_no_effect']:.3f}** assuming "
               f"price has no effect; direction correct in {bt['direction_hit_rate']:.0%}; correlation {bt['corr']:.2f}. "
               f"Median elasticity implied by these events: {bt['implied_elasticity_median']:.2f} vs pooled model median "
               f"{el['elasticity'].median():.2f}. "
               + ("**Real price changes imply STRONGER price sensitivity than the model**, so recommended price increases are "
                  "likely too large; weight the price tests above the model." if bt['implied_elasticity_median'] < el['elasticity'].median()
                  else "Real price changes do not suggest the model underestimates price sensitivity.")
               + (f" Only {bt['n_events']} events: treat this as indicative." if bt['n_events'] < 30 else ""))
    summary = f"""# Profit-maximising pricing: UCI Online Retail

Generated by `src/pricing_profit_optimizer.py`. Data {df['Date'].min().date()} to {data_end.date()}.
Decision horizon: the {HORIZON_DAYS} days after the data ends ({fut['Date'].min().date()} to {fut['Date'].max().date()}).
Objective: **profit = (price − unit cost) × expected units**.

> **Assumptions:** competitor prices are **SIMULATED** (seeded) and only shift demand in the `comp_*` scenarios.
> Unit costs are **ASSUMED** ({COST_RATIO_DEFAULT:.0%} of each item's median list price). Absolute £ figures are illustrative.

## 0. Data
- Cancellations are **netted** against the same customer's earlier orders: {clean_stats['cancel_units_matched']:,.0f} units removed,
  {clean_stats['cancel_units_unmatched']:,.0f} units of cancellations had no earlier order in the data.
- Lines without a CustomerID ({clean_stats['guest_line_share']:.0%} of lines, {clean_stats['guest_revenue_share']:.0%} of revenue,
  mostly guest/retail buyers) are **{'included' if clean_stats['guests_included'] else 'excluded'}** (`INCLUDE_GUEST_LINES`).
  Recommendations therefore describe {'all' if clean_stats['guests_included'] else 'registered (mainly wholesale) customers'}.
- One price definition everywhere: the **list price** = median unit price of regular-size lines (bulk tier discounts excluded).

## 0b. Holidays and shop closures (checked against the data)
- Closure calendar: no trading on Saturdays, England & Wales bank holidays (incl. substitute days and the 2011 Royal
  Wedding), Easter Sunday, and 24 Dec – 2 Jan. Checked against all {hol['days_checked']} days in the data: the calendar marks
  {hol['closed_days']} days closed; mismatches with the actual trading days: **{len(hol['calendar_wrong_open']) + len(hol['calendar_wrong_closed'])}**
  {('(' + ', '.join(hol['calendar_wrong_open'] + hol['calendar_wrong_closed']) + ')') if hol['calendar_wrong_open'] or hol['calendar_wrong_closed'] else ''}.
- The first trading day after a closure of 2+ days sells **{hol['reopening_effect']:+.0%}** vs the previous 10 trading days
  (median of {hol['reopening_days']} reopenings): it is a slow day, not a catch-up day.
- Gifting events (wholesale customers stock up ahead of them) are model features: days to the next event, which
  event, days since the last one, closures in the next/previous 7 days and reopening days.

{md_table(pd.DataFrame(hol['events']))}

- Beyond next week, daily forecasts use next week's level × last year's pattern for the same days (store-wide,
  weekday-matched), because the model has seen only one Christmas. This part is not validated: only one prior year exists.

## 1. Baseline demand model
- Splits: train ≤ {train_end.date()}, calibration weeks ≤ {val_cal_end.date()}, selection weeks ≤ {val_end.date()}, test ≤ {data_end.date()}.
- **All {len(res_tbl)} candidates, including the naive baselines, get the same level calibration** (fitted on the calibration
  weeks) and are selected on the later selection weeks. Test weeks are never used for selection.
- **Selected: {best}.** Test WAPE on 7-day product totals: {best_row['test_WAPE%_7day']:.1f}% (best ML {best_ml:.1f}%, best naive {best_naive:.1f}%).
- Production level factor {calib_prod:.3f}, measured by refitting on data up to {val_end.date()} and scoring the test weeks,
  so the factor matches the refit model rather than the train-only model.

{md_table(mt)}

## 2. Own-price elasticity (REAL data)
- Poisson (PPML) panel per category with item and date fixed effects on every trading day (zero-sale days included).
  Item slopes use the category's date effects and Newey-West errors; category slopes use item-clustered errors.
- Item-level estimate needs ≥ {ELAST_MIN_WEEKS} weeks with sales and ≥ {ELAST_MIN_PRICE_LEVELS} weekly list prices each held ≥ {ELAST_MIN_WEEKS_PER_LEVEL} **weeks**.
- Evidence: {srcs.get(core.SRC_OWN, 0)} items "{core.SRC_OWN}", {srcs.get(core.SRC_PRIOR, 0)} "{core.SRC_PRIOR}",
  {srcs.get(core.SRC_CONTRA, 0)} "{core.SRC_CONTRA}" (own slope above {core.ELASTICITY_CEILING}; not used, item flagged low confidence).
- **Level calibration:** daily prices are measured with noise, which pulls panel slopes toward zero. The shop's
  {level['n_events'] if level else 0} real list-price changes (sales {EVENT_WINDOW_WEEKS} weeks before vs after, net of the item's category) give a pooled
  elasticity of **{level['elasticity'] if level else float('nan'):.2f}** (s.e. {level['se'] if level else float('nan'):.2f}). All panel slopes are rescaled by
  **{level_k:.2f}** to match that level; the panel still decides how categories and items differ.
- A category slope is used only with ≥ {MIN_CLUSTERS_CATEGORY} items that changed price; otherwise the global prior
  ({g_mu:.2f}, between-item sd {g_tau:.2f}) is used. No posterior elasticity is above {core.ELASTICITY_CEILING}.
- Pooled elasticity: median {el['elasticity'].median():.2f}, IQR [{el['elasticity'].quantile(.25):.2f}, {el['elasticity'].quantile(.75):.2f}].

{md_table(prior_tbl.reset_index().rename(columns={'index': 'category'}).sort_values('prior_mean'))}

**Placebo test** ({N_PLACEBO} time-shuffled price paths per category). A real slope well below its placebo distribution
(z < −2) is evidence the estimate is not an artefact of the panel design:

{md_table(pl.reset_index())}

**Out-of-sample backtest.** Elasticities re-estimated on data up to {train_end.date()} predict the effect of later real
price changes: {bt_line}

## 3. Recommendations
- Decision drivers across all {len(rec)} items: {', '.join(f'{k}: {v}' for k, v in drivers.items())}.
  Items at a limit would move further without it: **the limit, not the data, sets their price.**
- **Lerner check:** for {implausible:.0%} of items the elasticity implies an unconstrained optimal price more than 1.5× the
  current price (optimal margin −1/e far above the assumed {1 - COST_RATIO_DEFAULT:.0%}). Either elasticities are biased toward zero
  by non-experimental price variation or costs are wrong; treat price *increases* at the upper limit with caution.
- Confidence (top {TOP_N_REPORT}): high {conf_counts['high']}, medium {conf_counts['medium']}, low {conf_counts['low']}.
  **High requires the item's own price evidence**; category-only items are capped at medium.
- **Expected profit gain, cross-fitted** (price chosen on half the elasticity draws, scored on the other half):

| scope | high | medium | low | high + medium |
|---|---|---|---|---|
| top {TOP_N_REPORT} | £{g_top['high']:,.0f} | £{g_top['medium']:,.0f} | £{g_top['low']:,.0f} | **£{g_top['high'] + g_top['medium']:,.0f}** |
| all {len(rec)} | £{g_all['high']:,.0f} | £{g_all['medium']:,.0f} | £{g_all['low']:,.0f} | **£{g_all['high'] + g_all['medium']:,.0f}** |

  Baseline profit at current prices (top {TOP_N_REPORT}): £{top['profit_at_current'].sum():,.0f}. These gains are model
  predictions; the backtest above is the only out-of-sample evidence on the elasticities behind them.
- P10–P90 profit ranges include **both** elasticity uncertainty and baseline-forecast error (empirical test-week
  actual/forecast ratios by volume band).
- Colour/design variants sharing a price are priced **jointly** ({int((rec['joint_with'] > 0).sum())} items).
- A **price-test plan** (switchback design, 5% significance, 80% power to tell elasticity −1 from −2) is in
  `price_test_plan.csv` for every item below high confidence; {int(plan['test_feasible'].sum())} of {len(plan)} are feasible within {core.MAX_TEST_WEEKS} weeks
  as single-item tests, {int((plan['test_weeks_pooled_category'] <= core.MAX_TEST_WEEKS).sum())} when the same % change is tested across the item's category.

{md_table(show)}

## 4. Competitor scenarios (SIMULATED competitor)
The competitor enters as demand × (competitor level / reference)^{core.CROSS_ELASTICITY}. It does **not** add to the own-price
elasticity (the real-data elasticity already includes competitor responses), so a level shift changes profit, not the price.

{md_table(pd.DataFrame([{"scenario": s, "profit @ current £": top[f"{s}_profit_at_current"].sum(),
                         "profit @ recommended £": top[f"{s}_profit"].sum()} for s in core.COMP_SCENARIOS]))}

## 5. Limitations
- Price changes in this data are not experiments; the placebo test and backtest bound, but do not remove, the bias.
  Run the price tests in `price_test_plan.csv` before acting on low/medium-confidence items.
- Constant elasticity, applied only within ±{core.MAX_PRICE_CHANGE:.0%} of current prices.
- The horizon is the week before Christmas 2011. Re-run with `--as-of` to see other weeks.
- Costs are assumed; set `COST_OVERRIDES_PRODUCT` / `COST_RATIO_BY_CATEGORY`. Competitor data is simulated unless `COMPETITOR_PATH` is set.
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    # ---------------- artifacts for the apps (portable formats) ----------------
    app_cols = csv_cols + ["joint_with"]
    rec[list(dict.fromkeys(app_cols))].to_parquet(OUTPUT_DIR / "app_products.parquet", index=False)
    meta = {"horizon_start": str(fut["Date"].min().date()), "horizon_end": str(fut["Date"].max().date()),
            "data_end": str(data_end.date()), "model": best, "test_wape_7day": float(best_row["test_WAPE%_7day"]),
            "best_naive_test_wape_7day": float(best_naive), "calibration_production": calib_prod,
            "calibration_source": "refit-to-test check", "z": z.round(6).tolist(),
            "forecast_band_edges": band_edges, "forecast_ratio_samples": [s.round(5).tolist() for s in ratio_samples],
            "backtest": bt, "holidays": hol, "calendar": day_calendar(daily_fc["Date"].unique()), "elasticity_level": level, "elasticity_level_scale": level_k, "clean_stats": clean_stats, "gains_top": g_top.to_dict(), "gains_all": g_all.to_dict()}
    (OUTPUT_DIR / "app_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log("PIPELINE COMPLETE")
    return rec, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", help="pretend the data ends on this date (YYYY-MM-DD); outputs go to outputs/asof_<date>/")
    run(ap.parse_args().as_of)


if __name__ == "__main__":
    main()
