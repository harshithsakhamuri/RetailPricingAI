"""
Pricing decision core: the maths shared by the pipeline, the Streamlit app and
(via a line-by-line port in docs/pricing.js) the static webpage.

Pure functions only: no module-level mutable state, no I/O, no ML libraries.
tests/test_parity.py checks that docs/pricing.js gives the same answers.

Demand model for one product over the decision horizon:

    units(p) = baseline_units * (p / p0) ** elasticity * comp_mult ** CROSS_ELASTICITY

`elasticity` is the TOTAL (reduced-form) own-price response estimated from real
sales, so it already contains whatever competitors did when our prices moved.
The competitor term therefore only shifts demand for a competitor price level
that differs from the reference (comp_mult = competitor price / reference); it
never adds to the own-price slope.
"""
from __future__ import annotations

import math

import numpy as np

# --- Decision settings --------------------------------------------------------
ELASTICITY_CEILING = -0.10   # draws capped here: demand must fall when price rises
CROSS_ELASTICITY = 0.5       # demand response to the competitor's price level
COMP_SCENARIOS = {"comp_minus10": 0.90, "comp_same": 1.00, "comp_plus10": 1.10}
MAX_PRICE_CHANGE = 0.20      # at most +/-20% vs current price
MIN_MARGIN = 0.20            # (price - cost) / price must be >= this
PRICE_HEADROOM = 0.10        # may go up to 10% above the highest price ever charged (flagged)
PRICE_ENDINGS = (0.05, 0.10, 0.25, 0.45, 0.49, 0.55, 0.65, 0.75, 0.79, 0.85, 0.95, 0.99)
ROBUST_QUANTILE = 0.25       # choose the price maximising the 25th-percentile profit

# --- Confidence rules ---------------------------------------------------------
RISK_HIGH = 0.02             # expected profit lost vs current price <= 2% -> may be high
RISK_MEDIUM = 0.05           # <= 5% -> may be medium
P_IMPROVE_HIGH = 0.80
P_IMPROVE_MEDIUM = 0.60
MIN_FORECAST_UNITS = 5.0

# Evidence behind an item's elasticity
SRC_OWN = "own price changes + category"
SRC_PRIOR = "category only (item never changed price)"
SRC_CONTRA = "category only (item's own data contradictory)"

# --- Price-test planner -------------------------------------------------------
TEST_Z_ALPHA = 1.959964      # two-sided 5% significance
TEST_Z_POWER = 0.841621      # 80% power
TEST_ELASTICITY_GAP = 1.0    # detect elasticity -1 vs -2 (difference of 1)
TEST_DEFAULT_STEP = 0.10     # test +10% when the recommendation is "keep price"
OPEN_DAYS_PER_WEEK = 6       # the shop trades Sunday-Friday
MAX_TEST_WEEKS = 12


def price_grid(lo: float, hi: float) -> np.ndarray:
    """Candidate prices with realistic endings: 5p steps under £1, typical retail
    endings above £1. Built from integer pence so JS and Python agree exactly."""
    pence = {5 * k for k in range(1, 20)}
    endings = [int(round(e * 100)) for e in PRICE_ENDINGS]
    for whole in range(1, int(math.floor(hi)) + 2):
        pence.add(100 * whole)
        pence.update(100 * whole + e for e in endings)
    return np.array(sorted(c / 100 for c in pence if lo - 1e-9 <= c / 100 <= hi + 1e-9))


def demand_matrix(prices, p0, base_units, e_draws, comp_mult=1.0):
    """Expected units, shape (draws, prices)."""
    prices = np.atleast_1d(np.asarray(prices, float))
    own = (prices[None, :] / p0) ** np.asarray(e_draws, float)[:, None]
    return base_units * own * comp_mult ** CROSS_ELASTICITY


def elasticity_draws(e, se, z):
    return np.minimum(e + se * np.asarray(z, float), ELASTICITY_CEILING)


def price_bounds(r: dict):
    """Allowed price range and which rule sets each end."""
    p0, cost = r["current_price"], r["unit_cost"]
    floor = cost / (1 - MIN_MARGIN)
    lo_opts = {"-20% cap": p0 * (1 - MAX_PRICE_CHANGE),
               "lowest price charged": min(r["hist_min_price"], p0),
               "minimum margin": floor}
    head = 1.0 if r.get("custom") else 1 + PRICE_HEADROOM
    hi_opts = {"+20% cap": p0 * (1 + MAX_PRICE_CHANGE),
               "highest price charged": max(r["hist_max_price"] * head, p0)}
    lo_rule = max(lo_opts, key=lo_opts.get)
    hi_rule = min(hi_opts, key=hi_opts.get)
    return lo_opts[lo_rule], hi_opts[hi_rule], lo_rule, hi_rule, floor


def _choose(profit, feasible):
    robust = np.where(feasible, np.quantile(profit, ROBUST_QUANTILE, axis=0), -np.inf)
    return int(np.argmax(robust))


def optimise_product(r: dict, z, stock: float = math.inf, forced_price: float | None = None) -> dict:
    """Robust profit-maximising price for one product.

    r needs: current_price, unit_cost, hist_min_price, hist_max_price,
    baseline_units, elasticity, elasticity_se; optional: comp_mult, custom.
    `stock` caps units (pass it in; nothing is read from global state).
    `forced_price` evaluates one given price (used for jointly priced variants)."""
    z = np.asarray(z, float)
    p0, cost = float(r["current_price"]), float(r["unit_cost"])
    comp_mult = float(r.get("comp_mult", 1.0))
    e_draws = elasticity_draws(r["elasticity"], r["elasticity_se"], z)
    lo, hi, lo_rule, hi_rule, floor = price_bounds(r)
    flags = []
    if forced_price is not None:
        cands = np.array([float(forced_price)])
    elif lo > hi:
        flags.append("no feasible price (minimum-margin price is above the allowed range)")
        cands = np.array([p0])
    else:
        grid = price_grid(lo, hi)
        cands = np.unique(np.r_[grid, [p0] if lo - 1e-9 <= p0 <= hi + 1e-9 else []])
        if len(cands) == 0:
            flags.append("no allowed price on the price grid; current price kept")
            cands = np.array([p0])
    if p0 < floor - 1e-9:
        flags.append("current price is below the minimum-margin price")

    prices = np.unique(np.r_[cands, p0])
    feasible = np.isin(prices, cands)
    k0 = int(np.where(np.isclose(prices, p0))[0][0])
    out = {}
    for scen, mult in COMP_SCENARIOS.items():
        units = np.minimum(demand_matrix(prices, p0, r["baseline_units"], e_draws, comp_mult * mult), stock)
        profit = (prices - cost)[None, :] * units
        k = _choose(profit, feasible)
        best_draw = np.where(feasible[None, :], profit, -np.inf).max(1)
        out[scen] = {
            "price": float(prices[k]), "units": float(units[:, k].mean()), "profit": float(profit[:, k].mean()),
            "profit_draws": profit[:, k], "profit_current_draws": profit[:, k0],
            "profit_current": float(profit[:, k0].mean()), "units_current": float(units[:, k0].mean()),
            "regret": float(np.mean(best_draw - profit[:, k]) / max(profit[:, k].mean(), 1e-9)),
            "downside": float(np.mean(np.maximum(profit[:, k0] - profit[:, k], 0)) / max(profit[:, k0].mean(), 1e-9)),
            "p_improve": float((profit[:, k] > profit[:, k0]).mean()) if k != k0 else float("nan"),
        }
        if scen == "comp_same":
            # Cross-fitted gain: choose on half the draws, score on the other half,
            # so the reported gain is not inflated by picking the best of noisy draws.
            a, b = np.arange(len(z)) % 2 == 0, np.arange(len(z)) % 2 == 1
            gains = []
            for pick, score in ((a, b), (b, a)):
                kk = _choose(profit[pick], feasible)
                gains.append(profit[score, kk].mean() - profit[score, k0].mean())
            out[scen]["profit_change_crossfit"] = float(np.mean(gains))

    best = out["comp_same"]["price"]
    multi = len(cands) > 1
    if not multi:
        driver = "fixed (no price choice)" if forced_price is None else "priced jointly with its variants"
        at_bound = False
    elif np.isclose(best, cands.max()) and best > p0 + 1e-9:
        driver, at_bound = f"upper limit: {hi_rule}", True
    elif np.isclose(best, cands.min()) and best < p0 - 1e-9:
        driver, at_bound = f"lower limit: {lo_rule}", True
    else:
        driver, at_bound = ("data: keep current price" if np.isclose(best, p0) else "data: interior optimum"), False
    out.update(flags=flags, bounds=(float(lo), float(hi)), decision_driver=driver, at_bound=bool(at_bound),
               e_draws=e_draws)
    return out


def lerner(e: float, cost: float, p0: float) -> dict:
    """Unconstrained profit-maximising price and margin implied by a constant
    elasticity e (Lerner rule: margin = -1/e). e >= -1 has no finite optimum."""
    cur_margin = (p0 - cost) / p0
    if e < -1:
        return {"lerner_margin": -1 / e, "unconstrained_price": cost * e / (1 + e), "current_margin": cur_margin}
    return {"lerner_margin": float("nan"), "unconstrained_price": float("inf"), "current_margin": cur_margin}


def profit_range(profit_draws, ratio_samples):
    """P10/P90 of profit including BOTH elasticity and baseline-forecast error.
    ratio_samples: empirical actual/forecast ratios (same length as the draws)."""
    d = np.asarray(profit_draws, float) * np.asarray(ratio_samples, float)
    return float(np.quantile(d, 0.10)), float(np.quantile(d, 0.90))


def confidence(r: dict):
    """Decision-based confidence. High needs BOTH low risk under the model AND the
    item's own price evidence; category-only items are capped at medium because
    their risk number only reflects the model agreeing with itself."""
    hard = []
    if r["baseline_units"] < MIN_FORECAST_UNITS:
        hard.append("very low forecast volume")
    flags = r.get("flags") or []
    hard.extend([flags] if isinstance(flags, str) else flags)
    if r["elasticity_source"] == SRC_CONTRA:
        hard.append("item's own price history shows demand not falling with price (likely confounded)")
    if hard:
        return "low", "; ".join(h for h in hard if h)
    risk, pi = r["downside"], r["p_improve"]
    pi_nan = pi is None or (isinstance(pi, float) and math.isnan(pi))
    untested = (not r.get("custom")) and r["recommended_price"] > max(r["hist_max_price"], r["current_price"]) + 1e-9
    own = r["elasticity_source"] == SRC_OWN
    notes = []
    if untested:
        notes.append(f"price is above the highest ever charged (£{r['hist_max_price']:.2f}); test before rolling out")
    if not own:
        notes.append("custom item: price sensitivity is the category average" if r.get("custom")
                     else "no price changes of its own: price sensitivity is the category average")
    if risk > RISK_HIGH:
        notes.append(f"{risk:.1%} expected profit at risk vs current price")
    if not pi_nan and pi < P_IMPROVE_HIGH:
        notes.append(f"new price beats current price in {pi:.0%} of scenarios")
    if risk <= RISK_HIGH and (pi_nan or pi >= P_IMPROVE_HIGH) and not untested and own:
        return "high", ""
    if risk <= RISK_MEDIUM and (pi_nan or pi >= P_IMPROVE_MEDIUM):
        return "medium", "; ".join(notes)
    return "low", "; ".join(notes)


def test_plan(p0: float, recommended: float, daily_log_sd: float, lo: float, hi: float) -> dict:
    """Switchback price test (alternate weeks at current vs test price).
    Days per arm needed to tell elasticity -1 from -2 at 5% significance / 80%
    power: n = 2 (z_a + z_b)^2 sd^2 / delta^2, delta = gap * |ln(test/p0)|."""
    if abs(recommended / p0 - 1) >= 0.02:
        test = recommended
    else:
        grid = price_grid(max(lo, p0 * 1.05), min(hi, p0 * (1 + MAX_PRICE_CHANGE)))
        test = float(grid[np.argmin(np.abs(grid - p0 * (1 + TEST_DEFAULT_STEP)))]) if len(grid) else p0 * 1.1
    delta = TEST_ELASTICITY_GAP * abs(math.log(test / p0))
    sd = max(float(daily_log_sd), 0.05)
    days = math.ceil(2 * (TEST_Z_ALPHA + TEST_Z_POWER) ** 2 * sd ** 2 / delta ** 2)
    weeks = math.ceil(2 * days / OPEN_DAYS_PER_WEEK)
    return {"test_price": round(test, 2), "test_days_per_arm": days, "test_weeks_total": weeks,
            "test_feasible": weeks <= MAX_TEST_WEEKS}
