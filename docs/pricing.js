/*
 * Pricing decision core — line-by-line port of src/pricing_core.py.
 * tests/test_parity.py runs this file in Node against Python outputs, so any
 * change here must be made in pricing_core.py too (and vice versa).
 * Works in the browser (window.PricingCore) and in Node (module.exports).
 */
(function (root) {
  "use strict";
  const CFG = {
    ELASTICITY_CEILING: -0.10, CROSS_ELASTICITY: 0.5,
    COMP_SCENARIOS: {comp_minus10: 0.90, comp_same: 1.00, comp_plus10: 1.10},
    MAX_PRICE_CHANGE: 0.20, MIN_MARGIN: 0.20, PRICE_HEADROOM: 0.10,
    PRICE_ENDINGS: [0.05, 0.10, 0.25, 0.45, 0.49, 0.55, 0.65, 0.75, 0.79, 0.85, 0.95, 0.99],
    ROBUST_QUANTILE: 0.25, RISK_HIGH: 0.02, RISK_MEDIUM: 0.05, P_IMPROVE_HIGH: 0.80, P_IMPROVE_MEDIUM: 0.60,
    MIN_FORECAST_UNITS: 5.0,
    SRC_OWN: "own price changes + category",
    SRC_PRIOR: "category only (item never changed price)",
    SRC_CONTRA: "category only (item's own data contradictory)",
    TEST_Z_ALPHA: 1.959964, TEST_Z_POWER: 0.841621, TEST_ELASTICITY_GAP: 1.0, TEST_DEFAULT_STEP: 0.10,
    OPEN_DAYS_PER_WEEK: 6, MAX_TEST_WEEKS: 12,
  };

  const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
  function quantile(a, q) {           // numpy default (linear interpolation)
    const s = Float64Array.from(a).sort(), pos = (s.length - 1) * q, lo = Math.floor(pos);
    return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (pos - lo);
  }
  const close = (a, b) => Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b);   // numpy isclose

  function priceGrid(lo, hi) {
    const pence = new Set();
    for (let k = 1; k < 20; k++) pence.add(5 * k);
    const endings = CFG.PRICE_ENDINGS.map(e => Math.round(e * 100));
    for (let w = 1; w <= Math.floor(hi) + 1; w++) { pence.add(100 * w); endings.forEach(e => pence.add(100 * w + e)); }
    return [...pence].map(c => c / 100).filter(x => x >= lo - 1e-9 && x <= hi + 1e-9).sort((a, b) => a - b);
  }
  const elasticityDraws = (e, se, z) => z.map(v => Math.min(e + se * v, CFG.ELASTICITY_CEILING));
  const unitsAt = (p, p0, base, eDraws, compMult, stock) =>
    eDraws.map(e => Math.min(base * Math.pow(p / p0, e) * Math.pow(compMult, CFG.CROSS_ELASTICITY), stock));

  function priceBounds(r) {
    const p0 = r.current_price, floor = r.unit_cost / (1 - CFG.MIN_MARGIN);
    const loOpts = {"-20% cap": p0 * (1 - CFG.MAX_PRICE_CHANGE), "lowest price charged": Math.min(r.hist_min_price, p0),
                    "minimum margin": floor};
    const head = r.custom ? 1 : 1 + CFG.PRICE_HEADROOM;
    const hiOpts = {"+20% cap": p0 * (1 + CFG.MAX_PRICE_CHANGE), "highest price charged": Math.max(r.hist_max_price * head, p0)};
    let loRule = null, hiRule = null;
    for (const k in loOpts) if (loRule === null || loOpts[k] > loOpts[loRule]) loRule = k;
    for (const k in hiOpts) if (hiRule === null || hiOpts[k] < hiOpts[hiRule]) hiRule = k;
    return {lo: loOpts[loRule], hi: hiOpts[hiRule], loRule, hiRule, floor};
  }

  function choose(profitCols, feasible, rows) {   // profitCols[j] = draws at price j
    let k = 0, best = -Infinity;
    profitCols.forEach((col, j) => {
      const v = feasible[j] ? quantile(rows ? rows.map(i => col[i]) : col, CFG.ROBUST_QUANTILE) : -Infinity;
      if (v > best) { best = v; k = j; }
    });
    return k;
  }

  function optimise(r, z, stock = Infinity, forcedPrice = null) {
    const p0 = r.current_price, cost = r.unit_cost, compMult = r.comp_mult == null ? 1 : r.comp_mult;
    const eDraws = elasticityDraws(r.elasticity, r.elasticity_se, z);
    const {lo, hi, loRule, hiRule, floor} = priceBounds(r);
    const flags = [];
    let cands;
    if (forcedPrice != null) cands = [forcedPrice];
    else if (lo > hi) { flags.push("no feasible price (minimum-margin price is above the allowed range)"); cands = [p0]; }
    else {
      cands = priceGrid(lo, hi);
      if (p0 >= lo - 1e-9 && p0 <= hi + 1e-9 && !cands.includes(p0)) cands.push(p0);
      cands.sort((a, b) => a - b);
      if (!cands.length) { flags.push("no allowed price on the price grid; current price kept"); cands = [p0]; }
    }
    if (p0 < floor - 1e-9) flags.push("current price is below the minimum-margin price");
    const prices = [...new Set([...cands, p0])].sort((a, b) => a - b);
    const feasible = prices.map(p => cands.includes(p));
    const k0 = prices.findIndex(p => close(p, p0));
    const out = {};
    for (const [scen, mult] of Object.entries(CFG.COMP_SCENARIOS)) {
      const U = prices.map(p => unitsAt(p, p0, r.baseline_units, eDraws, compMult * mult, stock));
      const P = prices.map((p, j) => U[j].map(u => (p - cost) * u));
      const k = choose(P, feasible);
      const bestDraw = z.map((_, i) => Math.max(...P.map((col, j) => feasible[j] ? col[i] : -Infinity)));
      const pk = P[k], pc = P[k0];
      const res = {
        price: prices[k], units: mean(U[k]), profit: mean(pk), profit_draws: pk, profit_current_draws: pc,
        profit_current: mean(pc), units_current: mean(U[k0]),
        regret: mean(bestDraw.map((v, i) => v - pk[i])) / Math.max(mean(pk), 1e-9),
        downside: mean(pc.map((v, i) => Math.max(v - pk[i], 0))) / Math.max(mean(pc), 1e-9),
        p_improve: k !== k0 ? pk.filter((v, i) => v > pc[i]).length / pk.length : NaN,
      };
      if (scen === "comp_same") {
        const a = z.map((_, i) => i).filter(i => i % 2 === 0), b = z.map((_, i) => i).filter(i => i % 2 === 1);
        const gains = [[a, b], [b, a]].map(([pick, score]) => {
          const kk = choose(P, feasible, pick);
          return mean(score.map(i => P[kk][i])) - mean(score.map(i => P[k0][i]));
        });
        res.profit_change_crossfit = mean(gains);
      }
      out[scen] = res;
    }
    const best = out.comp_same.price, cmax = Math.max(...cands), cmin = Math.min(...cands);
    let driver, atBound = false;
    if (cands.length <= 1) driver = forcedPrice == null ? "fixed (no price choice)" : "priced jointly with its variants";
    else if (close(best, cmax) && best > p0 + 1e-9) { driver = `upper limit: ${hiRule}`; atBound = true; }
    else if (close(best, cmin) && best < p0 - 1e-9) { driver = `lower limit: ${loRule}`; atBound = true; }
    else driver = close(best, p0) ? "data: keep current price" : "data: interior optimum";
    Object.assign(out, {flags, bounds: [lo, hi], decision_driver: driver, at_bound: atBound, e_draws: eDraws});
    return out;
  }

  function lerner(e, cost, p0) {
    const cur = (p0 - cost) / p0;
    return e < -1 ? {lerner_margin: -1 / e, unconstrained_price: cost * e / (1 + e), current_margin: cur}
                  : {lerner_margin: NaN, unconstrained_price: Infinity, current_margin: cur};
  }

  function profitRange(draws, ratios) {
    const d = draws.map((v, i) => v * ratios[i]);
    return [quantile(d, 0.10), quantile(d, 0.90)];
  }
  const bandOf = (u, edges) => edges.filter(e => e < u).length;   // numpy searchsorted (left)

  function confidence(r) {
    const hard = [];
    if (r.baseline_units < CFG.MIN_FORECAST_UNITS) hard.push("very low forecast volume");
    hard.push(...(r.flags || []));
    if (r.elasticity_source === CFG.SRC_CONTRA) hard.push("item's own price history shows demand not falling with price (likely confounded)");
    if (hard.length) return ["low", hard.filter(Boolean).join("; ")];
    const risk = r.downside, pi = r.p_improve, piNan = pi == null || Number.isNaN(pi);
    const untested = !r.custom && r.recommended_price > Math.max(r.hist_max_price, r.current_price) + 1e-9;
    const own = r.elasticity_source === CFG.SRC_OWN;
    const notes = [];
    if (untested) notes.push(`price is above the highest ever charged (£${r.hist_max_price.toFixed(2)}); test before rolling out`);
    if (!own) notes.push(r.custom ? "custom item: price sensitivity is the category average"
                                  : "no price changes of its own: price sensitivity is the category average");
    if (risk > CFG.RISK_HIGH) notes.push(`${(risk * 100).toFixed(1)}% expected profit at risk vs current price`);
    if (!piNan && pi < CFG.P_IMPROVE_HIGH) notes.push(`new price beats current price in ${Math.round(pi * 100)}% of scenarios`);
    if (risk <= CFG.RISK_HIGH && (piNan || pi >= CFG.P_IMPROVE_HIGH) && !untested && own) return ["high", ""];
    if (risk <= CFG.RISK_MEDIUM && (piNan || pi >= CFG.P_IMPROVE_MEDIUM)) return ["medium", notes.join("; ")];
    return ["low", notes.join("; ")];
  }

  function testPlan(p0, recommended, dailyLogSd, lo, hi) {
    let test;
    if (Math.abs(recommended / p0 - 1) >= 0.02) test = recommended;
    else {
      const g = priceGrid(Math.max(lo, p0 * 1.05), Math.min(hi, p0 * (1 + CFG.MAX_PRICE_CHANGE)));
      if (g.length) { let j = 0; g.forEach((x, i) => { if (Math.abs(x - p0 * 1.1) < Math.abs(g[j] - p0 * 1.1)) j = i; }); test = g[j]; }
      else test = p0 * 1.1;
    }
    const delta = CFG.TEST_ELASTICITY_GAP * Math.abs(Math.log(test / p0)), sd = Math.max(dailyLogSd, 0.05);
    const days = Math.ceil(2 * (CFG.TEST_Z_ALPHA + CFG.TEST_Z_POWER) ** 2 * sd ** 2 / delta ** 2);
    const weeks = Math.ceil(2 * days / CFG.OPEN_DAYS_PER_WEEK);
    return {test_price: Math.round(test * 100) / 100, test_days_per_arm: days, test_weeks_total: weeks,
            test_feasible: weeks <= CFG.MAX_TEST_WEEKS};
  }

  const API = {CFG, quantile, mean, priceGrid, elasticityDraws, unitsAt, priceBounds, optimise, lerner,
               profitRange, bandOf, confidence, testPlan};
  if (typeof module !== "undefined" && module.exports) module.exports = API; else root.PricingCore = API;
})(typeof self !== "undefined" ? self : this);
