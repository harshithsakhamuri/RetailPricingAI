// Replays tests/fixtures/parity_cases.json through docs/pricing.js; prints JSON results.
const path = require("path"), fs = require("fs");
const PC = require(path.join(__dirname, "..", "..", "docs", "pricing.js"));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "fixtures", "parity_cases.json"), "utf8"));
const out = fx.cases.map(c => {
  const o = PC.optimise(c.row, fx.z, c.stock == null ? Infinity : c.stock, c.forced), b = o.comp_same;
  const [conf, reason] = PC.confidence({...c.row, recommended_price: b.price, downside: b.downside, p_improve: b.p_improve, flags: o.flags});
  const [p10, p90] = PC.profitRange(b.profit_draws, fx.ratios[PC.bandOf(c.row.baseline_units, fx.edges)]);
  return {name: c.name, price: b.price, profit: b.profit, units: b.units, downside: b.downside,
    p_improve: Number.isNaN(b.p_improve) ? null : b.p_improve, profit_change_crossfit: b.profit_change_crossfit,
    regret: b.regret, minus10_price: o.comp_minus10.price, plus10_profit: o.comp_plus10.profit,
    lo: o.bounds[0], hi: o.bounds[1], driver: o.decision_driver, flags: o.flags, confidence: conf, reason, p10, p90,
    test: PC.testPlan(c.row.current_price, b.price, c.row.daily_log_sd, o.bounds[0], o.bounds[1])};
});
process.stdout.write(JSON.stringify(out));
