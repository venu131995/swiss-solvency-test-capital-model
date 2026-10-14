"""Full study: SST ratio today, capital breakdown, IFRS-style bridge, sensitivities, history, backtest,
verification. Writes results/summary.json and figures/fig*.png."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sst import (PERILS, BalanceSheet, block_bootstrap, book_bridge, expected_shortfall, load_factors,  # noqa: E402
                 natcat_mean, overlapping_annual, revalue, run_sst, state, value_at_risk)
from sst.balance_sheet import T  # noqa: E402
from sst.risk import H, daily_changes  # noqa: E402

FIG, RES = ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.linewidth": 1.2, "axes.titleweight": "bold"})
t0 = time.time()
f = load_factors(ROOT / "data")
today = f.index[-1]
out = {"valuation_date": str(today.date()), "history": [str(f.index[0].date()), str(today.date())]}

# ------------------------------------------------------------------ 1. SST today
r = run_sst(f, today, n=200_000)
out["sst"] = {k: round(float(r[k]), 4) for k in ("rbc", "mvm", "es", "tc", "ratio", "es_market", "es_insurance",
                                                  "var_market", "sc0")}
out["sst"]["stand_alone_market_es"] = {k: round(float(v), 4) for k, v in r["stand_alone"].items()}
ins = r["insurance"]
out["sst"]["stand_alone_insurance_es"] = {k: round(float(expected_shortfall(-v)), 4)
                                          for k, v in ins.items() if not k.startswith("_") and k != "new business (expected)"}
out["sst"]["natcat_expected_loss"] = round(natcat_mean(), 4)
out["sst"]["expected_new_business_result"] = round(float(ins["new business (expected)"][0]), 4)
out["balance_sheet"] = {k: round(float(v), 3) for k, v in r["base"].items()}

# ------------------------------------------------------------------ 2. verification
bs, s0 = r["bs"], state(f, today)
ver = {}
# (a) every bond ladder prices exactly to its stated market value on the build curve
ver["pricing_identity_max_abs_error"] = float(max(abs(r["base"][k] - getattr(bs, k)) for k in ("gov_usd", "gov_eur", "corp_usd")))
# (b) DV01 by full revaluation vs analytic sum t*CF*DF (1 bp parallel USD shift)
dz = 1e-4
up = bs.values(s0["usd_z"] + dz, s0["eur_z"], s0["spread"])
dv01_num = float(up["rbc"] - r["base"]["rbc"])
du = np.exp(-s0["usd_z"] * T)
dc = np.exp(-(s0["usd_z"] + s0["spread"]) * T)
dv01_an = float(-dz * (T * du) @ (bs.cf["gov_usd"] - bs.cf["pc_usd"] - bs.cf["lh_usd"]) - dz * (T * dc) @ bs.cf["corp_usd"])
ver["dv01_full_vs_analytic"] = {"full": dv01_num, "analytic_first_order": dv01_an,
                                "rel_diff": abs(dv01_num - dv01_an) / abs(dv01_an)}
# (c) ES estimator vs closed form (normal and Student-t, 2e6 samples)
rng = np.random.default_rng(7)
x = rng.standard_normal(2_000_000)
es_n = stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01
nu = 4
xt = rng.standard_t(nu, 2_000_000)
q = stats.t.ppf(0.99, nu)
es_t = stats.t.pdf(q, nu) / 0.01 * (nu + q * q) / (nu - 1)
ver["es_estimator"] = {"normal_sample": expected_shortfall(x), "normal_exact": es_n,
                       "t4_sample": expected_shortfall(xt), "t4_exact": es_t}
# (d) Monte Carlo noise of the SST ratio across 10 seeds
ratios = [run_sst(f, today, n=200_000, seed=s)["ratio"] for s in range(10)]
ver["ratio_seed_mean_sd"] = [float(np.mean(ratios)), float(np.std(ratios, ddof=1))]
# (e) bootstrap vs overlapping-history scenario sets (market ES)
ro = run_sst(f, today, method="overlap")
ver["market_es_bootstrap_vs_overlapping"] = [r["es_market"], ro["es_market"]]
# (f) balance-sheet identity in every scenario: RBC = assets - BEL
v = revalue(bs, s0, r["dX"][:1000], r["cols"])
ver["identity_max_abs_error"] = float(np.abs(v["rbc"] - (v["assets"] - v["bel"])).max())
# (g) nat-cat simulated mean vs exact capped-Pareto mean
cat = sum(ins["_cat_by_peril"].values())
ver["natcat_mean_sim_vs_exact"] = [float(cat.mean()), natcat_mean(), float(cat.std() / np.sqrt(cat.size))]
out["verification"] = ver

# ------------------------------------------------------------------ 3. bridge book -> SST
locked = f.loc[today - np.timedelta64(5 * 365, "D"):today]
lu = locked[[f"usd_z{t}" for t in range(1, 31)]].mean().to_numpy()
le = locked[[f"eur_z{t}" for t in range(1, 31)]].mean().to_numpy()
bridge = book_bridge(r, lu, le)
out["bridge"] = {k: round(v, 4) for k, v in bridge}
out["bridge_closure_error"] = abs(sum(v for _, v in bridge) - (r["rbc"] - r["mvm"]))

# ------------------------------------------------------------------ 4. sensitivities (instantaneous)
cols = r["cols"]


def shock(**kw):
    d = np.zeros((1, len(cols)))
    for c in cols:
        if c.startswith("usd_z"):
            d[0, cols.index(c)] = kw.get("usd", 0)
        if c.startswith("eur_z"):
            d[0, cols.index(c)] = kw.get("eur", 0)
    d[0, cols.index("spread")] = kw.get("spread", 0)
    d[0, cols.index("spx")] = np.log1p(kw.get("eq", 0))
    d[0, cols.index("eurusd")] = np.log1p(kw.get("fx", 0))
    return float(revalue(bs, s0, d, cols)["rbc"][0] - r["base"]["rbc"])


hur = next(p for p in PERILS if p.name.startswith("North Atlantic"))
rp200 = min(hur.xmin * (hur.freq / (1 / 200)) ** (1 / hur.alpha), hur.cap)   # 200-yr single-event loss
sens = {"Rates -100 bp": shock(usd=-0.01, eur=-0.01), "Rates +100 bp": shock(usd=0.01, eur=0.01),
        "Credit spreads +100 bp": shock(spread=0.01), "Equities -30 %": shock(eq=-0.30),
        "EUR/USD -15 %": shock(fx=-0.15), "1-in-200 hurricane": -rp200}
out["sensitivities"] = {}
for k, d_rbc in sens.items():
    # ratio after the shock: RBC moves, ES and MVM held fixed (first-order, as in SST sensitivity reporting)
    out["sensitivities"][k] = {"delta_rbc": round(d_rbc, 4),
                               "ratio": round((r["rbc"] + d_rbc - r["mvm"]) / r["es"], 4)}

# ------------------------------------------------------------------ 5. history: ratio at each year-end, rebuilt
hist = []
for y in range(2008, 2026):
    d = f.loc[:f"{y}-12-31"].index[-1]
    rr = run_sst(f, d, n=50_000, seed=y)
    hist.append({"date": str(d.date()), "es_market": rr["es_market"],
                 "usd10": float(f.loc[d, "usd_z10"])})
out["history_market_es"] = hist

# ------------------------------------------------------------------ 6. backtest of market VaR99 (monthly starts)
starts = f.loc["2008-09-01":f.index[-H - 1]].resample("MS").first().index
bt = []
for d in starts:
    d = f.index[f.index.get_indexer([d], method="bfill")[0]]
    i = f.index.get_loc(d)
    b = BalanceSheet()
    s = state(f, d)
    b.build(s["usd_z"], s["eur_z"], s["spread"])
    base = b.values(s["usd_z"], s["eur_z"], s["spread"])["rbc"]
    sc = block_bootstrap(daily_changes(f.iloc[:i + 1]), 20_000, seed=i)
    loss = -(revalue(b, s, sc, cols)["rbc"] - base)
    real = (f.iloc[i + H] - f.iloc[i]).to_numpy()[None, :]
    rl = float(-(revalue(b, s, real, cols)["rbc"][0] - base))
    bt.append({"date": str(d.date()), "var99": value_at_risk(loss), "es99": expected_shortfall(loss), "realised": rl})
exc = sum(b["realised"] > b["var99"] for b in bt)
n_bt = len(bt)
p = 0.01
lr = -2 * (np.log((1 - p) ** (n_bt - exc) * p ** exc + 1e-300)
           - np.log((1 - exc / n_bt) ** (n_bt - exc) * max(exc / n_bt, 1e-300) ** exc))
out["backtest"] = {"windows": n_bt, "independent_years": round(n_bt / 12, 1), "exceedances": int(exc),
                   "expected": round(p * n_bt, 2), "kupiec_LR": float(lr), "kupiec_p": float(1 - stats.chi2.cdf(lr, 1)),
                   "worst": max(bt, key=lambda b: b["realised"])}
out["runtime_s"] = round(time.time() - t0, 1)
(RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
(RES / "backtest.json").write_text(json.dumps(bt, indent=0))

# ------------------------------------------------------------------ figures
def box(ax):
    for sp in ax.spines.values():
        sp.set_linewidth(1.4)
    ax.tick_params(width=1.4)


# fig1: curves
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for d, c in [("2008-12-31", "tab:blue"), ("2020-12-31", "tab:orange"), (str(today.date()), "k")]:
    dd = f.loc[:d].index[-1]
    ax[0].plot(T, 100 * f.loc[dd, [f"usd_z{t}" for t in range(1, 31)]], color=c, lw=2, label=str(dd.date()))
    ax[1].plot(T, 100 * f.loc[dd, [f"eur_z{t}" for t in range(1, 31)]], color=c, lw=2, label=str(dd.date()))
for a, t in zip(ax, ["USD zero curve (bootstrapped from Treasuries)", "EUR AAA zero curve (ECB)"]):
    a.set_title(t); a.set_xlabel("maturity (years)"); a.set_ylabel("zero rate (%)"); a.legend(frameon=True); box(a)
fig.tight_layout(); fig.savefig(FIG / "fig1_curves.png", dpi=160); plt.close(fig)

# fig2: one-year result distributions
fig, ax = plt.subplots(figsize=(9, 4.5))
for x, lab, c in [(r["market"], "market", "tab:blue"), (r["total"] - r["market"], "insurance", "tab:orange"),
                  (r["total"], "total", "k")]:
    ax.hist(x, bins=300, density=True, histtype="step", lw=2, color=c, label=lab)
ax.axvline(-r["es"], color="tab:red", lw=2, ls="--", label=f"ES 99% total = {r['es']:.1f}")
ax.set_xlim(-20, 10); ax.set_xlabel("one-year change in risk-bearing capital (USD bn)"); ax.set_ylabel("density")
ax.legend(frameon=True); ax.set_title("One-year result distribution"); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig2_distribution.png", dpi=160); plt.close(fig)

# fig3: target capital build-up
sa_m = out["sst"]["stand_alone_market_es"]; sa_i = out["sst"]["stand_alone_insurance_es"]
items = list(sa_m.items()) + list(sa_i.items()) + [("expected new-business result", -out["sst"]["expected_new_business_result"])]
div = r["es"] - sum(v for _, v in items)
labels = [k for k, _ in items] + ["diversification", "MVM", "target capital", "RBC"]
vals = [v for _, v in items] + [div, r["mvm"]]
fig, ax = plt.subplots(figsize=(11, 4.8))
cum = 0
for i, v in enumerate(vals):
    ax.bar(i, abs(v), bottom=min(cum, cum + v), color="tab:green" if v < 0 else "tab:gray", edgecolor="k")
    cum += v
ax.bar(len(vals), r["tc"], color="tab:red", edgecolor="k")
ax.bar(len(vals) + 1, r["rbc"], color="tab:blue", edgecolor="k")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_ylabel("USD bn"); ax.set_title(f"Target capital build-up (ES 99 %): SST ratio {100 * r['ratio']:.0f} %"); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig3_target_capital.png", dpi=160); plt.close(fig)

# fig4: bridge
fig, ax = plt.subplots(figsize=(9, 4.2))
cum = 0
for i, (k, v) in enumerate(bridge):
    ax.bar(i, abs(v), bottom=0 if i == 0 else min(cum, cum + v), edgecolor="k",
           color="tab:blue" if i == 0 else ("tab:green" if v >= 0 else "tab:red"))
    cum += v
ax.bar(len(bridge), cum, color="tab:purple", edgecolor="k")
ax.set_xticks(range(len(bridge) + 1)); ax.set_xticklabels([k for k, _ in bridge] + ["RBC - MVM (SST numerator)"],
                                                           rotation=25, ha="right")
ax.set_ylabel("USD bn"); ax.set_title("Bridge: book equity to SST capital"); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig4_bridge.png", dpi=160); plt.close(fig)

# fig5: sensitivities
fig, ax = plt.subplots(figsize=(9, 4.2))
ks = list(out["sensitivities"]); rs = [100 * out["sensitivities"][k]["ratio"] for k in ks]
ax.barh(ks, [x - 100 * r["ratio"] for x in rs], left=100 * r["ratio"], color="tab:gray", edgecolor="k")
ax.axvline(100 * r["ratio"], color="k", lw=2); ax.axvline(100, color="tab:red", lw=2, ls="--", label="100 % threshold")
for i, x in enumerate(rs):
    ax.text(x, i, f" {x:.0f} % " , va="center", fontweight="bold", ha="left" if x >= 100 * r["ratio"] else "right")
ax.set_xlabel("SST ratio (%)"); ax.set_title("Sensitivity of the SST ratio (instantaneous shocks)"); ax.legend(); box(ax)
fig.tight_layout(); fig.savefig(FIG / "fig5_sensitivities.png", dpi=160); plt.close(fig)

# fig6: history + backtest
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot([np.datetime64(h["date"]) for h in hist], [h["es_market"] for h in hist], "o-", lw=2, color="k")
ax[0].set_ylabel("market ES 99 % (USD bn)"); ax[0].set_title("Market risk calibrated at each year-end"); box(ax[0])
dts = [np.datetime64(b["date"]) for b in bt]
ax[1].plot(dts, [b["var99"] for b in bt], lw=2, color="tab:red", label="model VaR 99 % (market)")
ax[1].plot(dts, [b["realised"] for b in bt], lw=1.5, color="tab:blue", label="realised one-year loss")
ax[1].set_ylabel("USD bn"); ax[1].set_title(f"Backtest: {exc} exceedances in {n_bt} monthly windows")
ax[1].legend(frameon=True); box(ax[1])
fig.tight_layout(); fig.savefig(FIG / "fig6_history_backtest.png", dpi=160); plt.close(fig)
print(json.dumps({k: out[k] for k in ("sst", "verification", "backtest", "sensitivities", "bridge")}, indent=1,
                 default=float))
print("runtime", out["runtime_s"])
