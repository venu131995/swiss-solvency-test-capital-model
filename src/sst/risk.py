"""One-year risk: market scenarios from history, insurance losses by simulation, ES, MVM and the SST ratio."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

H = 250  # trading days in the one-year SST horizon


# ---------------------------------------------------------------- statistics
def expected_shortfall(loss: np.ndarray, alpha: float = 0.99) -> float:
    """ES_alpha of a loss sample (losses positive): mean of the worst (1-alpha) tail, with the boundary
    atom weighted fractionally so the estimator is exact for any sample size."""
    x = np.sort(np.asarray(loss))[::-1]
    k = (1.0 - alpha) * x.size
    n = int(np.floor(k))
    return (x[:n].sum() + (k - n) * x[n]) / k


def value_at_risk(loss: np.ndarray, alpha: float = 0.99) -> float:
    return float(np.quantile(loss, alpha))


# ---------------------------------------------------------------- market scenarios
def daily_changes(f: pd.DataFrame) -> np.ndarray:
    """Daily factor changes: absolute for rates and spread, log for equity and FX (already logged)."""
    return np.diff(f.to_numpy(), axis=0)


def overlapping_annual(f: pd.DataFrame) -> np.ndarray:
    """All overlapping one-year changes in the history (strongly autocorrelated; ~len/250 independent)."""
    a = f.to_numpy()
    return a[H:] - a[:-H]


def block_bootstrap(dx: np.ndarray, n: int, block: int = 25, seed: int = 0) -> np.ndarray:
    """One-year changes as the sum of exactly 250 daily changes drawn in contiguous blocks (keeps
    volatility clustering and cross-factor correlation within each block)."""
    assert H % block == 0, "block length must divide 250"
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, dx.shape[0] - block + 1, size=(n, H // block))
    cs = np.vstack([np.zeros(dx.shape[1]), np.cumsum(dx, axis=0)])
    return (cs[starts + block] - cs[starts]).sum(axis=1)


# ---------------------------------------------------------------- insurance risk
@dataclass
class Peril:
    name: str
    freq: float       # Poisson events per year affecting the portfolio
    xmin: float       # Pareto scale: smallest loss counted, USD bn (company share)
    alpha: float      # Pareto tail index
    cap: float        # per-event cap (treaty limits), USD bn


PERILS = [
    Peril("North Atlantic hurricane", 0.60, 0.30, 1.5, 8.0),
    Peril("Europe windstorm", 0.50, 0.20, 1.8, 3.0),
    Peril("Japan earthquake", 0.15, 0.30, 1.4, 5.0),
    Peril("California earthquake", 0.08, 0.40, 1.3, 6.0),
    Peril("Other secondary perils", 2.00, 0.05, 2.0, 1.0),
]


def natcat_losses(n: int, perils=PERILS, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for p in perils:
        k = rng.poisson(p.freq, n)
        sev = np.minimum(p.xmin * (1 - rng.random(k.sum())) ** (-1 / p.alpha), p.cap)
        out[p.name] = np.bincount(np.repeat(np.arange(n), k), weights=sev, minlength=n)
    return out


def natcat_mean(perils=PERILS) -> float:
    """Exact expected annual loss of capped Pareto severities (for the nat-cat budget)."""
    m = 0.0
    for p in perils:
        a, x0, c = p.alpha, p.xmin, p.cap
        # E[min(X,c)] for Pareto(x0,a): x0 a/(a-1) - x0^a c^(1-a)/(a-1)
        m += p.freq * (x0 * a / (a - 1) - x0 ** a * c ** (1 - a) / (a - 1))
    return m


def insurance_result(bs, n: int, seed: int = 2, reserve_cv: float = 0.06, pandemic_p: float = 0.01,
                     pandemic_shock: float = 0.08, lh_trend_sd: float = 0.01, noncat_sd: float = 0.03) -> dict:
    """One-year change in RBC from insurance (positive = gain), by source, for n simulated years."""
    rng = np.random.default_rng(seed)
    cat = natcat_losses(n, seed=seed + 1)
    cat_total = sum(cat.values())
    budget = natcat_mean()
    nb = bs.pc_premium * (1 - bs.pc_noncat_ratio) - budget                      # expected new-business result
    noncat = -bs.pc_premium * noncat_sd * rng.standard_normal(n)
    reserves = bs.pc_usd_reserves + bs.pc_eur_reserves
    s = np.sqrt(np.log(1 + reserve_cv ** 2))
    reserve_dev = -reserves * (np.exp(s * rng.standard_normal(n) - s * s / 2) - 1)  # mean zero
    lh = bs.lh_usd_cashflows
    pandemic = -lh * pandemic_shock * (rng.random(n) < pandemic_p)
    trend = -lh * lh_trend_sd * rng.standard_normal(n)
    return {"new business (expected)": np.full(n, nb), "nat cat": budget - cat_total, "non-cat": noncat,
            "reserve development": reserve_dev, "L&H pandemic": pandemic, "L&H trend": trend,
            "_cat_by_peril": cat}


# ---------------------------------------------------------------- MVM and SST ratio
def market_value_margin(bs, z_usd: np.ndarray, sc0: float, coc: float = 0.06) -> float:
    """Cost-of-capital MVM: coc x sum_t SC_t DF(t+1), with the one-year capital SC_t running off in
    proportion to the remaining best-estimate liabilities."""
    from .balance_sheet import T
    liab = bs.cf["pc_usd"] + bs.cf["pc_eur"] + bs.cf["lh_usd"]
    remaining = liab[::-1].cumsum()[::-1]                 # undiscounted liabilities outstanding at start of year t
    sc = sc0 * remaining / remaining[0]
    df_next = np.exp(-z_usd * T)                          # capital cost paid at the end of each year
    return float(coc * (sc * df_next).sum())


def sst_ratio(rbc: float, mvm: float, es: float) -> float:
    """SST ratio = (RBC - MVM) / (TC - MVM), with target capital TC = ES_99%(one-year loss) + MVM."""
    return (rbc - mvm) / es
