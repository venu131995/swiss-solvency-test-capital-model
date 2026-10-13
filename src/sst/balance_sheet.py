"""Stylised reinsurer balance sheet as annual cash flows, valued market-consistently.

Every position is a vector of cash flows on years 1..30 in its own currency (plus equity and cash held as
market values). Assets and best-estimate liabilities are discounted with the current risk-free zero curve;
corporate bonds add a credit spread. The balance sheet is illustrative: sizes are of the order of a large
reinsurer (USD bn) but are NOT any company's actual figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

T = np.arange(1, 31, dtype=float)


def par_bond_ladder(z: np.ndarray, maturities: range, mv: float, spread: float = 0.0) -> np.ndarray:
    """Equal-MV ladder of annual par bonds (coupon = par yield incl. spread), scaled to total value mv."""
    df = np.exp(-(z + spread) * T)
    cf = np.zeros(30)
    for m in maturities:
        c = (1.0 - df[m - 1]) / df[:m].sum()           # par coupon at this maturity
        bond = np.zeros(30)
        bond[:m] = c
        bond[m - 1] += 1.0
        cf += bond * (mv / len(maturities))             # each bond priced at par = its MV share
    return cf


def payout(total: float, mean_term: float, horizon: int = 30) -> np.ndarray:
    """Undiscounted liability cash flows with a geometric payout pattern of the given mean term (years)."""
    q = 1.0 - 1.0 / mean_term
    w = (1 - q) * q ** (T[:horizon] - 1)
    return total * w / w.sum()


@dataclass
class BalanceSheet:
    # assets (USD bn, market value at the valuation date)
    gov_usd: float = 45.0
    gov_eur: float = 15.0          # in USD at the valuation date
    corp_usd: float = 25.0
    equity: float = 6.0
    cash: float = 5.0
    intangibles: float = 3.0       # counted in IFRS equity, not in SST risk-bearing capital
    # liabilities (undiscounted cash flows, USD bn)
    pc_usd_reserves: float = 42.0
    pc_eur_reserves: float = 14.0  # in USD at the valuation date
    pc_mean_term: float = 3.5
    lh_usd_cashflows: float = 40.0
    lh_mean_term: float = 11.0
    # one-year new business
    pc_premium: float = 24.0
    pc_noncat_ratio: float = 0.80  # expected non-cat claims + expenses as share of premium
    cf: dict = field(default_factory=dict)

    def build(self, usd_z: np.ndarray, eur_z: np.ndarray, spread: float) -> "BalanceSheet":
        """Fix the cash flows so that assets price at their stated market values on this date's curves."""
        fx = 1.0  # EUR positions are given in USD at t0; cash flows are stored in USD-at-t0 units
        self.cf = {
            "gov_usd": par_bond_ladder(usd_z, range(1, 16), self.gov_usd),
            "gov_eur": par_bond_ladder(eur_z, range(1, 11), self.gov_eur * fx),
            "corp_usd": par_bond_ladder(usd_z, range(1, 11), self.corp_usd, spread),
            "pc_usd": payout(self.pc_usd_reserves, self.pc_mean_term),
            "pc_eur": payout(self.pc_eur_reserves, self.pc_mean_term),
            "lh_usd": payout(self.lh_usd_cashflows, self.lh_mean_term),
        }
        return self

    def values(self, usd_z, eur_z, spread, spx_ret=0.0, fx_ret=0.0) -> dict:
        """Market values of every line; inputs may carry a leading scenario axis.

        usd_z, eur_z: (..., 30) zero rates; spread: (...); spx_ret, fx_ret: (...) log returns of
        S&P 500 and EUR/USD (EUR lines are revalued in USD).
        """
        usd_z, eur_z = np.asarray(usd_z), np.asarray(eur_z)
        spread = np.asarray(spread)[..., None]
        du, de = np.exp(-usd_z * T), np.exp(-eur_z * T)
        dc = np.exp(-(usd_z + spread) * T)
        fx = np.exp(np.asarray(fx_ret))
        v = {
            "gov_usd": du @ self.cf["gov_usd"],
            "gov_eur": (de @ self.cf["gov_eur"]) * fx,
            "corp_usd": dc @ self.cf["corp_usd"],
            "equity": self.equity * np.exp(np.asarray(spx_ret)),
            "cash": self.cash * np.ones_like(np.asarray(spx_ret, dtype=float)),
            "pc_usd": du @ self.cf["pc_usd"],
            "pc_eur": (de @ self.cf["pc_eur"]) * fx,
            "lh_usd": du @ self.cf["lh_usd"],
        }
        v["assets"] = v["gov_usd"] + v["gov_eur"] + v["corp_usd"] + v["equity"] + v["cash"]
        v["bel"] = v["pc_usd"] + v["pc_eur"] + v["lh_usd"]
        v["rbc"] = v["assets"] - v["bel"]
        return v
