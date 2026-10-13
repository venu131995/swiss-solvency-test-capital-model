"""Load the public market data used by the model into one aligned daily frame.

Sources (all public, downloaded by scripts/download_data.py):
- US Treasury constant-maturity yields (FRED DGS1..DGS30), treated as annual-coupon par yields.
- ECB euro-area AAA government spot curve (continuously compounded zero rates, 1y..30y).
- Moody's Baa corporate yield minus 10y Treasury (FRED BAA10Y): long-history BBB credit-spread proxy.
- S&P 500 index and EUR/USD (Yahoo Finance).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

USD_NODES = [1, 2, 3, 5, 7, 10, 20, 30]
EUR_NODES = [1, 2, 3, 5, 7, 10, 15, 20, 30]
TENORS = np.arange(1, 31)


def _fred(path: Path) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def bootstrap_par(par: np.ndarray) -> np.ndarray:
    """Annual-coupon par yields on tenors 1..30 (decimal, shape (..., 30)) -> discount factors."""
    par = np.atleast_2d(par)
    df = np.empty_like(par)
    annuity = np.zeros(par.shape[0])
    for i in range(par.shape[1]):
        df[:, i] = (1.0 - par[:, i] * annuity) / (1.0 + par[:, i])
        annuity += df[:, i]
    return df


def load_factors(data_dir: Path, start: str = "2004-09-06") -> pd.DataFrame:
    """Daily risk factors on common business days, forward-filled across holidays.

    Columns: usd_z1..usd_z30, eur_z1..eur_z30 (continuous zero rates, decimal), spread (decimal),
    spx (log level), eurusd (log level).
    """
    usd = pd.concat({n: _fred(data_dir / f"DGS{n}.csv") for n in USD_NODES}, axis=1) / 100.0
    eur = pd.concat({n: pd.read_csv(data_dir / f"ecb_SR_{n}Y.csv", index_col=0, parse_dates=True).iloc[:, 0]
                     for n in EUR_NODES}, axis=1) / 100.0
    spread = _fred(data_dir / "BAA10Y.csv").rename("spread") / 100.0
    yh = pd.read_csv(data_dir / "yahoo_spx_eurusd.csv", index_col=0, parse_dates=True)
    idx = eur.dropna().index
    idx = idx[idx >= start]
    usd, eur = usd.reindex(idx).ffill(), eur.reindex(idx).ffill()
    spread, yh = spread.reindex(idx).ffill(), yh.reindex(idx).ffill()
    # interpolate par (USD) and zero (EUR) nodes onto integer tenors 1..30
    usd_par = np.array([np.interp(TENORS, USD_NODES, r) for r in usd.to_numpy()])
    usd_z = -np.log(bootstrap_par(usd_par)) / TENORS
    eur_z = np.array([np.interp(TENORS, EUR_NODES, r) for r in eur.to_numpy()])
    out = pd.DataFrame(np.hstack([usd_z, eur_z]), index=idx,
                       columns=[f"usd_z{t}" for t in TENORS] + [f"eur_z{t}" for t in TENORS])
    out["spread"] = spread.to_numpy()
    out["spx"] = np.log(yh["^GSPC"].to_numpy())
    out["eurusd"] = np.log(yh["EURUSD=X"].to_numpy())
    return out.dropna()
