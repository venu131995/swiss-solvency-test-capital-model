"""Put the pieces together: market and insurance one-year results, target capital, MVM, SST ratio, bridge."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .balance_sheet import BalanceSheet, T
from .risk import (block_bootstrap, daily_changes, expected_shortfall, insurance_result, market_value_margin,
                   overlapping_annual, sst_ratio, value_at_risk)

GROUPS = {
    "USD rates": lambda c: c.startswith("usd_z"),
    "EUR rates": lambda c: c.startswith("eur_z"),
    "credit spread": lambda c: c == "spread",
    "equity": lambda c: c == "spx",
    "EUR/USD": lambda c: c == "eurusd",
}


def state(f: pd.DataFrame, date) -> dict:
    row = f.loc[date]
    return {"usd_z": row[[f"usd_z{t}" for t in range(1, 31)]].to_numpy(),
            "eur_z": row[[f"eur_z{t}" for t in range(1, 31)]].to_numpy(),
            "spread": float(row["spread"])}


def revalue(bs: BalanceSheet, s0: dict, dX: np.ndarray, cols: list[str], mask=None) -> dict:
    """Instantaneous one-year shocks dX (n, n_factors) applied to the state s0; optional factor mask."""
    if mask is not None:
        dX = dX * np.array([mask(c) for c in cols], dtype=float)
    iu = [cols.index(f"usd_z{t}") for t in range(1, 31)]
    ie = [cols.index(f"eur_z{t}") for t in range(1, 31)]
    return bs.values(s0["usd_z"] + dX[:, iu], s0["eur_z"] + dX[:, ie], s0["spread"] + dX[:, cols.index("spread")],
                     dX[:, cols.index("spx")], dX[:, cols.index("eurusd")])


def run_sst(f: pd.DataFrame, date, bs: BalanceSheet | None = None, n: int = 200_000, method: str = "bootstrap",
            alpha: float = 0.99, coc: float = 0.06, seed: int = 0, hist_start=None) -> dict:
    """Full SST calculation at `date`, calibrating market risk on history up to `date`."""
    bs = bs or BalanceSheet()
    s0 = state(f, date)
    bs.build(s0["usd_z"], s0["eur_z"], s0["spread"])
    base = bs.values(s0["usd_z"], s0["eur_z"], s0["spread"])
    hist = f.loc[hist_start:date] if hist_start else f.loc[:date]
    cols = list(f.columns)
    dX = block_bootstrap(daily_changes(hist), n, seed=seed) if method == "bootstrap" else overlapping_annual(hist)
    m = n if method == "bootstrap" else dX.shape[0]
    mkt = revalue(bs, s0, dX, cols)["rbc"] - base["rbc"]
    stand_alone = {g: expected_shortfall(-(revalue(bs, s0, dX, cols, fn)["rbc"] - base["rbc"]), alpha)
                   for g, fn in GROUPS.items()}
    ins = insurance_result(bs, m, seed=seed + 10)
    ins_total = sum(v for k, v in ins.items() if not k.startswith("_"))
    total = mkt + ins_total
    es_total = expected_shortfall(-total, alpha)
    es_mkt = expected_shortfall(-mkt, alpha)
    es_ins = expected_shortfall(-ins_total, alpha)
    # MVM capital: one-year insurance run-off risk of the existing liabilities (reserves + L&H)
    runoff = ins["reserve development"] + ins["L&H pandemic"] + ins["L&H trend"]
    sc0 = expected_shortfall(-runoff, alpha)
    mvm = market_value_margin(bs, s0["usd_z"], sc0, coc)
    rbc = float(base["rbc"])
    return {"date": pd.Timestamp(date), "bs": bs, "base": base, "rbc": rbc, "mvm": mvm, "es": es_total,
            "tc": es_total + mvm, "ratio": sst_ratio(rbc, mvm, es_total), "es_market": es_mkt,
            "es_insurance": es_ins, "var_market": value_at_risk(-mkt, alpha), "stand_alone": stand_alone,
            "market": mkt, "insurance": ins, "total": total, "dX": dX, "cols": cols, "sc0": sc0}


def book_bridge(r: dict, locked_usd: np.ndarray, locked_eur: np.ndarray) -> list[tuple[str, float]]:
    """Bridge from a book-value view (liabilities discounted at locked-in rates, intangibles counted) to the
    SST numerator RBC - MVM. Returns ordered (label, amount) steps whose running sum is exact."""
    bs, base = r["bs"], r["base"]
    locked_bel = (np.exp(-locked_usd * T) @ (bs.cf["pc_usd"] + bs.cf["lh_usd"])
                  + np.exp(-locked_eur * T) @ bs.cf["pc_eur"])
    book_equity = float(base["assets"]) + bs.intangibles - float(locked_bel)
    return [("Book equity (locked-in discount rates)", book_equity),
            ("Remove intangibles", -bs.intangibles),
            ("Liabilities at current market rates", float(locked_bel - base["bel"])),
            ("Market value margin", -r["mvm"])]
