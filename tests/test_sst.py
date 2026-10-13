import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sst import (BalanceSheet, block_bootstrap, bootstrap_par, expected_shortfall, market_value_margin,  # noqa: E402
                 natcat_losses, natcat_mean, par_bond_ladder, payout)
from sst.balance_sheet import T  # noqa: E402

Z = 0.03 + 0.01 * np.log1p(T) / np.log(31)          # smooth upward curve


def test_bootstrap_reprices_par_bonds():
    par = 0.02 + 0.02 * T / 30
    df = bootstrap_par(par[None, :])[0]
    for m in (1, 5, 17, 30):
        price = par[m - 1] * df[:m].sum() + df[m - 1]
        assert price == pytest.approx(1.0, abs=1e-13)


def test_ladder_prices_to_market_value_with_and_without_spread():
    for s in (0.0, 0.015):
        cf = par_bond_ladder(Z, range(1, 16), 45.0, s)
        assert np.exp(-(Z + s) * T) @ cf == pytest.approx(45.0, abs=1e-12)


def test_payout_pattern_sums_and_mean_term():
    cf = payout(42.0, 3.5)
    assert cf.sum() == pytest.approx(42.0)
    assert (T * cf).sum() / cf.sum() == pytest.approx(3.5, rel=1e-3)


def test_balance_sheet_identity_and_dv01():
    bs = BalanceSheet().build(Z, Z, 0.015)
    v0 = bs.values(Z, Z, 0.015)
    assert v0["rbc"] == pytest.approx(v0["assets"] - v0["bel"])
    h = 1e-6
    up = bs.values(Z + h, Z, 0.015)["rbc"]
    dn = bs.values(Z - h, Z, 0.015)["rbc"]
    du, dc = np.exp(-Z * T), np.exp(-(Z + 0.015) * T)
    analytic = -(T * du) @ (bs.cf["gov_usd"] - bs.cf["pc_usd"] - bs.cf["lh_usd"]) - (T * dc) @ bs.cf["corp_usd"]
    assert (up - dn) / (2 * h) == pytest.approx(analytic, rel=1e-7)


@pytest.mark.parametrize("dist,exact", [
    (lambda r: r.standard_normal(1_000_000), stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01),
    (lambda r: r.exponential(1.0, 1_000_000), 1.0 + np.log(100)),
])
def test_expected_shortfall_matches_closed_form(dist, exact):
    x = dist(np.random.default_rng(3))
    assert expected_shortfall(x, 0.99) == pytest.approx(exact, rel=0.01)


def test_expected_shortfall_fractional_tail_is_exact():
    x = np.arange(1, 1001, dtype=float)            # worst 1 % = 10 values 991..1000
    assert expected_shortfall(x, 0.99) == pytest.approx(995.5)
    assert expected_shortfall(x, 0.9985) == pytest.approx((1000 + 0.5 * 999) / 1.5)


def test_block_bootstrap_sums_250_days_and_scales_variance():
    rng = np.random.default_rng(0)
    dx = rng.standard_normal((400_000, 2)) * [1.0, 2.0]
    y = block_bootstrap(dx, 40_000, block=25, seed=1)
    assert y.std(axis=0) == pytest.approx(np.sqrt(250) * np.array([1.0, 2.0]), rel=0.03)


def test_natcat_simulation_mean_matches_exact():
    cat = sum(natcat_losses(400_000, seed=5).values())
    assert cat.mean() == pytest.approx(natcat_mean(), rel=0.02)


def test_mvm_equals_coc_times_capital_for_one_year_liability():
    bs = BalanceSheet()
    bs.cf = {"pc_usd": np.r_[10.0, np.zeros(29)], "pc_eur": np.zeros(30), "lh_usd": np.zeros(30)}
    z = np.full(30, 0.02)
    assert market_value_margin(bs, z, sc0=5.0, coc=0.06) == pytest.approx(0.06 * 5.0 * np.exp(-0.02))
