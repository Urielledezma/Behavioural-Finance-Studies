"""The security universe.

The whole price panel is drawn before a single agent exists and is never
re-drawn, re-ordered or conditioned on agent state.  That is how the exercise's
big constraint is satisfied: there is no code path by which a trading decision
can reach the return that follows it.  `forward_return_placebo` in this module
tests the claim empirically rather than trusting the argument.
"""

from dataclasses import dataclass

import numpy as np

from .config import PriceConfig


@dataclass
class Universe:
    prices: np.ndarray        # (n_days, n_securities) mid prices
    log_returns: np.ndarray   # (n_days, n_securities), first row is zero
    beta_market: np.ndarray   # (n_securities,)
    beta_style: np.ndarray    # (n_securities, n_style_factors)
    idio_vol: np.ndarray      # (n_securities,) annualised
    factors: np.ndarray       # (n_days, 1 + n_style_factors) daily shocks
    momentum_z: np.ndarray    # (n_days, n_securities) cross-sectional z-score

    @property
    def n_days(self) -> int:
        return self.prices.shape[0]

    @property
    def n_securities(self) -> int:
        return self.prices.shape[1]


def simulate_universe(cfg: PriceConfig, rng: np.random.Generator) -> Universe:
    """Draw a correlated panel of daily prices from a three-factor structure.

    Security j has log return

        r_jt = a_j + b_j^M f^M_t + b_j^V f^V_t + b_j^S f^S_t + e_jt,

    with one market factor carrying the drift and two orthogonal zero-mean style
    factors.  The intercept a_j is set so that the *arithmetic* expected annual
    return obeys a CAPM-like relation, rf + b_j^M * ERP, which keeps the drift
    tied to the risk the security actually carries instead of being a free knob.
    """
    T, J, K = cfg.n_days, cfg.n_securities, cfg.n_style_factors
    dt = 1.0 / cfg.trading_days

    beta_m = np.clip(
        rng.normal(cfg.beta_market_mean, cfg.beta_market_sd, J),
        *cfg.beta_market_clip,
    )
    beta_s = rng.normal(0.0, cfg.beta_style_sd, size=(J, K))
    idio = rng.uniform(cfg.idio_vol_low, cfg.idio_vol_high, J)

    # Daily factor shocks, independent across factors and across time.
    f_m = rng.normal(0.0, cfg.market_vol_annual * np.sqrt(dt), size=(T, 1))
    f_s = rng.normal(0.0, cfg.style_vol_annual * np.sqrt(dt), size=(T, K))
    factors = np.hstack([f_m, f_s])

    eps = rng.normal(0.0, 1.0, size=(T, J)) * (idio * np.sqrt(dt))

    total_var = (
        (beta_m * cfg.market_vol_annual) ** 2
        + (beta_s ** 2).sum(axis=1) * cfg.style_vol_annual ** 2
        + idio ** 2
    )
    if cfg.homogeneous_drift:
        # Every security carries the same expected log return.  Under a CAPM-like
        # drift, expected log return is mu_j - 0.5 * var_j, which falls sharply
        # with idiosyncratic volatility; any rule that selects on past returns
        # then picks up a drift differential it never had any information about.
        # That is indistinguishable from leakage in the gross-return test, so we
        # remove it at the source.  Expected *arithmetic* returns still differ by
        # half the variance, a Jensen artefact that cannot make a portfolio
        # compound faster.
        alpha = np.full(J, cfg.common_log_drift_annual * dt)
    else:
        mu_arith = cfg.risk_free_annual + beta_m * cfg.market_premium_annual
        alpha = (mu_arith - 0.5 * total_var) * dt

    log_r = alpha + f_m @ beta_m[None, :] + f_s @ beta_s.T + eps
    log_r[0, :] = 0.0

    p0 = rng.uniform(cfg.price_low, cfg.price_high, J)
    prices = p0 * np.exp(np.cumsum(log_r, axis=0))

    momentum_z = _momentum_zscore(prices, cfg.momentum_window)

    return Universe(
        prices=prices,
        log_returns=log_r,
        beta_market=beta_m,
        beta_style=beta_s,
        idio_vol=idio,
        factors=factors,
        momentum_z=momentum_z,
    )


def _momentum_zscore(prices: np.ndarray, window: int) -> np.ndarray:
    """Cross-sectionally standardised trailing return, used by scenario 8.

    This is a property of the price panel alone.  Agents read it; nothing they do
    writes back to it.
    """
    T = prices.shape[0]
    trail = np.zeros_like(prices)
    trail[window:] = prices[window:] / prices[:-window] - 1.0
    mu = trail.mean(axis=1, keepdims=True)
    sd = trail.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (trail - mu) / sd


def forward_return_placebo(
    universe: Universe, buy_days: np.ndarray, buy_secs: np.ndarray, horizon: int = 20
) -> dict:
    """Compare the forward return of what agents bought against the universe.

    If predictive information had leaked into the trade decision, purchased names
    would systematically out- or underperform the average security over the
    following `horizon` days.

    Two things have to be right for the test to mean anything.  A thousand
    accounts buying the same security on the same day is one purchase decision
    observed a thousand times, so the (day, security) pairs are de-duplicated
    first; leaving them in produced a t-statistic near -8 on a panel that is
    demonstrably free of reversal.  And purchases made on the same day share a
    market shock, so the statistic is averaged within a day and the standard
    error is taken across days.
    """
    T = universe.n_days
    keep = buy_days + horizon < T
    d, s = buy_days[keep], buy_secs[keep]
    if d.size == 0:
        return {"n_trades": 0, "n_pairs": 0, "n_days": 0, "diff": np.nan,
                "t_stat": np.nan, "diff_log": np.nan, "t_log": np.nan}

    pairs = np.unique(np.stack([d, s], axis=1), axis=0)
    d, s = pairs[:, 0], pairs[:, 1]

    logp = np.log(universe.prices)
    simple = universe.prices[horizon:] / universe.prices[:-horizon] - 1.0
    loggr = logp[horizon:] - logp[:-horizon]

    def _clustered(panel):
        excess = panel[d, s] - panel[d].mean(axis=1)
        days = np.unique(d)
        per_day = np.array([excess[d == u].mean() for u in days])
        se = per_day.std(ddof=1) / np.sqrt(per_day.size)
        return float(excess.mean()), float(per_day.mean() / se) if se > 0 else np.nan

    diff, t = _clustered(simple)
    diff_log, t_log = _clustered(loggr)
    return {
        "n_trades": int(keep.sum()),
        "n_pairs": int(pairs.shape[0]),
        "n_days": int(np.unique(d).size),
        "diff": diff, "t_stat": t,
        "diff_log": diff_log, "t_log": t_log,
    }
