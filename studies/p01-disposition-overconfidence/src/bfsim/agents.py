"""The trader cross-section.

delta and kappa are drawn from separate generators spawned off the master seed,
so the two are independent by construction rather than by inspection.  The
scenario's tabulated value is the *mean* of a Beta draw, not a constant: without
cross-sectional dispersion the independence diagnostics would have nothing to
correlate and the turnover regression would lose most of its identification.
"""

from dataclasses import dataclass

import numpy as np

from .config import (
    PopulationConfig,
    STREAM_DELTA,
    STREAM_KAPPA,
    STREAM_NPOS,
    STREAM_WEALTH,
)


@dataclass
class Population:
    delta: np.ndarray    # (N,) disposition strength in [0, 1]
    kappa: np.ndarray    # (N,) overprecision / churn intensity in [0, 1]
    wealth: np.ndarray   # (N,) starting capital in USD
    n_pos: np.ndarray    # (N,) number of positions held

    @property
    def n_traders(self) -> int:
        return self.delta.size


def _beta_draw(mean: float, concentration: float, size: int,
               rng: np.random.Generator) -> np.ndarray:
    """Beta variate with the requested mean, degenerate when the mean is 0 or 1."""
    if mean <= 0.0:
        return np.zeros(size)
    if mean >= 1.0:
        return np.ones(size)
    a = mean * concentration
    b = (1.0 - mean) * concentration
    return rng.beta(a, b, size)


def _coupled_draw(cfg, n, rng):
    """Gaussian copula linking delta and kappa at a stated rank correlation.

    Used only to show what the assignment's independence requirement is buying.
    """
    from scipy.stats import beta as beta_dist, norm
    rho = cfg.copula_rho
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    u = norm.cdf(z)
    def inv(mean, col):
        if mean <= 0.0:
            return np.zeros(n)
        a, b = mean * cfg.beta_concentration, (1.0 - mean) * cfg.beta_concentration
        return beta_dist.ppf(u[:, col], a, b)
    return inv(cfg.delta_mean, 0), inv(cfg.kappa_mean, 1)


def draw_population(cfg: PopulationConfig, rngs) -> Population:
    n = cfg.n_traders
    if cfg.copula_rho != 0.0:
        delta, kappa = _coupled_draw(cfg, n, rngs[STREAM_DELTA])
    else:
        delta = _beta_draw(cfg.delta_mean, cfg.beta_concentration, n, rngs[STREAM_DELTA])
        kappa = _beta_draw(cfg.kappa_mean, cfg.beta_concentration, n, rngs[STREAM_KAPPA])

    # Log-uniform capital: the wealth distribution is right-skewed in every
    # brokerage sample, and a uniform draw would put half the population above
    # 255k dollars.
    w = np.exp(rngs[STREAM_WEALTH].uniform(
        np.log(cfg.wealth_low), np.log(cfg.wealth_high), n))
    n_pos = rngs[STREAM_NPOS].integers(cfg.n_pos_low, cfg.n_pos_high + 1, n)

    return Population(delta=delta, kappa=kappa, wealth=w, n_pos=n_pos)
