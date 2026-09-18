"""Frozen configuration objects and the master seed policy.

Every stochastic component draws from its own stream spawned off a single
SeedSequence, so that changing the number of traders cannot silently change the
price path, and drawing delta cannot shift the draw of kappa.  The independence
claim in the report rests on this separation.
"""

from dataclasses import dataclass, field, replace

import numpy as np

MASTER_SEED = 20260913

# Named positions of the child streams spawned from the master seed.
STREAM_PRICES = 0
STREAM_DELTA = 1
STREAM_KAPPA = 2
STREAM_WEALTH = 3
STREAM_NPOS = 4
STREAM_HAZARD = 5
STREAM_SELECTION = 6
STREAM_BOOTSTRAP = 7
N_STREAMS = 8


def stream_rngs(scenario_index: int, master_seed: int = MASTER_SEED):
    """Return the list of independent generators used by one scenario run."""
    root = np.random.SeedSequence([master_seed, scenario_index])
    return [np.random.default_rng(s) for s in root.spawn(N_STREAMS)]


@dataclass(frozen=True)
class PriceConfig:
    """Factor structure for the security universe."""

    n_securities: int = 60
    n_days: int = 750
    trading_days: int = 252
    risk_free_annual: float = 0.02
    market_premium_annual: float = 0.06
    market_vol_annual: float = 0.16
    style_vol_annual: float = 0.10
    n_style_factors: int = 2
    beta_market_mean: float = 1.00
    beta_market_sd: float = 0.35
    beta_market_clip: tuple = (0.20, 2.00)
    beta_style_sd: float = 0.50
    idio_vol_low: float = 0.20
    idio_vol_high: float = 0.45
    # Expected LOG return, imposed identically on every security.  See
    # prices.simulate_universe for why this is the invariance that matters.
    homogeneous_drift: bool = True
    common_log_drift_annual: float = 0.06
    price_low: float = 20.0
    price_high: float = 200.0
    momentum_window: int = 20


@dataclass(frozen=True)
class PopulationConfig:
    """Cross-section of traders."""

    n_traders: int = 1000
    delta_mean: float = 0.0
    kappa_mean: float = 0.0
    beta_concentration: float = 6.0
    wealth_low: float = 10_000.0
    wealth_high: float = 500_000.0
    n_pos_low: int = 5
    n_pos_high: int = 30
    # Deliberate violation of the independence requirement, used only by the
    # counterfactual in the independence section.  Zero everywhere else.
    copula_rho: float = 0.0


@dataclass(frozen=True)
class CostConfig:
    """Explicit transaction costs.  Both legs are charged on every trade."""

    commission_bps: float = 10.0     # 0.10 % of notional, per side
    commission_min_usd: float = 1.00  # floor, in USD, per trade
    half_spread_bps: float = 15.0    # quoted spread of 30 bps around the mid


@dataclass(frozen=True)
class EngineConfig:
    """Behavioural mechanism and market-microstructure switches."""

    base_hazard: float = 0.004       # daily probability of closing a position
    kappa_multiplier: float = 4.0    # kappa = 1 quintuples the base hazard
    hazard_cap: float = 0.95
    reversal_strength: float = 0.60  # scenario 8: loading on standardised momentum
    rebalance_every: int = 21        # scenario 7: monthly
    rebalance_band: float = 0.25     # trim when weight exceeds 1.25x target
    settlement_lag: int = 0          # 0 = same-day redeployment, no cash drag
    min_deploy_usd: float = 100.0    # cash below this is not worth a ticket


@dataclass(frozen=True)
class ScenarioConfig:
    """One full population regenerated under a stated configuration."""

    index: int
    name: str
    delta_mean: float
    kappa_mean: float
    confound: str = "none"           # 'none' | 'rebalance' | 'reversal'
    engine: EngineConfig = field(default_factory=EngineConfig)
    prices: PriceConfig = field(default_factory=PriceConfig)
    n_traders: int = 1000
    copula_rho: float = 0.0

    @property
    def population(self) -> PopulationConfig:
        return PopulationConfig(
            n_traders=self.n_traders,
            delta_mean=self.delta_mean,
            kappa_mean=self.kappa_mean,
            copula_rho=self.copula_rho,
        )

    def with_overrides(self, **kwargs) -> "ScenarioConfig":
        return replace(self, **kwargs)
