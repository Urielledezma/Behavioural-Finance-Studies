"""P01: the simulator must recover what it is given.

These are the properties the report rests on, asserted qualitatively rather
than to five decimals, so that a change of numpy version on another machine
does not fail a test whose conclusion still holds.  A change that breaks one of
them has changed a finding, and the report has to be re-read, not the test.
"""

import numpy as np
import pytest

from bfsim.estimators import pgr_plr
from bfsim.scenarios import SCENARIOS, analyse, common_universe

N_BOOT = 20  # the bootstrap is not under test here, only the point estimates


@pytest.fixture(scope="module")
def runs():
    universe = common_universe()
    return {cfg.index: analyse(cfg, universe, n_boot=N_BOOT)
            for cfg in SCENARIOS if cfg.index in (1, 2, 3, 5, 6, 7)}


def test_pgr_plr_reproduces_odean_table_one():
    # Odean (1998), Table I, entire year: the published counts give 0.148 and 0.098.
    d = pgr_plr(13_883, 79_658, 11_930, 110_348)
    assert d["PGR"] == pytest.approx(0.148, abs=5e-4)
    assert d["PLR"] == pytest.approx(0.098, abs=5e-4)
    assert d["diff"] == pytest.approx(0.050, abs=1e-3)


def test_position_at_basis_enters_no_count():
    d = pgr_plr(0, 0, 0, 0)
    assert np.isnan(d["PGR"]) and np.isnan(d["PLR"])


def test_null_is_quiet(runs):
    assert abs(runs[1].disp["diff"]) < 0.005


def test_disposition_is_monotone_in_delta(runs):
    assert runs[2].disp["diff"] > 0.01
    assert runs[3].disp["diff"] > runs[2].disp["diff"]


def test_disposition_estimator_ignores_kappa(runs):
    assert abs(runs[5].disp["diff"]) < 0.005


def test_net_turnover_slope_is_a_cost(runs):
    assert runs[5].over["ret_net"]["beta"] < 0


def test_injected_parameters_are_independent(runs):
    n = len(runs[6].df)
    assert abs(runs[6].indep["corr_delta_kappa"]) < 3 / np.sqrt(n - 3)


def test_rebalancing_confound_fires_without_any_bias(runs):
    # The documented failure of the estimator, not a bug: delta is zero here.
    assert SCENARIOS[6].delta_mean == 0
    assert runs[7].disp["diff"] > 0.01
