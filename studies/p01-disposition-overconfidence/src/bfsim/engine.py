"""The daily trading engine.

Everything is vectorised over a (n_traders, n_slots) grid of positions.  A slot
is one lot: a purchase creates a lot and a full sale destroys it, so the default
cost basis is per-lot.  An average-cost basis is accumulated in parallel on the
same pass, which is what makes the cost-basis sensitivity table free.

The injected mechanism lives in `_hazard`.  Nothing in this file parametrises
PGR, PLR or turnover; all three are read back out of the simulated record.
"""

from dataclasses import dataclass, field

import numpy as np

from .agents import Population, draw_population
from .config import (
    CostConfig,
    STREAM_HAZARD,
    STREAM_PRICES,
    STREAM_SELECTION,
    ScenarioConfig,
)
from .prices import Universe, simulate_universe


@dataclass
class RunResult:
    """Everything the estimators need, at account level."""

    scenario: ScenarioConfig
    population: Population
    universe: Universe

    # Disposition counters, per account.  Per-lot basis, position-level counting.
    gains_realized: np.ndarray
    gains_paper: np.ndarray
    losses_realized: np.ndarray
    losses_paper: np.ndarray

    # The same four counters under the two alternative conventions.
    frac_counts: dict = field(default_factory=dict)     # share-weighted realisation
    avgcost_counts: dict = field(default_factory=dict)  # average-cost basis
    allday_counts: dict = field(default_factory=dict)   # silent days included too

    at_basis: int = 0          # positions sitting exactly at their purchase price
    sale_days: np.ndarray = None
    active_days: np.ndarray = None

    # Performance and turnover, per account.
    turnover: np.ndarray = None
    ret_net: np.ndarray = None
    ret_gross: np.ndarray = None
    vol_net: np.ndarray = None
    beta_exposure: np.ndarray = None
    pv_net: np.ndarray = None
    pv_gross: np.ndarray = None
    costs_paid: np.ndarray = None

    # For the leakage placebo.
    buy_days: np.ndarray = None
    buy_secs: np.ndarray = None

    # Discriminating statistic: trailing-momentum z of what was sold versus what
    # was held on the same days.  A reference-point rule does not tilt it; a
    # belief in reversal does.
    sold_z: tuple = (0.0, 0.0)
    held_z: tuple = (0.0, 0.0)
    sold_frac: tuple = (0.0, 0.0)


def _commission(notional, cost):
    return np.maximum(cost.commission_min_usd, notional * cost.commission_bps * 1e-4)


def _hazard(cfg, kappa_level, delta, gain, loss, universe, t, idx):
    """Daily probability of closing a position.

    kappa sets the level, delta tilts it by domain, and the reversal confound
    multiplies it by a function of the security's trailing return.  PGR appears
    nowhere in this expression; it is a consequence of it.
    """
    eng = cfg.engine
    domain = np.where(gain, 1.0 + delta, np.where(loss, 1.0 - delta, 1.0))
    h = kappa_level * domain
    if cfg.confound == "reversal":
        z = universe.momentum_z[t][idx]
        h = h * np.maximum(1.0 + eng.reversal_strength * z, 0.05)
    return np.clip(h, 0.0, eng.hazard_cap)


def run_scenario(cfg, rngs, universe=None, cost=None):
    cost = cost or CostConfig()
    eng = cfg.engine
    universe = universe if universe is not None else simulate_universe(
        cfg.prices, rngs[STREAM_PRICES])

    pop = draw_population(cfg.population, rngs)
    N = pop.n_traders
    J = universe.n_securities
    T = universe.n_days
    S = int(pop.n_pos.max())

    prices = universe.prices
    hs = cost.half_spread_bps * 1e-4
    rng_h = rngs[STREAM_HAZARD]
    rng_s = rngs[STREAM_SELECTION]

    usable = np.arange(S)[None, :] < pop.n_pos[:, None]
    row = np.arange(N)[:, None]

    state = {
        "sec": np.full((N, S), -1, dtype=np.int64),
        "sh": np.zeros((N, S)),
        "shg": np.zeros((N, S)),
        "bas": np.zeros((N, S)),
        "act": np.zeros((N, S), dtype=bool),
        "cash": pop.wealth.copy(),
        "cashg": pop.wealth.copy(),
    }
    # Ring buffer of sale proceeds awaiting settlement, one column per day of
    # lag, so settlement_lag = 3 really means three idle days rather than one.
    lag = max(int(eng.settlement_lag), 0)
    pending = np.zeros((max(lag, 1), N))
    pendingg = np.zeros((max(lag, 1), N))

    Gr, Gp, Lr, Lp = (np.zeros(N) for _ in range(4))
    fGr, fGp, fLr, fLp = (np.zeros(N) for _ in range(4))
    aGr, aGp, aLr, aLp = (np.zeros(N) for _ in range(4))
    uGr, uGp, uLr, uLp = (np.zeros(N) for _ in range(4))
    at_basis = 0
    sale_days = np.zeros(N)
    active_days = np.zeros(N)

    traded_mid = np.zeros(N)
    costs_paid = np.zeros(N)
    zs_sold = zn_sold = zs_held = zn_held = 0.0
    frac_sum = frac_n = 0.0
    beta_sum = np.zeros(N)
    pv_net = np.zeros((T, N))
    pv_gross = np.zeros((T, N))
    buy_days, buy_secs = [], []

    delta = pop.delta[:, None]
    kappa_level = eng.base_hazard * (1.0 + eng.kappa_multiplier * pop.kappa[:, None])
    target_w = (1.0 / pop.n_pos)[:, None]

    def execute_buys(alloc_w, day):
        """Deploy alloc_w (a fraction of current cash per slot) into named slots."""
        take = alloc_w > 0
        if not take.any():
            return
        sec, sh, shg, bas, act = (state["sec"], state["sh"], state["shg"],
                                  state["bas"], state["act"])
        px_mid = prices[day][sec.clip(0)]
        ask = px_mid * (1.0 + hs)
        alloc = alloc_w * state["cash"][:, None]
        allocg = alloc_w * state["cashg"][:, None]
        comm = np.where(take, np.minimum(_commission(alloc, cost), alloc), 0.0)
        notional = np.where(take, alloc - comm, 0.0)
        add_sh = np.where(take, notional / ask, 0.0)
        add_shg = np.where(take, allocg / px_mid, 0.0)

        new_sh = sh + add_sh
        # Topping up an existing lot blends the reference point; a fresh slot
        # takes the fill price it actually paid.
        state["bas"] = np.where(
            take & act,
            (sh * bas + add_sh * ask) / np.maximum(new_sh, 1e-12),
            np.where(take, ask, bas))
        state["sh"] = new_sh
        state["shg"] = shg + add_shg
        state["act"] = act | take
        state["cash"] = state["cash"] - alloc.sum(axis=1)
        state["cashg"] = state["cashg"] - allocg.sum(axis=1)
        costs_paid[:] += comm.sum(axis=1) + (add_sh * px_mid * hs).sum(axis=1)
        traded_mid[:] += (add_sh * px_mid).sum(axis=1)
        r, c = np.nonzero(take)
        buy_days.append(np.full(r.size, day))
        buy_secs.append(sec[r, c])

    # ---- day 0: initial deployment -----------------------------------------
    state["sec"] = np.where(usable, rng_s.integers(0, J, size=(N, S)), -1)
    execute_buys(np.where(usable, 1.0 / pop.n_pos[:, None], 0.0), 0)
    idx0 = state["sec"].clip(0)
    pv_net[0] = state["cash"] + (state["sh"] * prices[0][idx0] * state["act"]).sum(axis=1)
    pv_gross[0] = state["cashg"] + (state["shg"] * prices[0][idx0] * state["act"]).sum(axis=1)
    # Forming the initial portfolio is an artefact of the simulation, not
    # behaviour, so it is charged to net returns but excluded from turnover.
    traded_at_entry = traded_mid.copy()

    # ---- main loop ----------------------------------------------------------
    for t in range(1, T):
        if lag > 0:
            slot = (t - 1) % lag
            state["cash"] += pending[slot]
            state["cashg"] += pendingg[slot]
            pending[slot] = 0.0
            pendingg[slot] = 0.0

        sec, sh, shg, bas, act = (state["sec"], state["sh"], state["shg"],
                                  state["bas"], state["act"])
        idx = sec.clip(0)
        px = prices[t][idx]
        val = sh * px * act
        pv = state["cash"] + val.sum(axis=1)
        beta_sum += (val * universe.beta_market[idx]).sum(axis=1) / np.maximum(pv, 1e-9)

        # --- classification against the purchase price -----------------------
        gain = act & (px > bas)
        loss = act & (px < bas)
        at_basis += int((act & (px == bas)).sum())

        flat = (row * J + idx).ravel()
        tot_sh = np.bincount(flat, weights=(sh * act).ravel(),
                             minlength=N * J).reshape(N, J)
        tot_cost = np.bincount(flat, weights=(sh * bas * act).ravel(),
                               minlength=N * J).reshape(N, J)
        avg_bas = np.take_along_axis(
            np.where(tot_sh > 1e-12, tot_cost / np.maximum(tot_sh, 1e-12), 0.0),
            idx, axis=1)
        gain_a = act & (px > avg_bas)
        loss_a = act & (px < avg_bas)

        # --- the injected mechanism ------------------------------------------
        h = _hazard(cfg, kappa_level, delta, gain, loss, universe, t, idx)
        sell_full = act & (rng_h.random((N, S)) < h)

        # --- confound: mechanical rebalancing --------------------------------
        trim_frac = np.zeros((N, S))
        rebal_day = (cfg.confound == "rebalance") and (t % eng.rebalance_every == 0)
        if rebal_day:
            w = val / np.maximum(pv, 1e-9)[:, None]
            over = act & ~sell_full & (w > target_w * (1.0 + eng.rebalance_band))
            trim_frac = np.where(over, 1.0 - target_w / np.maximum(w, 1e-12), 0.0)

        frac = np.where(sell_full, 1.0, trim_frac)
        realized = frac > 0
        sold_today = realized.any(axis=1)
        sale_days += sold_today
        active_days += act.any(axis=1)

        # --- accumulate the four counters, only on days with a sale ----------
        m = sold_today[:, None]
        Gr += (gain & realized & m).sum(axis=1)
        Gp += (gain & ~realized & m).sum(axis=1)
        Lr += (loss & realized & m).sum(axis=1)
        Lp += (loss & ~realized & m).sum(axis=1)
        aGr += (gain_a & realized & m).sum(axis=1)
        aGp += (gain_a & ~realized & m).sum(axis=1)
        aLr += (loss_a & realized & m).sum(axis=1)
        aLp += (loss_a & ~realized & m).sum(axis=1)
        fGr += np.where(gain & m, frac, 0.0).sum(axis=1)
        fGp += np.where(gain & m, 1.0 - frac, 0.0).sum(axis=1)
        fLr += np.where(loss & m, frac, 0.0).sum(axis=1)
        fLp += np.where(loss & m, 1.0 - frac, 0.0).sum(axis=1)
        # Same four counts with the sale-day filter switched off, so the report
        # can price what that convention costs instead of only declaring it.
        uGr += (gain & realized).sum(axis=1)
        uGp += (gain & ~realized).sum(axis=1)
        uLr += (loss & realized).sum(axis=1)
        uLp += (loss & ~realized).sum(axis=1)

        zt = universe.momentum_z[t][idx]
        zs_sold += float(zt[realized].sum()); zn_sold += float(realized.sum())
        held = act & ~realized
        zs_held += float(zt[held].sum()); zn_held += float(held.sum())
        frac_sum += float(frac[realized].sum()); frac_n += float(realized.sum())

        # --- execute the sales ------------------------------------------------
        if realized.any():
            sh_sold = sh * frac
            shg_sold = shg * frac
            notional = sh_sold * px
            bid_proceeds = notional * (1.0 - hs)
            comm = np.where(realized,
                            np.minimum(_commission(notional, cost), bid_proceeds), 0.0)
            got = np.where(realized, bid_proceeds - comm, 0.0).sum(axis=1)
            gotg = (shg_sold * px).sum(axis=1)
            if lag > 0:
                slot = (t - 1) % lag
                pending[slot] += got
                pendingg[slot] += gotg
            else:
                state["cash"] += got
                state["cashg"] += gotg
            costs_paid += comm.sum(axis=1) + (notional * hs).sum(axis=1)
            traded_mid += notional.sum(axis=1)

            act = act & ~sell_full
            state["act"] = act
            state["sh"] = np.where(act, sh - sh_sold, 0.0)
            state["shg"] = np.where(act, shg - shg_sold, 0.0)
            state["bas"] = np.where(act, bas, 0.0)
            state["sec"] = np.where(act, sec, -1)

        # --- redeploy ---------------------------------------------------------
        if rebal_day:
            sec, sh, act = state["sec"], state["sh"], state["act"]
            val_now = sh * prices[t][sec.clip(0)] * act
            need = np.maximum(0.0, target_w * np.maximum(pv, 1e-9)[:, None] - val_now) * act
            tot_need = need.sum(axis=1)
            cash_now = state["cash"]
            spend = np.minimum(tot_need, np.maximum(cash_now - eng.min_deploy_usd, 0.0))
            alloc_w = np.where(
                tot_need[:, None] > 0,
                need / np.maximum(tot_need, 1e-9)[:, None]
                * (spend / np.maximum(cash_now, 1e-9))[:, None], 0.0)
            execute_buys(np.where(act, alloc_w, 0.0), t)

        empty = usable & ~state["act"]
        n_empty = empty.sum(axis=1)
        can_buy = (n_empty > 0) & (state["cash"] > eng.min_deploy_usd * np.maximum(n_empty, 1))
        if can_buy.any():
            pick = rng_s.integers(0, J, size=(N, S))
            fill = empty & can_buy[:, None]
            state["sec"] = np.where(fill, pick, state["sec"])
            execute_buys(np.where(fill, 1.0 / np.maximum(n_empty, 1)[:, None], 0.0), t)

        idx = state["sec"].clip(0)
        pv_net[t] = state["cash"] + (state["sh"] * prices[t][idx] * state["act"]).sum(axis=1)
        pv_gross[t] = state["cashg"] + (state["shg"] * prices[t][idx] * state["act"]).sum(axis=1)

    years = (T - 1) / cfg.prices.trading_days
    ret_net = (pv_net[-1] / pv_net[0]) ** (1.0 / years) - 1.0
    ret_gross = (pv_gross[-1] / pv_gross[0]) ** (1.0 / years) - 1.0
    daily = np.diff(np.log(np.maximum(pv_net, 1e-9)), axis=0)
    vol_net = daily.std(axis=0) * np.sqrt(cfg.prices.trading_days)
    turnover = ((traded_mid - traded_at_entry) / 2.0) / pv_net.mean(axis=0) / years

    return RunResult(
        scenario=cfg, population=pop, universe=universe,
        gains_realized=Gr, gains_paper=Gp, losses_realized=Lr, losses_paper=Lp,
        frac_counts=dict(Gr=fGr, Gp=fGp, Lr=fLr, Lp=fLp),
        avgcost_counts=dict(Gr=aGr, Gp=aGp, Lr=aLr, Lp=aLp),
        allday_counts=dict(Gr=uGr, Gp=uGp, Lr=uLr, Lp=uLp),
        at_basis=at_basis, sale_days=sale_days, active_days=active_days,
        turnover=turnover, ret_net=ret_net, ret_gross=ret_gross,
        vol_net=vol_net, beta_exposure=beta_sum / (T - 1),
        pv_net=pv_net, pv_gross=pv_gross, costs_paid=costs_paid,
        buy_days=np.concatenate(buy_days) if buy_days else np.array([], dtype=int),
        buy_secs=np.concatenate(buy_secs) if buy_secs else np.array([], dtype=int),
        sold_z=(zs_sold / max(zn_sold, 1.0), zn_sold),
        held_z=(zs_held / max(zn_held, 1.0), zn_held),
        sold_frac=(frac_sum / max(frac_n, 1.0), frac_n),
    )
