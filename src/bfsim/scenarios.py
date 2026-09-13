"""The eight scenarios and the runner that turns them into tables.

All eight share one price panel.  Regenerating prices per scenario would mix a
behavioural difference with a different realised market, and the comparisons the
assignment asks for -- null recovery, monotonicity in delta, monotonicity in
kappa -- are precisely comparisons *across* scenarios.  The price path is
therefore a fixed common factor, and `replicate_seeds` checks separately that
the conclusions survive redrawing it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    MASTER_SEED,
    PriceConfig,
    STREAM_BOOTSTRAP,
    STREAM_PRICES,
    ScenarioConfig,
    stream_rngs,
)
from .engine import run_scenario
from .estimators import (
    account_frame,
    bootstrap_accounts,
    bootstrap_transactions,
    independence_block,
    overconfidence_block,
    pgr_plr,
)
from .prices import forward_return_placebo, simulate_universe

SCENARIOS = [
    ScenarioConfig(1, "Null", 0.0, 0.0, "none"),
    ScenarioConfig(2, "Disposition only, low", 0.3, 0.0, "none"),
    ScenarioConfig(3, "Disposition only, high", 0.8, 0.0, "none"),
    ScenarioConfig(4, "Turnover only, low", 0.0, 0.3, "none"),
    ScenarioConfig(5, "Turnover only, high", 0.0, 0.8, "none"),
    ScenarioConfig(6, "Both active", 0.8, 0.8, "none"),
    ScenarioConfig(7, "Rebalancing confound", 0.0, 0.0, "rebalance"),
    ScenarioConfig(8, "Mean-reversion confound", 0.0, 0.0, "reversal"),
]


def common_universe(price_cfg: PriceConfig = None, seed: int = MASTER_SEED):
    """One price panel, drawn before any agent exists, shared by every scenario."""
    price_cfg = price_cfg or PriceConfig()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 999]))
    return simulate_universe(price_cfg, rng)


@dataclass
class ScenarioOutput:
    cfg: ScenarioConfig
    run: object
    df: pd.DataFrame
    disp: dict
    boot: dict
    boot_naive: dict
    over: dict
    indep: dict
    placebo: dict
    disp_frac: dict
    disp_avgcost: dict


def analyse(cfg, universe, n_boot=1000, master_seed=MASTER_SEED) -> ScenarioOutput:
    rngs = stream_rngs(cfg.index, master_seed)
    run = run_scenario(cfg, rngs, universe=universe)
    df = account_frame(run)
    boot_rng = rngs[STREAM_BOOTSTRAP]

    disp = pgr_plr(run.gains_realized, run.gains_paper,
                   run.losses_realized, run.losses_paper)
    boot = bootstrap_accounts(run.gains_realized, run.gains_paper,
                              run.losses_realized, run.losses_paper,
                              n_boot=n_boot, rng=boot_rng)
    boot_naive = bootstrap_transactions(run.gains_realized, run.gains_paper,
                                        run.losses_realized, run.losses_paper,
                                        n_boot=n_boot, rng=boot_rng)
    fc, ac = run.frac_counts, run.avgcost_counts
    return ScenarioOutput(
        cfg=cfg, run=run, df=df, disp=disp, boot=boot, boot_naive=boot_naive,
        over=overconfidence_block(df),
        indep=independence_block(df),
        placebo=forward_return_placebo(universe, run.buy_days, run.buy_secs),
        disp_frac=pgr_plr(fc["Gr"], fc["Gp"], fc["Lr"], fc["Lp"]),
        disp_avgcost=pgr_plr(ac["Gr"], ac["Gp"], ac["Lr"], ac["Lp"]),
    )


def run_all(universe=None, n_boot=1000, scenarios=None, verbose=True):
    universe = universe if universe is not None else common_universe()
    scenarios = scenarios or SCENARIOS
    out = {}
    for cfg in scenarios:
        out[cfg.index] = analyse(cfg, universe, n_boot=n_boot)
        if verbose:
            o = out[cfg.index]
            print(f"  [{cfg.index}] {cfg.name:<26s} "
                  f"PGR-PLR={o.disp['diff']:+.4f} ({o.boot['se_diff']:.4f})  "
                  f"beta_net={o.over['ret_net']['beta']:+.5f}")
    return out


def results_table(outputs) -> pd.DataFrame:
    rows = []
    for k in sorted(outputs):
        o = outputs[k]
        d, b, ov = o.disp, o.boot, o.over
        rows.append({
            "#": k,
            "Scenario": o.cfg.name,
            "delta_inj": o.cfg.delta_mean,
            "kappa_inj": o.cfg.kappa_mean,
            "Confound": o.cfg.confound,
            "PGR": d["PGR"], "se_PGR": b["se_PGR"],
            "PLR": d["PLR"], "se_PLR": b["se_PLR"],
            "PGR-PLR": d["diff"], "se_diff": b["se_diff"],
            "t_diff": d["diff"] / b["se_diff"] if b["se_diff"] > 0 else np.nan,
            "PGR/PLR": d["ratio"], "se_ratio": b["se_ratio"],
            "Turnover": o.df["turnover"].mean(),
            "beta_gross": ov["ret_gross"]["beta"], "se_gross": ov["ret_gross"]["se"],
            "beta_net": ov["ret_net"]["beta"], "se_net": ov["ret_net"]["se"],
            "t_gross": ov["ret_gross"]["t"], "t_net": ov["ret_net"]["t"],
        })
    return pd.DataFrame(rows).set_index("#")


def independence_table(outputs) -> pd.DataFrame:
    rows = []
    for k in sorted(outputs):
        o = outputs[k]
        r = dict(o.indep)
        r["#"] = k
        r["Scenario"] = o.cfg.name
        rows.append(r)
    cols = ["#", "Scenario", "corr_delta_kappa", "se_corr_null",
            "corr_delta_turnover", "corr_kappa_turnover",
            "corr_delta_retgross", "corr_turnover_retgross"]
    return pd.DataFrame(rows)[cols].set_index("#")


def convention_table(outputs) -> pd.DataFrame:
    rows = []
    for k in sorted(outputs):
        o = outputs[k]
        rows.append({
            "#": k, "Scenario": o.cfg.name,
            "per-lot, position": o.disp["diff"],
            "per-lot, share-weighted": o.disp_frac["diff"],
            "average cost, position": o.disp_avgcost["diff"],
            "silent days included": pgr_plr(**{k: v for k, v in
                                              zip(("Gr", "Gp", "Lr", "Lp"),
                                                  (o.run.allday_counts[x] for x in ("Gr", "Gp", "Lr", "Lp")))})["diff"],
            "at basis (obs)": o.run.at_basis,
            "se (account)": o.boot["se_diff"],
            "se (position-day)": o.boot_naive["se_diff"],
            "understatement": o.boot["se_diff"] / o.boot_naive["se_diff"],
        })
    return pd.DataFrame(rows).set_index("#")


def discrimination_table(outputs) -> pd.DataFrame:
    """Two statistics that separate mechanisms the headline estimator cannot."""
    rows = []
    for k in sorted(outputs):
        o = outputs[k]
        rows.append({
            "#": k, "Scenario": o.cfg.name,
            "PGR-PLR": o.disp["diff"],
            "mean fraction sold": o.run.sold_frac[0],
            "share-weighted PGR-PLR": o.disp_frac["diff"],
            "momentum z, sold": o.run.sold_z[0],
            "momentum z, held": o.run.held_z[0],
            "z tilt": o.run.sold_z[0] - o.run.held_z[0],
        })
    return pd.DataFrame(rows).set_index("#")


def placebo_table(outputs) -> pd.DataFrame:
    rows = []
    for k in sorted(outputs):
        p = dict(outputs[k].placebo)
        p["#"] = k
        p["Scenario"] = outputs[k].cfg.name
        rows.append(p)
    return pd.DataFrame(rows)[["#", "Scenario", "n_trades", "n_pairs", "n_days",
                               "diff", "t_stat", "diff_log", "t_log"]].set_index("#")


def replicate_seeds(cfg, seeds, n_boot=400):
    """Re-run one scenario over fresh price panels and fresh populations."""
    rows = []
    for s in seeds:
        u = common_universe(seed=s)
        o = analyse(cfg, u, n_boot=n_boot, master_seed=s)
        rows.append({
            "seed": s,
            "PGR-PLR": o.disp["diff"], "se_diff": o.boot["se_diff"],
            "ratio": o.disp["ratio"],
            "beta_gross": o.over["ret_gross"]["beta"],
            "beta_net": o.over["ret_net"]["beta"],
        })
    return pd.DataFrame(rows).set_index("seed")
