"""The two estimators, plus the diagnostics that tell us whether to believe them.

Nothing here reads an injected parameter.  Both estimators see only what a
researcher with a brokerage extract would see: dated positions, purchase prices,
sale dates, portfolio values and trade volume.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import STREAM_BOOTSTRAP


# ---------------------------------------------------------------------------
# Disposition effect
# ---------------------------------------------------------------------------

def pgr_plr(Gr, Gp, Lr, Lp):
    """Pooled proportion of gains and losses realised.

    Conventions, all of them consequential and all of them stated in the report:
      * only days on which the account sold something contribute to any count;
      * a sale is counted at position level, so trimming half a lot is one
        realisation and the surviving stub is classified again the next day;
      * the basis is per-lot by default;
      * a position sitting exactly at its purchase price enters no count.
    """
    Gr, Gp, Lr, Lp = (float(np.sum(x)) for x in (Gr, Gp, Lr, Lp))
    pgr = Gr / (Gr + Gp) if (Gr + Gp) > 0 else np.nan
    plr = Lr / (Lr + Lp) if (Lr + Lp) > 0 else np.nan
    return {
        "Gr": Gr, "Gp": Gp, "Lr": Lr, "Lp": Lp,
        "PGR": pgr, "PLR": plr,
        "diff": pgr - plr,
        "ratio": pgr / plr if plr and plr > 0 else np.inf,
    }


def bootstrap_accounts(Gr, Gp, Lr, Lp, n_boot=1000, rng=None):
    """Resample whole accounts with replacement.

    Positions inside an account share the account's delta, its holding period and
    its securities, so they are anything but independent.  Resampling accounts is
    the only level at which the i.i.d. assumption is defensible here.
    """
    rng = rng or np.random.default_rng(0)
    Gr, Gp, Lr, Lp = (np.asarray(x, float) for x in (Gr, Gp, Lr, Lp))
    n = Gr.size
    idx = rng.integers(0, n, size=(n_boot, n))
    gr, gp = Gr[idx].sum(axis=1), Gp[idx].sum(axis=1)
    lr, lp = Lr[idx].sum(axis=1), Lp[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pgr = gr / (gr + gp)
        plr = lr / (lr + lp)
        ratio = pgr / plr
    diff = pgr - plr
    return {
        "se_PGR": float(np.nanstd(pgr, ddof=1)),
        "se_PLR": float(np.nanstd(plr, ddof=1)),
        "se_diff": float(np.nanstd(diff, ddof=1)),
        "se_ratio": float(np.nanstd(ratio, ddof=1)),
        "ci_diff": tuple(np.nanpercentile(diff, [2.5, 97.5])),
        "ci_ratio": tuple(np.nanpercentile(ratio, [2.5, 97.5])),
        "draws_diff": diff,
    }


def bootstrap_transactions(Gr, Gp, Lr, Lp, n_boot=1000, rng=None):
    """The naive alternative: resample position-days as if independent.

    Reported only so that the report can show by how much it understates the
    standard error.  It is not used for inference anywhere.
    """
    rng = rng or np.random.default_rng(0)
    counts = np.array([np.sum(Gr), np.sum(Gp), np.sum(Lr), np.sum(Lp)], float)
    total = counts.sum()
    draws = rng.multinomial(int(total), counts / total, size=n_boot).astype(float)
    gr, gp, lr, lp = draws.T
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = gr / (gr + gp) - lr / (lr + lp)
    return {"se_diff": float(np.nanstd(diff, ddof=1))}


# ---------------------------------------------------------------------------
# Overconfidence
# ---------------------------------------------------------------------------

CONTROLS = ["log_wealth", "n_pos", "beta_exposure", "vol_net"]


def account_frame(run) -> pd.DataFrame:
    pop = run.population
    df = pd.DataFrame({
        "delta": pop.delta,
        "kappa": pop.kappa,
        "wealth": pop.wealth,
        "log_wealth": np.log(pop.wealth),
        "n_pos": pop.n_pos.astype(float),
        "turnover": run.turnover,
        "ret_net": run.ret_net,
        "ret_gross": run.ret_gross,
        "vol_net": run.vol_net,
        "beta_exposure": run.beta_exposure,
        "cost_drag": run.ret_gross - run.ret_net,
        "sale_days": run.sale_days,
    })
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def turnover_regression(df, dep="ret_net", regressor="turnover", controls=None):
    """r_i = alpha + beta * Turnover_i + gamma' X_i + e_i, HC1 standard errors.

    One observation per account, so there is no cluster structure left to
    correct for; heteroskedasticity across account sizes is the live problem.
    """
    controls = CONTROLS if controls is None else controls
    if df[regressor].std() == 0:
        # A degenerate regressor (kappa in a scenario that injects none) has no
        # slope to estimate.  Returning zero here would read as a finding.
        return {"beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                "r2": np.nan, "n": int(len(df)), "model": None}
    X = sm.add_constant(df[[regressor] + list(controls)].astype(float))
    model = sm.OLS(df[dep].astype(float), X).fit(cov_type="HC1")
    return {
        "beta": float(model.params[regressor]),
        "se": float(model.bse[regressor]),
        "t": float(model.tvalues[regressor]),
        "p": float(model.pvalues[regressor]),
        "r2": float(model.rsquared),
        "n": int(model.nobs),
        "model": model,
    }


def overconfidence_block(df):
    """The Barber-Odean contrast, plus the exogenous-parameter cross-check.

    Regressing on kappa instead of on realised turnover replaces an outcome with
    the injected parameter that caused it.  If the turnover slope and the kappa
    slope disagree in sign, the turnover slope is being read backwards.
    """
    out = {}
    for dep in ("ret_gross", "ret_net"):
        out[dep] = turnover_regression(df, dep=dep)
        out[dep + "_kappa"] = turnover_regression(df, dep=dep, regressor="kappa")
    return out


# ---------------------------------------------------------------------------
# Independence diagnostics
# ---------------------------------------------------------------------------

def independence_block(df):
    def corr(a, b):
        return float(np.corrcoef(df[a], df[b])[0, 1]) if df[a].std() > 0 and df[b].std() > 0 else np.nan

    n = len(df)
    return {
        "n": n,
        "corr_delta_kappa": corr("delta", "kappa"),
        "se_corr_null": 1.0 / np.sqrt(max(n - 3, 1)),
        "corr_delta_turnover": corr("delta", "turnover"),
        "corr_kappa_turnover": corr("kappa", "turnover"),
        "corr_delta_retgross": corr("delta", "ret_gross"),
        "corr_turnover_retgross": corr("turnover", "ret_gross"),
    }
