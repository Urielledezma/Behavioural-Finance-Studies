"""Assemble the P01 report notebook from source, so the document is itself
reproducible.  Run `python build_notebook.py`, then execute the notebook.
"""

import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).parent
C = []


def md(src):
    C.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src):
    C.append(nbf.v4.new_code_cell(src.strip("\n")))


# The pre-analysis is embedded verbatim from the registered file rather than
# paraphrased, since a pre-registration rewritten after the results is not one.
PRE = (ROOT / "PRE_ANALYSIS.md").read_text(encoding="utf-8")
PRE_BODY = PRE.split("\n", 4)[4].strip()
PRE_QUOTED = "\n".join("> " + line if line else ">" for line in PRE_BODY.splitlines())

# =========================================================================== 0
md(r"""
# **Recovering injected behavioural biases from a synthetic trading record**
## **P01 — Behavioural Finance**

---

**Authors:** Alan Jesús Hernández Soto · Francisco Uriel Ledezma Chávez · Esteban Vega Campos · *(fourth member)*  
**Programme:** Ingeniería Financiera  
**Institution:** ITESO — Universidad Jesuita de Guadalajara  
**Course:** Comportamiento en las Finanzas y Toma de Decisiones  
**Professor:** Luis Felipe Gómez Estrada  
**Date:** 17 September 2026

---
""")

md(r"""
## Contents

0. [The project in plain terms](#s0)
1. [Pre-analysis statement](#s1)
2. [Simulator design](#s2)
   - [2.1 The security universe](#s21)
   - [2.2 Traders and the injected mechanism](#s22)
   - [2.3 Trading costs, and what "gross" means here](#s23)
   - [2.4 Estimator conventions](#s24)
3. [Results across the eight scenarios](#s3)
4. [Validation](#s4)
5. [The scenarios where something happened](#s5)
6. [Diagnosing the turnover slope on gross returns](#s6)
7. [Independence of the injected parameters](#s7)
8. [Standard errors and clustering](#s8)
9. [What the estimators cannot distinguish](#s9)
10. [Conclusions](#s10)
11. [References](#s11)
12. [Reproducibility](#s12)

---
""")

# =========================================================================== 0
md(r"""
<a id="s0"></a>

## 0. The project in plain terms

With real brokerage data nobody can check whether a measured bias is genuine or the product of a coding error, because the true behaviour of each investor is unknown, so we built a simulated market in which it is known by construction. A thousand simulated investors trade sixty simulated stocks for three years, and we hide two behaviours in their decision rules at strengths we fix in advance, namely a tendency to sell winners faster than losers (the **disposition effect**) and a tendency to trade too often (**overprecision**, the form of overconfidence in which investors treat their own information as more precise than it is), after which we run the two standard measurements from the literature on the resulting trading record and compare what they report against what we hid.

The table below is the whole result in one place, and the rest of the notebook explains how each row was obtained and why the wrong answers occur.

| # | What we hid in the traders | What the disposition measure reports | What the turnover regression reports | Correct? |
|---|---|---|---|---|
| 1 | Nothing | No effect (ratio 1.02) | No effect | Yes |
| 2 | Mild disposition | Disposition (ratio 1.74) | Trading *raises* returns | Only the first |
| 3 | Strong disposition | Strong disposition (ratio 7.39) | Trading *raises* returns | Only the first |
| 4 | Mild overtrading | No effect (ratio 1.00) | Trading costs 0.78 pp of return per unit of turnover | Yes |
| 5 | Strong overtrading | No effect (ratio 1.00) | Trading costs 0.65 pp per unit of turnover, weakly significant | Yes |
| 6 | Both | Disposition (ratio 6.67) | Trading *raises* returns | Only the first |
| 7 | Nothing, but traders rebalance monthly | Disposition (ratio 1.83) | No effect | **No** |
| 8 | Nothing, but traders expect prices to revert | Disposition (ratio 1.41) | Trading *raises* returns | **No** |

In five of the eight scenarios at least one measurement returns a confident answer that is wrong, and none of the five is a bug, since each measurement computes exactly what it claims to compute and the error lies in reading that number as the behaviour behind it.

**A short glossary, used throughout.** A position's *basis* is its purchase price, so a position is a *paper gain* when today's price is above the basis and a *paper loss* when it is below. **PGR** (proportion of gains realised) is the share of paper gains an investor sells on the days they sell anything, and **PLR** is the same share for losses, so if an investor holding 10 gains and 10 losses sells 2 gains and 1 loss that day, PGR = 2/10 = 0.20 and PLR = 1/10 = 0.10, and a ratio PGR/PLR above one is the disposition effect. A *hazard* is simply the probability of selling a position on a given day, which is how we write the behaviour into each trader. **Turnover** is the fraction of the portfolio traded in a year, where a turnover of 1 means the whole portfolio was sold and replaced once. **Gross** returns ignore trading costs and **net** returns include them. A **standard error** measures how much an estimate would move if the experiment were repeated, a **t-statistic** is the estimate divided by it (values beyond about 2 are conventionally called significant), and a **bootstrap** estimates the standard error by recomputing the statistic on many resamples of the data.

---
""")

# =========================================================================== 1
md(r"""
<a id="s1"></a>

## 1. Pre-analysis statement

The assignment asks for one paragraph, written before any estimator is run, that states the parameters to be injected and what each estimator should recover. Ours was written on 13 September 2026 before the first estimator was run, and it is reproduced below exactly as it stands in `PRE_ANALYSIS.md` so that it cannot be adjusted after the fact, while how each prediction fared is scored at the end of section 4.
""")

md(PRE_QUOTED)

code(r"""
import os, sys, pathlib, warnings

# Work from the repository root whether the notebook is launched from here or
# from notebooks/, so that `src/` and `results/` resolve the same way either way.
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
            if (p / "src" / "bfsim").is_dir())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace

from bfsim.config import (MASTER_SEED, PriceConfig, PopulationConfig, CostConfig,
                          EngineConfig, stream_rngs)
from bfsim.agents import draw_population
from bfsim import figures as fx
from bfsim.scenarios import (SCENARIOS, common_universe, analyse, run_all, results_table,
                             independence_table, convention_table, placebo_table,
                             discrimination_table, replicate_seeds)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.5f}")
FIG = "results/figures"
""")

# =========================================================================== 2
md(r"""
---

<a id="s2"></a>

## 2. Simulator design

<a id="s21"></a>

### 2.1 The security universe

Sixty stocks are simulated over 750 trading days, roughly three years, and instead of moving independently they share common drivers the way real stocks do, since each daily log return is built as

$$r_{j,t} \;=\; a_j \;+\; \beta^{M}_j f^{M}_t \;+\; \beta^{V}_j f^{V}_t \;+\; \beta^{S}_j f^{S}_t \;+\; \varepsilon_{j,t},$$

where $f^M$ is a market factor with 16 % annual volatility, $f^V$ and $f^S$ are two independent style factors with 10 % each, the betas measure how strongly stock $j$ responds to each factor (market betas are drawn around 1 and clipped to $[0.2,\,2.0]$), and $\varepsilon_{j,t}$ is the stock's own noise with an annual volatility between 20 % and 45 %. This factor structure matters for two reasons, because it reproduces the average pairwise correlation of about 0.19 that a real cross-section of stocks shows, and because it gives the overconfidence regression a meaningful risk control, since each portfolio's market beta then varies for reasons unrelated to how often its owner trades.

The intercept $a_j$ was changed during construction, and the reason is worth recording because it looked exactly like a violation of the assignment's central constraint. With a CAPM-style drift, in which riskier stocks earn more on average, the expected *log* return still falls with volatility, so across our sixty stocks it varied with a standard deviation of 3.0 percentage points a year and a correlation of $-0.90$ with idiosyncratic volatility, which means that any rule selecting stocks on their past performance acquires a drift advantage or disadvantage it never had information about. We therefore impose the same expected log return of 6 % a year on every stock, which leaves the arithmetic means differing only by half the variance, an effect that cannot make a portfolio compound faster.

**The big constraint holds by construction**, because the entire price panel is drawn from its own random stream before a single trader exists and the trading engine only ever reads it, so no trading decision can reach the return that follows it, and section 4 tests the claim empirically in any case.
""")

code(r"""
price_cfg = PriceConfig()
universe = common_universe(price_cfg)

r = np.diff(np.log(universe.prices), axis=0)
corr = np.corrcoef(r.T)
iu = np.triu_indices_from(corr, 1)
print(f"stocks {universe.n_securities}   trading days {universe.n_days}")
print(f"annualised volatility   mean {r.std(0).mean()*np.sqrt(252):.3f}   "
      f"range [{r.std(0).min()*np.sqrt(252):.3f}, {r.std(0).max()*np.sqrt(252):.3f}]")
print(f"mean pairwise return correlation   {corr[iu].mean():.4f}")
print(f"market beta   mean {universe.beta_market.mean():.3f}   sd {universe.beta_market.std():.3f}")

# What the rejected CAPM-style drift would have implied for expected log returns
total_var = ((universe.beta_market * price_cfg.market_vol_annual) ** 2
             + (universe.beta_style ** 2).sum(1) * price_cfg.style_vol_annual ** 2
             + universe.idio_vol ** 2)
capm_log_drift = (price_cfg.risk_free_annual
                  + universe.beta_market * price_cfg.market_premium_annual - 0.5 * total_var)
print(f"CAPM-style drift, rejected:  sd of expected log return {capm_log_drift.std():.4f}   "
      f"corr with idiosyncratic vol {np.corrcoef(capm_log_drift, universe.idio_vol)[0,1]:+.3f}")
print(f"drift used instead: {price_cfg.common_log_drift_annual:.3f} per year on every stock")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for j in np.linspace(0, universe.n_securities - 1, 12).astype(int):
    ax[0].plot(universe.prices[:, j] / universe.prices[0, j], lw=0.9, color=fx.BLUE, alpha=0.5)
ax[0].plot((universe.prices / universe.prices[0]).mean(axis=1), lw=2.2, color=fx.DARK,
           label="universe mean")
ax[0].set_title("Twelve of the sixty price paths"); ax[0].legend(fontsize=8.5)
ax[0].set_xlabel("trading day"); ax[0].set_ylabel("price, indexed to 1")
ax[1].hist(corr[iu], bins=40, color=fx.GOLD, edgecolor="white", linewidth=0.4)
ax[1].axvline(corr[iu].mean(), color=fx.DARK, lw=1.6, ls="--", label="mean")
ax[1].set_title("Pairwise daily return correlations"); ax[1].legend(fontsize=8.5)
ax[1].set_xlabel("correlation")
plt.tight_layout(); plt.savefig(f"{FIG}/universe.png", dpi=160, bbox_inches="tight"); plt.show()
""")

md(r"""
<a id="s22"></a>

### 2.2 Traders and the injected mechanism

Each scenario generates a fresh population of $N=1000$ accounts, and each account receives a disposition strength $\delta_i$, an overprecision intensity $\kappa_i$, a starting capital $W_i$ between \$10,000 and \$500,000 drawn so that small accounts are more common than large ones, and a target number of positions $n_i$ between 5 and 30.

We read the $\delta$ and $\kappa$ of the scenario table as the *average* of the population rather than as a value every trader shares, drawing each trader's parameter from a Beta distribution with that mean, because without differences across traders there would be nothing to correlate in section 7 and the turnover regression would have almost no variation to work with. The two are drawn from separate random streams, so their independence follows from the construction rather than from checking it afterwards.

The behaviour is written into a single daily probability of selling each position the trader holds,

$$h_{i,j,t} \;=\; \underbrace{h_0\,\bigl(1 + 4\kappa_i\bigr)}_{\text{how often the trader acts}} \;\times\; \underbrace{\begin{cases} 1+\delta_i & \text{price above basis} \\ 1 & \text{price equal to basis} \\ 1-\delta_i & \text{price below basis}\end{cases}}_{\text{which positions it closes}}$$

with $h_0 = 0.004$, which makes an unbiased trader replace their portfolio about once a year, while $\kappa$ raises that frequency and $\delta$ tilts the selling toward winners. Nothing in this expression is PGR, which is the point the assignment insists on, since the estimator reads a quantity one level above the one we inject and the gap between them depends on how long positions are held and on the price path they follow, while the money from every sale is reinvested the same day in a stock chosen at random, independently of its past and its future.
""")

code(r"""
demo = draw_population(PopulationConfig(n_traders=4000, delta_mean=0.8, kappa_mean=0.3),
                       stream_rngs(99))
print(f"delta   mean {demo.delta.mean():.3f}  sd {demo.delta.std():.3f}")
print(f"kappa   mean {demo.kappa.mean():.3f}  sd {demo.kappa.std():.3f}")
print(f"realised corr(delta, kappa) = {np.corrcoef(demo.delta, demo.kappa)[0,1]:+.4f}  "
      f"(standard error under independence {1/np.sqrt(demo.n_traders-3):.4f})")
print(f"capital  median ${np.median(demo.wealth):,.0f}   positions  mean {demo.n_pos.mean():.1f}")

eng = EngineConfig()
d = np.linspace(0, 1, 100)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(d, eng.base_hazard*(1+d), color=fx.GOLD, lw=2.2, label="price above basis")
ax[0].plot(d, eng.base_hazard*(1-d), color=fx.BLUE, lw=2.2, label="price below basis")
ax[0].axhline(eng.base_hazard, color=fx.DARK, ls="--", lw=1.1, label=r"$h_0$")
ax[0].set_xlabel(r"$\delta_i$"); ax[0].set_ylabel("daily probability of selling")
ax[0].set_title(r"$\delta$ tilts selling toward winners"); ax[0].legend(fontsize=8)
k = np.linspace(0, 1, 100)
ax[1].plot(k, eng.base_hazard*(1+eng.kappa_multiplier*k)*252, color=fx.BLUE, lw=2.2)
ax[1].set_xlabel(r"$\kappa_i$"); ax[1].set_ylabel("expected sales per position-year")
ax[1].set_title(r"$\kappa$ sets how often the trader acts")
plt.tight_layout(); plt.savefig(f"{FIG}/mechanism.png", dpi=160, bbox_inches="tight"); plt.show()
""")

md(r"""
<a id="s23"></a>

### 2.3 Trading costs, and what "gross" means here

Every trade pays a commission of 10 basis points of its value with a minimum of one dollar, and crosses a bid-ask spread of 30 basis points, so buyers pay 15 basis points above the mid price and sellers receive 15 below it, which makes a full round trip of the portfolio cost $2\times(10+15)=50$ basis points, the number to keep in mind when reading the overconfidence slope in section 3.

Each account also runs a *shadow portfolio* that copies its decisions exactly, buying and selling the same stocks on the same days, but trades at the mid price and pays no commission, and **gross return is the return of that shadow portfolio while net return is the return of the real one**. This definition removes by construction one of the problems the assignment warns about, since a gross return computed by adding costs back at the end would still carry the half-spread inside the purchase and sale prices, whereas the shadow portfolio never paid it, so any gap between gross and net is trading cost and nothing else.
""")

md(r"""
<a id="s24"></a>

### 2.4 Estimator conventions

The disposition estimator counts, on every day an account sells something, how many of the positions it holds are gains sold ($G_r$), gains kept ($G_p$), losses sold ($L_r$) and losses kept ($L_p$), and pools the four counts across all accounts and days,

$$\widehat{PGR} = \frac{G_r}{G_r+G_p}, \qquad \widehat{PLR} = \frac{L_r}{L_r+L_p}.$$

Four choices decide what those counts mean, and because each one changes the answer we compute the alternative alongside our choice, with the magnitudes reported in section 5.

| Question | Our choice | Alternative also computed | Where it matters |
|---|---|---|---|
| Do days without a sale count? | No, only sale days contribute | Every day counts | Everywhere, since the difference shrinks by a factor of 4 to 12 when silent days are included |
| Is selling half a position one sale or half a sale? | One sale, and the remaining half is classified again the next day | Weighted by the fraction sold | Only in scenario 7, the only one that sells partial positions |
| Which purchase price? | Each purchase is its own lot | Average cost across lots of the same stock | Attenuates the estimate by about 8 % where the effect is large |
| A position exactly at its purchase price? | Counted as neither gain nor loss | Not needed | Never observed, since prices are continuous |

Standard errors come from a bootstrap that resamples whole **accounts**, since the positions of one account share that account's behaviour and are therefore not independent observations, and the naive version that resamples individual positions is computed only to show by how much it understates the uncertainty.

---
""")

# =========================================================================== 3
md(r"""
<a id="s3"></a>

## 3. Results across the eight scenarios

Each scenario regenerates the full population, while all eight share one price panel, because null recovery and monotonicity are comparisons *across* scenarios and a different market for each would mix the behavioural difference with a different realised market, a choice section 4 checks by redrawing the market eight times.
""")

code(r"""
%%time
outputs = run_all(universe, n_boot=2000, verbose=False)
table = results_table(outputs)
table.to_csv("results/scenarios.csv")

disp_cols = ["Scenario", "delta_inj", "kappa_inj", "PGR", "se_PGR", "PLR", "se_PLR",
             "PGR-PLR", "se_diff", "t_diff", "PGR/PLR", "se_ratio", "Turnover"]
table[disp_cols]
""")

code(r"""
table[["Scenario", "beta_gross", "se_gross", "t_gross", "beta_net", "se_net", "t_net"]]
""")

code(r"""
fx.pgr_plr_bars(table, f"{FIG}/pgr_plr.png"); plt.show()
cost_rate = 2 * (CostConfig().commission_bps + CostConfig().half_spread_bps) * 1e-4
fx.slopes(table, cost_rate, f"{FIG}/slopes.png"); plt.show()
drag = pd.DataFrame({k: {"mean turnover": outputs[k].df.turnover.mean(),
                         "mean cost drag, per year": outputs[k].df.cost_drag.mean()}
                     for k in outputs}).T
print(f"round-trip cost per unit of annual turnover: {cost_rate:.5f}")
drag
""")

md(r"""
**Reading the table.** The null is quiet, the disposition estimator rises with the injected $\delta$, and the two overtrading scenarios leave it undisturbed at $+0.0001$ and $-0.0003$, while on the overconfidence side scenarios 4 and 5 reproduce the pattern Barber and Odean report, with a gross slope indistinguishable from zero ($-0.0018$ and $-0.0006$) and a negative net slope ($-0.0078$ and $-0.0065$). Both net slopes lie within one standard error of $-0.005$, which is the round-trip cost written into the model, so the regression is not measuring a behavioural parameter at all but recovering the commission and spread schedule, which is at once the strongest validation in the exercise and the clearest limit on what the test can claim.

This also explains why the net slope does not become more negative from scenario 4 to scenario 5, contrary to what we predicted, since the slope is a cost per unit of turnover and the cost schedule is identical in both, so what rises with $\kappa$ is the level of the drag, from 1.34 % of return a year in scenario 4 to 2.53 % in scenario 5, and not the coefficient.

The rows that deserve attention are 6, 7 and 8, taken up in section 5, and the positive gross slopes of scenarios 2, 3, 6 and 8, taken up in section 6.

---
""")

# =========================================================================== 4
md(r"""
<a id="s4"></a>

## 4. Validation

Four things have to hold before the results can be believed, namely that the null comes back empty, that the recovered effect moves with the injected parameter, that each estimator stays quiet when only the other one's parameter is active, and that no information about future prices leaked into the trading decisions.

**Leakage.** The test compares the return over the next 20 days of every stock an agent bought against the average stock on the same day, so a rule that systematically picked future winners or losers would show a non-zero difference. The first version of this test returned a $t$ of $-7.5$ in scenario 7, which would have meant a violation of the constraint, yet the cell below shows the panel itself carries no predictability and the fault lay in the test, because a thousand accounts buying the same stock on the same day is one purchase counted a thousand times. Keeping each (day, stock) pair once and computing the standard error from daily averages, so that purchases sharing the same market day are not treated as independent, brings every scenario inside $|t|\le 2.2$, which across eight tests is what the absence of leakage looks like.
""")

code(r"""
placebo = placebo_table(outputs)
placebo.to_csv("results/placebo.csv")

# The first, naive version of the test: every account's purchase counted separately.
run7, h = outputs[7].run, 20
keep = run7.buy_days + h < universe.n_days
d7, s7 = run7.buy_days[keep], run7.buy_secs[keep]
fwd = universe.prices[d7 + h, s7] / universe.prices[d7, s7] - 1
bench = (universe.prices[h:] / universe.prices[:-h] - 1)[d7].mean(axis=1)
excess = fwd - bench
print(f"naive placebo, scenario 7: n = {excess.size:,}   "
      f"t = {excess.mean() / (excess.std(ddof=1) / np.sqrt(excess.size)):+.2f}")

# Does the price panel itself contain short-horizon reversal?  It should not.
P = universe.prices
past = P[h:-h] / P[:-2*h] - 1;  past -= past.mean(axis=1, keepdims=True)
nxt = P[2*h:] / P[h:-h] - 1;    nxt -= nxt.mean(axis=1, keepdims=True)
losers = np.take_along_axis(nxt, np.argsort(past, axis=1)[:, :12], axis=1).mean(axis=1)
print(f"panel check: mean corr(past 20d, next 20d) = "
      f"{np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(past, nxt)]):+.4f}   "
      f"bottom-quintile forward excess t = {losers.mean() / (losers.std(ddof=1) / np.sqrt(losers.size)):+.2f}")
placebo
""")

md(r"""
**Null recovery.** Within the shared price panel the null returns $\widehat{PGR}-\widehat{PLR} = 0.00085$ with a standard error of 0.00044, a $t$ of 1.95 that sits on the edge of significance for an effect that is zero by construction, whereas across eight independently redrawn markets the mean is 0.00022 with a standard deviation of 0.00050, a $t$ of 1.2, so the null is genuinely quiet and what misleads is the single-panel $t$. The reason is that every account in the sample lives through the same market, and resampling accounts, which correctly handles the dependence *within* an account, still treats a shared market shock as independent information across accounts, so a disposition ratio of 1.02 with a $t$ of 2 measured on one brokerage dataset is not evidence of anything, and finding this out requires redrawing the market, which a researcher with a single real dataset cannot do.

**Monotonicity.** The recovered difference rises from 0.00085 to 0.0266 to 0.0749 as $\delta$ goes from 0 to 0.3 to 0.8, and the ratio from 1.02 to 1.74 to 7.39, with the scenario 3 ratio stable at $7.34\pm0.38$ across the eight redrawn markets, and the recovery is monotone without being proportional because $\delta$ multiplies a daily probability while the ratio measures what that multiplier does to a proportion once holding periods and price paths intervene.

**Cross-estimator quiet holds in one direction only.** The disposition estimator stays silent in the two overtrading scenarios, at $t = 0.33$ and $t = -1.24$, but the overconfidence estimator does not stay silent in the two disposition scenarios, returning gross slopes of $+0.063$ and $+0.151$ with $t$ of 4.2 and 11.2 in populations with no overprecision whatsoever, so the disposition mechanism contaminates the overconfidence estimator while the reverse does not happen, a failed check that section 6 traces to reverse causality.
""")

code(r"""
%%time
seeds = [20260913, 7, 101, 4242, 31337, 2, 13, 55]
reps = {}
for i in (0, 2, 6):
    cfg = SCENARIOS[i]
    reps[cfg.index] = replicate_seeds(cfg, seeds, n_boot=300)
    reps[cfg.index].to_csv(f"results/replication_s{cfg.index}.csv")

summary = pd.DataFrame({
    f"Scenario {k}": r[["PGR-PLR", "ratio", "beta_gross"]].agg(["mean", "std", "min", "max"]).stack()
    for k, r in reps.items()})
summary
""")

code(r"""
fx.monotonicity(table, f"{FIG}/monotonicity.png"); plt.show()
""")

md(r"""
**How the pre-registered predictions fared.** Most of the predictions held, three failed in whole or in part, and one outcome was not predicted at all.

| Prediction in section 1 | Outcome | Verdict |
|---|---|---|
| Scenario 1 quiet on both estimators | Difference 0.00085 ($t$ = 1.95 in one market, 1.2 across eight) and gross slope $t$ = 0.08 | Held |
| Scenarios 2 and 3 positive, larger in 3, monotone but not equal to $\delta$ | 0.0266 and 0.0749, ratios 1.74 and 7.39 | Held |
| Scenarios 4 and 5 quiet on disposition, gross slope near zero, net slope negative | Disposition $t$ of 0.33 and $-1.24$, gross slope $t$ of $-0.53$ and $-0.16$, net slope $-0.0078$ and $-0.0065$ | Held |
| Net slope *more negative* in 5 than in 4 | $-0.0065$ against $-0.0078$, since the slope is a cost rate | **Failed** |
| Scenario 6 ratio attenuated relative to scenario 3 | 6.67 against 7.39, although the difference rose | Held |
| Scenarios 7 and 8 positive and significant with $\delta = 0$ | $t$ of 70 and 38 | Held |
| Reverse causality makes the gross slope positive in scenarios 2, 3, 6 and 7 | Positive and significant in 2, 3 and 6, while in 7 it is $t$ = 0.89 in the shared market yet positive in all eight redrawn ones | Partly failed |
| No cash drag and no spread in the gross measure | Gross slope $t$ of $-0.53$ and $-0.16$ in scenarios 4 and 5 | Held |
| $\mathrm{corr}(\delta,\kappa)$ near zero | $-0.040$ against a standard error of 0.032 | Held |
| $\mathrm{corr}(\delta,\text{turnover})$ materially negative | $+0.136$ in scenario 2, $-0.112$ in 3 and $-0.617$ in 6 | **Failed** at low $\delta$ |
| *Not predicted* | Scenario 8 also produces a positive gross slope, $+0.064$ with $t$ = 4.4 | New |

---
""")

# =========================================================================== 5
md(r"""
---

<a id="s5"></a>

## 5. The scenarios where something happened

### 5.1 Scenario 6: the difference is not a scale-free measure of the bias

Scenario 6 injects $\delta=0.8$ and $\kappa=0.8$ together. We predicted the measured
disposition would be **attenuated** relative to scenario 3, on the reasoning that higher churn
shortens holding periods and leaves positions closer to their purchase price. **That prediction
is wrong in the difference and right in the ratio**, and the two disagreeing is the point.
The difference rises from 0.0749 to 0.0985 while the ratio falls from 7.39 to 6.67.

The reason is that $\widehat{PGR}-\widehat{PLR}$ is not scale-free. Raising $\kappa$ raises
both realisation rates roughly proportionally — PGR goes from 0.0866 to 0.1158 and PLR from
0.0117 to 0.0174 — so a fixed multiplicative tilt produces a larger absolute gap simply because
both levels are larger. The ratio strips the level out and does fall, by the modest 10 % that
the shortened holding period justifies. A researcher reporting only the difference would
conclude that scenario 6's population is 32 % more loss-averse than scenario 3's, when the two
carry an identical injected $\delta$ and differ only in how often they trade. Odean reports
both statistics; this is why.
""")

code(r"""
conv = convention_table(outputs)
conv.to_csv("results/conventions.csv")
conv
""")

md(r"""
### 5.2 Scenario 7: rebalancing, and the convention that gives it away

Scenario 7 injects nothing at all. Accounts hold a target of equal weights and, once a month,
trim any position whose weight has drifted more than 25 % above target, redistributing the
proceeds to the positions that are below it. There is no reference point anywhere in the rule
and no parameter to recover. The estimator nevertheless returns
$\widehat{PGR}-\widehat{PLR} = 0.0334$ with a $t$ of 70, and a ratio of 1.83 — squarely inside
the range Odean reports for real brokerage accounts, and larger than the 1.74 that scenario 2
produces from a genuine injected $\delta$ of 0.3. Across eight redrawn panels it is
$0.0320\pm0.0027$, so it is not a fluke of one market. **A mechanical portfolio rule with no
behavioural content is indistinguishable from moderate loss aversion on the headline statistic.**

The mechanism is arithmetic rather than psychological. A position that rises gains weight and
breaches the band; a position that falls loses weight and is topped up. The rule sells winners
and buys losers because that is what equal-weighting means, and the disposition estimator has
no way to know that the seller was following a weight target rather than a reference point.

What does give it away is the partial-sale convention. Rebalancing trims: the mean fraction of
a position sold when a sale occurs is 0.712 in scenario 7 and exactly 1.000 in all seven other
scenarios. Counting realisations share-weighted rather than at position level therefore
collapses scenario 7's estimate from 0.0334 to 0.0075, a factor of 4.4, while leaving every
other scenario unchanged to five decimal places. That is a usable test on real data: if a
measured disposition effect largely disappears when partial sales are weighted by size, the
selling is rebalancing.

### 5.3 Scenario 8: a belief in reversal, wearing the same clothes

Scenario 8 also injects nothing. Agents believe past winners revert, so the sell hazard is
scaled by $1 + 0.6 z_{j,t}$ where $z$ is the security's cross-sectionally standardised
trailing 20-day return. The rule never looks at the purchase price. Because trailing return
and gain-versus-basis are correlated, the estimator returns 0.0169 with a $t$ of 38 and a ratio
of 1.41.

This one is harder to unmask than rebalancing, because the sales are whole positions and the
share-weighted convention does not move it at all (0.0169 under both). The tell has to come
from outside the purchase price.
""")

code(r"""
disc = discrimination_table(outputs)
# The ratio is only meaningful where a disposition effect was measured at all;
# dividing a momentum tilt of 0.002 by a difference of 0.0001 is noise over noise.
disc["z tilt per unit of measured disposition"] = np.where(
    disc["PGR-PLR"] > 0.01, disc["z tilt"] / disc["PGR-PLR"], np.nan)
disc.to_csv("results/discrimination.csv")
disc
""")

md(r"""
The trailing-momentum tilt — the mean standardised past return of what was sold, minus that of
what was held on the same days — separates the mechanisms, but only in ratio to the disposition
measured alongside it. A reference-point rule does tilt momentum, because winners against basis
are usually winners against the recent past too: scenario 3 sells positions whose trailing
$z$ is 0.267 above what it holds. But per unit of measured disposition the confounds are far
more momentum-driven, at 32.4 for scenario 8 and 17.2 for scenario 7 against 3.6 for scenario 3
and 3.3 for scenario 2. An account whose sales are extremely momentum-selective yet only mildly
disposition-selective is not following a reference point.

---
""")

# =========================================================================== 6
md(r"""
<a id="s6"></a>

## 6. Diagnosing the turnover slope on gross returns

The assignment asks which of four causes is responsible for a negative slope of gross returns
on turnover. **We never observe one.** The gross slope is quiet in every scenario where
$\kappa$ carries the variation — $-0.0018$ with $t=-0.53$ in scenario 4 and $-0.0006$ with
$t=-0.16$ in scenario 5 — and it is strongly *positive* in scenarios 2, 3, 6 and 8. The
diagnosis therefore runs in two parts: why nothing negative appears, and what the positive
slope is.

**Why nothing negative appears.** Leakage is excluded by the placebo of section 4, with every
scenario inside $|t|\le2.2$ on de-duplicated, day-clustered purchases. The spread is excluded
by definition, since gross return is a shadow portfolio filled at the mid rather than an
after-the-fact subtraction from a price path that already contains the spread. Cash drag is
excluded by construction, because proceeds are redeployed on the day of sale. The fourth cause,
compounding path effects, is present but small here: in scenario 5 the correlation between
turnover and gross return is $-0.020$, because redeployment is instantaneous and into an
equally random security, so turnover changes little about the portfolio's geometric return.

Cash drag is worth switching on rather than merely asserting away, so we re-run scenario 5 with
settlement lags of one, three and five days.
""")

code(r"""
%%time
rows = []
for lag in (0, 1, 3, 5):
    o = analyse(replace(SCENARIOS[4], engine=replace(EngineConfig(), settlement_lag=lag)),
                universe, n_boot=300)
    rows.append({"settlement lag": lag,
                 "beta_gross": o.over["ret_gross"]["beta"], "t_gross": o.over["ret_gross"]["t"],
                 "beta_net": o.over["ret_net"]["beta"], "t_net": o.over["ret_net"]["t"],
                 "mean gross return": o.df.ret_gross.mean(),
                 "mean turnover": o.df.turnover.mean()})
drag = pd.DataFrame(rows).set_index("settlement lag")
drag.to_csv("results/cash_drag.csv")
fx.cash_drag(drag.index.tolist(), drag["beta_gross"], drag["beta_net"], f"{FIG}/cash_drag.png")
plt.show()
drag
""")

md(r"""
One idle day between sale and redeployment takes the gross slope from $-0.0006$ ($t=-0.16$) to
$-0.0220$ ($t=-4.77$), and drives the population's mean gross return from 18.6 % to 16.2 % and
on to 13.1 % at a five-day lag. The mechanism needs no informed trading whatever: a
high-turnover account simply spends more of its life out of a market that drifts upward. This
is the scenario in which a researcher would wrongly conclude that frequent traders pick bad
securities, and the tell is that the damage scales with time spent in cash rather than with
anything about the securities bought — which the placebo would confirm by staying silent.

**What the positive slope is.** In scenarios 2, 3, 6 and 8 the gross slope is positive and
significant, reaching $+0.151$ with $t=11.2$ in scenario 3. This is reverse causality, and the
simulator lets us prove it rather than argue it. The injected parameter $\delta$ is
uncorrelated with gross return at $-0.026$, so disposition strength does not cause performance.
Yet realised turnover correlates $+0.397$ with gross return in the same scenario. Turnover is
therefore partly an *outcome* of the return: an account whose securities happened to rise is
holding more winners, and a rule that sells winners at $(1+\delta)$ and holds losers at
$(1-\delta)$ converts that luck into trades. The regression reads the arrow backwards.

Where $\kappa$ provides genuine exogenous variation we can go further and replace the outcome
with its cause.
""")

code(r"""
inst = pd.DataFrame({
    k: {"beta on realised turnover, gross": outputs[k].over["ret_gross"]["beta"],
        "beta on injected kappa, gross": outputs[k].over["ret_gross_kappa"]["beta"],
        "beta on injected kappa, net": outputs[k].over["ret_net_kappa"]["beta"],
        "corr(delta, ret_gross)": outputs[k].indep["corr_delta_retgross"],
        "corr(turnover, ret_gross)": outputs[k].indep["corr_turnover_retgross"]}
    for k in (2, 3, 5, 6, 7, 8)}).T
inst.index.name = "#"
inst
""")

md(r"""
In scenario 5, where the selling rule never looks at the purchase price, regressing on the
injected $\kappa$ instead of on realised turnover gives $+0.009$ on gross and $-0.015$ on net —
the Barber–Odean contrast in its clean form, with the cost effect intact and no endogeneity. In
scenario 6, where both parameters are active, the same substitution flips the gross coefficient
from $+0.028$ to $+0.008$ and turns the net coefficient from $+0.022$ to $-0.009$. The sign of
the net slope, which is the entire behavioural claim, reverses depending on whether the
right-hand side holds an outcome or a cause. In scenarios 2, 3, 7 and 8 no such substitution is
available because $\kappa$ is degenerate, which is precisely the situation a researcher with
real data is always in.

---
""")

# =========================================================================== 7
md(r"""
<a id="s7"></a>

## 7. Independence of the injected parameters
""")

code(r"""
indep = independence_table(outputs)
indep.to_csv("results/independence.csv")
indep
""")

code(r"""
%%time
rows = []
for rho in (0.0, 0.7, -0.7):
    o = analyse(replace(SCENARIOS[5], copula_rho=rho), universe, n_boot=300)
    rows.append({"copula rho": rho,
                 "realised corr(delta,kappa)": o.indep["corr_delta_kappa"],
                 "corr(delta,turnover)": o.indep["corr_delta_turnover"],
                 "corr(kappa,turnover)": o.indep["corr_kappa_turnover"],
                 "beta_gross": o.over["ret_gross"]["beta"],
                 "beta_net": o.over["ret_net"]["beta"],
                 "PGR-PLR": o.disp["diff"]})
copula = pd.DataFrame(rows).set_index("copula rho")
copula.to_csv("results/copula.csv")
copula
""")

code(r"""
fx.turnover_vs_parameters(outputs[6].df, outputs[5].df, f"{FIG}/turnover_params.png"); plt.show()
""")

md(r"""
**(1) The draws are independent.** $\delta_i$ and $\kappa_i$ come from separate generator
streams spawned off one master seed, so neither draw can shift the other.

**(2) The realised correlation.** Scenario 6 is the only scenario in which both parameters
carry dispersion, and there $\mathrm{corr}(\delta_i,\kappa_i) = -0.040$ against a null standard
error of $1/\sqrt{N-3} = 0.032$, a $t$ of $-1.28$. Everywhere else one of the two is degenerate
and the correlation is undefined rather than zero, which the table reports as a missing value.

**(3) The correlation between $\delta_i$ and realised turnover is not zero, and its sign
changes.** It is $+0.136$ in scenario 2, $-0.112$ in scenario 3 and $-0.617$ in scenario 6. The
injected parameter and the behaviour it produces are different objects, and the map between
them runs through the composition of the portfolio. Write the account's average hazard as
$h_0\bigl[(1+\delta)p_g + (1-\delta)p_l\bigr] = h_0\bigl[1+\delta(p_g-p_l)\bigr]$, where $p_g$
is the share of positions currently showing a gain. Two forces act in opposite directions. For
a given composition, raising $\delta$ raises the average hazard, because a drifting market
leaves more gains than losses and the $(1+\delta)$ arm applies to the larger pool — this is what
produces the positive correlation at $\delta=0.3$. But composition is itself endogenous to
$\delta$: an account that holds losers at $(1-\delta)$ accumulates them, $p_g$ falls, and the
tilt has a smaller pool to act on. At $\delta=0.8$ the second force dominates and the
correlation turns negative, strongly so when higher churn in scenario 6 lets the composition
effect express itself faster. Turnover is thus a *fixed point* of the mechanism rather than a
monotone function of the parameter, and no researcher measuring turnover is measuring $\kappa$.

**(4) What correlated parameters would do.** Coupling the two draws through a Gaussian copula
leaves the disposition estimate almost untouched, at 0.0985, 0.1040 and 0.1004 for
$\rho = 0,\,+0.7,\,-0.7$, and moves the overconfidence slope by more than half its value: the
gross coefficient runs $+0.028$, $+0.045$, $+0.022$ and the net coefficient $+0.022$, $+0.038$,
$+0.016$. The coefficient that is supposed to price trading moves by 60 % because of a
parameter that has nothing to do with costs. The reason is visible in the same table: a
positive $\rho$ partly cancels the composition effect described above, taking
$\mathrm{corr}(\delta,\text{turnover})$ from $-0.617$ to $-0.257$, so accounts with strong
disposition no longer trade distinctively little and the endogenous part of turnover is
reweighted. With the parameters correlated by construction, $\beta$ is a blend of the cost of
trading and the disposition mechanism's effect on trade frequency, and is not interpretable as
either.

---
""")

# =========================================================================== 8
md(r"""
<a id="s8"></a>

## 8. Standard errors and clustering

Positions inside an account share the account's $\delta$, its holding-period distribution and
its securities. Treating position-days as independent is therefore wrong, and the simulator
prices exactly how wrong.
""")

code(r"""
se = conv[["Scenario", "se (account)", "se (position-day)", "understatement"]].copy()
se["t if clustered"] = table["PGR-PLR"] / conv["se (account)"]
se["t if not clustered"] = table["PGR-PLR"] / conv["se (position-day)"]
se
""")

code(r"""
draws = {k: (f"{k}. {outputs[k].cfg.name}", outputs[k].boot["draws_diff"], k in (7, 8))
         for k in (1, 3, 7, 8)}
fx.bootstrap_density(draws, f"{FIG}/bootstrap.png"); plt.show()
for k in (1, 3, 6, 7, 8):
    lo, hi = outputs[k].boot["ci_diff"]
    rlo, rhi = outputs[k].boot["ci_ratio"]
    print(f"scenario {k}: diff 95% CI [{lo:+.5f}, {hi:+.5f}]   ratio 95% CI [{rlo:.3f}, {rhi:.3f}]")
""")

md(r"""
The account bootstrap gives standard errors between 1.0 and 4.6 times the naive position-day
bootstrap, and the multiple tracks how concentrated the effect is within accounts: it is 1.00
in the null and in the turnover-only scenarios, where realisations really are close to
independent coin flips, 2.32 in scenario 3 and 4.63 in scenario 6, where each account's
$\delta$ makes all of its position-days move together. In scenario 6 that factor takes the
$t$-statistic from 66 to 308 — both reject, so nothing turns on it here, but the same factor
applied to a genuinely marginal effect manufactures significance out of nothing.

Section 4 adds the caveat that matters more. Clustering by account handles dependence *within*
an account and does nothing about the dependence induced by a single shared price path across
accounts, which is why the null's within-panel $t$ of 1.95 coexists with a mean effect of
0.00022 across redrawn panels. In real data the panel cannot be redrawn, so the honest reading
is that account-clustered standard errors are a lower bound on the uncertainty and a disposition
ratio inside roughly $1.00\pm0.03$ should not be reported as a finding at all.

For the overconfidence regression the unit of observation is the account itself, so there is no
cluster structure left to correct; HC1 handles the heteroskedasticity that account size
introduces.

---
""")

# =========================================================================== 9
md(r"""
<a id="s9"></a>

## 9. What the estimators cannot distinguish

The eight scenarios produce three pairs that the headline statistics cannot tell apart, and
each has a different remedy.

**A reference point cannot be told from a weight target.** Scenario 2 injects $\delta=0.3$ and
returns a ratio of 1.74; scenario 7 injects nothing, rebalances monthly, and returns 1.83. Both
sit inside the range reported for real brokerage data. Separating them needs the *size* of each
sale relative to the position, which brokerage extracts do carry: rebalancing trims, with a mean
sold fraction of 0.712 against 1.000 for a reference-point rule, and weighting realisations by
share collapses the rebalancing estimate by a factor of 4.4 while leaving the genuine effect
untouched. Holdings-level weights and, better, any record of a stated target allocation would
settle it outright.

**A reference point cannot be told from a belief about the price path.** Scenario 8's agents
never observe their purchase price, yet return a ratio of 1.41, because trailing return and
gain-versus-basis are correlated. Partial-sale weighting does not help, since the sales are
whole. What separates them is the trailing return of the security, which is observable: per unit
of measured disposition the momentum tilt is 32.4 in scenario 8 against 3.6 in scenario 3. The
decisive test is a joint model of the sale decision on both the gain indicator and the trailing
return, which is how Grinblatt and Han separate the two channels; if the gain indicator loses
its explanatory power once momentum is included, the reference point was never doing the work.
Neither PGR nor PLR can perform that test, because both compress the decision into a single
binary classification against one reference price.

**A cost of trading cannot be told from an effect of performance on trading.** Wherever the
selling rule conditions on the purchase price, turnover is partly an outcome of the return, and
the regression of return on turnover reads the arrow backwards: scenario 3's gross slope is
$+0.151$ even though the injected $\delta$ correlates $-0.026$ with gross return. Within the
simulator the fix is to regress on the injected $\kappa$, which flips scenario 6's net slope
from $+0.022$ to $-0.009$. Real data has no injected parameter, so it needs an instrument that
moves turnover without being moved by returns — a change in the commission schedule, a broker
platform migration, a tax rule change — or a panel long enough for account fixed effects with
lagged turnover. The cross-sectional regression on its own cannot do it, and this is the
limitation that most directly qualifies Barber and Odean's cross-sectional result.

A fourth thing no statistic here can separate is the magnitude of a bias from the frequency of
acting on it. Scenario 6 has the same injected $\delta$ as scenario 3 and a 32 % larger measured
difference, purely because it trades more. Reporting the ratio alongside the difference is the
minimum defence; recovering the underlying hazard tilt would require modelling the holding
period explicitly, which is a duration model rather than a proportion.

---
""")

# =========================================================================== 10
md(r"""
<a id="s10"></a>

## 10. Conclusions

The simulator recovers what it is given. The null returns a disposition difference of 0.00085
against a true value of zero, the recovered difference rises monotonically through 0.0266 and
0.0749 as $\delta$ goes from 0.3 to 0.8, each estimator stays silent when only the other's
parameter is active, and the turnover slope on net returns comes back at $-0.0078$ and
$-0.0065$ against a round-trip cost schedule of $-0.005$ written into the code. That last
number is the clearest statement of what the overconfidence regression actually measures: not a
psychological parameter, but the commission and spread the account paid.

The failures are more instructive than the recoveries. Two scenarios with no injected bias at
all return disposition ratios of 1.83 and 1.41, both inside the range reported for real
brokerage accounts and one of them larger than what a genuine $\delta$ of 0.3 produces. A
mechanical rebalancing rule and a belief in mean reversion are, on the headline statistic,
indistinguishable from loss aversion — and since the estimator is correct in the sense that it
computes exactly what it claims to compute, no amount of care in implementing it would have
revealed the problem. Only the ground truth does.

Three methodological consequences follow, and they are the ones we would carry to real data.
First, the difference $\widehat{PGR}-\widehat{PLR}$ is not a scale-free measure of a bias,
since scenario 6 posts a 32 % larger difference than scenario 3 on an identical injected
$\delta$ purely by trading more; the ratio must be reported alongside it, and neither is the
parameter. Second, clustering has to be argued rather than chosen: the account bootstrap
inflates standard errors by up to 4.6 times relative to the naive position-day version, and yet
is still a lower bound, because within a single price panel the null's $t$ reached 1.95 while
the effect across eight redrawn panels was 0.00022. Third, wherever a selling rule conditions on
the purchase price, turnover is an outcome and not a regressor, and the sign of the behavioural
claim depends on which of the two is on the right-hand side.

The exercise also produced one error worth naming, since the assignment asks for the diagnosis
rather than a clean result. Our first price specification gave each security a CAPM-like drift,
which makes expected log returns vary by three percentage points a year across the
cross-section and correlate $-0.90$ with idiosyncratic volatility; a rebalancing rule that buys
laggards then acquires a drift differential it had no information about, which is
observationally identical to leakage. Imposing a common expected log return removed it at the
source. The leakage placebo that caught it had to be fixed too, because counting the same
purchase once per account produced $t=-7.5$ on a panel with no predictability in it — the same
pseudo-replication failure the clustering discussion is about, appearing in the diagnostic
rather than in the estimator.

What we would want from real data, in order, is the size of each sale relative to the position,
the trailing return of the security alongside its purchase price, and an instrument for
turnover. The first separates rebalancing from a reference point, the second separates a belief
about prices from a reference point, and the third is the only thing that makes the
overconfidence coefficient mean what it is usually taken to mean.

---
""")

md(r"""
<a id="s11"></a>

## 11. References

1. Barber, B. M., & Odean, T. (2000). Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors. *The Journal of Finance*, 55(2), 773–806.
2. Barberis, N., & Xiong, W. (2009). What Drives the Disposition Effect? An Analysis of a Long-Standing Preference-Based Explanation. *The Journal of Finance*, 64(2), 751–784.
3. Grinblatt, M., & Han, B. (2005). Prospect Theory, Mental Accounting, and Momentum. *Journal of Financial Economics*, 78(2), 311–339.
4. Odean, T. (1998). Are Investors Reluctant to Realize Their Losses? *The Journal of Finance*, 53(5), 1775–1798.
5. Odean, T. (1999). Do Investors Trade Too Much? *The American Economic Review*, 89(5), 1279–1298.

---
""")

md(r"""
<a id="s12"></a>

## 12. Reproducibility

The simulator lives in `src/bfsim/` as six modules: `config.py` holds every parameter as a
frozen dataclass and the seed policy, `prices.py` the factor model and the leakage placebo,
`agents.py` the population draws, `engine.py` the daily loop, `estimators.py` the two
estimators and their bootstraps, and `scenarios.py` the eight configurations and the tables
below. All randomness descends from a single master seed through named, independently spawned
streams, so re-running this notebook from a clean kernel reproduces every number in it exactly.
Every table is written to `results/` and every figure to `results/figures/`.
""")

code(r"""
import platform, statsmodels, matplotlib, nbformat
print(f"python      {platform.python_version()}")
for m in (np, pd, statsmodels, matplotlib):
    print(f"{m.__name__:<12}{m.__version__}")
print(f"master seed {MASTER_SEED}")
print(f"scenarios   {len(SCENARIOS)}   traders/scenario {SCENARIOS[0].n_traders}   "
      f"securities {price_cfg.n_securities}   days {price_cfg.n_days}")
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.5"},
}
path = pathlib.Path("notebooks/P01_behavioral_finance.ipynb")
nbf.write(nb, str(path))
print(f"wrote {path} with {len(C)} cells")
