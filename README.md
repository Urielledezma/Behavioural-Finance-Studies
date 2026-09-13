# P01 — Behavioural Finance

A simulator that generates synthetic brokerage records, injects a disposition mechanism
and an overprecision mechanism at magnitudes fixed in advance, and then tries to recover
those magnitudes with the estimators used on real data (Odean 1998; Barber & Odean 2000).

The report is `notebooks/P01_behavioral_finance.ipynb`. Everything it claims is computed
by the notebook itself; nothing is pasted in.

## Headline result

Two of the eight scenarios inject no bias at all and still return disposition ratios of
**1.83** and **1.41**, both inside the range reported for real brokerage accounts, and one
of them larger than the 1.74 produced by a genuine injected `delta = 0.3`. A mechanical
rebalancing rule and a belief in mean reversion are indistinguishable from loss aversion on
the headline statistic. The estimators are not broken — they compute exactly what they
claim to compute — which is the point of having ground truth.

## Layout

```
src/bfsim/
  config.py       every parameter as a frozen dataclass, plus the seed policy
  prices.py       three-factor price panel and the leakage placebo
  agents.py       trader population draws
  engine.py       the vectorised daily trading loop
  estimators.py   PGR/PLR, bootstraps, turnover regressions
  scenarios.py    the eight configurations and the result tables
  figures.py      plots
notebooks/        the written report
results/          tables (CSV) and figures (PNG), regenerated on every run
PRE_ANALYSIS.md   registered before any estimator was run
build_notebook.py assembles the report notebook from source
run_analysis.py   command-line driver; writes every table without the notebook
```

## Reproducing

```bash
pip install -r requirements.txt

python run_analysis.py                 # every table into results/, ~3 minutes
python build_notebook.py               # rebuild the report from source
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/P01_behavioral_finance.ipynb
```

All randomness descends from a single master seed (`config.MASTER_SEED = 20260913`)
through named, independently spawned generator streams — one each for prices, `delta`,
`kappa`, wealth, position counts, the sell hazard, security selection and the bootstrap.
Re-running from a clean interpreter reproduces every number exactly. `delta` and `kappa`
draw from separate streams, so their independence is a property of the construction rather
than of an inspection afterwards.

## Design decisions a reader should know about

**The scenario table's `delta` and `kappa` are population means, not constants.** Each
account draws from a Beta distribution with that mean. Without cross-sectional dispersion
there is no `corr(delta, kappa)` to report and the turnover regression loses most of its
identification.

**Every security carries the same expected log return.** Under a CAPM-like drift, expected
log return varies by three percentage points a year across the cross-section and correlates
−0.90 with idiosyncratic volatility, so any rule selecting on past returns acquires a drift
differential it had no information about — observationally identical to leakage. This was
found and fixed during construction; the episode is documented in the report.

**Gross return is a shadow portfolio, not a subtraction.** Each account runs a parallel
portfolio taking identical decisions but filling at the mid with no commission. Gross is
therefore the counterfactual return of the same behaviour in a frictionless market, which
keeps the spread out of the gross measure by definition rather than by assertion.

**The leakage placebo de-duplicates and clusters by day.** A thousand accounts buying the
same security on the same day is one decision observed a thousand times; counting them
separately produced *t* = −7.5 on a panel with no predictability in it.

## Estimator conventions

Only days on which the account sold something contribute to any count. A sale is counted at
position level, with the share-weighted alternative computed alongside. The cost basis is
per-lot, with average cost computed alongside. Positions exactly at their purchase price
enter no count (realised frequency: zero). Standard errors resample **accounts**, never
transactions; the naive position-day bootstrap is reported only to show that it understates
the standard error by up to a factor of 4.6.
