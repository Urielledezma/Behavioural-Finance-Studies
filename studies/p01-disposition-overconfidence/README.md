# P01 — Recovering injected behavioural biases from a synthetic trading record

With real brokerage data nobody can tell whether a measured bias is genuine or the product
of a coding error, because the true behaviour of each investor is unknown, so this study
builds a simulated market in which it is known by construction. A thousand simulated
traders carry a disposition effect and a tendency to overtrade at strengths fixed in
advance, and the two standard estimators from the literature (Odean 1998; Barber and
Odean 2000) are then asked to recover them across eight scenarios. In five of the eight
at least one estimator returns a confident answer that is wrong, most notably two
scenarios with no injected bias at all whose disposition ratios of 1.83 and 1.41 are
indistinguishable from the real thing.

---

## Contents

| File | What it is |
|---|---|
| [`P01_behavioral_finance.ipynb`](P01_behavioral_finance.ipynb) | The written report, with every number computed by its own code cells |
| [`PRE_ANALYSIS.md`](PRE_ANALYSIS.md) | The pre-analysis statement, written before any estimator was run and embedded unedited in the report |
| [`build_notebook.py`](build_notebook.py) | Assembles the report notebook from source. **The notebook's source of truth** |
| [`run_analysis.py`](run_analysis.py) | Command-line driver that writes every table without opening the notebook |
| [`results/`](results) | Tables as CSV and figures as PNG, regenerated on every run |

The simulator itself is the `bfsim` package in [`src/bfsim/`](../../src/bfsim) at the
repository root, and its recovery properties are asserted in
[`tests/test_p01_recovery.py`](../../tests/test_p01_recovery.py).

---

## Running it

From this folder:

```bash
python run_analysis.py            # every table into results/, about three minutes
python run_analysis.py --quick    # smoke test with small bootstraps

python build_notebook.py          # rebuild the report from source
jupyter nbconvert --to notebook --execute --inplace P01_behavioral_finance.ipynb
```

All randomness descends from a single master seed (`config.MASTER_SEED = 20260913`)
through separate named streams for prices, each behavioural parameter, capital, portfolio
size, the selling decision, stock selection and the bootstrap, so a run from a clean
interpreter reproduces every number in the report exactly. The notebook is versioned with
its outputs included, because its interpretation cells quote concrete values that would
lose their support if the cells were cleared. The two writers of `results/` differ only in
bootstrap size, so `run_analysis.py --quick` overwrites the tables with noisier standard
errors until the notebook is executed again.

---

## Design decisions worth knowing before reading the code

The scenario table's `delta` and `kappa` are population means rather than constants, with
each trader drawing from a Beta distribution with that mean, because without differences
across traders there is no correlation to report and the turnover regression has almost no
variation to work with.

Every stock carries the same expected log return, since a CAPM-style drift makes that
quantity vary by 3 percentage points a year and correlate −0.90 with idiosyncratic
volatility, which lets any rule that selects on past returns acquire a drift advantage it
never had information about, an effect indistinguishable from leakage.

Gross return is the return of a shadow portfolio that takes the same decisions at the mid
price with no commission, rather than net return with costs added back, which keeps the
bid-ask spread out of the gross measure by construction.

The leakage test keeps each (day, stock) purchase once and computes its standard error
from daily averages, because counting the same purchase once per account produced a
t-statistic of −7.5 on a price panel with no predictability in it.
