"""Command-line driver: reproduce every table in the report without the notebook.

    python run_analysis.py [--boot 2000] [--seeds 8] [--quick]

Writes CSVs to this study's results/ and a plain-text digest to results/summary.txt.
"""

import argparse
import pathlib
import sys
import time
from dataclasses import replace

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from bfsim.config import CostConfig, EngineConfig  # noqa: E402
from bfsim.scenarios import (  # noqa: E402
    SCENARIOS, analyse, common_universe, convention_table, discrimination_table,
    independence_table, placebo_table, replicate_seeds, results_table, run_all,
)

RESULTS = HERE / "results"
SEEDS = [20260913, 7, 101, 4242, 31337, 2, 13, 55]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replications")
    ap.add_argument("--seeds", type=int, default=8, help="panels for the replication check")
    ap.add_argument("--quick", action="store_true", help="small run for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.boot, args.seeds = 200, 2

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    log, t0 = [], time.time()

    def emit(name, df):
        df.to_csv(RESULTS / f"{name}.csv")
        log.append(f"### {name}\n{df.round(5).to_string()}\n")
        print(f"### {name}")
        print(df.round(5).to_string(), end="\n\n")

    universe = common_universe()
    outputs = run_all(universe, n_boot=args.boot, verbose=True)
    print()

    table = results_table(outputs)
    emit("scenarios", table)
    emit("independence", independence_table(outputs))
    emit("conventions", convention_table(outputs))
    emit("discrimination", discrimination_table(outputs))
    emit("placebo", placebo_table(outputs))

    c = CostConfig()
    rate = 2 * (c.commission_bps + c.half_spread_bps) * 1e-4
    log.append(f"round-trip cost per unit of annual turnover: {rate:.5f}\n")

    rows = []
    for lag in (0, 1, 3, 5):
        o = analyse(replace(SCENARIOS[4], engine=replace(EngineConfig(), settlement_lag=lag)),
                    universe, n_boot=max(args.boot // 4, 100))
        rows.append({"settlement lag": lag,
                     "beta_gross": o.over["ret_gross"]["beta"], "t_gross": o.over["ret_gross"]["t"],
                     "beta_net": o.over["ret_net"]["beta"], "t_net": o.over["ret_net"]["t"],
                     "mean gross return": o.df.ret_gross.mean()})
    emit("cash_drag", pd.DataFrame(rows).set_index("settlement lag"))

    rows = []
    for rho in (0.0, 0.7, -0.7):
        o = analyse(replace(SCENARIOS[5], copula_rho=rho), universe,
                    n_boot=max(args.boot // 4, 100))
        rows.append({"copula rho": rho,
                     "realised corr(delta,kappa)": o.indep["corr_delta_kappa"],
                     "corr(delta,turnover)": o.indep["corr_delta_turnover"],
                     "beta_gross": o.over["ret_gross"]["beta"],
                     "beta_net": o.over["ret_net"]["beta"],
                     "PGR-PLR": o.disp["diff"]})
    emit("copula", pd.DataFrame(rows).set_index("copula rho"))

    for i in (0, 2, 6):
        cfg = SCENARIOS[i]
        emit(f"replication_s{cfg.index}",
             replicate_seeds(cfg, SEEDS[:args.seeds], n_boot=max(args.boot // 6, 100)))

    log.append(f"elapsed {time.time() - t0:.0f}s")
    (RESULTS / "summary.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"wrote {RESULTS} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
