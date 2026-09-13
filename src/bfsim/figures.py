"""Figures.

One dark anchor colour and one accent carry every field; gold marks an injected
or theoretical value and red marks a scenario whose finding is spurious.  Colour
is therefore information, never decoration.
"""

import matplotlib.pyplot as plt
import numpy as np

ANCHOR = "#1f3251"   # fields: realised gains, gross returns
ACCENT = "#6f9bb0"   # fields: realised losses, net returns
GOLD = "#c9992e"     # events: injected or theoretical value
RED = "#a4303f"      # events: a finding that is spurious
GREY = "#9aa3ad"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10,
})

CONFOUNDED = {7, 8}


def _save(fig, path):
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def pgr_plr_bars(table, path=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2, table["PGR"], w, yerr=table["se_PGR"], color=ANCHOR,
           label="PGR (gains realised)", capsize=2.5, error_kw=dict(lw=0.9))
    ax.bar(x + w / 2, table["PLR"], w, yerr=table["se_PLR"], color=ACCENT,
           label="PLR (losses realised)", capsize=2.5, error_kw=dict(lw=0.9))
    for i, (k, r) in enumerate(table.iterrows()):
        col = RED if k in CONFOUNDED else "#33383d"
        ax.text(i, max(r["PGR"], r["PLR"]) + 0.006,
                f"{r['PGR/PLR']:.2f}x", ha="center", fontsize=8.5, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}. {s}" for k, s in zip(table.index, table["Scenario"])],
                       rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("proportion realised, per sale day")
    ax.set_title("Realisation rates by domain, with the ratio above each pair\n"
                 "Red marks a scenario in which no disposition was injected")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def monotonicity(table, path=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    clean = table[~table.index.isin(CONFOUNDED)]
    conf = table[table.index.isin(CONFOUNDED)]
    ax.errorbar(clean["delta_inj"], clean["PGR-PLR"], yerr=2 * clean["se_diff"],
                fmt="o", color=ANCHOR, ms=7, capsize=3, lw=1.1, label="no confound")
    ax.errorbar(conf["delta_inj"], conf["PGR-PLR"], yerr=2 * conf["se_diff"],
                fmt="D", color=RED, ms=7, capsize=3, lw=1.1,
                label="confound active, injected $\\delta=0$")
    for k, r in table.iterrows():
        ax.annotate(str(k), (r["delta_inj"], r["PGR-PLR"]),
                    textcoords="offset points", xytext=(8, -3), fontsize=8.5,
                    color=RED if k in CONFOUNDED else "#33383d")
    ax.axhline(0, color=GOLD, lw=1.2, ls="--", label="no disposition effect")
    ax.set_xlabel("injected $\\delta$ (population mean)")
    ax.set_ylabel("$\\widehat{PGR}-\\widehat{PLR}$")
    ax.set_title("Recovery is monotone in $\\delta$, and two scenarios\nreach the same place with $\\delta=0$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    return _save(fig, path)


def slopes(table, cost_rate, path=None):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2, table["beta_gross"], w, yerr=2 * table["se_gross"], color=ANCHOR,
           label="gross returns", capsize=2.5, error_kw=dict(lw=0.9))
    ax.bar(x + w / 2, table["beta_net"], w, yerr=2 * table["se_net"], color=ACCENT,
           label="net returns", capsize=2.5, error_kw=dict(lw=0.9))
    ax.axhline(0, color="#33383d", lw=0.9)
    ax.axhline(-cost_rate, color=GOLD, lw=1.2, ls="--",
               label=f"round-trip cost rate, $-${cost_rate:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}. {s}" for k, s in zip(table.index, table["Scenario"])],
                       rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel(r"$\beta$ on turnover")
    ax.set_title("Turnover slope, gross and net. Bars are two HC1 standard errors")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def turnover_vs_parameters(df_disp, df_churn, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (df, key, lab, col) in zip(axes, [
            (df_disp, "delta", r"injected $\delta$", ANCHOR),
            (df_churn, "kappa", r"injected $\kappa$", ACCENT)]):
        ax.scatter(df[key], df["turnover"], s=7, alpha=0.35, color=col, edgecolors="none")
        b = np.polyfit(df[key], df["turnover"], 1)
        xs = np.linspace(df[key].min(), df[key].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=GOLD, lw=1.8)
        r = np.corrcoef(df[key], df["turnover"])[0, 1]
        ax.set_xlabel(lab)
        ax.set_title(f"corr = {r:+.3f},  slope = {b[0]:+.2f}")
    axes[0].set_ylabel("realised annual turnover")
    fig.suptitle("The injected parameter is not the behaviour it produces: "
                 "disposition suppresses turnover it never targets", y=1.02, fontsize=11)
    return _save(fig, path)


def cash_drag(lags, betas_gross, betas_net, path=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(lags))
    w = 0.38
    ax.bar(x - w / 2, betas_gross, w, color=ANCHOR, label="gross")
    ax.bar(x + w / 2, betas_net, w, color=ACCENT, label="net")
    ax.axhline(0, color="#33383d", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}-day settlement" for l in lags])
    ax.set_ylabel(r"$\beta$ on turnover")
    ax.set_title("Cash drag, switched on deliberately.\n"
                 "One idle day turns a flat gross slope negative")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, path)


def bootstrap_density(draws_by_scenario, path=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for k, (label, draws, conf) in draws_by_scenario.items():
        col = RED if conf else (ANCHOR if k <= 3 else ACCENT)
        ax.hist(draws, bins=60, histtype="step", lw=1.4, color=col, label=label)
    ax.axvline(0, color=GOLD, lw=1.2, ls="--")
    ax.set_xlabel(r"$\widehat{PGR}-\widehat{PLR}$, bootstrap draws")
    ax.set_ylabel("frequency")
    ax.set_title("Account-clustered bootstrap distributions")
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, path)
