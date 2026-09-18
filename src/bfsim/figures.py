"""Figures, in the Phoenix Industries palette.

The two brand accents carry the data: Electric Blue for the first series and
Lightning Gold for the second.  Scarlet is reserved for alerts, which here means
a scenario whose finding is spurious, and Dark Gray draws every reference line
(zero, a theoretical value, a fitted slope).  Figures sit on a white page because
the report is read on paper as often as on screen, so the neutral ramp supplies
the ink, the muted labels and the hairline grid.
"""

import matplotlib.pyplot as plt
import numpy as np

# Brand palette (references/brand/brand.md)
BLUE = "#00B7FF"      # Electric Blue: first series
GOLD = "#FFB703"      # Lightning Gold: second series
SCARLET = "#FF2020"   # Scarlet: alert, a spurious finding
DARK = "#1A1B21"      # Dark Gray: reference lines, error bars, edges

# Applied neutrals for a light page (references/brand/house-style.md, section 2)
INK = "#14161A"
MUTED = "#5B6472"
RULE = "#D8DCE6"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "text.color": INK,
    "errorbar.capsize": 2.5,
})

CONFOUNDED = {7, 8}


def _save(fig, path):
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def _scenario_ticks(ax, table):
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels([f"{k}. {s}" for k, s in zip(table.index, table["Scenario"])],
                       rotation=20, ha="right", fontsize=8.5)
    for tick, k in zip(ax.get_xticklabels(), table.index):
        tick.set_color(SCARLET if k in CONFOUNDED else INK)


def pgr_plr_bars(table, path=None):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(table))
    w = 0.38
    err = dict(lw=0.9, ecolor=DARK)
    ax.bar(x - w / 2, table["PGR"], w, yerr=table["se_PGR"], color=GOLD,
           label="PGR, share of gains sold", error_kw=err)
    ax.bar(x + w / 2, table["PLR"], w, yerr=table["se_PLR"], color=BLUE,
           label="PLR, share of losses sold", error_kw=err)
    for i, (k, r) in enumerate(table.iterrows()):
        ax.text(i, max(r["PGR"], r["PLR"]) + 0.006, f"{r['PGR/PLR']:.2f}x",
                ha="center", fontsize=8.5,
                color=SCARLET if k in CONFOUNDED else INK,
                fontweight="bold" if k in CONFOUNDED else "normal")
    ax.set_ylim(0, table[["PGR", "PLR"]].to_numpy().max() * 1.15)
    _scenario_ticks(ax, table)
    ax.set_ylabel("proportion realised, per sale day")
    ax.set_title("Realisation rates by domain, with the PGR/PLR ratio above each pair\n"
                 "Scenarios in scarlet had no disposition injected")
    ax.legend(fontsize=9, loc="upper left")
    return _save(fig, path)


def monotonicity(table, path=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    clean = table[~table.index.isin(CONFOUNDED)]
    conf = table[table.index.isin(CONFOUNDED)]
    ax.axhline(0, color=DARK, lw=1.1, ls="--", label="no disposition effect")
    ax.errorbar(clean["delta_inj"], clean["PGR-PLR"], yerr=2 * clean["se_diff"],
                fmt="o", color=BLUE, ecolor=DARK, ms=8, lw=1.0, label="no confound")
    ax.errorbar(conf["delta_inj"], conf["PGR-PLR"], yerr=2 * conf["se_diff"],
                fmt="D", color=SCARLET, ecolor=DARK, ms=7, lw=1.0,
                label=r"confound active, injected $\delta=0$")
    offsets = {1: (8, 6), 4: (8, -4), 5: (8, -13)}
    for k, r in table.iterrows():
        ax.annotate(str(k), (r["delta_inj"], r["PGR-PLR"]), textcoords="offset points",
                    xytext=offsets.get(k, (8, -3)), fontsize=8.5,
                    color=SCARLET if k in CONFOUNDED else INK)
    ax.set_xlabel(r"injected $\delta$ (population mean)")
    ax.set_ylabel(r"$\widehat{PGR}-\widehat{PLR}$")
    ax.set_title("Recovery is monotone in $\\delta$, and two scenarios\n"
                 "reach the same place with $\\delta=0$")
    ax.legend(fontsize=8.5, loc="upper left")
    return _save(fig, path)


def slopes(table, cost_rate, path=None):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(table))
    w = 0.38
    err = dict(lw=0.9, ecolor=DARK)
    ax.bar(x - w / 2, table["beta_gross"], w, yerr=2 * table["se_gross"], color=BLUE,
           label="gross returns", error_kw=err)
    ax.bar(x + w / 2, table["beta_net"], w, yerr=2 * table["se_net"], color=GOLD,
           label="net returns", error_kw=err)
    ax.axhline(0, color=DARK, lw=0.9)
    ax.axhline(-cost_rate, color=DARK, lw=1.1, ls="--",
               label=f"round-trip cost per unit of turnover, $-${cost_rate:.4f}")
    _scenario_ticks(ax, table)
    ax.set_ylabel(r"$\beta$ on turnover")
    ax.set_title("Turnover slope on gross and net returns, with two HC1 standard errors")
    ax.legend(fontsize=9)
    return _save(fig, path)


def turnover_vs_parameters(df_disp, df_churn, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (df, key, lab, col) in zip(axes, [
            (df_disp, "delta", r"injected $\delta$ (scenario 6)", BLUE),
            (df_churn, "kappa", r"injected $\kappa$ (scenario 5)", GOLD)]):
        ax.scatter(df[key], df["turnover"], s=8, alpha=0.45, color=col, edgecolors="none")
        b = np.polyfit(df[key], df["turnover"], 1)
        xs = np.linspace(df[key].min(), df[key].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=DARK, lw=1.8)
        r = np.corrcoef(df[key], df["turnover"])[0, 1]
        ax.set_xlabel(lab)
        ax.set_title(f"corr = {r:+.3f},  slope = {b[0]:+.2f}", fontweight="normal")
    axes[0].set_ylabel("realised annual turnover")
    fig.suptitle("The injected parameter is not the behaviour it produces", y=1.02,
                 fontsize=11, fontweight="bold", color=INK)
    return _save(fig, path)


def cash_drag(lags, betas_gross, betas_net, path=None):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(lags))
    w = 0.38
    ax.bar(x - w / 2, betas_gross, w, color=BLUE, label="gross")
    ax.bar(x + w / 2, betas_net, w, color=GOLD, label="net")
    ax.axhline(0, color=DARK, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}-day settlement" for l in lags])
    ax.set_ylabel(r"$\beta$ on turnover")
    ax.set_title("Cash drag, switched on deliberately")
    ax.legend(fontsize=9)
    return _save(fig, path)


def bootstrap_density(draws_by_scenario, path=None):
    styles = {1: (DARK, "-"), 3: (BLUE, "-"), 7: (SCARLET, "-"), 8: (SCARLET, "--")}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvline(0, color=DARK, lw=1.0, ls=":")
    for k, (label, draws, _) in draws_by_scenario.items():
        col, ls = styles.get(k, (GOLD, "-"))
        ax.hist(draws, bins=60, histtype="step", lw=1.6, color=col, ls=ls, label=label)
    ax.set_xlabel(r"$\widehat{PGR}-\widehat{PLR}$, bootstrap draws")
    ax.set_ylabel("frequency")
    ax.set_title("Account-clustered bootstrap distributions")
    ax.legend(fontsize=8.5)
    return _save(fig, path)
