"""Supplementary Fig. S2 and Supplementary Table S2: benchmark model comparison.

Reads the per-subject fits written by ``experiments/setsize_model_comparison.py``
(Item-Limit, Slots-plus-Averaging, Equal-Precision and Variable-Precision,
each fitted by maximum likelihood to every simulated subject's errors).

  * Fig. S2: uniform-von Mises mixture statistics (w, CSD) per set size.
    Points: mean +/- SEM across subjects of each subject's statistics.
    Lines: mean model prediction across subjects; band: +/- SEM across
    subjects. RMSE compares each subject's prediction with that subject's
    statistics, over all subjects and set sizes.
                                 -> figS_model_comparison.{pdf,png}
  * Table S2: per model, mean AIC / BIC difference from each subject's best
    model +/- SEM across subjects, and the number of subjects for which the
    model has the lowest AIC.    -> table_S2_information_criteria.{csv,tex}

Run:  python -m figures.figS_model_comparison
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from figures import nature_style as ns

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODELS = ["IL", "SA", "EP", "VP"]
MODEL_LABELS = {
    "IL": "Item-Limit",
    "SA": "Slots-plus-Averaging",
    "EP": "Equal-Precision",
    "VP": "Variable-Precision",
}
MODEL_COLORS = {"IL": "#CB181D", "SA": "#31A354", "EP": "#E6AB02", "VP": "#2171B5"}


def load():
    ic = pd.read_csv(ns.result_path("setsize", "model_comparison_subjects_ic.csv"))
    curves = pd.read_csv(ns.result_path("setsize", "model_comparison_subjects_curves.csv"))
    return ic, curves


def table(ic):
    ic = ic.copy()
    ic["dAIC"] = ic["AIC"] - ic.groupby("subject")["AIC"].transform("min")
    ic["dBIC"] = ic["BIC"] - ic.groupby("subject")["BIC"].transform("min")
    best = ic.loc[ic.groupby("subject")["AIC"].idxmin(), "model"].value_counts()
    n_subj = ic["subject"].nunique()
    rows = []
    for m in MODELS:
        g = ic[ic["model"] == m]
        rows.append(dict(
            Model=m, k=int(g["k"].iloc[0]), n_per_subject=int(g["n"].iloc[0]),
            dAIC_mean=g["dAIC"].mean(), dAIC_sem=g["dAIC"].sem(),
            dBIC_mean=g["dBIC"].mean(), dBIC_sem=g["dBIC"].sem(),
            best_AIC=int(best.get(m, 0)), n_subjects=n_subj,
        ))
    df = pd.DataFrame(rows).sort_values("dAIC_mean").reset_index(drop=True)
    df.to_csv(ns.out_path("table_S2_information_criteria") + ".csv", index=False)

    lines = [
        r"\begin{tabular}{lcrrc}",
        r"\toprule",
        r"\textbf{Model} & $k$ & $\Delta$\textbf{AIC} & $\Delta$\textbf{BIC} & "
        r"\textbf{Best (AIC)} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r.Model} & {r.k} & ${r.dAIC_mean:.1f} \\pm {r.dAIC_sem:.1f}$ & "
            f"${r.dBIC_mean:.1f} \\pm {r.dBIC_sem:.1f}$ & {r.best_AIC}/{r.n_subjects} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(ns.out_path("table_S2_information_criteria") + ".tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(df.to_string(index=False))
    return df


def group_curves(curves):
    """Mean and SEM across subjects of w and CSD, per model (and empirical) and N."""
    return curves.groupby(["model", "N"])[["w", "csd"]].agg(["mean", "sem"])


def rmse(curves, model, stat):
    """RMSE of the model's prediction against each subject's own statistic,
    over all subjects and set sizes (van den Berg et al. 2012, Fig. 4)."""
    emp = curves[curves["model"] == "empirical"].set_index(["subject", "N"])[stat]
    pred = curves[curves["model"] == model].set_index(["subject", "N"])[stat]
    return float(np.sqrt(np.mean((pred - emp.loc[pred.index]) ** 2)))


def main():
    ic, curves = load()
    table(ic)
    g = group_curves(curves)
    Ns = sorted(int(n) for n in curves["N"].unique())
    xs = np.asarray(Ns, float)

    def series(model, stat):
        return (np.array([g.loc[(model, N), (stat, "mean")] for N in Ns]),
                np.array([g.loc[(model, N), (stat, "sem")] for N in Ns]))

    w_emp, w_sem = series("empirical", "w")
    c_emp, c_sem = series("empirical", "csd")
    rms_w = {m: rmse(curves, m, "w") for m in MODELS}
    rms_c = {m: rmse(curves, m, "csd") for m in MODELS}
    best_w = min(MODELS, key=rms_w.get)
    best_c = min(MODELS, key=rms_c.get)
    for m in MODELS:
        print(f"  {m}: RMSE_w={rms_w[m]:.3f} RMSE_CSD={rms_c[m]:.3f}")

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True)
    for col, m in enumerate(MODELS):
        c = MODEL_COLORS[m]
        for row, (stat, emp, sem, rms, best) in enumerate((
                ("w", w_emp, w_sem, rms_w, best_w),
                ("csd", c_emp, c_sem, rms_c, best_c))):
            ax = axes[row, col]
            pred, pred_sem = series(m, stat)
            ax.fill_between(xs, pred - pred_sem, pred + pred_sem, color=c,
                            alpha=0.25, linewidth=0)
            ax.plot(xs, pred, color=c, linewidth=1.5)
            ax.errorbar(xs, emp, yerr=sem, fmt="s", color="black", capsize=2,
                        markersize=3.5, markerfacecolor="white",
                        markeredgewidth=0.8, linewidth=0.8, zorder=5)
            star = r" $\bigstar$" if m == best else ""
            ax.text(0.95, 0.06, f"RMSE\n{rms[m]:.3f}{star}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7)
            ax.set_xticks(Ns)
            ax.set_ylim(0, 1.08) if stat == "w" else ax.set_ylim(0, None)
        axes[0, col].set_title(MODEL_LABELS[m], color=c, fontweight="bold", fontsize=8)
        axes[1, col].set_xlabel("Set size")
    axes[0, 0].set_ylabel("$w$")
    axes[1, 0].set_ylabel("CSD (rad)")

    fig.tight_layout(w_pad=1.0, h_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(ns.out_path("figS_model_comparison") + f".{ext}")
    plt.close(fig)
    print(f"saved figS_model_comparison.pdf / .png in {os.path.dirname(ns.out_path('x'))}")


if __name__ == "__main__":
    main()
