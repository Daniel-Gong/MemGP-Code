"""
Figure 2 - Set-size effects.

a   continuous-reproduction task paradigm (array -> delay -> probe wheel)
b-e HUMAN signatures of load from published behavioural data:
    b,c  Ma, Husain & Bays (2014) Fig. 1b–c (replot of Bays et al. 2009):
         target-relative error distributions; recall circular SD vs set size
    d    Bays, Catalao & Husain (2009) Fig. 1b: precision (rad^-1) power law
    e    Bays et al. (2009) Fig. 2b, N=4: non-target-relative report distribution
f-i MemGP reproduces all four signatures (real simulation outputs)
j   hyper-parameter: representational capacity (# inducing points)
k   hyper-parameter: forgetting weight beta
l   hyper-parameter: kernel lengthscale -- error vs set size for several
    lengthscales; each shifts the error level but preserves the set-size gradient
m   swap errors are spatially graded and load-dependent: heatmap of misbinding
    rate (probe reported as a non-target) over set size x target-distractor dist.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from figures import nature_style as ns

ns.set_style(9)

SEQ = ns.SETSIZE_SEQ  # blue gradient for N = 1,2,4,6,8
_HUMAN_NPZ = ns.data_path("bays2009", "bays2009_fig2_human.npz")


def circ_err(pred, target):
    return (np.asarray(pred) - np.asarray(target) + 180.0) % 360.0 - 180.0


# ===========================================================================
# Panel a : paradigm
# ===========================================================================
def screen_inset(parent, rect):
    iax = parent.inset_axes(rect)
    iax.set_xlim(0, 1); iax.set_ylim(0, 1); iax.set_aspect("equal")
    iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_visible(False)
    return iax


def draw_screen(iax, fc="white", margin=0.06):
    iax.add_patch(Rectangle((margin, margin), 1 - 2 * margin, 1 - 2 * margin,
                            fc=fc, ec=ns.INK, lw=1.4))
    iax.plot([0.47, 0.53], [0.5, 0.5], color=ns.INK, lw=1.0)
    iax.plot([0.5, 0.5], [0.47, 0.53], color=ns.INK, lw=1.0)


def place_items(iax, items, r=0.26, sq=0.1, outline=False, only=None):
    cx, cy = 0.5, 0.5
    for i, (l, c) in enumerate(items):
        a = np.deg2rad(l)
        px, py = cx + r * np.cos(a), cy + r * np.sin(a)
        if outline or (only is not None and i != only):
            iax.add_patch(Rectangle((px - sq/2, py - sq/2), sq, sq, fc="white",
                                    ec=ns.INK, lw=1.1))
        else:
            iax.add_patch(Rectangle((px - sq/2, py - sq/2), sq, sq,
                                    fc=ns.color_to_rgb(c), ec=ns.INK, lw=1.0))


def panel_a(ax, gap_in=0.60):
    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    items = [(-130, -120), (-35, 70), (60, 150), (150, -40)]

    # Screens keep a 1:1 aspect, so size their slots square in display units
    # (rather than leaving matplotlib to centre a square inside a wide slot)
    # and centre the row of three.
    fig = ax.figure
    pos = ax.get_position()
    w_in, h_in = pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()

    y0, h = 0.17, 0.82
    w = h * h_in / w_in
    gap = gap_in / w_in
    xs = [(1.0 - (3 * w + 2 * gap)) / 2 + i * (w + gap) for i in range(3)]
    y_mid = y0 + h / 2
    labels = ["Array", "Delay", "Probe"]
    for x, lab in zip(xs, labels):
        iax = screen_inset(ax, [x, y0, w, h])
        draw_screen(iax, margin=0.03)
        if lab == "Array":
            place_items(iax, items)
        elif lab == "Probe":
            # color wheel + empty squares + cued location cursor
            ns.color_wheel(iax, (0.5, 0.5), r_out=0.43, r_in=0.375)
            place_items(iax, items, outline=True, sq=0.085)
            a = np.deg2rad(items[0][0])
            sq = 0.085
            px, py = 0.5 + 0.26*np.cos(a), 0.5 + 0.26*np.sin(a)
            iax.add_patch(Rectangle((px-sq/2, py-sq/2), sq, sq, fc="white",
                                    ec=ns.CUE, lw=1.6, zorder=6))
            iax.annotate("", xy=(px+0.03, py-0.02), xytext=(px+0.16, py-0.14),
                         arrowprops=dict(arrowstyle="-|>", color=ns.INK, lw=1.3))
        ax.text(x + w/2, y0 - 0.04, lab, ha="center", va="top", fontsize=8.5,
                color=ns.INK)
    arrow_pad = 0.004
    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - arrow_pad, y_mid),
            xytext=(xs[i] + w + arrow_pad, y_mid),
            arrowprops=dict(
                arrowstyle="-|>", color=ns.INK, lw=1.7,
                mutation_scale=12, shrinkA=0, shrinkB=0,
            ),
            zorder=10,
        )
    # Figure 2 is scaled to 0.78\textheight in the manuscript, so 10.6 pt
    # here is 8 pt on the page, the Figure 3 panel title size.
    ax.text(0.0, 1.02, "Continuous reproduction task", ha="left", va="bottom",
            fontsize=10.6, fontweight="bold", color=ns.INK)


# ===========================================================================
# shared small-panel plotters
# ===========================================================================
def plot_error_dists(ax, errors_by_N, Ns, colors=None, title_color=ns.INK):
    x = np.linspace(-180, 180, 400)
    palette = SEQ if colors is None else colors
    for i, N in enumerate(Ns):
        e = np.asarray(errors_by_N[N])
        e = e[np.abs(e) <= 180]
        kde = stats.gaussian_kde(e, bw_method=0.25)
        ax.plot(x, kde(x), color=palette[i], lw=1.8, label=f"{N}")
    ax.axvline(0, color=ns.GREY, lw=0.7, ls="--")
    ax.set_xlim(-180, 180); ax.set_xticks([-180, 0, 180])
    ax.set_xlabel("Recall error (deg)")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    leg = ax.legend(title="Set size", fontsize=6.6, title_fontsize=6.8,
                    loc="upper right", handlelength=1.0, labelspacing=0.2,
                    borderpad=0.2)
    leg._legend_box.align = "left"


def plot_mae(ax, Ns, mae, color, marker="o", yerr=None, ylabel="Error (deg)"):
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        ax.errorbar(Ns, mae, yerr=yerr, fmt=f"{marker}-", color=color, ms=4, lw=1.8,
                    markeredgecolor="white", markeredgewidth=0.6, capsize=2.5,
                    elinewidth=1.0, zorder=4)
        ymax = float(np.max(np.asarray(mae) + yerr))
    else:
        ax.plot(Ns, mae, marker=marker, color=color, ms=4, lw=1.8,
                markeredgecolor="white", markeredgewidth=0.6)
        ymax = float(np.max(mae))
    ax.set_xlabel("Set size $N$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(Ns)
    ax.set_ylim(0, ymax * 1.2)


def plot_confidence(ax, err, conf):
    """Split errors by confidence (low vs high posterior variance)."""
    med = np.median(conf)
    lo = err[conf <= med]   # low variance = high confidence
    hi = err[conf > med]    # high variance = low confidence
    x = np.linspace(-180, 180, 400)
    for e, c, lab in [(lo, ns.MEM, "high conf."), (hi, ns.CUE, "low conf.")]:
        e = e[np.abs(e) <= 180]
        kde = stats.gaussian_kde(e, bw_method=0.28)
        ax.plot(x, kde(x), color=c, lw=1.9, label=f"{lab}\n(CSD {ns.circ_sd_deg(e):.0f}\u00b0)")
        ax.fill_between(x, kde(x), color=c, alpha=0.10)
    ax.axvline(0, color=ns.GREY, lw=0.7, ls="--")
    ax.set_xlim(-180, 180); ax.set_xticks([-180, 0, 180])
    ax.set_xlabel("Recall error (deg)"); ax.set_ylabel("Density"); ax.set_yticks([])
    ax.legend(fontsize=6.3, loc="upper right", handlelength=1.0,
              labelspacing=0.5, borderpad=0.2)


def plot_swap(ax, nt_err, color=ns.SLOT):
    edges = np.arange(-180, 181, 12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hn, _ = np.histogram(nt_err, bins=edges, density=True)
    ax.bar(centers, hn, width=11, color=color, alpha=0.85,
           label="vs non-target")
    ax.axhline(1/360, color=ns.GREY, lw=0.9, ls=":", label="chance")
    ax.axvline(0, color=ns.GREY, lw=0.7, ls="--")
    ax.set_xlim(-180, 180); ax.set_xticks([-180, 0, 180])
    ax.set_xlabel("Response \u2212 non-target (deg)"); ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.legend(fontsize=6.2, loc="upper left", handlelength=1.0,
              labelspacing=0.25, borderpad=0.2)


# ===========================================================================
# HUMAN behavioural data (Bays et al. 2009 OSF; Ma et al. 2014 Fig. 1b–c)
# ===========================================================================
def load_human():
    """Load processed Bays et al. (2009) trial statistics.

    Ma, Husain & Bays (2014) Fig. 1b–c replot this dataset (their caption:
    panel c adapted from Bays et al. 2009). Rebuild with
    ``data/bays2009/build_cache.py`` if the NPZ is missing.
    Raw trials live in ``data/bays2009/Exp_data.mat`` (OSF c2yx5).
    """
    if not os.path.isfile(_HUMAN_NPZ):
        raise FileNotFoundError(
            f"Missing {_HUMAN_NPZ}. Run data/bays2009/build_cache.py"
        )
    return ns.load_npz(_HUMAN_NPZ)


def human_setsize_errors():
    """Target-relative recall errors (deg) by set size — Ma et al. 2014 Fig. 1b."""
    d = load_human()
    Ns = [int(n) for n in d["Ns"]]
    errs = {int(N): np.asarray(d[f"err_deg_N{int(N)}"], float) for N in Ns}
    return Ns, errs


def human_recall_sd():
    """Recall circular SD (deg) vs set size — Ma et al. 2014 Fig. 1c."""
    d = load_human()
    return (np.asarray(d["Ns"], int),
            np.asarray(d["recall_sd_deg_mean"], float),
            np.asarray(d["recall_sd_deg_sem"], float))


def human_precision():
    """Bays precision (rad^-1; 1/circSD − chance) — Bays et al. 2009 Fig. 1b."""
    d = load_human()
    return (np.asarray(d["Ns"], int),
            np.asarray(d["precision_mean"], float),
            np.asarray(d["precision_sem"], float))


def human_nontarget_N4():
    """Non-target-relative errors at N=4 — Bays et al. 2009 Fig. 2b."""
    d = load_human()
    return np.asarray(d["nt_err_deg_N4"], float)


# ===========================================================================
# MODEL data loaders
# ===========================================================================
SD_CHANCE = np.pi / np.sqrt(3.0)   # circular SD of uniform responses (Bays et al. 2009)


def _main_trials():
    """Every display item; only the probed row has a report."""
    return pd.read_csv(ns.result_path("setsize", "setsize_main_trials.csv"))


def _probed(df):
    return df[df["probed"]]


def model_setsize_errors():
    """Pooled signed errors (deg) per set size, all subjects."""
    df = _probed(_main_trials())
    Ns = sorted(int(n) for n in df["N"].unique())
    return Ns, {N: df.loc[df["N"] == N, "error"].to_numpy(float) for N in Ns}


def _circ_sd_rad(errs_deg):
    return np.deg2rad(ns.circ_sd_deg(errs_deg))


def model_subject_stats():
    """Per subject and set size: circular SD (deg) and Bays precision (rad^-1).

    Returns (Ns, sd_mean, sd_sem, prec_mean, prec_sem) across subjects.
    """
    df = _probed(_main_trials())
    g = df.groupby(["subject", "N"])["error"]
    sd = g.apply(lambda e: ns.circ_sd_deg(e.to_numpy(float))).unstack("N")
    prec = g.apply(lambda e: 1.0 / _circ_sd_rad(e.to_numpy(float)) - 1.0 / SD_CHANCE).unstack("N")
    Ns = [int(n) for n in sd.columns]

    def mean_sem(t):
        return t.mean(axis=0).to_numpy(), t.sem(axis=0, ddof=1).to_numpy()

    return (Ns, *mean_sem(sd), *mean_sem(prec))


def model_swap(N=4):
    """Probed report relative to each non-target color (deg) at set size N, pooled."""
    df = _main_trials()
    df = df[df["N"] == N]
    nt = []
    for _, g in df.groupby(["subject", "trial"]):
        report = float(g.loc[g["probed"], "report"].iloc[0])
        nt.extend(circ_err(report, c) for c in g.loc[~g["probed"], "color"])
    return np.asarray(nt, float)


def sweep_mae(design):
    """Mean absolute error per (level, N): mean and SEM across subjects."""
    df = _probed(pd.read_csv(ns.result_path("setsize", f"setsize_{design}_trials.csv")))
    subj = (df.assign(abs_err=df["error"].abs())
              .groupby(["level", "N", "subject"])["abs_err"].mean())
    stat = subj.groupby(["level", "N"]).agg(["mean", "sem"])
    levels = sorted(df["level"].unique())
    Ns = sorted(int(n) for n in df["N"].unique())
    mae = np.array([[stat.loc[(lv, N), "mean"] for N in Ns] for lv in levels])
    sem = np.array([[stat.loc[(lv, N), "sem"] for N in Ns] for lv in levels])
    return levels, Ns, mae, sem


# ===========================================================================
# Representational-dynamics helpers
# ===========================================================================


# ===========================================================================
# Precision power-law panels (d = human, h = MemGP)
# ===========================================================================
def _powerlaw_precision(ax, N, prec, color, marker, yerr=None):
    """Plot precision (rad^-1) vs set size with a fitted power law
    precision = c * N^(-beta)."""
    N = np.asarray(N, float); prec = np.asarray(prec, float)
    slope, logc = np.polyfit(np.log(N), np.log(prec), 1)
    xf = np.linspace(N.min(), N.max(), 200)
    ax.plot(xf, np.exp(logc) * xf ** slope, color=color, lw=1.9, zorder=2)
    ax.errorbar(N, prec, yerr=yerr, fmt=marker, color=color, ms=4,
                mfc="white", mec=color, mew=1.4, lw=0, elinewidth=1.2,
                capsize=2.5, zorder=3)
    ax.set_xlabel("Number of items")
    ax.set_ylabel(r"Precision (rad$^{-1}$)")
    ax.set_xticks([int(n) for n in N])
    ax.set_ylim(bottom=0)


def plot_precision_human(ax):
    """Bays, Catalao & Husain (2009) Fig. 1b: recall precision (rad^-1)."""
    N, prec, yerr = human_precision()
    _powerlaw_precision(ax, N, prec, ns.PLUM, "s", yerr=yerr)


def plot_precision_model(ax, Ns, prec, sem):
    """MemGP precision (1/circular SD - 1/SD_chance, rad^-1), mean +/- SEM
    across simulated subjects, with a power law fitted to the means."""
    _powerlaw_precision(ax, Ns, prec, ns.MEM, "o", yerr=sem)


# ===========================================================================
def main():
    fig = plt.figure(figsize=(7.2, 9.3))
    # Two-level grid: the paradigm row needs far less air under it than the
    # data rows need between them, which a single uniform hspace cannot give.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.55, 6.30], hspace=0.10,
                             left=0.085, right=0.975, top=0.965, bottom=0.05)
    gs = outer[1].subgridspec(3, 4, hspace=0.46, wspace=0.5)

    # Row A : paradigm
    ax_a = fig.add_subplot(outer[0])
    panel_a(ax_a)
    ns.panel(ax_a, "a", x=-0.03, y=1.0)

    # ---- HUMAN row (Ma 2014 Fig. 1b–c; Bays 2009 Fig. 1b / 2b) ----
    Ns_h, h_err = human_setsize_errors()
    Ns_sd, h_sd, h_sd_sem = human_recall_sd()
    h_nt = human_nontarget_N4()

    axc = fig.add_subplot(gs[0, 0]); plot_error_dists(axc, h_err, Ns_h, colors=ns.PLUM_SEQ)
    axd = fig.add_subplot(gs[0, 1]); plot_mae(axd, Ns_sd, h_sd, ns.PLUM, "s",
                                               yerr=h_sd_sem,
                                               ylabel="Recall SD (deg)")
    axe = fig.add_subplot(gs[0, 2]); plot_precision_human(axe)
    axf = fig.add_subplot(gs[0, 3]); plot_swap(axf, h_nt, color=ns.PLUM)
    for a, L in zip([axc, axd, axe, axf], "bcde"):
        ns.panel(a, L)

    # ---- MODEL row ----
    Ns_m, m_err = model_setsize_errors()
    Ns_s, m_csd, m_csd_sem, m_prec, m_prec_sem = model_subject_stats()
    m_nt = model_swap()

    axg = fig.add_subplot(gs[1, 0]); plot_error_dists(axg, m_err, Ns_m)
    axh = fig.add_subplot(gs[1, 1]); plot_mae(axh, Ns_s, m_csd, ns.MEM, "o",
                                               yerr=m_csd_sem,
                                               ylabel="Recall SD (deg)")
    axi = fig.add_subplot(gs[1, 2]); plot_precision_model(axi, Ns_s, m_prec, m_prec_sem)
    axj = fig.add_subplot(gs[1, 3]); plot_swap(axj, m_nt, color=ns.MEM)
    for a, L in zip([axg, axh, axi, axj], "fghi"):
        ns.panel(a, L)

    # row identity labels
    # Printed at the Figure 3 row-label size (8 pt) after the manuscript scales
    # this figure down to 0.78\textheight.
    fig.text(0.012, _row_center(fig, gs, 0), "Human", rotation=90,
             va="center", ha="center", fontsize=10.6, fontweight="bold",
             color=ns.PLUM)
    fig.text(0.012, _row_center(fig, gs, 1), "MemGP", rotation=90,
             va="center", ha="center", fontsize=10.6, fontweight="bold",
             color=ns.MEM)

    # ---- hyperparameters + representational dynamics ----
    axk = fig.add_subplot(gs[2, 0]); panel_sweep(axk, "M")
    axl = fig.add_subplot(gs[2, 1]); panel_sweep(axl, "beta")
    axm = fig.add_subplot(gs[2, 2]); panel_sweep(axm, "lambda")
    axn = fig.add_subplot(gs[2, 3]); panel_misbinding(axn)
    for a, L in zip([axk, axl, axm, axn], "jklm"):
        ns.panel(a, L)

    ns.save_fig(fig, "fig2_setsize")


def _row_center(fig, gs, row):
    pos = gs[row, 0].get_position(fig)
    return (pos.y0 + pos.y1) / 2


# ---- hyperparameter panels (mean +/- SEM across simulated subjects) ----
SWEEPS = {
    "M": dict(title="inducing pts $M$", label=lambda v: f"{int(v) ** 2}",
              colors=lambda n: plt.cm.viridis(np.linspace(0.1, 0.85, n))),
    "beta": dict(title=r"forgetting $\beta$", label=lambda v: f"{v:g}",
                 colors=lambda n: [ns.GREY, "#E6A23C", "#D2691E", ns.BAD][:n]),
    "lambda": dict(title=r"lengthscale $\lambda$", label=lambda v: f"{v:g}",
                   colors=lambda n: plt.cm.GnBu(np.linspace(0.42, 0.95, n))),
}


def panel_sweep(ax, design):
    """Error vs set size for each level of one parameter (Fig. 2j-l)."""
    spec = SWEEPS[design]
    levels, Ns, mae, sem = sweep_mae(design)
    colors = spec["colors"](len(levels))
    for i, lv in enumerate(levels):
        ax.errorbar(Ns, mae[i], yerr=sem[i], fmt="o-", color=colors[i], ms=4,
                    lw=1.7, markeredgecolor="white", markeredgewidth=0.5,
                    capsize=2, elinewidth=1.0, zorder=4, label=spec["label"](lv))
    ax.set_xlabel("Set size $N$"); ax.set_ylabel("Error (deg)")
    ax.set_xticks(Ns)
    ax.set_ylim(0, float(np.nanmax(mae + sem)) * 1.15)
    leg = ax.legend(title=spec["title"], fontsize=6.0, title_fontsize=6.2,
                    loc="upper left", handlelength=1.0, labelspacing=0.2,
                    borderpad=0.2)
    leg._legend_box.align = "left"


def misbinding_rate():
    """Pooled misbinding rate (%) per (N, distance), and SEM across subjects."""
    df = pd.read_csv(ns.result_path("setsize", "setsize_misbinding_trials.csv"))
    Ns = sorted(int(n) for n in df["N"].unique())
    dists = sorted(int(d) for d in df["distance"].unique())
    rate = df.pivot_table(index="N", columns="distance", values="swap",
                          aggfunc="mean").reindex(index=Ns, columns=dists) * 100.0
    subj = df.groupby(["N", "distance", "subject"])["swap"].mean() * 100.0
    sem = subj.groupby(["N", "distance"]).sem().unstack("distance").reindex(
        index=Ns, columns=dists)
    return Ns, dists, rate.to_numpy(), sem.to_numpy()


def panel_misbinding(ax):
    """Panel m: swap errors are spatially graded and grow with load. Heatmap of
    the misbinding rate (probe reported as a non-target item) as a joint function
    of set size and target-distractor spatial distance."""
    Ns, dists, rate, _ = misbinding_rate()
    vmax = float(np.nanmax(rate))
    im = ax.imshow(rate, aspect="auto", origin="lower", cmap="Oranges",
                   vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(dists))); ax.set_xticklabels(dists)
    ax.set_yticks(range(len(Ns))); ax.set_yticklabels(Ns)
    ax.set_xlabel("Target-distractor dist. (deg)")
    ax.set_ylabel("Set size $N$")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label("Misbinding (%)", fontsize=6.2)
    cb.outline.set_visible(False)


if __name__ == "__main__":
    main()
