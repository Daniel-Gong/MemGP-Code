"""Figure 3 - Retrospective cues and goal-directed attention.

Panels
  a    retro-cue task: array -> delay -> retro-cue -> delay -> probe
  b    spatial-proximity gain a(l) that reweights the maintenance ELBO
  c-d  human (Souza et al. 2014): cue/focus benefit; benefit grows with load
  e    human (van Moorselaar et al. 2015): cue can be redirected (Cowan's K)
  f-h  MemGP reproduces all three (simulation outputs)
  i-j  recall-error distributions, neutral vs cued, at N=2 and N=4
  k    cued and uncued activation both decline; the difference is not significant
  l    the cued item's read-out variance falls: selective sharpening

Sizing: drawn at the printed size (6.5 in = \\textwidth at 1 in margins) so the
LaTeX scale factor is 1.0 and every font here is the font on the page. No text
below 7 pt.

Style: the paired human / MemGP rows are colour-coded by source. Panels c--e
(human) are a black/grey family; panels f--h (MemGP) are the MemGP blue
family. Within a row, darker = cued, lighter = neutral. In e and h, item A
and item B share the row colour and are separated by marker (circle vs
square, unfilled). Human bars are open, MemGP bars are solid. Cue red is reserved
for the schematic (a--b) and the delay traces (i--l).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import offset_copy
from scipy import stats

from figures import nature_style as ns


ns.set_style(8)
FIG_W, FIG_H = 6.5, 6.48

FS_MIN = 7.0       # legends, dense corner text (absolute floor)
FS_NOTE = 7.5      # in-panel annotations
FS_HEAD = 8.0      # panel headings, row banners
FS_PANEL = 11.0    # panel letters

OUT = Path(ns.out_path("fig3_retrocue"))

INK = ns.INK
CUE = ns.CUE
NEUTRAL = ns.NEUTRAL
GREY = ns.GREY
GREY_DK = ns.GREY_DK
MEM = ns.MEM
MEM_LT = ns.MEM_LT

# Row palettes: colour family = data source.
HUMAN = dict(
    accent=ns.PLUM,
    cued=ns.PLUM,
    neutral=ns.PLUM_LT,
)
MODEL = dict(
    accent=MEM,
    cued=MEM,
    neutral=MEM_LT,
)

ITEMS = [(-130, -120), (-35, 70), (60, 150), (150, -40)]
CUED = 2

SOUZA_CSV = {
    2: Path(ns.data_path("souza2014", "exp_11_Souza_et_al_2014_-_Exp_2_-_2_items.csv")),
    4: Path(ns.data_path("souza2014", "exp_13_Souza_et_al_2014_-_Exp_2_-_4_items.csv")),
}
VANMOOR_CSV = Path(ns.data_path("vanmoorselaar2015", "fig2b_capacity.csv"))


# ---------------------------------------------------------------------------
# shared rendering: bars — human = filled, no outline; MemGP = solid with a white edge
# ---------------------------------------------------------------------------
def bar_kw(color, solid):
    if solid:
        return dict(color=color, edgecolor="white", linewidth=0.8)
    return dict(color=color, linewidth=0)


def head(ax, text, color):
    ax.set_title(text, loc="left", fontsize=FS_HEAD, fontweight="bold",
                 color=color, pad=3)


# ===========================================================================
# Panel a : retro-cue paradigm
# ===========================================================================
def _screen(parent, rect):
    iax = parent.inset_axes(rect)
    iax.set_xlim(0, 1)
    iax.set_ylim(0, 1)
    iax.set_xticks([])
    iax.set_yticks([])
    for s in iax.spines.values():
        s.set_visible(False)
    iax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, fc="white", ec=INK,
                            lw=1.2))
    iax.plot([0.47, 0.53], [0.5, 0.5], color=INK, lw=0.9)
    iax.plot([0.5, 0.5], [0.47, 0.53], color=INK, lw=0.9)
    return iax


def _items(iax, outline=False, r=0.27, sq=0.115):
    for loc, col in ITEMS:
        a = np.deg2rad(loc)
        px, py = 0.5 + r * np.cos(a), 0.5 + r * np.sin(a)
        fc = "white" if outline else ns.color_to_rgb(col)
        iax.add_patch(Rectangle((px - sq / 2, py - sq / 2), sq, sq, fc=fc,
                                ec=INK, lw=1.0, zorder=5))


def _ring_cue(iax, r=0.27, sq=0.115):
    a = np.deg2rad(ITEMS[CUED][0])
    px, py = 0.5 + r * np.cos(a), 0.5 + r * np.sin(a)
    iax.add_patch(Rectangle((px - sq / 2, py - sq / 2), sq, sq, fc="white",
                            ec=CUE, lw=1.8, zorder=6))


def panel_a(ax):
    """Three displays joined by two labelled delays; the empty fixation
    screens the delays used to occupy said nothing worth their width."""
    ns.blank(ax, equal=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    w, h, y0 = 0.257, 0.745, 0.165
    xs = [0.020, 0.3635, 0.707]
    labels = ["Array", "Retrocue", "Probe"]
    y_mid = y0 + h / 2

    for x, lab in zip(xs, labels):
        iax = _screen(ax, [x, y0, w, h])
        if lab == "Array":
            _items(iax)
        elif lab == "Retrocue":
            _items(iax, outline=True)
            _ring_cue(iax)
        else:
            ns.color_wheel(iax, (0.5, 0.5), r_out=0.455, r_in=0.395)
            _items(iax, outline=True)
            _ring_cue(iax)
        ax.text(x + w / 2, y0 - 0.05, lab, ha="center", va="top",
                fontsize=FS_NOTE,
                color=CUE if lab == "Retrocue" else INK,
                fontweight="bold" if lab == "Retrocue" else "normal")

    for i in range(2):
        x0 = xs[i] + w + 0.008
        x1 = xs[i + 1] - 0.008
        ax.annotate("", xy=(x1, y_mid), xytext=(x0, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4,
                                    mutation_scale=10, shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, y_mid + 0.055, "delay", ha="center",
                va="bottom", fontsize=FS_MIN, color=GREY_DK, style="italic")

    ax.text(0.0, 1.02, "Continuous reproduction with a retrospective cue",
            ha="left", va="bottom", fontsize=FS_HEAD, fontweight="bold",
            color=INK)


def panel_b(ax):
    """The gain profile the cue installs on the maintenance objective."""
    cued_loc = ITEMS[CUED][0]
    x = np.linspace(-180, 180, 512)
    a = np.exp(-0.5 * (ns.circ_dist(x, cued_loc) / 42.0) ** 2)

    ax.fill_between(x, 0, a, color=CUE, alpha=0.15, lw=0, zorder=1)
    ax.plot(x, a, color=CUE, lw=1.9, zorder=3)
    # Items sit on the location axis itself, inside the axes, so they cannot
    # crowd the tick labels.
    for loc, col in ITEMS:
        ax.add_patch(Rectangle((loc - 10, 0.012), 20, 0.082,
                               fc=ns.color_to_rgb(col), ec=INK, lw=0.6,
                               zorder=7))
    # Marker hangs just below the axis at the cued location (no tick there),
    # pointing up at the item; the label sits under the marker. Mixed
    # transform: x in data, y in axes fraction.
    trans = ax.get_xaxis_transform()
    ax.plot(cued_loc, -0.048, marker="^", ms=5.0, color=CUE, zorder=6,
            markeredgecolor="white", markeredgewidth=0.6,
            transform=trans, clip_on=False)
    ax.text(cued_loc, -0.078, "cued", ha="center", va="top",
            fontsize=FS_MIN, color=CUE, fontweight="bold",
            transform=trans, clip_on=False)

    ax.set_xlim(-180, 180)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["1", "$G$"])
    ax.set_xlabel("Location (deg)", labelpad=5.5)
    ax.set_ylabel(r"Gain $a(\ell)$")
    ns.clean(ax)
    ax.spines["left"].set_bounds(0, 1)
    ax.spines["bottom"].set_bounds(-180, 180)
    head(ax, "Attention gain", INK)


# ===========================================================================
# paired human / model panels
# ===========================================================================
def _err_kw(color):
    return dict(lw=1.0, capthick=1.0, capsize=2.2, ecolor=color)


def cue_benefit(ax, neutral, cued, neutral_sem, cued_sem, n_items, solid,
                theme):
    """Mean recall error, neutral vs cued, at one set size."""
    ax.bar(0, neutral, 0.60, yerr=neutral_sem,
           error_kw=_err_kw(theme["neutral"]),
           **bar_kw(theme["neutral"], solid))
    ax.bar(1, cued, 0.60, yerr=cued_sem,
           error_kw=_err_kw(theme["cued"]),
           **bar_kw(theme["cued"], solid))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["neutral", "cued"])
    ax.set_xlim(-0.62, 1.62)
    ax.set_ylabel("Recall error (deg)")
    ax.set_ylim(0, max(neutral + neutral_sem, cued + cued_sem) * 1.34)
    ns.clean(ax)
    head(ax, f"Cue lowers error ($\\boldsymbol{{N}}$={n_items})", theme["accent"])


def load_benefit(ax, sizes, benefit, sem, solid, theme):
    """Retrocue benefit as a function of set size."""
    x = np.arange(len(sizes))
    ax.bar(x, benefit, 0.58, yerr=sem, error_kw=_err_kw(theme["cued"]),
           **bar_kw(theme["cued"], solid))
    ax.set_xticks(x)
    # The load is in the tick labels rather than an axis label, which would
    # otherwise drop into the row below.
    ax.set_xticklabels([f"$N={int(n)}$" for n in sizes])
    ax.set_xlim(-0.62, len(sizes) - 1 + 0.62)
    ax.set_ylabel("Cue benefit (deg)")
    ax.set_ylim(0, max(b + s for b, s in zip(benefit, sem)) * 1.34)
    ns.clean(ax)
    head(ax, "Benefit grows with load", theme["accent"])


def reallocation(ax, cue_a, cue_b, sem, theme,
                 ylabel="Recall error (deg)",
                 title="Cue can be redirected",
                 xticklabels=("cue A", "cue B")):
    """Error (or capacity) for each item after cue A and after cue B."""
    x = [0, 1]
    ea = [cue_a["A"], cue_b["A"]]
    eb = [cue_a["B"], cue_b["B"]]
    sa = [sem["cueA"]["A"], sem["cueB"]["A"]]
    sb = [sem["cueA"]["B"], sem["cueB"]["B"]]
    colour = theme["accent"]
    for y, s, marker, label in ((ea, sa, "o", "item A"),
                                (eb, sb, "s", "item B")):
        ax.plot(x, y, color=colour, ls="-", lw=1.7, marker=marker, ms=4.6,
                mfc="white", mec=colour, mew=1.2, zorder=4, label=label)
        ax.errorbar(x, y, yerr=s, fmt="none", ecolor=colour, elinewidth=1.0,
                    capsize=3.2, capthick=1.0, zorder=5)
    ax.set_xlim(-0.32, 1.32)
    ax.set_xticks(x)
    ax.set_xticklabels(list(xticklabels))
    ax.set_ylabel(ylabel)
    ymax = max(ea + eb) + max(sa + sb)
    ax.set_ylim(0, ymax * 1.28)
    ns.clean(ax)
    ax.legend(fontsize=FS_MIN, loc="upper center", ncol=2, handlelength=1.4,
              columnspacing=0.9, handletextpad=0.4, borderpad=0.15)
    head(ax, title, theme["accent"])


# ===========================================================================
# Human behavioural loaders
# ===========================================================================
def human_souza():
    """Souza et al. (2014) Exp. 2: no-delay vs delay (focused) after the recall cue.

    ``condition_index`` 0 = no-delay (respond immediately; plotted as neutral);
    1 = delay (extra focused-attention interval; plotted as cued). Delay has
    *lower* absolute error — the focusing benefit.
    """
    parts = []
    for N, path in SOUZA_CSV.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["set_size"] = N
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["abs_err_deg"] = np.rad2deg(np.abs(df["error"].to_numpy(float)))
    sm = (df.groupby(["subject", "set_size", "condition_index"], as_index=False)
            ["abs_err_deg"].mean())

    def mean_sem(series):
        x = np.asarray(series, float)
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))

    # Panel c: N=4 neutral (0) vs cued/focused (1)
    w4 = sm[sm.set_size == 4].pivot(
        index="subject", columns="condition_index", values="abs_err_deg"
    ).dropna()
    n_mean, n_sem = mean_sem(w4[0])
    c_mean, c_sem = mean_sem(w4[1])
    t_c, p_c = stats.ttest_rel(w4[0], w4[1])

    # Panel d: cue benefit = neutral − focused, by set size
    sizes, benefits, benefit_sems = [], [], []
    benefit_by_N = {}
    for N in (2, 4):
        w = sm[sm.set_size == N].pivot(
            index="subject", columns="condition_index", values="abs_err_deg"
        ).dropna()
        b = w[0] - w[1]
        m, s = mean_sem(b)
        sizes.append(N)
        benefits.append(m)
        benefit_sems.append(s)
        benefit_by_N[N] = b
    common = benefit_by_N[2].index.intersection(benefit_by_N[4].index)
    d = benefit_by_N[4].loc[common] - benefit_by_N[2].loc[common]
    t_d, p_d = stats.ttest_1samp(d, 0.0)

    return dict(
        n4_neutral=n_mean, n4_cued=c_mean,
        n4_neutral_sem=n_sem, n4_cued_sem=c_sem,
        p_cued=float(p_c), t_cued=float(t_c),
        sizes=sizes, benefits=benefits, benefit_sems=benefit_sems,
        p_load=float(p_d) / 2.0,  # one-tailed: larger benefit at higher load
        t_load=float(t_d),
    )


def human_van_moorselaar():
    """van Moorselaar et al. (2015) Exp. 2: Cowan's K under cue redirect."""
    if not VANMOOR_CSV.is_file():
        raise FileNotFoundError(VANMOOR_CSV)
    df = pd.read_csv(VANMOOR_CSV, comment="#")
    cue_a = {
        "A": float(df[(df.condition == "cue_A") & (df.item == "A")]["K_mean"].iloc[0]),
        "B": float(df[(df.condition == "cue_A") & (df.item == "B")]["K_mean"].iloc[0]),
    }
    cue_b = {
        "A": float(df[(df.condition == "cue_B") & (df.item == "A")]["K_mean"].iloc[0]),
        "B": float(df[(df.condition == "cue_B") & (df.item == "B")]["K_mean"].iloc[0]),
    }
    sem = {
        "cueA": {
            "A": float(df[(df.condition == "cue_A") & (df.item == "A")]["K_sem"].iloc[0]),
            "B": float(df[(df.condition == "cue_A") & (df.item == "B")]["K_sem"].iloc[0]),
        },
        "cueB": {
            "A": float(df[(df.condition == "cue_B") & (df.item == "A")]["K_sem"].iloc[0]),
            "B": float(df[(df.condition == "cue_B") & (df.item == "B")]["K_sem"].iloc[0]),
        },
    }
    return cue_a, cue_b, sem


def match_y(*axes):
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def sig_bracket(ax, p, x0=0, x1=1, frac=0.86):
    """Drawn after the y scales are matched so the bracket sits inside."""
    top = ax.get_ylim()[1]
    y = frac * top
    tick = 0.028 * top
    ax.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=INK, lw=0.9,
            clip_on=False, zorder=6)
    ax.text((x0 + x1) / 2, y + 0.01 * top, stars(p), ha="center", va="bottom",
            fontsize=FS_HEAD + 1, fontweight="bold", color=INK)


# ===========================================================================
# MemGP-only panels
# ===========================================================================
def error_dist(ax, dist, N, show_labels):
    """Recall-error distribution, neutral vs cued. The cue sharpens the
    central peak and thins the wide-error (swap/guess) tails."""
    if dist is None:
        ax.text(0.5, 0.5, "pending", ha="center", transform=ax.transAxes)
        return
    xs = np.linspace(-180, 180, 361)
    for key, colour, label in ((f"neutral_N{N}", NEUTRAL, "neutral"),
                               (f"cued_N{N}", CUE, "cued")):
        e = np.asarray(dist[key], float)
        k = stats.gaussian_kde(e, bw_method=0.30)(xs)
        ax.fill_between(xs, k, color=colour, alpha=0.24, lw=0)
        ax.plot(xs, k, color=colour, lw=1.7, zorder=3)
    ax.axvline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, 0, 180])
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_xlabel("Error (deg)")
    ax.set_ylabel("Density")
    ns.clean(ax)
    if show_labels:
        ax.text(0.96, 0.92, "cued", transform=ax.transAxes, ha="right",
                va="top", fontsize=FS_MIN, color=CUE, fontweight="bold")
        # Same gap as the panel k legend (labelspacing=0.2 at FS_MIN).
        gap = FS_MIN * (1.0 + 0.2)
        trans = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=-gap, units="points")
        ax.text(0.96, 0.92, "neutral", transform=trans, ha="right",
                va="top", fontsize=FS_MIN, color=NEUTRAL, fontweight="bold")
    head(ax, f"$\\boldsymbol{{N}}$ = {N}", MODEL["accent"])


def dynamics(ax, dyn, key_cued, key_uncued, ylabel, title,
             dash_uncued=False, legend_loc="lower right", bottom_pad=0.06,
             other_label="uncued"):
    """Change from cue onset. Band is the standard error across trials.

    When the two traces overlie, the second curve is dashed and drawn on top.
    """
    if dyn is None:
        ax.text(0.5, 0.5, "pending", ha="center", transform=ax.transAxes)
        return
    cue = int(dyn["cue_epoch"])
    cued = np.asarray(dyn[key_cued], float) * 1e3
    uncued = np.asarray(dyn[key_uncued], float) * 1e3
    cued = cued - cued[:, cue][:, None]
    uncued = uncued - uncued[:, cue][:, None]
    epochs = np.arange(cued.shape[1])
    mean_c, mean_u = cued.mean(axis=0), uncued.mean(axis=0)
    se_c, se_u = stats.sem(cued, axis=0), stats.sem(uncued, axis=0)
    lo_c, hi_c = mean_c - se_c, mean_c + se_c
    lo_u, hi_u = mean_u - se_u, mean_u + se_u

    ax.axhline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.fill_between(epochs, lo_u, hi_u, color=NEUTRAL, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(epochs, lo_c, hi_c, color=CUE, alpha=0.22, lw=0, zorder=2)
    if dash_uncued:
        ax.plot(epochs, mean_c, color=CUE, lw=1.6, label="cued", zorder=3)
        ax.plot(epochs, mean_u, color=NEUTRAL, lw=1.6, ls=(0, (4, 1.6)),
                label=other_label, zorder=4)
    else:
        ax.plot(epochs, mean_u, color=NEUTRAL, lw=1.6, label=other_label, zorder=3)
        ax.plot(epochs, mean_c, color=CUE, lw=1.6, label="cued", zorder=4)
    ax.axvline(cue, color=GREY, lw=0.8, ls=(0, (3, 2.5)), zorder=2)

    y_lo = min(lo_c.min(), lo_u.min())
    y_hi = max(hi_c.max(), hi_u.max())
    span = y_hi - y_lo or 1.0
    ax.set_xlim(0, epochs[-1])
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(y_lo - bottom_pad * span, y_hi + 0.16 * span)
    ax.text(cue + 2, y_hi + 0.12 * span, "cue", ha="left", va="top",
            fontsize=FS_MIN, color=GREY_DK)
    ax.set_xlabel("Delay (epoch)")
    ax.set_ylabel(ylabel, fontsize=FS_NOTE)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("cued"), labels.index(other_label)]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc=legend_loc, fontsize=FS_MIN, borderpad=0.2,
              handlelength=1.6, labelspacing=0.2)
    ns.clean(ax)
    head(ax, title, MODEL["accent"])


# ===========================================================================
# data
# ===========================================================================
def model_retrocue():
    return pd.read_csv(ns.result_path("retrocue", "retrocue_setsize_results.csv"))


def model_error_dist():
    """Probed-item signed errors pooled over subjects, keyed e.g. 'cued_N4'."""
    out = {}
    for N in (2, 4):
        df = pd.read_csv(ns.result_path("retrocue", f"retrocue_trials_N{N}.csv"))
        for cond in ("neutral", "cued"):
            out[f"{cond}_N{N}"] = df.loc[df.condition == cond, "signed_error"].to_numpy(float)
    return out


def model_dynamics(trial_level=False):
    """Trajectories for Fig. 3k-l.

    Default is the subject mean (subjects x epochs), used by the paired tests.
    ``trial_level`` flattens subjects and trials (trials x epochs) for the
    across-trial bootstrap band.
    """
    d = ns.load_npz(ns.result_path("retrocue", "retrocue_dynamics.npz"))
    skip = {"subject", "cue_epoch"}
    out = {}
    for k, v in d.items():
        if k in skip:
            continue
        arr = np.asarray(v, float)
        if arr.ndim < 2:
            continue
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1]) if trial_level else arr.mean(axis=1)
        out[k] = arr
    out["cue_epoch"] = int(d["cue_epoch"])
    return out


def model_load_p():
    """One-tailed test that the cue benefit is larger at N=4 than at N=2."""
    comp = pd.read_csv(ns.result_path("retrocue", "retrocue_setsize_comparison.csv"))
    return float(comp.p_val.iloc[0])


def model_reallocation():
    """Subject means (20 subjects, 100 redirect trials each)."""
    df = pd.read_csv(ns.result_path("retrocue", "cue_flexibility_subjects.csv"))
    cue_a = {"A": df["A_post_cueA"].mean(), "B": df["B_post_cueA"].mean()}
    cue_b = {"A": df["A_post_cueB"].mean(), "B": df["B_post_cueB"].mean()}
    sem = {
        "cueA": {"A": stats.sem(df["A_post_cueA"]),
                 "B": stats.sem(df["B_post_cueA"])},
        "cueB": {"A": stats.sem(df["A_post_cueB"]),
                 "B": stats.sem(df["B_post_cueB"])},
    }
    return cue_a, cue_b, sem


# ---------------------------------------------------------------------------
def row_label(fig, axes, label, colour, x=0.020):
    """Names the row down its left edge."""
    y0 = min(ax.get_position().y0 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    fig.text(x, (y0 + y1) / 2, label, rotation=90, va="center", ha="center",
             fontsize=FS_HEAD, fontweight="bold", color=colour)


def write_html(svg_path):
    svg = svg_path.read_text()
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Figure 3 - Retrospective cues and goal-directed attention</title>
<style>
@page {{ size: {FIG_W}in {FIG_H}in; margin: 0; }}
html, body {{ margin: 0; padding: 0; background: white; }}
main {{ width: {FIG_W}in; height: {FIG_H}in; display: flex; align-items: center;
        justify-content: center; overflow: hidden; }}
svg {{ width: {FIG_W}in; height: {FIG_H}in; display: block; }}
</style></head><body><main>{svg}</main></body></html>"""
    OUT.with_suffix(".html").write_text(html)


def main():
    df = model_retrocue()
    r4 = df[df.set_size == 4].iloc[0]
    cue_a, cue_b, sem = model_reallocation()
    dist = model_error_dist()
    dyn = model_dynamics(trial_level=True)
    other = "neutral" if "neutral_act" in dyn else "uncued"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    outer = fig.add_gridspec(
        4, 1, height_ratios=[1.16, 1.0, 1.0, 0.96],
        left=0.108, right=0.976, top=0.952, bottom=0.062, hspace=0.44,
    )

    top = outer[0].subgridspec(1, 2, width_ratios=[2.62, 1.0], wspace=0.30)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    panel_a(ax_a)
    panel_b(ax_b)

    human = outer[1].subgridspec(1, 3, wspace=0.46)
    ax_c = fig.add_subplot(human[0, 0])
    ax_d = fig.add_subplot(human[0, 1])
    ax_e = fig.add_subplot(human[0, 2])

    model = outer[2].subgridspec(1, 3, wspace=0.46)
    ax_f = fig.add_subplot(model[0, 0])
    ax_g = fig.add_subplot(model[0, 1])
    ax_h = fig.add_subplot(model[0, 2])

    # Wider than the rows above: k and l carry a long single-line y label.
    bottom = outer[3].subgridspec(1, 4, wspace=0.70)
    ax_i = fig.add_subplot(bottom[0, 0])
    ax_j = fig.add_subplot(bottom[0, 1])
    ax_k = fig.add_subplot(bottom[0, 2])
    ax_l = fig.add_subplot(bottom[0, 3])

    # ---- human benchmark (Souza 2014; van Moorselaar 2015) ----
    hum = human_souza()
    vm_a, vm_b, vm_sem = human_van_moorselaar()
    cue_benefit(ax_c, hum["n4_neutral"], hum["n4_cued"],
                hum["n4_neutral_sem"], hum["n4_cued_sem"],
                n_items=4, solid=False, theme=HUMAN)
    load_benefit(ax_d, hum["sizes"], hum["benefits"], hum["benefit_sems"],
                 solid=False, theme=HUMAN)
    reallocation(ax_e, vm_a, vm_b, vm_sem, theme=HUMAN,
                 ylabel="Cowan's $K$",
                 title="Cue can be redirected",
                 xticklabels=("cue A", "cue B"))

    # ---- MemGP ----
    cue_benefit(ax_f, r4.neutral_mae, r4.cued_mae, r4.neutral_sem, r4.cued_sem,
                n_items=int(r4.set_size), solid=True, theme=MODEL)
    load_benefit(ax_g, df.set_size.tolist(), df.benefit_mean.tolist(),
                 df.benefit_sem.tolist(), solid=True, theme=MODEL)
    reallocation(ax_h, cue_a, cue_b, sem, theme=MODEL)

    error_dist(ax_i, dist, 2, show_labels=True)
    error_dist(ax_j, dist, 4, show_labels=False)
    dynamics(ax_k, dyn, "cued_act", f"{other}_act",
             "$\\Delta$ mean ($\\times10^{-3}$)", "Strength",
             dash_uncued=True, legend_loc="lower left", other_label=other)
    dynamics(ax_l, dyn, "cued_var", f"{other}_var",
             "$\\Delta$ variance ($\\times10^{-3}$)", "Uncertainty",
             legend_loc="lower left", bottom_pad=0.42, other_label=other)

    match_y(ax_c, ax_f)
    match_y(ax_d, ax_g)
    # e (Cowan's K) and h (recall error deg) use different units — do not match
    match_y(ax_i, ax_j)
    sig_bracket(ax_c, hum["p_cued"])
    sig_bracket(ax_d, hum["p_load"])
    sig_bracket(ax_f, float(r4.p_val))
    sig_bracket(ax_g, model_load_p())

    # Panel a's stage labels sit inside its axes, but b hangs tick labels and
    # an axis label below its own, which lands them on top of the c-e titles.
    # Lift b's floor rather than loosening the gap for the whole grid.
    pos = ax_b.get_position()
    lift = 0.040
    ax_b.set_position([pos.x0, pos.y0 + lift, pos.width, pos.height - lift])
    pos_b = ax_b.get_position()
    pos_e = ax_e.get_position()
    ax_b.set_position([pos_e.x0, pos_b.y0, pos_e.width, pos_b.height])

    pos_a = ax_a.get_position()
    pos_c = ax_c.get_position()
    c_right = pos_c.x0 - 0.28 * pos_c.width
    x_a = (c_right - pos_a.x0) / pos_a.width
    ns.panel(ax_a, "a", x=x_a, y=1.005, size=FS_PANEL)
    ns.panel(ax_b, "b", x=-0.30, y=1.045, size=FS_PANEL)
    for ax, letter in ((ax_c, "c"), (ax_d, "d"), (ax_e, "e"),
                       (ax_f, "f"), (ax_g, "g"), (ax_h, "h")):
        ns.panel(ax, letter, x=-0.28, y=1.03, size=FS_PANEL)
    pos_i = ax_i.get_position()
    x_i = (c_right - pos_i.x0) / pos_i.width
    ns.panel(ax_i, "i", x=x_i, y=1.03, size=FS_PANEL)
    for ax, letter in ((ax_j, "j"), (ax_k, "k"), (ax_l, "l")):
        ns.panel(ax, letter, x=-0.36, y=1.03, size=FS_PANEL)

    row_label(fig, [ax_c], "Human", HUMAN["accent"])
    row_label(fig, [ax_f], "MemGP", MODEL["accent"])

    svg_path = OUT.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches=None, facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=400, bbox_inches=None,
                facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches=None, facecolor="white")
    write_html(svg_path)
    plt.close(fig)
    print("saved fig3_retrocue.svg / .png / .pdf / .html")


if __name__ == "__main__":
    main()
